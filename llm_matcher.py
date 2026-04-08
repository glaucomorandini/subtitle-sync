"""LLM-based semantic line matching via OpenRouter."""

import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─── Shared rate-limit state ─────────────────────────────────────────────────
# When any worker hits a 429/overload, it sets _rl_until to the time the
# server told us (or a backoff estimate) and all other workers sleep until
# then before issuing their next request. This avoids the "everyone retries
# simultaneously and gets 429'd again" pile-up.
_rl_lock = threading.Lock()
_rl_until = 0.0


def _rl_wait_if_paused(label=""):
    first = True
    while True:
        with _rl_lock:
            now = time.monotonic()
            if now >= _rl_until:
                return
            sleep_for = _rl_until - now
        if first:
            print(f"  [rate-limit] waiting {sleep_for:.0f}s{(' ' + label) if label else ''}...", flush=True)
            first = False
        time.sleep(min(5.0, sleep_for))


def _rl_set_pause(seconds):
    target = time.monotonic() + max(0.0, seconds)
    with _rl_lock:
        global _rl_until
        if target > _rl_until:
            _rl_until = target

BACKENDS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
    },
    "local": {
        "url": "http://localhost:11434/v1/chat/completions",
        "api_key_env": None,
    },
}
DEFAULT_BACKEND = "openrouter"


def _resolve_backend(name):
    if name not in BACKENDS:
        print(f"Error: unknown backend '{name}'. Choices: {', '.join(BACKENDS)}", file=sys.stderr)
        sys.exit(1)
    cfg = BACKENDS[name]
    api_key = None
    env_var = cfg["api_key_env"]
    if env_var:
        api_key = os.environ.get(env_var)
        if not api_key:
            print(f"Error: {env_var} environment variable not set.", file=sys.stderr)
            sys.exit(1)
    return {"name": name, "url": cfg["url"], "api_key": api_key}


SYSTEM_PROMPT = """/no_think
You match PT-BR subtitle lines to EN subtitle lines (same content, different translations).

Rules:
- A PT-BR line maps to one or more CONSECUTIVE EN lines (e.g. [3] or [3,4], never [3,7]).
- Each EN line should be claimed by AT MOST ONE PT line across your entire response. Do not reuse an EN index.
- Generic/short lines like "OK", "Sim", "Não", "Ei!", "Hã?", "Ah!", interjections, and single-word reactions are almost impossible to match reliably — OMIT them unless the surrounding lines unambiguously anchor the match. When in doubt, omit.
- It is far better to omit a PT line than to guess. Do not invent matches to maximise coverage.
- Some edge lines in the batch belong to adjacent batches; omit them if no match.
- "c": "high" only if you are certain the meaning matches AND the surrounding context agrees. Use "low" if there is ANY ambiguity.

Return ONLY a JSON object like {"matches":[{"pt":0,"en":[0],"c":"high"},{"pt":1,"en":[1,2],"c":"low"}]}. Omit PT lines with no confident match."""


class RateLimitError(Exception):
    pass


def _parse_retry_after(value):
    """Retry-After can be seconds (int) or an HTTP-date. Return seconds or None."""
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime
        import datetime as _dt
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        delta = (dt - _dt.datetime.now(dt.tzinfo)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


_request_counter = 0
_request_lock = threading.Lock()


def _chat_completion(backend, model, messages, temperature=0.1):
    """Call OpenAI-compatible chat completions API for the given backend."""
    # Honor any active global rate-limit pause before issuing the request.
    _rl_wait_if_paused()

    global _request_counter
    with _request_lock:
        _request_counter += 1
        req_id = _request_counter
    print(f"  [req {req_id}] sending request...", flush=True)

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if backend["name"] == "local":
        body["options"] = {"num_ctx": 8192}
        body["think"] = False
    elif backend["name"] == "groq":
        # Only reasoning models accept reasoning_effort; others 400.
        if any(tag in model.lower() for tag in ("qwen3", "gpt-oss")):
            body["reasoning_effort"] = "none"
    payload = json.dumps(body).encode("utf-8")

    url = backend["url"]
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "User-Agent": "subtitle-sync/1.0",
    }
    if backend["api_key"]:
        headers["Authorization"] = f"Bearer {backend['api_key']}"

    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        print(f"  [req {req_id}] HTTP {e.code}: {body[:500]}", flush=True)
        raise
    if "choices" not in data:
        err = data.get("error", {})
        # Treat upstream rate limit errors as retryable
        if err.get("code") in (429, 502) or "rate" in str(err.get("message", "")).lower():
            raise RateLimitError(f"Upstream rate limit: {err.get('message', data)}")
        raise RuntimeError(f"Unexpected API response: {data}")
    return data["choices"][0]["message"]["content"]


def _format_lines(lines, label):
    """Format lines for the prompt as [index] "text"."""
    parts = []
    for i, line in enumerate(lines):
        parts.append(f'[{i}] "{line["text"]}"')
    return f"{label} lines:\n" + "\n".join(parts)


def _parse_response(text, pt_count, en_count):
    """Extract and validate JSON matches from LLM response."""
    # Strip markdown code fences if present
    text = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Also try to find a bare JSON array
    if not text.startswith('['):
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            text = m.group(0)

    parsed = json.loads(text)
    if isinstance(parsed, dict):
        matches = parsed.get('matches', [])
    else:
        matches = parsed

    # Validate
    valid = []
    for entry in matches:
        pt_idx = entry.get('pt')
        en_indices = entry.get('en', [])
        conf = entry.get('c', 'high')
        if conf not in ('high', 'low'):
            conf = 'high'
        if not isinstance(pt_idx, int) or not isinstance(en_indices, list):
            continue
        if pt_idx < 0 or pt_idx >= pt_count:
            continue
        en_indices = [e for e in en_indices if isinstance(e, int) and 0 <= e < en_count]
        if en_indices:
            valid.append({'pt': pt_idx, 'en': en_indices, 'c': conf})

    return valid


def _create_batches(pt_lines, en_lines, num_batches=None, overlap=5, max_en_per_batch=None):
    if num_batches is None:
        num_batches = max(3, min(12, len(pt_lines) // 40))
    """Split lines into time-window batches with overlap.

    Uses the PT-BR timeline to create windows, then gathers EN lines
    that fall in approximately the same relative time region using a
    rough linear estimate.
    """
    if not pt_lines or not en_lines:
        return []

    pt_start = pt_lines[0]['start_s']
    pt_end = pt_lines[-1]['end_s']
    en_start = en_lines[0]['start_s']
    en_end = en_lines[-1]['end_s']
    pt_duration = pt_end - pt_start
    en_duration = en_end - en_start

    if pt_duration <= 0 or en_duration <= 0:
        return []

    # Rough linear mapping: en_time ≈ en_start + (pt_time - pt_start) / pt_duration * en_duration
    def pt_to_en_estimate(t):
        return en_start + (t - pt_start) / pt_duration * en_duration

    # Create time windows on the PT timeline
    window_size = pt_duration / num_batches
    batches = []

    for i in range(num_batches):
        win_start = pt_start + i * window_size
        win_end = pt_start + (i + 1) * window_size

        # Expand window slightly for overlap
        if i > 0:
            win_start -= window_size * 0.15
        if i < num_batches - 1:
            win_end += window_size * 0.15

        # Gather PT lines in this window
        batch_pt = [l for l in pt_lines if l['start_s'] < win_end and l['end_s'] > win_start]

        # Map window to EN timeline and gather EN lines.
        # Use a generous margin (half the window size, min 60s) so that edit
        # drift between the two videos can't push the correct EN match outside
        # the batch — otherwise the LLM has nothing valid to match against.
        en_margin = max(60.0, window_size * 0.5)
        en_win_start = pt_to_en_estimate(win_start) - en_margin
        en_win_end = pt_to_en_estimate(win_end) + en_margin
        batch_en = [l for l in en_lines if l['start_s'] < en_win_end and l['end_s'] > en_win_start]

        if batch_pt and batch_en:
            batches.append((batch_pt, batch_en))

    return batches


MAX_RETRIES = 5
RATE_LIMIT_MAX_WAIT = 120.0  # cap on per-attempt sleep for rate-limit retries


def _is_rate_limit(e):
    if isinstance(e, RateLimitError):
        return True
    if isinstance(e, urllib.error.HTTPError) and e.code in (429, 502, 503, 529):
        return True
    return False


def _run_batch(backend, model, batch_pt, batch_en):
    """Run a single batch with retry/backoff.

    Rate-limit / overload errors (429, 502, 503, 529, RateLimitError) retry
    indefinitely with capped exponential backoff — these are expected on free
    tiers and high-concurrency runs and should never poison the result.

    All other errors (timeouts, 5xx-internal, JSON parse, connection reset,
    etc.) get up to MAX_RETRIES attempts then return an error string so the
    caller can decide whether to abort.
    """
    prompt = _format_lines(batch_pt, "PT-BR") + "\n\n" + _format_lines(batch_en, "EN")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    other_attempts = 0
    rl_attempts = 0
    while True:
        try:
            raw = _chat_completion(backend, model, messages=messages)
            return _parse_response(raw, len(batch_pt), len(batch_en)), None
        except Exception as e:
            if _is_rate_limit(e):
                rl_attempts += 1
                # Prefer the server's Retry-After header if present.
                retry_after = None
                if isinstance(e, urllib.error.HTTPError):
                    retry_after = _parse_retry_after(e.headers.get("Retry-After"))
                if retry_after is None:
                    retry_after = min(RATE_LIMIT_MAX_WAIT,
                                      (2 ** min(rl_attempts, 6)) + random.random() * 2)
                # Pause ALL workers for this duration (one shared timer) so we
                # don't pile up parallel retries that all get 429'd again.
                _rl_set_pause(retry_after)
                _rl_wait_if_paused(f"(attempt {rl_attempts}, {retry_after:.0f}s)")
                continue
            other_attempts += 1
            if other_attempts >= MAX_RETRIES:
                return [], f"FAILED after {MAX_RETRIES} attempts: {e}"
            wait = (2 ** other_attempts) + random.random()
            time.sleep(wait)


def _run_batches(backend, model, batches, pt_index_map, en_index_map, label, concurrency, verbose):
    """Execute a list of (batch_pt, batch_en) pairs and return raw global matches."""
    all_matches = []

    def _worker(idx, batch_pt, batch_en):
        matches, err = _run_batch(backend, model, batch_pt, batch_en)
        return idx, batch_pt, batch_en, matches, err

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = [ex.submit(_worker, i, bp, be) for i, (bp, be) in enumerate(batches)]
        for fut in as_completed(futures):
            idx, batch_pt, batch_en, matches, err = fut.result()
            if verbose:
                tag = f"  {label} {idx + 1}/{len(batches)} ({len(batch_pt)} PT + {len(batch_en)} EN)"
                print(f"{tag}: {err if err else f'{len(matches)} matches'}")
            for m in matches:
                global_pt = pt_index_map[id(batch_pt[m['pt']])]
                global_en = tuple(en_index_map[id(batch_en[e])] for e in m['en'])
                all_matches.append({'pt': global_pt, 'en': global_en, 'c': m.get('c', 'high')})

    return all_matches


def _dedup_matches(all_matches):
    """Cross-batch consensus dedup: pick the most-agreed EN answer per PT line."""
    from collections import defaultdict, Counter
    by_pt = defaultdict(list)
    for m in all_matches:
        by_pt[m['pt']].append((m['en'], m['c']))

    deduped = []
    for pt_idx, candidates in by_pt.items():
        en_counts = Counter(en for en, _ in candidates)
        best_en, best_count = en_counts.most_common(1)[0]
        agreed_confs = [c for en, c in candidates if en == best_en]
        if best_count == len(candidates) and 'high' in agreed_confs:
            conf = 'high'
        elif best_count > len(candidates) // 2:
            conf = 'medium'
        else:
            conf = 'low'
        deduped.append({'pt': pt_idx, 'en': list(best_en), 'c': conf})

    deduped.sort(key=lambda m: m['pt'])
    return deduped


def _rescue_batches(unmatched_pt, en_lines, anchors, rescue_margin=180.0, target_batches=10):
    """Build rescue batches for unmatched PT lines.

    Splits unmatched PT lines into roughly `target_batches` chunks. For each,
    gathers EN lines within the chunk's estimated EN time range ± rescue_margin.
    Accepts large EN windows (no per-line splitting) so the total request
    count stays bounded — better for free-tier daily token limits.
    """
    from time_mapper import interpolate as _interp
    if not anchors or not unmatched_pt:
        return []

    src_arr = [a[0] for a in anchors]
    dst_arr = [a[1] for a in anchors]

    chunk_size = max(1, (len(unmatched_pt) + target_batches - 1) // target_batches)
    batches = []
    for i in range(0, len(unmatched_pt), chunk_size):
        chunk = unmatched_pt[i:i + chunk_size]
        en_center_min = _interp(chunk[0]['start_s'], src_arr, dst_arr) - rescue_margin
        en_center_max = _interp(chunk[-1]['end_s'], src_arr, dst_arr) + rescue_margin
        batch_en = [l for l in en_lines if l['start_s'] < en_center_max and l['end_s'] > en_center_min]
        if batch_en:
            batches.append((chunk, batch_en))
    return batches


def match_lines(pt_lines, en_lines, backend, model, verbose=True,
                concurrency=5, num_batches=None):
    """Match PT-BR lines to EN lines using LLM semantic matching.

    Returns list of {'pt': global_pt_idx, 'en': [global_en_idx, ...]} dicts.

    Two-pass strategy:
      Pass 1 — time-windowed batches with a linear EN estimate. Handles the
               majority of lines whose PT→EN offset is smooth.
      Pass 2 — rescue pass for unmatched PT lines. Uses anchor-based EN
               estimates (built from pass-1 results) with a wide margin so
               scene-reordered lines find their correct EN match even when
               they appear far from where the linear estimate placed them.
    """
    backend = _resolve_backend(backend) if isinstance(backend, str) else backend
    pt_index_map = {id(l): i for i, l in enumerate(pt_lines)}
    en_index_map = {id(l): i for i, l in enumerate(en_lines)}

    # ── Pass 1: time-windowed batches ────────────────────────────────────────
    batches = _create_batches(pt_lines, en_lines, num_batches=num_batches)
    if verbose:
        print(f"  LLM matching: {len(batches)} batches, concurrency={concurrency}, model={model}, backend={backend['name']}")

    raw_matches = _run_batches(backend, model, batches, pt_index_map, en_index_map,
                               "Batch", concurrency, verbose)
    deduped = _dedup_matches(raw_matches)

    # ── Pass 2: rescue unmatched PT lines ────────────────────────────────────
    matched_pt_set = {m['pt'] for m in deduped}
    unmatched = [pt_lines[i] for i in range(len(pt_lines)) if i not in matched_pt_set]

    from time_mapper import build_anchors, filter_outlier_anchors, make_monotonic

    def _do_rescue(label, target_batches):
        nonlocal deduped
        matched = {m['pt'] for m in deduped}
        unmatched_local = [pt_lines[i] for i in range(len(pt_lines)) if i not in matched]
        if not unmatched_local:
            return 0
        high = [m for m in deduped if m.get('c', 'high') != 'low']
        anchors_local = make_monotonic(filter_outlier_anchors(build_anchors(high, pt_lines, en_lines)))
        rescue = _rescue_batches(unmatched_local, en_lines, anchors_local, target_batches=target_batches)
        if not rescue:
            return 0
        if verbose:
            print(f"  {label}: {len(unmatched_local)} unmatched PT lines → {len(rescue)} batches")
        rescue_raw = _run_batches(backend, model, rescue, pt_index_map, en_index_map,
                                  label, concurrency, verbose)
        rescue_deduped = _dedup_matches(rescue_raw)
        existing_pt = {m['pt'] for m in deduped}
        added = 0
        for m in rescue_deduped:
            if m['pt'] not in existing_pt:
                deduped.append(m)
                added += 1
        deduped.sort(key=lambda m: m['pt'])
        return added

    if unmatched:
        added1 = _do_rescue("Rescue", target_batches=10)
        # Second pass only if the first rescue actually helped (added new
        # matches → better anchors → potentially recover more lines).
        if added1 > 0:
            _do_rescue("Rescue2", target_batches=10)

    if verbose:
        print(f"  Total unique matches: {len(deduped)}/{len(pt_lines)} PT lines matched")

    return deduped
