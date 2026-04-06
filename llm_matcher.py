"""LLM-based semantic line matching via OpenRouter."""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

DEFAULT_MODEL = "qwen/qwen3.6-plus:free"

SYSTEM_PROMPT = """You are a subtitle synchronization tool. You will receive two lists of subtitle lines:
- PT-BR (Portuguese) lines with index and timestamp
- EN (English) lines with index and timestamp

These are translations of the same content. Match each PT-BR line to its corresponding EN line(s).

Rules:
- A PT-BR line may correspond to 1 or more EN lines (if EN splits what PT-BR merges)
- Multiple PT-BR lines may correspond to 1 EN line (if PT-BR splits what EN merges)
- Some lines at the start/end of the batch may not have a match (they belong to adjacent batches)
- Use the dialogue content to match, not the timestamps (timestamps are from different video edits)

Return ONLY a JSON array of objects:
[
  {"pt": 0, "en": [0]},
  {"pt": 1, "en": [1, 2]},
  {"pt": 2, "en": [3]}
]

Where "pt" is the PT-BR line index and "en" is a list of matching EN line indices.
If a PT-BR line has no match in this batch, omit it."""


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    return api_key


def _chat_completion(api_key, model, messages, temperature=0.1):
    """Call OpenRouter chat completions API directly."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "choices" not in data:
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

    matches = json.loads(text)

    # Validate
    valid = []
    for entry in matches:
        pt_idx = entry.get('pt')
        en_indices = entry.get('en', [])
        if not isinstance(pt_idx, int) or not isinstance(en_indices, list):
            continue
        if pt_idx < 0 or pt_idx >= pt_count:
            continue
        en_indices = [e for e in en_indices if isinstance(e, int) and 0 <= e < en_count]
        if en_indices:
            valid.append({'pt': pt_idx, 'en': en_indices})

    return valid


def _create_batches(pt_lines, en_lines, num_batches=10, overlap=5):
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

        # Map window to EN timeline and gather EN lines
        en_win_start = pt_to_en_estimate(win_start) - 10  # extra margin
        en_win_end = pt_to_en_estimate(win_end) + 10
        batch_en = [l for l in en_lines if l['start_s'] < en_win_end and l['end_s'] > en_win_start]

        if batch_pt and batch_en:
            batches.append((batch_pt, batch_en))

    return batches


def match_lines(pt_lines, en_lines, model=DEFAULT_MODEL, verbose=True):
    """Match PT-BR lines to EN lines using LLM semantic matching.

    Returns list of {'pt': global_pt_idx, 'en': [global_en_idx, ...]} dicts.
    Indices are into the original pt_lines and en_lines lists.
    """
    api_key = _get_api_key()
    batches = _create_batches(pt_lines, en_lines)

    if verbose:
        print(f"  LLM matching: {len(batches)} batches, model={model}")

    all_matches = []

    # Build index maps: line object -> index in original list
    pt_index_map = {id(l): i for i, l in enumerate(pt_lines)}
    en_index_map = {id(l): i for i, l in enumerate(en_lines)}

    for batch_num, (batch_pt, batch_en) in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_num + 1}/{len(batches)}: {len(batch_pt)} PT + {len(batch_en)} EN lines...", end=" ", flush=True)

        prompt = _format_lines(batch_pt, "PT-BR") + "\n\n" + _format_lines(batch_en, "EN")

        matches = None
        attempt = 0
        while True:
            try:
                raw = _chat_completion(
                    api_key, model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                matches = _parse_response(raw, len(batch_pt), len(batch_en))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = min(20 * (attempt + 1), 80)
                    if verbose:
                        print(f"rate limited, waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                    attempt += 1
                    continue
                # Non-429 HTTP error: retry up to 3 times then skip
                if attempt < 3:
                    if verbose:
                        print(f"retry ({e})...", end=" ", flush=True)
                    attempt += 1
                else:
                    if verbose:
                        print(f"FAILED ({e})")
                    break
            except Exception as e:
                if attempt < 3:
                    if verbose:
                        print(f"retry ({e})...", end=" ", flush=True)
                    time.sleep(5)
                    attempt += 1
                else:
                    if verbose:
                        print(f"FAILED ({e})")
                    break
        # Delay between batches to avoid rate limits on free tier
        time.sleep(10)

        if matches:
            # Remap batch-local indices to global indices
            for m in matches:
                global_pt = pt_index_map[id(batch_pt[m['pt']])]
                global_en = [en_index_map[id(batch_en[e])] for e in m['en']]
                all_matches.append({'pt': global_pt, 'en': global_en})
            if verbose:
                print(f"{len(matches)} matches")
        elif verbose:
            print("skipped")

    # Deduplicate: if same PT line matched in multiple batches, keep first
    seen_pt = set()
    deduped = []
    for m in all_matches:
        if m['pt'] not in seen_pt:
            seen_pt.add(m['pt'])
            deduped.append(m)

    deduped.sort(key=lambda m: m['pt'])

    if verbose:
        print(f"  Total unique matches: {len(deduped)}/{len(pt_lines)} PT lines matched")

    return deduped
