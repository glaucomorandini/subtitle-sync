#!/usr/bin/env python3
"""
Sync subtitle timings to match a reference subtitle file using LLM matching.

Usage: python3 sync_subs.py <reference_file> <target_file> [--model MODEL] [--dry-run]

The reference file has correct timing. The target file will be remapped
to match. A .bak backup of the target is always created before writing.
"""

import argparse
import shutil
import sys

from ass_parser import (
    format_time, get_all_speech_styles, get_primary_speech_style,
    get_speech_lines, parse_line, parse_time,
)
from llm_matcher import match_lines, BACKENDS, DEFAULT_BACKEND
from time_mapper import build_anchors, filter_outlier_anchors, interpolate, make_monotonic


def read_lines(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return f.readlines()


def remap_file(lines, src_arr, dst_arr, direct_overrides=None,
               max_stretch=3.0, min_orig_dur=0.5):
    """Remap timestamps in all Dialogue/Comment lines.

    `direct_overrides` maps original-file line-index -> (new_start, new_end)
    seconds. Lines listed there are snapped exactly to the override (no
    interpolation, no stretch cap). Used for PT lines with a confident LLM
    match so that scene-reordered lines land on the correct EN timing
    instead of being smoothed away by anchor interpolation.
    """
    from ass_parser import DIALOGUE_RE

    overrides = direct_overrides or {}
    new_lines = []
    for i, line in enumerate(lines):
        m = DIALOGUE_RE.match(line)
        if m:
            old_start = parse_time(m.group(3))
            old_end = parse_time(m.group(4))
            old_dur = max(0.0, old_end - old_start)
            if i in overrides:
                new_start, new_end = overrides[i]
            else:
                new_start = interpolate(old_start, src_arr, dst_arr)
                new_end = interpolate(old_end, src_arr, dst_arr)
                cap_basis = max(old_dur, min_orig_dur)
                max_new_dur = cap_basis * max_stretch
                if (new_end - new_start) > max_new_dur:
                    new_end = new_start + max_new_dur
            # Ensure end >= start
            if new_end < new_start:
                new_end = new_start + 0.02
            new_line = (
                f"{m.group(1)}: {m.group(2)},"
                f"{format_time(new_start)},"
                f"{format_time(new_end)},"
                f"{m.group(5)}\n"
            )
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return new_lines


def print_verification(matches, pt_lines, en_lines, src_arr, dst_arr):
    """Print sample matched lines with timing deltas."""
    print("\n--- Verification (sample matched lines) ---")
    print(f"{'PT text':<40} {'PT old':>10} {'PT new':>10} {'EN ref':>10} {'Delta':>7}")
    print("-" * 82)

    # Sample evenly across matches
    step = max(1, len(matches) // 12)
    for i in range(0, len(matches), step):
        m = matches[i]
        pt = pt_lines[m['pt']]
        en_first = en_lines[m['en'][0]]

        old_t = pt['start_s']
        new_t = interpolate(old_t, src_arr, dst_arr)
        ref_t = en_first['start_s']
        delta = abs(new_t - ref_t)

        text = pt['text'][:38]
        marker = " !!" if delta > 2.0 else ""
        print(f"{text:<40} {format_time(old_t):>10} {format_time(new_t):>10} {format_time(ref_t):>10} {delta:>6.2f}s{marker}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Sync subtitle timings using LLM matching.")
    parser.add_argument("reference", help="Reference subtitle file (correct timing)")
    parser.add_argument("target", help="Target subtitle file (to be remapped)")
    parser.add_argument("--backend", choices=sorted(BACKENDS.keys()), default=DEFAULT_BACKEND,
                        help=f"LLM backend (default: {DEFAULT_BACKEND})")
    parser.add_argument("--model", required=True,
                        help="Model name, required. Examples: "
                             "openai/gpt-oss-120b:free (openrouter), "
                             "llama-3.3-70b-versatile (groq), "
                             "qwen3.5:4b (local)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of LLM batches to run in parallel (default: 5)")
    parser.add_argument("--batches", type=int, default=None,
                        help="Override number of batches (default: adaptive based on line count)")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without writing")
    parser.add_argument("--cache", default=None, help="JSON file to load/save LLM matches (skips API call if file exists)")
    parser.add_argument("--debug-pt", type=float, default=None, help="Print anchors near this PT timestamp (seconds)")
    args = parser.parse_args()

    # Read files
    print(f"Reference: {args.reference}")
    print(f"Target:    {args.target}")

    ref_lines = read_lines(args.reference)
    tgt_lines = read_lines(args.target)

    # Detect speech styles
    ref_styles = get_all_speech_styles(ref_lines)
    tgt_styles = get_all_speech_styles(tgt_lines)
    print(f"Reference speech styles: {', '.join(sorted(ref_styles))}")
    print(f"Target speech styles:    {', '.join(sorted(tgt_styles))}")

    if not ref_styles or not tgt_styles:
        print("Error: Could not detect speech styles in one or both files.", file=sys.stderr)
        sys.exit(1)

    # Extract speech lines from all speech styles
    ref_speech = get_speech_lines(ref_lines, ref_styles)
    tgt_speech = get_speech_lines(tgt_lines, tgt_styles)
    print(f"Reference speech lines: {len(ref_speech)}")
    print(f"Target speech lines:    {len(tgt_speech)}")

    # LLM matching (with optional cache)
    import json, os
    if args.cache and os.path.exists(args.cache):
        print(f"\nLoading matches from cache: {args.cache}")
        with open(args.cache) as f:
            matches = json.load(f)
    else:
        print("\nMatching lines...")
        matches = match_lines(tgt_speech, ref_speech, backend=args.backend, model=args.model,
                              concurrency=args.concurrency, num_batches=args.batches)
        if args.cache:
            with open(args.cache, 'w') as f:
                json.dump(matches, f)
            print(f"Cached matches to {args.cache}")

    if not matches:
        print("Error: No matches found.", file=sys.stderr)
        sys.exit(1)

    # Drop low-confidence matches before building anchors
    high_conf = [m for m in matches if m.get('c', 'high') != 'low']
    n_low = len(matches) - len(high_conf)
    if n_low:
        print(f"Dropped {n_low} low-confidence matches")

    # Build anchors, filter outliers (sliding-median offset filter), then enforce monotonicity
    anchors = build_anchors(high_conf, tgt_speech, ref_speech)
    n_raw = len(anchors)
    anchors = filter_outlier_anchors(anchors)
    n_after_outlier = len(anchors)
    anchors = make_monotonic(anchors)
    print(f"Anchor points: {len(anchors)} (raw={n_raw}, after_outlier={n_after_outlier})")

    if args.debug_pt is not None:
        center = args.debug_pt
        print(f"\n--- Anchors near PT {center}s ---")
        # Show all matches whose pt line is within 30s of center
        for m in matches:
            pt_line = tgt_speech[m['pt']]
            if abs(pt_line['start_s'] - center) <= 30:
                en_first = ref_speech[m['en'][0]]
                en_last = ref_speech[m['en'][-1]]
                print(f"  PT {pt_line['start_s']:.1f}s '{pt_line['text'][:50]}'")
                print(f"     -> EN {en_first['start_s']:.1f}..{en_last['end_s']:.1f}s '{en_first['text'][:50]}' c={m.get('c','?')}")
        print(f"\n--- Surviving anchors near PT {center}s (after outlier+monotonic) ---")
        for src, dst in anchors:
            if abs(src - center) <= 30:
                print(f"  ({src:.2f}, {dst:.2f})  offset={dst-src:+.1f}s")

    src_arr = [a[0] for a in anchors]
    dst_arr = [a[1] for a in anchors]

    # Verification
    print_verification(matches, tgt_speech, ref_speech, src_arr, dst_arr)

    if args.dry_run:
        print("Dry run — no files written.")
        return

    # Backup and write
    bak_path = args.target + ".bak"
    shutil.copy2(args.target, bak_path)
    print(f"Backup: {bak_path}")

    # Build direct timing overrides from confident matches: each PT line that
    # the LLM mapped to one or more EN lines gets snapped to the EN line(s)'
    # exact start/end. This handles scene reorderings (where a single PT line
    # lives far from its EN counterpart) — anchor interpolation can't, since
    # such anchors look like outliers and get filtered.
    direct_overrides = {}
    for m in high_conf:
        en_idx = m['en']
        if not en_idx:
            continue
        pt_line = tgt_speech[m['pt']]
        en_first = ref_speech[en_idx[0]]
        en_last = ref_speech[en_idx[-1]]
        direct_overrides[pt_line['idx']] = (en_first['start_s'], en_last['end_s'])
    print(f"Direct timing overrides from matches: {len(direct_overrides)}")

    new_lines = remap_file(tgt_lines, src_arr, dst_arr, direct_overrides=direct_overrides)
    with open(args.target, 'w', encoding='utf-8-sig') as f:
        f.writelines(new_lines)

    print(f"Written: {args.target}")
    print("Done.")


if __name__ == '__main__':
    main()
