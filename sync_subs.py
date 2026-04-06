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
    format_time, get_primary_speech_style, get_speech_lines, parse_line, parse_time,
)
from llm_matcher import match_lines, DEFAULT_MODEL
from time_mapper import build_anchors, interpolate, make_monotonic


def read_lines(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return f.readlines()


def remap_file(lines, src_arr, dst_arr):
    """Remap timestamps in all Dialogue/Comment lines."""
    import re
    from ass_parser import DIALOGUE_RE

    new_lines = []
    for line in lines:
        m = DIALOGUE_RE.match(line)
        if m:
            old_start = parse_time(m.group(3))
            old_end = parse_time(m.group(4))
            new_start = interpolate(old_start, src_arr, dst_arr)
            new_end = interpolate(old_end, src_arr, dst_arr)
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without writing")
    args = parser.parse_args()

    # Read files
    print(f"Reference: {args.reference}")
    print(f"Target:    {args.target}")

    ref_lines = read_lines(args.reference)
    tgt_lines = read_lines(args.target)

    # Detect speech styles
    ref_style = get_primary_speech_style(ref_lines)
    tgt_style = get_primary_speech_style(tgt_lines)
    print(f"Reference speech style: {ref_style}")
    print(f"Target speech style:    {tgt_style}")

    if not ref_style or not tgt_style:
        print("Error: Could not detect speech style in one or both files.", file=sys.stderr)
        sys.exit(1)

    # Extract speech lines
    ref_speech = get_speech_lines(ref_lines, ref_style)
    tgt_speech = get_speech_lines(tgt_lines, tgt_style)
    print(f"Reference speech lines: {len(ref_speech)}")
    print(f"Target speech lines:    {len(tgt_speech)}")

    # LLM matching
    print("\nMatching lines...")
    matches = match_lines(tgt_speech, ref_speech, model=args.model)

    if not matches:
        print("Error: No matches found.", file=sys.stderr)
        sys.exit(1)

    # Build anchors
    anchors = build_anchors(matches, tgt_speech, ref_speech)
    anchors = make_monotonic(anchors)
    print(f"Anchor points: {len(anchors)}")

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

    new_lines = remap_file(tgt_lines, src_arr, dst_arr)
    with open(args.target, 'w', encoding='utf-8-sig') as f:
        f.writelines(new_lines)

    print(f"Written: {args.target}")
    print("Done.")


if __name__ == '__main__':
    main()
