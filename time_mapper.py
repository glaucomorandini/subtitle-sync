"""Time mapping: anchor building and piecewise-linear interpolation."""

from bisect import bisect_right
from statistics import median


def build_anchors(matches, pt_lines, en_lines):
    """Convert LLM match dicts to (pt_time, en_time) anchor pairs.

    For each match {pt: i, en: [j, ..., k]}:
    - anchor_start = (pt[i].start_s, en[j].start_s)
    - anchor_end = (pt[i].end_s, en[k].end_s)
    """
    anchors = []
    for m in matches:
        pt_idx = m['pt']
        en_indices = m['en']
        if not en_indices:
            continue
        pt_line = pt_lines[pt_idx]
        en_first = en_lines[en_indices[0]]
        en_last = en_lines[en_indices[-1]]
        anchors.append((pt_line['start_s'], en_first['start_s']))
        anchors.append((pt_line['end_s'], en_last['end_s']))
    return anchors


def filter_outlier_anchors(anchors, k=4, max_dev=8.0):
    """Keep an anchor if at least one of its k nearest PT-time neighbors has
    a similar offset (within max_dev seconds).

    This handles two cases correctly:
    - Isolated bad LLM matches: no neighbors agree → dropped.
    - Real video cuts: a small *cluster* of anchors with a different offset
      band still mutually support each other → all kept.

    A simple median filter would incorrectly drop the cluster as outliers,
    even though it represents the correct mapping across a scene cut.
    """
    if len(anchors) < k + 1:
        return list(anchors)
    sorted_a = sorted(anchors, key=lambda a: a[0])
    offsets = [a[1] - a[0] for a in sorted_a]
    n = len(sorted_a)
    kept = []
    for i in range(n):
        my_off = offsets[i]
        my_pt = sorted_a[i][0]
        # k nearest neighbors by PT-time distance, expanding outward
        l, r = i - 1, i + 1
        neighbor_offs = []
        while len(neighbor_offs) < k and (l >= 0 or r < n):
            if l < 0:
                neighbor_offs.append(offsets[r]); r += 1
            elif r >= n:
                neighbor_offs.append(offsets[l]); l -= 1
            elif (my_pt - sorted_a[l][0]) <= (sorted_a[r][0] - my_pt):
                neighbor_offs.append(offsets[l]); l -= 1
            else:
                neighbor_offs.append(offsets[r]); r += 1
        if any(abs(my_off - no) <= max_dev for no in neighbor_offs):
            kept.append(sorted_a[i])
    return kept


def make_monotonic(anchors, **_ignored):
    """Sort by src (PT time) and dedup near-equal src timestamps.

    Note: we deliberately do NOT enforce dst (EN time) to be non-decreasing.
    When the two video edits reorder scenes, the correct mapping is genuinely
    non-monotonic in EN time, and forcing monotonicity drops valid anchors.
    Outlier filtering already removes bad LLM matches; what remains may
    legitimately go backward in EN time across a scene reorder.
    """
    anchors = sorted(set(anchors), key=lambda a: a[0])
    if not anchors:
        return anchors
    result = [anchors[0]]
    for a in anchors[1:]:
        if a[0] <= result[-1][0] + 0.005:
            continue
        result.append(a)
    return result


def interpolate(t, src_arr, dst_arr):
    """Piecewise-linear interpolation with edge extrapolation."""
    if not src_arr:
        return t
    if t <= src_arr[0]:
        if len(src_arr) >= 2 and src_arr[1] != src_arr[0]:
            slope = (dst_arr[1] - dst_arr[0]) / (src_arr[1] - src_arr[0])
        else:
            slope = 1.0
        return dst_arr[0] + slope * (t - src_arr[0])
    if t >= src_arr[-1]:
        if len(src_arr) >= 2 and src_arr[-1] != src_arr[-2]:
            slope = (dst_arr[-1] - dst_arr[-2]) / (src_arr[-1] - src_arr[-2])
        else:
            slope = 1.0
        return dst_arr[-1] + slope * (t - src_arr[-1])
    idx = bisect_right(src_arr, t) - 1
    idx = min(idx, len(src_arr) - 2)
    t1, e1 = src_arr[idx], dst_arr[idx]
    t2, e2 = src_arr[idx + 1], dst_arr[idx + 1]
    if t2 == t1:
        return e1
    return e1 + (t - t1) / (t2 - t1) * (e2 - e1)
