"""Time mapping: anchor building and piecewise-linear interpolation."""

from bisect import bisect_right


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


def make_monotonic(anchors):
    """Sort, deduplicate, and enforce monotonicity on anchor pairs."""
    anchors = sorted(set(anchors))
    if not anchors:
        return anchors
    result = [anchors[0]]
    for a in anchors[1:]:
        if a[0] > result[-1][0] + 0.005 and a[1] >= result[-1][1]:
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
