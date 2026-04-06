"""ASS subtitle file parsing utilities."""

import re


# ─── Time helpers ────────────────────────────────────────────────────────────

def parse_time(t):
    """Parse ASS time format H:MM:SS.CC to seconds."""
    h, m, rest = t.split(':')
    s, cs = rest.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def format_time(seconds):
    """Format seconds to ASS time format H:MM:SS.CC."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    seconds -= h * 3600
    m = int(seconds // 60)
    seconds -= m * 60
    s = int(seconds)
    cs = round((seconds - s) * 100)
    if cs >= 100:
        s += 1; cs = 0
    if s >= 60:
        m += 1; s = 0
    if m >= 60:
        h += 1; m = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def strip_tags(text):
    """Remove {\\override} tags and \\N line breaks."""
    text = re.sub(r'\{[^}]*\}', '', text)
    text = text.replace('\\N', ' ').replace('\\n', ' ')
    return text.strip()


# ─── ASS line parsing ────────────────────────────────────────────────────────

DIALOGUE_RE = re.compile(
    r'^(Dialogue|Comment): (\d+),'
    r'(\d:\d{2}:\d{2}\.\d{2}),'
    r'(\d:\d{2}:\d{2}\.\d{2}),'
    r'(.*)'
)


def parse_line(line):
    """Parse a Dialogue/Comment line into components."""
    m = DIALOGUE_RE.match(line)
    if m:
        return {
            'prefix_type': m.group(1), 'layer': m.group(2),
            'start': m.group(3), 'end': m.group(4), 'rest': m.group(5),
            'start_s': parse_time(m.group(3)), 'end_s': parse_time(m.group(4)),
        }
    return None


def get_style(rest_field):
    """Extract style name from the rest field."""
    return rest_field.split(',', 1)[0]


def get_text(rest_field):
    """Extract plain text from the rest field."""
    parts = rest_field.split(',', 4)
    return strip_tags(parts[4]) if len(parts) > 4 else ''


# ─── Style detection ─────────────────────────────────────────────────────────

SPEECH_BASES = {'main', 'secondary', 'flashbacks', 'thoughts', 'narrator'}


def get_primary_speech_style(lines):
    """Auto-detect the primary speech style (e.g. 'Main-207+' or 'Main')."""
    style_counts = {}
    for line in lines:
        if not line.startswith('Dialogue:'):
            continue
        parsed = parse_line(line)
        if not parsed:
            continue
        style = get_style(parsed['rest'])
        # Handle prefixed styles like "AR-Main", "JP-Main" by checking all parts
        parts = [p.strip() for p in style.lower().split('-')]
        base = next((p for p in parts if p in SPEECH_BASES), None)
        if base:
            style_counts[style] = style_counts.get(style, 0) + 1
    if not style_counts:
        return None
    return max(style_counts, key=style_counts.get)


def get_speech_lines(lines, primary_style):
    """Extract dialogue lines matching the primary speech style."""
    result = []
    for i, line in enumerate(lines):
        if not line.startswith('Dialogue:'):
            continue
        parsed = parse_line(line)
        if parsed and get_style(parsed['rest']) == primary_style:
            result.append({
                'idx': i,
                'start_s': parsed['start_s'],
                'end_s': parsed['end_s'],
                'text': get_text(parsed['rest']),
            })
    return result
