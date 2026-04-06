# subtitle-sync

Sync subtitle timings from one language to match a reference subtitle file using LLM semantic matching.

Common use case: you have a well-timed subtitle in one language (e.g. English, Arabic) and a translation in another language (e.g. Portuguese) timed for a different video edit. This tool remaps the translation's timestamps to match the reference, so both files are in sync with the same video.

Works with any language pair and any content — anime, movies, series, etc.

---

## How it works

### The problem

Two subtitle files covering the same content but produced from different video edits will have different timestamps. A simple offset doesn't work because the drift is not constant — it varies throughout the episode due to cuts, openings, recaps, and pacing differences between versions. On top of that, translators often merge or split lines differently between languages, making direct line-by-line comparison unreliable.

### The solution: LLM semantic matching

Instead of matching lines by position or keywords, the tool sends batches of subtitle text from both files to an LLM (via OpenRouter) and asks it to match lines by meaning — across languages, regardless of how they were split or merged.

### Pipeline

```
1. Parse both .ass files
2. Auto-detect the primary speech style in each file
3. Extract speech dialogue lines with timestamps
4. Split the episode into ~10 time windows
5. For each window: send PT + EN lines to the LLM, get back JSON match mappings
6. Build (source_time → reference_time) anchor pairs from the matches
7. Piecewise-linear interpolation to remap ALL lines in the target file
8. Backup original, write synced file
```

### Time anchoring

Each LLM match produces two anchor points:

- `target_line.start` → `reference_first_matched_line.start`
- `target_line.end` → `reference_last_matched_line.end`

This naturally handles merged/split lines. If Portuguese merges two English lines into one, the Portuguese line's start maps to the first English line's start and its end maps to the second English line's end.

All anchors are then sorted, deduplicated, and made monotonic. The interpolation engine uses piecewise-linear mapping between anchors, with linear extrapolation at the edges.

### What gets remapped

Every `Dialogue:` and `Comment:` line in the target file gets its timestamps remapped — not just speech, but also typesetting, karaoke, signs, captions, and credits.

### Partial batch failures

The tool splits the episode into ~10 time windows and calls the LLM once per window. If a batch fails (e.g. rate limit timeout), the remaining successful batches still provide anchors across the rest of the episode. The interpolation fills in the gaps. A few failed batches will not break the sync — only a large consecutive block of failures in the same region would cause drift there.

---

## Requirements

- Python 3.8+
- No external dependencies — uses only Python standard library
- An [OpenRouter](https://openrouter.ai) API key

---

## Setup

```bash
export OPENROUTER_API_KEY="your-key-here"
```

---

## Usage

```bash
python3 sync_subs.py <reference_file> <target_file> [--model MODEL] [--dry-run]
```

| Argument | Description |
|---|---|
| `reference_file` | Subtitle file with correct timing |
| `target_file` | Subtitle file to be remapped (will be modified in place) |
| `--model` | OpenRouter model ID (default: `qwen/qwen3.6-plus:free`) |
| `--dry-run` | Show matching and verification without writing any files |

### Examples

```bash
# Sync a Portuguese subtitle to match an English reference
python3 sync_subs.py "episode_en.ass" "episode_ptbr.ass"

# Sync using an Arabic reference
python3 sync_subs.py "episode_ar.ass" "episode_ptbr.ass"

# Preview without writing
python3 sync_subs.py "episode_en.ass" "episode_ptbr.ass" --dry-run

# Use a different model
python3 sync_subs.py "episode_en.ass" "episode_ptbr.ass" --model "deepseek/deepseek-chat-v3-0324:free"
```

### Backup

The original target file is always backed up as `<target_file>.bak` before any changes are written. The `--dry-run` flag skips writing entirely.

---

## Output

```
Reference: enieslobby 04 en.ass
Target:    04 Enies Lobby.ass
Reference speech style: Main-207+
Target speech style:    Main
Reference speech lines: 309
Target speech lines:    305

Matching lines...
  LLM matching: 10 batches, model=qwen/qwen3.6-plus:free
  Batch 1/10: 53 PT + 57 EN lines... 43 matches
  Batch 2/10: 45 PT + 54 EN lines... 40 matches
  ...
  Total unique matches: 280/305 lines matched

Anchor points: 452

--- Verification (sample matched lines) ---
PT text                                      PT old     PT new     EN ref   Delta
----------------------------------------------------------------------------------
E-Ei! Eu sei quem é esse espadachim      0:01:56.54 0:01:55.22 0:01:55.22   0.00s
Parem com isso! Eu sou uma idosa!        0:03:17.60 0:02:59.83 0:02:59.83   0.00s
...

Backup: 04 Enies Lobby.ass.bak
Written: 04 Enies Lobby.ass
Done.
```

The verification table samples ~12 matched lines spread across the episode, showing the original timestamp, remapped timestamp, reference timestamp, and the delta. Lines with delta > 2s are flagged with `!!`.

---

## File structure

```
subtitle-sync/
├── sync_subs.py      # CLI entry point and orchestration
├── ass_parser.py     # .ass file parsing (time helpers, style detection, line extraction)
├── llm_matcher.py    # LLM batching, prompting, response parsing via OpenRouter
├── time_mapper.py    # Anchor building and piecewise-linear interpolation
└── README.md
```

---

## Supported models (OpenRouter)

Any chat model on OpenRouter works. Free tier options:

| Model | Notes |
|---|---|
| `qwen/qwen3.6-plus:free` | Default. Strong multilingual, good structured output |
| `deepseek/deepseek-chat-v3-0324:free` | Strong at JSON output |
| `google/gemini-2.5-flash:free` | Fast |

Free tier models have rate limits. The tool handles 429 errors automatically by waiting and retrying indefinitely until the batch succeeds.

---

## Supported subtitle format

`.ass` (Advanced SubStation Alpha) only. The tool handles:

- Style auto-detection including prefixed styles (`AR-Main`, `Main-207+`, `Main`, etc.)
- All speech style bases: `Main`, `Secondary`, `Flashbacks`, `Thoughts`, `Narrator`
- Override tags (`{\i1}`, `{\pos(...)}`, etc.) are stripped for matching but preserved in output
- All line types are remapped: dialogue, typesetting, karaoke, signs, captions, credits
