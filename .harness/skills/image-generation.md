---
name: image-generation
description: Generate or edit images via Codex CLI — with prompt refinement, vision validation, retry loop, and error handling.
version: 1.0.0
tags: [image-generation, assets, media, codex, vision]
domain: [frontend, content, media, documentation]
---

# Image Generation

Generate or edit an image using the Codex CLI based on $ARGUMENTS.

Parse $ARGUMENTS as one of:
- `"<prompt>"` — generate a new image
- `"edit <path> <prompt>"` — edit an existing image
- Optional inline hints: `size:<WxH>`, `format:<png|jpg|webp>`, `out:<path>`, `style:<hint>`

Examples:
- `"a watercolor moonlit library"`
- `"edit assets/hero.png replace background with white"`
- `"a minimalist OCaml logo size:512x512 out:docs/logo.png"`

## Step 1 — Parse and Resolve Arguments

Extract from $ARGUMENTS:
- `MODE`: `generate` or `edit`
- `RAW_PROMPT`: the user's original description
- `INPUT_PATH`: (edit mode only) path to source image — verify it exists; abort if not
- `SIZE`: default `1024x1024`; override if `size:WxH` present
- `FORMAT`: default `png`; override if `format:X` present
- `OUTPUT_PATH`: if `out:<path>` present, use it; otherwise ask the user where to save the result

If `OUTPUT_PATH` is missing and $ARGUMENTS provides no hint, ask:
> "Where should I save the image? (e.g., `assets/hero.png`)"

Do not proceed until `OUTPUT_PATH` is resolved.

## Step 2 — Prompt Refinement

Before invoking Codex, rewrite `RAW_PROMPT` into a detailed, visually precise prompt:

- Add composition, lighting, mood, and style details the user left implicit
- Incorporate `SIZE` as an aspect ratio hint (e.g., "square 1:1 composition")
- Incorporate `FORMAT` only if it has visual implications (e.g., transparency for PNG)
- Keep the user's intent intact — do not substitute a different concept

Store the result as `REFINED_PROMPT`. Show the user both the original and refined version:
> Original: `<RAW_PROMPT>`
> Refined: `<REFINED_PROMPT>`

If the user rejects the refinement, use `RAW_PROMPT` as-is.

## Step 3 — Generate

Set `ATTEMPT=1`, `MAX_ATTEMPTS=3`.

**Generate mode:**
```
codex exec --full-auto "Generate a <SIZE> <FORMAT> image of: <REFINED_PROMPT>. Save the result to <OUTPUT_PATH>."
```

**Edit mode:**
```
codex exec --full-auto "Edit the image at <INPUT_PATH>: <REFINED_PROMPT>. Save the result to <OUTPUT_PATH>."
```

Capture both stdout and stderr. On completion, proceed to Step 4.

### Rate Limit Handling

If stderr contains `rate limit`, `429`, or `quota`:
- Notify the user: "Rate limit hit — waiting <WAIT>s before retry."
- Wait: `WAIT = 15 * ATTEMPT` seconds (15s, 30s, 45s)
- Retry the same command. Do not increment `ATTEMPT` for rate limit retries — at most 3 rate limit waits per attempt.
- If rate limit persists after 3 waits, abort and report: "Rate limit not cleared after 3 retries. Try again later."

### Fatal Error Handling

Abort immediately (do not retry) if stderr contains:
- `command not found` / `codex: not found` → "Codex CLI not installed. Run: `npm install -g @openai/codex`"
- `authentication` / `unauthorized` / `401` → "Codex authentication failed. Check your credentials."
- `not found` on `INPUT_PATH` (edit mode) → "Source image not found at `<INPUT_PATH>`."

## Step 4 — Validate

### 4a. File existence check

Verify `OUTPUT_PATH` exists and is non-empty (size > 0 bytes).

If missing or empty:
- This counts as a failed attempt. Increment `ATTEMPT`.
- If `ATTEMPT <= MAX_ATTEMPTS`, refine the prompt further (add more explicit detail) and go to Step 3.
- If `ATTEMPT > MAX_ATTEMPTS`, abort: "Codex did not produce an output file after 3 attempts."

### 4b. Partial file check

Read the file. If the image cannot be decoded (corrupted header, truncated):
- Treat as failed attempt. Same retry logic as 4a.

### 4c. Vision quality check

Inspect the generated image visually:
- Does the image depict what `RAW_PROMPT` requested?
- Are there obvious artifacts, blank regions, or clearly wrong content?
- Does the composition match the aspect ratio implied by `SIZE`?

Score the result: **pass** / **soft-fail** / **hard-fail**

| Result | Condition | Action |
|--------|-----------|--------|
| pass | Intent clearly satisfied | Proceed to Step 5 |
| soft-fail | Mostly correct but minor issues | Show user, offer retry or keep |
| hard-fail | Wrong content or unusable | Auto-retry with corrected prompt |

On **soft-fail**, present the image path and ask:
> "The image was generated but may not fully match your intent. Keep it, or retry with adjustments?"

On **hard-fail**, identify the specific mismatch, adjust `REFINED_PROMPT` to correct it, increment `ATTEMPT`, and go to Step 3 if `ATTEMPT <= MAX_ATTEMPTS`.

If `ATTEMPT > MAX_ATTEMPTS` on hard-fail:
- Save the last result anyway
- Report: "Could not fully satisfy the prompt after 3 attempts. Saved best result to `<OUTPUT_PATH>`. Issues: `<mismatch description>`."

## Step 5 — Log

Append a JSON entry to `.imagelog.json` in the project root (create if absent):

```json
{
  "timestamp": "<ISO 8601>",
  "mode": "<generate|edit>",
  "raw_prompt": "<RAW_PROMPT>",
  "refined_prompt": "<REFINED_PROMPT>",
  "output_path": "<OUTPUT_PATH>",
  "size": "<SIZE>",
  "format": "<FORMAT>",
  "attempts": <ATTEMPT>,
  "outcome": "<pass|soft-fail|hard-fail|aborted>"
}
```

If `.imagelog.json` exists, parse it as an array and append. If it does not exist, create it as a single-element array.

## Step 6 — Report

On success:
> "Image saved to `<OUTPUT_PATH>` (attempt <ATTEMPT>/<MAX_ATTEMPTS>). Logged to `.imagelog.json`."

On abort: summarize what failed and what the user should do next.

## Rules

- Never invoke Codex without a resolved `OUTPUT_PATH`.
- Never skip vision validation — even if the file exists and is non-empty.
- Never hard-retry more than `MAX_ATTEMPTS` times total (rate limit retries are separate).
- Never overwrite an existing file without confirming with the user first.
- Never expose raw Codex stderr to the user — translate errors into plain language.
- If the user explicitly says "skip validation", proceed directly to Step 5 after Step 3.
- `.imagelog.json` must always be updated, even on failure (log the abort).
