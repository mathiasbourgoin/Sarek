---
name: post-edit-lint
description: Auto-detect project linter and run it on edited files — informational, non-blocking.
event: PostToolUse
matcher: Edit|Write
version: 1.0.0
timeout: 15000
async: false
requires: ["jq"]
---

# Post-Edit Lint

Runs after every Edit or Write tool call. Detects the project's linter by checking for config files and runs it on the affected file. Results are informational (exit 0 always) so edits are never blocked.

## Detected Linters

| Config File | Linter | Command |
|-------------|--------|---------|
| `.eslintrc*` / `eslint.config.*` | ESLint | `npx eslint --no-warn-ignored <file>` |
| `pyproject.toml` (with `[tool.ruff]`) | Ruff | `ruff check <file>` |
| `pyproject.toml` / `setup.cfg` (with flake8) | Flake8 | `flake8 <file>` |
| `.ocamlformat` | OCaml fmt | `dune fmt 2>&1` |
| `rustfmt.toml` / `.rustfmt.toml` | Rustfmt | `cargo fmt -- --check <file>` |
| `Cargo.toml` | Clippy | `cargo clippy -- -W warnings 2>&1` |
| `.golangci.yml` | golangci-lint | `golangci-lint run <file>` |
| `biome.json` | Biome | `npx biome check <file>` |

## Command

```command
#!/bin/bash
INPUT=$(cat -)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_result.path // empty')

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
  exit 0
fi

DIR=$(dirname "$FILE")
PROJECT_ROOT="$DIR"
while [ "$PROJECT_ROOT" != "/" ]; do
  if [ -f "$PROJECT_ROOT/package.json" ] || [ -f "$PROJECT_ROOT/pyproject.toml" ] || \
     [ -f "$PROJECT_ROOT/Cargo.toml" ] || [ -f "$PROJECT_ROOT/dune-project" ] || \
     [ -f "$PROJECT_ROOT/go.mod" ] || [ -f "$PROJECT_ROOT/.git/HEAD" ]; then
    break
  fi
  PROJECT_ROOT=$(dirname "$PROJECT_ROOT")
done

cd "$PROJECT_ROOT" 2>/dev/null || cd "$DIR"

LINTER=""
LINT_CMD=""

# JavaScript / TypeScript — ESLint
if [ -z "$LINTER" ]; then
  for cfg in .eslintrc .eslintrc.js .eslintrc.json .eslintrc.yml eslint.config.js eslint.config.mjs eslint.config.ts; do
    if [ -f "$PROJECT_ROOT/$cfg" ]; then
      LINTER="eslint"
      LINT_CMD="npx eslint --no-warn-ignored \"$FILE\" 2>&1"
      break
    fi
  done
fi

# JavaScript / TypeScript — Biome
if [ -z "$LINTER" ] && [ -f "$PROJECT_ROOT/biome.json" ]; then
  LINTER="biome"
  LINT_CMD="npx biome check \"$FILE\" 2>&1"
fi

# Python — Ruff
if [ -z "$LINTER" ] && [ -f "$PROJECT_ROOT/pyproject.toml" ] && grep -q '\[tool\.ruff\]' "$PROJECT_ROOT/pyproject.toml" 2>/dev/null; then
  LINTER="ruff"
  LINT_CMD="ruff check \"$FILE\" 2>&1"
fi

# Python — Flake8
if [ -z "$LINTER" ]; then
  if ([ -f "$PROJECT_ROOT/setup.cfg" ] && grep -q '\[flake8\]' "$PROJECT_ROOT/setup.cfg" 2>/dev/null) || \
     ([ -f "$PROJECT_ROOT/.flake8" ]); then
    LINTER="flake8"
    LINT_CMD="flake8 \"$FILE\" 2>&1"
  fi
fi

# OCaml
if [ -z "$LINTER" ] && [ -f "$PROJECT_ROOT/.ocamlformat" ]; then
  LINTER="dune fmt"
  LINT_CMD="dune fmt 2>&1"
fi

# Rust — fmt + clippy
if [ -z "$LINTER" ] && ([ -f "$PROJECT_ROOT/rustfmt.toml" ] || [ -f "$PROJECT_ROOT/.rustfmt.toml" ] || [ -f "$PROJECT_ROOT/Cargo.toml" ]); then
  LINTER="cargo fmt"
  LINT_CMD="cargo fmt -- --check 2>&1"
fi

# Go
if [ -z "$LINTER" ] && [ -f "$PROJECT_ROOT/.golangci.yml" ]; then
  LINTER="golangci-lint"
  LINT_CMD="golangci-lint run \"$FILE\" 2>&1"
fi

if [ -z "$LINTER" ]; then
  exit 0
fi

echo "--- post-edit-lint: running $LINTER ---"
OUTPUT=$(eval "$LINT_CMD" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] && [ -n "$OUTPUT" ]; then
  echo "$OUTPUT" | head -30
  echo "--- $LINTER found issues (informational, not blocking) ---"
else
  echo "--- $LINTER: clean ---"
fi

# Always exit 0 — this hook is informational, never blocks edits
exit 0
```

## Installed As

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "#!/bin/bash\nINPUT=$(cat -)\nFILE=$(echo \"$INPUT\" | jq -r '.tool_input.file_path // .tool_result.path // empty')\nif [ -z \"$FILE\" ] || [ ! -f \"$FILE\" ]; then exit 0; fi\nDIR=$(dirname \"$FILE\")\nPROJECT_ROOT=\"$DIR\"\nwhile [ \"$PROJECT_ROOT\" != \"/\" ]; do\n  if [ -f \"$PROJECT_ROOT/package.json\" ] || [ -f \"$PROJECT_ROOT/pyproject.toml\" ] || [ -f \"$PROJECT_ROOT/Cargo.toml\" ] || [ -f \"$PROJECT_ROOT/dune-project\" ] || [ -f \"$PROJECT_ROOT/go.mod\" ] || [ -f \"$PROJECT_ROOT/.git/HEAD\" ]; then break; fi\n  PROJECT_ROOT=$(dirname \"$PROJECT_ROOT\")\ndone\ncd \"$PROJECT_ROOT\" 2>/dev/null || cd \"$DIR\"\nLINTER=\"\"\nLINT_CMD=\"\"\nfor cfg in .eslintrc .eslintrc.js .eslintrc.json .eslintrc.yml eslint.config.js eslint.config.mjs eslint.config.ts; do\n  if [ -f \"$PROJECT_ROOT/$cfg\" ]; then LINTER=\"eslint\"; LINT_CMD=\"npx eslint --no-warn-ignored \\\"$FILE\\\" 2>&1\"; break; fi\ndone\nif [ -z \"$LINTER\" ] && [ -f \"$PROJECT_ROOT/biome.json\" ]; then LINTER=\"biome\"; LINT_CMD=\"npx biome check \\\"$FILE\\\" 2>&1\"; fi\nif [ -z \"$LINTER\" ] && [ -f \"$PROJECT_ROOT/pyproject.toml\" ] && grep -q '\\[tool\\.ruff\\]' \"$PROJECT_ROOT/pyproject.toml\" 2>/dev/null; then LINTER=\"ruff\"; LINT_CMD=\"ruff check \\\"$FILE\\\" 2>&1\"; fi\nif [ -z \"$LINTER\" ] && ([ -f \"$PROJECT_ROOT/.flake8\" ] || ([ -f \"$PROJECT_ROOT/setup.cfg\" ] && grep -q '\\[flake8\\]' \"$PROJECT_ROOT/setup.cfg\" 2>/dev/null)); then LINTER=\"flake8\"; LINT_CMD=\"flake8 \\\"$FILE\\\" 2>&1\"; fi\nif [ -z \"$LINTER\" ] && [ -f \"$PROJECT_ROOT/.ocamlformat\" ]; then LINTER=\"dune fmt\"; LINT_CMD=\"dune fmt 2>&1\"; fi\nif [ -z \"$LINTER\" ] && ([ -f \"$PROJECT_ROOT/rustfmt.toml\" ] || [ -f \"$PROJECT_ROOT/.rustfmt.toml\" ] || [ -f \"$PROJECT_ROOT/Cargo.toml\" ]); then LINTER=\"cargo fmt\"; LINT_CMD=\"cargo fmt -- --check 2>&1\"; fi\nif [ -z \"$LINTER\" ] && [ -f \"$PROJECT_ROOT/.golangci.yml\" ]; then LINTER=\"golangci-lint\"; LINT_CMD=\"golangci-lint run \\\"$FILE\\\" 2>&1\"; fi\nif [ -z \"$LINTER\" ]; then exit 0; fi\necho \"--- post-edit-lint: running $LINTER ---\"\nOUTPUT=$(eval \"$LINT_CMD\" 2>&1)\nEXIT_CODE=$?\nif [ $EXIT_CODE -ne 0 ] && [ -n \"$OUTPUT\" ]; then echo \"$OUTPUT\" | head -30; echo \"--- $LINTER found issues (informational) ---\"; else echo \"--- $LINTER: clean ---\"; fi\nexit 0",
            "description": "Auto-detect linter and run on edited files (informational)"
          }
        ]
      }
    ]
  }
}
```
