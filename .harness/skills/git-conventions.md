---
description: Apply git workflow conventions — commits, branches, PRs.
version: 1.0.0
---

# Git Conventions

Apply standardized git conventions for the action described in $ARGUMENTS (e.g., "commit this", "create PR", "new branch for auth feature").

## Commit Format

```
<type>: <description>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

- Description: imperative mood, lowercase, no period, under 72 chars.
- Body (optional): blank line after subject, wrap at 80 chars, explain **why** not what.
- Footer (optional): `Closes #123`, `BREAKING CHANGE: <description>`.

Examples:
- `feat: add JWT authentication middleware`
- `fix: prevent null pointer in user lookup`
- `refactor: extract validation into shared module`

## Branch Naming

```
<type>/<short-description>
```

- Kebab-case description, max 4 words.
- Examples: `feat/add-auth`, `fix/null-pointer`, `refactor/extract-validation`
- For issue-linked work: `feat/123-add-auth`

## PR Workflow

When creating a PR:

1. Ensure the branch is pushed with `git push -u origin <branch>`.
2. Use `gh pr create` with this template:

```markdown
## Summary
- <1-3 bullet points describing what and why>

## Test plan
- [ ] <specific verification step>
- [ ] <edge case checked>
- [ ] <regression check>
```

3. Title: same format as commit subject (`<type>: <description>`), under 70 chars.
4. Request reviewers if the user specifies them.

## Commit Workflow

When committing:

1. Run `git status` and `git diff --staged` to understand what's staged.
2. If nothing staged, help the user stage relevant files (prefer explicit paths over `git add .`).
3. Draft commit message following the format above.
4. Use a HEREDOC for the message:
   ```bash
   git commit -m "$(cat <<'EOF'
   <type>: <description>
   EOF
   )"
   ```
5. Run `git status` after to confirm success.

## Rules

- **Never** force-push to `main` or `master`.
- **Always** push with `-u` to set upstream tracking.
- **Never** use `git add .` or `git add -A` — stage specific files.
- **Never** skip pre-commit hooks (`--no-verify`).
- **Never** commit `.env`, credentials, or secrets — warn the user if these are staged.
- PR descriptions must be comprehensive — reviewers should understand the change without reading code.
- One logical change per commit. Split unrelated changes into separate commits.
