---
name: kb-reindex
description: Build or update the LanceDB semantic search index for KB files — opt-in, cold-start or incremental.
version: 1.0.0
---

# KB Reindex

You build or incrementally update a LanceDB vector index over the project's `kb/` directory. This skill is **opt-in** — only run when `search_index: true` is set in the project's kb-agent tunable, or when explicitly invoked.

The index is a search cache. Markdown files remain the primary source of truth. The index is never committed to git.

## Pre-conditions

```bash
# Check opt-in flag (read from harness.json or inform user to set it)
# Check LanceDB is available (Python: pip install lancedb; JS: npm install @lancedb/lancedb)
[ -d kb ] && echo "KB present" || echo "KB absent — nothing to index"
```

If `kb/` is absent: report and stop.

**Migration warning**: if any KB file lacks `schema-version: 2` in frontmatter, emit:
> ⚠️ Some KB files have not been migrated to schema v2. Run `/kb-migrate` first for best results. Continuing anyway — legacy files will be indexed with status inferred from old values.

## Steps

### Step 1 — Detect mode

```bash
[ -d kb/.index ] && echo "incremental" || echo "cold-start"
```

**Cold start**: `kb/.index/` does not exist → full initial index.
**Incremental**: `kb/.index/` exists → upsert only changed files. If invoked in incremental mode *without* a changed-files list (e.g., called manually by user), fall back to cold start: delete and rebuild the full index.

---

### Step 2 — Cold start (full reindex)

1. Walk all `kb/**/*.md` files. Exclude:
   - `kb/.index/` (the index directory itself)
   - `kb/reports/` (audit reports — operational, not normative)

2. For each file, chunk by section:
   - Split content on `## ` heading boundaries
   - Each chunk: `{file_path, section_heading, content, status, schema_version}`
   - Infer `status` from frontmatter (`status:` field). Map legacy values: `draft`/`reviewed` → `live-doctrine`, `stale` → `historical`.

3. Embed each chunk using the configured embedding model (default: `text-embedding-3-small`, 1536 dimensions). Batch embed for efficiency.

4. Store in LanceDB table `kb_chunks`:

   | Column | Type | Description |
   |--------|------|-------------|
   | `id` | string | `<file_path>#<section_heading>` |
   | `file_path` | string | Relative path from kb/ root |
   | `section` | string | Section heading (or `"__preamble__"` for content before first `##`) |
   | `content` | string | Raw text of the chunk |
   | `status` | string | Mapped status value |
   | `schema_version` | int | 1 or 2 |
   | `embedding` | vector(1536) | Float32 embedding vector |

5. Write `kb/.index/.gitignore` containing `*` (ensures index is never committed).

6. Report:
   ```
   ✓ KB reindex complete — N files, M chunks, model: text-embedding-3-small
   ```

---

### Step 3 — Incremental update

Accepts: list of changed file paths (relative to repo root).

For each changed file:
1. Delete all existing chunks where `file_path` matches.
2. If file still exists: re-chunk and re-embed → upsert.
3. If file was deleted: only deletion needed.

Report:
```
✓ KB reindex (incremental) — N files updated, M chunks upserted/deleted
```

---

### Tunables (document in harness.json)

| Tunable | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `embedding_dim` | `1536` | Vector dimensions (must match model) |
| `chunk_by` | `section` | Chunking strategy: `section` (by `## ` heading) or `fixed` (N chars) |
| `index_dir` | `kb/.index` | LanceDB storage path |
| `exclude_dirs` | `[reports]` | KB subdirectories to exclude |

## Rules

- Index is never the source of truth — always re-derive from markdown if in doubt
- Never commit `kb/.index/` to git
- Re-embedding on model change requires a full cold-start reindex (old vectors are incompatible)
- Incremental mode must delete before upsert to avoid stale chunk accumulation
