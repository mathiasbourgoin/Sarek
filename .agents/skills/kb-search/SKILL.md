---
name: kb-search
description: Hybrid semantic+keyword search over the KB LanceDB index — returns ranked chunks with source, section, and status.
version: 1.0.0
---

# KB Search

You perform hybrid semantic and keyword search over the project KB's LanceDB index. Use this skill instead of loading all KB files when searching for a specific concept, term, or constraint.

**Prerequisite:** `kb/.index/` must exist. Run `/kb-reindex` first if not.

## Steps

### Step 1 — Pre-condition check

```bash
[ -d kb/.index ] && echo "index: present" || echo "index: absent"
```

If absent:
> ⛔ No KB index found at `kb/.index/`. Run `/kb-reindex` to build the index first.

---

### Step 2 — Execute hybrid search

Accept:
- `query` (string): the search query
- `top_k` (int, default: 5): number of results to return
- `include_all_statuses` (bool, default: false): if false, filter out `historical` and `superseded` chunks

**Hybrid search algorithm:**

1. **Vector search**: embed `query` with the same model used for indexing. Compute cosine similarity against all chunks in `kb_chunks`. Retrieve top `top_k * 3` candidates.

2. **Keyword search**: tokenize `query`. Score each candidate with BM25 against its `content` field.

3. **Score fusion**: combine vector score (weight 0.7) and BM25 score (weight 0.3). Re-rank by fused score.

4. **Status filter**: if `include_all_statuses = false`, remove chunks where `status` is `historical` or `superseded`. Apply this as a **pre-filter on the vector query** (not post-fusion) so that `top_k * 3` candidates are drawn only from live-doctrine and derived chunks, ensuring `top_k` results are always available when they exist.

5. Return top `top_k` results.

---

### Step 3 — Format results

Return a ranked list:

```
KB Search Results for: "<query>"
Showing top N results (historical/superseded filtered)

1. [score: 0.92] kb/properties.md § "Invariants"
   > "All database writes must be wrapped in explicit transactions..."
   Status: live-doctrine

2. [score: 0.87] kb/spec.md § "API Contract"
   > "The API must return errors in the format {error: string, code: int}..."
   Status: live-doctrine

3. [score: 0.61] kb/architecture.md § "Storage Layer"
   > "PostgreSQL is the primary storage backend..."
   Status: live-doctrine
```

**Low confidence notice**: if the top result's fused score is below 0.3, or if fewer than `top_k` results were returned (can happen when most KB content is filtered out):
> ⚠️ Low confidence results — no strong match found for "<query>". Consider: broader query, running `/kb-reindex` to refresh stale embeddings, or checking if the concept is documented at all.

---

### Step 4 — Follow-up

After returning results, offer:
> "Open any of these files for full context? Or refine the query?"

## Rules

- Never return `historical` or `superseded` chunks as primary results (unless `include_all_statuses: true`)
- Never modify the KB index from this skill — read-only
- If the index is stale (kb files modified after last reindex), note: "Index may be stale — run `/kb-reindex` for fresh results"
- Score threshold 0.3 is the default low-confidence floor — do not suppress results, just warn
