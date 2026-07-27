# Backend / Plugin Factorization Review (backlog-58)

**Status:** DESIGN REVIEW — analysis only, no code change proposed for merge here.
**Date:** 2026-07-25
**Scope:** structural duplication across Sarek's codegen backends
(`sarek/codegen/Sarek_ir_{cuda,opencl,glsl,wgsl,metal}.ml`, plus PTX split) and the
`sarek-{cuda,hip,opencl,vulkan,metal}` plugin runtime layer.
**Method disclosure:** `arch-index`/`arch-query` was attempted (`docs/architecture.db`)
but the database has no populated `functions`/`calls` tables (queries fail with
`no such table: functions`) — call-graph data is unavailable. **All findings below are
from `grep` + direct file reads.** File:line citations are against this worktree
(branch `design/backend-factorization-review`, off `main`).

---

## 1. What is already mutualized (precedent — the house style)

These are the patterns the codebase already prefers. Any new factorization should match
one of them: a **shared helper module with per-backend data/closures passed in**, a
**table-driven dispatcher**, or a **generic fold/traversal with per-node hooks**.

| Precedent | Home | PR | Shape |
|---|---|---|---|
| Kernel-compilation cache | `spoc/framework/Guarded_cache.ml` | PR #275 / backlog-66 | shared module |
| `gen_intrinsic` (5 backends → 1) | `sarek/codegen/Sarek_ir_intrinsic_dispatch.ml` | PR #276 / backlog-49 | table-driven dispatcher |
| 7 `*_uses_*` detectors → 1 | `spoc/ir/Sarek_ir_analysis.ml` (`expr_fold`/`stmt_fold` + `'a folder`) | PR #277 / backlog-47 | generic fold, per-node record |
| Variant-type emission | `spoc/ir/Sarek_ir_codegen.ml` (`mangle_name`, `gen_variant_def ~type_of_elttype`, `gen_variant_def_glsl`) | — | shared module, callback for the one backend-specific datum |
| Backend error funnel | `Sarek_backend_error.Backend_error.Make(struct let name end)` | — | functor; each `*_error.ml` is ~20 lines |
| Backend codegen re-export | `sarek-*/Sarek_ir_*.ml` are 9-line `include Sarek_codegen.*` shims | — | codegen already lives once in `sarek_codegen` |

Two consequences worth stating up front:

- **The plugin codegen files are not duplication.** `sarek-cuda/Sarek_ir_cuda.ml` et al. are
  re-export shims; the generators live once in the `sarek_codegen` library. No action.
- **`spoc/ir/Sarek_ir_codegen.ml` is the natural home** for further codegen factoring: it
  already establishes the exact `~type_of_elttype` callback idiom and is `include`d by all
  five backends. Opportunities 3–4 below extend it rather than inventing a new module.

**Regression safety net:** golden-output tests exist under `sarek/tests/codegen_golden/`
(`test_codegen_golden.ml`, `test_glsl_name_collision.ml`, `test_shader_ematch_payload.ml`,
`test_glsl_intrinsic_fallback.ml`) plus per-backend `test_sarek_ir_{cuda,opencl,metal,glsl}.ml`.
This materially raises the safety of every codegen refactor below: a byte-diff in emitted
source is caught. (Note a gap: there is no dedicated `test_sarek_ir_wgsl.ml` unit file — WGSL
is covered only via golden + `sarek/transpile/web/test/webgpu_wgsl_test.mjs`.)

---

## 2. Ranked opportunities

Ranking is **value × safety**, both on a High/Med/Low scale. Value = duplication removed +
bug-class closed. Safety = regression surface, readability/flexibility preserved, test cover.
LOC estimates are approximate and count *removed* duplication, not net (a shared helper adds
some lines back).

### Summary table

| # | Opportunity | Dup. today | LOC saved | Value | Safety | Rank | Task tie |
|---|---|---|---|---|---|---|---|
| 1 | Shadow-rename twins (GLSL/WGSL) → shared IR pre-pass | ~110 of ~118 lines identical | ~100 | High | High | **1** | backlog-71 / backlog-72 (PR #282, PR #283) |
| 2 | EMatch/SMatch payload handling → shared helper + fix divergence | guard dup ×2, SMatch handler dup ×5, EMatch discard dup ×5 | ~80–120 | High | High | **2** | backlog-73 / backlog-75 (PR #284) |
| 3 | C-family `gen_param`/`is_vec_type`/record typedef → `Sarek_ir_codegen` | verbatim ×3 | ~60–70 | Med | High | **3** | — |
| 4 | `gen_lvalue` shared traversal (5 backends) | 4 arms ×5, identical | ~40 | Med | High | **4** | — |
| 5 | `gen_expr` mechanical arms + `gen_stmt` control-flow (C-family trio) | ~90% of trio traversal | large (~hundreds) | High | Med | **5** | — |
| R1 | Full uniform `gen_expr`/`gen_stmt` incl. WGSL | — | — | — | — | **AGAINST** | — |
| R2 | Header/prelude unification | ~0 | ~0 | Low | — | **AGAINST** | — |
| R3 | Reserved-name escaping mechanism | 1-liner ×2 | ~3–5 | Low | High | **AGAINST (defer)** | — |
| R4 | WGSL variant-type emission → shared | — | — | Low | Low | **AGAINST** | — |
| R5 | nvrtc/hiprtc RTC unification | looks twinned, is ~90% divergent | — | Low | Low | **AGAINST** | — |

---

### Opportunity 1 — Shadow-rename twins → one parameterized IR pre-pass  **[RANK 1]**

**The duplication.**
- GLSL `rename_pc_shadowing_locals` — `sarek/codegen/Sarek_ir_glsl.ml:1518-1637` (120 lines),
  counter `pc_shadow_counter` at `:1489-1492`.
- WGSL `rename_scalar_shadowing_locals` — `sarek/codegen/Sarek_ir_wgsl.ml:1035-1150` (116 lines),
  counter `scalar_shadow_counter` at `:1001-1004`.

Both are `stmt → stmt` alpha-renames threading an immutable `string SM.t` env, renaming any
body binder that collides with a "protected" name and rewriting references. Binder coverage is
identical (`SLet`, `SLetMut`, `SFor` var, `SMatch`/`EMatch` `PConstr` binders; full structural
recursion elsewhere). **Verified byte-for-byte identical:** GLSL `:1557-1637` == WGSL `:1070-1150`
(the entire `re_expr`/`re_lvalue`/`bind`/`re_stmt` region, 81 lines). `ren` and `bind_pattern`
are also structurally identical.

The **only** genuine differences:

| Aspect | GLSL | WGSL |
|---|---|---|
| collision predicate | `escape` then `mem n pc_names \|\| mem n len_names` (2 protected sets) | `mem name scalar_names` (1 set, raw name) |
| fresh-name prefix | `sarek_pc_shadow_` | `sarek_scalar_shadow_` |
| escape fn | `escape_glsl_name` (`:221`) | `escape_wgsl_name` (`:127`) |
| counter | `pc_shadow_counter` | `scalar_shadow_counter` |

CUDA/OpenCL/Metal/PTX have **no** such pass and correctly need none (they emit params as
ordinary C-style identifiers; the hazard is specific to GLSL's `#define name pc.name` macros and
WGSL's `params.<name>` global rewrite). So this is a genuine **2-backend** twin, not a 5-way one.

**Proposed abstraction (matches house style — shared module + injected closures).**
Add to `spoc/ir/Sarek_ir_codegen.ml`:

```
val rename_shadowing_locals :
  collides:(string -> bool) ->
  fresh_name:(string -> string) ->
  Sarek_ir_types.stmt -> Sarek_ir_types.stmt
```

GLSL passes `~collides:(fun n -> let n = escape_glsl_name n in List.mem n pc_names || List.mem n len_names)`
and `~fresh_name:(fun o -> Printf.sprintf "sarek_pc_shadow_%s_%d" (escape_glsl_name o) (post_incr pc_shadow_counter))`;
WGSL passes its 1-set predicate and its prefix. ~93% of the two functions (≈100 LOC) collapses to one.

**Risk.** Low. Pure AST→AST, no FFI, no perf path. The two protected-set shapes and prefix are
fully captured by the two closures. `test_glsl_name_collision.ml` + golden tests pin the output.
The one judgment call: whether to keep the counters per-backend (they must, for name stability of
each backend's golden files) — the `~fresh_name` closure owns its own counter, so this is fine.
**Recommend as the first factorization.** Highest value×safety: real duplication, tiny blast
radius, direct test coverage, and it consolidates two fixes (backlog-71 / backlog-72, PR #282 and PR #283) that were
landed in parallel and produced the twin.

---

### Opportunity 2 — EMatch/SMatch payload handling → shared helper + close the divergence  **[RANK 2]** (ties backlog-75)

This is the messiest area and the most valuable to get right, because it is **both** duplication
**and** a live correctness divergence. Three distinct sub-problems are tangled together — keep them
separate.

**IR fact:** `EMatch of expr * (pattern * expr) list`, `pattern = PConstr of string * string list | PWild`
(`spoc/ir/Sarek_ir_types.ml:131-132`, `:106-108`). The payload binder names are the `string list`
in `PConstr`. `SMatch` is the statement-level twin (`:148`).

**2a — Expression-position EMatch: the buggy lowering is copy-pasted ×5.**
Each shader/native backend lowers a value-position match to a nested ternary/`select`, and each
writes `PConstr (name, _)` — **dropping the binder list**:
- CUDA `Sarek_ir_cuda.ml:190-219` (discard at `:208`), OpenCL `:208-237` (`:226`),
  Metal `:212-239` (`:228`) — **no guard, silently wrong** if a body uses a binder.
- GLSL `:482-516` and WGSL `:348-384` added a fail-loud guard (`case_binds_used_payload`,
  GLSL `:299-328`, WGSL `:196-223`) that raises `unsupported_construct` instead — the backlog-73 fix
  (PR #284). The `.tag == name` emitter itself (GLSL `:504`, WGSL `:378`) is still the same snippet.

So today CUDA/OpenCL/Metal **silently emit undefined-identifier code** while GLSL/WGSL fail loud —
an inconsistency, not just duplication. PTX (`Sarek_ir_ptx_expr.ml:809-823` via `bind_pattern_vars`
`:86-130`) and the interpreter (`sarek/interp/Sarek_ir_interp_eval.ml:177-209`) are **correct** —
they bind the payload.

**2b — Statement-position SMatch: the payload-binding handler is duplicated ×5 but correct.**
All five backends emit a `switch(scrut.tag)` that *does* bind payloads via
`data.<Ctor>_v[._N]` and a `find_constr_types` lookup over `current_variants`
(CUDA `Sarek_ir_cuda.ml:505-549`; the same shape recurs in opencl/glsl/wgsl/metal — grep
`find_constr_types`/`current_variants` hits 7–8 per file). This is real structural duplication.

**Proposed abstraction.**
1. Move `case_binds_used_payload` + `expr_mentions` into `Sarek_ir_codegen.ml` **once**, and call
   it from all five value-position arms. **Extend the guard to CUDA/OpenCL/Metal** so they fail loud
   too — this closes the silent-wrong divergence (2a). Small, high-value correctness win.
2. Factor the SMatch payload-extraction skeleton (2b) into a shared
   `gen_match_payload_bindings ~type_of_elttype ~field_access buf ...` — the lookup and the
   iteri-over-payload loop are identical; only the field-access spelling (`.data.Ctor_v._N` vs
   GLSL/WGSL member syntax) and the decl form vary, both injectable.

**The honest limit (judgment call, ties backlog-75).** A shared *ternary/select* helper unifies the
fail-loud behavior and de-dups the emitter, but it does **not** make payload-using
match-*expressions* compile on the ternary backends — a nested ternary is a single expression with
nowhere to introduce a declaration, which is exactly why GLSL/WGSL chose to fail loud. Genuinely
supporting payloads in expression position requires **promoting EMatch to SMatch-style lowering**
(hoist to statement position with real binder decls), which PTX and the interpreter already do.
That is a *feature*, not a factorization, and should be tracked as such under backlog-75. **Do not
conflate "fix backlog-75 in one place" with "make it compile":** the factorization delivers *uniform
fail-loud + one guard site + one SMatch handler*; the feature is a separate follow-up.

**Risk.** Low-Med. Extending the guard to 3 more backends changes their behavior from
silent-wrong to a clean error — strictly an improvement, but it is a behavior change, so it needs a
CHANGELOG note and a negative test. `test_shader_ematch_payload.ml` already exists to pin it.

---

### Opportunity 3 — C-family `gen_param` / `is_vec_type` / record typedef → `Sarek_ir_codegen`  **[RANK 3]**

**The duplication (verbatim, C-family trio).**
- `gen_param`: CUDA `Sarek_ir_cuda.ml:572-591`, OpenCL `:575-598`, Metal `:693-716`. Identical
  skeleton: emit `<type> <name>`, and if `is_vec_type`, append `, int sarek_<name>_length`. The
  only variable is the type-map fn (`cuda_param_type`/`opencl_param_type`/`metal_param_type`).
- `is_vec_type` is copy-pasted verbatim (CUDA `:570`, Metal `:642`, + OpenCL).
- Record typedef: inlined identically in each `generate_with_types` (CUDA `:722-734`,
  OpenCL `:712-724`, Metal `:984-996`): `typedef struct {` + per-field `<type> <name>;` + `} <mangled>;`.

**Proposed abstraction.** Extend `Sarek_ir_codegen.ml` (the module that already did variants):
`is_vec_type` (0-arg move), `gen_param ~param_type buf`, `gen_record_def ~type_of_elttype buf`.
Metal's `gen_param_metal` buffer-attribute variant (`:646-692`, `[[buffer(n)]]` threading) is
**genuinely divergent — leave it**. ~60–70 LOC removed with no behavioral change.

**Risk.** Low. Same callback idiom already proven for variants; golden tests pin output.
The GLSL/WGSL `gen_record_def` (`glsl:1743`, `wgsl:817`) use different member syntax
(`type name;` vs `name : type,`) — a shared skeleton with a field-emitter callback is *possible*
but the payoff is marginal and it slightly obscures each backend's syntax; **do C-family only.**

---

### Opportunity 4 — `gen_lvalue` shared traversal  **[RANK 4]**

`gen_lvalue` has 4 arms (`LVar`, `LArrayElem`, `LArrayElemExpr`, `LRecordField`) that are
**identical across all five** backends (CUDA `:327-343`, OpenCL `:347-365`, GLSL `:786+`,
WGSL `:533-551`, Metal `:409+`); the only variation is a name-escaping hook and a cosmetic
`")["` vs `')' '['` delimiter. A shared `gen_lvalue ~escape ~gen_expr buf` covers all five with
zero loss. **Cleanest full factor-out**, but small (~40 LOC) — hence rank 4, not higher.

**Risk.** Low, *except* the mutual recursion: `gen_lvalue` calls `gen_expr` which calls
`gen_lvalue`. A shared `gen_lvalue` must take `~gen_expr` as a parameter (or the two share a
`rec` group via a functor). This is a mild structural wrinkle — the value is small enough that it
is only worth doing *together with* opportunity 5 (same recursion group), not on its own.

---

### Opportunity 5 — Shared `gen_expr` mechanical arms + `gen_stmt` control-flow (C-family trio only)  **[RANK 5 — biggest prize, most caution]**

This is where the raw LOC is (the traversal skeleton across five ~1000-line files), and where
over-abstraction is the classic mistake. **Be skeptical.** The honest measurements:

- **CUDA vs OpenCL vs Metal traversal ≈ 90% identical.** `EBinop`/`EUnop`/`EArrayRead`/
  `EArrayReadExpr`/`ERecordField`/`ETuple`/`EApp`/`EVariant`/`EArrayLen` are near byte-identical
  (e.g. `EBinop` is the same 5-line block in CUDA `:105`, OpenCL `:121`, Metal `:125`, WGSL `:263`).
  Control-flow `SIf`/`SWhile`/`SReturn`/`SExpr`/`SBlock`/`SSeq`/`SEmpty`/`SAssign` likewise. The
  trio duplication is pure cost.
- **GLSL still ~70% aligned;** WGSL drops to ~55–60%.

**Where mutualization HURTS (leave per-backend even in a shared skeleton):**
- `EConst` — literal suffixes (`LL`/`L`/`i`/`f`), `true/false` vs `1/0`, and WGSL/GLSL *hard
  errors* on i64/f64 (WGSL `:237`,`:248`) / GLSL non-finite reconstruction (`:339`). Genuinely
  per-backend.
- `ECast` — prefix `(type)expr` (C-family) vs constructor `type(expr)` (GLSL/WGSL): an emit-*order*
  difference, not a string swap.
- `ERecord` — three distinct shapes (positional literal / `.field =` designated / `Name(...)`).
- `EIf` / `EMatch` multi-case — ternary (C-family/GLSL) vs WGSL `select()` with a *restructured,
  separately-duplicated* pre-buffering recursion (`wgsl:362-405`).
- Declarations — `SLet`/`SLetMut`/`SFor` are C `type name = e;` vs WGSL `let/var name : T = e;`
  (`gen_var_decl`, `wgsl:593`): different token order + keyword, plus WGSL/GLSL Shared-array
  hoisting and error paths. **Load-bearing.**
- `SNative` — real injection (C-family) vs `/* not supported */` stub (GLSL/WGSL).

**Recommendation.** A shared traversal-with-hooks is worthwhile **for the C-family trio
(CUDA/OpenCL/Metal)** and reasonable to extend to GLSL. Provide named hooks: `~escape`,
`~gen_binop`, `~gen_unop`, `~gen_var_decl`, `~barrier_keywords`, `~type_of_elttype`, and a
per-arm emitter for each of `EConst`/`ECast`/`ERecord`/`EIf`/`EMatch`. Share `gen_lvalue` (op. 4)
and the ~9 mechanical `gen_expr` arms + the control-flow half of `gen_stmt`; **leave the ~5
structural arms and the declaration arms per-backend.**

**Risk.** Med. Largest regression surface (touches every emit path), and the readability tradeoff
is real: a reader chasing "how does Metal emit an if" now hops module → hook. Golden tests bound
the *correctness* risk well, but the *maintainability* verdict is a genuine judgment call. **Do
this last, incrementally (start with the trio's mechanical arms behind hooks, measure the
readability cost before extending), and stop the moment a hook body becomes as large as the arm it
replaced** — at that point the abstraction is relocating code, not removing it.

---

## 3. Explicitly recommend AGAINST

- **R1 — Forcing one uniform `gen_expr`/`gen_stmt` across all five including WGSL.** Net-negative.
  WGSL's `select()` match/`EIf` lowering, typed-literal hard-errors, `params.<name>` scalar rewrite,
  and `let/var name : T` declaration syntax are load-bearing divergences. A hook-per-arm scheme
  would merely *relocate* them, producing an abstraction as big as what it replaces while hiding
  each language's shape behind indirection. The five languages are not one; do not pretend.
- **R2 — Header/extension/prelude unification.** ~0 LOC. Each header is dictated by its language
  (`extern "C"` / OpenCL `#pragma` / Metal `#include <metal_stdlib>` / GLSL `#version`+`#extension`
  / WGSL `@compute` entry signature). The only shared bit is a `// Sarek-generated` comment. Not
  worth an abstraction.
- **R3 — Reserved-name escaping *mechanism*.** The 1-liner `if mem name reserved then name^"v"`
  is shared only between GLSL (`:221`) and WGSL (`:127`); the lists correctly stay per-backend.
  Factoring saves ~3–5 LOC and the explicit form is clearer. **Defer** until a third escaping
  backend appears.
- **R4 — WGSL variant emission → shared.** WGSL re-implements `gen_variant_def` inline
  (`:836-895`) because its syntax (`const … : i32`, `fn make_…`, trailing-comma fields, no union)
  materially diverges from the C-family union form already shared in `Sarek_ir_codegen.ml`.
  Unifying would add branches and reduce clarity.
- **R5 — nvrtc/hiprtc RTC unification.** `sarek-cuda/Cuda_nvrtc.ml` (396 LOC) and
  `sarek-hip/Hip_rtc.ml` (301 LOC) *look* like twins (HIP is the AMD analog of NVRTC) but a diff
  shows ~90% divergence: different result-code enums, different library symbols, and a different
  pipeline (hiprtc returns a finalized code object; nvrtc returns editable PTX needing a separate
  ptxas/JIT stage). The FFI reality is genuinely different. Leave separate.
  *(Note the plugin `*_error.ml` files are already shared via `Backend_error.Make` — no action.)*

---

## 4. Recommended execution order

Ordered by value × safety, smallest-blast-radius-first so each lands independently behind the
golden tests:

1. **Opportunity 1 — shadow-rename twins.** Isolated, ~100 LOC, direct test cover, consolidates
   backlog-71 / backlog-72. Do first.
2. **Opportunity 2a — EMatch guard: share + extend to CUDA/OpenCL/Metal.** Closes the silent-wrong
   divergence (a real correctness bug), then 2b (share the SMatch handler). Ties backlog-75; explicitly
   file the "EMatch→SMatch promotion for payload support" as a *separate feature* under backlog-75, not
   part of this factorization.
3. **Opportunity 3 — C-family `gen_param`/`is_vec_type`/record typedef into `Sarek_ir_codegen`.**
   Proven callback idiom, verbatim duplication, low risk.
4. **Opportunity 4 + 5 together — `gen_lvalue` (full) + C-family trio mechanical `gen_expr` arms +
   control-flow `gen_stmt`, behind named hooks.** Largest prize, largest caution. Do incrementally,
   trio-only first, measure readability before extending to GLSL, and stop where a hook body grows
   to the size of the arm it replaces.

Steps 1–3 are unambiguous wins. Step 4 is the one requiring a human call on the
readability/flexibility tradeoff — it is deliberately last.

---

## 5. Open judgment calls for the reviewer

- **backlog-75 semantics.** Does backlog-75 want *uniform fail-loud* on payload-using match-expressions (op. 2a
  suffices) or *actual payload support* (needs the separate EMatch→SMatch promotion feature)? This
  changes whether op. 2 "closes" backlog-75 or only de-risks it.
- **How far to push op. 5.** C-family trio only, or through GLSL? My recommendation: trio first,
  GLSL only if the trio abstraction stays readable; never WGSL.
- **Counter/name-stability.** The shadow-rename fresh-name counters and prefixes are per-backend by
  design (golden-file stability). The op. 1 `~fresh_name` closure preserves this; confirm no golden
  file is expected to change.
</content>
</invoke>
