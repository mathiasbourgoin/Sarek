# Intake Brief — pure-codegen-extraction (v2, corrected scope)

**Date:** 2026-06-02
**Status:** VALIDATED
**Type:** refactor

> v2 supersedes the v1 brief. Dual-voice planning (architect + skeptic) showed v1
> materially under-scoped the FFI coupling and the test situation; code-verified
> corrections are folded in below. Decisions made during planning are recorded under
> **Decisions**; genuinely-open high risks remain under **Open Questions**.

## Goal

Phase 0 of the website overhaul (`docs/plans/website-overhaul-2026-06-02.md`). Carve a
**pure, FFI-free** path `OCaml source → ppxlib AST → typed AST → IR → per-backend GPU
source` out of the FFI-linked packages, and expose
`Sarek_transpile.of_source : string -> (backend, string) result` so the Phase 2 in-browser
playground can transpile client-side (no GPU/runtime). GPU backend *runtimes*
(ctypes/CUDA/Vulkan/Metal) stay unchanged.

This is **multi-library architectural surgery**, not a file move — see Architecture.

## Scope Boundary

OUT of scope:
- js_of_ocaml target / playground UI (Phase 2). *But:* a bytecode (or jsoo) compile of the
  pure libraries is the only real proof the carve-out worked — see Open Q5.
- Backend runtime behaviour (memory, launch, device mgmt) — unchanged.
- The native-CPU generator (`Sarek_native_gen_*`) and `Kirc_Ast.Native` — legitimately
  FFI-coupled; stay in the FFI-linked library.
- **`[%native]` kernels in `of_source`.** `Sarek_ast.ENative` carries
  `gpu/ocaml : Ppxlib.expression` (unevaluated AST); `Sarek_quote` itself `failwith`s on
  quoting ENative. A runtime string transpiler cannot evaluate them. `of_source` returns a
  structured error for sources using `[%native]` — explicitly unsupported.
- Any change to generated GPU source bytes — must stay byte-identical (gated by a NEW
  golden harness, see below).

## Relevant Files

| File | Role | Verified fact |
|---|---|---|
| `spoc/ir/Sarek_ir_types.ml` (`sarek_ir`, pure) | IR types | FFI-clean leaf |
| `sarek/ppx/Sarek_parse.ml` | source AST → Sarek AST | `open Ppxlib`; consumes **ppxlib** `expression` (not compiler-libs Parsetree) |
| `sarek/ppx/Sarek_ast.ml` | Sarek AST | `ENative {gpu:Ppxlib.expression; ocaml:Ppxlib.expression}` — unevaluated nodes |
| `sarek/ppx/{Sarek_typer,Sarek_convergence,Sarek_mono,Sarek_tailrec,Sarek_lower,Sarek_lower_ir}.ml` | type→convergence→mono→tailrec→lower→IR | 0 *direct* FFI refs, BUT `Sarek_lower` produces `Kirc_Ast.k_ext` |
| `sarek/ppx/Kirc_Ast.ml` | legacy AST on the lower path | `Native of (Spoc_core.Device.t -> string)` → **transitive FFI dep** |
| `sarek/ppx/Sarek_ppx_registry.ml` + `Sarek_core_primitives.ml` | intrinsic registry consulted by the typer | `all_intrinsics()` reads a **mutable global** populated at module-init by FFI `sarek_stdlib`/`sarek_float64` |
| `sarek/ppx/dune` (`sarek_ppx_lib`) | one lib mixing pure frontend + FFI native-gen | `(libraries ppxlib spoc_core)` — must be **split** |
| `sarek-cuda/Sarek_ir_cuda.ml`, `sarek-opencl/…opencl.ml`, `sarek-metal/…metal.ml` | IR→source | `current_device:Device.t option ref` used only in `SNative` (reads `dev.framework`); **25/23/27 `*_error.*` calls** |
| `sarek-vulkan/Sarek_ir_glsl.ml` | IR→GLSL | no current_device; `Spoc_core.Log.debugf` ×2; 16 `Vulkan_error.*` calls |
| `sarek-*/{Cuda,Opencl,Metal,Vulkan}_error.ml` | error helpers | `include Spoc_framework.Backend_error.Make(...)` |
| `spoc/framework/{Backend_error,Device_type,Framework_sig,Typed_value}.ml` (`spoc_framework`) | error/device substrate | `spoc_framework` **links ctypes**; `Framework_sig`/`Typed_value` use it; `Backend_error`/`Device_type` purity is UNPROVEN (Open Q1) |
| `sarek-*/test/test_sarek_ir_*.ml` | the ONLY codegen tests | fragment asserts (`String.contains`), **no golden/byte-exact snapshots** |
| `sarek-{cuda,opencl,metal,vulkan}/dune` | backend pkgs (`wrapped` default true) | host the generators today; reference `Sarek_ir_<b>` **unqualified** as siblings |

## Architecture Notes

**Corrected FFI coupling (all on the transpile path):**
1. `Device.t` in cuda/opencl/metal — only `SNative` via `dev.framework` (a string).
2. `Spoc_core.Log` in glsl — 2 debug calls.
3. `*_error` modules → `Spoc_framework.Backend_error` → `spoc_framework` (**ctypes**).
4. `Kirc_Ast.Native (Spoc_core.Device.t -> string)` on the `Sarek_lower` path → spoc_core.
5. Intrinsic registry: typing reads a mutable global filled by FFI `sarek_stdlib` at init.

**Target library graph (to finalize in re-plan):**
- `sarek_ir` (pure, existing) + a pure `Backend_error` lib. **Verified:** `Backend_error.ml`
  is ctypes-free and self-contained; the `*_error` modules only `include
  Backend_error.Make(...)`. Extract `Backend_error` into a pure leaf lib; the ctypes users
  (`Framework_sig`/`Device_type`/`Typed_value`) stay in `spoc_framework`. (Resolves Q1.)
- `sarek_frontend` (pure: parse, typer, convergence, mono, tailrec, lower, lower_ir, env,
  registry, core_primitives; deps `ppxlib`+`sarek_ir`) — split out of `sarek_ppx_lib`.
- `sarek_codegen` (the 4 device-decoupled generators + error substrate; deps `sarek_ir`).
- `sarek_native_gen` (Kirc_Ast, Sarek_native_gen_*; keeps spoc_core).
- `sarek_ppx_lib` kept as a **façade** re-exporting the above so existing consumers/tests
  don't break; backend packages re-export `Sarek_ir_<b>` under their wrapped namespace.
- `Sarek_transpile` depends on `sarek_frontend` + `sarek_codegen`.

**Pipeline order (must match for byte-identical):** parse → typer → **convergence → mono →
tailrec** → lower_ir → `Sarek_ir_types.kernel` → per-backend `generate_with_types
~types:kern.kern_types`. (`Sarek_ppx.ml`'s `Unix.gettimeofday` is driver-only; omit.)

**`of_source`:** parse the string via ppxlib's parser (compiler-libs `Parse.expression`
then `Ppxlib_ast.Selected_ast.of_ocaml` migration to the pinned AST — Decision below),
require a bare `fun … -> …` (no `[%kernel]` wrapper), run the pipeline, emit
`(backend,string) result` with structured location/message errors. `backend` is a new
variant `Cuda|Opencl|Metal|Glsl` the transpiler defines.

## Decisions (resolved during planning)

| Point | Decision |
|---|---|
| Device decoupling | (a) replace `current_device:Device.t option ref` with `current_framework:string option ref`; `generate_for_device` becomes a 2-line backend-side wrapper. Minimal diff, lowest byte-identical-regression risk. |
| GLSL logging | inject `?log:(string->unit)` defaulting to no-op; backend wrapper passes the `Spoc_core.Log` impl. |
| `sarek_ppx_lib` | split is **mandatory** (frontend vs native-gen) with a re-export façade. |
| Parser | ppxlib parse + `Selected_ast.of_ocaml` migration (robust to ocaml↔ppxlib version skew); verify the exact bridge in implementation. |
| `[%native]` | excluded from `of_source` (structured error). |
| Golden harness | build FIRST, before any decoupling, as the regression gate (see gates). |

## Quality Gates

```bash
# 0. NEW prerequisite — golden-snapshot harness (must exist before refactor)
#    Capture current byte-exact generated source for representative kernels across
#    cuda/opencl/metal/glsl; reset mutable codegen refs (current_variants/current_device)
#    between captures to ensure determinism. This is the real byte-identical detector.
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek/tests/runtest
opam exec --switch=/home/mathias/dev/SPOC -- dune build
opam exec --switch=/home/mathias/dev/SPOC -- dune build @sarek-vulkan/all
opam exec --switch=/home/mathias/dev/SPOC -- dune fmt --auto-promote
# Acceptance proof: pure libs compile FFI-free (bytecode at minimum) — Open Q5
```

## Open Questions

- [x] **Q1 (RESOLVED) — `Backend_error` is ctypes-free.** Verified: 0 ctypes/Framework_sig/
  Typed_value refs; the `*_error` modules only `include Backend_error.Make(...)`. Extract
  `Backend_error` to a pure leaf lib; `Framework_sig`/`Device_type`/`Typed_value` (ctypes
  users) stay in `spoc_framework`. Carve-out feasible.
- [~] **Q2 (RESOLVED — scope expands).** Investigated: the registry is itself FFI-coupled.
  `intrinsic_info`/`type_info`/`const_info` carry `*_device : Spoc_core.Device.t -> string`
  device-codegen closures (`Sarek_ppx_registry.ml:24,35,46`), and every `let%sarek_intrinsic`
  in `sarek_stdlib` builds one. Two consequences for the pure path:
  (a) **Type decoupling:** extend the framework-name decision to the registry — change
  `*_device : Spoc_core.Device.t -> string` to `framework:string -> string` (or a pure
  device descriptor), touching the registry type AND every stdlib intrinsic `device`
  closure. Same root cause as the generators' `current_device`, but a wider surface.
  (b) **Population:** the registry is filled by FFI `sarek_stdlib` module-init (its runtime
  bodies use `Spoc_core.Vector`). The pure path needs a **pure intrinsic-metadata split**
  (name/type/device-codegen) separable from the FFI runtime bodies — or a build-time static
  metadata table. The plan must validate this on a minimal case (one `Float32` intrinsic)
  in its de-risking spike before the full rollout.
- [ ] **Q3 — regression fixtures.** The only codegen tests are fragment asserts. Do we add
  golden fixtures covering SNative and stdlib-heavy kernels BEFORE the device/Log refactor
  (Steps that change those paths)? (Strongly implied yes by both voices.)
- [ ] **Q5 — acceptance proof.** Does a bytecode/jsoo compile of `sarek_frontend` +
  `sarek_codegen` gate Phase 0 "done"? (Without it, "compiles to jsoo" is asserted, not
  verified.)
