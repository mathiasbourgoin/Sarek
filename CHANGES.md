## 2026-07

### Added

- Direct PTX emitter completion: records/variants with aligned C-ABI layout,
  match, helper-function inlining, static + dynamic `.shared`, `.local`
  arrays, full atomic set (CAS, wrapping inc/dec, 64-bit add/exch), f64
  softmath (trig, exp/log family, hypot, …), scalar cast matrix
- `Sarek_worklist` — portable dynamic-parallelism (work-queue) library
- `Sarek_gemm` — portable tiled SGEMM library for shared-memory backends
- `Sarek_df64` — float-float extended-precision arithmetic in pure Sarek
- Portable float64: `G`-suffix float64 literals, `sarek.real64` host
  plumbing, complete interpreter float64 math oracle
- Tier-0 defunctionalization pass (first-class kernel-local functions)
- Tuple-typed vectors on device and Native paths (shared shape registry)
- Static tag erasure for kernel-local variant slots
- Aligned C-ABI aggregate host layout (L8), with the `PtxLayout.v` formal
  model restated for the aligned ABI (0 admits)
- float32 `fma` intrinsic on all backends; GLSL `precise` qualifier on
  float locals
- CI installs `cuda-nvdisasm-12-6` and `ci/assert-toolchain.sh` asserts
  `nvdisasm` (version + a ptxas→nvdisasm probe). It was absent from the built
  image, so `test_cuda_f16_sass` — the gate the NVIDIA f16 guarantee rests on,
  and the only one that could surface a 12.6-vs-13.3 ptxas divergence —
  self-skipped in CI while reporting green.
- `docs/fp-contraction-policy.md` — cross-backend floating-point contraction
  policy: what each backend may contract, what actually prevents it, and
  whether that mechanism is verified or merely believed. The interpreter is
  named as the cross-backend oracle. Linked from every site that previously
  carried an ad-hoc contraction comment. §7 ("what cannot be verified without
  NVIDIA hardware") is now partly closed: an f16 kernel HAS been executed on
  an NVIDIA GPU — GTX 1070 Max-Q, sm_61, CUDA 12.9, driver 580.119.02 —
  agreeing with the interpreter on all 63488 finite binary16 inputs, with a
  liveness control (63085 mismatches on deliberately mismatched kernels)
  proving the sweep can go red. That also settles the tie-rounding question
  the section named, exercises the *driver's* ptxas rather than the offline
  one, adds a second toolkit version (12.9, between CI's 12.6 and the
  document's 13.3), and independently confirms on hardware that the CUDA
  barrier removed in #110 was inert. sm_61 is below the sm_75…sm_121 range
  the host-side sweeps cover, so it is a new sample, not a repeat.
- FP-conformance guard on the CUDA/nvrtc path: `-use_fast_math`, `-ftz=true`,
  `--prec-div=false` and `--prec-sqrt=false` are now REJECTED at the point an
  option array reaches `nvrtcCompileProgram` (`--fmad=true` warns). The guard
  screens the whole ARRAY, because nvrtc accepts an option and its value as two
  elements: `["--ftz"; "true"]` compiled a subnormal-flushing kernel past a
  per-element check (confirmed against libnvrtc 13.3, `.ftz` in the emitted
  PTX). A value-taking name consumes the next element and is fail-closed. They flush
  binary32 subnormals or downgrade div/sqrt, and no later flag undoes that.
  `sarek-cuda/test/test_cuda_fp_conformance.ml` reproduces the hazard
  host-side (`-ftz=true` turns `FMUL`/`FADD` into `FMUL.FTZ`/`FADD.FTZ` at
  sm_90, CUDA 13.3) and each of its four cases was proved red by mutation.
- `sarek/tests/e2e/test_vulkan_no_contraction.ml` (on the `e2e-gpu` alias, so
  it executes rather than merely building) — settles whether a Vulkan driver
  honours SPIR-V `NoContraction` (#126). Same shader compiled twice,
  differing only by `precise`, run on the same device/driver/process; the
  contracted target is taken from the device's own `fma()` rather than an IEEE
  model, because RADV's `fma` is not correctly rounded and a modelled target
  can make a genuinely contracted result read as clean. **Measured on RX 7900
  XTX (RADV NAVI31) and the Raphael iGPU, Mesa 26.1.4-arch3.1: 0 of 7
  contraction shapes contracted with or without `precise`, and the emitted RDNA
  ISA is opcode-identical between the two builds.** Both in-tree claims are
  refuted: RADV neither ignores the decoration nor needs it. Mesa ANV remains
  unmeasured — no Intel GPU on this machine.
- `sarek-hip/test/test_hip_f16_shapes.ml` + `scripts/f16_shape_isa_audit.sh` —
  exhaustive f16 expression-shape audit for AMDGPU fusion demotion (#106). All
  20 shapes the DSL can emit, each swept over all 63488 finite binary16 inputs,
  on gfx1100 and gfx1036: **0 disagreements as shipped**. Removing the opacity
  barrier breaks 9 of 20 and reproduces the original 620 exactly — the harness
  was calibrated against that known positive before its nulls were trusted, and
  it fails closed if the barrier-removed control stops going red. Disassembly
  additionally shows *four* demotion opcodes, not two (`v_mul_f16` and
  `v_sub_f16` are new), and three shapes that are demoted in machine code yet
  numerically clean — which a numeric-only audit would have mis-reported as
  unaffected. See `docs/optimization/amdgpu-f16-fusion-shape-audit.md`.
- ZLUDA/AMD support for the CUDA/PTX backend (PTX launch ABI fix)
- T3-SEMANTIC milestone lock for both formal projects; conformance +
  mutation tests wired into `dune runtest`

### Changed

- **BREAKING (`Sarek_ir_codegen`, shipped in `spoc.opam`'s `spoc.ir` library).**
  The record-declaration ordering surface added for backlog-203 is replaced by
  one that also orders variants (see the backlog-211 entry under Fixed).
  Removed: `sort_record_types_by_dependency`, `referenced_record_names`,
  `gen_record_typedefs`. Renamed: `Record_type_cycle` → `Type_decl_cycle`, which
  now reports a cycle with either kind on it. Added: `type_decl`, `tie_break`,
  `sort_type_decls_by_dependency`, `referenced_type_names`, `gen_type_decls`,
  `gen_c_type_decls`. No compatibility aliases, deliberately: an alias for
  `sort_record_types_by_dependency` would still compile inside a per-kind
  emission loop and silently reintroduce the cross-kind bug. The three C-family
  backends also lose their local `gen_variant_def` wrappers —
  `gen_c_type_decls` takes `~constructor_prefix` directly. Nothing outside
  `spoc/ir` and its tests referenced any of the removed names.
- Breaking: `Sarek_type_helpers.HELPERS` gained a required `val field_names :
  string list` — the immediate field names in `to_values` order. An in-place
  record-field store needs a field's POSITION in the `VRecord`'s value array,
  and nothing in the previous signature could supply one: `get_field` only
  reads, and `to_values`/`from_value` round-trip through the OCaml record and
  therefore copy. For a PPX-generated RECORD helper the correspondence holds by
  construction: `field_names` comes from the same label list that generates
  `to_values` and `get_field`, so the three cannot drift. Two carve-outs, because
  the justification for a breaking change should not be stated more widely than
  it holds: the PPX emits `field_names = []` for a VARIANT (no named fields, so
  every lookup returns `None` and a field store on one is refused by name rather
  than writing an arbitrary slot), and one in-tree implementor is *not* PPX-
  generated — the hand-written test mock in
  `sarek/sarek/test/test_sarek_type_helpers.ml`, which maintains the order
  manually and whose own test checks `field_index` against `to_values`' output
  rather than against hardcoded slots so that a drift there fails. A hand-written
  implementation outside this repository has to add the list, and carries the
  ordering obligation itself. There is no non-breaking spelling — a module type
  cannot carry a default, and the index is not derivable from the rest of the
  signature.

- The CUDA branch of `sarek_f32_barrier` no longer emits
  `asm volatile("" : "+f"(x))`. At an f16 narrowing it contributes zero PTX
  instructions, so `ptxas` receives an identical instruction stream and the
  cubins are byte-identical with and without it — re-measured on CUDA 13.3 for
  sm_75 through sm_121. What keeps the f32 multiply out of the narrowing on
  NVIDIA is `ptxas`, machine-checked by `test_cuda_f16_sass`. NOTE the same
  barrier is *not* inert at a `mul`→`add` site (PTX `mul.f32`+`add.f32` instead
  of `fma.rn.f32`), but `ptxas -O1`+ re-contracts that under the default
  `-fmad=true`, so it is still not a usable contraction barrier on NVIDIA — use
  `Sarek_df64`'s `mul_rn`. The AMDGPU `"+v"` barrier, which IS load-bearing, is
  unchanged. Removing a no-op that read as protection; no behaviour change.

### Fixed

- A shared-memory array of a record type behaved four different ways across
  nine devices, none of them right. `let%shared (s : tri) = 4l` followed by
  `s.(i).a <- e`, measured on this host: the Interpreter ×2 raised
  `assignment target of .a (got unit)`; Native accepted the store and wrote
  EVERY slot (`7 7 7 7`, want `7 0 0 0`); CUDA/PTX ×2 raised
  `unsupported construct: btype of custom type`, naming neither the array nor
  the type; and OpenCL ×2 and Vulkan ×2 failed inside the DEVICE compiler,
  OpenCL with `unknown type name 'Test_..._tri'` and Vulkan with a glslang
  syntax error at the shared declaration. Three separate causes. (1) Native's
  `alloc_shared_with_key` filled the array with `Array.make size default`, so
  every slot of a boxed element type was the same allocation — it now takes a
  per-slot thunk and calls `Array.init`, and the identical `Array.make` in the
  `create_array n Local` path is fixed with it. (2) The interpreter mapped a
  record element type to `VUnit`, so there was nothing to store into — it now
  builds a zeroed `VRecord` — or, for a variant, a `VVariant` carrying the
  first NULLARY constructor, the same one Native's `default_value_for_type`
  picks, falling back to the first constructor with zeroed payloads where
  there is no nullary one (a case in which Native has no default at all and
  raises, so the interpreter is more defined there rather than agreeing) — per slot in
  `Sarek_ir_interp_value.alloc_kernel_array`. The tag is
  `Hashtbl.hash ctor mod 256`, the encoding `EVariant` and the two matchers
  already use and which is now the single `variant_tag_of_ctor`; a positional
  index there produced a `VVariant` no arm could match, so reading a default
  variant slot raised `Pattern match failure in SMatch` on the Interpreter
  while Native answered the nullary constructor. That replaces three identical
  copies of the old init table in `Sarek_ir_interp_eval` plus a fourth, NARROWER
  one on `Sarek_ir_interp`'s `DShared` kernel-parameter path, which mapped only
  `TInt32` and `TFloat32` and everything else to `VUnit`. Unifying them
  therefore also widens that path: `TInt64`, `TFloat64`, `TBool`, `TFloat16` and
  `TUint8` go from `VUnit` to their typed zeros there. That is a fix, but it is
  a behaviour change outside backlog-206's shape and it is stated rather than
  folded into the word "copies". (3) OpenCL and Vulkan had no shared-memory gap at all:
  `register_types_from_typ` ran over PARAMETER types only, so a record named
  nowhere but the shared declaration was never emitted as a `struct`; it is now
  top-level and runs at both kernel-array declaration sites. The tell was that
  the same kernel written with a whole-slot store (`s.(tid) <- {...}`) compiled
  and ran correctly on all seven non-PTX devices throughout, because the record
  literal registered the type — so a whole-slot-store test would have been green
  the whole time, and `test_shared_record_slots.ml` exercises both shapes.
  CUDA/PTX still refuses, and the test asserts the refusal rather than
  tolerating it: PTX has no struct type and
  `Sarek_ir_ptx_mem.emit_agg_elem_addr` refuses state-space aggregates on
  purpose. What changed there is that the message names the array, the element
  type and the state space. Metal, HIP and WGSL were not measured — no such
  device on this machine — and nothing is claimed about them.

- A `[@@sarek.type]` record with a record-typed field had its inner struct
  emitted AFTER the struct that referenced it, so no kernel touching a nested
  record compiled on any shader backend. Both declaration-emission loops walked
  `kern_types` in list order, and that is not a dependency order: the PPX
  prepends the types reachable through the registry to the ones the kernel
  payload declares. Measured on an RX 7900 XTX host, a read-only nested access
  (`dst.(tid) <- src.(tid).mid.b`) failed on OpenCL ×2 with
  `unknown type name 'Test_..._triple'` and on Vulkan ×2 with a glslang parse
  error at the field line; a three-level chain and two independent nested types
  failed the same way. `Sarek_ir_codegen.sort_record_types_by_dependency` now
  orders the declarations, stably — ties break on the incoming list position,
  so an already-correct order is returned unchanged and no committed golden
  moved. A cycle between distinct record declarations raises `Record_type_cycle`
  instead of being emitted in some wrong order (a self-referencing field is
  dropped, not reported); that path is unreachable through the PPX, which
  refuses a self- or mutually-referencing record field while resolving its
  alignment (measured: both spellings stop at *unknown alignment for field
  type*), and is kept as a backstop for hand-built IR. Interpreter and Native
  were never affected — they carry values, not struct declarations. This orders
  records against records only; the two cross-kind halves it left live are the
  entry below, which also replaced the two identifiers named here — see the
  BREAKING note under Changed for what they became.
- A dependency edge CROSSING between a record declaration and a variant
  declaration was ordered by nothing, because each backend family sorted one
  kind inside its own emission loop and the families disagreed on which loop ran
  first. Both halves are reachable from ordinary `[@@sarek.type]` source and both
  were measured on an RX 7900 XTX / Ryzen 7950X host, two OpenCL devices and two
  Vulkan devices. A variant with a RECORD payload is red on both OpenCL devices
  (`error: unknown type name 'Test_record_variant_decl_order_probe_pt'` at the
  union member) and green on both Vulkan devices; a record with a VARIANT-typed
  field is the exact inverse — red on both Vulkan devices
  (`syntax error, unexpected IDENTIFIER` at the field line) and green on both
  OpenCL devices. Each family being green on the shape the other fails on is why
  a reproducer built on either half alone reports the ordering fixed. Reaching
  either half needs a RUNTIME-SELECTED constructor: static tag erasure (L14
  S1/S2) reduces a variant-typed local or record field written by a literal
  constructor application, and an erased variant never reaches `kern_variants`,
  so a literal-constructor reproducer is green on every device while covering
  nothing. All five generators that declare struct types (CUDA — also HIP —
  OpenCL, Metal, GLSL, WGSL) now emit records and variants from ONE interleaved
  topological pass over both lists together, `Sarek_ir_codegen.gen_type_decls`,
  instead of two per-kind loops. The tie-break is still the incoming index and
  each backend passes a `~tie_break` naming the order its own two loops used to
  run, so an edge-free kernel's emitted source is byte-identical and no committed
  golden moved. Node identity in the sort is the POSITION in the list, not the
  type name: a name-keyed self-edge drop discards a record's genuine edge to a
  SAME-NAMED variant, which `mangle_name` and fusion can both produce, and that
  is precisely the edge class this entry is about. The cyclic-value guard stays
  keyed on PHYSICAL node identity, so a cyclic `elttype` value still terminates
  whatever closes the cycle — including `let rec t = TVec t`, which has neither a
  record nor a variant on it. What this does NOT fix: a type referenced but never
  declared. The PPX registers a variant in `kern_variants` without registering a
  record that appears only in its payload, and no ordering pass can supply a
  declaration that is absent — it still surfaces as the backend's own *unknown
  type name*. Interpreter, Native and PTX were never affected: the first two
  carry values, and the PTX emitter declares no struct types.

- `v.(i).f <- e` — an in-place record-field store on a vector element — did
  something different on each CPU backend, while working on CUDA/PTX. The
  Interpreter **refused** it (`Unsupported_operation "record field assignment"
  / "not fully supported"`). Native had two failure modes split by mutability:
  on a **mutable** field it accepted the store and **silently dropped** it — the
  generated OCaml was a setfield on the fresh record `Vector.get` had just
  marshalled out of storage, so the write hit a temporary, nothing raised, and
  the vector kept its old values — while on an **immutable** field the same
  setfield did not compile (`The record field b is not mutable`), loud but
  misdiagnosed, which is what pushed users to add `mutable` and so into the
  silent half. Both halves are pinned.

  The nested form `v.(i).f.g <- e` needed a second fix on top of the depth-1
  one, on both CPU backends: with depth 1 fixed the Interpreter still read the
  intermediate record through the registry's copying `get_field` and Native still
  matched only the depth-1 shape, so the chained store was silently dropped by
  both. (Before either fix the Interpreter refused every field store, nested or
  not; the silent nested drop is the state the depth-1 fix alone would have
  shipped.) Asserted on every enumerated device, reading the values back and
  checking the neighbouring fields at both levels, because a dropped store is
  otherwise indistinguishable from a kernel that never ran.

  Two limits, stated rather than implied. On Native the store is now a
  whole-element read-modify-write, which makes two threads writing *different*
  fields of the *same* element a lost-update **race** there — an interleaving
  that reads both elements before either writes back loses one store, a
  serialised one loses neither — where the C-family, PTX and interpreter paths
  touch disjoint locations and cannot interfere. Read off the expansion, not
  observed, and covered by no test (backlog-207). And a record type declared
  *inside* the kernel payload is read through a generated getter that has no
  setter, so a store to one is now **refused with a message** instead of silently
  emitting a setfield.

- A record bound to a `let` inside a kernel was a live window onto vector
  storage in the Interpreter rather than a copy, so mutating the local mutated
  the vector element it came from. Detaching it copies through nested records and
  variant payloads, and refuses a chain deeper than 64 links. What that guard
  replaces is not a hang: the recursion is not tail-recursive, so an unguarded
  cyclic value dies with `Stack_overflow` — an untyped crash naming neither the
  value nor the binding — after tens of seconds of allocation and GC thrash
  (measured on one host; the seconds are a property of that `ulimit -s`, so the
  direction is the claim, not the figure). The depth-65 refusal itself is read
  off the marshaller and `bind_var`, not observed, and no committed test covers a
  65-link chain.

- `Sarek_df64` silently ran at plain float32 precision on real NVIDIA
  hardware (CUDA/PTX and NVIDIA OpenCL): `ptxas` contracted the multiply in
  `two_prod` into the `add`/`sub` of the `quick_two_sum` closing `df64_mul`,
  rebuilding the exact product and cancelling the TwoProd error term.
  `mul`/`div`/`sqrt` degraded from ~2^-47 to ~2^-24 with no error reported.
  `two_prod` now forms its product with `fma a b 0.0`, which cannot be fused
  again. Measured on a GTX 1070 (sm_61, CUDA 12.9): mul 5.92e-08 → 9.07e-15,
  div 5.64e-08 → 5.08e-15, throughput unchanged. A CPU-only regression guard
  (`test_df64_no_contraction`) asserts the emitted PTX contains no
  contractable `mul.f32`. The df64 per-backend precision table now names the
  device and toolchain behind every figure — the previous table generalised
  AMD-only measurements to "CUDA/PTX", which is why this went unseen.
- df64 precision gates could not distinguish a working df64 from a collapsed
  one. `test_df64`, `test_real64` and `test_real64_single_source` all widened
  their tolerance to `0x1p-22` (2.38e-07 — four times the float32 unit
  roundoff) on the backends with a documented deviation, so a df64 that had
  degraded all the way to plain float32 (measured 5.84e-08 on RADV) and one
  meeting its contract (9.07e-15) both read PASS. The widening also keyed on
  the bare `Vulkan` framework tag, sweeping in NVIDIA Vulkan, which meets the
  full contract. All three tests now hold every device to the derived contract
  bound (2^-47 add/sub, 2^-46 mul/div/sqrt) and express the documented
  deviations as an explicit expected-failure band — upper end 2^-23, twice the
  float32 unit roundoff — keyed on driver identity (Mesa RADV, Mesa ANV) in
  `Test_helpers.df64_known_deviation`. Degrading past plain float32 now FAILs
  even on an allowlisted device, and so does an allowlisted device that starts
  MEETING the contract (strict XPASS, as in pytest's `xfail(strict=True)`) —
  the run goes red naming the match arm to delete, so the allowlist cannot rot
  behind a green exit code. Red-proved twice: replacing `df64_mul` with a plain
  float32 multiply (old gate PASS on both RADV devices and Native at 1.43e-07
  vs tol 2.38e-07, new gate FAIL), and adding a bogus allowlist entry for an op
  that already meets the contract (all three tests exit 1 with the stale-entry
  message).
- PTX backend emits `sqrt.rn.f32` rather than `sqrt.approx.f32` for the float32
  `sqrt` intrinsic — the same correctness fix already applied to division
  (`div.approx.f32` → `div.rn.f32`, audit finding M2), which had missed `sqrt`.
  Dumping the generated PTX showed `sqrt.approx.f32` (~1 ulp) was the only
  non-correctly-rounded instruction in the whole `df64_sqrt` body, where it
  serves as the Newton seed. This is a global change: every f32 sqrt in every
  PTX kernel is now correctly rounded. Measured worst-case relative error over
  each test's own input set, on a GTX 1070 Max-Q (sm_61, CUDA 12.9, driver
  580.119.02): `df64_sqrt` 1.42e-14 (failing) → 8.53e-15 in `test_df64`, and
  1.68e-14 (failing) → 8.87e-15 in `test_real64`'s df64 fallback. Both post-fix
  figures coincide with the interpreter's for the same inputs — agreement
  between summary statistics, not element-wise identity — so the seed accounted
  for the whole gap these tests can see. Sampled maxima on one device and
  toolchain, not bounds. Costs ~12% kernel time on a sqrt-dominated benchmark
  (`bench_nbody` n=4096: 1.535 ms → 1.722 ms); `rsqrt` is unchanged for code
  that wants the fast form. The same bug class remains open in the OpenCL
  backend (no `-cl-fp32-correctly-rounded-divide-sqrt`, sqrt 1.81e-14) and the
  Vulkan backend (1.68e-14 on NVIDIA); see the `KNOWN RESIDUAL` block in
  `sarek/Sarek_df64/Sarek_df64.ml`.
- Indexed kernel-argument container with strict launch validation, honored
  across all six backends (out-of-order/sparse `set_arg` now correct)
- Unambiguous, collision-resistant compile-cache keys (kernel name included,
  digest-per-field); CUDA kernel-cache eviction on context destroy
- e2e test alias actually executes tests; vacuous tests now assert real
  results; negative suite runs in CI
- Vulkan push-constant scalar binding by logical index; 64-bit write
  alignment; buffer-index validation
- Benchmark dashboard HTML escaping at all sinks (stored-XSS hardening)
- CPU data kept authoritative after Interpreter runs

## 2026-06

### Added

- CUDA backend split into CUDA/PTX and CUDA/C devices (PTX is the default)
- `.shared` (DShared) emission in the PTX backend with formal spec
  (`specs/ptx-dshared-formal.md`)
- T3-SEMANTIC theorem work for the convergence-safety and type-safety
  formal projects
- WGSL/WebGPU codegen backend, in-browser playground and Learn course

### Changed

- Breaking: removed first-party `Obj` escape hatches from Sarek/SPOC execution
  paths.
  - Kernel vector arguments now cross framework/plugin boundaries through typed
    existential accessors and runtime type witnesses.
  - Custom value conversion now uses typed helper lookup instead of raw runtime
    representation casts.
  - Native and interpreter plugin buffer copies use typed Bigarray operations.
  - Custom shared-memory arrays are keyed by typed witnesses to reject
    mismatched reuse.
  - Legacy native direct execution now accepts typed `Framework_sig.exec_arg`
    arrays.

## 2026-01

### Changed

- Documentation cleanup and modernization
  - Comprehensive README rewrite with clear SPOC/Sarek distinction
  - Added sarek/ directory navigation guide
  - Created CONTRIBUTING.md with project guidelines
  - Removed pretentious language ("Grade A", "100%", etc.)
  - Added AI assistance acknowledgment
- Code quality improvements
  - Eliminated 49 failwith calls from Native and Interpreter plugins
  - Added structured error handling using Backend_error pattern
  - Created test suites for Native and Interpreter plugins
  - Added READMEs for Native, Interpreter, ppx_intrinsic, Sarek_float64
- CI/CD modernization
  - Added unit test execution (dune runtest)
  - Created fast benchmark suite (~20s) for CI
  - Integrated coverage measurement with bisect_ppx
  - Simplified workflow to single build job
- Repository cleanup
  - Removed AI artifact files (AGENTS.md, etc.) from history
  - Removed tracked build artifacts (*.exe, *.log, etc.)
  - Updated .gitignore for better hygiene

## 2025-12

### Added

- Full PPX rewrite replacing Camlp4
  - PPX-based kernel syntax ([%kernel ...], [%ktype ...])
  - Type registry for cross-module GPU types
  - Intrinsic registration system (Sarek_stdlib)
  - Float32 stdlib module with math intrinsics
  - Pragma support for loop unrolling hints
- Kernel fusion system (Sarek_fusion.ml)
  - Vertical fusion for producer-consumer patterns
  - Reduction fusion (map + reduce)
  - Stencil fusion with radius tracking
  - Auto-fusion heuristics
- Clean intermediate representation (Sarek_ir.ml)
- BSP superstep syntax with barrier synchronization
- Warp convergence tracking (Sarek_convergence.ml)
- Core primitives with convergence information
- Comprehensive test suite (unit, negative, e2e tests)

### Changed

- Removed Camlp4 dependency entirely
- Improved type inference for kernel parameters
- Better compile-time error messages

## 2024-2025

### Changed

- GPU Backend Overhaul (all 4 backends: CUDA, OpenCL, Vulkan, Metal)
  - Eliminated all failwith calls (94 total removed)
  - Implemented Backend_error.Make functor pattern for structured errors
  - Refactored code generators for maintainability
  - Added comprehensive unit test suites (19-20 tests per backend)
  - Created professional documentation (9-26KB per backend)
  - Improved error messages with context
- OCaml 5.x migration
  - Updated to OCaml 5.4.0
  - Migrated from Domains to Effect handlers where appropriate
  - Maintained backward compatibility

## 20210823

- Add PPX extension to declare external GPGPU kernels
- Update Samples to use PPX instead of Camlp4 extension
- Update dune/opam files for opam release

## 20210816

### Added

- Build with dune
- Compatible with OCaml 4.12
- Switch to github actions for CI
