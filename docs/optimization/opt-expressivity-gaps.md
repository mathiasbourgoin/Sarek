# Expressivity gaps: Sarek vs CUDA / OpenCL / Triton

Status: reference document (read-only on code; no code changed). Date:
2026-07-22.

## Purpose

The `ptx-limits-campaign` docs so far (L8/L9/L10) catalog what the PTX
backend *rejects today* for constructs Sarek's type system otherwise
allows. This document looks the other way: what does Sarek's surface
language and stdlib *not let the user say at all*, that CUDA, OpenCL C, or
Triton do — and where does that absence force a user off Sarek entirely, or
into a slower hand-rolled formulation. Target framing per the campaign
brief: "performance and expressivity better or comparable with
CUDA/OpenCL/Triton."

Method: read `sarek/ppx/Sarek_core_primitives.ml{,.mli}` (the semantic
primitive registry), `sarek/codegen/Sarek_ir_{cuda,opencl,metal,glsl,wgsl,
ptx_*}.ml` (the six backend emitters), `sarek/Sarek_stdlib/*`, and the
`sarek/tests/e2e/*` idiom surface (36 kernels: reduce, scan, sort,
histogram, matrix_mul, transpose, stencil, convolution, mandelbrot, nbody,
ray, module_poly, ...). Every claim below is grounded in a specific file;
this is a code inventory, not a survey of intentions in READMEs.

---

## (a) Already better than CUDA — claim this explicitly

These are places where Sarek's host language (OCaml) gives it structural
capability CUDA/OpenCL C cannot express at all, at zero extra engineering
cost. The doc framing should lead with these, not apologize for them.

1. **Algebraic data types on the GPU.** `[@@sarek.type]` supports records
   *and* variants (`sarek/ppx/Sarek_ppx.ml:29-207`), compiled through to
   every backend (`test_ktype_record.ml`, `test_registered_variant.ml`,
   `test_klet_variant.ml`, `test_complex_types.ml`, `test_nested_types.ml`).
   `match` on a GPU-resident variant with exhaustiveness checking is not
   expressible in CUDA or OpenCL C at all — C has tagged unions only by
   convention, with zero compiler-checked exhaustiveness. This is a
   category Sarek wins outright, not a "comparable" feature.
2. **Genuine parametric polymorphism, not C++ template instantiation.**
   `[@sarek.module] identity (x : 'a) : 'a = x` is usable at multiple
   concrete types in one kernel (`test_module_poly.ml:27-38`) via OCaml's
   own polymorphism — no template-instantiation-per-call-site explosion,
   no separate compilation unit per type, no SFINAE. CUDA's answer is C++
   templates (compile-time monomorphization with attendant binary bloat
   and error-message opacity); OpenCL C has no generics at all.
3. **`__restrict__` / `restrict` emitted automatically**, not something
   the user has to remember to annotate. Verified in codegen golden tests:
   every kernel pointer parameter gets `__restrict__` in CUDA
   (`test_codegen_golden.ml:456-457`) and `restrict` in OpenCL
   (`:533-535`) unconditionally. In hand-written CUDA this is a
   correctness-affecting opt-in the user must remember on every pointer
   parameter; Sarek gives it for free because its aliasing model already
   guarantees no user-visible pointer aliasing between kernel vector
   arguments.
4. **Six backends from one kernel source** (CUDA/PTX, OpenCL, Vulkan,
   Metal, WGSL, Native, Interpreter — `sarek/codegen/Sarek_ir_{cuda,opencl,
   metal,glsl,wgsl,ptx_*}.ml`, `sarek/plugins/{native,interpreter}`).
   Triton targets NVIDIA/AMD GPUs through its own compiler stack; CUDA is
   NVIDIA-only; OpenCL is portable but is what Sarek compiles *to*, not an
   alternative to it. No single-source path in the CUDA/Triton world
   reaches Vulkan/Metal/WGSL from the same kernel body.
5. **Compile-time convergence/variance analysis as a real static check**,
   not a runtime UB trap. `Sarek_core_primitives.ml`'s `variance` lattice
   (`Uniform ≤ BlockVarying ≤ WarpVarying ≤ ThreadVarying`) plus
   `Sarek_convergence.ml` reject a warp-collective call from
   divergent control flow *at compile time*
   (`sarek/tests/negative/test_warp_diverged.ml` — "Warp collective
   'warp_shuffle' called in diverged control flow"). CUDA's equivalent
   mistake (calling `__shfl_sync` under a divergent predicate with a
   stale mask) is a runtime hang or wrong-answer bug with no compiler
   diagnostic.

---

## (b) Gap table

| # | Gap | CUDA/OpenCL/Triton offer | What Sarek needs | User value | Cost | Priority |
|---|---|---|---|---|---|---|
| 1 | **Warp-level primitives** (shuffle/vote/ballot) | PTX `shfl.sync`, `vote.sync`, `__ballot_sync`; OpenCL/Vulkan/Metal subgroup ops (`sub_group_shuffle`, `subgroupBallot`) | See below — **this is a half-built gap, not a from-scratch one** | High — needed for any fast warp-level reduction/scan, avoiding shared-mem+barrier round trips | M | **P0** |
| 2 | **Half precision (f16/bf16)** + packed math | PTX `.f16`/`.f16x2`, `__half2`, `hfma2`; Triton `tl.float16`/`tl.bfloat16` native | New numeric type end-to-end: PPX type + typer rule, IR type, 6 codegen emitters, host-side packing in `Vector`/`Bigarray` (no f16 in OCaml `Bigarray` — needs a packed-int16 bit-reinterpret layer) | High for ML workloads; zero relevance for the FP32/FP64 numeric-kernel user base this project currently serves | L | P2 |
| 3 | **Grid-stride loop idiom** | A `for` loop with `blockDim.x*gridDim.x` stride; pure convention, no language support in CUDA either | Nothing — already expressible today with `global_thread_id`, `block_dim_x`, `grid_dim_x` (all in `Sarek_core_primitives.ml:75-179`) and an ordinary bounded loop | None — non-gap | — | N/A |
| 4 | **Multi-dimensional strided views (Triton tile model)** | Triton `tl.load`/`tl.store` over block pointers with masks, 2D/3D tiles, compiler-managed vectorization/coalescing; CUDA/OpenCL: manual `row*width+col` (same as Sarek today) | Two separable asks — see the dedicated section (c) below | View layer: Medium value, Medium cost. Tile-programming model: different compiler, out of scope | View: M / Tile: XL (out of scope) | View: **P1**, Tile: not recommended |
| 5 | **Block-level reduce/scan in stdlib** | CUB (`cub::BlockReduce`), Triton `tl.sum`/`tl.max`/associative_scan | A `[@sarek.module]` library building on warp prims (once #1 lands) + shared-memory tree pattern, generic over `+`/`min`/`max`/user monoid | High — every one of `test_reduce.ml`/`test_scan.ml`/`test_histogram.ml`/`test_sort.ml` currently hand-rolls this per-kernel with no shared abstraction | M (blocked on #1) | **P1** |
| 6 | **Tensor cores / `mma`/`wmma`** | PTX `mma.sync`, CUDA `wmma::` API, Triton `tl.dot` (lowers to tensor cores transparently) | New instruction class in IR + PTX emitter `mma.sync` support + matching ABI in the other 5 backends (Metal has its own `simdgroup_matrix`, Vulkan needs `VK_KHR_cooperative_matrix`, OpenCL has no portable equivalent) | High in principle (matmul/conv-bound ML), but every other backend either lacks a real equivalent or needs a distinct extension path — this fragments the "one kernel, six backends" value prop rather than extending it | XL | **doc-only** — not recommended as an implementation target now |
| 7 | **Dynamic parallelism / cooperative groups / cluster sync** | CUDA `cudaLaunchDevice`/cooperative-groups API, Hopper cluster/`cga` primitives | New launch model, host-device re-entrancy, backend-specific (OpenCL/Vulkan/Metal have no equivalent at all) | Low — niche even in native CUDA; no equivalent to target on 5 of 6 backends | XL | **verdict: skip** |
| 8 | **`printf`/`assert` in kernels** | PTX `vprintf` (device-side, host-buffered); OpenCL 1.2+ `printf` built-in; Metal/Vulkan: no portable device printf | IR node + PTX `vprintf` call convention (varargs marshalling through `.param` space — same class of problem L10 already flags indirect calls as blocked by) + OpenCL builtin passthrough; Metal/Vulkan/WGSL: no target, document as CUDA/OpenCL-only debug feature | Very high DX value — today a wrong-answer kernel bug means bisecting via host-side dumps only | M | **P0** |
| 9 | **Constant memory (`__constant__`)** | CUDA `__constant__` (broadcast-optimized read-only cache), OpenCL `__constant` address space | New address-space qualifier threaded through PPX param annotations, IR, and the 4 GPU backends' memory-space syntax (`Sarek_reserved.ml:36` already reserves `constant`/`restrict`/`__restrict__` as keywords — suggesting this was anticipated and stubbed, not built) | Medium — matters for broadcast-heavy kernels (small shared read-only tables); shared-memory arrays already give most of this benefit for block-scoped data | S–M | P2 |
| 10 | **Texture/surface memory** | CUDA texture objects (hardware interpolation/clamping, 2D cache locality), OpenCL images | Full new resource type: PPX annotation, IR node, 4 backends' texture-binding ABI, host-side image upload path | Low relative to cost — niche outside graphics/image-processing kernels, and none of the current e2e test surface (compute-oriented) exercises it | L | **verdict: skip** (revisit only if a graphics-kernel use case appears) |
| 11 | **Occupancy/launch-tuning surface** (`launch_bounds`) | CUDA `__launch_bounds__(maxThreads, minBlocks)` — hints the compiler for register allocation | A PPX attribute forwarded to the PTX emitter's register-budget decision (today implicit, not user-tunable) | Medium — matters once a kernel is register-bound; today the user has no lever at all, whereas CUDA gives one | S | P2 |
| 12 | **Kernel templates/metaprogramming** | already covered in (a).3 — Sarek's OCaml-level polymorphism is a genuine structural win here, not a gap | — | — | — | (a) |
| 13 | **Stdlib breadth** (sort networks, prefix-sum, histogram as reusable modules) | CUB, Thrust, Triton's growing `tl.*` library | `[@sarek.module]`-based libraries; mechanically straightforward once #1/#5 exist, since the *kernels* for these already exist as e2e tests and just need extracting into a reusable, generic (not per-test-hardcoded) form | High — closes the gap between "the pattern is demonstrated" and "the pattern is a one-line library call" | S–M per primitive (bitonic sort M, prefix-sum S once #5 lands, histogram S) | P1 |

---

## (c) The Triton question, treated seriously

Triton's actual value proposition is **not** "arrays with strides" — that
part is a thin convenience layer. Its real claim is: the user writes a
*tile-level* program (`BLOCK_SIZE`-wide vectors, `tl.load`/`tl.store`
with masks) and the compiler owns memory coalescing, vectorized loads,
software pipelining, and register allocation across the tile — the user
never writes a per-thread index computation at all.

These are two different asks and they should not be conflated in a
roadmap:

**Ask 1 — a strided/multi-dim view layer over `Vector.t`.** This is
squarely in scope and comparably cheap. Today every 2D kernel
(`test_matrix_mul.ml:88-91`, presumably `test_transpose.ml`,
`test_stencil.ml`, `test_convolution.ml`) hand-computes
`row = ty + block_dim_y*block_idx_y` and indexes a flat 1D
`Vector.t` with `row*n+col` by convention — the same idiom a raw CUDA
kernel uses, so Sarek is at parity with CUDA/OpenCL here, not behind them.
A thin `View2D`/`View3D` wrapper (record of `{ base : vector; rows;
cols; row_stride }` plus `[@sarek.module]` accessor functions
`get view i j` / `set view i j v` that the PPX inlines to the same
index arithmetic) would remove the manual-stride bugs class without
touching codegen at all — pure library-level sugar over the existing 1D
`vector` type (`sarek/core/Vector.ml`). Cost: M, mechanical, no new IR.

**Ask 2 — the tile/block programming model itself (compiler infers
vectorization/coalescing from a tile abstraction).** This is a
different compiler, not a library on top of Sarek's current IR. Sarek's
kernel body compiles per-thread scalar code (one thread = one lane of
control flow, matching CUDA/OpenCL's SPMD model exactly); Triton's `tl.*`
API compiles per-*program-instance* tile code where a single load/store
instruction denotes a whole block's memory traffic, and the pass that
turns that into per-lane vector loads with masking, then chooses
tile-to-warp mapping, is Triton's actual compiler middle-end (its MLIR
dialects: TritonGPU, plus the layout/coalescing/pipelining passes).
Retrofitting that onto Sarek's current single-pass-per-backend emitters
(`Sarek_ir_{cuda,opencl,...}.ml`, each ~a direct AST-to-text walk with no
layout-optimization pass in between) is not a feature addition, it is a
second compiler with its own IR layer, cost-modeled scheduling, and a
much larger backend-emitter rewrite across all 6 targets. Honest verdict:
**out of scope for this project as currently architected.** If tile-level
programming becomes a hard requirement, the right framing is "Sarek gets
a seventh, tile-oriented front end that lowers into a new mid-level IR,"
not "extend the existing emitters." That is a multi-quarter, dedicated-team
effort, not a backlog item alongside L8/L9/L10.

---

## (d) Recommended shortlist (5 items)

Ranked by (user value × how much of the remaining engineering is
"already half-done" evidence found in the tree):

1. **Finish warp-level primitives (#1).** The semantic layer already
   exists and is tested (`Sarek_core_primitives.ml:380-457`, convergence
   checks in `test_warp_diverged.ml`, `test_core_primitives.ml`) — but
   *zero* backend emits them (`grep` for `warp_shuffle`/`shfl`/`ballot`
   across all 6 codegen files returns nothing outside PTX-adjacent
   analysis code). This is the single highest-leverage item: the hard
   design work (variance/convergence semantics) is done, only the
   mechanical emitter work (PTX `shfl.sync` variants, OpenCL/Vulkan/Metal
   `sub_group_*`/`simd_*` intrinsics) remains, and it directly unblocks
   #5 and #13.
2. **`printf`/`assert` in kernels (#8).** Highest DX-value-per-cost item
   on the list; PTX and OpenCL both have a real target, and the current
   debug workflow (host-side dump-and-diff) is a genuine friction point
   for every kernel author today.
3. **Multi-dim strided view layer (#4, ask 1 only — not the tile model).**
   Removes a real, recurring bug class (manual stride arithmetic) at
   library cost, with no IR/codegen changes required.
4. **Block-level reduce/scan stdlib (#5) + broader stdlib breadth (#13).**
   Bundled because #5 is a prerequisite pattern for #13's prefix-sum/
   histogram entries, and because the *kernels* already exist as e2e
   tests today — this is "extract and genericize," not "invent."
5. **Occupancy hint surface, `launch_bounds` (#11).** Cheapest item with
   a real user complaint behind it (no register-budget lever today),
   good low-risk filler once 1-4 are underway.

Explicitly **not** recommended for the near-term roadmap: tensor cores/
`mma` (#6, doc-only), dynamic parallelism/cooperative groups (#7, skip),
texture/surface memory (#10, skip), and the Triton tile-programming model
proper (different compiler, out of scope as argued in (c)). Half
precision (#2) and constant memory (#9) are real but lower priority than
the shortlist given the project's current FP32/FP64 numeric-kernel focus.
