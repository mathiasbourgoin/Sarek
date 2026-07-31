# No eighth backend

**Decision:** decline to add an eighth backend. Seven already exceeds what the
project's story needs, and every additional one is a full column in every
cross-cutting change from here on.

**Date:** 2026-07-31.

**Method:** the same as `docs/design/width-addition-cost.md` (backend-column
cost, not width-addition cost) — counted by reading real diffs and code, not
estimated. Every count below states the command or file:line that produced it.

---

## 1. What "seven" counts

`README.md:70-81` lists the devices that appear in `Device.init` /
`sarek-device-info`: `CUDA/PTX`, `CUDA/C`, `HIP`, `OpenCL`, `Vulkan`, `Metal`,
`Native`, `Interpreter` — eight table rows. `README.md:89` notes the CUDA
row splits into two only because it registers two frameworks
(`Device.filter_cuda ()` matches both) from what the codebase and its docs
otherwise treat as one backend. Collapsed, that is **seven**: CUDA, HIP,
OpenCL, Vulkan, Metal, Native, Interpreter.

WGSL is listed separately, at `README.md:83-88`, under an explicit "Not a
device — a code-generation target only" heading: `sarek/plugins/webgpu` is a
stub whose `is_available ()` returns `false`, so WGSL never appears in device
enumeration. It has a real emitter (`sarek/codegen/Sarek_ir_wgsl.ml`) and a
real CI gate (`naga` validation), but deliberately no runtime — it is the
project's own precedent for what *not* promoting a target to full-backend
status looks like, and it is why this decision is "no eighth", not "no
ninth": WGSL was never the seventh, or the eighth.

## 2. Every cross-cutting change costs a full column

`docs/design/width-addition-cost.md` measured this for adding a scalar type,
not a backend, but the mechanism is the same one that makes a backend
expensive: a change that must reach every target touches each target's own
emitter, its own tests, and its own idioms, none of which the compiler can
find for you in one pass. Its headline number, reproduced here because it is
the strongest single data point in this repository for "cross-cutting is
not cheap": adding one scalar width (`TBFloat16`) touched **40** production
files; the OCaml compiler forced only **21** of them, across **seven**
separate build rounds, because each layer's errors are hidden behind the
previous layer's failure; the other **20** — including the FFI layer, the
host storage layer, and the exec-arg dispatch — the compiler never asks for
at all and must be found by reading.

A backend is not a scalar type, but the same shape of cost shows up whenever
a change must reach every emitter rather than one:

- **The dispatch-tag divergence** (§3 below) was found by reading five
  emitter files end to end, not by a type error — nothing forces
  `Dispatch.framework` to be consistent across emitters, so the four correct
  ones and the one constant one type-checked identically.
- **The SNative three-way split** (§3 below) is a match arm present, in some
  form, in all five codegen files with a `Dispatch.framework` field
  (`Sarek_ir_cuda.ml`, `Sarek_ir_opencl.ml`, `Sarek_ir_metal.ml`,
  `Sarek_ir_glsl.ml`, `Sarek_ir_wgsl.ml`) plus the separate PTX emitter
  (`Sarek_ir_ptx_stmt.ml`) — six sites, three different behaviors, and each
  one independently decided by whoever wrote that emitter's arm.
- **The per-generation dispatch state is not shared machinery — it is five
  independent copies.** Four of the five emitters each define their own
  module-level `current_framework : string option ref = ref None`:
  `Sarek_ir_cuda.ml:36`, `Sarek_ir_opencl.ml:47`, `Sarek_ir_metal.ml:36`,
  `Sarek_ir_wgsl.ml:41` — four separate `ref` cells, four separate
  definitions, not one shared module. The fifth, GLSL, has no such ref at
  all (§3 below). Converting that pattern to a properly threaded
  per-generation value — replacing a global mutable ref with a value passed
  through the call chain — is therefore not one change; it is five,
  because there is no single definition site whose fix reaches all five
  emitters. It is the same shape as §2's headline number: a change that
  must reach every target, done once per file because nothing forces it to
  be done once at all.

Seven backends is already seven columns for every change of this shape. An
eighth is not a 1/7 increase in cost — an eighth emitter would add a sixth
site to the dispatch-tag divergence, a seventh to the SNative split, and a
sixth independent ref cell (or a sixth silent exception to the pattern) to
the state-threading count.

## 3. The columns are not equivalent, so the cost is not even linear

If every backend answered every design question the same way, "add a
backend" would still cost a column, but a predictable one: copy the pattern
that worked for the other seven. It does not work that way. Four instances,
each found by reading the actual emitter rather than assumed:

- **WGSL refuses whole-value array/vector equality that four C-family
  backends accept.** An earlier change (referenced in commit `68b7a39d`'s own
  message) had refused aggregate `=`/`<>` (tuple, record, variant, function)
  at the typer, deliberately leaving vectors and arrays out of the refused
  set, because on CUDA, OpenCL, Metal and GLSL a
  kernel-vector or local-array parameter is a device pointer, so `src = dst`
  emits `src == dst` (pointer equality) and both `clang -x cl` and
  `glslangValidator` accept it, exit 0. WGSL does not generalize the same
  way: naga has no equality operator on `array<T>` / `array<T, N>` at all
  and rejects the identical shape with `"Incompatible operands: Equal(Array
  …)"`, reproduced through the full frontend pipeline and confirmed against
  naga 30.0.0. Fixed by refusing the shape at exactly one emitter,
  `sarek/codegen/Sarek_ir_wgsl.ml`'s `EBinop (Eq | Ne, ...)` arm — commit
  message states directly: "The four C-family emitters are untouched."
  (regression test: `sarek/tests/codegen_golden/test_wgsl_array_equality_refusal.ml`)

- **GLSL's dispatch tag is a compile-time constant where four other
  emitters carry a mutable, per-generation reference.** `sarek/codegen/Sarek_ir_glsl.ml:778`
  reads `Dispatch.framework = (fun () -> "GLSL")` — a literal string, no
  ref, no parameter. `sarek/codegen/Sarek_ir_cuda.ml:303-304`,
  `Sarek_ir_opencl.ml:349-350`, `Sarek_ir_metal.ml:407-408`, and
  `Sarek_ir_wgsl.ml:747-748` all instead read
  `Option.value ~default:"<Name>" !current_framework` — a ref that can be
  set per generation. This is not a bug (nothing currently retargets GLSL's
  string), but it means the four-emitter pattern one might copy for an
  eighth backend is, in fact, a four-out-of-five pattern with one silent
  exception that a naive port would either miss or copy wrong.

- **Metal narrowed `float64` to `float` silently, and no other backend's
  fix generalized to it.** Before the fix landed in commit `6bdc751e`
  (`#64` slice 1), `sarek/codegen/Sarek_ir_metal.ml`'s `TFloat64` arm
  emitted `"float"` with a comment noting Metal has no `double` — and no
  refusal anywhere on the path. Commit `2bc893d1` (`#141`) found the defect
  was worse than the comment claimed: the IR element type also fixes the
  host buffer's stride (`float64` is 8 bytes/element on the host,
  `Vector.float64`), so the emitted `device float*` strode the buffer at 4,
  and every element after the first was a bit-half of its neighbor — a
  wrong-answer defect, not a precision one. The fix is Metal-specific: no
  other backend needed it, because CUDA, OpenCL, HIP, GLSL and WGSL either
  support `double`/`f64` natively or already refused it — Metal alone had
  silently narrowed. See `docs/design/capability-model.md` §1 for the
  broader point this instance illustrates (a device capability collapsed to
  one boolean reads as "permitted" in every direction it is wrong).

- **`SNative` (raw target-language pass-through) gets a three-way answer,
  not a two-way one.** Read directly from the six sites that match on it:
  `sarek/codegen/Sarek_ir_ptx_stmt.ml:248-253` serves it unconditionally —
  "Pass-through: caller must supply valid PTX as the gpu closure," no
  device-context check. `Sarek_ir_cuda.ml:489-501`,
  `Sarek_ir_opencl.ml:533-546`, and `Sarek_ir_metal.ml:711-723` each serve it
  *conditionally* — only if `!current_framework` is `Some _`, else raise —
  which is exactly the ref that §3's GLSL bullet shows GLSL does not carry.
  `Sarek_ir_glsl.ml:1275-1277` and `Sarek_ir_wgsl.ml:1084-1086` both refuse
  it outright, emitting a `/* native code not supported in GLSL|WGSL */`
  comment and nothing else. Three distinct answers to the same construct,
  across backends that otherwise share a dispatch interface — an eighth
  backend would need its own answer to this one too, and there is no
  majority pattern to default to (2 conditional-serve, 1 unconditional-serve,
  2 refuse).

Each of these is a case where the seven existing backends do not agree with
each other, so there is no "the backend way" to extend to an eighth by
analogy. The cost of a new backend is not "the existing pattern, times one
more" — it is "find out, per construct, which of the several existing
disagreements this target lands in," which is closer to redoing part of the
analysis for all seven than doing it once for the new one.

## 4. What would reverse this

This decision rests on §2 and §3's premises — the cost is a full column, and
the columns disagree with each other. Any of the following breaks one of
those premises, and is a legitimate reason to revisit the decision — not a
reason to have decided differently now:

1. **A hardware or platform target with no path through the existing
   seven.** All seven current backends are CPU (`Native`, `Interpreter`) or
   GPU-compute (`CUDA`, `HIP`, `OpenCL`, `Vulkan`, `Metal`) reached through a
   shader/kernel-source or IR-execution model this codebase already
   generates for. A target that is neither — e.g. an FPGA/dataflow model,
   or a GPU API with no path through any of `Sarek_ir_{cuda,opencl,metal,glsl,wgsl}`'s
   shared `Dispatch` interface, or WGSL's promotion from codegen-only to a
   real device once `sarek/plugins/webgpu` stops being a stub — would not
   be "an eighth column of the same kind"; it would be filling a gap the
   seven cannot reach at all. That is a different decision than the one
   recorded here.

2. **A backend that subsumes two existing ones, so the count does not
   increase.** `README.md:89`'s CUDA/PTX-vs-CUDA/C split is the existing
   precedent for this: two registered frameworks, one backend, one column,
   because `Sarek_ir_cuda.ml` already serves both CUDA/C and HIP
   (`docs/design/width-addition-cost.md` §5.2: "CUDA and HIP share
   `Sarek_ir_cuda.ml` verbatim") — three framework names, one emitter file,
   counted as one column throughout this record. A genuinely new target
   that replaces, say, both `Vulkan` and `Metal` — rather than sitting
   beside them — would leave the column count at seven or fewer, not eight,
   and this decision would not apply to it.

3. **A structural change to the cost itself, such that a column stops being
   a full column.** Nothing currently prevents §3's kind of divergence
   (`Dispatch.framework` inconsistent across emitters with nothing to
   check it, `SNative` answered three different ways with nothing forcing
   agreement) or requires a cross-cutting change to touch every emitter's
   own file. If a future refactor made the seven emitters share enough
   structure that a typical cross-cutting change forced a compiler error in
   one place rather than a silent divergence in five — the way
   `test_backend_type_width_totality` and
   `test_type_width_totality` already close part of that gap for scalar
   widths (`docs/design/width-addition-cost.md` §3) — an eighth column
   might cost meaningfully less than the current seven do, and the
   arithmetic behind this decision would need redoing with the new cost.

None of these three conditions holds today. If a specific candidate
motivated re-asking the question in this session, it does not change that:
this record does not evaluate any specific eighth backend, only the
standing cost of having one more than seven, and that cost is unchanged by
which target might exercise it.

---

*Every count in §1 and every code claim in §2 and §3 was read from the file
and line cited, at the commit this record was written against
(`ed245581`, `origin/main`). §2's width-addition numbers are reproduced from
`docs/design/width-addition-cost.md`, not re-measured here.*
