# No eighth backend

**Decision:** decline to add an eighth backend. Every additional one is a
full column in every cross-cutting change from here on (§2, §3) — that is
the premise this record argues and the one §4's reversal condition attaches
to. (Seven already feels like enough for this project's story, but that is
the author's judgement: it is not argued in this record and carries no
reversal condition of its own.)

**Date:** 2026-07-31.

**Method:** the same as `docs/design/width-addition-cost.md` (backend-column
cost, not width-addition cost) — counted by reading real diffs and code, not
estimated. Every count below states the command, file:line, or sibling
record that produced it.

---

## 1. What "seven" counts

`README.md:70-81` lists the devices that appear in `Device.init` /
`sarek-device-info`: `CUDA/PTX`, `CUDA/C`, `HIP`, `OpenCL`, `Vulkan`, `Metal`,
`Native`, `Interpreter` — eight table rows. `README.md:89-90` states only
that "The CUDA family registers as two frameworks (`CUDA/PTX` and `CUDA/C`);
use `Device.filter_cuda ()` to match both" — it does not itself collapse the
count to seven. The collapse is this record's own definitional choice, and
it matches the passage that already makes it, `README.md:134,138-143`: five
GPU backend packages (`sarek-cuda`, `sarek-hip`, `sarek-opencl`,
`sarek-vulkan`, `sarek-metal` — with `sarek-cuda` covering both `CUDA/PTX`
and `CUDA/C`) plus `Native` and `Interpreter` from `sarek/plugins/`.
Collapsed that way, it is **seven**: CUDA, HIP, OpenCL, Vulkan, Metal,
Native, Interpreter.

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
files; the OCaml compiler forced 20 of those 40 — **21** in total, counting
one file that postdates the slice — across **seven** separate build rounds,
because each layer's errors are hidden behind the previous layer's failure;
the other **20** — including the FFI layer, the host storage layer, and the
exec-arg dispatch — the compiler never asks for at all and must be found by
reading.

A backend is not a scalar type, but the same shape of cost shows up whenever
a change must reach every emitter rather than one:

- **The dispatch-tag divergence** (§3 below) was found by reading five
  emitter files end to end, not by a type error — nothing forces
  `Dispatch.framework` to be consistent across emitters, so the four
  ref-carrying ones and the one constant one type-checked identically.
- **The SNative three-way split** (§3 below) is a match arm present, in some
  form, in all five codegen files with a `Dispatch.framework` field
  (`Sarek_ir_cuda.ml`, `Sarek_ir_opencl.ml`, `Sarek_ir_metal.ml`,
  `Sarek_ir_glsl.ml`, `Sarek_ir_wgsl.ml`) plus the separate PTX emitter
  (`Sarek_ir_ptx_stmt.ml`) — six sites, three different behaviors, and each
  one independently decided by whoever wrote that emitter's arm.
- **The per-generation dispatch state is not shared machinery.** Four of
  the five `Dispatch.framework`-carrying emitters each define their own
  module-level `current_framework : string option ref = ref None`:
  `Sarek_ir_cuda.ml:36`, `Sarek_ir_opencl.ml:47`, `Sarek_ir_metal.ml:36`,
  `Sarek_ir_wgsl.ml:41` — four separate `ref` cells, four separate
  definitions, not one shared module. The fifth, GLSL, has no such ref at
  all (§3 below). Converting that pattern to a properly threaded
  per-generation value — replacing a global mutable ref with a value passed
  through the call chain — is therefore not one change: it is four separate
  ref cells across four files, plus a fifth emitter (GLSL) that would have
  to be brought into a pattern it never joined — no single definition site
  reaches them. It is the same shape as §2's headline number: a change that
  must reach every target, done once per file because nothing forces it to
  be done once at all.

This record measures three different column counts for changes of this
shape, not one: five dispatch-tag sites (`Dispatch.framework`, above), six
SNative-deciding sites (three codegen files' conditional arms, PTX's
unconditional pass-through, and GLSL/WGSL's refusal — see §3), and four
independent ref cells. None of the three is seven; "seven" is §1's
device/framework count, not a column count for this shape of change, and
this record does not conflate the two again below. An eighth backend is not
a uniform increase on any of the three — it would add a sixth site to the
dispatch-tag divergence, a seventh to the SNative split, and a fifth
independent ref cell (or a second silent exception to the pattern,
alongside GLSL) to the state-threading count.

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
  kernel-vector or local-array parameter either is a device pointer (CUDA,
  OpenCL, Metal), so `src = dst` emits `src == dst` as pointer equality, or
  is compared element-wise as a value (GLSL, which has no pointers) — either
  way it type-checks. Verified by execution on this host: `clang -x cl
  -cl-std=CL1.2` (OpenCL) and `glslangValidator -V` (GLSL, both whole-value
  array `==` and SSBO-buffer `==`) both exit 0. CUDA and Metal accept the
  same shape by C-family analogy; neither was executed here (`nvcc` and a
  Metal toolchain are both absent from this host). WGSL does not generalize
  the same way: naga has no equality operator on `array<T>` / `array<T, N>`
  at all and rejects the identical shape with `"Incompatible operands:
  Equal(Array …)"`, reproduced through the full frontend pipeline and
  confirmed against naga 30.0.0 on this host. Fixed by refusing the shape at
  exactly one emitter,
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
  fix generalized to it.** Before the fix landed in commit `6bdc751e` (the
  Metal-capability slice that first gave the codegen a `float64`-support
  signal to check), `sarek/codegen/Sarek_ir_metal.ml`'s `TFloat64` arm
  emitted `"float"` with a comment noting Metal has no `double` — and no
  refusal anywhere on the path. Commit `2bc893d1` (a follow-up review that
  re-examined Metal's float64 handling for concurrency and buffer-stride
  issues) found the defect was worse than the comment claimed: the IR
  element type also fixes the
  host buffer's stride (`float64` is 8 bytes/element on the host,
  `Vector.float64`), so the emitted `device float*` strode the buffer at 4,
  and every element after the first was a bit-half of its neighbor — a
  wrong-answer defect, not a precision one. The fix is Metal-specific: no
  other backend needed it, because CUDA, OpenCL, HIP, GLSL and WGSL either
  support `double`/`f64` natively or already refused it — Metal alone had
  silently narrowed. See `docs/design/capability-model.md` §1 for the
  broader point this instance illustrates (a device capability collapsed to
  one boolean reads as "permitted" in every direction it is wrong).

- **`SNative` (raw target-language pass-through) gets at least a three-way
  answer among the codegen emitters, and a fourth from the Interpreter.**
  Read directly from the six emitter sites that decide an answer for it —
  not every site that merely pattern-matches it in passing; `SNative` also
  appears in pass-through traversal arms in `Sarek_ir_inline_vec.ml` and
  `Sarek_ir_softmath.ml` that don't decide its behavior:
  `sarek/codegen/Sarek_ir_ptx_stmt.ml:248-253` serves it unconditionally —
  "Pass-through: caller must supply valid PTX as the gpu closure," no
  device-context check. `Sarek_ir_cuda.ml:489-501`,
  `Sarek_ir_opencl.ml:533-546`, and `Sarek_ir_metal.ml:711-723` each serve it
  *conditionally* — only if `!current_framework` is `Some _`, else raise —
  which is exactly the ref that §3's GLSL bullet shows GLSL does not carry.
  `Sarek_ir_glsl.ml:1275-1277` and `Sarek_ir_wgsl.ml:1084-1086` both refuse
  it outright, emitting a `/* native code not supported in GLSL|WGSL */`
  comment and nothing else. That is three distinct answers across the five
  codegen emitters plus the separate PTX emitter — (3 conditional-serve, 1
  unconditional-serve, 2 refuse), no pattern held by more than half. The
  Interpreter — one of the seven backends §1 counts, though it has no
  codegen emitter — supplies a fourth answer of its own:
  `sarek/interp/Sarek_ir_interp_eval.ml:433` calls the construct's typed
  OCaml fallback directly (`ocaml.run ...`), neither serving generated code
  nor refusing. An eighth backend would need its own answer to this one
  too, and there is no majority among the existing four answers to default
  to.

Each of these is a case where the existing columns do not agree with each
other — including WGSL, one of the five `Dispatch.framework` emitters even
though §1 does not count it among the seven backends — so there is no "the
backend way" to extend to an eighth by analogy. The
cost of a new backend is not "the existing pattern, times one more" — it is
"find out, per construct, which of the several existing disagreements this
target lands in," which is closer to redoing part of that analysis for all
of them than doing it once for the new one.

## 4. What would reverse this — and what wouldn't, but looks like it might

This decision rests on §2 and §3's premises — the cost is a full column, and
the columns disagree with each other. Exactly one of the three items below
breaks either premise and is a legitimate reason to revisit the decision.
The other two are scope boundaries, not reversals: they describe cases this
record does not govern, listed because each is the kind of case most likely
to be mistaken for a reason this decision was wrong.

**Scope boundaries — not reversals:**

1. **A hardware or platform target with no path through the existing
   seven.** All seven current backends are CPU (`Native`, `Interpreter`) or
   GPU-compute (`CUDA`, `HIP`, `OpenCL`, `Vulkan`, `Metal`) reached through a
   shader/kernel-source or IR-execution model this codebase already
   generates for. A target that is neither — e.g. an FPGA/dataflow model,
   or a GPU API with no path through any of `Sarek_ir_{cuda,opencl,metal,glsl,wgsl}`'s
   shared `Dispatch` interface — would not be "an eighth column of the same
   kind"; it would be filling a gap the seven cannot reach at all. That
   does not make §2 or §3's premises false — it says the premises don't
   apply to that target, which is a different decision than the one
   recorded here. The one crisply observable version of this boundary is
   WGSL's own promotion from codegen-only to a real device: it would be
   observable the day `sarek/plugins/webgpu`'s `Webgpu_plugin.ml:47 let
   is_available () = false` flips to `true`. An FPGA/dataflow target is a
   real category but has no comparably countable trigger.

2. **A backend that subsumes two existing ones, so the count does not
   increase.** `README.md:89-90` records a precedent for one kind of
   subsumption already in this codebase: `CUDA/PTX` and `CUDA/C` are two
   registered frameworks that §1 folds into a single column, `CUDA`. A
   different kind of subsumption is visible at the emitter-file level:
   `sarek/codegen/Sarek_ir_cuda.ml` already serves both `CUDA/C` and `HIP`
   (`docs/design/width-addition-cost.md` §5.2: "CUDA and HIP share
   `Sarek_ir_cuda.ml` verbatim") — but §1 still counts `HIP` as its own
   column, separate from `CUDA`, so sharing an emitter file is not by
   itself what collapses the device/framework count; it is a precedent for
   "one file, several names," not for "one column" in the sense §1 uses. A
   genuinely new target that replaced, say, both `Vulkan` and `Metal` —
   rather than sitting beside them — would leave the column count at seven
   or fewer, not eight. That is a scope boundary, not a reason this
   decision would have been wrong: it describes a different backend than
   any candidate on the table today.

**The one reversal condition:**

3. **A structural change to the cost itself, such that a column stops being
   a full column.** Nothing currently prevents §3's kind of divergence
   (`Dispatch.framework` inconsistent across emitters with nothing to
   check it, `SNative` answered four different ways with nothing forcing
   agreement) or requires a cross-cutting change to touch every emitter's
   own file. A concrete, countable version of this condition: today there
   are four independent `current_framework` ref cells (§2 above); if a
   future refactor collapsed them to one shared, properly-threaded value —
   the same shape of change `sarek/tests/unit/test_host_ir_width_agreement.ml`
   made for scalar-width agreement, closing the gap that
   `test_backend_type_width_totality` and `test_type_width_totality` leave
   between them, since those two "cover the ends of the pipeline while
   meeting nowhere" (`docs/design/width-addition-cost.md` §3) — a typical
   cross-cutting change would force a compiler error in one place rather
   than a silent divergence across four. An eighth column might then cost
   meaningfully less than the current ones do, and the arithmetic behind
   this decision would need redoing with the new cost.

Neither scope boundary applies today, and condition 3 — the one reversal
condition this record identifies — does not hold either: the four
`current_framework` ref cells have not been collapsed, and no
compiler-enforced check has replaced the silent-divergence pattern §3
describes. This record does not claim condition 3 is the only conceivable
way the cost premise could break, only the one its own analysis in §2 and
§3 exposes. If a specific candidate motivated re-asking the question in
this session, it does not change that: this record does not evaluate any
specific eighth backend, only the standing cost of having one more than
seven, and that cost is unchanged by which target might exercise it.

---

*Every count in §1 and every code claim in §2 and §3 was read from the file
and line cited, at the commit this record was written against
(`ed245581`, `origin/main`). §2's width-addition numbers are reproduced from
`docs/design/width-addition-cost.md`, not re-measured here.*
