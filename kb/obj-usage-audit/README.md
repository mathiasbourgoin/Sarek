# Obj Usage Audit

Audit scope: first-party OCaml sources under `spoc/`, `sarek/`, backend
packages, benchmarks, tools, and scripts.

## What is actually removed — census re-run 2026-07-27 at `c92a9a28`

This section used to claim the implementation "removes active source hits for
`Obj.magic`, `Obj.repr`, `Obj.obj`, `Obj.t`, and the old
`internal_get_vector_obj` escape hatch". **That was true of one and a half of the
five and read as true of all five.** It is the summary-versus-evidence drift this
project has spent time removing elsewhere: a heading true of part of its section,
believed about the whole.

| Symbol | Live hits | Status |
| --- | --- | --- |
| `Obj.magic` | **0** | removed — the unsound cast the audit existed for |
| `internal_get_vector_obj` | **0** | removed |
| `Obj.repr` | 3 code + 1 doc-comment | **still present**, see roles below |
| `Obj.obj` | 2 | **still present** |
| `Obj.t` | 3 code + 2 doc-comment | **still present** |

The eight numbered boundaries below were each replaced, and the replacements are
in the tree — that part of the old claim holds. What did not hold is the leap
from "these eight are fixed" to "the symbols are gone". The remaining hits are in
three roles, none of them the type-recovery cast the audit targeted, and each is
argued in place rather than merely surviving:

1. **Liveness anchors for ctypes** — `sarek/core/Memory.ml:101` passes
   `Obj.repr ba` to `Fat.make ~managed` purely to keep a bigarray alive. The
   value is never read back at a type. (`Memory.ml:63` and
   `sarek/tests/unit/test_float16.ml:429` are doc comments describing the same
   call, not code.)
2. **Kernarg retention** — `sarek-cuda/Cuda_api.ml:66,740` and
   `sarek-hip/Hip_api.ml:40,549` keep `(params, refs)` as `Obj.t` in
   `pending_kernargs` for liveness only, again never recovered at a type. Both
   files say so in the declaration's comment. This is the lifetime machinery
   whose absence produced a five-hour misattribution to ZLUDA.
3. **A remaining untyped registry** — `sarek/tuple_vec/Sarek_tuple_vec.ml:102`
   is a `(string, Obj.t) Hashtbl.t` with `Obj.obj` on lookup at `:121` and
   `:135`. **This is a genuine escape hatch of the kind the audit targeted, and
   it postdates the audit**; `Sarek_tuple_vec` is the tuple-shape registry, a
   different thing from the `Sarek_type_helpers` registry that
   [02-custom-helper-registry.md](02-custom-helper-registry.md) records as fixed.
   It carries its own soundness argument in the source — the mangled shape name
   uniquely determines the OCaml type, and host and native side compare the same
   `Type_id` via `Type_id.equal` — plus a mutex closing the OCaml 5 data race
   (audit finding L1). That argument is *by-construction*, not machine-checked.
   It is the one open item in this audit.

Census command, and the exact spelling matters — this is a lexical count, so a
different spelling of the same construct would not appear:

```sh
grep -rn 'Obj\.\(magic\|repr\|obj\|t\)' --include=*.ml --include=*.mli \
  spoc sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal sarek-hip
```

## Priority Index

1. [Native vector execution boundary](01-native-vector-boundary.md)
2. [Custom value helper registry](02-custom-helper-registry.md)
3. [Interpreter plugin bridge](03-interpreter-plugin-bridge.md)
4. [Plugin buffer copies](04-plugin-buffer-copies.md)
5. [Custom shared memory arrays](05-custom-shared-memory-arrays.md)
6. [Native runtime test direct vector access](06-native-runtime-test-vector-access.md)
7. [Legacy native direct API](07-legacy-native-direct-api.md)
8. [PPX custom descriptor qualification](08-ppx-custom-descriptor-qualification.md)

## Implemented Direction

The replacement strategy is a mix of GADT-style runtime witnesses, typed
existentials, first-class modules, and typed Bigarray loops. This keeps values in
their real static types across backend boundaries and turns mismatches into
explicit errors instead of undefined representation casts.

## Verification Target

The source audit command used by the **original** audit (superseded as a status
claim by the 2026-07-27 census above, which is narrower and current):

```sh
rg -n "Obj\.(magic|repr|obj)|Obj\.t|\bObj\b|internal_get_vector_obj|shared_key|custom_key|get_any|set_any" -S --glob '*.ml' --glob '*.mli' spoc sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal benchmarks tools scripts
```
