# Implementation Report — JS WebGPU runner + ABI descriptor

**Branch:** `roadmap/webgpu-runner-abi`
**Steps:** 7/7 complete. Do NOT push (lead pushes/opens PR).

## New / changed files

| File | Change |
|------|--------|
| `sarek/codegen/Sarek_wgsl_abi.ml` | NEW — ABI descriptor type + hand-rolled JSON serializer |
| `sarek/codegen/Sarek_wgsl_abi.mli` | NEW — public `.mli` with docstrings |
| `sarek/codegen/dune` | +`Sarek_wgsl_abi` to modules list |
| `sarek/codegen/Sarek_ir_wgsl.ml` | +`abi` builder function (reuses split_params/escape_wgsl_name) |
| `sarek/transpile/Sarek_transpile.ml` | extract `run_pipeline` helper; add `of_source_with_abi` |
| `sarek/transpile/Sarek_transpile.mli` | +`of_source_with_abi` val declaration |
| `sarek/transpile/web/transpile_js.ml` | +`transpileWithAbi` jsoo export; `transpile` unchanged |
| `gh-pages/javascripts/sarek_webgpu_runner.js` | NEW — generic WebGPU runner script |
| `sarek/transpile/web/test/webgpu_wgsl_test.mjs` | rewritten to use transpileWithAbi + SarekWebGPU |

## Commits (7 steps)

```
af54f8c1  Add Sarek_wgsl_abi: ABI descriptor type + hand-rolled JSON serializer
1dd878be  Add abi builder to Sarek_ir_wgsl reusing split_params/escape_wgsl_name
f8341d0b  Add of_source_with_abi to Sarek_transpile; extract run_pipeline helper
44ce9578  Add transpileWithAbi jsoo export; keep transpile export byte-identical
7ed7626e  Add sarek_webgpu_runner.js: generic ABI-driven WebGPU kernel runner
828a88ee  Rewrite GPU acceptance test to use transpileWithAbi + SarekWebGPU runner
ddd22db2  Apply ocamlformat and SPDX; all new ML files carry CECILL-B headers
```

## Actual `to_json` output

### vector_add

Source: `fun (a:float32 vector)(b:float32 vector)(c:float32 vector) -> let i = global_thread_id in c.(i) <- a.(i) +. b.(i)`

```json
{"kernelName":"sarek_kern","workgroupSize":[256,1,1],"buffers":[{"name":"a","binding":0,"elementType":"f32","access":"read_write"},{"name":"b","binding":1,"elementType":"f32","access":"read_write"},{"name":"c","binding":2,"elementType":"f32","access":"read_write"}],"params":{"binding":3,"byteSize":16,"fields":[{"name":"sarek_a_length","type":"i32","offset":0,"kind":"length","of":"a"},{"name":"sarek_b_length","type":"i32","offset":4,"kind":"length","of":"b"},{"name":"sarek_c_length","type":"i32","offset":8,"kind":"length","of":"c"}]}}
```

Note: `byteSize=16` because 3 fields × 4 bytes = 12, rounded up to the next multiple of 16.

### bounds_check

Source: `fun (a:float32 vector)(b:float32 vector)(n:int32) -> let i = global_thread_id in b.(i) <- (if i < n then a.(i) else 0.0)`

```json
{"kernelName":"sarek_kern","workgroupSize":[256,1,1],"buffers":[{"name":"a","binding":0,"elementType":"f32","access":"read_write"},{"name":"b","binding":1,"elementType":"f32","access":"read_write"}],"params":{"binding":2,"byteSize":16,"fields":[{"name":"sarek_a_length","type":"i32","offset":0,"kind":"length","of":"a"},{"name":"sarek_b_length","type":"i32","offset":4,"kind":"length","of":"b"},{"name":"n","type":"i32","offset":8,"kind":"scalar"}]}}
```

## SarekWebGPU runner API

`globalThis.SarekWebGPU` (loaded via `<script src="sarek_webgpu_runner.js">`):

- `async getAdapter()` — requests `{powerPreference:'high-performance'}`; returns adapter or null.
- `async run(wgsl, abi, {inputs, scalars})` — `{outputs}`:
  - `inputs`: `{[bufferName]: TypedArray}` for each storage buffer.
  - `scalars`: `{[name]: number}` for `kind:"scalar"` params fields.
  - Packs uniform per `params.fields`, each at `field.offset` (4-byte slots).
  - Dispatches `ceil(maxElements / workgroupSize[0])` workgroups.
  - Returns `outputs:{[bufferName]: TypedArray}` for all storage buffers.
  - Throws on compile error (includes WGSL error messages + line numbers).

## Quality gate results

| Gate | Result |
|------|--------|
| `dune build sarek/codegen` | PASS |
| `dune build @sarek/tests/runtest` (26 goldens) | PASS — all byte-identical |
| `dune exec sarek/transpile/test/test_transpile_proof.exe` | PASS |
| `dune build sarek/transpile/web/transpile_js.bc.js` | PASS |
| `dune build @fmt` | PASS — clean |
| `grep SarekWebGPU sarek_webgpu_runner.js` | PASS |
| `grep transpileWithAbi webgpu_wgsl_test.mjs` | PASS |

## Residual risks

- `make wgsl-gpu-test` (Playwright + real GPU): not run by implementer — the lead runs this on the RX 7900 XTX per the brief.
- Other backends (CUDA, Metal, OpenCL): not affected; ABI is WGSL-only; `of_source_with_abi` returns `Internal_error` for non-WGSL as specified.
- `kernelName` is always `"sarek_kern"` (the default from the transpiler); the brief says to use `k.kern_name` and the emitted value matches whatever the frontend sets.
