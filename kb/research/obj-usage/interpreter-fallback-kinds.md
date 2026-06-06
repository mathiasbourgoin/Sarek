# Explicit Interpreter Fallback Kinds

Priority: P2. **Status (2026-06-02):** the interpreter no longer contains `Obj` casts
(none remain in `Sarek_ir_interp*.ml`). The interpreter was also split (see
[interpreter](../../sarek/runtime/interpreter.md)); the vector bridge functions stayed in
the reduced main module. References repointed to current locations.

## Obj Sites (former)

- `sarek/sarek/Sarek_ir_interp.ml:239` (`vector_to_array`): formerly the fallback
  `Vector.get` cast to `VInt32`; no `Obj` cast present now.
- `sarek/sarek/Sarek_ir_interp.ml:274` (`array_to_vector`): formerly the fallback
  `to_int32` cast into `Vector.set`; no `Obj` cast present now.

## Assumed Invariant

Every scalar vector kind not explicitly matched can be represented as an `int32`. The actual unmatched scalar kinds are `Char` and `Complex32` from `sarek/core/Vector_types.ml:22-23`.

## Existing Project Pattern

`sarek/sarek/Execute.ml:349-437` already handles these cases without `Obj`:

- `Char` becomes `VInt32 (Int32.of_int (Char.code c))`.
- Writeback converts `VInt32` back through `Char.chr`.
- `Complex32` is explicitly unsupported/skipped.

## Viable Replacement

Copy the explicit handling into `Sarek_ir_interp.vector_to_array` and `array_to_vector`:

```ocaml
| Scalar Char ->
    Array.init len (fun i ->
      VInt32 (Int32.of_int (Char.code (Vector.get vec i))))
| Scalar Complex32 ->
    Interp_error.raise_error
      (Unsupported_operation { operation = "vector_to_array";
                               reason = "Complex32 vectors are not supported" })
```

For writeback:

```ocaml
| Scalar Char ->
    Vector.set vec i (Char.chr (Int32.to_int (to_int32 arr.(i))))
| Scalar Complex32 -> structured unsupported error
```

## Expected Impact

Small but clean: removes two `Obj.magic` calls and makes unsupported `Complex32` deterministic. Runtime performance is unchanged.

## Tests

- Add `Sarek_ir_interp` vector bridge tests for `Char`.
- Add an unsupported `Complex32` test that checks the structured error.
- Compare behavior with `Execute.vector_to_interpreter_array` to keep both bridges aligned.

