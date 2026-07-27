# Interpreter Plugin Bridge

## Original Risk

The interpreter plugin converted framework execution arguments back into kernel
arguments through raw representation conversion. That made interpreter execution
depend on the hidden layout of vector and scalar wrappers.

## Implemented Alternative

`Sarek_ir_interp` now executes directly from `Framework_sig.exec_arg list` using
typed conversion helpers:

- scalar `typed_value` conversion for scalar arguments,
- vector-to-array and array-to-vector writeback through typed accessors,
- custom conversion through typed helper lookup.

## Why This Improves The Code

The interpreter no longer needs to reconstruct static types from opaque runtime
objects. Writeback is explicit, and unsupported conversions are surfaced as
normal interpreter errors.
