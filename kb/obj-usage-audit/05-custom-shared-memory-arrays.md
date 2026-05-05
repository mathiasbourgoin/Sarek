# Custom Shared Memory Arrays

## Original Risk

Custom shared-memory arrays were stored in an erased slot. Reusing a name for a
different custom type could recover the old array at the new type.

## Implemented Alternative

`Sarek_cpu_runtime.alloc_shared_with_key` stores each custom shared array with a
`Type_id` witness. Reuse checks witness equality before returning the array.

## Why This Improves The Code

Shared memory reuse remains fast for matching calls and now rejects type
mismatches at the allocation boundary. Unit tests cover allocation, reuse, and
mismatch rejection.
