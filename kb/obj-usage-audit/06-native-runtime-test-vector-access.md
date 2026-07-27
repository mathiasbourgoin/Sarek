# Native Runtime Test Vector Access

## Original Risk

The native runtime integration test extracted raw Bigarray vectors from
`EXEC_VECTOR` modules through a direct object escape hatch. That test codified
the unsafe API and prevented removing it.

## Implemented Alternative

The test kernel now reads and writes vector elements through `V.get` and `V.set`
typed-value accessors. Native runtime buffers implement scalar typed-value
access for Bigarray-backed buffers.

## Why This Improves The Code

The integration test now exercises the same typed public execution surface as
native kernels receive. It also verifies that runtime `ArgBuffer` values can be
used without accessing backend-private storage.
