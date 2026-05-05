# Obj Usage Audit

Audit scope: first-party OCaml sources under `spoc/`, `sarek/`, backend
packages, benchmarks, tools, and scripts. The implementation now removes active
source hits for `Obj.magic`, `Obj.repr`, `Obj.obj`, `Obj.t`, and the old
`internal_get_vector_obj` escape hatch.

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

The source audit command used for this PR:

```sh
rg -n "Obj\.(magic|repr|obj)|Obj\.t|\bObj\b|internal_get_vector_obj|shared_key|custom_key|get_any|set_any" -S --glob '*.ml' --glob '*.mli' spoc sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal benchmarks tools scripts
```
