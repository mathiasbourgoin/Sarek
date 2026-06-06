# Native And Interpreter Plugins

## Component Inventory

Reviewed plugin files: `sarek/plugins/native/**` and `sarek/plugins/interpreter/**`, including plugin READMEs, base implementations, plugin registration files, error modules, dunes, and colocated tests.

## Per-File Purpose

- `sarek/plugins/native/README.md`: native plugin design and usage notes.
- `sarek/plugins/native/Native_plugin_base.ml`: native backend memory buffers, kernel argument storage, launch compatibility, direct execution registration, and registry helpers.
- `sarek/plugins/native/Native_plugin.ml`: framework registration for the native backend and direct execution hook.
- `sarek/plugins/native/Native_error.ml`: native plugin errors.
- `sarek/plugins/native/test/test_native_error.ml`: native error tests.
- `sarek/plugins/interpreter/README.md`: interpreter plugin design and usage notes.
- `sarek/plugins/interpreter/Interpreter_plugin_base.ml`: interpreter backend memory buffers, kernel argument storage, launch compatibility, direct execution registration, and registry helpers.
- `sarek/plugins/interpreter/Interpreter_plugin.ml`: framework registration for interpreter backend and direct execution hook.
- `sarek/plugins/interpreter/Interpreter_error.ml`: interpreter plugin errors.
- `sarek/plugins/interpreter/test/test_interpreter_error.ml`: interpreter plugin error tests.

## Features/APIs

- Plugin-level backend registration with the framework registry.
- Direct execution functions for native and interpreter backends.
- Compatibility memory module exposing allocate/free/copy operations.
- Compatibility kernel module exposing `set_arg_*` and `launch`.
- Plugin-local kernel registries and listing helpers.
- Error modules for plugin-specific failures.

## Invariants

- `set_arg_* idx value` must set the argument at `idx`, independent of call order.
- Legacy compatibility launch and direct execution must agree on argument order and vector element types.
- Plugin backend priority should match framework documentation or the docs should match code.
- Plugin registries should either be initialization-only or synchronized.
- Vector arguments should preserve type, length, and writeback semantics after interpreter/native execution.

## Potential Invariant Violations/Bugs

- Native plugin argument setters ignore `idx` and prepend to a list at `sarek/plugins/native/Native_plugin_base.ml:568-601`; launch reverses call order at `sarek/plugins/native/Native_plugin_base.ml:611-617`. This works only if callers set arguments monotonically in order.
- Interpreter plugin argument setters have the same issue at `sarek/plugins/interpreter/Interpreter_plugin_base.ml:536-570`.
- Interpreter legacy `Kernel.launch` type-detects vector buffers only by element size at `sarek/plugins/interpreter/Interpreter_plugin_base.ml:575-661`. `elem_size = 4` becomes float32 and `elem_size = 8` becomes float64 at `sarek/plugins/interpreter/Interpreter_plugin_base.ml:590-631`, so int32/int64 vectors are not represented correctly. The direct execution path in `sarek/plugins/interpreter/Interpreter_plugin.ml:102-128` is better and should be preferred.
- Native backend priority is registered as 10 at `sarek/plugins/native/Native_plugin.ml:361-369`, and interpreter priority as 5 at `sarek/plugins/interpreter/Interpreter_plugin.ml:168-176`. The framework README documents native 50 and interpreter 30 at `sarek/framework/README.md:103-109`.
- Plugin kernel registries are mutable unsynchronized hashtables: native registry at `sarek/plugins/native/Native_plugin_base.ml:30-35` with register/list at `sarek/plugins/native/Native_plugin_base.ml:663-670`, interpreter registry at `sarek/plugins/interpreter/Interpreter_plugin_base.ml:23-24` with register/list at `sarek/plugins/interpreter/Interpreter_plugin_base.ml:686-693`.

## Performance Or Maintainability Risks

- Compatibility `Kernel.launch` paths duplicate direct execution behavior and are easier to get wrong.
- Argument order depends on call convention rather than data structure invariants.
- Backend priority drift between docs and code can produce surprising backend selection.
- Unsynchronized plugin-local registries can race during concurrent registration.
- Interpreter vector type inference by byte width cannot distinguish integer and float vectors.

## Related Tests

- `sarek/plugins/native/test/test_native_error.ml`: native error formatting.
- `sarek/plugins/interpreter/test/test_interpreter_error.ml`: interpreter error formatting.

No scoped plugin tests were found for memory copies, kernel argument order, direct execution, or legacy launch behavior.

## Missing Tests

- `set_arg_*` called out of order and overwritten by index.
- Native and interpreter direct execution argument order.
- Interpreter legacy launch with int32, int64, float32, and float64 vectors.
- Vector writeback after interpreter execution.
- Plugin priority consistency with framework documentation.
- Concurrent registration/listing of plugin kernels.

## Concrete Improvement/Fix Candidates

- Store kernel arguments in an indexed array or map keyed by `idx`; support replacement and detect missing indices at launch.
- Prefer direct execution APIs and deprecate legacy `Kernel.launch`, or make legacy launch delegate to direct execution after typed argument conversion.
- Carry vector element type in plugin buffers instead of inferring from byte width.
- Align plugin priorities with `sarek/framework/README.md` or update the documentation.
- Add mutex protection or initialization-only guards around plugin-local registries.
