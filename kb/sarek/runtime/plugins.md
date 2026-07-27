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

- `set_arg_* idx value` must set the argument at `idx`, independent of call order. **Enforced 2026-07-02 (merged)** via `Spoc_framework.Kernel_args` — see Potential Invariant Violations/Bugs below for the fix.
- Legacy compatibility launch and direct execution must agree on argument order and vector element types.
- Plugin backend priority should match framework documentation or the docs should match code.
- Plugin registries should either be initialization-only or synchronized.
- Vector arguments should preserve type, length, and writeback semantics after interpreter/native execution.

## Potential Invariant Violations/Bugs

- **`set_arg_*` ignoring `idx` — fixed 2026-07-02 (merged) via `Spoc_framework.Kernel_args`.** Both native and interpreter plugins now route argument storage through the shared indexed container (`spoc/framework/Kernel_args.ml`/`.mli`, see [../../spoc/framework.md](../../spoc/framework.md)):
  - `Native_plugin_base.ml:565` defines `type args = Framework_sig.exec_arg Spoc_framework.Kernel_args.t`; `set_arg_buffer`/`set_arg_int32`/`set_arg_int64`/`set_arg_float32`/`set_arg_float64` (`:721-733`) all call `Kernel_args.set args idx (...)`, storing at the caller-supplied `idx` (last-set-wins on duplicates) instead of ignoring it and prepending to a list. Launch validates via `Kernel_args.validate_and_extract args ~expected_count` (`:755-757`), which rejects internal gaps/duplicates/out-of-range indices before launch.
  - `Interpreter_plugin_base.ml:528-680` has the identical pattern: `type args = Framework_sig.exec_arg Spoc_framework.Kernel_args.t`, `set_arg_*` at `:653-665` calling `Kernel_args.set`, and `validate_and_extract` at `:680`.
  - Practical effect: callers may now set arguments in any order (or overwrite by index) and get a validated, gap-free array at launch, rather than depending on strictly monotonic call order. Verified against source 2026-07-02.
- **Fixed as of 2026-07-02** (was: interpreter legacy `Kernel.launch` type-detected vector buffers only by byte width, e.g. `elem_size = 4` → float32 / `elem_size = 8` → float64, misrepresenting int32/int64 vectors): `Kernel.launch` now consumes typed `Framework_sig.exec_arg` values built by the `set_arg_*` setters (`EA_Vec`/`EA_Int32`/`EA_Int64`/`EA_Float32`/`EA_Float64`) and converts each to the interpreter's `ArgArray`/`ArgScalar` representation by matching on the exec_arg's own type tag, not on buffer byte size — `sarek/plugins/interpreter/Interpreter_plugin_base.ml:534-674` (`launch` at `:669-711`). No byte-width sniffing remains in this path.
- Native backend priority is registered as 10 at `sarek/plugins/native/Native_plugin.ml:361-369`, and interpreter priority as 5 at `sarek/plugins/interpreter/Interpreter_plugin.ml:168-176`. The framework README documents native 50 and interpreter 30 at `sarek/framework/README.md:103-109`.
- Plugin kernel registries are mutable unsynchronized hashtables: native registry at `sarek/plugins/native/Native_plugin_base.ml:30-35` with register/list at `sarek/plugins/native/Native_plugin_base.ml:663-670`, interpreter registry at `sarek/plugins/interpreter/Interpreter_plugin_base.ml:23-24` with register/list at `sarek/plugins/interpreter/Interpreter_plugin_base.ml:686-693`.

## Performance Or Maintainability Risks

- Compatibility `Kernel.launch` paths duplicate direct execution behavior and are easier to get wrong.
- Argument order depends on call convention rather than data structure invariants.
- Backend priority drift between docs and code can produce surprising backend selection.
- Unsynchronized plugin-local registries can race during concurrent registration.

## Related Tests

- `sarek/plugins/native/test/test_native_error.ml`: native error formatting.
- `sarek/plugins/interpreter/test/test_interpreter_error.ml`: interpreter error formatting.

No scoped plugin tests were found for memory copies, kernel argument order, direct execution, or legacy launch behavior.

## Missing Tests

- (Verify) `set_arg_*` called out of order and overwritten by index now exercises `Kernel_args.set`/`validate_and_extract` for both native and interpreter plugins — the 2026-07-02 fix landed the mechanism; confirm plugin-level test coverage exists beyond `spoc/framework/test/` unit tests for `Kernel_args` itself.
- Native and interpreter direct execution argument order.
- Interpreter legacy launch with int32, int64, float32, and float64 vectors (regression coverage for the 2026-07-02 typed-exec_args fix, to lock in that int32/int64 are no longer confused with float32/float64).
- Vector writeback after interpreter execution.
- Plugin priority consistency with framework documentation.
- Concurrent registration/listing of plugin kernels.

## Concrete Improvement/Fix Candidates

- **Done 2026-07-02 (merged):** ~~store kernel arguments in an indexed array or map keyed by `idx`; support replacement and detect missing indices at launch~~ — implemented via `Spoc_framework.Kernel_args`.
- Prefer direct execution APIs and deprecate legacy `Kernel.launch`, or make legacy launch delegate to direct execution after typed argument conversion.
- Align plugin priorities with `sarek/framework/README.md` or update the documentation.
- Add mutex protection or initialization-only guards around plugin-local registries.
