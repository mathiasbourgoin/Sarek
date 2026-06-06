# spoc/registry

## Component Inventory

- `spoc/registry/README.md`: explains runtime registration, PPX-generated registration pattern, lookup APIs, and examples.
- `spoc/registry/dune`: builds public library `spoc.registry` from `Sarek_registry`, depending on `spoc_framework` (`spoc/registry/dune:4-11`).
- `spoc/registry/Sarek_registry.ml`: global runtime registries and lookup/code-generation helpers.
- `spoc/registry/test/test_sarek_registry.ml`: tests type, record, variant, function, short-name, and device-code behavior.

## Per-File Purpose

- `Sarek_registry.ml` stores primitive type metadata, record metadata, variant metadata, and intrinsic/function metadata in process-global hash tables (`spoc/registry/Sarek_registry.ml:42-89`).
- Registration functions mutate those registries (`spoc/registry/Sarek_registry.ml:91-123`).
- Lookup helpers expose exact lookup, membership checks, device-code rendering, short-record-name lookup, field/constructor accessors, and CUDA/OpenCL code selection (`spoc/registry/Sarek_registry.ml:125-256`).
- Module initialization registers only fundamental `bool` and `unit` types (`spoc/registry/Sarek_registry.ml:235-247`).

## Features and APIs

- `register_type`, `register_record`, `register_variant`, and `register_fun` are intended for PPX/generated stdlib registration (`spoc/registry/Sarek_registry.ml:91-123`).
- `find_type`, `find_record`, `find_variant`, and `find_fun` return options for exact lookups (`spoc/registry/Sarek_registry.ml:125-136`).
- `is_type`, `is_record`, `is_variant`, and `is_fun` expose membership checks (`spoc/registry/Sarek_registry.ml:138-149`).
- `type_device_code` and `fun_device_code` fail if a name is unknown (`spoc/registry/Sarek_registry.ml:151-163`).
- `fun_device_template` invokes a function's device-code generator using a synthetic generic device (`spoc/registry/Sarek_registry.ml:165-197`).
- `find_record_by_short_name` supports resolving `Module.Type` by `Type` (`spoc/registry/Sarek_registry.ml:199-217`).
- `cuda_or_opencl` returns OpenCL code only when `dev.framework = "OpenCL"`; all other framework strings get CUDA-style code (`spoc/registry/Sarek_registry.ml:252-256`).

## Invariants

- Registered record names may be fully qualified and should remain unique across the process.
- Function registry keys include both `module_path` and `name`, so `Float32.sin` and unqualified `sin` are distinct (`spoc/registry/Sarek_registry.ml:87-89`, `113-123`).
- `fi_arity` should equal `List.length fi_arg_types`, though registration currently does not enforce it (`spoc/registry/Sarek_registry.ml:112-123`).
- `type_device_code` assumes the requested type was registered in the primitive type registry, not the record or variant registries (`spoc/registry/Sarek_registry.ml:151-155`).
- Only `bool` and `unit` are registered in this module; numeric primitives are expected to come from stdlib modules when linked (`spoc/registry/Sarek_registry.ml:235-247`, `258-260`).

## Potential Invariant Violations or Bugs

- All `register_*` functions silently overwrite existing entries via `Hashtbl.replace` (`spoc/registry/Sarek_registry.ml:91-123`). Duplicate PPX output or conflicting libraries can replace metadata without warning.
- `find_record_by_short_name` returns the first matching record discovered by hash-table fold (`spoc/registry/Sarek_registry.ml:202-217`). Multiple `ModuleA.point`/`ModuleB.point` registrations make lookup ambiguous and nondeterministic.
- `register_fun` does not validate `arity` against `arg_types` length (`spoc/registry/Sarek_registry.ml:112-123`).
- `fun_device_template` passes a fake `"generic"` device and assumes most intrinsics ignore device details (`spoc/registry/Sarek_registry.ml:165-197`). Device-sensitive intrinsics can produce misleading templates.
- `cuda_or_opencl` defaults unknown frameworks to CUDA syntax (`spoc/registry/Sarek_registry.ml:252-256`), which may be wrong for Vulkan/Metal/future backends.
- `record_fields` falls back to short-name lookup and raises `Failure` on miss (`spoc/registry/Sarek_registry.ml:219-227`); callers cannot distinguish absent vs ambiguous records.

## Performance and Maintainability Risks

- Registry state is global and lacks reset/snapshot APIs, so tests and plugin loading order can affect later behavior.
- Short-name lookup scans the entire record registry (`spoc/registry/Sarek_registry.ml:202-217`); acceptable for small registries but linear and ambiguous.
- Stringly typed names for types, fields, modules, and frameworks make accidental mismatches runtime-only errors.
- `failwith` errors are simple but unstructured; backend/codegen callers cannot classify registry failures without parsing messages.

## Related Tests

- `spoc/registry/test/test_sarek_registry.ml` verifies startup types, primitive registration/lookup, device-code callbacks, record registration/field access, short-name lookup, variant registration/constructor access, function registration with and without modules, template lookup, and `cuda_or_opencl`.

## Missing Tests

- Duplicate registration behavior for types, records, variants, and functions.
- Ambiguous short-name lookup.
- `register_fun` with mismatched `arity` and `arg_types`.
- Failure messages from `type_device_code`, `fun_device_code`, `record_fields`, and `variant_constructors`.
- Device-sensitive `fun_device_template` behavior.
- Unknown framework behavior in `cuda_or_opencl`.
- Interaction with stdlib numeric primitive registrations when those libraries are linked.

## Concrete Improvement Candidates

- Return `('a, registry_error) result` or raise structured errors instead of raw `Failure`.
- Add duplicate-detecting registration variants and reserve `replace` behavior for intentional overrides.
- Maintain a secondary short-name index that detects ambiguity.
- Validate `arity = List.length arg_types` in `register_fun`.
- Replace `cuda_or_opencl` with backend-specific codegen dispatch or return an error for unknown frameworks.
