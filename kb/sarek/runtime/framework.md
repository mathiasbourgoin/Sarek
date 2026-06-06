# Framework Support

## Component Inventory

Reviewed framework files: `sarek/framework/README.md`, `Framework_registry.ml`, `Intrinsic_registry.ml`, `Framework_cache.ml`, `Framework_error.ml`, `dune`, and colocated tests under `sarek/framework/test/**`.

## Per-File Purpose

- `sarek/framework/README.md`: backend framework design, registration examples, priority model, and extension guidance.
- `sarek/framework/Framework_registry.ml`: global backend framework registry, backend lookup, priority selection, and capability queries.
- `sarek/framework/Intrinsic_registry.ml`: intrinsic registration and lookup by backend/language/name.
- `sarek/framework/Framework_cache.ml`: disk-backed artifact cache keyed by strings, plus hashing helper.
- `sarek/framework/Framework_error.ml`: framework-specific errors and formatting.
- `sarek/framework/test/dummy_backend.ml`: dummy backend used by integration tests.
- `sarek/framework/test/test_framework_registry.ml`: registry behavior tests.
- `sarek/framework/test/test_intrinsic_registry.ml`: intrinsic lookup tests.
- `sarek/framework/test/test_framework_cache.ml`: cache basics.
- `sarek/framework/test/test_framework_integration.ml`: dummy backend integration.

## Features/APIs

- Framework registration with backend kind, priority, capabilities, and kernel operations.
- Lookup by backend and automatic selection of the highest-priority available backend.
- Intrinsic registration per backend and language.
- Cache directory creation, `compute_key`, `get`, `put`, `exists`, `clear`, and `stats`-style helpers.
- Framework-specific error variants for missing backends and unsupported features.

## Invariants

- Backend names and priorities should produce deterministic selection.
- Re-registering an intrinsic should not create duplicate backend capability entries.
- Cache keys must resolve inside the cache directory.
- Cache writes should not publish partial artifacts.
- Registry mutation should be initialization-only or synchronized if used concurrently.

## Potential Invariant Violations/Bugs

- `Framework_registry` stores global mutable `Hashtbl`s at `sarek/framework/Framework_registry.ml:28-35` and mutates/reads them without locks at `sarek/framework/Framework_registry.ml:37-131`. Concurrent plugin initialization or tests can race.
- `Framework_cache.get` and `put` turn the caller-provided `key` directly into a path component at `sarek/framework/Framework_cache.ml:96-100` and `sarek/framework/Framework_cache.ml:124-131`. Although `compute_key` returns MD5 at `sarek/framework/Framework_cache.ml:89-94`, the API accepts arbitrary keys, so `../` traversal is possible if exposed to untrusted callers.
- Cache I/O is not atomic and not close-safe on exceptions: reads and writes open channels directly at `sarek/framework/Framework_cache.ml:102-113` and `sarek/framework/Framework_cache.ml:129-137`. Concurrent writers can expose partial files.
- `Intrinsic_registry.Global.register` appends backend names when an intrinsic is re-registered at `sarek/framework/Intrinsic_registry.ml:57-61`, and `backends_for` returns the accumulated list at `sarek/framework/Intrinsic_registry.ml:77-79`. Duplicate backend entries are possible.

## Performance Or Maintainability Risks

- Registry priority and backend availability are global mutable state, which makes tests sensitive to execution order unless reset helpers are used consistently.
- The framework README documents backend priorities that do not match current plugin priorities; see plugin notes for exact native/interpreter refs.
- Cache serialization is byte-string oriented and has no schema/version guard in the path, which can make stale artifacts hard to invalidate after compiler changes.
- Cache stats and cleanup can become slow if the cache directory grows large because they scan filesystem contents directly.

## Related Tests

- `sarek/framework/test/test_framework_registry.ml`: backend registration, lookup, and selection.
- `sarek/framework/test/test_intrinsic_registry.ml`: intrinsic registration and backend lookup.
- `sarek/framework/test/test_framework_cache.ml`: cache put/get/exists/clear basics.
- `sarek/framework/test/test_framework_integration.ml`: dummy backend integration through the framework facade.

## Missing Tests

- Concurrent backend registration and lookup.
- Re-registering the same intrinsic/backend and checking for duplicate backend entries.
- Cache path traversal attempts such as `../outside`.
- Concurrent `put` to the same key and reader behavior while a write is in progress.
- Cache channel cleanup when `input_binary_int`, `really_input_string`, or output raises.
- Compatibility test that documented backend priorities match registered plugin priorities.

## Concrete Improvement/Fix Candidates

- Add a mutex around registry mutation and read paths, or document/enforce single-threaded initialization.
- Restrict cache keys to a validated digest format, or make public `get`/`put` accept source bytes and compute the key internally.
- Write cache entries to a temporary file and `rename` atomically after `close_out`.
- Wrap channel operations in `Fun.protect` so descriptors close on exceptions.
- Deduplicate backend names in `Intrinsic_registry.Global.register`, preferably by replacing existing `(name, backend, language)` entries.
- Add a registry reset/test namespace if global state is intended to be mutable in tests.
