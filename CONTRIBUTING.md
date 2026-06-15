# Contributing to SPOC/Sarek

## Code Quality Guidelines

### OCaml Standards

Follow standard OCaml best practices:
- [OCaml Programming Guidelines](https://ocaml.org/docs/guidelines)
- [OCaml Style Guide](https://github.com/ocaml/ocaml/blob/trunk/HACKING.adoc)

### Project-Specific Rules

**Type Safety**
- Favor GADTs and phantom types over `Obj.t`
- Use structured types (records, variants) over tuples for complex data
- Leverage the type system for correctness guarantees

**Error Handling**
- No `failwith` or `invalid_arg` in library code
- Use structured error types with inline records
- Provide context in error messages

**Code Organization**
- Extract helper functions for clarity
- Keep functions focused and maintainable
- Use meaningful names

**Testing**
- Write tests for new features
- Test error paths
- Include e2e tests for backend changes

### Build and Test

```bash
# Build
dune build

# Run tests
dune runtest

# Run specific backend tests
_build/default/sarek-cuda/test/test_cuda_error.exe
```

### Documentation

- Document public APIs
- Keep README files factual and technical
- Avoid marketing language or performance claims without evidence
