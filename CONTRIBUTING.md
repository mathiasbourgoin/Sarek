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

#### Referring to issues, PRs and backlog items

GitHub gives issues and pull requests a *single shared counter* (it is past
330 in this repository) and auto-links every bare `#NN` in rendered Markdown
to whatever holds that number. Our internal task tracker has its own,
much lower numbering, so a bare `#NN` meaning a backlog item almost always
renders as a live link to an unrelated PR — `#63` means "Metal
`simdgroup_matrix`" to us and "Cleanup 2026 01" to GitHub.

Three forms, and nothing else:

| Referring to | Write | Renders as |
| --- | --- | --- |
| A real GitHub PR or issue | `PR #290`, `issue #135` | a link, to the right thing |
| An internal backlog item | `backlog-63` | plain text |
| A numbered row or item **inside the same document** | name it — "row 3, `ld.global.nc`" | plain text |

`backlog-NN` deliberately drops the `#`: GitHub will not link it, so it
cannot promise the reader a page that does not exist. **The backlog is not
public.** A `backlog-NN` reference is for people working in this repository;
it is not something an outside reader can look up, and it should not be
dressed up to look like it is.

Before writing `#NN`, check that the number resolves to the thing you are
describing (`gh issue view NN`). If it does not, you mean `backlog-NN`.

**Do not put a GitHub noun in front of a `backlog-NN`.** "Issue backlog-106"
or "PR backlog-64" re-introduces exactly the claim the form exists to drop:
the reader is told this is a GitHub issue, and it is not. Front-matter that
used to read `**Issue:** #107` should read `**Tracked as:** backlog-107`.
The nouns are fine where they govern a real number — `PR #275 / backlog-66`
is correct, because `PR` applies to `#275`.

**PR titles and commit subjects are out of scope.** Every existing PR here
uses a bare `(#NNN)` for a backlog item, and rewriting history to match
costs more than the ambiguity does — a PR title is read in a context where
the reader already knows the number is ours. For *new* titles, `(backlog-NN)`
is preferred; nothing existing needs changing.
