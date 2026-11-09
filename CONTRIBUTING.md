# Contributing to Sarek

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

### CI workflows

These rules apply to every workflow in `.github/workflows/`.

- **Minimal permissions.** Each workflow declares a top-level `permissions:` block, read-only by default (`contents: read`). A job that really needs more (publishing a package, posting a comment, pushing to a deployment branch) raises it in its own job-level `permissions:` block, with only the scopes it uses.
- **Pinned actions.** Every third-party action is pinned to a full commit SHA, with the version in a trailing comment: `uses: owner/action@<40-hex-sha> # v1.2.3`. Resolve the SHA of the tag you want (dereferencing annotated tags to their commit) instead of copying a floating tag such as `@v4`.
- **Untrusted toolchains and tools.** A workflow that builds or executes third-party toolchains or tools (package managers, compilers, language runtimes, installers) triggers on `pull_request` only. It does not trigger on `push` to the default branch or on `workflow_dispatch`, and it never uses `pull_request_target`. The one exception is a job whose purpose is to publish after a merge (documentation deploy, container image push): it may run on `push` to the default branch, executes only reviewed code from that branch, and holds its write scope at job level.
- **No secrets in such workflows.** They run with no `secrets.*` and with the token restricted to `contents: read`.

### Documentation

- Document public APIs
- Keep README files factual and technical
- Avoid marketing language or performance claims without evidence

#### Documentation snippets

Every `ocaml` code block in the front-page `README.md`, `gh-pages/index.md`, `gh-pages/docs/*.md`, `gh-pages/examples/*.md`, `sarek/core/README.md`, `sarek/sarek/README.md` and `sarek/ppx/README.md` is compiled by `dune runtest`. To run only this check:

```bash
dune build @snippets
```

How it works (`sarek/tests/doc_snippets/`): `extract.exe` writes the blocks of one Markdown file, in order, as one OCaml module; a library compiles the modules with `sarek`, `sarek.core`, `sarek.stdlib`, `sarek.native`, `sarek.interpreter`, the five GPU plugins and `(pps sarek.ppx)`. Nothing is linked or run. Blocks of one page form a single program, so a later block may use names an earlier one defined. A compile error names the generated file (`snip_<page>.ml`); the comment above each block in that file gives the Markdown line.

- **Add a page**: add a `(rule (targets snip_<name>.ml) ...)` to `sarek/tests/doc_snippets/dune`, copying a neighbour. A page in `gh-pages/docs/`, `gh-pages/examples/` or the listed READMEs that has an `ocaml` block but no rule fails the check.
- **Skip a block** that cannot be a structure item (a signature fragment, pseudo-code, code that must not compile): put `<!-- snippet: skip -->` on the line directly above the opening fence. The comment is invisible in the rendered page. Say why in the review; prefer rewriting the block so it compiles.
- **Use another language** for a block that is not OCaml (`bash`, `lisp`, `cuda`): any fence other than `ocaml` is ignored, except a fence that looks like an attempt at one (`OCaml`, `ml`, `ocaml foo`, `~~~` fences, four backticks with an ocaml/ml info string), which fails the check with its file and line.
- **Build a page's program as an executable**: mark a block `<!-- snippet: NAME -->` and extract it with `extract.exe --only NAME`. `sarek/tests/doc_snippets/getting_started/` does this for the Getting Started program and runs it on the CPU backends.

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
