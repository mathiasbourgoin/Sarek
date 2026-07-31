# Sarek PPX Parser

## Component Inventory

Parser scope is primarily `sarek/ppx/Sarek_parse.ml` (the expression dispatcher and kernel/module-item parsing) and `sarek/ppx/Sarek_parse_helpers.ml` (extracted leaf helpers), with input/output shapes in `sarek/ppx/Sarek_ast.ml` and error reporting in `sarek/ppx/Sarek_error.ml`. `sarek/ppx/Sarek_ppx.ml` calls the parser from `expand_kernel` at `sarek/ppx/Sarek_ppx.ml:1430-1433`. The former monolithic `Sarek_parse.ml` (~839 lines) was reduced (pure move) by extracting leaf helpers into `Sarek_parse_helpers.ml` and decomposing `parse_expression`.

## Per-File Purpose

- `sarek/ppx/Sarek_ast.ml`: source AST for parsed kernels, including `type_expr`, `pattern`, `expr_desc`, `module_item`, and `kernel`.
- `sarek/ppx/Sarek_parse_helpers.ml`: extracted leaf helpers — `Parse_error_exn` (`sarek/ppx/Sarek_parse_helpers.ml:12`), `parse_type` (`:17`), pattern extractors (`extract_type_from_pattern`/`extract_name_from_pattern`/`extract_param_from_pattern`, `:66-83`), `parse_pattern` (`:104`), `parse_binop`/`parse_unop` (`:121`/`:145`), the AST-502 compatibility shims (`:162-188`), and `collect_fun_params` (`:211`).
- `sarek/ppx/Sarek_parse.ml`: maps `Parsetree` payloads into `Sarek_ast`. The `parse_expression` dispatcher spans `sarek/ppx/Sarek_parse.ml:115-331` — **~215-217 lines** (graph-verified 2026-07-02; the previous "~115 lines" KB figure was stale/wrong, not just outdated — it undercounted the function by roughly half) — and delegates five arm bodies to extracted helpers — `parse_assign_form` (`:332`), `parse_pragma_form` (`:343`), `parse_create_array_form` (`:365`), `parse_binop_or_app_form` (`:383`), and `parse_let_form` (`:399`); it also parses `let%shared`/`let%superstep` (`:36`/`:86`), kernel bodies, type declarations, and Sarek module items. Helper names are re-aliased from `Sarek_parse_helpers` at `sarek/ppx/Sarek_parse.ml:18-32`.
- `sarek/ppx/Sarek_error.ml`: carries parse failures as structured PPX errors.

## Features And APIs

- Parameter parsing requires `(name : type)` shapes in `sarek/ppx/Sarek_parse_helpers.ml:83-101` (`extract_param_from_pattern`).
- Core expression parser handles literals, variables, one-level qualified names, unary/binary ops, array/vector access, assignment, `if`, loops, `let`, local functions, record/variant construction, match, tuple, and extension nodes in `sarek/ppx/Sarek_parse.ml:115-331` (`parse_expression`, ~215-217 lines), delegating five arm bodies to `parse_assign_form`/`parse_pragma_form`/`parse_create_array_form`/`parse_binop_or_app_form`/`parse_let_form` (`sarek/ppx/Sarek_parse.ml:332-453`).
- `let%shared` is parsed in `sarek/ppx/Sarek_parse.ml:36-81` (`parse_let_shared`).
- `let%superstep` and optional `[@divergent]` binding attributes are parsed in `sarek/ppx/Sarek_parse.ml:86-112` (`parse_superstep`).
- Kernel payloads and module items are parsed in `sarek/ppx/Sarek_parse.ml:454-631` (`parse_kernel_function`, `parse_module_items_from_structure`).

## Invariants

- Kernel parameters must be typed; unannotated parameters fail in `sarek/ppx/Sarek_parse_helpers.ml:89-95` (`extract_param_from_pattern`).
- Multi-argument variant constructors are rejected by the parser in `sarek/ppx/Sarek_parse_helpers.ml:44-65` (`parse_variant_constructors`, rejection at `:51`).
- Only syntax explicitly recognized by `parse_expression` should enter typing; unsupported forms end at `sarek/ppx/Sarek_parse.ml:327`.
- Source locations should be preserved well enough for user-facing errors.

## Potential Invariant Violations Or Bugs

- **FIXED (backlog-191), was: `when` guards on match cases silently dropped.** `parse_expression`'s `Pexp_match` arm read `case.pc_lhs` and `case.pc_rhs` and never looked at `case.pc_guard`, so the guard was discarded in the PARSER — upstream of type-checking, of every lowering pass and of backend selection — and the arm became unconditional in the single AST all backends are generated from. Measured before the fix with a load-bearing guard (`| Circle r when r >. 10.0 -> r *. 2.0 | Circle r -> r +. 100.0`): the kernel returned the dropped-guard answer on all 9 devices present on the dev host — Interpreter (sequential + parallel), Native, CUDA/PTX ×2 (ZLUDA on AMD), OpenCL ×2 (radeonsi), Vulkan ×2 (RADV) — 10/64 elements wrong against the source semantics and 0/64 wrong against the dropped-guard oracle, identically on every one.

  **How silent it was, precisely.** Not silent for *this* kernel: a wrong-**answer** repro needs two arms on the same constructor, one guarded and one not, because only then is the guard what chooses between them — and dropping it leaves two syntactically identical arms, which is OCaml's warning 11 `redundant-case`. Under the e2e suites' flags (`(:standard -w -32-33-34-69)`, which do not disable 11) that warning is a hard error. So this shape did emit a diagnostic, pointing at a redundant case rather than at a discarded guard. The genuinely silent shapes are those where no two arms share a constructor: the guard vanishes, nothing looks redundant, the arm becomes unconditional, and there is no warning at all. Warning 11 is therefore not a usable backstop for this bug — which is why the refusal is the fix.

  **Where the error surfaces.** `parse_expression` now raises `Parse_error_exn` located on the guard expression. On the **`[%kernel]` route** that becomes a located `Location.raise_errorf` at PPX time — the conversion is at `sarek/ppx/Sarek_ppx.ml:1999`, inside the `[%kernel]` expander's handler. It is the **only** such conversion in the file, so the **`[@sarek.module]` route behaves differently and worse**: `process_structure_for_module_items` (`Sarek_ppx.ml:2289`) has no `Parse_error_exn` handler. Measured — an in-file guarded `let[@sarek.module]` is swallowed by the self-scan's catch-all (`Sarek_ppx.ml:571-581`), which prints `Sarek PPX: scanning … failed (Parse_error_exn(...)); skipping this file's Sarek registrations` and leaves the user with `Error: Unbound variable: <helper>`; a guarded helper in a sibling file escapes as an unlocated `Fatal error: exception Sarek_parse_helpers.Parse_error_exn(...)`. This is **pre-existing and not specific to guards** (reproduced with a lambda, no guard involved) and **fail-closed in every case** — compilation fails on all three paths, none of them yields a silently wrong kernel — so it is a diagnostic-quality defect, not a correctness one. Tracked as **issue #391**; deliberately not fixed alongside the guard refusal.

  This is a **refusal, not an implementation**: `Sarek_ir_ppx`'s `EMatch`/`SMatch` carry `(pattern * _) list` with no guard slot — and there are two IRs to add one to (`Sarek_ir_ppx` for the PPX, `Sarek_ir_types` in `spoc/ir` for the code generators, bridged by `Sarek_ir_conv`) plus a third like-named `EMatch` on the surface `Sarek_ast`. `git grep -l "EMatch\|SMatch"` reports 37 non-test source files; the blast radius includes at least the six device emitters (`Sarek_ir_cuda`, `Sarek_ir_opencl`, `Sarek_ir_metal`, `Sarek_ir_ptx` with its `_expr`/`_stmt` halves, `Sarek_ir_glsl`, `Sarek_ir_wgsl`) and roughly a dozen IR passes, and is not a closed list — re-run the grep rather than trusting an enumeration. Beyond the IR: every C-family emitter builds a match as a nested tag ternary whose LAST arm is emitted unconditionally (and PTX branches to the last arm unconditionally), so a guarded arm has no way to fall through; and `Sarek_ir_ptx_expr.check_match_exhaustive` reasons on constructor coverage alone, which a guarded arm satisfies syntactically but not semantically — once guards exist a match can fail at run time and the device backends have no trap to fail into (the interpreter's `Pattern_match_failure` has no GPU counterpart). Hence the message says "not yet implemented", not "cannot be supported". Negative-test coverage: `sarek/tests/negative/test_when_guard.ml`, wired into `make test_negative`; it covers the `[%kernel]` route only (see #391), and the case is deliberately self-contained (no `open Spoc`/`open Kirc`) so its red is caused by the refusal alone.
- **FIXED (backlog-192), was: unsupported OCaml core types became `TEConstr ("unknown", [])`** in `parse_type`'s fallback arm instead of failing at parse time. Worse than "moves diagnostics later": `Sarek_types.type_of_type_expr` maps an unrecognised type constructor to `TRecord (name, [])`, so the annotation became an EMPTY RECORD type named `unknown` — a phantom type, which is why it did not reliably become a diagnostic at all. `parse_type` now refuses, naming the core_type shape (`Sarek_unsupported.core_type_refusal`).
- Confirmed limitation: qualified IDENTIFIERS (`Pexp_ident`) support one module segment via `Ldot (Lident modname, name)`; deeper paths are refused, and since backlog-192 the refusal names the limit instead of saying "Unsupported expression". `let open`'s module path is a DIFFERENT site and is no longer limited: it used to flatten to the empty path beyond depth two, which `Sarek_native_gen_expr` turns into `failwith "empty module path in TEOpen"` — measured as `Sarek internal error: Failure("empty module path in TEOpen")` for a kernel containing `let open A.B.C in`. It now flattens at any depth (`sarek/tests/unit/test_parse.ml`).
- Confirmed limitation: standalone lambda expressions are rejected with a parse error directing the user to let-bound functions at `sarek/ppx/Sarek_parse.ml:592-597` (`parse_expression`'s `_ when is_function_expression expr` arm — a GUARDED wildcard, not a constructor match, which is why `Sarek_unsupported`'s `Pexp_function` arm records its unreachability as resting on that guard). (Prior KB text describing nested `ELet` construction was already stale before this refactor; the reject-and-redirect behavior is the moved-verbatim baseline.)
- **REMOVED (backlog-192), was: `is_array_access`**, a helper whose body was the literal `false` behind a comment calling itself "a simplified check", guarding an arm that therefore never fired while advertising an `a.(i)` route that did not exist. `a.(i)` needs no arm: OCaml desugars it to `Array.get a i`, which the two `Array.get` arms match. Supported access forms are `Array.get`/`Array.set` and `.%[]`/`.%[]<-`.
- Confirmed limitation: mutable assignment only handles simple variable left-hand sides for `:=` in `sarek/ppx/Sarek_parse.ml:332-341` (`parse_assign_form`, error at `:340`).

## Performance Or Maintainability Risks

- Parser support is spread across many `Parsetree` shapes and has several syntax-specific paths for arrays, records, and module identifiers.
- Silent `unknown` type nodes make parser behavior harder to reason about because later failures may look like type errors rather than syntax support gaps.
- Location conversion uses partial position reconstruction in `sarek/ppx/Sarek_ast.ml:31-58`; this may reduce diagnostic precision.

## Related Tests

- `sarek/tests/unit/test_parse.ml:501-562` runs parser tests for operators, primitive/vector/arrow/tuple/var types, kernels, and basic expressions.
- E2E tests exercise PPX parsing through all executables in `sarek/tests/e2e/dune:56-93`.

## Missing Tests

- Direct lambda expressions inside kernels, either accepted with correct behavior or rejected clearly.
- Generic OCaml array syntax cases not routed through the explicit access forms.
- `:=` assignment with non-variable lvalues, if intended to be rejected.

## Concrete Improvement/Fix Candidates

- Add a recursive longident-to-name helper shared by variable, constructor, and intrinsic parsing.
- Either remove lambda parsing or represent lambdas explicitly and type them.

## The accepted subset is a refusal boundary (backlog-192)

`Sarek_parse` used to have three answers for a construct the kernel subset does
not implement: implement it, refuse it, or DROP it and carry on. The third is the
one that produces silently wrong device code — backlog-191's `when` guard was one
instance of it, not the only one. backlog-192 is the systematic sweep: every arm
and every record field the parser can encounter, walked from the ppxlib Parsetree
type definitions, classified, with every drop turned into a located refusal.

Three mechanisms hold the boundary now, and only the first is a matter of style:

1. `Sarek_unsupported.ml` holds one refusal table per Parsetree variant, and
   **none of them has a wildcard arm**, so the OCaml compiler is what notices
   when ppxlib grows a constructor: the next Parsetree arm stops the build rather
   than reaching a user as silence. What makes it stop is `-warn-error +8` in
   `sarek/ppx/dune`, on `sarek_frontend` only. Measured 2026-07-31 by deleting
   one arm: dune's dev `:standard` already errors on warning 8, but
   `--profile=release` only warned and built at exit 0 until that flag was added.
   `-w +8`, which an earlier revision of this section credited, changes nothing
   in either profile.
2. Each refusal that a source file can reach has a negative-compile case in
   `sarek/tests/negative` asserting on the compiler's real stderr, wired into
   `make test_negative` (and `scripts/check-negative-case-coverage.sh` refuses a
   declared case nothing asserts).
3. The two refusals the OCaml parser cannot reach are asserted on AST nodes in
   `sarek/tests/unit/test_parse.ml`, each with the check that establishes the
   unreachability written beside it.

### Scope of the enumeration, stated exactly

Walked: every arm of `expression_desc`, `pattern_desc`, `core_type_desc` and
`structure_item_desc`, and every FIELD of `case`, `value_binding`,
`value_constraint`, `type_declaration`, `type_kind`, `constructor_declaration`,
`constructor_arguments`, `label_declaration`, `open_infos`, `module_expr_desc`,
`function_param_desc`, `function_body`, `type_constraint`, `arg_label`,
`constant` and `longident`, as declared in **ppxlib 0.38.0**'s
`lib/ppxlib/ast/ast.ml` — the selected AST this PPX compiles against, not a
list of constructs anybody thought of. (An earlier revision of this line said
"ppxlib 5.2.1". There is no such ppxlib release; 5.2 is the OCaml version whose
Parsetree shape that file carries, and the two were conflated. Re-check with
`opam list --installed ppxlib` in `/home/mathias/dev/SPOC/_opam` and by diffing
its constructor set against the switch's own `lib/ppxlib/astlib/ast_502.ml`,
which is equal.)

**This walk raised the ppxlib floor, and the bound was raised to match.**
**Declared ppxlib floor: `0.37.0`.** It used to say `(ppxlib (>= 0.22.0))`,
which was false of the code: the tables match `Ptyp_open` and the parameterised
`Pexp_function`, both of which need the OCaml-5.2-shaped AST. Measured on the
`octez-setup` switch (ppxlib 0.35.0, OCaml 5.3.0, dune 3.23.0): `987b0c30`
builds `sarek_frontend` at exit 0 and this branch fails at exit 1 with
`no constructor "Ptyp_open"` and an unmatched `Pexp_fun (_, _, _, _)`.

The floor was then bisected over the versions that actually exist, in a
throwaway local switch on OCaml 5.4.0 with dune 3.24.1, destroyed afterwards.
`opam show ppxlib --field all-versions` gives `… 0.35.0 0.36.0 0.36.2 0.37.0
0.38.0~5.5preview 0.38.0`, so the candidate set between the known-bad and the
known-good is four releases, not a range of numbers:

| ppxlib | `opam install` | `dune build sarek/ppx/sarek_frontend.cma` | outcome |
|---|---|---|---|
| 0.36.0 | exit 20, `Package conflict!` | not reached | **install-failure** — the package itself caps `"ocaml" {>= "4.08.0" & < "5.4.0"}`, and this project requires `ocaml >= 5.4.0` |
| 0.36.2 | exit 20, `Package conflict!` | not reached | **install-failure**, same cap |
| 0.37.0 | exit 0 | **exit 0** | **BUILDS** — the floor |
| 0.38.0 | exit 0 | exit 0 | builds (the switch this repo develops in) |

`0.38.0~5.5preview` is a prerelease and was not tried; opam will not select it
without an explicit request.

**Two different floors, and only one of them is expressible.** The floor the
CODE needs is `0.36.0`: that release already carries `Ptyp_open` and had already
replaced the four-argument `Pexp_fun` with the parameterised `Pexp_function`.
Established by source inspection rather than by a build, because 0.36.0 cannot
be installed on the OCaml this package requires —

```
opam source ppxlib.0.35.0 --dir=p35 ; grep -c Ptyp_open p35/ast/ast.ml   -> 0
opam source ppxlib.0.36.0 --dir=p36 ; grep -c Ptyp_open p36/ast/ast.ml   -> 14
grep -n '| Pexp_fun of' p35/ast/ast.ml  -> 373:  | Pexp_fun of arg_label * …
grep -n '| Pexp_fun of' p36/ast/ast.ml  -> (no match)
```

The floor that can be BUILT, and therefore the one declared, is `0.37.0`. A
declared `0.36.0` would be a bound no run in this repository can support, which
is the same class of unbacked claim as the `0.22.0` it replaced.

**What keeps this honest from now on: `scripts/check-ppxlib-floor.sh`.** It
refuses a `dune-project` and a `sarek.opam` that disagree (the tracked `.opam`
is what opam clients read, and it can go stale behind `dune-project`), refuses a
floor recorded here that differs from the declared one, and — the rule that ties
the number to the code — carries a table of Parsetree constructors with the
first ppxlib release providing each, and refuses a declared floor below the
introducing release of any constructor the PPX matches. It also exits 2 rather
than 0 if none of the tabled constructors appears at all, so a rename cannot
turn rule 3 into a permanent vacuous pass.

**A CI lane on the minimum ppxlib was considered and not built.** It is the only
thing that would catch a BRAND-NEW constructor raising the floor without anybody
adding a table row, but it costs a full dependency build (ctypes, js_of_ocaml,
the runtime) per run, and the return is currently near zero: opam's solver
already makes the floor unreachable from below, because this package requires
`ocaml >= 5.4.0` and every ppxlib before 0.37.0 caps itself at `ocaml < 5.4.0`.
If the OCaml floor is ever lowered, that reasoning expires and the lane becomes
worth its cost.

Reached from: `sarek/ppx/Sarek_parse.ml` and its callee
`sarek/ppx/Sarek_parse_helpers.ml`.

**NOT walked: `sarek/ppx/Sarek_ppx.ml`.** It is the caller, not a callee: it
does its own `Parsetree` matching for the attribute/extension surface
(`[@@sarek.type]`, `[%%sarek_include]`, `[@sarek.module]`, the self-scan) before
handing a payload to the parser. Class-comparable drops may exist there and this
PR did not look. Same for `Sarek_typer`, the lowering passes and the emitters.

Class-comparable arms the OCaml *parser* cannot produce are marked
**unreachable**. TWO of them state a re-runnable check — the `Lapply` field path
and the `Lapply` let-open, both settled with `ocamlc -stop-after parsing` quoted
at their definition in `Sarek_parse.ml`, and both asserted on AST nodes in
`test_parse.ml`. The other fourteen, all in `Sarek_unsupported.ml`, state a
structural ARGUMENT instead ("parsed by an arm of its own that matches the
constructor totally") and no check: nothing machine-verifies that those arms are
total, so if a future guard narrows one, the arm becomes reachable and its text
lies to the user. An earlier revision of this line claimed every unreachable row
said how it was checked; that is true of two rows out of sixteen.

### `expression_desc` — `Sarek_parse.parse_expression`

| construct | site | before | after |
|---|---|---|---|
| `Pexp_ident` `x` / `M.x` | Sarek_parse.ml `parse_expression` | implemented | implemented |
| `Pexp_ident` deeper than `M.x`, or `Lapply` | final arm | refused, "Unsupported expression" | refused, names the depth limit and the alias workaround |
| `Pexp_constant` int / `1l` / `1L` / `1.0` / `1.0G` | `parse_expression` | implemented | implemented |
| `Pexp_constant` char, string, other int suffix | final arm | refused, "Unsupported expression" | refused, lists the literal forms a kernel accepts |
| `Pexp_let` non-rec single binding | `parse_let_form` | implemented | implemented |
| `Pexp_let` recursive, or `let a = … and b = …` | final arm | refused, "Unsupported expression" | refused, names both and says to nest |
| `Pexp_function` | `is_function_expression` arm | refused (let-bound advice) | unchanged |
| `Pexp_apply` — callee and positional args | `parse_expression` | implemented | implemented |
| `Pexp_apply` — `arg_label` on each argument | generic apply arm | **SILENTLY DROPPED**: `f ~b:2 ~a:1` lowered as positional `f 2 1` | **refused**, `test_labelled_arg` |
| `Pexp_match` scrutinee + `pc_lhs`/`pc_rhs` | `parse_expression` | implemented | implemented |
| `case.pc_guard` | `parse_expression` | refused (backlog-191) | unchanged |
| `Pexp_try` | final arm | refused, generic | refused, names it, `test_expr_table_try` |
| `Pexp_tuple` | `parse_expression` | implemented | implemented |
| `Pexp_construct` `()`/`true`/`false`/`C e` | `parse_expression` | implemented | implemented |
| `Pexp_construct` qualified `M.C e` | final arm | refused, generic | refused, names it |
| `Pexp_variant` `` `Tag `` | final arm | refused, generic | refused, names it |
| `Pexp_record` field list | `parse_expression` | implemented | implemented |
| `Pexp_record` **`with` base** | `parse_expression` | **DROPPED**: `{r with x=e}` parsed as `{x=e}`, base gone. NOT silent, measured: OCaml rejects the re-emitted native literal with `Some record fields are undefined: y` on the user's `with` line, never mentioning the `with`. A bad diagnostic, not wrong device code | **refused**, `test_record_update` |
| `Pexp_record` field longident `Lapply` | `parse_expression` | **DROPPED** to the invented field name `"field"` | **refused** — **unreachable** from source (`r.F(X).f` is a syntax error; `ocamlc -stop-after parsing` on `let f r = r.F(X).lbl` reports "Syntax error"), so asserted on the AST node in `test_parse.ml` |
| `Pexp_field` `r.f` | `parse_expression` | implemented | implemented |
| `Pexp_field` qualified `r.M.f` | final arm | refused, generic | refused, names it |
| `Pexp_setfield` `r.f <- e` | `parse_expression` | implemented | implemented |
| `Pexp_setfield` qualified | final arm | refused, generic | refused, names it |
| `Pexp_array` | final arm | refused, generic | refused, names `create_array` as the route |
| `Pexp_ifthenelse` | `parse_expression` | implemented | implemented |
| `Pexp_sequence` | `parse_expression` | implemented | implemented |
| `Pexp_while` | `parse_expression` | implemented | implemented |
| `Pexp_for` with a variable binder | `parse_expression` | implemented | implemented |
| `Pexp_for` with `_` or any other pattern | final arm | refused, generic | refused, says a `for` binder must be a variable |
| `Pexp_constraint` | `parse_expression` | implemented | implemented |
| `Pexp_coerce` | final arm | refused, generic | refused, names it |
| `Pexp_send`, `Pexp_new`, `Pexp_setinstvar`, `Pexp_override`, `Pexp_object` | final arm | refused, generic | refused, each named |
| `Pexp_letmodule` (inside a body) | final arm | refused, generic | refused, says it is read only at the payload top |
| `Pexp_letexception`, `Pexp_assert`, `Pexp_lazy`, `Pexp_poly`, `Pexp_newtype`, `Pexp_pack`, `Pexp_letop`, `Pexp_unreachable` | final arm | refused, generic | refused, each named with its reason |
| `Pexp_open` `M` / `M.N` | `parse_expression` | implemented | implemented |
| `Pexp_open` **deeper than `M.N`** | `parse_expression` | **SILENTLY DROPPED** to the EMPTY path — which `Sarek_native_gen_expr` turns into `failwith "empty module path in TEOpen"` and `Sarek_env.open_module` treats as a no-op. Measured: `Sarek internal error: Failure("empty module path in TEOpen")` | **implemented**: the path is preserved at any depth by the parser and re-emitted in full by the native backend (`test_parse.ml`). Intrinsic RESOLUTION is unchanged and keys on the last segment only (`Sarek_env.short_module_name`), exactly as it did at depth 2 — no e2e case exercises a 3-deep open end to end |
| `Pexp_open` with a `Lapply` module path | `parse_expression` | dropped to the empty path | **refused** — **unreachable** (`let open F(X) in` parses to `Pmod_apply`, not `Pmod_ident` + `Lapply`; checked with `-stop-after parsing`), asserted on the AST node in `test_parse.ml` |
| `Pexp_open` with a non-`Pmod_ident` module expr | final arm | refused, generic | refused, names functor application and `struct` |
| `open_infos.popen_override` (`open!`) | `parse_expression` | not read | **residual, documented**: `open!` differs from `open` only in whether OCaml warns about shadowing, and that is settled before the kernel is parsed |
| `open_infos.popen_attributes` | `parse_expression` (`Sarek_parse.ml:576`, swallowed by the record's `_`) | not read | **residual, DROPPED and documented** — the same class as `pvb_attributes` on an ordinary kernel-body `let`, and it was missing from this table entirely while the scope line above claimed `open_infos` had been walked field by field. An attribute on a kernel-body `let open` is still discarded in silence. Found by the adversarial review of this PR |
| `Pexp_extension` `global` / `native` / `shared` / `superstep` | `parse_expression` | implemented | implemented |
| `Pexp_extension` any other | final arm | refused, generic | refused, **naming the extension** and listing the four |

### `pattern_desc` — `Sarek_parse_helpers.parse_pattern` and the binder extractors

| construct | site | before | after |
|---|---|---|---|
| `Ppat_any`, `Ppat_var`, `Ppat_tuple`, `Ppat_construct` (unqualified) | `parse_pattern` | implemented | implemented |
| `Ppat_construct` existential binders (`C (type a) p`) | `parse_pattern` | **SILENTLY DROPPED**: parsed as the plain `C p` | **refused**, `test_existential_pattern` |
| `Ppat_construct` qualified `M.C` | final arm | refused, "Unsupported pattern" | refused, names it |
| `Ppat_alias` in a `match` pattern | final arm | refused, "Unsupported pattern" | refused, names it |
| `Ppat_alias` in a **binder** (`let (p as x) = e`, `fun (p as x) ->`) | `extract_name_from_pattern` | **SILENTLY DROPPED**: answered the alias name, discarded the inner pattern, so every name in `p` was absent and surfaced as an unbound variable at the USE | **refused**, `test_alias_binder` |
| `Ppat_constant`, `Ppat_interval`, `Ppat_variant`, `Ppat_record`, `Ppat_array`, `Ppat_or`, `Ppat_type`, `Ppat_lazy`, `Ppat_unpack`, `Ppat_exception`, `Ppat_extension`, `Ppat_open` | final arm | refused, "Unsupported pattern" | refused, each named with its reason (`test_pattern_table_or`) |
| `Ppat_constraint` in a pattern | `parse_pattern` | annotation looked through | **residual, documented**: a pattern's type is fixed by the scrutinee and by the constructor declaration, so the annotation cannot change what is lowered — and looking through it is what makes the documented `let ((a, b) : t) = e` spelling work |
| `Ppat_alias` arm of `extract_type_from_pattern` | helpers | live | **unreachable**: all three call sites call `extract_name_from_pattern` first, which now refuses an alias |

### `value_binding` / `value_constraint` / `function_body` — the three annotation spellings

| construct | site | before | after |
|---|---|---|---|
| `let (x : t) = e` (annotation in the pattern) | `extract_type_from_pattern` | implemented | implemented |
| `let x : t = e` (`pvb_constraint`, the spelling this tree uses) | `parse_let_form`, both payload folds | **SILENTLY DROPPED** — nothing read `pvb_constraint`. A kernel-local `let sum : float = …` was typed by inference with the declared width ignored; an annotated module constant hit "must have type annotations", a message false of the code in front of it | **implemented** (`binding_type`), `test_local_annotation_read` |
| `let f x : t = e` (`Pexp_function`'s `type_constraint`) | `collect_fun_params` | **SILENTLY DROPPED** — every kernel helper's declared result type discarded | **implemented** (`fun_return_type` → `ELetRec`'s result slot), `test_helper_return_read` |
| the same constraint on a NESTED `fun` (`let f x = fun y : t -> e`) | `fun_return_type` | **SILENTLY DROPPED**, and the first version of this branch's fix kept dropping it while asserting the drop was correct. `collect_fun_params` flattens the inner `fun` into the binding's parameter list, so after flattening there is no inner function for the annotation to belong to. Measured: the nested spelling compiled at exit 0, the flattened spelling of the same function failed to unify | **implemented** — every `Pexp_function` on the descent is inspected, the last constraint wins, peeled by the parameters collected after it |
| same, on a payload module item (`MFun` has no type field) | both payload folds | **SILENTLY DROPPED** | **implemented** as an `ETyped` constraint on the body, `test_module_helper_return_read` — but for an annotation containing a type VARIABLE the constraint is weaker than the source says: `Sarek_typer`'s `ETyped` arm converts in a FRESH type-variable context while `MFun`'s parameters use the item's, so the two `'a`s in `let f (x : 'a) : 'a = …` are different variables. The `ELetRec` route does not share this. Not refused (`test_module_poly.ml` writes exactly that shape); recorded, cross-runtime review |
| whole-binding annotation on a FUNCTION (`let (f : a -> b) = fun x -> …`) | `binding_result_type` | put in the RESULT slot **unpeeled**, unifying an arrow against the body's type | **refused**. This branch's first draft peeled one arrow per parameter instead, which silently discarded every DOMAIN the user wrote — `let (f : float32 -> int32) = fun (x : int32) -> x` accepted as `int32 -> int32`. The cross-runtime review caught it. Refusing costs nothing: kernel parameters must carry their own annotations anyway. `test_whole_binding_annotation` |
| a result annotation with fewer arrows than the parameters collected after it | `fun_return_type` | n/a (the whole slot was unread) | **refused**, `test_return_annotation_arity` |
| a binding annotated in BOTH places (`let (x : t1) : t2 = e`) | `binding_type` | the pattern one won, the other was discarded unchecked | **refused**, `test_double_annotation` |
| TWO result annotations on one binding (outer and inner `fun`) | `fun_return_type` | n/a | **refused** — the first draft took the inner one silently, discarding the relationship the outer one states. `test_two_result_annotations` |
| annotation on a tuple-pattern `let` | `parse_let_form` | dropped (`EMatch` has no type slot) | **refused**, `test_tuple_let_annotation` |
| `Pvc_coercion` (`let x :> t = e`) | `binding_type` | dropped | **refused**, `test_binding_coercion` |
| `Pvc_constraint` with `locally_abstract_univars` | `binding_type` | dropped | **refused** (no committed case: see "what is not covered") |
| `Pcoerce` in return position (`let f x :> t = e`) | `fun_return_type` | dropped | **refused**, `test_return_coercion` |
| return annotation on the KERNEL function | `parse_kernel_function` | dropped (`Sarek_ast.kernel` has no return type) | **refused**, `test_kernel_return_type` |
| `pvb_attributes` on a `let%superstep` binding: `divergent` | `parse_superstep` | implemented | implemented |
| `pvb_attributes` on a `let%superstep` binding: anything else | `parse_superstep` | **SILENTLY DROPPED**: `[@divergnt]` read as "not divergent", and the convergence checker was then applied to a step the user had declared divergent | **refused**, `test_superstep_attribute` |
| `pvb_attributes` on an ordinary kernel-body `let` | `parse_let_form` | not read | **residual, DROPPED and documented** — a stray attribute on a kernel-body binding is still discarded in silence. Closing it is the same three lines as the superstep one; it is listed as follow-up rather than done, because it needs its own case and this PR is already large |
| `pvb_loc` | — | used only for locations | n/a |

### `core_type_desc` — `Sarek_parse_helpers.parse_type`

| construct | site | before | after |
|---|---|---|---|
| `Ptyp_constr`, `Ptyp_var`, `Ptyp_tuple`, `Ptyp_arrow` (unlabelled) | `parse_type` | implemented | implemented |
| `Ptyp_arrow` **labelled / optional** | `parse_type` | **SILENTLY DROPPED**: `x:t -> u` and `t -> u` parsed to the same `TEArrow` | **refused**, `test_labelled_arrow_type` |
| `Ptyp_constr` path containing `Lapply` (`F(X).t`) | `parse_type`'s `flatten` | **SILENTLY DROPPED** to the name `""` | **refused**, `test_functor_type_path` |
| `Ptyp_any`, `Ptyp_object`, `Ptyp_class`, `Ptyp_alias`, `Ptyp_variant`, `Ptyp_package`, `Ptyp_open`, `Ptyp_extension` | `parse_type`'s last arm | **SILENTLY DROPPED** to `TEConstr ("unknown", [])`, which `Sarek_types.type_of_type_expr` maps to `TRecord ("unknown", [])` — an empty record type, i.e. a phantom type rather than an error | **refused**, each named (`test_wildcard_type`) |
| `Ptyp_poly` quantifier list | `parse_type` | not read | **residual, documented**: it binds the variables `Ptyp_var` already carries by name, so nothing about the type is lost |

### `type_declaration` and friends — `Sarek_parse.parse_payload`

A payload type declaration is consumed by the PPX and never reaches OCaml, so a
field nobody reads here is a field nobody reads at all.

| construct | site | before | after |
|---|---|---|---|
| `ptype_name`, `Ptype_record`, `Ptype_variant` | `parse_payload` | implemented | implemented |
| `ptype_params` (`type 'a t`) | `check_payload_type_decl` | **DROPPED**. Not silent for the measured shape (a field mentioning the parameter): OCaml rejects the re-emitted declaration with `A type wildcard _ is not allowed in this type declaration`, pinned to the whole payload and naming nothing the user wrote. The parameter-unused shape was not measured | **refused**, `test_payload_type_params` |
| `ptype_cstrs` (`constraint 'a = t`) | same | **SILENTLY DROPPED** | **refused** (no committed case: see below) |
| `ptype_private` | same | **SILENTLY DROPPED** | **refused** (no committed case: see below) |
| `ptype_manifest` with `Ptype_abstract` (`type t = u`) | same | refused, "Unsupported type declaration in kernel payload" | refused, names the alias and says aliases are not followed, `test_payload_type_alias` |
| `ptype_manifest` with a representation (`type t = u = {…}`) | same | **SILENTLY DROPPED** | **refused** (no committed case) |
| `Ptype_abstract` with no manifest, `Ptype_open` | same | refused, generic | refused, each named |
| `ptype_attributes` | same | **SILENTLY DROPPED** | **refused**, except `sarek.type` / `sarek.type_private`, which are accepted and ignored because a payload type is registered unconditionally (no committed case) |
| `Pstr_type`'s `rec_flag` | `parse_payload` | **DROPPED** | **refused** for `nonrec`. This branch first documented the drop as harmless, on the false ground that a kernel type cannot refer to another kernel type in its fields — nested records are supported and exercised (`sarek/tests/e2e/test_nested_types.ml`), so `nonrec` is meaningful and was being ignored. Cross-runtime review. `test_payload_type_nonrec` |
| `constructor_declaration.pcd_res` / `pcd_vars` (GADT) | `parse_variant_constructors` | **SILENTLY DROPPED**: `C : int -> t` recorded as `C of int` | **refused**, `test_payload_gadt` |
| `Pcstr_tuple` with 2+ args, `Pcstr_record` | same | refused at `Location.none` — no caret | refused **at `pcd_loc`** |
| `constructor_declaration.pcd_attributes`, `label_declaration.pld_attributes` | `parse_variant_constructors`, `parse_record_fields` | not read | **residual, documented**: these two helpers are shared with the TOP-LEVEL `[@@sarek.type]` route, where the same declaration also reaches OCaml and the attributes are its business. Refusing here would break code that compiles today. For a payload-local type they are genuinely dropped |
| `label_declaration.pld_name` / `pld_type` / `pld_mutable` | `parse_record_fields` | implemented | implemented |

### `structure_item_desc` inside a payload `let module M = struct … end`

| construct | site | before | after |
|---|---|---|---|
| `Pstr_type`, `Pstr_value` | `parse_module_items_from_structure` | implemented | implemented |
| `Pstr_value` with no params and no annotation | same | **DROPPED** — the binding was not registered. That the fold discarded it is mechanical (it returned its accumulator); the CONSEQUENCE was not isolated by measurement, because with the refusal removed this case fails with `Unbound module M`, which the probing showed is a pre-existing limitation of the payload `let module` route (bare-name registration) and not the drop | **refused**, `test_module_const_no_type` |
| `Pstr_eval`, `Pstr_primitive`, `Pstr_typext`, `Pstr_exception`, `Pstr_module`, `Pstr_recmodule`, `Pstr_modtype`, `Pstr_open`, `Pstr_class`, `Pstr_class_type`, `Pstr_include`, `Pstr_extension` | same, last arm | **DROPPED** — the fold returned its accumulator unchanged, which is mechanical. Same caveat as the row above: the residual error with the refusal removed (`Unbound module M`) is the payload `let module` route's own limitation, not a measurement of the drop's consequence | **refused**, each named (`test_module_item_dropped`) |
| `Pstr_attribute` named `ocaml.doc` / `ocaml.text` / `doc` / `text` | same | dropped (harmlessly — a doc paragraph carries no semantics for the fold to lose) | **accepted and skipped**, `test_parse_payload_floating_doc_accepted` in `sarek/tests/unit/test_parse.ml`. A standalone `(** … *)` between two items of a payload module IS a `Pstr_attribute`; refusing it rejected documentation. Same predicate and same allow-list as `check_payload_type_decl`'s `ptype_attributes`, which CodeRabbit caught one site earlier on this branch |
| `Pstr_attribute` with any other name (`[@@@warning "-32"]`) | same, last arm | **DROPPED** | **refused** (no committed negative case) |

### `module_expr_desc` and the payload top — `Sarek_parse.collect_mods`

| construct | site | before | after |
|---|---|---|---|
| `Pmod_structure` | `collect_mods` | implemented | implemented |
| `Pmod_ident`, `Pmod_apply`, `Pmod_apply_unit`, `Pmod_functor`, `Pmod_constraint`, `Pmod_unpack`, `Pmod_extension` | `collect_mods` | **SILENTLY DROPPED** — contributed `([], [])` | **refused**, `test_module_not_structure`. NOTE: unlike the four variants above, `module_expr_desc` has NO table in `Sarek_unsupported.ml`. These arms are covered by a real wildcard (`Sarek_parse.ml:1082`, `| _ -> raise ... "only a literal structure is supported here"`). That is behaviourally safe — the default is a refusal, not a drop — but the "the compiler is what notices when ppxlib grows a constructor" guarantee does NOT extend here: a new `Pmod_*` would be absorbed into the generic message with no build failure. Found by the adversarial review of this PR |
| `Pexp_letmodule` module NAME | `collect_mods` | not read (`tdecl_module = None`) | **residual, documented and deliberate**: `None` is what makes a payload-local module's types resolvable by their BARE name, since `Sarek_typer` qualifies the registered name when it is `Some`. Writing it in would make `M.t` required and `t` unresolvable |
| `Pexp_letmodule` with `txt = None` (`let module _ =`) | `collect_mods` | fell through and failed later as "Kernel must be a function" | **refused**, named (no committed case) |
| `Pexp_open` at the payload top | `collect_mods` | skipped, path dropped | **residual, documented, and the reason it stays.** The open scopes the kernel body, and stripping it means the body is resolved without it (cross-runtime review's point, and it is right). It cannot be closed by re-wrapping the body in an `EOpen`: the native backend re-emits `let open M in` into generated OCaml, so every payload using the form would then need `M` to resolve in the user's file — which is exactly the backlog-208 trap, and `test_inline_node_exhaustion.ml` (payload-top `let open Std in`, no `module Std` alias) would break. Names a kernel needs are in Sarek's own environment, which the open does not populate |
| `Pexp_let` with 2+ bindings at the payload top | `collect_mods` | only the first was looked at, the rest fell through as if they were the kernel function | **refused**, named (no committed case) |
| `function_param_desc.Pparam_val` labelled/optional, `Pparam_newtype` | `collect_fun_params` | refused | unchanged |
| `Pfunction_cases` | `collect_fun_params` | refused at all four call sites | unchanged |

### What is NOT covered, precisely

* `Sarek_ppx.ml` was not walked (see scope, above).
* **Seven** refusals added here have **no committed negative case**:
  `ptype_cstrs`, `ptype_private`, manifest-plus-representation,
  `ptype_attributes`, `locally_abstract_univars`, the anonymous payload module,
  and the 2+-binding payload `let`. All seven were **executed by hand** during
  the sweep through a scratch negative stanza and each printed its own message at
  exit 1 (the outputs are in the PR discussion), so this is not a
  by-inspection claim — but only the 29 committed cases are wired into
  `make test_negative`, and an unasserted refusal is one that can rot.
* Two refusals are **unreachable from OCaml source** and are asserted on AST
  nodes in `test_parse.ml` instead, each with the check that establishes the
  unreachability written at its definition.
* **Seven** **residual drops** are documented rather than closed:
  `popen_override`, `Ptyp_poly`'s quantifiers, a pattern's `Ppat_constraint`, the
  payload module NAME, the payload-top `open`, `pvb_attributes` on an ordinary
  kernel-body `let`, and the type-variable weakening of a module-item result
  annotation. The first three are argued at their site to lose nothing. The
  payload module NAME is load-bearing as it is. The last three are real remaining
  holes with a stated reason for staying: the payload-top `open` cannot be closed
  without breaking `test_inline_node_exhaustion`, the type-variable weakening
  cannot be refused without breaking `test_module_poly`, and `pvb_attributes` on
  an ordinary `let` is simply not done — the same three lines as the superstep
  one, deferred rather than justified. `Pstr_type`'s `rec_flag` was on this list
  and is now REFUSED instead; `constructor_declaration.pcd_attributes` and
  `label_declaration.pld_attributes` stay, because those helpers are shared with
  the top-level route where OCaml is entitled to the attributes.
* The claim "every arm is now read or refused" holds for the four variants and
  the record fields listed at the top of this table, in
  `Sarek_parse.ml` + `Sarek_parse_helpers.ml`, against ppxlib 0.38.0. It is not a
  claim about the PPX as a whole, and for `module_expr_desc` it is a claim about
  the DEFAULT being a refusal rather than about the compiler enforcing the
  enumeration (see that section's note).

