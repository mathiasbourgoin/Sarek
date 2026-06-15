# TypeSafety — Findings

## Finding template

| ID | Severity | Target (file:line) | Property | Status |
|----|----------|--------------------|----------|--------|
| F-TS-01 | MAJOR | `sarek/ppx/Sarek_typer.ml:538` (ELet) | lexical scoping | RESOLVED |

Findings discovered during proof work (e.g. a soundness lemma that fails to
prove, revealing a checker bug or an over-strong spec) are recorded here with a
concrete reproduction and a pointer to the failing theorem.

---

## F-TS-01: ELet scope leak in Sarek_typer.infer

**Status**: RESOLVED
**Severity**: MAJOR
**Found by**: T1-CMBT differential QCheck (formal/type-safety)
**Counterexample**: `let y = (let x = 0 in 0) in x`

**Description**: Inner let-bindings inside a let-value expression leaked into the
continuation body. Sarek_typer.infer accepted `x` as in-scope in the body where the
Coq model (correct HM lexical scoping) correctly rejects it as UnboundVar.

**Root cause**: `infer_let_binding` built the body env from the env accumulated during
value inference (`env'`), which included nested-let variable bindings from inside
the value expression. Should have been built from the pre-value env (vars restored).
`enter_level`/`exit_level` (Sarek_env.ml:170/172) only touch `current_level`, not
`vars`, so the leak came purely from threading the post-value env into the body env.

**Fix**: Restore `vars` to pre-value state before building body env
(`env_clean = {env_after with vars = env_in.vars}`), preserving `current_level` and
all other unification side-state from `env_after`.
See commit: 20b79b36 (fix(typer): F-TS-01 — restore ELet body env to pre-value variable scope)
