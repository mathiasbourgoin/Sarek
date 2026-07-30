#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Guard against silently-unwired test executables (the "vacuous e2e" failure
# mode fixed in 2026-07: an (executable) with no run rule builds green but
# never executes). Every declared target must be classified: wired to an
# alias/rule, or declared in scripts/unwired-targets.tsv with a reason.
#
# WIDENED TO THE WHOLE TREE (backlog-69). It used to read ONE hardcoded path,
# sarek/tests/e2e/dune, so a gate against unwired tests was itself unwired
# everywhere else. Measured on widening: 45 unwired out of 275 declared. Five of
# them were real — test_transpile_proof (the only test of the transpiler, and the
# transpiler is what #356 found routing five backends through a lossy emit path),
# test_native_runtime, test_runtime_comparison, and gate_numeric + gate_framework,
# which are GATES that nothing invoked. All five now wired; all five verified
# passing BEFORE wiring, so this does not turn CI red.
#
# The other 40 are legitimately not runnable by `dune test` — GPU timing
# benchmarks, js_of_ocaml browser targets, hardware probes, one operator tool —
# and are declared in scripts/unwired-targets.tsv. A gate reporting 45 red on 39
# non-problems gets switched off, which is why the widening ships with the
# declaration rather than after it.
#
# BOTH DIRECTIONS ARE RED. Unwired with no declaration: red. A declaration for a
# target that is now wired, or that no longer exists: also red — a stale
# exemption silently keeps covering something already fixed or deleted, which is
# the "declared but unverified" shape the declaration exists to close.
set -euo pipefail

# ROOT and DECL_FILE are arguments so prove-red can exercise this against a
# small fixture tree instead of the repository. That is not a convenience: the
# gate's input IS the whole tree, so prove-red's copy-a-few-files scratch world
# would leave most dune files absent and every declaration for them would read
# as STALE — a baseline that is red for a reason that has nothing to do with the
# property. Defaults are the real tree, which is what CI runs.
ROOT="${1:-.}"
DECL_FILE="${2:-scripts/unwired-targets.tsv}"
[ -d "$ROOT" ] || { echo "ERROR: root $ROOT not found"; exit 2; }
[ -f "$DECL_FILE" ] || { echo "ERROR: $DECL_FILE not found (run from repo root)"; exit 2; }

python3 - "$DECL_FILE" "$ROOT" <<'PYEOF'
import re, sys, pathlib

decl_path = sys.argv[1]
root = pathlib.Path(sys.argv[2])
# Read the declaration: <dune file>::<target>\t<tag>\t<justification>.
declared_exempt = {}
for lineno, line in enumerate(open(decl_path), 1):
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        continue
    parts = line.split("\t")
    if len(parts) < 3 or "::" not in parts[0] or not parts[1].strip() or not parts[2].strip():
        print(f"ERROR: {decl_path}:{lineno}: expected '<dune>::<target>\\t<tag>\\t<why>', got: {line}")
        sys.exit(2)
    declared_exempt[parts[0]] = parts[1].strip()

# Target tags, then the two ALIAS tags (backlog-186): an alias may be
# legitimately uninvoked in CI only because it needs hardware CI lacks, or
# because it is an operator-run entry point. Both must still be DECLARED.
VALID_TAGS = {"benchmark", "browser-target", "hardware-probe", "tool",
              "gpu-required", "manual-only"}
bad = {k: v for k, v in declared_exempt.items() if v not in VALID_TAGS}
if bad:
    for k, v in sorted(bad.items()):
        print(f"ERROR: {k}: unknown reason tag {v!r} (expected one of {sorted(VALID_TAGS)})")
    sys.exit(2)


def toplevel_forms(s):
    forms, depth, start = [], 0, None
    for i, c in enumerate(s):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and start is not None:
                forms.append(s[start : i + 1])
                start = None
    return forms


total_declared = 0
unwired = []          # keys that are unwired in the tree

# This gate's OWN prove-red fixtures are dune files describing deliberately
# unwired targets. Once the walk went whole-tree they were scanned as if they
# were real, and CI reported `alias-coverage/exempt/dune::fixture_exempt` as an
# undeclared unwired target — correct by the letter, meaningless in fact. Never
# seen locally because every local run passed the fixture dir as the root.
#
# The exclusion is conditional on the root, and that is the load-bearing part:
# when prove-red invokes this WITH the fixture dir as root, the fixtures are the
# subject and must still be scanned. A blanket skip would make those mutations
# find nothing, which is why the `unwired-executable` mutation is what catches a
# regression here — it asserts UNWIRED is still detected inside that tree.
root_in_fixtures = "prove-red-fixtures" in root.resolve().parts

for dune in sorted(root.rglob("dune")):
    if "_build" in dune.parts:
        continue
    if not root_in_fixtures and "prove-red-fixtures" in dune.parts:
        continue
    raw = dune.read_text()
    # Strip line comments so names mentioned in prose do not count as wiring.
    src = re.sub(r";[^\n]*", "", raw)
    if src.count("(") != src.count(")"):
        print(f"ERROR: unbalanced parentheses after comment-stripping in {dune} - parser assumptions violated")
        sys.exit(2)
    forms = toplevel_forms(src)
    declared = set()
    for form in forms:
        head = form[1:].split(None, 1)[0] if form[1:].split(None, 1) else ""
        if head in ("executables", "tests"):
            m = re.search(r"\(names((?:[^()])*)\)", form)
            if m:
                declared.update(m.group(1).split())
        elif head in ("executable", "test"):
            m = re.search(r"\(name\s+([A-Za-z0-9_]+)\s*\)", form)
            if m:
                declared.add(m.group(1))
    if not declared:
        continue
    total_declared += len(declared)
    # A (test)/(tests) stanza is self-wiring: dune attaches it to runtest.
    self_wired = set()
    for form in forms:
        head = form[1:].split(None, 1)[0] if form[1:].split(None, 1) else ""
        if head == "test":
            m = re.search(r"\(name\s+([A-Za-z0-9_]+)\s*\)", form)
            if m:
                self_wired.add(m.group(1))
        elif head == "tests":
            m = re.search(r"\(names((?:[^()])*)\)", form)
            if m:
                self_wired.update(m.group(1).split())
    referenced = set(re.findall(r"([A-Za-z0-9_]+)\.exe", src)) | self_wired
    for name in sorted(declared - referenced):
        rel = dune.relative_to(root)
        unwired.append(f"{rel}::{name}")

if total_declared == 0:
    print("ERROR: no executables or tests parsed from any dune file - parser broken or tree restructured")
    sys.exit(2)

# DIRECTION 1 -- unwired with no declaration.
undeclared = [k for k in unwired if k not in declared_exempt]
# DIRECTION 2 -- a declaration that no longer describes anything unwired. This is
# what stops the file rotting: a stale exemption would keep silently covering a
# target that has since been wired or deleted.
unwired_set = set(unwired)
# `alias::` keys live in the same file but belong to DIRECTION 3, which has its
# own staleness check below. Without this exclusion every alias declaration
# reads STALE here, because it is not in the TARGET unwired set — two
# namespaces sharing one declaration file.
stale = [k for k in declared_exempt
         if not k.startswith("alias::") and k not in unwired_set]

if undeclared or stale:
    for k in sorted(undeclared):
        print(f"UNWIRED: {k} is declared in a dune file, referenced by no alias or run rule,")
        print(f"         and has no line in {decl_path}.")
    for k in sorted(stale):
        print(f"STALE:   {decl_path} exempts {k}, but it is not unwired (wired now, or gone).")
    print()
    if undeclared:
        print("If it is a TEST that can run, WIRE it:")
        print("  (rule (alias runtest) (action (run %{dep:NAME.exe})))")
        print(f"Only if it genuinely cannot run under `dune test`, add a line to {decl_path}")
        print("with one of: benchmark | browser-target | hardware-probe | tool.")
    if stale:
        print(f"Remove the stale line(s) from {decl_path} - the exemption is no longer describing")
        print("anything, and leaving it there means it silently covers whatever takes that name next.")
    sys.exit(1)

# DIRECTION 3 -- an alias that nothing outside dune ever invokes (backlog-186).
#
# Directions 1 and 2 ask whether a TARGET is reached by a rule in its own dune
# file. They do not ask whether the ALIAS that rule attaches to is ever invoked.
# So attaching tests to a brand-new alias satisfies them completely: the .exe
# name appears in the file, `declared - referenced` is empty, and the gate says
# OK. That is exactly how the whole sarek-hip suite sat outside CI while this
# gate reported 277 targets all accounted for -- 6 rules on `e2e-hip`, 6 on
# `e2e-gpu`, 12 executables, invoked from nowhere. One level of reachability
# short.
#
# Self-executing aliases need no invoker: dune runs them itself. `dune build`
# builds @default, which depends on @all; `dune runtest` builds @runtest; @fmt is
# built by this repo's own `dune build @fmt` formatting gate.
#
# `check`, `doc` and `install` were in this set and are NOT self-executing --
# nothing builds them unless something names them. Keeping them here meant a rule
# attached to `(alias check)` was skipped as though dune ran it, which is the
# gate's own vacuous-pass shape: the exemption was granted by the name of the
# alias rather than by any evidence it gets built. Removing them costs nothing on
# this tree (its only rule aliases are runtest, e2e-gpu and e2e-hip) and closes
# the hole. `doc` is genuinely reachable anyway when it is used -- docs.yml runs
# `dune build @doc` -- so it now passes on evidence rather than by assumption.
# Reported by CodeRabbit on PR #383.
SELF_EXECUTING = {"runtest", "default", "fmt", "all"}

rule_aliases = {}
for dune in sorted(root.rglob("dune")):
    if "_build" in dune.parts:
        continue
    if not root_in_fixtures and "prove-red-fixtures" in dune.parts:
        continue
    src = re.sub(r";[^\n]*", "", dune.read_text(errors="replace"))
    for m in re.finditer(r"\(alias\s+([A-Za-z0-9_-]+)\s*\)", src):
        # A dune file with six rules on one alias must not be listed six times.
        rel = str(dune.relative_to(root))
        bucket = rule_aliases.setdefault(m.group(1), [])
        if rel not in bucket:
            bucket.append(rel)

# Anything that could invoke an alias. Read from the SCANNED ROOT, so the
# fixture tree exercises this direction too rather than skipping it -- a
# direction that silently does not apply is the vacuous-pass shape this whole
# file exists to prevent.
invoker_globs = [".github/workflows/*.yml", ".github/workflows/*.yaml",
                 "Makefile", "makefile", "scripts/*", "ci/*"]
invoker_text = []
invoker_count = 0
for g in invoker_globs:
    for f in root.glob(g):
        # The declaration file lives under scripts/ and its JUSTIFICATIONS name
        # the very aliases they exempt, so counting it as an invoker makes every
        # declared alias look reachable — and then DIRECTION 3 reports its own
        # declaration STALE. Caught by this gate on itself.
        if f.resolve() == pathlib.Path(decl_path).resolve():
            continue
        if f.is_file():
            invoker_count += 1
            try:
                invoker_text.append(f.read_text(errors="replace"))
            except OSError:
                pass
# Strip #-comments before testing for an invocation, for exactly the reason
# the dune scan above strips ;-comments: a name MENTIONED IN PROSE is not
# wiring. Caught by this gate on its own first run — a ci.yml comment
# explaining why @e2e-hip is not invoked made @e2e-hip look invoked.
# Erring toward more reds is the safe direction here: a stripped `#` inside
# a quoted string can only HIDE an invocation, never invent one.
blob = "\n".join(re.sub(r"#[^\n]*", "", t) for t in invoker_text)

if invoker_count == 0:
    print("ERROR: no invoker files (.github/workflows, Makefile, scripts/, ci/) under "
          f"{root} - cannot tell whether any alias is reachable, and reporting OK "
          "would be a vacuous pass")
    sys.exit(2)

# An alias is REACHABLE only if something actually BUILDS it. A bare "@name"
# appearing anywhere in the invoker text is not that.
#
# The previous test was `("@" + alias) in blob` after stripping #-comments, and
# an adversarial review broke it in two ways:
#   - a script containing `echo "we deliberately do not build @e2e-fixture"`
#     satisfied reachability, so a ci.yml step `name:` field, an echo, or a
#     Python docstring all counted as wiring. The stripped-comments note claimed
#     "a name MENTIONED IN PROSE is not wiring", which held only for #-prefixed
#     prose.
#   - the match was a PREFIX match, so a future alias `e2e` would be satisfied
#     by the existing string `@e2e-hip`. No live collision, but latent.
#
# Now it must appear as an argument to a command that builds it, and the alias
# name must END at a word boundary.

def alias_is_built(alias, text):
    """True only if some `dune build`-shaped command names this exact alias.

    A DIRECTORY-SCOPED invocation counts: `dune build @somedir/somealias` builds
    `somealias` in somedir/ and below, and the `@@` form is the same thing
    restricted to that one directory -- both are how dune is actually driven. The
    previous pattern required the alias name to sit immediately after `@`, so
    every prefixed form read as UNREACHABLE and the gate would have demanded an
    exemption for an alias CI really did build. That direction of error is the
    expensive one: a gate that reports a correct tree dirty gets switched off, and
    its true findings go with it. Reported by CodeRabbit on PR #383.

    The example above uses deliberately fake names. Writing a REAL alias here
    makes this very file wire it: scripts/* is invoker text, `#`-comments are
    stripped before the search but a Python docstring is not, and an earlier draft
    of this docstring said `@sarek-hip/e2e-hip` -- which made alias::e2e-hip read
    as reachable and turned its exemption STALE. Same trap the DIRECTION 3 notes
    above record against a ci.yml comment, one layer in.

    The trailing `(?![\\w/-])` still does the anti-prefix work it was added for --
    alias `e2e` is not satisfied by `@e2e-hip`, since the lookahead rejects the
    `-`, and the optional directory groups cannot rescue the match because they
    each require a `/`.
    """
    pat = re.compile(
        r"dune\s+(?:build|exec|runtest|test)[^\n;&|]*?@@?"
        + r"(?:[A-Za-z0-9_.-]+/)*"
        + re.escape(alias)
        + r"(?![\w/-])"
    )
    return pat.search(text) is not None

unreachable = []
for alias, files in sorted(rule_aliases.items()):
    if alias in SELF_EXECUTING:
        continue
    if alias_is_built(alias, blob):
        continue
    unreachable.append((alias, files))

undeclared_aliases = [a for a, _ in unreachable if ("alias::" + a) not in declared_exempt]
stale_aliases = [k for k in declared_exempt
                 if k.startswith("alias::")
                 and k[len("alias::"):] not in {a for a, _ in unreachable}]

if undeclared_aliases or stale_aliases:
    for a in sorted(undeclared_aliases):
        files = dict(unreachable)[a]
        print(f"UNREACHABLE ALIAS: @{a} is attached to rules in {', '.join(files)},")
        print(f"                   but no workflow, Makefile or script ever invokes @{a}.")
        print("                   Every target behind it builds and never runs.")
    for k in sorted(stale_aliases):
        print(f"STALE:   {decl_path} exempts {k}, but that alias is reachable now (or gone).")
    print()
    if undeclared_aliases:
        print("Either invoke it (a CI step running `dune build @NAME`), or declare it in")
        print(f"{decl_path} as `alias::NAME` with one of: gpu-required | manual-only.")
    sys.exit(1)

print(f"OK: {total_declared} target(s) across the tree; {len(unwired)} unwired, all declared with a reason; {len(rule_aliases)} rule alias(es), all reachable or declared")
PYEOF

# ---------------------------------------------------------------------------
# Red-path evidence for this gate, executed by scripts/prove-red.sh. Until
# backlog-151 this was one of the four gates kb/properties.md recorded as
# `red_path: null` -- its refusals (missing file, nothing parsed) were written
# and never observed firing. The `empty-declared-set` mutation below is the one
# that matters: this gate's whole job is to notice unwired executables, and a
# parse that finds zero of them would otherwise report "OK: all 0 ... wired".
#
# The subject runs against a FIXTURE tree, not the repository. That is forced,
# not preferred: this gate's input is the whole tree, so a scratch world holding
# only a few copied dune files would leave most of them absent and every
# declaration for them would read STALE — a baseline red for a reason unrelated
# to the property being proven. The fixture carries a wired target AND a
# legitimately-unwired declared one, so "refuses an undeclared target" and
# "refuses every unwired target" cannot be confused.
#
# BEGIN prove-red-spec
# copy: scripts/check-test-alias-coverage.sh
# copy: scripts/prove-red-fixtures/alias-coverage/decl.tsv
# copy: scripts/prove-red-fixtures/alias-coverage/wired/dune
# copy: scripts/prove-red-fixtures/alias-coverage/exempt/dune
# copy: scripts/prove-red-fixtures/alias-coverage/Makefile
# invoke: scripts/check-test-alias-coverage.sh
# baseline-argv: scripts/prove-red-fixtures/alias-coverage scripts/prove-red-fixtures/alias-coverage/decl.tsv
# baseline-exit: 0
# baseline-message: 1 unwired, all declared with a reason
#
# mutation: alias-unreachable
#   desc: remove the Makefile's `dune build @e2e-fixture` line, leaving a non-self-executing alias that nothing builds. This is DIRECTION 3's own defect -- a dune file wired to an alias that no workflow, Makefile or script ever invokes, so its tests are declared, built by nobody and run by nobody. Without this mutation DIRECTION 3 was covered by nothing: the fixture had no invoker at all, the gate exited 2, and the baseline was not green (observed in CI on PR #383).
#   apply: python3 - <<'PYEOF'
#   apply: p = "scripts/prove-red-fixtures/alias-coverage/Makefile"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "\tdune build @wired/e2e-fixture\n"
#   apply: assert s.count(old) == 1, ("invoker line not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "\t@echo nothing\n")
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: e2e-fixture
#
# mutation: alias-prefix-collision
#   desc: the invoker builds `@e2e`, a strict PREFIX of the declared `e2e-fixture` alias. It must NOT count as building it. Without the trailing word-boundary lookahead in alias_is_built this reported the alias reachable, so a future short alias would have been satisfied by an unrelated longer one that merely starts the same way. Pins the guard in the same breath as the directory-prefix support added beside it, since a pattern loose enough to accept `@dir/name` is exactly the one at risk of accepting `@name-suffix`.
#   apply: python3 - <<'PYEOF'
#   apply: p = "scripts/prove-red-fixtures/alias-coverage/Makefile"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "\tdune build @wired/e2e-fixture\n"
#   apply: assert s.count(old) == 1, ("invoker line not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "\tdune build @wired/e2e\n")
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: e2e-fixture
#
# mutation: unwired-undeclared
#   desc: a target declared with no alias, no run rule and no line in the declaration — the "builds green, never executes" shape this gate exists for.
#   apply: printf '\n(executable (name zz_probe))\n' >> scripts/prove-red-fixtures/alias-coverage/wired/dune
#   argv: scripts/prove-red-fixtures/alias-coverage scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   expect-exit: 1
#   expect-message: UNWIRED: wired/dune::zz_probe
#
# mutation: stale-declaration
#   desc: the OTHER direction, and the one that stops the declaration rotting. A line exempting a target that is NOT unwired (wired since, or renamed) must be red — otherwise the exemption silently keeps covering whatever takes that name next.
#   apply: printf 'wired/dune::fixture_wired\tbenchmark\tnot actually unwired\n' >> scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   argv: scripts/prove-red-fixtures/alias-coverage scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   expect-exit: 1
#   expect-message: STALE
#
# mutation: bad-reason-tag
#   desc: a tag outside the closed set. Free-text reasons would let "TODO" or "flaky" become exemptions, so the vocabulary is fixed and an unknown tag is a usage error, not a silent pass.
#   apply: printf 'exempt/dune::zz_tag\twhatever\tunknown tag\n' >> scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   argv: scripts/prove-red-fixtures/alias-coverage scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   expect-exit: 2
#   expect-message: unknown reason tag
#
# mutation: malformed-declaration-line
#   desc: a line missing its justification column. A declaration whose reason is empty is an exemption with no stated cause, which is the thing this file exists to prevent.
#   apply: printf 'exempt/dune::zz_bad\tbenchmark\n' >> scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   argv: scripts/prove-red-fixtures/alias-coverage scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   expect-exit: 2
#   expect-message: expected
#
# mutation: missing-declaration-file
#   desc: the declaration is gone. An environment mutation: with no exemptions the gate must refuse rather than proceed, because "no exemptions read" and "nothing needs exempting" are not the same statement.
#   apply: rm -f scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   argv: scripts/prove-red-fixtures/alias-coverage scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   expect-exit: 2
#   expect-message: not found
#
# mutation: empty-declared-set
#   desc: the tree parses cleanly but declares no targets. An empty declared set turns this strict check into a permissive one, so it must be exit 2 rather than "OK: 0 target(s)".
#   apply: printf '(rule (alias runtest))\n' > scripts/prove-red-fixtures/alias-coverage/wired/dune && printf '(rule (alias runtest))\n' > scripts/prove-red-fixtures/alias-coverage/exempt/dune
#   argv: scripts/prove-red-fixtures/alias-coverage scripts/prove-red-fixtures/alias-coverage/decl.tsv
#   expect-exit: 2
#   expect-message: no executables or tests parsed
# END prove-red-spec
# ---------------------------------------------------------------------------
