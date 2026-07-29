#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Every negative-compile case declared in sarek/tests/negative/dune must be
# ASSERTED by the Makefile's test_negative target (backlog-165).
#
# Why this gate exists. The negative suite's real assertions do not live beside
# the cases: sarek/tests/negative/dune only declares a profile-gated library per
# case, and the exact-stderr checks live in the Makefile's test_negative target.
# Nothing connects the two. Adding a case to the dune file gives you a library
# that nothing ever builds, so the case silently never runs -- and, because a
# negative case is "expected to fail", its absence looks exactly like its
# success. That is the same shape as the ptxas-sweep recipe gate in
# test_ptx_intrinsic_sweep.ml, which exists for the same reason and caught a
# real omission the day this was written.
#
# Both directions are checked, because each is a different defect:
#   * declared, not asserted -> a case that never runs (silent hole)
#   * asserted, not declared -> the target builds a target that does not exist,
#     so `dune build` fails for a reason unrelated to the property, and the
#     grep-for-message check reports the wrong thing
#
# NOT in scope, deliberately: whether each asserted message is the RIGHT one for
# its case. That is a semantic claim about a diagnostic and no lexical gate can
# settle it. The known live weakness in that direction is recorded in
# kb/sarek/tests/negative.md (a coincidental substring match remains possible),
# and test_superstep_diverged's documented KNOWN-ISSUE fallback is deliberately
# non-blocking and is NOT treated as an omission here -- it is asserted, it just
# tolerates one outcome on purpose.
#
# Exit 0 = every declared case is asserted and vice versa.
# Exit 1 = a real mismatch, with the offending case names printed.
# Exit 2 = the gate could not do its job (a file is missing, or either side
#          came out EMPTY -- an empty set would make the comparison vacuously
#          pass, which is the failure mode this project has hit before).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DUNE="sarek/tests/negative/dune"
MK="Makefile"

for f in "$DUNE" "$MK"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found -- cannot check negative-case coverage" >&2
    exit 2
  fi
done

# Declared: every `(name neg_...)` library stanza in the negative dune file.
declared="$(grep -oP '^\s*\(name \Kneg_[a-z0-9_]+' "$DUNE" | sort -u)"

# Asserted: every neg_* target the test_negative recipe actually builds. Scoped
# to that recipe only -- a mention anywhere else in the Makefile is not an
# assertion. The path prefix is what makes this unambiguous.
asserted="$(sed -n '/^test_negative:/,/^$/p' "$MK" \
            | grep -oP 'sarek/tests/negative/\Kneg_[a-z0-9_]+' | sort -u)"

n_declared="$(printf '%s' "$declared" | grep -c . || true)"
n_asserted="$(printf '%s' "$asserted" | grep -c . || true)"

# Fail closed on an empty side. Without this, a change that breaks either
# extraction (a dune reformat, a Makefile retarget) would empty both sets and
# the comparison would report perfect agreement over nothing.
if [ "$n_declared" -eq 0 ]; then
  echo "ERROR: found 0 declared negative cases in $DUNE -- the extraction is broken, not the tree" >&2
  exit 2
fi
if [ "$n_asserted" -eq 0 ]; then
  echo "ERROR: found 0 asserted negative cases in $MK's test_negative recipe -- the extraction is broken, or the target no longer builds them by path" >&2
  exit 2
fi

missing="$(comm -23 <(printf '%s\n' "$declared") <(printf '%s\n' "$asserted"))"
extra="$(comm -13 <(printf '%s\n' "$declared") <(printf '%s\n' "$asserted"))"

rc=0

if [ -n "$missing" ]; then
  echo "FAIL: declared in $DUNE but never asserted by 'make test_negative':" >&2
  printf '%s\n' "$missing" | sed 's/^/  - /' >&2
  echo "" >&2
  echo "  Each of these is a negative case that never runs. Add a line to the" >&2
  echo "  test_negative recipe that builds it and greps for its exact expected" >&2
  echo "  message -- 'it failed to compile' is not an assertion, since a typo" >&2
  echo "  or an unrelated build error satisfies it too." >&2
  rc=1
fi

if [ -n "$extra" ]; then
  [ "$rc" -eq 1 ] && echo "" >&2
  echo "FAIL: asserted by 'make test_negative' but not declared in $DUNE:" >&2
  printf '%s\n' "$extra" | sed 's/^/  - /' >&2
  echo "" >&2
  echo "  The recipe builds a target that does not exist, so the build fails" >&2
  echo "  for a reason unrelated to the property under test and the message" >&2
  echo "  grep reports something misleading." >&2
  rc=1
fi

if [ "$rc" -eq 0 ]; then
  echo "OK: $n_declared negative case(s) declared, all asserted by 'make test_negative'."
fi

exit "$rc"

# ---------------------------------------------------------------------------
# BEGIN prove-red-spec
# copy: scripts/check-negative-case-coverage.sh
# copy: sarek/tests/negative/dune
# copy: Makefile
# invoke: scripts/check-negative-case-coverage.sh
# baseline-exit: 0
# baseline-message: negative case(s) declared, all asserted
#
# mutation: declared-not-asserted
#   desc: a new negative case is declared in the dune file and given no assertion in the test_negative recipe. This is the hole the gate exists for -- the case's library is never built, so it never runs, and because a negative case is "expected to fail" its absence is indistinguishable from its success.
#   apply: python3 - <<'PYEOF'
#   apply: p = "sarek/tests/negative/dune"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: s += """
#   apply: (library
#   apply:  (name neg_test_freshly_added_case)
#   apply:  (modules test_freshly_added_case)
#   apply:  (libraries sarek sarek.stdlib sarek.ppx.lib)
#   apply:  (preprocess
#   apply:   (pps sarek_ppx))
#   apply:  (flags
#   apply:   (:standard -w -33))
#   apply:  (enabled_if
#   apply:   (= %{profile} "negative")))
#   apply: """
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: never asserted by
#
# mutation: asserted-not-declared
#   desc: the recipe asserts a case that the dune file does not declare. The build then fails because the target is unknown, not because the property is violated, and the message grep reports a misleading result.
#   apply: python3 - <<'PYEOF'
#   apply: p = "Makefile"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: needle = "\t@echo \"=== Negative tests (expected compile errors) ===\"\n"
#   apply: add = "\t@out=$$(mktemp); dune build --profile=negative sarek/tests/negative/neg_test_does_not_exist.cma > \"$$out\" 2>&1; if grep -q \"nope\" \"$$out\"; then echo \"  PASS\"; else cat \"$$out\"; rm -f \"$$out\"; exit 1; fi; rm -f \"$$out\"\n"
#   apply: assert s.count(needle) == 1
#   apply: open(p, "w", encoding="utf-8").write(s.replace(needle, needle + add))
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: not declared in
#
# mutation: empty-declared-side
#   desc: the dune extraction is broken (here by renaming the stanza key) so the declared set comes out empty. A set-comparison gate reports perfect agreement over nothing, which is the vacuous-pass shape this project has shipped before -- it must be exit 2, not 0.
#   apply: python3 - <<'PYEOF'
#   apply: p = "sarek/tests/negative/dune"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: open(p, "w", encoding="utf-8").write(s.replace("(name neg_", "(nom neg_"))
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: extraction is broken
# END prove-red-spec
# ---------------------------------------------------------------------------
