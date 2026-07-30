#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# The type-variable id space has exactly ONE allocator (backlog-183).
#
# WHY A SOURCE GATE AND NOT A TEST. backlog-184 added a mechanical uniqueness
# assertion over the lowered IR, and the review that prompted it expected that
# assertion to cover this class too. It cannot: the defect this guards makes
# `unify` REJECT A LEGAL PROGRAM during type inference, so the kernel never
# reaches lowering and there is no lowered IR to assert over. The failure is also
# arithmetic-dependent — it needs the two counters to have drifted into collision
# — so a runtime test would be a coin flip on counter state rather than a gate.
# A source-level invariant is the shape that actually holds here.
#
# WHAT WENT WRONG. Sarek_typer had exactly one site building a tvar with
# `fresh_var_id ()` — the TERM-variable counter in Sarek_typed_ast — while every
# other tvar uses `fresh_tvar_id ()` from Sarek_types. Two separate Atomics, both
# starting at 0, so the two id spaces overlapped. Since `float_literal_ids` and
# `numeric_required_ids` are keyed on tvar ids and consulted inside `unify` as set
# membership, a leaked term id could be spuriously present in one of those sets
# and reject a program that type-checks correctly.
#
# Exit 0 = one allocator. Exit 1 = a violation. Exit 2 = the gate could not run
# (missing sources), which is a FAILURE and not a pass: a gate that silently
# checks nothing is the failure mode this repo has hit repeatedly.

set -uo pipefail

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; NC=$'\033[0m'

# Every file that may construct a type variable. Listed explicitly rather than
# globbed so that a NEW typer file is a deliberate addition here, not a silent
# omission from the check.
SOURCES=(
    "sarek/ppx/Sarek_types.ml"
    "sarek/ppx/Sarek_typer.ml"
)

missing=0
for f in "${SOURCES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "${RED}CANNOT RUN${NC}: expected source not found: $f" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "${RED}FAIL${NC}: the tvar-allocator gate could not inspect its sources." >&2
    echo "  A moved or renamed typer file must UPDATE this gate's SOURCES list," >&2
    echo "  not disappear from it. Refusing to report success." >&2
    exit 2
fi

# --- Negative half: no tvar may draw from the term-variable counter. ----------
# Matches `Unbound (fresh_var_id` with any spacing. `fresh_var_id` is legitimate
# elsewhere (term binders), so the pattern is deliberately anchored to Unbound —
# the tvar constructor — rather than banning the function outright.
violations=$(grep -nE 'Unbound[[:space:]]*\([[:space:]]*fresh_var_id' "${SOURCES[@]}" || true)

# --- Positive half: the correct allocator must actually be in use. ------------
# Without this the gate would pass on a tree where tvar construction had been
# deleted or renamed away entirely — green because there is nothing left to
# check, which is not the same as the invariant holding.
correct=$(grep -cE 'Unbound[[:space:]]*\([[:space:]]*fresh_tvar_id' "${SOURCES[@]}" \
    | awk -F: '{n += $2} END {print n + 0}')

status=0

if [ -n "$violations" ]; then
    echo "${RED}FAIL${NC}: a type variable is drawing its id from the TERM-variable counter." >&2
    echo "$violations" | sed 's/^/  /' >&2
    echo >&2
    echo "  Use Sarek_types.fresh_tvar_id (), not Sarek_typed_ast.fresh_var_id ()." >&2
    echo "  These are two independent Atomics both starting at 0, so mixing them" >&2
    echo "  makes the tvar id space non-injective. float_literal_ids and" >&2
    echo "  numeric_required_ids are keyed on tvar ids and are read inside unify," >&2
    echo "  so a collision REJECTS A LEGAL PROGRAM rather than merely looking untidy." >&2
    status=1
fi

if [ "$correct" -eq 0 ]; then
    echo "${RED}FAIL${NC}: found no 'Unbound (fresh_tvar_id ...)' construction at all." >&2
    echo "  The gate has nothing to guard, which means either tvar construction" >&2
    echo "  moved to a file not in SOURCES, or it is now spelled differently." >&2
    echo "  Either way this gate is no longer checking the invariant it claims to." >&2
    status=2
fi

if [ "$status" -eq 0 ]; then
    echo "${GREEN}PASS${NC}: one tvar id allocator ($correct construction site(s), 0 from the term counter)"
fi
exit "$status"

# ---------------------------------------------------------------------------
# BEGIN prove-red-spec
# copy: scripts/check-tvar-id-allocator.sh
# copy: sarek/ppx/Sarek_typer.ml
# copy: sarek/ppx/Sarek_types.ml
# invoke: scripts/check-tvar-id-allocator.sh
# baseline-exit: 0
# baseline-message: one tvar id allocator
#
# mutation: term-counter-leak
#   desc: reintroduce the backlog-183 defect verbatim -- a tvar built with fresh_var_id, the term-variable counter. This is the real historical bug, not a synthetic edit: it shipped, and it became able to reject legal programs once the float-literal and numeric-required registries started being read inside unify.
#   apply: python3 - <<'PYEOF'
#   apply: p = "sarek/ppx/Sarek_typer.ml"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "Unbound (fresh_tvar_id (), env_inner.current_level)"
#   apply: assert s.count(old) == 1, ("expected exactly one site, found %d" % s.count(old))
#   apply: s = s.replace(old, "Unbound (fresh_var_id (), env_inner.current_level)")
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: drawing its id from the TERM-variable counter
#
# mutation: gate-blinded-by-a-moved-source
#   desc: point the gate at a file that does not exist, simulating a typer file being renamed without updating SOURCES. The gate must REFUSE rather than report success -- a check whose inputs vanished is the vacuous-green failure mode, and it has bitten this repo before (7 unwired suites, the nvdisasm self-skip, the Objective-C self-skip on a real Mac).
#   apply: python3 - <<'PYEOF'
#   apply: p = "scripts/check-tvar-id-allocator.sh"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = '"sarek/ppx/Sarek_typer.ml"\n)'
#   apply: assert s.count(old) == 1, ("SOURCES anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, '"sarek/ppx/Sarek_typer_RENAMED.ml"\n)')
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: could not inspect its sources
#
# mutation: positive-control-removed
#   desc: remove EVERY correct construction, across both source files. The negative half alone would go GREEN here -- there is no fresh_var_id to find -- which is exactly why the positive half exists: without it, tvar construction being renamed or moved out of SOURCES would read as the invariant holding. Removing only ONE of the two sites deliberately does NOT trip this, and that is correct rather than a weakness: `correct` is a whole-tree "at least one" count, so one site remaining means the gate still has something real to guard. An earlier version of this mutation blanked only the typer's site and the subject stayed green -- the mutation was wrong about the gate, not the gate about the invariant, and prove-red.sh is what surfaced the difference.
#   apply: python3 - <<'PYEOF'
#   apply: import re
#   apply: pat = r"Unbound\s*\(\s*fresh_tvar_id\s*\(\)"
#   apply: a = "sarek/ppx/Sarek_typer.ml"
#   apply: b = "sarek/ppx/Sarek_types.ml"
#   apply: sa, na = re.subn(pat, "Unbound (0", open(a, encoding="utf-8").read())
#   apply: sb, nb = re.subn(pat, "Unbound (0", open(b, encoding="utf-8").read())
#   apply: assert na + nb >= 2, ("expected to blank at least 2 sites, blanked %d" % (na + nb))
#   apply: open(a, "w", encoding="utf-8").write(sa)
#   apply: open(b, "w", encoding="utf-8").write(sb)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: found no 'Unbound (fresh_tvar_id
# END prove-red-spec
# ---------------------------------------------------------------------------
