#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Tie every prove-red-spec block to its kb/properties.md gate-red-path row
# (backlog-218).
#
# WHY THIS EXISTS
#
# check-kb-properties.sh's `gate-red-path` check verifies that a declared
# `red_path` file exists and is invoked from a carrier. It never looks at
# prove-red.sh's own discovery, so two drifts pass it silently:
#
#   * a script can carry a live, working `BEGIN prove-red-spec` block while
#     the kb/properties.md row beside it still says `red_path: null` --
#     ci/assert-toolchain.sh sat in exactly that state for months, until
#     PR #384 fixed it by hand;
#   * a row can claim `red_path: scripts/prove-red.sh` -- this repository's
#     convention for "the red path IS a prove-red-spec block beside the tool,
#     not a dedicated *.test.sh" -- for a tool whose block has since been
#     deleted, or edited into something prove-red.sh's own parser would
#     refuse, and nothing today reads far enough into either file to notice.
#
# So the KB's claim that every gate's red path is real is currently
# unverified against the actual spec blocks. This script closes that. It
# rediscovers subjects the same way prove-red.sh does -- a comment-delimited
# `BEGIN prove-red-spec` / `END prove-red-spec` block, top-level files only
# under scripts/ and ci/, *.test.sh and *.test.js excluded -- and cross-checks
# that set against kb/properties.md's gate-red-path rows in both directions.
#
# THE kb/properties.md OBSTACLE, since it is the reason this is its own gate
# rather than three more lines inside prove-red.sh: prove-red.sh builds each
# subject's scratch tree from that SUBJECT's own `copy:` list, and no existing
# subject copies kb/properties.md -- it has never needed to. This checker's
# own `copy:` list (see the spec block at the bottom of this file) DOES
# include kb/properties.md, deliberately, so that its own prove-red-spec
# mutations can edit the copy sitting in the scratch tree, and this script
# reads that mutated copy rather than the pristine one.
#
# The alternative -- point this script outside its scratch tree at the real
# kb/properties.md regardless of where it is invoked from -- was rejected and
# is recorded here rather than silently avoided: a mutation to the KB row
# would then be invisible to it (it would keep reading the un-mutated file
# from the real working tree no matter what the scratch copy said), which
# would make half of this gate's own red-path evidence unfalsifiable. Reading
# strictly from argv[1]/cwd, as below, is what keeps this script's own
# baseline and mutations meaningful under prove-red.sh's isolation model; it
# is also, incidentally, why this script takes an explicit root argument
# rather than walking up to a git toplevel the way some gates here do --
# prove-red.sh's scratch is a bare mktemp directory with no git in it.
#
# WHAT THIS DOES NOT DO
#
# It does not replicate prove-red.sh's MUTATION semantics (apply:/argv:/
# stdin: execution) -- that duplication would be a second, divergence-prone
# implementation of the same grammar. "Unparseable" here means only the
# structural half prove-red.sh's own parser would refuse on before ever
# running anything: not exactly one BEGIN/END pair, END before BEGIN, a line
# between them that is not `key: value`, or no `invoke:` key anywhere in that
# range. A block that parses but is otherwise broken (a `copy:` target that
# does not exist, say) is prove-red.sh's own concern to report, the next time
# it actually runs that subject.
#
# A block that HAS a marker but fails this structural check is still reported
# under "carries a `BEGIN prove-red-spec` block" for direction (a) if it also
# has no matching row -- deliberately: an unparseable block is not a reason to
# excuse the missing row. Only direction (b) -- a row that already claims
# `red_path: scripts/prove-red.sh` -- distinguishes "no marker at all" from
# "marker present but does not parse".
#
# Exit codes:
#   0  every discovered subject is named by a row whose red_path is not null,
#      and every row claiming red_path: scripts/prove-red.sh names a tool that
#      genuinely carries a block prove-red.sh's parser would accept
#   1  at least one of those two things is false -- a real mismatch between
#      the KB's claim and what prove-red.sh would actually discover
#   2  kb/properties.md is missing or malformed (no/multiple ```code-intel
#      fences, a line that is not valid JSON, a gate-red-path row with no
#      `check.tool`, two rows naming the same tool), or neither scripts/ nor
#      ci/ exists under the root. Never a skip.
#
# Usage:
#   scripts/check-provered-kb-link.sh [root]
#
# `root` defaults to this repository. It exists so prove-red.sh can point this
# script at a scratch copy of the tree -- see the spec block below -- and so a
# fresh clone runs it with no argument at all.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -d "$ROOT" ] || { echo "::error::root '$ROOT' is not a directory. This is a" \
     "mechanism failure, not a linkage finding, so it must be exit 2." >&2; exit 2; }
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
import os
import re
import sys

ROOT = os.getcwd()
PROPS = "kb/properties.md"
SCAN_DIRS = ["scripts", "ci"]
BEGIN = "BEGIN prove-red-spec"
END = "END prove-red-spec"
PROVE_RED = "scripts/prove-red.sh"


def fail(msg, code=2):
    print("::error::%s" % msg)
    sys.exit(code)


def strip_comment(line):
    s = line.strip()
    for marker in ("#", "//"):
        if s.startswith(marker):
            s = s[len(marker):]
            if s.startswith(" "):
                s = s[1:]
            return s
    return s


# ---------------------------------------------------------------------------
# kb/properties.md: gate-red-path rows, tool -> red_path (or None).
# ---------------------------------------------------------------------------
if not os.path.isfile(PROPS):
    fail("%s is missing under %s. There is nothing to cross-check a "
         "prove-red-spec block against." % (PROPS, ROOT))

try:
    props_text = open(PROPS, encoding="utf-8", errors="replace").read()
except OSError as exc:
    fail("%s could not be read: %s" % (PROPS, exc))
blocks = re.findall(r"^```code-intel[ \t]*\n(.*?)^```[ \t]*$",
                    props_text, re.M | re.S)
if len(blocks) != 1:
    fail("%s carries %d ```code-intel block(s); exactly one is required to "
         "cross-check prove-red subjects against gate-red-path rows."
         % (PROPS, len(blocks)))

tool_red_path = {}
n_rows = 0
for n, ln in enumerate((l for l in blocks[0].split("\n") if l.strip()), 1):
    try:
        obj = json.loads(ln)
    except json.JSONDecodeError as exc:
        fail("%s code-intel line %d is not valid JSON: %s" % (PROPS, n, exc))
    if not isinstance(obj, dict) or obj.get("type") != "gate-red-path":
        continue
    check = obj.get("check", {})
    tool = check.get("tool") if isinstance(check, dict) else None
    if not isinstance(tool, str) or not tool:
        fail("%s code-intel line %d (%s): a gate-red-path row has no "
             "`check.tool`." % (PROPS, n, obj.get("id", "?")))
    if tool in tool_red_path:
        fail("%s: two gate-red-path rows both name `check.tool`: %s. Which "
             "one this cross-check believes would be undefined."
             % (PROPS, tool))
    tool_red_path[tool] = check.get("red_path")
    n_rows += 1

# ---------------------------------------------------------------------------
# scripts/ and ci/: which top-level files carry a block, and whether it is
# structurally the shape prove-red.sh's own parser would accept.
# ---------------------------------------------------------------------------
def discover():
    found = {}
    roots_present = 0
    for d in SCAN_DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        roots_present += 1
        for name in sorted(os.listdir(full)):
            path = os.path.join(full, name)
            if not os.path.isfile(path):
                continue
            if name.endswith(".test.sh") or name.endswith(".test.js"):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                continue
            lines = src.split("\n")
            starts = [i for i, l in enumerate(lines) if strip_comment(l) == BEGIN]
            ends = [i for i, l in enumerate(lines) if strip_comment(l) == END]
            if not starts and not ends:
                continue
            rel = os.path.join(d, name)
            ok = len(starts) == 1 and len(ends) == 1 and ends[0] > starts[0]
            if ok:
                has_invoke = False
                for i in range(starts[0] + 1, ends[0]):
                    raw = strip_comment(lines[i])
                    if not raw.strip():
                        continue
                    if ":" not in raw:
                        ok = False
                        break
                    if raw.split(":", 1)[0].strip() == "invoke":
                        has_invoke = True
                if ok and not has_invoke:
                    ok = False
            found[rel] = ok
    if roots_present == 0:
        fail("none of %s exists under %s, so this cross-check read nothing."
             % (", ".join(SCAN_DIRS), ROOT))
    return found


subjects = discover()
live = {rel for rel, ok in subjects.items() if ok}

violations = []

# direction (a): ANY discovered marker with no row, or a row whose red_path is
# null -- including one whose block does not fully parse. An unparseable
# block is not a reason to excuse the missing row: prove-red.sh's own
# EXPECTED_SUBJECTS pin is a separate, coarser backstop (it would refuse the
# whole run over a subject-count mismatch), not a substitute for this row-level
# report, so this iterates over every `subjects` key rather than only `live`.
for rel in sorted(subjects):
    if rel not in tool_red_path:
        violations.append(
            "%s carries a `BEGIN prove-red-spec` block with no matching "
            "kb/properties.md gate-red-path row (check.tool == %r). A red "
            "path nothing declares is indistinguishable from no red path at "
            "all -- add a row." % (rel, rel))
    elif tool_red_path[rel] is None:
        violations.append(
            "%s carries a `BEGIN prove-red-spec` block but its "
            "kb/properties.md row declares red_path: null. This is the exact "
            "state ci/assert-toolchain.sh sat in for months before PR #384 "
            "fixed it by hand -- point red_path at %r (or at a dedicated "
            "*.test.sh)." % (rel, rel))

# direction (b): a row claiming red_path: scripts/prove-red.sh for a tool
# whose block is missing or does not parse.
for tool, red_path in sorted(tool_red_path.items()):
    if red_path != PROVE_RED:
        continue
    if tool not in subjects:
        violations.append(
            "kb/properties.md declares gate `%s` red_path: %s, but %s "
            "carries no `BEGIN prove-red-spec` block at all -- the KB's "
            "claim of red-path coverage is currently false."
            % (tool, PROVE_RED, tool))
    elif not subjects[tool]:
        violations.append(
            "kb/properties.md declares gate `%s` red_path: %s, but %s's "
            "prove-red-spec block does not parse (not exactly one BEGIN/END "
            "pair, a line that is not `key: value`, or no `invoke:` key) -- "
            "prove-red.sh would refuse it too, so the KB's claim of "
            "red-path coverage is currently false." % (tool, PROVE_RED, tool))

if violations:
    print("::error::%d prove-red <-> kb/properties.md linkage violation(s)."
          % len(violations))
    for v in violations:
        print("    - %s" % v)
    sys.exit(1)

print("OK: %d live prove-red-spec subject(s) reconciled against %d "
      "kb/properties.md gate-red-path row(s)." % (len(live), n_rows))
PY

# ---------------------------------------------------------------------------
# BEGIN prove-red-spec
# copy: scripts
# copy: ci
# copy: kb/properties.md
# invoke: scripts/check-provered-kb-link.sh
# baseline-exit: 0
# baseline-message: kb/properties.md gate-red-path row(s)
#
# mutation: kb-row-red-path-nulled
#   desc: ci/assert-toolchain.sh's row is edited to red_path: null while its own prove-red-spec block is left untouched -- the exact ci/assert-toolchain.sh gap PR #384 fixed by hand, direction (a), first half.
#   apply: python3 - <<'PYEOF'
#   apply: p = "kb/properties.md"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = '"check": {"tool": "ci/assert-toolchain.sh", "red_path": "scripts/prove-red.sh"}}'
#   apply: assert s.count(old) == 1, ("anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, '"check": {"tool": "ci/assert-toolchain.sh", "red_path": null}}')
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: row declares red_path: null
#
# mutation: kb-row-deleted-entirely
#   desc: the whole KB-GATE-TOOLCHAIN-ASSERT row is deleted, so the tool has no row at all rather than a null one -- direction (a), second half; a row that never existed must be caught the same as one that says null.
#   apply: python3 - <<'PYEOF'
#   apply: p = "kb/properties.md"
#   apply: lines = open(p, encoding="utf-8").read().split("\n")
#   apply: kept = [l for l in lines if not l.startswith('{"id": "KB-GATE-TOOLCHAIN-ASSERT"')]
#   apply: assert len(kept) == len(lines) - 1, ("expected to drop exactly one line, dropped %d" % (len(lines) - len(kept)))
#   apply: open(p, "w", encoding="utf-8").write("\n".join(kept))
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: no matching kb/properties.md gate-red-path row
#
# mutation: claimed-tool-block-vanished
#   desc: check-tvar-id-allocator.sh's row still claims red_path: scripts/prove-red.sh, but its block is deleted from the copied source -- direction (b), missing half.
#   apply: python3 - <<'PYEOF'
#   apply: p = "scripts/check-tvar-id-allocator.sh"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: idx = s.index("# BEGIN prove-red-spec")
#   apply: assert idx > 0, "BEGIN marker not found"
#   apply: open(p, "w", encoding="utf-8").write(s[:idx])
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: carries no `BEGIN prove-red-spec` block at all
#
# mutation: claimed-tool-block-unparseable
#   desc: check-negative-case-coverage.sh's row still claims red_path: scripts/prove-red.sh, but its block is given a SECOND BEGIN marker -- prove-red.sh's own parser refuses two -- direction (b), unparseable half.
#   apply: python3 - <<'PYEOF'
#   apply: p = "scripts/check-negative-case-coverage.sh"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "# BEGIN prove-red-spec\n"
#   apply: assert s.count(old) == 1, ("anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, old + old, 1)
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: does not parse
#
# mutation: kb-properties-malformed-json
#   desc: KB-GATE-SELF's row is truncated mid-JSON. The mechanism half of this gate -- it must refuse rather than silently skip the row or the whole file -- rather than a finding about a real tool.
#   apply: python3 - <<'PYEOF'
#   apply: p = "kb/properties.md"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = '{"id": "KB-GATE-SELF", "type": "gate-red-path"'
#   apply: assert s.count(old) == 1, ("anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, '{"id": "KB-GATE-SELF", "type": "gate-red-path"' + ", broken json here", 1)
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: is not valid JSON
# END prove-red-spec
# ---------------------------------------------------------------------------
