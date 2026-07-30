#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Every workflow step must have exactly one action, and no duplicate keys.
#
# WHY THIS EXISTS. A step was inserted into ci.yml immediately BEFORE an
# existing step's `run:` line rather than after it:
#
#     - name: Check every negative case is asserted (backlog-165)
#     - name: Compile-gate the HIP backend and its tests
#       run: dune build @sarek-hip/all
#
#       run: ./scripts/check-negative-case-coverage.sh
#
# The negative-case step was left with NO action at all, and the HIP step got
# TWO `run:` keys. YAML keeps the last duplicate silently, so the step NAMED
# "Compile-gate the HIP backend" actually ran check-negative-case-coverage.sh
# and the HIP compile-gate -- the entire deliverable of backlog-186 -- never
# executed once. A step that runs something other than what it is named is the
# worst shape of all: the log shows a green check under the right name.
#
# Nothing caught it because no tool in this repo reads the workflow's STRUCTURE.
# ocamlformat does not, dune does not, and GitHub's own parser accepted the
# duplicate key rather than rejecting it.
#
# WHAT IT CHECKS, per step, in every .github/workflows/*.yml:
#   1. exactly one of `run:` / `uses:` (zero = the step does nothing; two = the
#      last silently wins and the name is a lie)
#   2. no key repeated within one step
#
# Deliberately dependency-free: a line scanner, not a YAML library. PyYAML's
# safe_load is the wrong tool here -- it SILENTLY drops duplicate keys, which is
# the exact defect being detected, so parsing with it would report a clean tree.
#
# TWO THINGS IT MUST NOT DESCEND INTO, both measured as false positives on the
# first run of this gate (5 findings, all spurious, 0 real):
#   - nested mappings: `with: { name: coverage }` is an ARTIFACT name, not a
#     second step name. Only DIRECT children of the step are step keys.
#   - block scalars: deploy-pr-preview.yml embeds JavaScript under `script: |`
#     whose object literals contain `owner:`, `repo:`, `body:`. Those are JS,
#     not YAML, and read as duplicate keys to a naive line scan.
#
# Exit codes: 0 = every step well-formed, 1 = a malformed step,
# 2 = cannot run (no workflows, no python3) -- fail closed.

set -uo pipefail

git rev-parse --show-toplevel >/dev/null 2>&1 || {
  echo "check-workflow-steps: not inside a git work tree" >&2
  exit 2
}
cd "$(git rev-parse --show-toplevel)" || exit 2

command -v python3 >/dev/null 2>&1 || {
  echo "check-workflow-steps: python3 not found" >&2
  exit 2
}

python3 - <<'PY'
import glob, re, sys

files = sorted(glob.glob(".github/workflows/*.yml") + glob.glob(".github/workflows/*.yaml"))
if not files:
    print("check-workflow-steps: no workflow files found", file=sys.stderr)
    sys.exit(2)

# A step begins at "- name:" / "- uses:" / "- run:" and continues until the next
# item at the same indentation or a dedent. Keys are recorded with the step they
# belong to, including ones that appear AFTER a blank line -- that is precisely
# how the orphaned `run:` attached itself to the wrong step.
STEP_START = re.compile(r"^(\s*)-\s+(\w[\w-]*):")
# A step written as a YAML FLOW mapping: `- {name: x, run: a, run: b}`. The
# block-mapping scanner below cannot see one -- STEP_START needs `word:` right
# after the dash, so the line matches nothing and the step is INVISIBLE, which
# is worse than a miss: `steps_seen` does not even count it. Found by running
# this gate's own attack list against it (flow mapping with a duplicate `run`
# was accepted, exit 0). Single-line flow mappings are parsed here; anything
# more exotic is REFUSED rather than skipped.
FLOW_STEP = re.compile(r"^(\s*)-\s*\{(.*)\}\s*$")
FLOW_STEP_OPEN = re.compile(r"^\s*-\s*\{")
KEY = re.compile(r"^(\s*)([\w-]+):(.*)$")
# `key: |`, `key: >-`, `key: |+` … everything indented under it is opaque text.
BLOCK_SCALAR = re.compile(r"^\s*[|>][+-]?\s*(#.*)?$")

problems = []
steps_seen = 0

for f in files:
    lines = open(f, encoding="utf-8").read().splitlines()
    cur = None  # (indent, start_line, name, [(key, line)])
    def close(cur):
        global steps_seen
        if cur is None:
            return
        steps_seen += 1
        indent, start, name, keys, _child_indent = cur
        names = [k for k, _ in keys]
        actions = [k for k in names if k in ("run", "uses")]
        label = name or f"(unnamed, line {start})"
        if len(actions) == 0:
            problems.append(
                f"{f}:{start}: step {label!r} has neither `run:` nor `uses:` — it does nothing"
            )
        elif len(actions) > 1:
            where = ", ".join(f"line {ln}" for k, ln in keys if k in ("run", "uses"))
            problems.append(
                f"{f}:{start}: step {label!r} has {len(actions)} actions ({where}) — "
                f"YAML keeps the LAST, so this step does not do what its name says"
            )
        dups = sorted({k for k in names if names.count(k) > 1})
        for d in dups:
            at = ", ".join(f"line {ln}" for k, ln in keys if k == d)
            problems.append(f"{f}:{start}: step {label!r} repeats key {d!r} ({at})")
    # Indentation of a step's DIRECT children: the `- ` marker's indent plus the
    # two columns the dash and space occupy.
    skip_until_indent = None  # inside a block scalar; None = not in one
    for i, line in enumerate(lines, 1):
        if skip_until_indent is not None:
            if not line.strip():
                continue
            cur_ind = len(line) - len(line.lstrip())
            if cur_ind > skip_until_indent:
                continue  # still inside the block scalar's text
            skip_until_indent = None
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fm = FLOW_STEP.match(line)
        if fm:
            close(cur)
            cur = None
            steps_seen += 1
            # Split on top-level commas only; a nested {..} or [..] value is one
            # item. Keys are what precedes the first colon of each item.
            inner, depth_b, item, items = fm.group(2), 0, [], []
            for ch in inner:
                if ch in "{[":
                    depth_b += 1
                elif ch in "}]":
                    depth_b -= 1
                if ch == "," and depth_b == 0:
                    items.append("".join(item))
                    item = []
                else:
                    item.append(ch)
            items.append("".join(item))
            fkeys = [it.split(":", 1)[0].strip() for it in items if ":" in it]
            label = next(
                (it.split(":", 1)[1].strip() for it in items
                 if it.split(":", 1)[0].strip() == "name"),
                f"(flow mapping, line {i})",
            )
            facts = [k for k in fkeys if k in ("run", "uses")]
            if len(facts) == 0:
                problems.append(
                    f"{f}:{i}: step {label!r} has neither `run:` nor `uses:` — it does nothing"
                )
            elif len(facts) > 1:
                problems.append(
                    f"{f}:{i}: step {label!r} has {len(facts)} actions in one flow "
                    f"mapping — YAML keeps the LAST, so this step does not do what "
                    f"its name says"
                )
            for d in sorted({k for k in fkeys if fkeys.count(k) > 1}):
                problems.append(f"{f}:{i}: step {label!r} repeats key {d!r}")
            continue
        if FLOW_STEP_OPEN.match(line):
            # A flow mapping this scanner cannot bound on one line. Refuse: a
            # step it cannot analyse must not be reported as well-formed.
            problems.append(
                f"{f}:{i}: step is a multi-line flow mapping, which this gate "
                f"cannot analyse — rewrite it as a block mapping so it can be checked"
            )
            close(cur)
            cur = None
            continue
        m = STEP_START.match(line)
        if m:
            close(cur)
            indent = len(m.group(1))
            nm = None
            if m.group(2) == "name":
                nm = line.split(":", 1)[1].strip()
            # Child indent DERIVED from where the first key actually sits, not
            # assumed to be dash+2. `-   name: A` / `    run: one` (three spaces
            # after the dash) is valid and common, and the hardcoded dash+2
            # rejected it with "has neither run: nor uses:" — a false positive
            # whose message was also wrong, since the step does have an action.
            child_indent = line.index(m.group(2))
            cur = (indent, i, nm, [(m.group(2), i)], child_indent)
            if BLOCK_SCALAR.match(line.split(":", 1)[1]):
                skip_until_indent = child_indent - 1
            continue
        if cur is not None:
            km = KEY.match(line)
            if km:
                ind = len(km.group(1))
                if ind == cur[4]:
                    cur[3].append((km.group(2), i))
                    if BLOCK_SCALAR.match(km.group(3)):
                        skip_until_indent = ind
                elif ind < cur[4]:
                    close(cur)
                    cur = None
                # ind > cur[4] is a nested mapping (`with:`'s children) — not a
                # step key, so it is neither recorded nor a terminator.
    close(cur)

for p in problems:
    print(p)

if problems:
    print()
    print(f"{len(problems)} malformed workflow step(s). A step with no action is dead;")
    print("a step with two actions runs only the last one, under the other's name.")
    sys.exit(1)

# A gate that examined ZERO steps has verified nothing. The previous version
# printed "OK — 0 steps ..." and exited 0, so a workflow written entirely in a
# form the scanner cannot see reported full coverage of a file it had not read.
# The fail-closed case below covered zero FILES; it never covered zero STEPS.
# Found by adversarial review, reproduced on a flow-mapping-only workflow.
if steps_seen == 0:
    print(
        f"check-workflow-steps: examined {len(files)} workflow file(s) and "
        f"recognised ZERO steps.",
        file=sys.stderr,
    )
    print(
        "  Exit 2, not 0: either the files contain no steps (in which case this "
        "gate is guarding nothing) or they use a step form this scanner cannot "
        "parse. Both need a human, and neither is a pass.",
        file=sys.stderr,
    )
    sys.exit(2)

print(
    f"check-workflow-steps: OK — {steps_seen} steps across {len(files)} workflow file(s), "
    "each with exactly one action and no duplicate keys"
)
PY
