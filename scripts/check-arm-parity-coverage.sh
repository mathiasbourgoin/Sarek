#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Every backend `arm` name must be covered by the arm-parity matrix (task #94).
#
# WHY A SECOND, LEXICAL CHECK
#
# sarek/tests/unit/test_backend_arm_parity.ml probes each of the five source
# backends with each intrinsic in its table and asserts the observed behaviour
# has not changed. That is the real check — it sees through `pre_hook` and
# `post_hook`, which a grep cannot.
#
# But it can only probe names it was told about. A name added to one backend's
# `arm` and to no list is invisible to it: the suite stays green, having verified
# nothing about the new name. That is not a hypothetical shape — it is the
# defining property of a hand-maintained list, and this repository has spent two
# days removing instances of it.
#
# So: this script extracts every string literal used as a match arm inside each
# backend's `arm` field and asserts it appears in the test's `names` list. The
# pair is the same arrangement as the Rocq `admit` grep beside `rocq check` — a
# cheap lexical tripwire guarding the entrance to an expensive semantic check,
# each catching what the other structurally cannot.
#
# It deliberately does NOT check the converse (a name in the test with no arm
# anywhere). That is legitimate and load-bearing: `log10` has no `arm` on GLSL
# and is handled by `pre_hook`, and several names are reached only through the
# FFI registry. Requiring an arm for every tested name would forbid exactly the
# indirection the behavioural test exists to see through.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

TEST=sarek/tests/unit/test_backend_arm_parity.ml
BACKENDS="cuda opencl metal wgsl glsl"

[ -f "$TEST" ] || {
  echo "::error::$TEST is missing — the arm-parity matrix it guards is gone, \
so this check would pass having compared nothing."
  exit 1
}

# python3 rather than sed: the extraction has to find the `arm = ...` field and
# stop at its terminating `| _ -> None)`, which is a bracket-matching job, and a
# regex that got it subtly wrong would silently extract too few names — the
# exact failure this script exists to prevent, committed by the script itself.
python3 - "$TEST" $BACKENDS <<'PY'
import re, sys

test_path, backends = sys.argv[1], sys.argv[2:]

covered = set(re.findall(r'^\s*\("([^"]+)",\s*\d+,\s*\[',
                         open(test_path, encoding="utf-8").read(), re.M))
if not covered:
    print("::error::no rows parsed out of %s. Either the matrix is empty or its "
          "shape changed and this check is now reading nothing." % test_path)
    sys.exit(1)

missing, scanned = [], 0
for be in backends:
    path = "sarek/codegen/Sarek_ir_%s.ml" % be
    src = open(path, encoding="utf-8").read()
    i = src.find("arm =")
    j = src.find("| _ -> None)", i)
    if i < 0 or j < 0:
        print("::error::could not locate the `arm` table in %s (looked for "
              "'arm =' then '| _ -> None)'). Refusing to report success on a "
              "backend this script could not read." % path)
        sys.exit(1)
    names = []
    for group in re.findall(r'\|\s*("(?:[^"\\]|\\.)*"(?:\s*\|\s*"(?:[^"\\]|\\.)*")*)\s*->',
                            src[i:j]):
        names += re.findall(r'"((?:[^"\\]|\\.)*)"', group)
    if not names:
        print("::error::%s: extracted zero arm names. The table cannot be "
              "empty; the parser is broken." % path)
        sys.exit(1)
    scanned += 1
    for n in sorted(set(names)):
        if n not in covered:
            missing.append((be, n))

if scanned != len(backends):
    print("::error::scanned %d of %d backends." % (scanned, len(backends)))
    sys.exit(1)

if missing:
    print("::error::intrinsic arm(s) not covered by the arm-parity matrix in "
          "%s:" % test_path)
    for be, n in missing:
        print("    %-8s %s" % (be, n))
    print("Add a row for each. Until then nothing checks whether the other four "
          "backends handle these names, which is the divergence this matrix "
          "exists to make deliberate.")
    sys.exit(1)

print("OK: %d backends scanned, every arm name covered by %d matrix rows."
      % (scanned, len(covered)))
PY
