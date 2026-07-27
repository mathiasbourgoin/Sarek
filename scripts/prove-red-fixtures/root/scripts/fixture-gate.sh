#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# The subject scripts/prove-red.sh proves ITSELF against.
#
# It is a real gate in miniature: it reads a declared input, refuses when the
# input is absent (exit 2, a broken world), and fails when the input is wrong
# (exit 1, a real violation). It is committed rather than synthesised so that
# prove-red.sh's own spec block can break it in the four ways prove-red.sh can
# lie -- see that block. Nothing else in the repository reads it, and it is not
# discovered by prove-red.sh's own scan: the scan walks `scripts/` and `ci/`
# non-recursively, and this lives one level down.
#
# Exit codes:
#   0  the input carries the marker
#   1  the input does not carry the marker
#   2  the input is not there at all
# ---------------------------------------------------------------------------
set -euo pipefail

IN="data/input.txt"

if [ ! -f "$IN" ]; then
  echo "::error::$IN not found -- there is nothing to check and a pass would" \
       "mean nothing."
  exit 2
fi

if ! grep -q 'MARKER-OK' "$IN"; then
  echo "::error::$IN does not carry MARKER-OK."
  exit 1
fi

echo "OK: input.txt is well-formed"

# ---------------------------------------------------------------------------
# BEGIN prove-red-spec
# copy: scripts/fixture-gate.sh
# copy: data/input.txt
# invoke: scripts/fixture-gate.sh
# baseline-exit: 0
# baseline-message: OK: input.txt is well-formed
#
# mutation: marker-removed
#   desc: the input is present but wrong -- a real violation, exit 1.
#   apply: printf 'nothing here\n' > data/input.txt
#   expect-exit: 1
#   expect-message: does not carry MARKER-OK
#
# mutation: input-deleted
#   desc: the declared input is gone -- an environment mutation, not a source edit; the shape that made add-license-headers.sh report success about a tree it had never read.
#   apply: rm -f data/input.txt
#   expect-exit: 2
#   expect-message: not found
# END prove-red-spec
# ---------------------------------------------------------------------------
