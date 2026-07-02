#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Guard against a regression of the `make opam` target mutating sarek.opam
# (it used to append a bogus `available: [ os = "linux" ]` line on every
# run, and also tried to append to a non-existent sarek_ppx.opam). This
# script asserts:
#   - `make opam` can be run twice in a row without error
#   - sarek.opam gains no `available:` line
#   - sarek.opam gains no duplicated trailing line
#   - sarek.opam is left byte-identical to the tracked version (git diff clean)
#   - the Makefile no longer references sarek_ppx.opam (that package does
#     not exist in this repo)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

fail() {
    echo -e "${RED}FAIL: $1${NC}"
    exit 1
}

echo "Running 'make opam' twice to check for corruption/non-idempotence..."
make opam
make opam

if grep -q 'available:' sarek.opam; then
    fail "sarek.opam contains an 'available:' line after 'make opam'"
fi

# Duplicate trailing line check: last line must not equal the second-to-last
# non-empty line (guards against the old double-append corruption).
last_line="$(tail -n 1 sarek.opam)"
second_last_line="$(tail -n 2 sarek.opam | head -n 1)"
if [ -n "$last_line" ] && [ "$last_line" = "$second_last_line" ]; then
    fail "sarek.opam has a duplicated trailing line"
fi

if ! git diff --exit-code sarek.opam > /dev/null; then
    fail "sarek.opam differs from the tracked version after 'make opam'"
fi

if grep -q 'sarek_ppx.opam' Makefile; then
    fail "Makefile still references the non-existent sarek_ppx.opam"
fi

if [ -f sarek_ppx.opam ]; then
    fail "sarek_ppx.opam was created by 'make opam' but should not exist"
fi

echo -e "${GREEN}PASS: 'make opam' is clean and idempotent${NC}"
