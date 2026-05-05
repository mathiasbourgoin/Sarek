#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Run unit tests with code coverage measurement

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Clean previous coverage data
rm -f _coverage/*.coverage 2>/dev/null || true
mkdir -p _coverage

# Try to install bisect_ppx if not already installed
echo "Checking for bisect_ppx..."
if ! opam list bisect_ppx --installed --short 2>/dev/null; then
  echo "Installing bisect_ppx..."
  opam repo add alpha git+https://github.com/kit-ty-kate/opam-alpha-repository.git || true
  opam update
  opam install -y bisect_ppx || {
    echo "Warning: Failed to install bisect_ppx, coverage will be skipped"
    exit 0
  }
fi

echo "Building with coverage instrumentation..."
LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH \
  OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  dune build --force --instrument-with bisect_ppx \
              sarek/tests/unit/ \
              sarek/core/test/ \
              spoc/framework/test/ \
              spoc/ir/test/ \
              spoc/registry/test/

echo ""
echo "Running unit tests..."
export BISECT_FILE="$PROJECT_ROOT/_coverage/unit"

# Run all sarek unit tests
for test in _build/default/sarek/tests/unit/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

# Run core tests
for test in _build/default/sarek/core/test/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

# Run spoc tests
for test in _build/default/spoc/*/test/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

echo ""
echo "Generating coverage report..."
OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report html \
    --coverage-path _coverage \
    -o _coverage/unit-report

OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report summary \
    --coverage-path _coverage

echo ""
echo "Coverage report generated at: _coverage/unit-report/index.html"
