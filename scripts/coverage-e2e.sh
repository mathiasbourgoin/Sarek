#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Run e2e tests with code coverage measurement

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Clean previous coverage data
rm -f _coverage/*.coverage 2>/dev/null || true
mkdir -p _coverage

echo "Building with coverage instrumentation..."
LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH \
  OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  dune build --force --instrument-with bisect_ppx \
              sarek/tests/e2e/test_vector_add.exe \
              sarek/tests/e2e/test_reduce.exe \
              sarek/tests/e2e/test_complex_types.exe \
              sarek/tests/e2e/test_histogram.exe \
              sarek/tests/e2e/test_matrix_mul.exe \
              sarek/tests/e2e/test_transpose.exe \
              sarek/tests/e2e/test_scan.exe \
              sarek/tests/e2e/test_mandelbrot.exe \
              sarek/tests/e2e/test_polymorphism.exe

echo ""
echo "Running e2e tests..."
export BISECT_FILE="$PROJECT_ROOT/_coverage/e2e"
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

# Run tests with native backend (for consistent results)
_build/default/sarek/tests/e2e/test_vector_add.exe --native
_build/default/sarek/tests/e2e/test_reduce.exe --native
_build/default/sarek/tests/e2e/test_complex_types.exe --native
_build/default/sarek/tests/e2e/test_histogram.exe --native
_build/default/sarek/tests/e2e/test_matrix_mul.exe --native
_build/default/sarek/tests/e2e/test_transpose.exe --native
_build/default/sarek/tests/e2e/test_scan.exe --native
_build/default/sarek/tests/e2e/test_mandelbrot.exe --native
_build/default/sarek/tests/e2e/test_polymorphism.exe --native

echo ""
echo "Generating coverage report..."
OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report html \
    --coverage-path _coverage \
    -o _coverage/e2e-report

OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report summary \
    --coverage-path _coverage

echo ""
echo "Coverage report generated at: _coverage/e2e-report/index.html"
