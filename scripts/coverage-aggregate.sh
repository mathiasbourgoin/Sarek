#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Generate aggregate coverage report from all test types

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Clean all coverage data at start
rm -rf _coverage
mkdir -p _coverage

echo "=== Building with coverage instrumentation ==="
LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH \
  OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  dune build --force --instrument-with bisect_ppx \
              sarek/tests/unit/ \
              sarek/core/test/ \
              sarek/framework/test/ \
              spoc/framework/test/ \
              spoc/ir/test/ \
              spoc/registry/test/ \
              sarek/tests/e2e/test_vector_add.exe \
              sarek/tests/e2e/test_reduce.exe \
              sarek/tests/e2e/test_complex_types.exe \
              sarek/tests/e2e/test_histogram.exe \
              sarek/tests/e2e/test_matrix_mul.exe \
              sarek/tests/e2e/test_transpose.exe \
              sarek/tests/e2e/test_scan.exe \
              sarek/tests/e2e/test_mandelbrot.exe \
              sarek/tests/e2e/test_polymorphism.exe

export BISECT_FILE="$PROJECT_ROOT/_coverage/aggregate"
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

echo ""
echo "=== Running Unit Tests ==="
for test in _build/default/sarek/tests/unit/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

echo ""
echo "=== Running Core Tests ==="
for test in _build/default/sarek/core/test/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

echo ""
echo "=== Running Framework Tests ==="
for test in _build/default/sarek/framework/test/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

echo ""
echo "=== Running SPOC Tests ==="
for test in _build/default/spoc/*/test/test_*.exe; do
  echo "Running $(basename $test)..."
  $test
done

echo ""
echo "=== Running E2E Tests ==="
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
echo "=== Running Benchmarks ==="
_build/default/sarek/tests/e2e/test_matrix_mul.exe --native -s 128
_build/default/sarek/tests/e2e/test_transpose.exe --native -s 256
_build/default/sarek/tests/e2e/test_scan.exe --native -s 1024
_build/default/sarek/tests/e2e/test_reduce.exe --native -s 1024
_build/default/sarek/tests/e2e/test_histogram.exe --native -s 1024
_build/default/sarek/tests/e2e/test_mandelbrot.exe --native -s 512

echo ""
echo "=== Generating Coverage Reports ==="
OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report html \
    --coverage-path _coverage \
    -o _coverage/aggregate-report

OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report summary \
    --coverage-path _coverage

echo ""
echo "===================================================================="
echo "Aggregate coverage report generated at:"
echo "  _coverage/aggregate-report/index.html"
echo ""
echo "For individual test suite reports, run:"
echo "  ./scripts/coverage-unit.sh       (unit tests only)"
echo "  ./scripts/coverage-e2e.sh        (e2e tests only)"
echo "  ./scripts/coverage-benchmarks.sh (benchmarks only)"
echo "===================================================================="
