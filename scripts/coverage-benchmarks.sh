#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Run benchmarks with code coverage measurement

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
              sarek/tests/e2e/test_matrix_mul.exe \
              sarek/tests/e2e/test_transpose.exe \
              sarek/tests/e2e/test_scan.exe \
              sarek/tests/e2e/test_reduce.exe \
              sarek/tests/e2e/test_histogram.exe \
              sarek/tests/e2e/test_mandelbrot.exe

echo ""
echo "Running benchmarks (native backend only for consistent coverage)..."
export BISECT_FILE="$PROJECT_ROOT/_coverage/benchmarks"
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

# Run tests with small size and native backend (no --benchmark to avoid running all devices)
_build/default/sarek/tests/e2e/test_matrix_mul.exe --native -s 128
_build/default/sarek/tests/e2e/test_transpose.exe --native -s 256
_build/default/sarek/tests/e2e/test_scan.exe --native -s 1024
_build/default/sarek/tests/e2e/test_reduce.exe --native -s 1024
_build/default/sarek/tests/e2e/test_histogram.exe --native -s 1024
_build/default/sarek/tests/e2e/test_mandelbrot.exe --native -s 512

echo ""
echo "Generating coverage report..."
OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report html \
    --coverage-path _coverage \
    -o _coverage/benchmarks-report

OPAM_SWITCH_PREFIX="$PROJECT_ROOT/_opam" \
  PATH="$PROJECT_ROOT/_opam/bin:$PATH" \
  bisect-ppx-report summary \
    --coverage-path _coverage

echo ""
echo "Coverage report generated at: _coverage/benchmarks-report/index.html"
