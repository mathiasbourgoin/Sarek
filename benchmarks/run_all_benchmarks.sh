#!/bin/bash
# Run all benchmarks and update web data
# Usage: ./run_all_benchmarks.sh [output_dir] [--generate-backend-code] [--no-clean]

set -e  # Exit on error during setup phase only

# Show help
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  cat <<EOF
SAREK Benchmark Suite Runner

Usage: $0 [output_dir] [--generate-backend-code] [--no-clean]

Runs all benchmarks and updates web viewer data.

Arguments:
  output_dir              Base directory for results (default: benchmarks/results)
                          Results saved to output_dir/run_TIMESTAMP/
  --generate-backend-code Also regenerate backend code for all benchmarks
  --no-clean              Do not remove old results from this hostname before saving

Examples:
  $0                              # Save to benchmarks/results/run_TIMESTAMP/
  $0 my_results                   # Save to my_results/run_TIMESTAMP/
  $0 --generate-backend-code      # Also regenerate CUDA/OpenCL/Vulkan/Metal code
  make benchmarks                 # Same as running script directly

After running:
  1. Old results from this hostname are automatically replaced (use --no-clean to keep them)
  2. Review new results in benchmarks/results/
  3. Commit updated benchmarks/results/*.json files in PR
  4. Optionally run 'make bench-deduplicate' to check for cross-host duplicates
EOF
  exit 0
fi

# Parse arguments
OUTPUT_DIR="benchmarks/results"
GENERATE_CODE=false
CLEAN_OLD=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --generate-backend-code)
      GENERATE_CODE=true
      shift
      ;;
    --no-clean)
      CLEAN_OLD=false
      shift
      ;;
    *)
      if [[ ! "$1" == --* ]]; then
        OUTPUT_DIR="$1"
        shift
      fi
      ;;
  esac
done
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"

echo "================================================"
echo "  SAREK Benchmark Suite Runner"
echo "================================================"
echo "Output directory: ${RUN_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Create output directory
mkdir -p "${RUN_DIR}"

echo "Building all benchmarks..."
dune build \
  benchmarks/bench_matrix_mul.exe \
  benchmarks/bench_matrix_mul_tiled.exe \
  benchmarks/bench_vector_add.exe \
  benchmarks/bench_vector_copy.exe \
  benchmarks/bench_stream_triad.exe \
  benchmarks/bench_reduction.exe \
  benchmarks/bench_reduction_max.exe \
  benchmarks/bench_dot_product.exe \
  benchmarks/bench_transpose.exe \
  benchmarks/bench_transpose_tiled.exe \
  benchmarks/bench_mandelbrot.exe \
  benchmarks/bench_scan.exe \
  benchmarks/bench_bitonic_sort.exe \
  benchmarks/bench_histogram.exe \
  benchmarks/bench_gather_scatter.exe \
  benchmarks/bench_radix_sort.exe \
  benchmarks/bench_nbody.exe \
  benchmarks/bench_conv2d.exe \
  benchmarks/bench_stencil_2d.exe \
  benchmarks/to_web.exe

echo ""
echo "Running benchmarks..."
echo "This may take several minutes depending on your hardware."
echo ""

# Disable exit-on-error for benchmark runs (continue on failure)
set +e

# Track failed benchmarks
FAILED_BENCHMARKS=()
SUCCESSFUL_BENCHMARKS=()

# Helper function to run a benchmark
run_benchmark() {
  local name="$1"
  local exe="$2"
  
  echo "▶ ${name}..."
  if dune exec "benchmarks/${exe}.exe" -- --output "${RUN_DIR}"; then
    echo "  ✓ Complete"
    SUCCESSFUL_BENCHMARKS+=("${name}")
  else
    echo "  ✗ FAILED"
    FAILED_BENCHMARKS+=("${name}")
  fi
  echo ""
}

# Run all benchmarks
run_benchmark "Matrix Multiplication (naive)" "bench_matrix_mul"
run_benchmark "Matrix Multiplication (tiled)" "bench_matrix_mul_tiled"
run_benchmark "Vector Addition" "bench_vector_add"
run_benchmark "Vector Copy" "bench_vector_copy"
run_benchmark "STREAM Triad" "bench_stream_triad"
run_benchmark "Parallel Reduction (sum)" "bench_reduction"
run_benchmark "Parallel Reduction (max)" "bench_reduction_max"
run_benchmark "Dot Product" "bench_dot_product"
run_benchmark "Matrix Transpose (naive)" "bench_transpose"
run_benchmark "Matrix Transpose (tiled)" "bench_transpose_tiled"
run_benchmark "Mandelbrot Set" "bench_mandelbrot"
run_benchmark "N-Body" "bench_nbody"
run_benchmark "2D Convolution" "bench_conv2d"
run_benchmark "2D Stencil (Jacobi)" "bench_stencil_2d"
run_benchmark "Prefix Sum (Scan)" "bench_scan"
run_benchmark "Bitonic Sort" "bench_bitonic_sort"
run_benchmark "Histogram (256 bins)" "bench_histogram"
run_benchmark "Gather/Scatter" "bench_gather_scatter"
run_benchmark "Radix Sort" "bench_radix_sort"

# Re-enable exit-on-error
set -e

# Count results
RESULT_COUNT=$(ls -1 "${RUN_DIR}"/*.json 2>/dev/null | wc -l)
TOTAL_BENCHMARKS=$((${#SUCCESSFUL_BENCHMARKS[@]} + ${#FAILED_BENCHMARKS[@]}))

echo "================================================"
echo "Benchmark Summary"
echo "================================================"
echo "Total benchmarks: ${TOTAL_BENCHMARKS}"
echo "Successful: ${#SUCCESSFUL_BENCHMARKS[@]}"
echo "Failed: ${#FAILED_BENCHMARKS[@]}"
echo ""

if [ ${#FAILED_BENCHMARKS[@]} -gt 0 ]; then
  echo "⚠️  Failed benchmarks:"
  for bench in "${FAILED_BENCHMARKS[@]}"; do
    echo "  - ${bench}"
  done
  echo ""
  echo "Note: Failed benchmarks will not be included in results."
  echo "Please include this failure list in your PR description."
  echo ""
fi

echo "Generated ${RESULT_COUNT} result files in ${RUN_DIR}"
echo ""

# Move results to benchmarks/results/ for git tracking and CI
echo "Moving results to benchmarks/results/ for git tracking..."
if [ ${RESULT_COUNT} -gt 0 ]; then
  # Remove old results from this hostname to prevent duplicates
  if [ "$CLEAN_OLD" = true ]; then
    HOSTNAME=$(hostname)
    OLD_COUNT=$(ls -1 benchmarks/results/${HOSTNAME}_*.json 2>/dev/null | wc -l)
    if [ ${OLD_COUNT} -gt 0 ]; then
      rm -f benchmarks/results/${HOSTNAME}_*.json
      echo "  ✓ Removed ${OLD_COUNT} old results from ${HOSTNAME}"
    fi
  fi

  mv "${RUN_DIR}"/*.json "benchmarks/results/"
  echo "  ✓ Moved ${RESULT_COUNT} new results to benchmarks/results/"
  
  # Remove empty run directory
  rmdir "${RUN_DIR}" 2>/dev/null || rm -rf "${RUN_DIR}"
  echo "  ✓ Cleaned up ${RUN_DIR}"
else
  echo "  ⚠ No results found to move"
fi
echo ""

# Update web data
echo "Updating web viewer data..."
WEB_OUTPUT="gh-pages/benchmarks/data/latest.json"
dune exec benchmarks/to_web.exe -- "${WEB_OUTPUT}" benchmarks/results/*.json

echo "  ✓ Updated ${WEB_OUTPUT}"
echo ""

# Generate backend code if requested
if [ "$GENERATE_CODE" = true ]; then
  echo "================================================"
  echo "Regenerating backend code for all benchmarks..."
  dune build benchmarks/generate_backend_code.exe
  dune exec benchmarks/generate_backend_code.exe
  
  # Copy to gh-pages
  echo "Copying generated code to gh-pages..."
  mkdir -p gh-pages/benchmarks/descriptions/generated
  cp benchmarks/descriptions/generated/*.md gh-pages/benchmarks/descriptions/generated/
  
  echo "  ✓ Backend code regenerated"
  echo ""
fi

echo "================================================"
echo "✅ Benchmark run complete!"
echo ""
echo "Results are ready in benchmarks/results/"
echo "Web viewer data updated: ${WEB_OUTPUT}"
echo ""
echo "Next steps:"
echo "  1. (Optional) Check for duplicates:"
echo "     make bench-deduplicate"
echo ""
echo "  2. Add results to git:"
echo "     git add benchmarks/results/*.json"
echo "     git add ${WEB_OUTPUT}"
if [ "$GENERATE_CODE" = true ]; then
  echo "     git add benchmarks/descriptions/generated/"
  echo "     git add gh-pages/benchmarks/descriptions/generated/"
fi
echo ""
echo "  3. Commit and push:"
if [ ${#FAILED_BENCHMARKS[@]} -gt 0 ]; then
  echo "     git commit -m \"Add benchmark results from $(hostname) ($(date +%Y-%m-%d))"
  echo ""
  echo "Failed benchmarks (excluded from results):"
  for bench in "${FAILED_BENCHMARKS[@]}"; do
    echo "- ${bench}"
  done
  echo "\""
else
  echo "     git commit -m \"Add benchmark results from $(hostname) ($(date +%Y-%m-%d))\""
fi
echo "     git push origin <your-branch>"
echo ""
echo "  4. Create PR - CI will generate preview at:"
echo "     https://mathiasbourgoin.github.io/Sarek/preview/pr-<number>/benchmarks/"
echo ""
echo "Reviewers can deduplicate before merging with:"
echo "  dune exec benchmarks/deduplicate_results.exe"
echo "================================================"
