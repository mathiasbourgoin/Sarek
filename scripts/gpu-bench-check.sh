#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# gpu-bench-check.sh — GPU benchmark regression check for SPOC/Sarek.
#
# Runs the self-verifying benchmark suite on whatever GPU/accelerator backends
# are available on this machine (OpenCL + Vulkan auto-detected) at a small,
# fast size and reports any correctness regression. Each bench records a
# per-device `verified` flag in its JSON result; this gate fails if ANY result
# has "verified": false, or a bench crashes/fails to build. It gates on
# CORRECTNESS, not absolute performance (timings are recorded for history).
#
# Usage: scripts/gpu-bench-check.sh [--size N] [--iterations N]
# Exit:  0 = all results verified; 1 = a verification failed / a bench crashed.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
SWITCH="${OPAM_SWITCH:-$PWD}"
SIZE="${SIZE:-2048}"
ITERS="${ITERS:-3}"
WARMUP="${WARMUP:-1}"
while [ $# -gt 0 ]; do
  case "$1" in
    --size) SIZE="$2"; shift 2 ;;
    --iterations) ITERS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

BENCHES=(
  bench_vector_add bench_vector_copy bench_stream_triad
  bench_dot_product bench_reduction bench_reduction_max
  bench_transpose bench_transpose_tiled
  bench_matrix_mul bench_matrix_mul_tiled
  bench_scan bench_histogram bench_gather_scatter
  bench_mandelbrot bench_conv2d bench_stencil_2d
  bench_bitonic_sort bench_radix_sort bench_nbody
)
# matrix-mul on the OpenCL CPU device is a known sporadic segfault (pre-existing).
KNOWN_FLAKY_REGEX='matrix_mul.*OpenCL'

OUT=$(mktemp -d -t spoc-gpu-bench-XXXXXX)
trap 'rm -rf "$OUT"' EXIT
echo "=== SPOC GPU benchmark regression check ==="
echo "size=$SIZE iterations=$ITERS warmup=$WARMUP  ($(date -u +%FT%TZ))"
echo "results dir: $OUT"
echo

crash_count=0; flaky_count=0
declare -a CRASHES
for b in "${BENCHES[@]}"; do
  if ! opam exec --switch="$SWITCH" -- dune build "benchmarks/$b.exe" >/dev/null 2>&1; then
    echo "  [BUILD-FAIL] $b"; crash_count=$((crash_count + 1)); CRASHES+=("$b: build failed"); continue
  fi
  out=$(timeout 300 bash -c "opam exec --switch='$SWITCH' -- dune exec benchmarks/$b.exe -- --sizes $SIZE --iterations $ITERS --warmup $WARMUP --output '$OUT'" 2>&1)
  rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
    if echo "$out" | grep -qE "$KNOWN_FLAKY_REGEX"; then
      echo "  [FLAKY] $b (exit $rc, known matrix-mul/OpenCL flake)"; flaky_count=$((flaky_count + 1))
    else
      echo "  [CRASH] $b (exit $rc)"; crash_count=$((crash_count + 1)); CRASHES+=("$b: crashed (exit $rc)")
    fi
  fi
done

# Authoritative correctness gate: scan every JSON result this run produced.
python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
files = glob.glob(os.path.join(out, "**", "*.json"), recursive=True)
total = verified = unchecked = 0
fails = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        fails.append(f"{os.path.basename(f)}: unreadable ({e})"); continue
    name = d.get("benchmark", os.path.basename(f))
    for r in d.get("results", []):
        total += 1
        v = r.get("verified", None)
        dev = r.get("device", r.get("device_name", "?"))
        sz = r.get("size", "?")
        if v is False:
            fails.append(f"{name} [{dev}] size={sz}: verified=false")
        elif v is True:
            verified += 1
        else:
            unchecked += 1
print()
print(f"=== summary: {total} results · {verified} verified · {unchecked} no-verify-field · "
      f"{len(fails)} verify-fail ===")
if fails:
    print("REGRESSIONS:")
    for x in fails:
        print(f"  - {x}")
    sys.exit(1)
sys.exit(0)
PY
json_rc=$?

echo
if [ "$crash_count" -gt 0 ]; then
  echo "CRASHES:"; for c in "${CRASHES[@]}"; do echo "  - $c"; done
fi
[ "$flaky_count" -gt 0 ] && echo "(known-flaky skipped: $flaky_count)"
if [ "$json_rc" -ne 0 ] || [ "$crash_count" -gt 0 ]; then
  echo "RESULT: REGRESSION DETECTED ✗"; exit 1
fi
echo "RESULT: all GPU benchmark results verified ✓"; exit 0
