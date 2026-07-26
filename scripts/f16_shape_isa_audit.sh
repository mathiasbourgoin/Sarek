#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# ISA half of the f16 expression-shape audit (issue #106).
#
# test_hip_f16_shapes.exe answers "does this shape DISAGREE with the
# interpreter". That is necessary but not sufficient: a shape can be demoted by
# the AMDGPU ISel combine and still agree on every input, because the demoted
# computation happens to be exact on the whole binary16 domain (A13 `x *. x` is
# the clean example -- a product of two binary16 values needs at most 22
# significant bits, so it is exact in binary32 and the demoted binary16
# multiply rounds identically). A numeric null therefore cannot distinguish
# "not demoted" from "demoted, harmless here" -- and the second is a latent
# hazard, because the next expression shape may not be so lucky.
#
# This script settles that by disassembling each shape for gfx1100 with and
# without the barrier and reporting which demotion opcodes appear.
#
# Usage:  scripts/f16_shape_isa_audit.sh [outdir]

set -euo pipefail

OUT="${1:-${TMPDIR:-/tmp}/sarek-f16-isa}"
ARCH="${SAREK_F16_ARCH:-gfx1100}"
ROCM="${ROCM_PATH:-/opt/rocm}"
HIPCC="$ROCM/bin/hipcc"

if [ ! -x "$HIPCC" ]; then
  echo "hipcc not found at $HIPCC (set ROCM_PATH); skipping" >&2
  exit 0
fi

mkdir -p "$OUT"

# 1. Dump the generated HIP source for every shape, with and without the
#    barrier. The test itself does the substitution, so what is compiled below
#    is byte-for-byte what the runtime compiles.
echo "==> dumping generated sources to $OUT"
SAREK_F16_DUMP="$OUT" dune exec sarek-hip/test/test_hip_f16_shapes.exe >/dev/null

# 2. Disassemble each. hiprtc implicitly includes the HIP runtime and fp16
#    headers; hipcc does not, hence the two -include flags. -ffp-contract=off
#    mirrors Hip_rtc.base_options -- the point of the exercise is that it does
#    NOT prevent these combines.
echo "==> compiling for $ARCH"
for f in "$OUT"/*.hip; do
  b="${f%.hip}"
  "$HIPCC" -x hip -I"$ROCM/include" \
    -include hip/hip_runtime.h -include hip/hip_fp16.h \
    --offload-arch="$ARCH" -O3 -ffp-contract=off \
    -S -o "$b.s" "$f" 2>/dev/null || echo "  ! failed: $(basename "$f")" >&2
done

# 3. Report. Any of these opcodes means an f32 operation was folded into, or
#    demoted to, binary16:
#      v_fma_mixlo_f16  an f32 multiply/fma fused into the f32->f16 narrowing
#      v_add_f16        an f32 add/sub demoted to a binary16 add
#      v_sub_f16        an f32 subtract/negate demoted to binary16
#      v_mul_f16        an f32 multiply demoted to a binary16 multiply
#    v_fma_mix_f32 is NOT in the list: it keeps an f32 result and is how an
#    explicitly requested fma() is meant to be emitted.
DEMOTE='v_fma_mixlo_f16|v_mad_mixlo_f16|v_add_f16|v_sub_f16|v_mul_f16'

printf '\n%-6s %-34s %s\n' "shape" "with barrier" "barrier removed"
printf '%-6s %-34s %s\n' "-----" "----------------------------------" \
  "----------------------------------"
for f in "$OUT"/*_barrier.s; do
  case "$f" in *_nobarrier.s) continue ;; esac
  id="$(basename "$f" _barrier.s)"
  nb="$OUT/${id}_nobarrier.s"
  [ -f "$nb" ] || continue
  # `grep` exits 1 when a shape has no demotion at all -- which is the
  # RESULT, not an error. Without the `|| true` the pipeline fails under
  # `set -o pipefail` and the whole report silently comes out empty.
  fmt() {
    { grep -oE "$DEMOTE" "$1" || true; } | sort | uniq -c \
      | awk '{printf "%sx%s ", $1, $2}'
  }
  a="$(fmt "$f")"; b="$(fmt "$nb")"
  printf '%-6s %-34s %s\n' "$id" "${a:-(none)}" "${b:-(none)}"
done

echo
echo "A non-empty 'with barrier' cell is a REGRESSION: the barrier is supposed"
echo "to leave every narrowing as a standalone v_cvt_f16_f32."
