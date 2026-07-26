#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# #57 slice 2b — standalone reproducer for the RADV f16-narrowing fusion.
#
# WHAT THIS IS FOR, and how it differs from the test next to it.
#
#   sarek-vulkan/test/test_vulkan_f16_tripwire.ml
#       is the GATE. It runs the shaders on a device and fails when the fusion
#       stops, so the refusal in Sarek_ir_glsl cannot outlive its reason.
#
#   this script
#       is the DOCUMENTED REPRODUCER, and it exists because the gate cannot
#       cover one half of the claim in docs/fp-contraction-policy.md §6. That
#       claim is "the decoration is emitted AND the driver ignores it". The gate
#       measures the ignoring. Nothing measures the emitting for THIS shape — if
#       glslang ever stopped decorating, "ignored" would quietly become "never
#       asked", the gate would still pass, and §6 would be wrong in a way no
#       test could see.
#
# It needs no Vulkan device and no OCaml toolchain: glslangValidator and
# spirv-dis only. With a RADV device present, pass --asm to also dump the ISA.
#
# Measured 2026-07-26, glslangValidator + SPIRV-Tools from the Vulkan SDK on
# Mesa 26.1.4-arch3.1:
#
#   plain   : 0 NoContraction
#   precise : 1 NoContraction, on the OpFMul
#
# and the RADV ISA is byte-identical between the two, both fusing via
# v_fma_mixlo_f16 — see docs/fp-contraction-policy.md §6.
set -euo pipefail

need() { command -v "$1" >/dev/null || { echo "missing: $1"; exit 2; }; }
need glslangValidator
need spirv-dis

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

emit() { # $1 = "" | "precise "
  cat <<EOF
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 256) in;
layout(std430, binding = 0) volatile buffer Out { uint outb[]; };
layout(std430, binding = 1) readonly buffer In { uint inb[]; };
uint pack(float16_t r) {
  return packFloat2x16(f16vec2(r, float16_t(0.0))) & 0xFFFFu;
}
void main() {
  uint i = gl_GlobalInvocationID.x;
  float x = float(unpackFloat2x16(inb[i]).x);
  ${1}float p = x * 1.1;
  outb[i] = pack(float16_t(p));
}
EOF
}

status=0
for variant in plain precise; do
  qual=""
  [ "$variant" = precise ] && qual="precise "
  emit "$qual" > "$work/$variant.comp"
  glslangValidator -V -S comp "$work/$variant.comp" -o "$work/$variant.spv" \
    >/dev/null
  n=$(spirv-dis "$work/$variant.spv" | grep -c NoContraction || true)
  echo "$variant: $n NoContraction decoration(s)"

  # The expectations are asserted, not printed: an unasserted probe that drifts
  # is how a documented reproducer stops reproducing anything.
  case "$variant" in
    plain)
      [ "$n" -eq 0 ] || {
        echo "  UNEXPECTED: the undecorated variant should carry none"
        status=1
      } ;;
    precise)
      [ "$n" -eq 1 ] || {
        echo "  UNEXPECTED: glslang no longer decorates the OpFMul for this"
        echo "  shape. docs/fp-contraction-policy.md §6 says RADV IGNORES a"
        echo "  decoration that IS emitted. If it is no longer emitted, that"
        echo "  sentence is wrong and must be rewritten — do not just adjust"
        echo "  this number."
        status=1
      } ;;
  esac
done

spirv-dis "$work/precise.spv" | grep -E "NoContraction|OpFMul|OpFConvert" || true

if [ "${1:-}" = "--asm" ]; then
  echo
  echo "ISA (needs a RADV device; both variants are expected to be identical,"
  echo "and both to contain v_fma_mixlo_f16):"
  echo "  RADV_DEBUG=asm dune exec sarek-vulkan/test/test_vulkan_f16_tripwire.exe"
fi

exit "$status"
