#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Assert that the codegen-validation toolchain is present and runnable.
#
# WHY THIS EXISTS
#
# Every codegen gate in this repository self-skips when its tool is absent:
#
#   sarek/tests/unit/test_ptx_snapshot.ml         -> ptxas
#   sarek/tests/codegen_golden/test_codegen_golden.ml
#     glsl_validation_sweep                       -> glslangValidator
#     wgsl_validation_sweep                       -> naga
#   sarek/tests/unit/test_shader_recursion_vector.ml -> glslangValidator + naga
#   sarek-cuda/test/test_cuda_nvrtc_gate.ml       -> libnvrtc
#
# That skip is correct behaviour for a developer machine, which legitimately has
# none of these installed. It is a disaster in CI: the previous image carried
# none of them, so every gate printed "SKIP" and reported success. CI was green
# and had validated nothing. That is exactly how the f16 work shipped a
# generated `#include <cuda_fp16.h>` that NVRTC cannot resolve — caught only
# when a reviewer pasted the generated source into a real NVRTC by hand.
#
# The gates therefore KEEP their skip logic, and this script is the thing that
# makes a missing tool a CI FAILURE rather than a silent pass. It must run early
# and fail fast, before any test step: without it, one bad edit to ci/Dockerfile
# silently returns the whole suite to green-but-vacuous with nobody noticing.
#
# Each check runs the tool, not merely `command -v` it: a present-but-broken
# binary (wrong ABI, missing shared library) skips just as silently as an
# absent one.
# ---------------------------------------------------------------------------
set -uo pipefail

failures=0
checks=0

fail() {
  echo "::error::$*"
  failures=$((failures + 1))
}

ok() { echo "  OK  $*"; }

section() { echo; echo "== $1"; }

run_check() {
  # run_check <label> <command...>
  local label="$1"
  shift
  checks=$((checks + 1))
  local out
  if out=$("$@" 2>&1); then
    ok "$label — $(printf '%s' "$out" | head -1)"
  else
    fail "$label: '$*' failed (exit $?):
$out"
  fi
}

# --------------------------------------------------------------------------
section "ptxas (PTX assembler — host-side, no GPU required)"
if ! command -v ptxas >/dev/null 2>&1; then
  fail "ptxas is not on PATH. The PTX assembly gate in test_ptx_snapshot.ml \
will self-skip and CI will validate nothing. Install cuda-nvcc-12-6 (see \
ci/Dockerfile)."
else
  run_check "ptxas --version" ptxas --version
  # Positive control: assemble a minimal module. Proves ptxas can actually
  # produce a cubin here, not just print a version banner.
  tmpdir=$(mktemp -d)
  cat >"$tmpdir/probe.ptx" <<'PTX'
.version 7.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 p)
{
  .reg .u64 %rd<2>;
  ld.param.u64 %rd1, [p];
  ret;
}
PTX
  checks=$((checks + 1))
  # sm_75 rather than sm_70: CUDA 13 dropped Volta from ptxas's --gpu-name
  # list, and sm_75 is accepted by every 12.x and 13.x toolkit, so this probe
  # keeps working across a toolkit bump.
  if out=$(ptxas --compile-only --gpu-name sm_75 -o "$tmpdir/probe.cubin" \
             "$tmpdir/probe.ptx" 2>&1); then
    ok "ptxas assembled a probe module"
  else
    fail "ptxas is on PATH but cannot assemble a trivial module:
$out"
  fi
  rm -rf "$tmpdir"
fi

# --------------------------------------------------------------------------
section "NVRTC (CUDA-C -> PTX — host-side, no GPU required)"
# Sarek reaches NVRTC by dlopen (sarek-cuda/Cuda_nvrtc.ml tries libnvrtc.so,
# .so.12, .so.11), so assert exactly that: the library resolves AND dlopen
# succeeds AND nvrtcVersion is callable. `ls` alone would pass on a library
# whose own dependencies are missing.
nvrtc_probe=$(mktemp -d)
cat >"$nvrtc_probe/probe.c" <<'C'
#include <dlfcn.h>
#include <stdio.h>
int main(void) {
  const char *cands[] = {"libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11", 0};
  for (int i = 0; cands[i]; i++) {
    void *h = dlopen(cands[i], RTLD_LAZY);
    if (!h) continue;
    int (*ver)(int *, int *) = dlsym(h, "nvrtcVersion");
    if (!ver) { printf("%s: no nvrtcVersion symbol\n", cands[i]); return 1; }
    int maj = 0, min = 0;
    if (ver(&maj, &min) != 0) { printf("nvrtcVersion failed\n"); return 1; }
    printf("%s -> NVRTC %d.%d\n", cands[i], maj, min);
    return 0;
  }
  printf("no libnvrtc could be dlopen'd: %s\n", dlerror());
  return 1;
}
C
checks=$((checks + 1))
if ! cc -o "$nvrtc_probe/probe" "$nvrtc_probe/probe.c" -ldl 2>"$nvrtc_probe/cc.err"; then
  fail "could not build the NVRTC dlopen probe:
$(cat "$nvrtc_probe/cc.err")"
elif out=$("$nvrtc_probe/probe" 2>&1); then
  ok "NVRTC dlopen — $out"
else
  fail "libnvrtc is not loadable, so test_cuda_nvrtc_gate.ml will self-skip and \
the generated CUDA-C will not be compile-checked: $out
Install cuda-nvrtc-12-6 (see ci/Dockerfile)."
fi
rm -rf "$nvrtc_probe"

# CUDA headers: the f16 regression was an unresolvable #include, so assert the
# header tree the emitter may reference is actually on disk.
checks=$((checks + 1))
fp16=$(find "${CUDA_PATH:-/usr/local/cuda}" /usr/local/cuda -name cuda_fp16.h \
         2>/dev/null | head -1)
if [ -n "$fp16" ]; then
  ok "CUDA headers present ($fp16)"
else
  fail "cuda_fp16.h not found under \
${CUDA_PATH:-/usr/local/cuda}. The f16 regression this gate exists for was an \
#include NVRTC could not resolve, so the header tree must really be on disk. \
Install cuda-cudart-dev-12-6 (see ci/Dockerfile)."
fi

# --------------------------------------------------------------------------
section "glslangValidator (GLSL -> SPIR-V)"
if ! command -v glslangValidator >/dev/null 2>&1; then
  fail "glslangValidator is not on PATH. The GLSL shader-validation sweep will \
self-skip. Install glslang-tools (see ci/Dockerfile)."
else
  run_check "glslangValidator --version" glslangValidator --version
  tmpdir=$(mktemp -d)
  cat >"$tmpdir/probe.comp" <<'GLSL'
#version 450
layout(local_size_x = 1) in;
void main() {}
GLSL
  checks=$((checks + 1))
  if out=$(glslangValidator -V -S comp -o "$tmpdir/probe.spv" \
             "$tmpdir/probe.comp" 2>&1); then
    ok "glslangValidator compiled a probe shader"
  else
    fail "glslangValidator is on PATH but cannot compile a trivial compute \
shader:
$out"
  fi
  rm -rf "$tmpdir"
fi

# --------------------------------------------------------------------------
section "naga (WGSL validation)"
if ! command -v naga >/dev/null 2>&1; then
  fail "naga is not on PATH. WGSL has NO other executable validation anywhere \
in this repository, so the wgsl_validation_sweep would silently validate \
nothing. Built from naga-cli in the Dockerfile's builder stage."
else
  run_check "naga --version" naga --version
  tmpdir=$(mktemp -d)
  cat >"$tmpdir/probe.wgsl" <<'WGSL'
@compute @workgroup_size(1)
fn main() {}
WGSL
  checks=$((checks + 1))
  # A single positional argument with no output file makes naga run the full
  # front-end + validator. NOTE: `--validate all` does NOT work — the flag takes
  # a numeric ValidationFlags bitmask, and passing a keyword makes naga exit
  # non-zero during argument parsing. That mistake is what kept the WGSL sweep
  # from ever passing, so this probe deliberately pins the working invocation.
  if out=$(naga "$tmpdir/probe.wgsl" 2>&1); then
    ok "naga validated a probe shader — $out"
  else
    fail "naga is on PATH but cannot validate a trivial compute shader:
$out"
  fi
  rm -rf "$tmpdir"
fi

# --------------------------------------------------------------------------
section "OpenCL ICDs"
if ! command -v clinfo >/dev/null 2>&1; then
  fail "clinfo is not installed, so the OpenCL device inventory cannot be \
asserted. Install clinfo (see ci/Dockerfile)."
else
  clinfo_out=$(clinfo 2>&1 || true)
  echo "$clinfo_out" | grep -E '^\s*(Platform Name|Device Name|Device Type)' \
    | sed 's/^/    /' || true

  checks=$((checks + 1))
  if echo "$clinfo_out" | grep -qi 'portable computing language\|pocl'; then
    ok "pocl platform enumerates"
  else
    fail "no pocl platform in clinfo output. Task #74 had to classify the Intel \
oneAPI CPU runtime as a known-issue device, which leaves CI with zero \
trustworthy OpenCL coverage unless pocl is present and working. Install \
pocl-opencl-icd (see ci/Dockerfile)."
  fi

  # The known-issue suppression in sarek/tests/e2e/test_helpers.ml identifies
  # pocl by CL_DEVICE_NAME prefix ("pthread-" on pocl 1.x, "cpu-" from pocl 3.x
  # on). If pocl ever renames its CPU device, that predicate stops recognising
  # it and pocl failures would be silently EXCUSED as the Intel flake. Assert
  # the prefix here so a rename breaks the build loudly instead.
  checks=$((checks + 1))
  if echo "$clinfo_out" | grep -E '^\s*Device Name' \
       | grep -qE '(pthread-|cpu-)'; then
    ok "pocl CPU device name matches the prefixes test_helpers.ml keys on"
  else
    fail "no OpenCL device name starting with 'pthread-' or 'cpu-'. \
Test_helpers.is_pocl_device would no longer recognise the conformant device, \
and the CPU-OpenCL known-issue suppression (#74) would wrongly excuse real \
pocl failures. Update pocl_device_name_prefixes in \
sarek/tests/e2e/test_helpers.ml together with this check."
  fi

  # Deterministic single-ICD selection: each directory must resolve to exactly
  # one platform, so a CI step can pin one with OCL_ICD_VENDORS.
  for pair in "vendors-pocl:portable computing language" "vendors-intel:intel"; do
    dir="/etc/OpenCL/${pair%%:*}"
    want="${pair#*:}"
    checks=$((checks + 1))
    if [ ! -d "$dir" ]; then
      fail "$dir is missing; OCL_ICD_VENDORS cannot pin a single ICD."
    elif [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
      fail "$dir is empty; a package rename probably broke the ICD split in \
ci/Dockerfile."
    elif OCL_ICD_VENDORS="$dir" clinfo 2>&1 | grep -qi "$want"; then
      ok "OCL_ICD_VENDORS=$dir resolves to the expected platform ($want)"
    else
      fail "OCL_ICD_VENDORS=$dir did not enumerate a '$want' platform."
    fi
  done
fi

# --------------------------------------------------------------------------
section "Rocq / Coq proof checker"
# #45: the formal guarantee ("0 admits, N theorems") was enforced by grep only,
# and the committed .vo files were never re-verified on a branch. A real
# rocqchk needs the toolchain to be here.
if command -v rocq >/dev/null 2>&1; then
  run_check "rocq --version" rocq --version
elif command -v coqc >/dev/null 2>&1; then
  run_check "coqc --version" coqc --version
else
  echo "  NOTE: no rocq/coqc in this image — the proof-checking job runs in the" \
       "official rocq/rocq-prover container instead, so this is expected here."
fi

# --------------------------------------------------------------------------
echo
if [ "$failures" -gt 0 ]; then
  echo "::error::toolchain assertion FAILED: $failures of $checks checks failed."
  echo "A missing tool here means the corresponding codegen gate silently"
  echo "self-skips and CI goes green without validating anything. Fix"
  echo "ci/Dockerfile rather than relaxing this script."
  exit 1
fi
echo "toolchain assertion OK: $checks/$checks checks passed."
