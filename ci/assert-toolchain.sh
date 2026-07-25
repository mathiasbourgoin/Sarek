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
#     opencl_validation_sweep                     -> clang (OpenCL C)
#   sarek/tests/unit/test_opencl_gate.ml          -> clang (OpenCL C)
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

# How many checks must actually run. Pinned for the same reason
# check-formal-proofs.sh pins EXPECTED_PROJECTS: without it, deleting a section
# — or an edit that makes one stop running — shrinks this script's scope in
# silence and it still prints "OK: N/N checks passed" and exits 0. "N of N"
# is tautological by construction and can never report a gap on its own.
#
# The Rocq section below is deliberately NOT counted; see its comment.
EXPECTED_CHECKS=10

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
  # Counted: a failure that is not in the denominator reads as "1 of 6 failed"
  # out of a 6 that excludes it.
  checks=$((checks + 1))
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
#
# -L is load-bearing: /usr/local/cuda is a SYMLINK to /usr/local/cuda-<series>,
# and GNU find does not follow a symlinked starting point unless told to (a
# trailing slash also works, but -L says what is meant). Without it this check
# reported the header missing on an image where it was plainly present.
checks=$((checks + 1))
fp16=$(find -L "${CUDA_PATH:-/usr/local/cuda}" /usr/local/cuda \
         -name cuda_fp16.h 2>/dev/null | head -1)
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
  checks=$((checks + 1))
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
  checks=$((checks + 1))
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
section "clang (OpenCL C validation)"
# The OpenCL gate (#128) compiles generated OpenCL C with `clang -x cl` rather
# than through a vendor ICD, and that choice is the point: on the reference
# machine (RX 7900 XTX, rusticl/radeonsi) illegal generated OpenCL did not
# produce a build log, it took the host process down with SIGSEGV. A gate has to
# fail where we can read the failure.
if ! command -v clang >/dev/null 2>&1; then
  checks=$((checks + 1))
  fail "clang is not on PATH. OpenCL C is then the ONLY backend with committed \
goldens and no executable validation at all, so opencl_validation_sweep would \
silently validate nothing."
else
  run_check "clang --version" clang --version
  tmpdir=$(mktemp -d)
  cat >"$tmpdir/probe.cl" <<'CL'
__kernel void probe(__global int *o) { o[get_global_id(0)] = 1; }
CL
  checks=$((checks + 1))
  # Positive control, not `command -v`: a clang built without OpenCL support, or
  # without the default builtin header, is on PATH and useless. This probe needs
  # BOTH -x cl and -finclude-default-header to succeed, which is exactly the
  # invocation the gate uses (opencl_clang.ml). A clang that cannot resolve
  # get_global_id would fail every kernel on builtins rather than on defects.
  if out=$(clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header \
             -fsyntax-only "$tmpdir/probe.cl" 2>&1); then
    ok "clang compiled a probe OpenCL kernel"
  else
    fail "clang is on PATH but cannot compile a trivial OpenCL kernel:
$out"
  fi
  rm -rf "$tmpdir"
fi

# --------------------------------------------------------------------------
# NOT asserted here: the OpenCL ICD inventory.
#
# An earlier revision installed pocl as a second, conformant CPU ICD (task #79)
# and asserted it enumerated. Measured in this image, pocl 1.8 + LLVM 11 on
# jammy cannot compile any kernel at all — `error: unknown target CPU 'generic'`
# — so the assertion would have guarded a device that cannot run a test, while
# the Intel oneAPI runtime it was meant to replace computed sin/cos/exp/sqrt
# correctly here (worst relative error 1.3e-07). pocl moves to a separate
# experimental PR that installs it unpinned and reports whether it can compile
# on the real GitHub runner. When that lands, the checks belong here.
#
# --------------------------------------------------------------------------
section "Rocq / Coq proof checker (informational — NOT a gate)"
# Reported, not asserted, and deliberately excluded from EXPECTED_CHECKS.
#
# #45's guarantee is enforced by the separate formal-proofs job, which runs
# scripts/check-formal-proofs.sh inside the official rocq/rocq-prover image.
# This image ships no Rocq on purpose (see the comment above the formal-proofs
# job in .github/workflows/ci.yml), so there is nothing here to assert: a check
# that passes whether or not the tool is present is the exact anti-pattern this
# script exists to remove, and an earlier revision of this section was one.
#
# Absence is not asserted either — adding Rocq to this image would be wasteful
# but not wrong, and failing on it would make this script a style gate.
if command -v rocq >/dev/null 2>&1; then
  echo "  NOTE: rocq present ($(rocq --version 2>&1 | head -1)); the proof job" \
       "does not use it."
elif command -v coqc >/dev/null 2>&1; then
  echo "  NOTE: coqc present ($(coqc --version 2>&1 | head -1)); the proof job" \
       "does not use it."
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
# Reached only when every check that RAN passed — which is exactly when a
# silently-vanished check would otherwise be invisible.
if [ "$checks" -ne "$EXPECTED_CHECKS" ]; then
  echo "::error::toolchain assertion ran $checks checks, expected \
$EXPECTED_CHECKS. Every check passed, so this is not a broken tool — it is a \
check that stopped running. Restore it, or update EXPECTED_CHECKS at the top of \
this script if the removal was deliberate."
  exit 1
fi
echo "toolchain assertion OK: $checks/$EXPECTED_CHECKS checks passed."
