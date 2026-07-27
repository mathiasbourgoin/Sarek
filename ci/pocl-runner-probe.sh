#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Does pocl actually work on a bare GitHub runner? (backlog-99)
#
# WHY THIS EXISTS
#
# The CI image ships exactly one OpenCL ICD, the Intel oneAPI CPU runtime.
# An earlier attempt (backlog-79) added pocl as a second, conformant CPU ICD
# and pinned the e2e tests to it. That was strictly worse: pocl 1.8 against
# LLVM 11 on jammy compiled no kernel at all -- `error: unknown target CPU
# 'generic'` -- so the float32 math tests SKIPPED, and a SKIP is not a pass.
# ci/assert-toolchain.sh records the outcome and says the retry "moves to a
# separate experimental PR that installs it unpinned and reports whether it
# can compile on the real GitHub runner. When that lands, the checks belong
# here." This is that probe.
#
# The measured failure was a property of that IMAGE -- pocl 1.8 + LLVM 11 on
# Ubuntu 22.04 -- and says nothing about the runner. `ubuntu-latest` is a
# different distribution with a pocl several major versions newer, and nobody
# has measured what it does. Today the project's whole OpenCL story rests on
# the container image; whether a bare runner can execute a kernel at all is
# unmeasured, and that is the gap.
#
# MEASURED RESULT, 2026-07-27, run 30282252291 on `ubuntu-latest`:
#
#     distro:       Ubuntu 24.04.4 LTS
#     package:      pocl-opencl-icd 5.0-2.1build3
#     ICD:          pocl.icd -> libpocl.so.2.12.0
#     device:       cpu-haswell-AMD EPYC 9V74 80-Core Processor
#     version:      OpenCL 3.0 PoCL HSTR: cpu-x86_64-pc-linux-gnu-haswell
#     icd/enumerate/compile/execute/reject : all pass
#     VERDICT:      POCL-WORKS
#
# So the answer to backlog-99 is yes: a bare runner CAN compile and run an
# OpenCL kernel, 1024/1024 elements correct, and it rejects invalid source.
#
# This does not contradict backlog-79, it localises it. That failure was pocl
# 1.8 against LLVM 11 on jammy, inside the CI image; the runner is noble with
# pocl 5.0, four major versions on. The conclusion "pocl cannot compile" was
# never a fact about pocl, only about that pairing — which is exactly why it
# was worth measuring somewhere else rather than inferring.
#
# WHAT IT IMPLIES for the existing OpenCL gates: a second, independent OpenCL
# execution environment is now known to be available, so the project's OpenCL
# claims no longer HAVE to rest solely on the image's Intel oneAPI ICD. Two
# things that follow, neither done here:
#
#   * ci/assert-toolchain.sh's "NOT asserted here: the OpenCL ICD inventory"
#     note says the checks belong there "when that lands". They can now.
#   * A differential run — the same kernels under oneAPI and under pocl — would
#     catch the class of bug a single ICD cannot: generated OpenCL that one
#     runtime accepts and another does not. That is the real prize, and it is a
#     separate change with a real cost, so it is proposed rather than smuggled
#     in here.
#
# Promoting this job to a gate needs a second consideration this run cannot
# supply: one green is not a stability record, and pinning CI to an unpinned
# apt package is how a green turns red on someone else's release schedule.
#
# THIS IS NOT A GATE.
#
# It installs pocl, reports what happened, and exits 0 regardless. Nothing is
# pinned to pocl, no test is redirected to it, and a red result is a perfectly
# good result -- "pocl cannot compile a kernel on ubuntu-latest either" is
# information the project does not currently have. Pass --strict to make the
# verdict the exit code; the covering test uses that, and a future promotion
# to a real gate would.
#
# WHAT IT CHECKS, AND WHY IT IS IN THIS ORDER
#
# Each stage can fail independently, and lumping them together is how a probe
# ends up reporting "OpenCL is broken" when the truth was "no ICD file was
# installed". The stages, in the order a kernel actually needs them:
#
#   A  icd        an ICD manifest exists for pocl
#   B  enumerate  clGetPlatformIDs/clGetDeviceIDs return a pocl device
#   C  compile    clBuildProgram accepts a trivial kernel  <- where #79 died
#   D  execute    the kernel RUNS and the arithmetic is right
#   E  reject     an INVALID kernel is refused
#
# D and E are what make the green trustworthy. A build that succeeds is not a
# kernel that ran, which is the whole reason this repository distrusts a
# reported SKIP; and a compiler that accepts everything, including nonsense,
# reports success without compiling. E is the positive control for C: without
# it, "pocl compiled our kernel" and "pocl says yes to any string" produce the
# same output.
set -uo pipefail

STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

WORK="${POCL_PROBE_WORKDIR:-$(mktemp -d "${TMPDIR:-/tmp}/pocl-probe.XXXXXX")}"
mkdir -p "$WORK"

# Stage results, all "unknown" until measured. An unrun stage must never read
# as a passed one.
declare -A RESULT=([icd]=unknown [enumerate]=unknown [compile]=unknown \
  [execute]=unknown [reject]=unknown)

note() { printf '  %s\n' "$*"; }
stage() { printf '\n== %s ==\n' "$*"; }

# ---------------------------------------------------------------------------
stage "environment"
note "uname:    $(uname -srm)"
if [ -r /etc/os-release ]; then
  note "distro:   $(. /etc/os-release && echo "$PRETTY_NAME")"
fi
for pkg in pocl-opencl-icd libpocl2 ocl-icd-libopencl1; do
  if command -v dpkg-query >/dev/null 2>&1; then
    v="$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null)" && \
      note "package:  $pkg $v"
  fi
done
command -v clinfo >/dev/null 2>&1 && note "clinfo:   $(command -v clinfo)"

# ---------------------------------------------------------------------------
stage "A. ICD manifest"
ICD_DIR="${OCL_ICD_VENDORS:-/etc/OpenCL/vendors}"
if [ -d "$ICD_DIR" ]; then
  note "vendors dir: $ICD_DIR"
  for f in "$ICD_DIR"/*.icd; do
    [ -e "$f" ] || continue
    note "  $(basename "$f") -> $(cat "$f" 2>/dev/null)"
  done
  if ls "$ICD_DIR"/*pocl*.icd >/dev/null 2>&1; then
    RESULT[icd]=pass
  else
    RESULT[icd]=fail
    note "no pocl ICD manifest present"
  fi
else
  RESULT[icd]=fail
  note "no $ICD_DIR directory at all"
fi

# ---------------------------------------------------------------------------
# The probe proper. Written in C against the OpenCL API rather than driven
# through clinfo, because clinfo answers "is there a device" and the question
# is "can it run a kernel" -- the two came apart in exactly this way on #79,
# where pocl enumerated fine and compiled nothing.
#
# It reports one `PROBE <stage> <pass|fail>` line per stage so this script
# never has to infer a stage result from an exit code.
cat > "$WORK/probe.c" <<'CEOF'
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>

static const char *SRC_OK =
    "__kernel void twice(__global float *o, __global const float *i) {\n"
    "  size_t g = get_global_id(0);\n"
    "  o[g] = i[g] * 2.0f + 1.0f;\n"
    "}\n";

/* Not a kernel. If clBuildProgram accepts this, a successful build of the
   real kernel above means nothing. */
static const char *SRC_BAD =
    "__kernel void broken(__global float *o) {\n"
    "  this is not OpenCL C at all;\n"
    "}\n";

#define N 1024

static void report(const char *stage, int ok) {
  printf("PROBE %s %s\n", stage, ok ? "pass" : "fail");
  fflush(stdout);
}

int main(void) {
  cl_uint nplat = 0;
  cl_platform_id plats[16];
  cl_platform_id plat = NULL;
  cl_device_id dev = NULL;
  char name[512];

  if (clGetPlatformIDs(16, plats, &nplat) != CL_SUCCESS || nplat == 0) {
    printf("  no OpenCL platform enumerated\n");
    report("enumerate", 0);
    return 0;
  }
  for (cl_uint p = 0; p < nplat; p++) {
    name[0] = 0;
    clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof name, name, NULL);
    printf("  platform: %s\n", name);
    /* Match pocl specifically. Any other ICD present on the runner must not
       be able to make this probe report success on pocl's behalf. */
    if (strstr(name, "Portable Computing Language") || strstr(name, "PoCL") ||
        strstr(name, "pocl")) {
      cl_device_id d[8];
      cl_uint nd = 0;
      if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 8, d, &nd) == CL_SUCCESS &&
          nd > 0) {
        plat = plats[p];
        dev = d[0];
        name[0] = 0;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, NULL);
        printf("  pocl device: %s\n", name);
        name[0] = 0;
        clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof name, name, NULL);
        printf("  pocl device version: %s\n", name);
      }
    }
  }
  if (!dev) {
    printf("  no pocl device found among %u platform(s)\n", nplat);
    report("enumerate", 0);
    return 0;
  }
  report("enumerate", 1);

  cl_int err = CL_SUCCESS;
  cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  if (!ctx || err != CL_SUCCESS) {
    printf("  clCreateContext failed: %d\n", (int)err);
    report("compile", 0);
    return 0;
  }

  /* C -- compile. This is where pocl 1.8 + LLVM 11 failed. On failure the
     build log is the whole point: "it did not compile" without the compiler's
     reason is not a measurement anyone can act on. */
  cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC_OK, NULL, &err);
  err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len = 0;
    static char log[65536];
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, &len);
    printf("  clBuildProgram failed: %d\n", (int)err);
    printf("  --- build log ---\n%.*s\n  --- end log ---\n", (int)len, log);
    report("compile", 0);
    return 0;
  }
  report("compile", 1);

  /* D -- execute. A build is not a run. */
  cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
  float in[N], out[N];
  for (int i = 0; i < N; i++) { in[i] = (float)i; out[i] = -1.0f; }
  cl_mem din = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof in, in, &err);
  cl_mem dout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof out, NULL, &err);
  cl_kernel k = clCreateKernel(prog, "twice", &err);
  if (err != CL_SUCCESS) {
    printf("  clCreateKernel failed: %d\n", (int)err);
    report("execute", 0);
    return 0;
  }
  clSetKernelArg(k, 0, sizeof(cl_mem), &dout);
  clSetKernelArg(k, 1, sizeof(cl_mem), &din);
  size_t gws = N;
  err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("  clEnqueueNDRangeKernel failed: %d\n", (int)err);
    report("execute", 0);
    return 0;
  }
  clFinish(q);
  err = clEnqueueReadBuffer(q, dout, CL_TRUE, 0, sizeof out, out, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("  clEnqueueReadBuffer failed: %d\n", (int)err);
    report("execute", 0);
    return 0;
  }
  /* Check the ARITHMETIC, not just that the call returned. A queue that
     silently drops the kernel leaves the buffer at its initial value, and
     that must not read as success. */
  int bad = 0;
  for (int i = 0; i < N; i++) {
    float want = (float)i * 2.0f + 1.0f;
    if (out[i] != want) {
      if (bad < 3)
        printf("  out[%d] = %g, expected %g\n", i, out[i], want);
      bad++;
    }
  }
  if (bad) {
    printf("  %d/%d elements wrong\n", bad, N);
    report("execute", 0);
    return 0;
  }
  printf("  kernel ran; all %d elements correct\n", N);
  report("execute", 1);

  /* E -- reject. The positive control for C. */
  cl_program badp = clCreateProgramWithSource(ctx, 1, &SRC_BAD, NULL, &err);
  err = clBuildProgram(badp, 1, &dev, "", NULL, NULL);
  if (err == CL_SUCCESS) {
    printf("  INVALID kernel source COMPILED -- pocl's build step is not\n"
           "  checking anything, so the `compile` pass above is worthless.\n");
    report("reject", 0);
  } else {
    printf("  invalid kernel correctly rejected (%d)\n", (int)err);
    report("reject", 1);
  }
  return 0;
}
CEOF

stage "B-E. compile and run a kernel"
CC_BIN="${CC:-cc}"
# OPENCL_CFLAGS lets the covering test point at vendored Khronos headers when
# the distribution package is not installed.
# shellcheck disable=SC2086
if ! "$CC_BIN" ${OPENCL_CFLAGS:-} -O1 -o "$WORK/probe" "$WORK/probe.c" -lOpenCL \
     > "$WORK/cc.log" 2>&1; then
  note "could not build the probe (no CL headers or no libOpenCL):"
  /usr/bin/sed 's/^/    /' "$WORK/cc.log"
  RESULT[enumerate]=unbuilt
  RESULT[compile]=unbuilt
  RESULT[execute]=unbuilt
  RESULT[reject]=unbuilt
else
  "$WORK/probe" 2>&1 | /usr/bin/tee "$WORK/probe.log" | /usr/bin/sed 's/^/  /'
  while read -r _ st verdict; do
    [ -n "${st:-}" ] && RESULT[$st]="$verdict"
  done < <(/usr/bin/grep '^PROBE ' "$WORK/probe.log" || true)
fi

# ---------------------------------------------------------------------------
stage "VERDICT"
for s in icd enumerate compile execute reject; do
  printf '  %-10s %s\n' "$s" "${RESULT[$s]}"
done

# The single line a human or a later job reads. "unknown" is deliberately in
# the same bucket as "fail": a stage that did not run is not a stage that
# passed, which is the mistake this whole file is a reaction to.
if [ "${RESULT[compile]}" = "pass" ] && [ "${RESULT[execute]}" = "pass" ] \
   && [ "${RESULT[reject]}" = "pass" ]; then
  VERDICT="POCL-WORKS"
  MSG="pocl compiled, ran and correctly rejected -- a bare runner can execute an OpenCL kernel."
elif [ "${RESULT[compile]}" = "pass" ] && [ "${RESULT[reject]}" != "pass" ]; then
  VERDICT="POCL-UNTRUSTWORTHY"
  MSG="pocl built our kernel but also built invalid source; the build step is not checking anything."
elif [ "${RESULT[enumerate]}" = "pass" ]; then
  VERDICT="POCL-ENUMERATES-ONLY"
  MSG="pocl offers a device but cannot compile or run a kernel -- the backlog-79 outcome, again."
else
  VERDICT="POCL-ABSENT"
  MSG="no usable pocl device was reachable from this probe."
fi

echo
echo "POCL_PROBE_VERDICT=$VERDICT"
echo "  $MSG"
echo
echo "This step is informational (backlog-99). It gates nothing, and no test is"
echo "pinned to pocl. If the verdict is not POCL-WORKS, the practical reading is"
echo "that the project's OpenCL coverage still rests entirely on the container"
echo "image's Intel oneAPI runtime, and that a bare runner cannot be used as a"
echo "second, independent OpenCL execution environment."

if [ "$STRICT" -eq 1 ] && [ "$VERDICT" != "POCL-WORKS" ]; then
  exit 1
fi
exit 0
