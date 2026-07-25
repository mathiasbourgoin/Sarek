# ---------------------------------------------------------------------------
# naga (WGSL validator) — built in a throwaway stage.
#
# There is no apt package and no upstream prebuilt binary for naga-cli, so the
# only options are a Rust toolchain or Dawn's `tint` (a CMake/depot_tools build,
# far more expensive). A builder stage keeps the ~1.5 GB Rust toolchain out of
# the final image: only the 6.0 MB `naga` binary is copied across, and the whole
# stage is one cached layer keyed on this Dockerfile.
#
# Measured: 6.0 MB binary, 21 s to build on a 16-core workstation with a warm
# cargo registry; budget a few minutes for a cold 2-core GitHub runner.
# ---------------------------------------------------------------------------
# naga 30.0.0 declares rust-version = "1.87"; 1.90 is the pinned floor above it.
FROM rust:1.90-slim-bookworm AS naga-builder
ARG NAGA_VERSION=30.0.0
RUN cargo install naga-cli --version "${NAGA_VERSION}" --locked --root /out

FROM intel/oneapi-runtime:2024.2.1-0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG CACHEBUST
RUN echo "cache bust: ${CACHEBUST}" > /tmp/cachebust

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    clang \
    curl \
    git \
    gnupg \
    libffi-dev \
    libx11-dev \
    m4 \
    opam \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Codegen-validation toolchain.
#
# Every gate below already existed and already skipped itself when its tool was
# absent, which on this image meant ALL of them: CI was green because nothing
# ran. See ci/assert-toolchain.sh, which turns a future regression of this layer
# back into a red build instead of a silent return to self-skipping.
#
# TOTAL IMAGE COST, measured by summing `docker history` layer sizes for a build
# of this file against a build of the previous ci/Dockerfile:
#   baseline           4483 MB
#   with this file     4834 MB   -> +351 MB
# broken down as: CUDA layer 304 MB, glslang layer 35.8 MB, naga 6.8 MB, gnupg
# (needed for the NVIDIA apt keyring) 5 MB, symlink/ldconfig 0.1 MB.
#
#   glslang-tools       glslangValidator, the GLSL -> SPIR-V validator used by
#                       the shader-validation sweep and by the production Vulkan
#                       path (Sarek_vulkan.Vulkan_api_base). 7.4 MB itself, but
#                       35.8 MB as a layer: on jammy `glslang-tools` has a hard
#                       `Depends: spirv-tools` (25.5 MB), so spirv-tools arrives
#                       whether or not it is asked for. It is no longer listed
#                       explicitly, because nothing in this repository uses
#                       spirv-val and claiming it as a capability would be
#                       false — but it cannot actually be removed without
#                       losing glslangValidator.
#
#   clang               NOT installed here - already provided by the base
#                       toolchain layer at the top of this file, so the OpenCL
#                       validation gate (#128) adds 0 MB. Recorded in this list
#                       anyway because that install now has a SECOND consumer
#                       and nothing else would say so: `clang` there is no
#                       longer only a C toolchain, it is the OpenCL C front-end
#                       the gate and ci/assert-toolchain.sh's probe depend on.
#                       If it is ever trimmed as unused, assert-toolchain.sh
#                       fails loudly rather than letting the sweep return to
#                       self-skipping.
#
#                       Deliberately NOT an OpenCL ICD: the gate must be able
#                       to REPORT a rejection, and on the reference hardware
#                       (RX 7900 XTX, rusticl/radeonsi) illegal generated
#                       OpenCL crashed the host process instead of returning a
#                       build log (#53/#127). clang needs no device, so this
#                       does not change device enumeration. `clang -x cl` plus
#                       `-Xclang -finclude-default-header` supplies the builtin
#                       declarations an ICD would inject; the assert-toolchain
#                       probe fails if that header is missing, because without
#                       it every kernel would fail on get_global_id rather than
#                       on its own defects.
#
# NOT installed, and why:
#
#   pocl-opencl-icd
#                 A conformant CPU OpenCL ICD is still wanted (task #79: the
#                 image's only ICD is Intel's oneAPI CPU runtime, classified as
#                 a known-issue device by #74). But pocl 1.8 + LLVM 11 on jammy
#                 cannot compile a single kernel in this image: every build
#                 fails with `error: unknown target CPU 'generic'` against the
#                 device string `pthread-x86_64-pc-linux-gnu-generic`. Pinning
#                 e2e to it made test_float32_sin_pure and test_math_intrinsics
#                 SKIP while `make e2e-fast` logged 11 errors and still exited
#                 0 — strictly worse than the status quo. Measured in the same
#                 image, the Intel ICD computed sin/cos/exp/sqrt correctly
#                 (worst relative error 1.3e-07) and SPOC's own #74 expiry NOTE
#                 fired to say so. pocl also dragged in ~243 MB of LLVM.
#                 It moves to a separate experimental PR that installs it
#                 WITHOUT pinning and logs whether it can compile a kernel on
#                 the actual GitHub runner (Ice Lake / Zen 3), where LLVM 11 may
#                 resolve a real target CPU and where #74's flake was first
#                 seen. The inert Test_helpers.is_pocl_device carve-out stays in
#                 the tree for that PR to use.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install --no-install-recommends -y \
    glslang-tools \
  && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# CUDA host-side compilers: ptxas (PTX assembler) and NVRTC (CUDA-C -> PTX).
#
# BOTH ARE HOST-SIDE. Neither needs an NVIDIA device, a driver, or /dev/nvidia*,
# so this does not change device enumeration: no CUDA device appears, the CUDA
# plugin still finds no driver, and the CUDA E2E tests keep skipping. What it
# does buy is that the two static gates stop self-skipping:
#   - test_ptx_snapshot.ml assembles the emitted PTX with ptxas;
#   - test_cuda_nvrtc_gate.ml compiles the emitted CUDA-C with real NVRTC —
#     which is the check that would have caught the f16 backend emitting
#     `#include <cuda_fp16.h>`, a header NVRTC cannot resolve because it has no
#     filesystem include path unless one is passed. Verified locally: prepending
#     that include to the corpus turns all 17 cases red with "catastrophic
#     error: could not open source file \"cuda_fp16.h\" (no directories in
#     search list)".
#
# Granular packages only — deliberately NOT the `cuda-toolkit` metapackage
# (multi-GB). Sizes below are `dpkg-query Installed-Size` read out of the BUILT
# image, dependencies included, not estimates from the .deb payloads:
#   cuda-nvcc-12-6       147.6 MB  ptxas 31 MB, nvlink 31 MB, nvcc 23 MB,
#                                  libnvptxcompiler_static.a 60 MB
#   cuda-nvrtc-12-6       60.5 MB  libnvrtc.so.12 + builtins
#   cuda-nvvm-12-6        56.1 MB  hard dependency of cuda-nvcc (cicc/libnvvm);
#                                  not requested, not optional
#   cuda-cccl-12-6        12.4 MB  dependency of cuda-cudart-dev
#   cuda-cudart-dev-12-6   6.8 MB  cuda_fp16.h and the rest of the header tree
#   cuda-crt / cuda-cudart / config / driver-dev   ~2.2 MB
#   ------------------------------------------------------------------
#   packages             285.6 MB;  layer as built: 304 MB
#
# cuda-nvrtc-dev-12-6 (93 MB: libnvrtc_static.a, libnvptxcompiler_static.a,
# nvrtc.h) is deliberately NOT installed. Sarek reaches NVRTC by dlopen through
# ctypes (sarek-cuda/Cuda_nvrtc.ml), so it needs the shared object and no
# headers at all.
#
# Version-pinned to the 12.6 series so an upstream CUDA bump cannot silently
# change what the gates assemble against.
# ---------------------------------------------------------------------------
ARG CUDA_APT_SERIES=12-6
ARG CUDA_SERIES_DOTTED=12.6
RUN curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  && dpkg -i cuda-keyring_1.1-1_all.deb \
  && rm cuda-keyring_1.1-1_all.deb \
  && apt-get update && apt-get install --no-install-recommends -y \
       "cuda-nvcc-${CUDA_APT_SERIES}" \
       "cuda-nvrtc-${CUDA_APT_SERIES}" \
       "cuda-cudart-dev-${CUDA_APT_SERIES}" \
  && rm -rf /var/lib/apt/lists/*

# The /usr/local/cuda -> /usr/local/cuda-<series> symlink normally comes from
# the `cuda-toolkit` metapackage we are deliberately not installing, so create
# it here; everything below (PATH, CUDA_PATH, ld.so.conf) is written against the
# unversioned path so a series bump only touches CUDA_APT_SERIES.
#
# Setting CUDA_PATH also flips one `enabled_if` in the tree:
# sarek/tests/new_runtime/dune gates test_runtime_comparison on CUDA_PATH being
# non-empty, so that executable now gets BUILT in CI. It is an (executable) with
# no run rule, so this adds a compile check and no runtime dependency on a
# device — and NVRTC/driver are dlopen'd lazily, so it links without -lnvrtc.
RUN ln -sfn "/usr/local/cuda-${CUDA_SERIES_DOTTED}" /usr/local/cuda \
  && printf '%s\n%s\n' /usr/local/cuda/lib64 \
       /usr/local/cuda/targets/x86_64-linux/lib \
       > /etc/ld.so.conf.d/999-cuda.conf \
  && ldconfig \
  && /usr/local/cuda/bin/ptxas --version \
  && fp16="$(find -L /usr/local/cuda -name cuda_fp16.h | head -1)" \
  && [ -n "$fp16" ] \
  && echo "cuda_fp16.h: $fp16"
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

COPY --from=naga-builder /out/bin/naga /usr/local/bin/naga

# Intel oneAPI provides the ONLY OpenCL runtime in this image. pocl was tried as
# a second, conformant CPU ICD and removed — see the "NOT installed, and why"
# block above; it moves to a separate experimental PR.
ENV OPAMYES=1
