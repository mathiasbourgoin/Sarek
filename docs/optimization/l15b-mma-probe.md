# L15b tensor-core `mma` implementability probe

**Task:** `mma-probe` (roster RESEARCH, fast mode) — GO/NO-GO evidence for
**L15b**, the tensor-core `mma` optimization of the L15a tiled GEMM (PR #239).

**Verdict: NO-GO** on the current runtime stack (ZLUDA v7-preview.3 / RX 7900 XTX).
`mma.sync` and the legacy `wmma.mma.sync` PTX forms **assemble** with ptxas but are
**not executed** by ZLUDA — the translator silently drops any kernel containing them.

Date: 2026-07-24. Probe: `sarek-cuda/test/test_ptx_mma_probe.ml` (permanent,
skip-clean off-CUDA). Method mirrors `test_ptx_atomics_probe.ml`: hand-written PTX
loaded via `Cuda_api.Kernel.load_from_ptx`, no DSL/codegen changes.

## Environment

| Component | Value |
|-----------|-------|
| GPU | AMD Radeon RX 7900 XTX (RDNA3, `gfx1100`; has WMMA hardware) |
| Driver | ZLUDA v7-preview.3 (`~/opt/zluda/libnvcuda.so`, dlopen'd as `libcuda.so.1`) |
| Device compute capability reported by ZLUDA | **sm_88** (mma requires sm_80+, satisfied) |
| ptxas (static) | NVIDIA ptxas 13.3 V13.3.73 (`/opt/cuda/bin`); ZLUDA shim ptxas 12.8 |
| Probe kernel | `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`, one warp |

## (a) Static — ptxas assembles `mma.sync`: PASS

Hand-written `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` PTX (`.version 8.0`)
assembles cleanly on the real NVIDIA ptxas for every target tried:

```
/opt/cuda/bin/ptxas --gpu-name sm_80 mma.ptx  -> exit 0
/opt/cuda/bin/ptxas --gpu-name sm_86 mma.ptx  -> exit 0
/opt/cuda/bin/ptxas --gpu-name sm_88 mma.ptx  -> exit 0
~/opt/zluda/ptxas --gpu-name=sm_80 -o=z.cubin mma.ptx -> exit 0 (shim, pass-through)
```

The instruction form itself is valid PTX. Static assembly is **not** the gate.

## (b) Dynamic — ZLUDA does NOT execute `mma`/`wmma`: FAIL (decisive)

Loading the assembled module through the ZLUDA driver
(`cuModuleLoadData` + `cuModuleGetFunction`) fails:

```
[INFO] device compute capability sm_88
[INFO] mma.sync NOT viable: [CUDA Runtime] Context error during
       cuModuleGetFunction: CUDA_ERROR_NOT_FOUND
[INFO] legacy wmma.mma.sync also failed: [CUDA Runtime] Context error during
       cuModuleGetFunction: CUDA_ERROR_NOT_FOUND
```

The failure is at **function lookup**, not module load: ZLUDA's PTX→AMDGPU
translator accepts the module handle but silently drops the kernel that contains
the `mma`/`wmma` instruction, so the entry symbol does not exist afterward.

### Control experiment (rules out a PTX-structure artifact)

The `CUDA_ERROR_NOT_FOUND` could in principle be caused by the kernel's structure
(`.visible .entry`, the parameter, the stores) rather than the `mma` instruction.
Ruled out: an **identical** kernel — same name, same signature, same
per-lane stores — with the single `mma.sync` line removed (D fragments set to a
constant `10.0` instead) loads and launches correctly and returns `10.0` on all
128 lanes:

```
CTRL OK out[0]=10 out[127]=10
```

The only differing token between the passing control and the failing probe is the
`mma.sync` instruction. Therefore the instruction is the sole cause. Same result
for the `wmma.mma.sync` legacy form. This is the classic ZLUDA symptom of an
un-translated PTX opcode.

## Consequence for L15b

**Do not implement L15b against this runtime.** A tensor-core GEMM path emitted by
Sarek would assemble but produce kernels that vanish at load time on ZLUDA — worse
than a slow path, it is a hard `CUDA_ERROR_NOT_FOUND` at launch. The L15a tiled
GEMM (shared-memory blocking, no tensor cores) remains the correct ceiling on this
stack. RDNA3 has WMMA hardware, so the limitation is purely ZLUDA's PTX translator,
not the GPU.

## Re-probe criteria (flip NO-GO → GO)

Re-run `dune runtest sarek-cuda` (which reaches this probe) when **any** of:

1. **ZLUDA is upgraded** past v7-preview.3 with PTX `mma.sync` / `wmma` translation.
   The probe is self-flipping: once the module loads and the function resolves, the
   `Ok` branch fires and **hard-asserts** every lane equals `10.0` — so a correct
   implementation turns the test green-with-assertions, and a *miscompiling* one
   turns it red. No code change needed to re-evaluate; just run it.
2. **A native ROCm/HIP backend** (rocWMMA) is targeted instead of PTX-on-ZLUDA — then
   L15b is a separate, ROCm-side implementability question, not covered here.
3. **A real NVIDIA sm_80+ device** is available — then re-probe there; expected GO.

### What "GO" will require from this probe when re-run

- `[INFO] mma.sync loaded+launched; out[0]=10 out[127]=10`
- all 128 lane assertions `check (float 0.01) ... 10.0` pass.

Until then the probe prints `[SKIP] tensor-core mma not available on this driver`
and stays green (capability not claimed → skip-clean, per atomics-probe convention).

## Reproduce

```
LD_LIBRARY_PATH=$HOME/opt/zluda:/opt/cuda/lib64:$LD_LIBRARY_PATH \
  dune exec sarek-cuda/test/test_ptx_mma_probe.exe -- -v
```
