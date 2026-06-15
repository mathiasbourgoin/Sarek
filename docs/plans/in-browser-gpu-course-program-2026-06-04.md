# In-Browser GPU Course Program — master plan

**Date:** 2026-06-04
**Status:** IN PROGRESS (autonomous)
**Owner:** agent (autonomous), final SPOC-web-backend PR gated on human merge

## Context

The Sarek transpiler runs in-browser (jsoo playground, live). The WGSL backend
(PR #161) is merged and GPU-validated on the RX 7900 XTX via Playwright/WebGPU.
This program builds the in-browser **interactive GPU course** on top of that,
then hardens the website, then begins the pure SPOC-web backend that will
eventually replace the ad-hoc WebGPU runner.

## Phases (each its own PR vs `main`, roster pipeline, GPU + Playwright checks)

1. **JS WebGPU runner + ABI descriptor** *(auto-merge)*
   - Transpiler emits a structured **ABI** alongside the WGSL: ordered storage
     buffers (name, binding, element type, access), the `Params` uniform layout
     (vec-length i32 fields then scalars, with byte offsets), and workgroup size.
   - A reusable JS module `SarekWebGPU` binds buffers / packs the uniform /
     computes dispatch from the ABI + inputs and runs any kernel — generalizing
     the hardcoded test harness.
   - Acceptance: the existing 3 kernels run through the *generic* runner on the
     real GPU (Playwright), byte-for-byte vs the hardcoded path.

2. **First course pages** *(auto-merge)*
   - Intro: what GPGPU/a kernel is; how to write one with Sarek; the SPOC model.
   - Vector addition lesson with an **editable kernel + auto-check**: the page
     runs the user's kernel on their GPU and grades the result vs a CPU
     reference (auto-correction), with a clear pass/fail.
   - Then progressively harder problems (map, saxpy, reduction-style, etc.).
   - **Visual lessons (rendered to a `<canvas>` in the page):**
     - **Mandelbrot** (generation): per-pixel iteration count → JS colormap → canvas.
     - **Photo filter** (e.g. grayscale/invert/threshold): RGB input vectors →
       output vector → canvas; show input + output side by side.
     - Both stay within the float32-vector ABI; keep image size small (≈256²)
       with a resource/time disclaimer for larger images / more iterations.
   - Keep N and compute time **small** for fast iteration; bigger datasets later
     behind explicit resource/time disclaimers.
   - Sequencing note: the numeric harness + 4 lessons land first (agent in
     flight); the canvas-render mode + the two visual lessons extend the SAME
     `roadmap/course-pages` branch so Phase 2 ships as one PR.

3. **Website consistency pass** *(auto-merge)*
   - After the course covers most Sarek/GPGPU aspects, ensure nav, theming,
     layouts, cross-links, and the playground all remain coherent.

4. **README + website docs audit** *(auto-merge)*
   - Audit README and site docs; clean/fix/update stale or wrong content.

5. **SPOC web backend (pure-JS SPOC core)** *(PR only — NOT auto-merged)*
   - On a separate branch, begin splitting SPOC into pure parts that compile to
     JS, to eventually replace the hand-written WebGPU runner and to enable
     courses on host-side interaction and kernel composition.
   - Open a PR for human review; do **not** merge.

## Working rules (from the user)

- Roster pipeline per phase; regular commits; branch-per-PR vs `main`.
- Run `make bench-gpu-check` (real GPU) before pushing codegen/runtime changes.
- Validate WGSL/runtime changes through Playwright + real WebGPU.
- Phases 1–4 auto-merge; phase 5 stops at PR for human decision.
</content>
</invoke>
