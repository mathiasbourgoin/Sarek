---
layout: page
title: Launch Sarek on Binder
---

# Try Sarek in Your Browser

> **This page describes a superseded approach.** The Binder/Jupyter environment described here has been replaced by a native browser experience that requires no setup.

## Live Options — No Installation Required

Sarek now runs directly in the browser via WebGPU:

- **[Playground](/Sarek/playground.html)** — Transpile any Sarek kernel to GPU source code (WGSL, CUDA, OpenCL, GLSL, MSL) instantly. Runs the full Sarek compiler in-browser.
- **[Interactive Learn Course](/Sarek/learn/)** — Write and run Sarek kernels on your own GPU from the page. Covers vector addition, Mandelbrot generation, image filters, and more.

Unlike the Binder approach, these options run the actual GPU workloads on your hardware via WebGPU — not on a cloud CPU. The code you write is identical to the code you would compile natively with OCaml.
