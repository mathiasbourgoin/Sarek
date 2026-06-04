---
layout: page
title: "Learn GPU programming with Sarek"
---

# Learn GPU programming with Sarek

These interactive lessons teach the data-parallel programming model through
short, runnable exercises. Each lesson lets you edit a Sarek kernel, click
**Run on my GPU**, and instantly see whether your answer is correct — graded
against a CPU reference running in the same browser tab.

---

## What is GPGPU and what is a kernel?

A **GPU** (Graphics Processing Unit) has thousands of small cores that run in
parallel. **GPGPU** (General-Purpose GPU computing) uses those cores to
accelerate arbitrary numeric computation — not just graphics.

The central concept is the **kernel**: a function that runs once per element
of your dataset, each invocation handling exactly one element identified by
its **thread index** (`global_thread_id` in Sarek). If you have an array of
256 floats, the GPU launches 256 threads simultaneously, each computing one
output value independently. This is the *data-parallel* model.

```
Thread 0:  reads a[0], b[0]  → writes c[0]
Thread 1:  reads a[1], b[1]  → writes c[1]
...
Thread 255: reads a[255], b[255] → writes c[255]
```

No thread needs to wait for another — all 256 compute at once.

---

## How Sarek / SPOC works

**Sarek** lets you write GPU kernels in OCaml syntax. The compiler
(*transpiler*) reads your kernel expression and emits GPU source code for
multiple backends: CUDA, OpenCL, Metal, GLSL, and **WGSL**.

In these lessons, Sarek transpiles to **WGSL** (WebGPU Shading Language) and
runs the shader on **your own GPU** directly in your browser via the
[WebGPU API](https://www.w3.org/TR/webgpu/). The transpiler itself is a
WebAssembly + JavaScript bundle loaded from this page — no server is involved.

---

## How to read a Sarek kernel

A minimal kernel looks like:

```
fun (a : float32 vector) (b : float32 vector) ->
  let i = global_thread_id in
  b.(i) <- Float32.sin a.(i)
```

- `fun (a : float32 vector) ...` — the kernel's typed arguments. Buffers
  (arrays on the GPU) are `elementType vector`. Scalar values (ints, floats)
  are plain `int32` / `float32`.
- `let i = global_thread_id in` — binds the thread index. Each of the N
  threads gets a unique value of `i` from 0 to N-1.
- `b.(i) <- expr` — writes `expr` into output buffer `b` at position `i`.
  Read accesses use `a.(i)` without the `<-`.

---

## How the interactive checker works

1. You edit the kernel starter in the editor.
2. Click **Run on my GPU** — the page calls
   `SarekTranspile.transpileWithAbi(source, "wgsl")` to compile your kernel
   to WGSL, then `SarekWebGPU.run(...)` to dispatch it on your GPU.
3. The result is compared element-by-element against a CPU reference
   (JavaScript). A relative tolerance of ±0.1 % is allowed for floating-point
   rounding. You see **PASS** (green) or **FAIL** with the first mismatching
   index.
4. Use **Show generated WGSL** to inspect the shader the transpiler produced.

All datasets use **N = 256** for instant iteration. Larger datasets (N ≥ 64 K)
would show more realistic GPU speedups but take several seconds to grade in
the browser.

---

## Browser requirements

WebGPU requires a **recent Chrome or Edge** (version 113+). Firefox Nightly
also works with the `dom.webgpu.enabled` flag. If your browser does not
support WebGPU the lesson text still renders and you can read the kernel, but
the Run button will be disabled with an explanatory message.

---

## Lessons

1. [Lesson 1 — Vector addition]({{ site.baseurl }}/learn/01-vector-add.html) — the data-parallel hello world
2. [Lesson 2 — Scalar parameters (SAXPY)]({{ site.baseurl }}/learn/02-scale-saxpy.html) — passing a scalar into a kernel
3. [Lesson 3 — Elementwise map]({{ site.baseurl }}/learn/03-map-square.html) — mapping a function over every element
4. [Lesson 4 — Control flow and bounds]({{ site.baseurl }}/learn/04-bounds-if.html) — `if` expressions and index guarding
5. [Lesson 5 — Mandelbrot]({{ site.baseurl }}/learn/05-mandelbrot.html) — a per-pixel loop that **generates an image** on your GPU
6. [Lesson 6 — Image filter]({{ site.baseurl }}/learn/06-image-filter.html) — a grayscale **photo filter** rendered in the page (bring your own photo)
