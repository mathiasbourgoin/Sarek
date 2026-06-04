---
layout: page
title: Live Transpiler
---

# Live Transpiler

> **This page describes a superseded approach.** The Thebe-based interactive environment has been replaced by the native browser Playground.

## Transpile Kernels Live in Your Browser

The **[Playground](/Sarek/playground.html)** lets you write a Sarek kernel and see the generated GPU source code (WGSL, CUDA, OpenCL, GLSL, MSL) immediately — no installation, no account.

```
[Playground →](/Sarek/playground.html)
```

The Playground runs the full Sarek transpiler pipeline compiled to WebAssembly (via js_of_ocaml) directly in your browser. Every backend output tab is generated in real time as you type.
