---
layout: index
---

<div class="features-grid">
  <div class="feature-item">
    <h3>Sarek DSL</h3>
    <p>Write high-performance GPU kernels directly in OCaml syntax using a type-safe PPX. No need to learn CUDA C or OpenCL C for your logic.</p>
  </div>
  <div class="feature-item">
    <h3>Unified Runtime</h3>
    <p>The SPOC framework manages memory transfers, device detection, and kernel execution across multiple hardware backends seamlessly.</p>
  </div>
  <div class="feature-item">
    <h3>Modern Architecture</h3>
    <p>Optimized for OCaml 5 with support for <strong>CUDA</strong>, <strong>OpenCL</strong>, <strong>Vulkan</strong>, <strong>Metal</strong>, and parallel CPU execution via Domains.</p>
  </div>
</div>

<hr style="margin: 50px 0; border: 0; border-top: 1px solid var(--border-color);">

## Recent Developments (2024-2026)

Sarek has been recently modernized to leverage the latest OCaml features and modern GPU APIs:

- **OCaml 5.4 Integration**: Full support for effects and domains, providing high-performance CPU parallel execution.
- **Cross-Platform GPU**: Newly added **Vulkan** and **Apple Metal** backends for modern desktop and mobile hardware.
- **Improved Reliability**: A new structured error handling system and comprehensive test coverage.
- **Modular Design**: Backend implementations are now dynamic plugins, allowing for lightweight and extensible builds.

## Quick Start

You can install the core Sarek ecosystem via Opam:

```bash
opam install sarek spoc
```

Check out the [Getting Started](docs/getting_started.html) guide to write your first GPU kernel in minutes, or browse the [Examples](examples/) to see common patterns.

## How it works

Sarek allows you to express parallel logic as standard OCaml functions. These are compiled to native GPU code at runtime.

```ocaml
(* A simple vector addition kernel *)
let%kernel vector_add (a : float32 vector) (b : float32 vector) (c : float32 vector) =
  let idx = get_global_id 0 in
  c.(idx) <- a.(idx) + b.(idx)
```

<hr style="margin: 50px 0; border: 0; border-top: 1px solid var(--border-color);">

## Project Info

Sarek is the result of over a decade of academic research into high-level parallel programming abstractions. It is currently maintained by **Mathias Bourgoin**.

- [Full Project History](docs/history.html)
- [Publications & Talks](docs/publications.html)
- [Frequently Asked Questions](docs/faq.html)