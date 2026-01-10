---
layout: index_sample
title: GPGPU Concepts in Sarek
---

# GPGPU Concepts in Sarek

If you are coming from standard OCaml development, GPU programming requires a mental shift from "Task Parallelism" (like OCaml Domains) to "Data Parallelism" (SIMT).

## The SIMT Model
**SIMT** stands for *Single Instruction, Multiple Threads*. 

In OCaml, if you want to process an array, you might use `Array.map`. On a GPU, you write a **Kernel** (a single function) that is executed by thousands of threads simultaneously. Each thread knows its own ID and uses it to decide which piece of data to process.

## 1. Execution Hierarchy: Threads & Blocks

Sarek organizes parallel execution into a 3D hierarchy:

- **Thread**: The smallest unit of execution.
- **Block** (or Work-group): A collection of threads that can communicate and synchronize with each other.
- **Grid**: The total set of all blocks launched for a kernel.

In Sarek, you define these dimensions using `(x, y, z)` tuples. For example, a grid of `(4, 1, 1)` with blocks of `(256, 1, 1)` results in 1024 total threads.

### Terminology Mapping
- `thread_idx_x`: The ID of the thread within its block.
- `block_idx_x`: The ID of the block within the grid.
- `global_thread_id`: The unique ID of the thread across the entire grid.

## 2. Memory Hierarchy

GPU memory is not flat. Choosing the right memory space is the key to performance.

### Global Memory (`Vector.t`)
This is the main GPU memory. It is large but slow. Data transferred from OCaml via `Vector` lives here. All threads in all blocks can read/write to it.

### Shared Memory (`let%shared`)
Shared memory is a small, ultra-fast cache shared by all threads **within the same block**. It is used for collaborative computing (like reductions or matrix tiles).
```ocaml
let%shared sdata = Array.create Float32 256 in (* Shared by 256 threads *)
```

### Local Memory
Standard variables (`let x = ...` or `let mut x = ...`) are local to a single thread and usually stored in high-speed registers.

## 3. Synchronization & Supersteps

Because threads run in parallel, they often need to wait for each other.

### Barriers
A `barrier()` call forces every thread in a block to reach that point before any can proceed. This ensures that memory writes from one thread are visible to others.

### Supersteps (Sarek/BSP Model)
Sarek introduces the concept of **Supersteps**, derived from the Bulk Synchronous Parallel (BSP) model. A `let%superstep` block is a clean way to organize phases of computation separated by implicit barriers.
```ocaml
let%superstep loading_phase = 
  sdata.(tid) <- input.(gid)
in
(* Threads automatically synchronize here *)
let%superstep computing_phase = 
  process sdata.(tid)
in
...
```

## 4. Host & Device Transfers

- **Host**: Your CPU running OCaml.
- **Device**: The GPU.

The CPU and GPU usually have separate memory. Before running a kernel, you must **transfer** your data to the device. Sarek manages this through `Vector` handles. 
- When you pass a `Vec a` to `Execute.run`, Sarek ensures the data is present on the GPU.
- When you call `Vector.get` or `Vector.to_array`, Sarek pulls the data back to the OCaml heap.

## Summary Checklist
1. **Define Kernel**: Write the logic using Sarek PPX.
2. **Transfer Data**: Initialize `Vector` objects.
3. **Configure Layout**: Decide on `grid` and `block` sizes.
4. **Run**: Dispatch to CUDA, OpenCL, Vulkan, or Metal.
5. **Sync**: Retrieve results.
