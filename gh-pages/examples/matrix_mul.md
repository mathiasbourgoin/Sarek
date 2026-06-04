---
layout: page
title: Matrix Multiplication Example
---

# Matrix Multiplication

Matrix multiplication is a fundamental operation in high-performance computing. This example demonstrates two approaches: a naive implementation and an optimized tiled implementation using shared memory.

## Naive Implementation

Each thread computes one element of the output matrix `C`. It reads a full row from `A` and a full column from `B` from global memory.

```ocaml
open Sarek

let matmul_naive =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector)
        (m : int32) (n : int32) (k : int32) ->
      let open Sarek_stdlib.Std in
      let row = get_global_id 1 in (* y-dimension *)
      let col = get_global_id 0 in (* x-dimension *)

      if row < m && col < n then begin
        let sum = ref 0.0 in
        for i = 0 to k - 1 do
          sum := !sum +. (a.((row * k) + i) *. b.((i * n) + col))
        done;
        c.((row * n) + col) <- !sum
      end]
```

## Tiled Implementation (Shared Memory)

To reduce global memory accesses, we can load blocks ("tiles") of `A` and `B` into fast shared memory. All threads in a block cooperate to load the tiles, synchronize, and then compute a partial product.

```ocaml
let matmul_tiled =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (c : float32 vector)
        (m : int32) (n : int32) (k : int32) ->
      let open Sarek_stdlib.Std in
      (* Allocate shared memory for tiles *)
      let%shared (tile_a : float32) = 256l in (* 16x16 *)
      let%shared (tile_b : float32) = 256l in

      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in

      let tile_size = 16 in
      let num_tiles = (k + tile_size - 1) / tile_size in
      let sum = ref 0.0 in

      for t = 0 to num_tiles - 1 do
        (* Collaborative load into shared memory *)
        let%superstep load =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else
            tile_a.((ty * tile_size) + tx) <- 0.0;

          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else
            tile_b.((ty * tile_size) + tx) <- 0.0
        in

        (* Compute partial product for this tile *)
        for i = 0 to tile_size - 1 do
          sum := !sum +. (tile_a.((ty * tile_size) + i) *. tile_b.((i * tile_size) + tx))
        done
      done;

      if row < m && col < n then
        c.((row * n) + col) <- !sum]
```

## Host Code

```ocaml
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let run_matmul () =
  let m, n, k = 1024, 1024, 1024 in
  let dev = Device.best () in

  (* Get IR from kernel *)
  let _, kirc = matmul_tiled in
  let ir = match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir | None -> failwith "No IR" in

  (* Allocate matrices *)
  let a = Vector.create Vector.float32 (m * k) in
  let b = Vector.create Vector.float32 (k * n) in
  let c = Vector.create Vector.float32 (m * n) in

  (* 16x16 thread blocks *)
  let block = Execute.dims2d 16 16 in
  let grid  = Execute.dims2d ((n + 15)/16) ((m + 15)/16) in

  Execute.run_vectors
    ~device:dev ~ir ~block ~grid
    ~args:[Vec a; Vec b; Vec c; Int m; Int n; Int k]
    ()
```
