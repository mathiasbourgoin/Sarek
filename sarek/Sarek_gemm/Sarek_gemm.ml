(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_gemm - portable shared-memory tiled SGEMM (single-precision GEMM).
 *
 * Computes C = alpha * A * B + beta * C for row-major float32 matrices, in
 * PURE Sarek (FMA accumulation, no tensor cores), on every Sarek backend that
 * supports block-shared memory: CUDA/PTX (incl. ZLUDA), OpenCL, Vulkan, Metal,
 * Native. (The sequential Interpreter has no shared-memory model - see BACKEND
 * LIMITATIONS - so the tiled kernel is not run there, exactly like the reduce
 * and naive/tiled matmul e2e tests.)
 *
 * This is the FMA baseline for campaign item L15. A later step (L15b) will
 * optionally accelerate the inner tile product with tensor-core mma; the API
 * is kept so that fast path can slot behind it (see API / L15b HOOK below).
 *
 * TILING SCHEME  (classic BLOCK x BLOCK, one output element per thread)
 *   The output C is partitioned into {!tile} x {!tile} blocks; one thread block
 *   of {!tile} x {!tile} threads computes one output block. Thread (tx,ty)
 *   owns C[row,col] with row = ty + tile*block_idx_y, col = tx + tile*block_idx_x.
 *   Marching along K in steps of {!tile}:
 *     1. superstep LOAD  - each thread cooperatively loads one element of the
 *        current A tile and one of the current B tile into shared memory,
 *        zero-filling the halo when the global index is out of range. The
 *        implicit barrier at the end of the superstep publishes both tiles to
 *        the whole block.
 *     2. superstep COMPUTE - each thread accumulates the {!tile}-long dot
 *        product of its shared A row and shared B column into a private
 *        register with {!gemm_fma_f32} (fused multiply-add). The implicit
 *        barrier at the end guarantees every thread is done reading the tiles
 *        before the next iteration's LOAD overwrites them.
 *   After the K loop, the guarded thread writes alpha*acc + beta*C_old.
 *   Two barriers per K step (one per superstep) - the classic tiled count.
 *
 * TILE SIZE  (default 16 -> {!tile})
 *   16x16 = 256 threads/block and 2 * 16*16 * 4 B = 2 KiB shared memory - a
 *   safe, portable default (well under the 16-48 KiB/block and 1024 thread/
 *   block limits every backend here exposes). Larger tiles (e.g. 32 -> 8 KiB,
 *   1024 threads) raise arithmetic intensity / reuse but cut the number of
 *   resident blocks per multiprocessor (occupancy) and can exceed the work-
 *   group size on CPU-class OpenCL/Native devices. The shared-array extents are
 *   compile-time literals in a Sarek kernel, so the tile is fixed per kernel
 *   value: this library ships the 16x16 kernel and a matching {!Host} launch
 *   config; a different tile is a one-line edit of the two shared literals plus
 *   [tile_size] (see PATTERN) and the {!Host.tile} constant.
 *
 * BOUNDARY HANDLING  (non-multiple-of-tile dimensions)
 *   M, N, K need NOT be multiples of {!tile} and the matrices need NOT be
 *   padded. The LOAD superstep guards every global read (row<M, col<N, and the
 *   K index < K) and writes 0.0 into the shared tile on a miss, so out-of-range
 *   lanes contribute a genuine zero to the dot product; the final store is
 *   guarded by row<M && col<N. Zero-padding the tile - not the matrix - is what
 *   makes an arbitrary shape correct with no host-side copy. This is the
 *   classic GEMM boundary bug when omitted, so it is tested explicitly (see
 *   sarek/tests/e2e/test_sarek_gemm.ml: exact-multiple, non-multiple, and
 *   rectangular M<>N<>K cases).
 *
 * NUMERICS
 *   Tiled accumulation sums the K products in a different ORDER than a naive
 *   row*col loop (block by block), and FMA fuses each product-add with a single
 *   rounding. Results therefore match a naive/CPU reference within a float32
 *   epsilon, NOT bit-for-bit. Callers must compare with a tolerance (the e2e
 *   test uses a relative epsilon scaled by K).
 *
 * BACKEND LIMITATIONS
 *   - Interpreter: no block-shared-memory / barrier model (sequential executor);
 *     the tiled kernel is not run there. Use a naive kernel for the Interpreter.
 *   - All of CUDA/PTX (ZLUDA), OpenCL, Vulkan, Native provide __shared__ /
 *     local / shared / emulated tiles and a block barrier, and run this kernel.
 *
 * USAGE (from another compilation unit - same split as Sarek_worklist)
 *   dune:   add [sarek.gemm] to (libraries); add this file to
 *           (preprocessor_deps) of the kernel's stanza.
 *   source: [let%sarek_include _ = "path/to/Sarek_gemm.ml"] to reuse the pure
 *           {!gemm_fma_f32} helper from your own [%kernel]; OR use the ready
 *           kernel value {!sgemm_tiled_kernel} directly and {!Host} for the
 *           launch configuration (see the e2e test for the full flow).
 *
 * API / L15b HOOK
 *   The stable surface L15b (tensor-core mma) extends is:
 *     - {!gemm_fma_f32}     the element multiply-accumulate (swap for an mma
 *                           fragment op, or add [gemm_fma_f16]/df64 twins - the
 *                           kernel body is unchanged around it);
 *     - {!sgemm_tiled_kernel}  the (a b c m n k alpha beta) kernel value - an
 *                           mma fast path is a sibling kernel value with the
 *                           SAME argument shape, so {!Host} and every caller
 *                           are reused unchanged;
 *     - {!Host}             tile constant + block/grid/shared-mem launch config,
 *                           parameterised by the element type only.
 ******************************************************************************)

[@@@warning "-32-33-34"]

(* Alias so the pure [@sarek.module] helper type-checks as OCaml; inside a
   [%kernel] the same [float32] resolves to the device 32-bit float. *)
type float32 = float

(* OCaml-level binding of [fma] (as in Sarek_df64): the body below compiles as
   plain OCaml, while the PPX maps the bare [fma] call to the device fused-
   multiply-add intrinsic (fmaf / fma) inside a kernel. The plain-OCaml version
   runs at binary64 precision and is not a bit-faithful float32 reference. *)
let fma = Float.fma

(** Element multiply-accumulate: [acc + a * b] with a single rounding (FMA). The
    one op L15b swaps for a tensor-core fragment multiply-accumulate. *)
let[@sarek.module] gemm_fma_f32 (acc : float32) (a : float32) (b : float32) :
    float32 =
  fma a b acc

(* ========================================================================= *)
(* Directly-usable tiled SGEMM kernel value (float32, tile = 16).            *)
(*                                                                            *)
(* PATTERN: to retile, this is exactly the body to paste into your own       *)
(* [%kernel] - change the two [let%shared ... = 256l] extents to tile*tile   *)
(* and [tile_size] to the new tile (and {!Host.tile} to match).              *)
(* ========================================================================= *)

(** Tiled SGEMM: [c := alpha * a * b + beta * c] for row-major float32 [a]
    (MxK), [b] (KxN), [c] (MxN). Launch with {!Host}. Tile is 16x16. *)
let sgemm_tiled_kernel =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32)
        (alpha : float32)
        (beta : float32) ->
      let%shared (tile_a : float32) = 256l in
      let%shared (tile_b : float32) = 256l in
      let tx = thread_idx_x in
      let ty = thread_idx_y in
      let row = ty + (block_dim_y * block_idx_y) in
      let col = tx + (block_dim_x * block_idx_x) in
      let tile_size = 16l in
      let num_tiles = (k + tile_size - 1l) / tile_size in
      let sum = mut 0.0 in
      for t = 0 to num_tiles - 1l do
        let%superstep load =
          let a_col = (t * tile_size) + tx in
          if row < m && a_col < k then
            tile_a.((ty * tile_size) + tx) <- a.((row * k) + a_col)
          else tile_a.((ty * tile_size) + tx) <- 0.0 ;
          let b_row = (t * tile_size) + ty in
          if b_row < k && col < n then
            tile_b.((ty * tile_size) + tx) <- b.((b_row * n) + col)
          else tile_b.((ty * tile_size) + tx) <- 0.0
        in
        let%superstep _compute =
          for i = 0 to tile_size - 1l do
            sum :=
              gemm_fma_f32
                sum
                tile_a.((ty * tile_size) + i)
                tile_b.((i * tile_size) + tx)
          done
        in
        ()
      done ;
      if row < m && col < n then
        c.((row * n) + col) <- (alpha *. sum) +. (beta *. c.((row * n) + col))]

(** Host-side launch configuration for {!sgemm_tiled_kernel}. Kept element-type-
    agnostic so an f64/df64 or L15b mma kernel reuses it by construction. *)
module Host = struct
  (** Tile edge. MUST equal the kernel's shared extent (tile*tile) and
      [tile_size] literal; 16 -> 256 threads/block, 2 KiB shared. *)
  let tile = 16

  (** Block dimensions: one thread per output element of a tile. *)
  let block () = Sarek.Execute.dims2d tile tile

  (** Grid dimensions covering an MxN output (ceil-div, so partial edge blocks
      are launched and masked by the kernel's boundary guards). *)
  let grid ~m ~n =
    Sarek.Execute.dims2d ((n + tile - 1) / tile) ((m + tile - 1) / tile)

  (** Shared-memory footprint of the kernel's two float32 tiles, for
      occupancy estimation only. The kernel declares its tiles with
      [let%shared] (STATIC shared memory), so launches do NOT need to pass
      [~shared_mem] - passing it merely reserves the same bytes again as an
      unused dynamic region (audit finding L5). *)
  let shared_mem_bytes = 2 * tile * tile * 4
end
