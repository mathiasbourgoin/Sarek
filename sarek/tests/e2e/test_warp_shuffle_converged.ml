(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Kernels that must PASS the front end: every warp shuffle is in converged
   control flow, in every spelling. The assertion is successful compilation. *)

[@@@warning "-39"]

module Std = Sarek_stdlib.Std
module Gpu = Sarek_stdlib.Gpu

(* OK: unconditional, all four primitives, three value types *)
let kernel_all_primitives =
  [%kernel
    fun (v : int32 vector) (f : float32 vector) (d : float64 vector) ->
      let t = thread_idx_x in
      v.(t) <- warp_shuffle v.(t) 3l ;
      v.(t) <- warp_shuffle_down v.(t) 1l ;
      v.(t) <- warp_shuffle_up v.(t) 1l ;
      v.(t) <- warp_shuffle_xor v.(t) 4l ;
      f.(t) <- warp_shuffle_xor f.(t) 2l ;
      d.(t) <- warp_shuffle_down d.(t) 1l]

(* OK: after a thread-varying if, back in converged flow *)
let kernel_after_varying_if =
  [%kernel
    fun (v : int32 vector) ->
      let x = if thread_idx_x > 16 then 1l else 2l in
      v.(thread_idx_x) <- warp_shuffle_xor x 1l]

(* OK: under a block-uniform condition and a uniform loop bound *)
let kernel_uniform_control =
  [%kernel
    fun (v : float32 vector) (n : int32) ->
      if block_idx_x > 0l then
        for _i = 0 to 3 do
          v.(thread_idx_x) <-
            v.(thread_idx_x) +. warp_shuffle_xor v.(thread_idx_x) 1l
        done]

(* OK: the qualified spellings, and warp_size / lane_id as values *)
let kernel_qualified =
  [%kernel
    fun (v : int32 vector) ->
      let w = Gpu.warp_size in
      let l = Std.lane_id in
      v.(thread_idx_x) <- Gpu.warp_shuffle_xor (l + w) 1l]

(* OK: a helper containing a shuffle, called from converged flow *)
let kernel_helper =
  [%kernel
    fun (v : int32 vector) ->
      let reduce (x : int32) = warp_shuffle_xor x 1l in
      v.(thread_idx_x) <- reduce v.(thread_idx_x)]

(* OK: the shuffle result is thread-varying, and a varying if AFTER it is fine *)
let kernel_result_varying =
  [%kernel
    fun (v : int32 vector) ->
      let x = warp_shuffle v.(thread_idx_x) 0l in
      if x > 0l then v.(thread_idx_x) <- x]

(* OK: a later helper of the same name shadows the one that shuffles *)
let kernel_shadowed_helper =
  [%kernel
    fun (v : int32 vector) ->
      let reduce (x : int32) = warp_shuffle_xor x 1l in
      v.(thread_idx_x) <- reduce v.(thread_idx_x) ;
      let reduce (y : int32) = y + 1l in
      if thread_idx_x > 16 then v.(thread_idx_x) <- reduce v.(thread_idx_x)]

(* The strict analysis (thread-varying parameters, helper results, mutables
   assigned in divergence) must not refuse these legitimate kernels. *)

(* OK: a downstream 8-lane reduction, a shuffle tree in converged flow *)
let kernel_reduce8_xor =
  [%kernel
    fun (inp : float32 vector) (out : float32 vector) ->
      let g = (block_idx_x * 64l) + thread_idx_x in
      let s0 = inp.(g) in
      let s1 = s0 +. warp_shuffle_xor s0 4l in
      let s2 = s1 +. warp_shuffle_xor s1 2l in
      let s3 = s2 +. warp_shuffle_xor s2 1l in
      if thread_idx_x land 7l = 0l then out.(g / 8l) <- s3]

(* OK: a shuffle loop with a uniform bound, accumulating in a mutable that is
   varying (it only feeds a store, never a shuffle condition) *)
let kernel_loop_tree =
  [%kernel
    fun (v : float32 vector) ->
      let acc = mut v.(thread_idx_x) in
      for i = 0 to 2 do
        acc := acc +. warp_shuffle_down acc (1l lsl i)
      done ;
      v.(thread_idx_x) <- acc]

(* OK: a helper parameter that is thread-varying but only feeds the shuffled
   value, never a condition *)
let kernel_varying_param_value =
  [%kernel
    fun (v : int32 vector) ->
      let reduce (x : int32) = x + warp_shuffle_xor x 1l in
      v.(thread_idx_x) <- reduce (thread_idx_x + v.(thread_idx_x))]

(* OK: a helper whose shuffle is under a test on a parameter, called with a
   uniform argument *)
let kernel_uniform_param_condition =
  [%kernel
    fun (v : int32 vector) (n : int32) ->
      let gate (k : int32) (x : int32) =
        if k > 16l then warp_shuffle_xor x 1l else x
      in
      v.(thread_idx_x) <- gate n v.(thread_idx_x)]

(* OK: a helper result that is uniform (reads only block_idx_x) gating a
   shuffle *)
let kernel_uniform_helper_result =
  [%kernel
    fun (v : int32 vector) ->
      let blk (k : int32) = block_idx_x + k in
      if blk 0l > 0l then
        v.(thread_idx_x) <- warp_shuffle_xor v.(thread_idx_x) 1l]

(* OK: a mutable assigned only from uniform values in uniform control flow *)
let kernel_uniform_mutable =
  [%kernel
    fun (v : int32 vector) (n : int32) ->
      let m = mut 0l in
      if block_idx_x > 0l then m := n ;
      if m > 0l then v.(thread_idx_x) <- warp_shuffle_xor v.(thread_idx_x) 1l]

let () = print_endline "All warp shuffle convergence tests passed (compilation)"
