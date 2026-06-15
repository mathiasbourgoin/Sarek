(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Test kernels that should PASS convergence analysis.
   All barriers are in converged control flow. *)

(* OK: Unconditional barrier *)
let kernel_unconditional =
  [%kernel fun (v : float32 vector) -> block_barrier ()]

(* OK: Condition depends on block_idx (uniform within block) *)
let kernel_uniform_condition =
  [%kernel
    fun (v : float32 vector) ->
      if block_idx_x > 0 then begin
        block_barrier ()
      end
      else begin
        block_barrier ()
      end]

(* OK: Barrier in for loop with uniform bounds *)
let kernel_for_loop =
  [%kernel
    fun (v : float32 vector) ->
      for _i = 0 to 10 do
        block_barrier ()
      done]

(* OK: Barrier outside divergent if - back to converged *)
let kernel_barrier_after_if =
  [%kernel
    fun (v : float32 vector) ->
      let x = if thread_idx_x > 16 then 1.0 else 2.0 in
      block_barrier () ;
      v.(thread_idx_x) <- x]

(* OK: No barrier in divergent path *)
let kernel_divergent_no_barrier =
  [%kernel
    fun (v : float32 vector) ->
      if thread_idx_x > 16 then v.(thread_idx_x) <- 1.0
      else v.(thread_idx_x) <- 2.0]

let () = print_endline "All convergence tests passed (compilation success)"
