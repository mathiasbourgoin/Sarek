(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Unrolling of small innermost loops with a runtime trip count, for the PTX
    generator.

    [ptxas] unrolls the rolled loop PTX it is given by a factor of 4 at most,
    while NVVM (which compiles the CUDA-C of the same IR) unrolls by 16: for a
    loop that carries a few independent global loads per iteration that is four
    times fewer loads in flight. {!rewrite} puts the unrolling in the IR, where
    the loads of several iterations become independent statements of one loop
    body that [ptxas] schedules freely.

    A loop is rewritten into a {e main loop} that runs [u] iterations per trip
    while at least [u] iterations remain, followed by the {e original} loop for
    the remainder. The body is copied unchanged, in order, so every operation
    runs in the same order on the same operands: results are bit-identical, no
    reassociation is ever made.

    A loop qualifies when it is {e innermost} and:
    - it is a [for] loop with an [int32] index and at least one non-constant
      bound (constant-trip loops are left to [ptxas], which unrolls them fully),
      or a [while] loop of the shape [while k < bound do ... ; k := k + c done]
      ([<=] also accepted) with [k] an [int32] mutated only by that last
      statement, [c] a positive constant and [bound] built from constants and
      variables the body does not assign;
    - its body contains at least one global array read, has no nested loop, no
      return, no barrier or memory fence, no warp operation, no helper call, no
      array declaration and no inline native code, and is small;
    - it is not under an explicit [pragma] (the author's hint wins).

    The remaining-iterations test is made in 64 bits, so it cannot overflow for
    any [int32] bound, and the main loop only runs when the original loop would
    have run all [u] iterations: the sequence of iterations executed is exactly
    the original one.

    The unroll factor [u] is chosen so that about {!target_loads_in_flight}
    global loads of the body are in flight, between 2 and {!max_factor}.

    [SAREK_OPT_DISABLE=unroll] ([Sarek_policy.restrict Auto_unroll]) makes
    {!rewrite} the identity; with a reporter installed
    ({!Sarek_policy.set_reporter}) the decision is reported once per kernel as
    an {!Sarek_policy.Auto_unroll} record for the generator ["ptx"]. *)

(** 32 *)
val target_loads_in_flight : int

(** 16 *)
val max_factor : int

(** [rewrite k] unrolls every qualifying loop of the kernel body and of the
    helper bodies. *)
val rewrite : Sarek_ir_types.kernel -> Sarek_ir_types.kernel
