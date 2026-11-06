(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Launch-time variant of a cooperative-matrix kernel on the CUDA backends.

    [wmma] needs a 32-byte aligned address and a stride that is a multiple of 16
    bytes, which the IR does not promise. The generators emit the bare
    instruction for an access they can prove aligned, and for every other access
    a run-time test with a staging path through a per-warp shared buffer
    ({!Sarek_ir_coopmat_align}). A kernel whose accesses depend on scalar
    arguments (a GEMM's [k] and [n]) would pay that buffer on every launch, so
    the host decides instead: {!Sarek_ir_coopmat_align.derive} reduces "every
    access is aligned" to a predicate on the kernel's int32 parameters, and at
    launch the actual arguments either satisfy it (the kernel marked with
    {!Sarek_ir_coopmat_align.mark_aligned}: bare [wmma], no check, no buffer) or
    not (the kernel as given: the staged variant). The marked kernel is a
    distinct IR value, built once per kernel, so [Execute]'s source memo, which
    keys on the IR's physical identity, holds one source per variant (and per
    block) and each is compiled once.

    Only the CUDA frameworks are concerned; every other backend, and a kernel
    with no cooperative-matrix statement, gets the kernel it was given. The base
    pointers are assumed 32-byte aligned (device allocations). *)

open Sarek_ir_types
module A = Sarek_ir_coopmat_align

(** What a launch runs. *)
type outcome =
  | Aligned  (** the marked variant: no check, no staging buffer *)
  | Staged  (** the kernel as given: run-time check, staging when misaligned *)
  | Not_applicable  (** not a CUDA framework, or no cooperative matrix *)

let outcome_name = function
  | Aligned -> "aligned"
  | Staged -> "staged"
  | Not_applicable -> "not-applicable"

let frameworks = ["CUDA/C"; "CUDA/PTX"]

type entry = {
  kernel : kernel;  (** compared physically *)
  applicable : bool;
  predicate : A.predicate option;
  aligned : kernel;
}

let lock = Mutex.create ()

let entries : entry list ref = ref []

let max_entries = 256

let entry_of (k : kernel) : entry =
  Mutex.protect lock (fun () ->
      match List.find_opt (fun e -> e.kernel == k) !entries with
      | Some e -> e
      | None ->
          let applicable = Sarek_ir_analysis.kernel_has_coopmat_op k in
          let predicate = if applicable then A.derive k else None in
          let e =
            {
              kernel = k;
              applicable;
              predicate;
              aligned = (if predicate <> None then A.mark_aligned k else k);
            }
          in
          let kept =
            if List.length !entries >= max_entries then
              List.filteri (fun i _ -> i < max_entries - 1) !entries
            else !entries
          in
          entries := e :: kept ;
          e)

let aligned_launches = Atomic.make 0

let staged_launches = Atomic.make 0

(** [(aligned, staged)] launches counted so far. *)
let counts () = (Atomic.get aligned_launches, Atomic.get staged_launches)

(** [choose ~framework k scalars] is the kernel to generate and run for the
    scalar int32 arguments [scalars] (parameter name, value); [~count:false]
    leaves the launch counters alone (introspection). *)
let choose ?(count = true) ~framework (k : kernel)
    (scalars : (string * int) list) : kernel * outcome =
  if not (List.mem framework frameworks) then (k, Not_applicable)
  else
    let e = entry_of k in
    if not e.applicable then (k, Not_applicable)
    else
      match e.predicate with
      | Some p when A.holds p scalars ->
          if count then Atomic.incr aligned_launches ;
          (e.aligned, Aligned)
      | Some _ | None ->
          if count then Atomic.incr staged_launches ;
          (k, Staged)
