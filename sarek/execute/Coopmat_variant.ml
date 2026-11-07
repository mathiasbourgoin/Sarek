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
  | Staged
      (** the kernel as given: static proofs where they hold, run-time check and
          staging for the rest *)
  | Checked
      (** every access tested at run time: a launch argument's base pointer is
          not 32-byte aligned, which the static proofs assume *)
  | Not_applicable  (** not a CUDA framework, or no cooperative matrix *)

let outcome_name = function
  | Aligned -> "aligned"
  | Staged -> "staged"
  | Checked -> "checked"
  | Not_applicable -> "not-applicable"

let frameworks = ["CUDA/C"; "CUDA/PTX"]

(* What is derived once per kernel. A slot of the kernel's {!Ir_memo} record, so
   it lives as long as the kernel and no longer, and a hit costs the launch's
   own one-per-domain IR lookup (nothing, when the kernel is the previous launch
   of the domain). It must not refer to the kernel: a variant that is the kernel
   itself is [None], and the aligned and checked kernels share only their
   immutable fields with it. *)
type entry = {
  applicable : bool;
  predicate : A.predicate option;
  aligned : kernel option;  (** [None]: the kernel itself *)
  checked : kernel option;
  arrays : string list;  (** the array parameters its accesses touch *)
}

let entry_key : entry Ir_memo.key = Ir_memo.new_key ()

let entry_of (k : kernel) : entry =
  let facts = Ir_memo.facts k in
  match Ir_memo.find facts entry_key with
  | Some e -> e
  | None ->
      let applicable = Sarek_ir_analysis.kernel_has_coopmat_op k in
      let predicate = if applicable then A.derive k else None in
      let e =
        {
          applicable;
          predicate;
          aligned =
            (if predicate <> None then Some (A.mark_aligned k) else None);
          checked = (if applicable then Some (A.mark_checked k) else None);
          arrays = (if applicable then A.arrays_accessed k else []);
        }
      in
      Ir_memo.set facts entry_key e ;
      e

(** The alignment, in bytes, [wmma] needs of an address. *)
let required_alignment = 32

(** The largest power of two (at most 4096) dividing a device pointer; [0] is
    taken as maximally aligned. *)
let alignment_of_pointer (ptr : nativeint) : int =
  let p = Int64.of_nativeint ptr in
  if p = 0L then 4096
  else
    let low = Int64.logand p (Int64.neg p) in
    if Int64.compare low 4096L > 0 then 4096 else Int64.to_int low

(* [name] is an array parameter of [k]. *)
let is_param (k : kernel) name =
  List.exists
    (function DParam (v, _) -> String.equal v.var_name name | _ -> false)
    k.kern_params

(* Every array parameter the kernel's accesses touch has a base pointer aligned
   for [wmma]. [bases] (array parameter name, alignment in bytes) is what the
   launch knows; [None]: not asked (a device allocation is assumed aligned). An
   argument without an entry is not assumed aligned; an array that is not a
   parameter (a shared array) is the generators' business. *)
let bases_aligned (k : kernel) bases arrays =
  match bases with
  | None -> true
  | Some bases ->
      List.for_all
        (fun a ->
          (not (is_param k a))
          ||
          match List.assoc_opt a bases with
          | Some alignment -> alignment >= required_alignment
          | None -> false)
        arrays

(** Whether [choose] has anything to decide for [k] on [framework]: a CUDA
    framework and a kernel with cooperative-matrix statements. The launch path
    asks this first: for every other launch it is a string comparison against
    two names (a kernel on a CUDA framework: one physical comparison with the
    last kernel seen), and nothing is allocated. *)
let relevant ~framework (k : kernel) =
  List.mem framework frameworks && (entry_of k).applicable

let aligned_launches = Atomic.make 0

let checked_launches = Atomic.make 0

let staged_launches = Atomic.make 0

(** [(aligned, staged)] launches counted so far. *)
let counts () = (Atomic.get aligned_launches, Atomic.get staged_launches)

(** Launches that ran the checked variant (a misaligned base pointer). *)
let checked_count () = Atomic.get checked_launches

(** [choose ~framework k scalars] is the kernel to generate and run for the
    scalar int32 arguments [scalars] (parameter name, value); [~count:false]
    leaves the launch counters alone (introspection). *)
let choose_lazy ?(count = true) ~framework (k : kernel)
    ~(scalars : unit -> (string * int) list)
    ~(bases : unit -> (string * int) list option) : kernel * outcome =
  if not (List.mem framework frameworks) then (k, Not_applicable)
  else
    let e = entry_of k in
    if not e.applicable then (k, Not_applicable)
    else if not (bases_aligned k (bases ()) e.arrays) then begin
      if count then Atomic.incr checked_launches ;
      (Option.value e.checked ~default:k, Checked)
    end
    else
      match e.predicate with
      | Some p when A.holds p (scalars ()) ->
          if count then Atomic.incr aligned_launches ;
          (Option.value e.aligned ~default:k, Aligned)
      | Some _ | None ->
          if count then Atomic.incr staged_launches ;
          (k, Staged)

(** {!choose_lazy} with the arguments in hand. *)
let choose ?count ?bases ~framework (k : kernel) (scalars : (string * int) list)
    : kernel * outcome =
  choose_lazy
    ?count
    ~framework
    k
    ~scalars:(fun () -> scalars)
    ~bases:(fun () -> bases)
