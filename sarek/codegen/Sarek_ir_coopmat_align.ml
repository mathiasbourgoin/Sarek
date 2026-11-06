(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Static alignment proof for cooperative-matrix loads and stores on NVIDIA
    ([wmma]), shared by the CUDA/C and PTX generators.

    [wmma.load] / [wmma.store] need a 32-byte aligned address and a leading
    dimension that is a multiple of 16 bytes. The IR promises neither (any start
    index, any stride: {!Sarek_ir_types.CM_load}), so a generator either checks
    at run time and stages a misaligned tile through shared memory, or proves
    the access aligned and emits the bare instruction. This module is the proof.

    {b What is proved.} A power-of-two {e factor} of an int32 expression: [f]
    such that [f] divides the expression's value, whatever the run. It is
    computed from the expression alone, by rules that hold under wrapping 32-bit
    arithmetic (divisibility by a power of two is preserved modulo 2^32):
    - an integer constant [c]: its lowest set bit ([0]: unbounded);
    - [a + b], [a - b]: the minimum of the factors;
    - [a * b]: the product;
    - [a << c] with a constant [c]: the factor of [a] times [2^c];
    - a variable bound by an immutable [let]: the factor of its definition;
    - anything else (kernel parameters, loop variables, mutable variables,
      division, intrinsics, array reads): 1.

    An access is {e proven} when [factor index * element bytes >= 32] and
    [factor stride * element bytes >= 16] (both sides are powers of two, so [>=]
    is divisibility). {b The base pointer is assumed aligned}: a global buffer
    is a device allocation (256-byte aligned) and a [.shared] array a
    cooperative-matrix statement accesses is declared 32-byte aligned by the
    generators. A cooperative-matrix kernel that does not rely on either (a
    pointer into the middle of an allocation) must not pass a view to the
    launch.

    The walk is by name with lexical scoping, so a shadowing binder (a [let], a
    [for] variable, a match payload) hides an outer fact. The result is the list
    of proven operations, identified {e physically}: a generator asks about the
    very node it is about to emit, and an operation it cannot find (a rewritten
    tree, a helper body) is unproven, which is the safe answer. *)

open Sarek_ir_types
module C = Sarek_coopmat_types

(** Cap of a factor: [2^30], standing for "divisible by every power of two that
    matters". *)
let unbounded = 1 lsl 30

let lowest_bit n = if n = 0 then unbounded else min unbounded (n land -n)

type env = (string * int) list

let rec factor (env : env) (e : expr) : int =
  match e with
  | EConst (CInt32 c) -> lowest_bit (Int32.to_int c)
  | EVar v when v.var_mutable -> 1
  | EVar v -> Option.value (List.assoc_opt v.var_name env) ~default:1
  | EBinop ((Add | Sub), a, b) -> min (factor env a) (factor env b)
  | EBinop (Mul, a, b) -> min unbounded (factor env a * factor env b)
  | EBinop (Shl, a, EConst (CInt32 s)) when s >= 0l && s < 31l ->
      min unbounded (factor env a lsl Int32.to_int s)
  | _ -> 1

(** Bytes per element of the buffer a fragment is loaded from or stored to. *)
let element_bytes (f : C.fragment) =
  match f.C.frag_component with
  | C.Uint8 | C.Sint8 -> 1
  | C.Float16 -> 2
  | C.BFloat16 | C.Float32 | C.Uint32 | C.Sint32 -> 4

let access_is_proven env ~(frag : C.fragment) ~index ~stride =
  let bytes = element_bytes frag in
  factor env index * bytes >= 32 && factor env stride * bytes >= 16

(** Bytes of the 16x16 staging tile of a fragment. *)
let tile_bytes (f : C.fragment) = 256 * element_bytes f

(** The proven loads and stores of a kernel body. *)
let proven_ops (body : stmt) : coopmat_op list =
  let found = ref [] in
  let rec walk env = function
    | SCoopmat (CM_load {frag; index; stride; _} as op)
    | SCoopmat (CM_store {frag; index; stride; _} as op) ->
        if access_is_proven env ~frag ~index ~stride then found := op :: !found
    | SCoopmat (CM_decl _ | CM_muladd _) -> ()
    | SSeq ss -> List.iter (walk env) ss
    | SIf (_, a, b) ->
        walk env a ;
        Option.iter (walk env) b
    | SWhile (_, s) | SPragma (_, s) | SBlock s -> walk env s
    | SFor (v, _, _, _, s) -> walk ((v.var_name, 1) :: env) s
    | SLet (v, e, s) -> walk ((v.var_name, factor env e) :: env) s
    | SLetMut (v, _, s) -> walk ((v.var_name, 1) :: env) s
    | SMatch (_, arms) ->
        List.iter
          (fun (pat, s) ->
            let env =
              match pat with
              | PConstr (_, names) ->
                  List.fold_left (fun env n -> (n, 1) :: env) env names
              | PWild -> env
            in
            walk env s)
          arms
    | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
    | SMemFence | SNative _ ->
        ()
  in
  walk [] body ;
  !found

(** Whether [op] is one of the [proven] operations (physical identity). *)
let is_proven proven op = List.exists (fun p -> p == op) proven
