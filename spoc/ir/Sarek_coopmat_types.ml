(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative-matrix vocabulary, fragment type and subgroup calling convention
    — the half with NO dependencies (backlog-62, slices 2, 4b and 3).

    Split out of {!Sarek_coopmat} in slice 3 and for one reason only: the IR
    must be able to name a fragment. [Sarek_ir_types.coopmat_op] carries a
    {!fragment} and a {!config}, and [Sarek_capability] depends on
    [Sarek_ir_types] — so leaving the vocabulary in a module that also depends
    on [Sarek_capability] closes a cycle
    [Sarek_ir_types -> Sarek_coopmat -> Sarek_capability -> Sarek_ir_analysis ->
     Sarek_ir_types] that dune rejects outright.

    The cut is at the capability boundary, which is also where it belongs
    conceptually: WHAT a cooperative-matrix configuration is has no opinion
    about whether a device provides it. {!Sarek_coopmat} [include]s this module,
    so every existing [Sarek_coopmat.fragment] / [Sarek_coopmat.Uint8] reference
    keeps its meaning and its type equality; nothing outside needs to know the
    module was split.

    The .mli of {!Sarek_coopmat} carries the design rationale for every type
    here — why integers are admitted from the start, why the shape is a record
    rather than a closed variant, why an unrepresentable scope must not be
    rewritten. It is not repeated. *)

type component_type = Float16 | Float32 | Uint8 | Sint8 | Uint32 | Sint32

let component_name = function
  | Float16 -> "f16"
  | Float32 -> "f32"
  | Uint8 -> "u8"
  | Sint8 -> "s8"
  | Uint32 -> "u32"
  | Sint32 -> "s32"

let component_bits = function
  | Uint8 | Sint8 -> 8
  | Float16 -> 16
  | Float32 | Uint32 | Sint32 -> 32

(* An explicit match on every constructor rather than [<> Float16 && <>
   Float32]. A new float component type (bf16, fp8) added to the variant must
   be a compile error here, because the one thing this predicate decides is
   whether a configuration escapes the accuracy relaxation — and a new float
   type defaulting to "integer, therefore exact" is the permissive-default
   failure the capability model exists to prevent. *)
let component_is_integer = function
  | Uint8 | Sint8 | Uint32 | Sint32 -> true
  | Float16 | Float32 -> false

type scope = Subgroup | Workgroup | Device_scope | Queue_family

let scope_name = function
  | Subgroup -> "subgroup"
  | Workgroup -> "workgroup"
  | Device_scope -> "device"
  | Queue_family -> "queuefamily"

type shape = {m : int; n : int; k : int}

let shape_name s = Printf.sprintf "%dx%dx%d" s.m s.n s.k

type config = {
  cfg_shape : shape;
  cfg_a : component_type;
  cfg_b : component_type;
  cfg_c : component_type;
  cfg_result : component_type;
  cfg_saturating : bool;
  cfg_scope : scope;
}

let config_name c =
  Printf.sprintf
    "%s %s*%s+%s->%s%s %s"
    (shape_name c.cfg_shape)
    (component_name c.cfg_a)
    (component_name c.cfg_b)
    (component_name c.cfg_c)
    (component_name c.cfg_result)
    (if c.cfg_saturating then " saturating" else "")
    (scope_name c.cfg_scope)

(* Exactness is a property of the ACCUMULATION, so it is decided by the addend
   and result types, not by the operand types. The distinction is not academic
   on this hardware: every configuration the local device advertises has
   integer operands paired with integer accumulate, but f16*f16 pairs with BOTH
   f16 and f32 accumulate, and reading the operand types would have called both
   of those the same thing. *)
let accumulation_is_exact c =
  component_is_integer c.cfg_c && component_is_integer c.cfg_result

type accuracy_regime = Strict | Relaxed_bounded

let regime c = if accumulation_is_exact c then Strict else Relaxed_bounded

let regime_name = function
  | Strict -> "strict"
  | Relaxed_bounded -> "relaxed-bounded"

type device_support = {
  ds_configs : config list;
  ds_robust_buffer_access : bool;
  ds_subgroup_size : int;
  ds_advertised_count : int;
}

let config_matches ~shape ~a ~b ~c ~result cfg =
  cfg.cfg_shape = shape && cfg.cfg_a = a && cfg.cfg_b = b && cfg.cfg_c = c
  && cfg.cfg_result = result

let find_config ~support ~shape ~a ~b ~c ~result =
  match support with
  | None -> None
  | Some s -> (
      let candidates =
        List.filter (config_matches ~shape ~a ~b ~c ~result) s.ds_configs
      in
      (* Prefer the non-saturating variant: it is the one that computes the
         same function as a plain accumulate, so it is what an unqualified
         request means. A caller that wants saturation must ask for it, which
         it cannot do through this function — deliberately, until a slice
         exists that can emit either. *)
      match List.find_opt (fun cfg -> not cfg.cfg_saturating) candidates with
      | Some cfg -> Some cfg
      | None -> List.nth_opt candidates 0)

type use = Matrix_a | Matrix_b | Accumulator

let use_name = function
  | Matrix_a -> "A"
  | Matrix_b -> "B"
  | Accumulator -> "accumulator"

type fragment = {
  frag_use : use;
  frag_shape : shape;
  frag_component : component_type;
  frag_scope : scope;
}

let fragment_dims f =
  let s = f.frag_shape in
  match f.frag_use with
  | Matrix_a -> (s.m, s.k)
  | Matrix_b -> (s.k, s.n)
  | Accumulator -> (s.m, s.n)

let fragment_components f =
  let rows, cols = fragment_dims f in
  rows * cols

let fragments_of_config c =
  let frag frag_use frag_component =
    {
      frag_use;
      frag_shape = c.cfg_shape;
      frag_component;
      frag_scope = c.cfg_scope;
    }
  in
  ( frag Matrix_a c.cfg_a,
    frag Matrix_b c.cfg_b,
    frag Accumulator c.cfg_c,
    frag Accumulator c.cfg_result )

let components_per_invocation ~subgroup_size f =
  let total = fragment_components f in
  if subgroup_size <= 0 then
    Error
      (Printf.sprintf
         "subgroup size %d is not positive, so a %s fragment cannot be \
          distributed"
         subgroup_size
         (use_name f.frag_use))
  else if total mod subgroup_size <> 0 then
    Error
      (Printf.sprintf
         "a %s fragment of %s (%d components) cannot be distributed over a \
          subgroup of %d invocations: %d does not divide %d"
         (use_name f.frag_use)
         (shape_name f.frag_shape)
         total
         subgroup_size
         subgroup_size
         total)
  else Ok (total / subgroup_size)

let config_fits_subgroup ~subgroup_size c =
  let a, b, cc, r = fragments_of_config c in
  List.for_all
    (fun f ->
      match components_per_invocation ~subgroup_size f with
      | Ok _ -> true
      | Error _ -> false)
    [a; b; cc; r]
