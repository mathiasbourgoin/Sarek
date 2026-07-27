(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative-matrix vocabulary and fragment type (backlog-62, slices 2 and
    4b). See the .mli for why integers are admitted from the start and why the
    device gate cannot be keyed on the configuration list. *)

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

let device_lacks_config cfg =
  {
    Sarek_capability.cap_name = "cooperative-matrix";
    (* Device_optional, per docs/design/capability-model.md §2: the backend can
       spell the instruction, a given device may not provide it. The local box
       has one device that does and one that does not under the SAME driver,
       which is why a driver-keyed or backend-keyed refusal would be wrong
       here. *)
    cap_kind = Sarek_capability.Device_optional;
    cap_why =
      Printf.sprintf
        "the device advertises no cooperative-matrix configuration matching %s \
         (VK_KHR_cooperative_matrix absent, its cooperativeMatrix feature \
         false, or no advertised configuration with these dimensions and \
         component types)"
        (config_name cfg);
    cap_evidence =
      Sarek_capability.Measured
        "AMD Radeon RX 7900 XTX (RADV NAVI31) advertises \
         VK_KHR_cooperative_matrix revision 2 with 14 configurations; the AMD \
         Ryzen 9 7950X iGPU (RADV RAPHAEL_MENDOCINO) does not advertise the \
         extension and reports cooperativeMatrix = false, under the same radv \
         / Mesa 26.1.4-arch3.1 / Vulkan 1.4.354";
    cap_remedy =
      Some
        "Use an ordinary multiply-accumulate kernel, or select a device that \
         advertises this cooperative-matrix configuration.";
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

let verdict ~support cfg =
  match support with
  | None ->
      Sarek_capability.Unknown
        "device cooperative-matrix support was not probed, so no configuration \
         can be confirmed"
  | Some s ->
      if
        List.exists
          (fun advertised ->
            config_matches
              ~shape:cfg.cfg_shape
              ~a:cfg.cfg_a
              ~b:cfg.cfg_b
              ~c:cfg.cfg_c
              ~result:cfg.cfg_result
              advertised
            && advertised.cfg_saturating = cfg.cfg_saturating
            && advertised.cfg_scope = cfg.cfg_scope)
          s.ds_configs
      then Sarek_capability.Available
      else Sarek_capability.Unavailable (device_lacks_config cfg)

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
