(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Field-name resolution for a record field STORE agrees with the one for a READ.
 *
 * backlog-172 round 6. [Sarek_ir_interp_eval.assign_lvalue]'s [LRecordField] arm
 * resolves a field name to a slot in the [VRecord]'s [value array]. So does
 * [read_lvalue]'s [LRecordField] arm, on the way to fetching the base. They used
 * to disagree for a REGISTERED type:
 *
 *   read  (Some h): h.field_index field, and raise if it answers None.
 *   write (Some h): h.field_index field, and if None fall back to
 *                   positional_field_index field.
 *
 * [positional_field_index] recognises the tuple convention [_0], [_1], … which
 * synthesized tuple records use — those are never registered, which is why the
 * fallback exists at all under [None]. Under [Some h] it meant that a registered
 * record asked for a field it does not have, whose name happens to look
 * positional, had that name READ as an error and WRITTEN into whatever field sits
 * at that position. A store that silently lands in the wrong field is a worse
 * failure than the dropped store the rest of backlog-172 removes: a dropped store
 * leaves correct data, and this one does not.
 *
 * The previous comment in that arm described the divergence and called it benign
 * because the name "cannot occur in practice". That is an argument about what the
 * front end emits, not a property of the function, and it is the shape this
 * repository keeps paying for. The fallback under [Some h] is gone; write now
 * resolves exactly as read does.
 *
 * WHAT THIS FILE PINS, and why it needs a registered type with a HOLE in it: the
 * divergence is only reachable when [h.field_index] answers [None] for a name
 * [positional_field_index] answers [Some] for. So the mock below registers
 * [field_names = ["a"; "b"]] — no [_0] — and the cases drive both [_0] (the
 * positional-looking hole) and [b] (a real field) through the same store path.
 *
 * MEASURED RED, restoring the fallback on the [assign_lvalue] arm ALONE (line
 * position asserted, not pattern-matched — the two arms now carry identical text,
 * and a `perl -0pi` for it hits [read_lvalue] first, which mutates the wrong path
 * and produces a red that means something else):
 *
 *   case 1  FAIL  slot 0 is untouched: expected 1.0, got 99.0
 *   case 2  OK    (positive control: a real field still stores)
 *   case 3  FAIL  read and write agree on whether "_0" resolves: expected true,
 *                 got false
 *
 * Case 2 is what keeps "refuses" from being "refuses everything". Case 3 is
 * symmetric and catches the divergence from EITHER side: mutating [read_lvalue]
 * instead — which is what the first attempt at this proof actually did — leaves
 * cases 1 and 2 green and reddens only case 3.
 ******************************************************************************)

open Alcotest
module V = Sarek_interp.Sarek_value
module H = Sarek_interp.Sarek_type_helpers
module Eval = Sarek_interp.Sarek_ir_interp_eval
module Interp = Sarek_interp.Sarek_ir_interp
module Interp_error = Sarek_interp.Interp_error
module T = Sarek_ir_types

let type_name = "test_fsr_pair"

(* Two named fields and deliberately NO [_0]: the hole is the point. Registered,
   so [Sarek_type_helpers.lookup] answers [Some h] and the store path takes the
   arm under test rather than the positional branch. *)
module Pair_helpers : H.HELPERS with type t = float * float = struct
  type t = float * float

  let type_id = T.Type_id.create ()

  let from_values arr =
    match arr with
    | [|V.VFloat32 a; V.VFloat32 b|] -> (a, b)
    | _ -> failwith "invalid test_fsr_pair values"

  let to_values (a, b) = [|V.VFloat32 a; V.VFloat32 b|]

  let get_field (a, b) name =
    match name with
    | "a" -> V.VFloat32 a
    | "b" -> V.VFloat32 b
    | _ -> failwith ("unknown field: " ^ name)

  (* Same order as [to_values]. No [_0]. *)
  let field_names = ["a"; "b"]

  let to_value (a, b) = V.VRecord (type_name, [|V.VFloat32 a; V.VFloat32 b|])

  let from_value = function
    | V.VRecord (_, [|V.VFloat32 a; V.VFloat32 b|]) -> (a, b)
    | _ -> failwith "invalid test_fsr_pair record"
end

let () = H.register type_name (H.AnyHelpers (module Pair_helpers))

let state : Interp.thread_state =
  {
    thread_idx = (0, 0, 0);
    block_idx = (0, 0, 0);
    block_dim = (1, 1, 1);
    grid_dim = (1, 1, 1);
  }

(* A record of the registered type bound to a variable, so [LRecordField (LVar v,
   field)] resolves its base through [read_lvalue] exactly as the interpreter
   does. The [value array] is handed back by reference, which is what makes the
   store observable here at all. *)
let bind_record () =
  let env = Interp.create_env () in
  let fields = [|V.VFloat32 1.0; V.VFloat32 2.0|] in
  let var =
    {
      T.var_name = "r";
      var_id = 1;
      var_type = T.TRecord (type_name, []);
      var_mutable = true;
    }
  in
  Hashtbl.replace env.Interp.vars var.T.var_id (V.VRecord (type_name, fields)) ;
  (env, var, fields)

let store env var field value =
  Eval.assign_lvalue state env (T.LRecordField (T.LVar var, field)) value

(* Case 1. THE DEFECT. [_0] is not a field of this registered type, and it looks
   positional. It must be refused, not resolved to slot 0.

   Both halves are asserted: that neither slot moved, AND that it raises.

   THE SLOT CHECK COMES FIRST, deliberately. [Alcotest.check] raises on the first
   failure, so whichever assertion is written first is the one a reader sees at
   the pre-fix polarity — and "slot 0 holds 99 instead of 1" names the actual
   defect (a wrong-field WRITE), where "it did not raise" only names a missing
   diagnostic. Ordering them the other way round left the slot assertion never
   observed failing, which is a gate whose red is taken on faith. *)
let test_positional_name_on_registered_type_is_refused () =
  let env, var, fields = bind_record () in
  let raised =
    match store env var "_0" (V.VFloat32 99.0) with
    | () -> false
    | exception Interp_error.Interpreter_error _ -> true
  in
  (* Slot 0 is where the fallback wrote. *)
  (match fields.(0) with
  | V.VFloat32 x ->
      check
        (float 0.0001)
        "slot 0 is untouched: a name the type does not have must not resolve \
         to a position"
        1.0
        x
  | v -> failf "slot 0 changed shape: %s" (V.value_type_name v)) ;
  (match fields.(1) with
  | V.VFloat32 x -> check (float 0.0001) "slot 1 is untouched" 2.0 x
  | v -> failf "slot 1 changed shape: %s" (V.value_type_name v)) ;
  check
    bool
    "and the store is refused rather than silently doing nothing"
    true
    raised

(* Case 2 (positive control). A real field still stores, in the right slot. If
   the fix had tightened resolution too far this goes red, which is the polarity
   a "must raise" case cannot see. *)
let test_real_field_still_stores () =
  let env, var, fields = bind_record () in
  store env var "b" (V.VFloat32 42.0) ;
  (match fields.(1) with
  | V.VFloat32 x -> check (float 0.0001) "b stored into slot 1" 42.0 x
  | v -> failf "slot 1 changed shape: %s" (V.value_type_name v)) ;
  match fields.(0) with
  | V.VFloat32 x -> check (float 0.0001) "a untouched" 1.0 x
  | v -> failf "slot 0 changed shape: %s" (V.value_type_name v)

(* Case 3 (positive control, the other end). The store path and the READ path
   answer the same way for the same name — asserted rather than assumed, since
   "they agree" is the whole claim and the two arms are separate code. *)
let test_read_and_write_agree_on_the_same_name () =
  let env, var, _ = bind_record () in
  let read_raises field =
    match Eval.read_lvalue state env (T.LRecordField (T.LVar var, field)) with
    | _ -> false
    | exception Interp_error.Interpreter_error _ -> true
  in
  let write_raises field =
    let env, var, _ = bind_record () in
    match store env var field (V.VFloat32 7.0) with
    | () -> false
    | exception Interp_error.Interpreter_error _ -> true
  in
  List.iter
    (fun field ->
      check
        bool
        (Printf.sprintf
           "read and write agree on whether %S resolves on a registered type"
           field)
        (read_raises field)
        (write_raises field))
    ["_0"; "_1"; "b"; "a"; "nope"]

let () =
  run
    "interp field-store resolution"
    [
      ( "registered-type field resolution",
        [
          test_case
            "a positional-looking absent field is refused"
            `Quick
            test_positional_name_on_registered_type_is_refused;
          test_case
            "a real field still stores"
            `Quick
            test_real_field_still_stores;
          test_case
            "read and write agree"
            `Quick
            test_read_and_write_agree_on_the_same_name;
        ] );
    ]
