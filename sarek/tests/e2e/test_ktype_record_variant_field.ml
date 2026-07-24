(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E: a [@@sarek.type] record with a VARIANT-typed field, used as a VECTOR
 * ELEMENT (host-created, kernel-visible), round-tripping host->device->host.
 *
 * This is the RECORD-CONTAINS-VARIANT marshalling gap found during L14-S2
 * (PR #251): the deriver's generated interpreter helpers treated any
 * non-primitive record field as a [VRecord], so a variant field failed with
 * `Field 'kind' expected record` at the host/interpreter boundary. The fix
 * generates a value-model helper for the variant type too, and the record
 * helper delegates the field to it, so the field decodes/encodes as a
 * [VVariant].
 *
 * Scope (see briefs/deriver-variant-fields-impl.md): the interpreter is
 * value-based (no byte layout), so the NON-erasable / runtime-selected case
 * works here on Interpreter + Native. Device backends (CUDA/OpenCL/Vulkan)
 * would need a nested-variant byte layout in Sarek_ir_layout, which is
 * Rocq-coupled and out of scope; the erasable case already runs there via the
 * L14-S2 [_erec_] synthesis. This test therefore runs on Interpreter + Native
 * and verifies against a pure-OCaml reference.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

let () = Test_helpers.Benchmarks.init ()

type float32 = float

(* A variant with a nullary constructor, a second nullary, and a float32
   payload constructor - enough to exercise a tag switch and payload
   marshalling in both directions. *)
type color = Red | Green | Shade of float32 [@@sarek.type]

(* The record under test: a variant-typed field [kind] beside a scalar. *)
type cell = {kind : color; scale : float32} [@@sarek.type]

(* Kernel over a [cell vector]: read the element (host->interp decode of a
   VRecord whose [kind] field is a VVariant), tag-switch on the variant field,
   and write a cell back whose [kind] is the (runtime-selected) passed-through
   variant field (interp->host encode). *)
let cell_kirc =
  snd
    [%kernel
      fun (src : cell vector) (out : cell vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let c = src.(tid) in
          let s =
            match c.kind with Red -> 10.0 | Green -> 20.0 | Shade f -> f
          in
          out.(tid) <- {kind = c.kind; scale = c.scale +. s}
        end]

(* Standalone variant vector (D1): read a [color vector] element (VVariant
   decode) and write a transformed [color] back (VVariant encode). *)
let color_kirc =
  snd
    [%kernel
      fun (src : color vector) (out : color vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then
          out.(tid) <-
            (match src.(tid) with
            | Red -> Green
            | Green -> Red
            | Shade f -> Shade (f +. 1.0))]

let n = 64

let mk_color i =
  match i mod 3 with 0 -> Red | 1 -> Green | _ -> Shade (float_of_int i)

(* Pure-OCaml references. *)
let ref_cell {kind; scale} =
  let s = match kind with Red -> 10.0 | Green -> 20.0 | Shade f -> f in
  {kind; scale = scale +. s}

let ref_color = function
  | Red -> Green
  | Green -> Red
  | Shade f -> Shade (f +. 1.0)

let color_eq a b =
  match (a, b) with
  | Red, Red | Green, Green -> true
  | Shade x, Shade y -> abs_float (x -. y) < 1e-3
  | _ -> false

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "Kernel has no IR"

let launch dev kirc args =
  let threads = min 64 n in
  let grid_x = (n + threads - 1) / threads in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d grid_x)
    ~ir:(ir_of kirc)
    ~args
    () ;
  Transfer.flush dev

let run_cell dev =
  let src = Vector.create_custom cell_custom n in
  let out = Vector.create_custom cell_custom n in
  for i = 0 to n - 1 do
    Vector.set src i {kind = mk_color i; scale = float_of_int i}
  done ;
  launch
    dev
    cell_kirc
    [
      Sarek.Execute.Vec src;
      Sarek.Execute.Vec out;
      Sarek.Execute.Int32 (Int32.of_int n);
    ] ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let expected = ref_cell (Vector.get src i) in
    let got = Vector.get out i in
    if
      (not (color_eq expected.kind got.kind))
      || abs_float (expected.scale -. got.scale) > 1e-3
    then ok := false
  done ;
  !ok

let run_color dev =
  let src = Vector.create_custom color_custom n in
  let out = Vector.create_custom color_custom n in
  for i = 0 to n - 1 do
    Vector.set src i (mk_color i)
  done ;
  launch
    dev
    color_kirc
    [
      Sarek.Execute.Vec src;
      Sarek.Execute.Vec out;
      Sarek.Execute.Int32 (Int32.of_int n);
    ] ;
  let ok = ref true in
  for i = 0 to n - 1 do
    if not (color_eq (ref_color (Vector.get src i)) (Vector.get out i)) then
      ok := false
  done ;
  !ok

let () =
  let required = ["Interpreter"; "Native"] in
  let devs = Device.init ~frameworks:required () in
  (* Both backends are the stated contract: a Native or Interpreter regression
     must not be silently untested because one failed to initialize. *)
  let available =
    Array.to_list devs |> List.map (fun d -> d.Device.framework)
  in
  let missing = List.filter (fun f -> not (List.mem f available)) required in
  if missing <> [] then begin
    Printf.eprintf
      "Missing required backends: %s\n%!"
      (String.concat ", " missing) ;
    exit 1
  end ;
  let any_failure = ref false in
  Array.iter
    (fun dev ->
      Printf.printf "runtime [%s] %s:\n%!" dev.Device.framework dev.Device.name ;
      let check label f =
        try
          if f dev then Printf.printf "  %s: PASSED\n%!" label
          else begin
            any_failure := true ;
            Printf.printf "  %s: FAILED\n%!" label
          end
        with
        | Sarek.Interp_error.Interpreter_error err ->
            any_failure := true ;
            Printf.printf
              "  %s: ERROR (%s)\n%!"
              label
              (Sarek.Interp_error.error_to_string err)
        | e ->
            any_failure := true ;
            Printf.printf "  %s: ERROR (%s)\n%!" label (Printexc.to_string e)
      in
      check "record-with-variant-field round-trip" run_cell ;
      check "standalone variant vector round-trip" run_color)
    devs ;
  if !any_failure then exit 1
  else print_endline "test_ktype_record_variant_field PASSED"
