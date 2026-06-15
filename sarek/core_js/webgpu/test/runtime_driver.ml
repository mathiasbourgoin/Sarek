(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Browser driver for the SpocRT end-to-end test.

    Exports
    [globalThis.SpocRT.run(kernelSrc, inputsObj, scalarsObj, outName, cb)]
    which: 1. Calls [Sarek_transpile.of_source_with_abi WGSL kernelSrc] to
    obtain [(wgsl, abi_json)]. 2. Converts the JS [inputsObj] / [scalarsObj] to
    the OCaml input types. 3. Calls [Webgpu_runtime.run] with the parsed data.
    4. Delivers [outputs[outName]] as a plain JS array to [cb(null, array)] on
    success, or [cb(errorString, null)] on failure. *)

open Js_of_ocaml
open Spoc_webgpu_runtime

(* Compatibility shims for jsoo 6.x: no Js.to_int/of_int/of_float. *)
let js_to_int x = int_of_float (Js.to_float x)

let js_of_int n = Js.float (float_of_int n)

let js_of_float v = Js.float v

let js_get obj key = Js.Unsafe.get obj (Js.string key)

let js_array_length arr = js_to_int (Js.Unsafe.get arr (Js.string "length"))

(** Convert a JS object to an OCaml assoc list of (string * float array). *)
let js_obj_to_inputs obj =
  (* Use Object.keys to enumerate the properties. *)
  let keys =
    Js.Unsafe.fun_call
      (Js.Unsafe.get
         (Js.Unsafe.get Js.Unsafe.global (Js.string "Object"))
         (Js.string "keys"))
      [|Js.Unsafe.inject obj|]
  in
  let n = js_array_length keys in
  List.init n (fun i ->
      let key = Js.to_string (Js.Unsafe.get keys (js_of_int i)) in
      let arr_js = js_get obj (Js.to_string (Js.string key)) in
      let len = js_array_length arr_js in
      let data =
        Array.init len (fun j ->
            Js.to_float (Js.Unsafe.get arr_js (js_of_int j)))
      in
      (key, data))

(** Convert a JS object to an OCaml assoc list of (string * float). *)
let js_obj_to_scalars obj =
  let keys =
    Js.Unsafe.fun_call
      (Js.Unsafe.get
         (Js.Unsafe.get Js.Unsafe.global (Js.string "Object"))
         (Js.string "keys"))
      [|Js.Unsafe.inject obj|]
  in
  let n = js_array_length keys in
  List.init n (fun i ->
      let key = Js.to_string (Js.Unsafe.get keys (js_of_int i)) in
      let v = Js.to_float (js_get obj (Js.to_string (Js.string key))) in
      (key, v))

let run_fn kernel_src inputs_obj scalars_obj out_name cb =
  let src = Js.to_string kernel_src in
  let out_name_s = Js.to_string out_name in
  match Sarek_transpile.of_source_with_abi Sarek_transpile.WGSL src with
  | Error e ->
      let msg = "transpile error: " ^ Sarek_transpile.string_of_error e in
      Js.Unsafe.fun_call
        cb
        [|Js.Unsafe.inject (Js.string msg); Js.Unsafe.inject Js.null|]
      |> ignore
  | Ok (wgsl, abi_json) ->
      let inputs = js_obj_to_inputs inputs_obj in
      let scalars = js_obj_to_scalars scalars_obj in
      Webgpu_runtime.run
        ~wgsl
        ~abi_json
        ~inputs
        ~scalars
        ~outputs_wanted:[out_name_s]
        ~on_done:(fun results ->
          let out_arr =
            match List.assoc_opt out_name_s results with
            | None -> [||]
            | Some a -> a
          in
          let js_arr = Js.array (Array.map js_of_float out_arr) in
          Js.Unsafe.fun_call
            cb
            [|Js.Unsafe.inject Js.null; Js.Unsafe.inject js_arr|]
          |> ignore)
        ~on_error:(fun msg ->
          Js.Unsafe.fun_call
            cb
            [|Js.Unsafe.inject (Js.string msg); Js.Unsafe.inject Js.null|]
          |> ignore)
        ()

let () =
  let module_obj = Js.Unsafe.obj [||] in
  Js.Unsafe.set
    module_obj
    "run"
    (Js.Unsafe.callback (fun src inputs scalars out_name cb ->
         run_fn src inputs scalars out_name cb)) ;
  Js.Unsafe.set Js.Unsafe.global "SpocRT" module_obj
