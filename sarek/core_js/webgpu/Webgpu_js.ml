(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Low-level js_of_ocaml glue for the WebGPU runtime: promise helpers, JS
    object/array accessors, typed-array construction, and the WebGPU global enum
    readers. Kept separate from {!Webgpu_runtime} so each file stays within the
    project size limits. *)

open Js_of_ocaml

(* jsoo 6.x: Js.to_float/Js.float exist; Js.to_int/Js.of_int/Js.of_float do not. *)
let js_to_int x = int_of_float (Js.to_float x)

let js_of_int n = Js.float (float_of_int n)

let js_of_float v = Js.float v

let then_ p f =
  ignore
    (Js.Unsafe.meth_call p "then" [|Js.Unsafe.inject (Js.wrap_callback f)|])

let catch_ p f =
  ignore
    (Js.Unsafe.meth_call p "catch" [|Js.Unsafe.inject (Js.wrap_callback f)|])

let js_global name = Js.Unsafe.get Js.Unsafe.global (Js.string name)

let gpu =
  js_global "navigator" |> fun nav -> Js.Unsafe.get nav (Js.string "gpu")

let json_parse s =
  Js.Unsafe.fun_call
    (Js.Unsafe.get (js_global "JSON") (Js.string "parse"))
    [|Js.Unsafe.inject (Js.string s)|]

let js_get obj key = Js.Unsafe.get obj (Js.string key)

let js_int obj key = js_to_int (js_get obj key)

let js_str obj key = Js.to_string (js_get obj key)

let js_arr_len arr = js_to_int (Js.Unsafe.get arr (Js.string "length"))

let js_arr_get arr i = Js.Unsafe.get arr (js_of_int i)

let js_is_null_or_undef v =
  Js.Optdef.case
    (Js.Optdef.return v)
    (fun () -> true)
    (fun v' -> Js.Opt.case (Js.Opt.return v') (fun () -> true) (fun _ -> false))

let make_ta ctor (data : float array) =
  let arr =
    Js.Unsafe.new_obj ctor [|Js.Unsafe.inject (js_of_int (Array.length data))|]
  in
  Array.iteri
    (fun i v ->
      Js.Unsafe.set arr (js_of_int i) (Js.Unsafe.inject (js_of_float v)))
    data ;
  arr

let typed_array et data =
  match et with
  | "f32" -> make_ta (js_global "Float32Array") data
  | "i32" -> make_ta (js_global "Int32Array") data
  | "u32" -> make_ta (js_global "Uint32Array") data
  | _ -> failwith ("SpocRT: unknown elementType: " ^ et)

let bpe = function
  | "f32" | "i32" | "u32" -> 4
  | et -> failwith ("SpocRT: unknown elementType: " ^ et)

let usage s =
  js_to_int (Js.Unsafe.get (js_global "GPUBufferUsage") (Js.string s))

let map_read () =
  js_to_int (Js.Unsafe.get (js_global "GPUMapMode") (Js.string "READ"))

let shader_compute () =
  js_to_int (Js.Unsafe.get (js_global "GPUShaderStage") (Js.string "COMPUTE"))

let js_err err =
  Js.Optdef.case
    (Js.Optdef.return (js_get err "message"))
    (fun () -> "unknown error")
    Js.to_string
