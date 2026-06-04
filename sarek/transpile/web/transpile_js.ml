(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** JS shim: exposes [Sarek_transpile.of_source] as a js_of_ocaml module.

    Exports [globalThis.SarekTranspile] with:
    - [transpile(source, backend)] -> [{ok, code|null, error|null}]
    - [transpileWithAbi(source, backend)] ->
      [{ok, code|null, abi|null, error|null}] On WGSL success [abi] is a parsed
      JS object (JSON.parse of the ABI JSON).
    - [backends] array of supported backend names *)

open Js_of_ocaml

let backend_of_string s =
  match String.lowercase_ascii s with
  | "cuda" -> Some Sarek_transpile.CUDA
  | "opencl" -> Some Sarek_transpile.OpenCL
  | "metal" -> Some Sarek_transpile.Metal
  | "glsl" -> Some Sarek_transpile.GLSL
  | "wgsl" -> Some Sarek_transpile.WGSL
  | _ -> None

let transpile source backend_str =
  let source = Js.to_string source in
  let backend_str = Js.to_string backend_str in
  let result_obj = Js.Unsafe.obj [||] in
  (match backend_of_string backend_str with
  | None ->
      Js.Unsafe.set result_obj "ok" Js._false ;
      Js.Unsafe.set result_obj "code" Js.null ;
      Js.Unsafe.set
        result_obj
        "error"
        (Js.some
           (Js.string
              (Printf.sprintf
                 "unknown backend %S; expected cuda|opencl|metal|glsl|wgsl"
                 backend_str)))
  | Some backend -> (
      match Sarek_transpile.of_source backend source with
      | Ok code ->
          Js.Unsafe.set result_obj "ok" Js._true ;
          Js.Unsafe.set result_obj "code" (Js.some (Js.string code)) ;
          Js.Unsafe.set result_obj "error" Js.null
      | Error e ->
          Js.Unsafe.set result_obj "ok" Js._false ;
          Js.Unsafe.set result_obj "code" Js.null ;
          Js.Unsafe.set
            result_obj
            "error"
            (Js.some (Js.string (Sarek_transpile.string_of_error e))))) ;
  result_obj

let transpile_with_abi source backend_str =
  let source = Js.to_string source in
  let backend_str = Js.to_string backend_str in
  let result_obj = Js.Unsafe.obj [||] in
  (match backend_of_string backend_str with
  | None ->
      Js.Unsafe.set result_obj "ok" Js._false ;
      Js.Unsafe.set result_obj "code" Js.null ;
      Js.Unsafe.set result_obj "abi" Js.null ;
      Js.Unsafe.set
        result_obj
        "error"
        (Js.some
           (Js.string
              (Printf.sprintf
                 "unknown backend %S; expected cuda|opencl|metal|glsl|wgsl"
                 backend_str)))
  | Some backend -> (
      match Sarek_transpile.of_source_with_abi backend source with
      | Ok (code, abi_json) ->
          let abi_js =
            Js.Unsafe.fun_call
              (Js.Unsafe.get
                 (Js.Unsafe.get Js.Unsafe.global (Js.string "JSON"))
                 (Js.string "parse"))
              [|Js.Unsafe.inject (Js.string abi_json)|]
          in
          Js.Unsafe.set result_obj "ok" Js._true ;
          Js.Unsafe.set result_obj "code" (Js.some (Js.string code)) ;
          Js.Unsafe.set result_obj "abi" (Js.some abi_js) ;
          Js.Unsafe.set result_obj "error" Js.null
      | Error e ->
          Js.Unsafe.set result_obj "ok" Js._false ;
          Js.Unsafe.set result_obj "code" Js.null ;
          Js.Unsafe.set result_obj "abi" Js.null ;
          Js.Unsafe.set
            result_obj
            "error"
            (Js.some (Js.string (Sarek_transpile.string_of_error e))))) ;
  result_obj

let () =
  let module_obj = Js.Unsafe.obj [||] in
  Js.Unsafe.set
    module_obj
    "transpile"
    (Js.Unsafe.callback (fun source backend -> transpile source backend)) ;
  Js.Unsafe.set
    module_obj
    "transpileWithAbi"
    (Js.Unsafe.callback (fun source backend ->
         transpile_with_abi source backend)) ;
  Js.Unsafe.set
    module_obj
    "backends"
    (Js.array
       [|
         Js.string "cuda";
         Js.string "opencl";
         Js.string "metal";
         Js.string "glsl";
         Js.string "wgsl";
       |]) ;
  Js.Unsafe.set Js.Unsafe.global "SarekTranspile" module_obj
