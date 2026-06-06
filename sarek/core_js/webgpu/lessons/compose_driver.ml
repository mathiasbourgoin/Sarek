(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Browser driver: kernel-composition course lesson.

    Exports
    [globalThis.SpocCompose.run(kernelA_src, kernelB_src, inputsObj, cb)] where
    [inputsObj] is a JS object [{ a: Array, b: Array }] and
    [cb(resultArray, errorStringOrNull)].

    Composition: [out = a^2 + b], implemented as two kernels run in sequence.
    - Kernel A (square): reads [a], writes [tmp].
    - Kernel B (add): reads [tmp] and [b], writes [out].

    Buffer ownership: all GPU buffers are allocated per-call and destroyed after
    read-back by [Webgpu_runtime.run]; the OCaml float arrays passed as [inputs]
    are not mutated. [cb] fires exactly once on completion or error. *)

open Js_of_ocaml
open Spoc_webgpu_runtime

(* ── JS compatibility shims (jsoo 6.x: no Js.to_int / js_of_int) ─────────── *)

let js_to_int x = int_of_float (Js.to_float x)

let js_of_int n = Js.float (float_of_int n)

let js_of_float v = Js.float v

let js_array_length arr = js_to_int (Js.Unsafe.get arr (Js.string "length"))

(* ── Conversion helpers ───────────────────────────────────────────────────── *)

(** Extract a named float array from a JS object. Returns [[||]] if key absent.
*)
let js_get_float_array obj key =
  let arr_js = Js.Unsafe.get obj (Js.string key) in
  if Js.Unsafe.equals arr_js Js.undefined || Js.Unsafe.equals arr_js Js.null
  then [||]
  else
    let len = js_array_length arr_js in
    Array.init len (fun i -> Js.to_float (Js.Unsafe.get arr_js (js_of_int i)))

(** Build a JS Array from an OCaml float array. *)
let float_array_to_js arr = Js.array (Array.map js_of_float arr)

(** Invoke [cb(result, null)] on success. *)
let cb_ok cb result =
  Js.Unsafe.fun_call
    cb
    [|Js.Unsafe.inject (float_array_to_js result); Js.Unsafe.inject Js.null|]
  |> ignore

(** Invoke [cb(null, errorString)] on error. *)
let cb_err cb msg =
  Js.Unsafe.fun_call
    cb
    [|Js.Unsafe.inject Js.null; Js.Unsafe.inject (Js.string msg)|]
  |> ignore

(* ── Transpile helper ─────────────────────────────────────────────────────── *)

(** Transpile a kernel source string to [(wgsl, abi_json)] or call [cb_err]. *)
let transpile_or_err cb label src =
  match Sarek_transpile.of_source_with_abi Sarek_transpile.WGSL src with
  | Error e ->
      let msg =
        label ^ " transpile error: " ^ Sarek_transpile.string_of_error e
      in
      cb_err cb msg ;
      None
  | Ok (wgsl, abi_json) -> Some (wgsl, abi_json)

(* ── Kernel-B dispatch (step 2 of composition) ───────────────────────────── *)

(** After kernel A delivers [tmp], transpile and run kernel B. *)
let run_kernel_b wgsl_b abi_b n tmp b_arr cb =
  let zeros = Array.make n 0.0 in
  Webgpu_runtime.run
    ~wgsl:wgsl_b
    ~abi_json:abi_b
    ~inputs:[("tmp", tmp); ("b", b_arr); ("out", zeros)]
    ~scalars:[]
    ~outputs_wanted:["out"]
    ~on_done:(fun results ->
      match List.assoc_opt "out" results with
      | None -> cb_err cb "kernel B: output buffer 'out' missing from results"
      | Some out -> cb_ok cb out)
    ~on_error:(fun msg -> cb_err cb ("kernel B GPU error: " ^ msg))
    ()

(* ── Kernel-A dispatch (step 1 of composition) ───────────────────────────── *)

(** Run kernel A; on success chain into [run_kernel_b]. *)
let run_kernel_a wgsl_a abi_a wgsl_b abi_b n a_arr b_arr cb =
  let zeros = Array.make n 0.0 in
  Webgpu_runtime.run
    ~wgsl:wgsl_a
    ~abi_json:abi_a
    ~inputs:[("a", a_arr); ("tmp", zeros)]
    ~scalars:[]
    ~outputs_wanted:["tmp"]
    ~on_done:(fun results ->
      match List.assoc_opt "tmp" results with
      | None -> cb_err cb "kernel A: output buffer 'tmp' missing from results"
      | Some tmp -> run_kernel_b wgsl_b abi_b n tmp b_arr cb)
    ~on_error:(fun msg -> cb_err cb ("kernel A GPU error: " ^ msg))
    ()

(* ── Public entry point ───────────────────────────────────────────────────── *)

(** [run_fn kA_src kB_src inputs_obj cb] -- the JS-callable composition driver.

    [inputs_obj] must expose [.a] and [.b] (JS arrays of equal length). [cb] is
    called as [cb(resultArray, null)] on success or [cb(null, errorString)] on
    any failure. *)
let run_fn kernel_a_src kernel_b_src inputs_obj cb =
  let src_a = Js.to_string kernel_a_src in
  let src_b = Js.to_string kernel_b_src in
  let a_arr = js_get_float_array inputs_obj "a" in
  let b_arr = js_get_float_array inputs_obj "b" in
  let n = Array.length a_arr in
  match transpile_or_err cb "kernel A" src_a with
  | None -> ()
  | Some (wgsl_a, abi_a) -> (
      match transpile_or_err cb "kernel B" src_b with
      | None -> ()
      | Some (wgsl_b, abi_b) ->
          run_kernel_a wgsl_a abi_a wgsl_b abi_b n a_arr b_arr cb)

let () =
  let module_obj = Js.Unsafe.obj [||] in
  Js.Unsafe.set
    module_obj
    "run"
    (Js.Unsafe.callback (fun kA kB inputs cb -> run_fn kA kB inputs cb)) ;
  Js.Unsafe.set Js.Unsafe.global "SpocCompose" module_obj
