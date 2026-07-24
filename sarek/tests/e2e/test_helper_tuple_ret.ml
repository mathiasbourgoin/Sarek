(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for a module-level HELPER FUNCTION whose RETURN type is a
 * primitive-component tuple ([let mk x y = (x, y)]).
 *
 * This is the third "wrong-width" instance of the class the L13 local-tuple /
 * vector-of-tuple work already closed for data slots. A module-level helper's
 * return type was typed by the bare [elttype_of_typ] placeholder in
 * [Sarek_lower_ir.lower_expr] (the [TEApp]/fun_map branch), which maps
 * [TTuple]/[TFun] to [Ir.TInt32]. The helper body, however, lowers a primitive
 * tuple literal to the synthesized positional record ([_tup_*], fields
 * [_0.._n]), so the emitted helper was declared int-returning while its body
 * returned a compound value:
 *
 *     __device__ int mk(float x, float y) {
 *       return (struct _tup_float32_float32){ ... };   // int-vs-compound
 *     }
 *
 * — a silent miscompile with no diagnostic. The fix routes the helper return
 * type through [slot_elttype_of_typ] (the same data mapper vector/local slots
 * use) so a primitive-tuple return lowers to the synthesized [_tup_*] record
 * end-to-end, matching the body.
 *
 * Axis 1 (codegen): the emitted CUDA/OpenCL/GLSL declares the helper with the
 *   synthesized struct return type and defines that struct — never [int mk].
 * Axis 2 (behaviour): results match a pure-OCaml reference on every available
 *   device (CUDA/PTX under ZLUDA, OpenCL, Vulkan, Metal, Native, Interpreter).
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

type float32 = float

(* A: all-float32 tuple return. dst[i] = a + b with (a, b) = mk src[i] (src[i]+1)
   = (src[i], src[i]+1), so dst[i] = 2*src[i] + 1. *)
let k_ff =
  snd
    [%kernel
      let mk (x : float32) (y : float32) : float32 * float32 = (x, y) in
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let open Std in
        let tid = global_thread_id in
        if tid < n then begin
          let p = mk src.(tid) (src.(tid) +. 1.0) in
          match p with a, b -> dst.(tid) <- a +. b
        end]

(* B: mixed (float32 * int32) tuple return; int component consumed via float.
   dst[i] = a + float_of_int b with (a, b) = mk src[i] tid = (src[i], tid). *)
let k_fi =
  snd
    [%kernel
      let mkm (x : float32) (i : int32) : float32 * int32 = (x, i) in
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let open Std in
        let tid = global_thread_id in
        if tid < n then begin
          let p = mkm src.(tid) tid in
          match p with a, b -> dst.(tid) <- a +. float b
        end]

(* ---- OCaml references (must mirror the kernels exactly) ------------------ *)

let ref_src i = float_of_int i -. 8.0

let ref_ff i = ref_src i +. (ref_src i +. 1.0)

let ref_fi i = ref_src i +. float_of_int i

(* ---- Axis 1: emitted-source structural check ----------------------------- *)

let ir_of name kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith ("no IR for " ^ name)

let contains hay needle =
  let nh = String.length needle and h = String.length hay in
  let rec go i =
    if i + nh > h then false
    else if String.sub hay i nh = needle then true
    else go (i + 1)
  in
  nh = 0 || go 0

let codegen_ok = ref true

(* Pin the EXACT wrong-width signature the fix targets. Three direct assertions
   on the emitted source, one per backend:

   (a) The helper is DECLARED returning the synthesized [_tup_*] record — the
       emitted source contains ["<struct_name> <helper>("] (the struct-return
       form all three C-family backends emit: [cuda/opencl/glsl_type_of_elttype]
       maps the tuple's [TRecord] to the mangled [_tup_*] name, then a space,
       then the helper name and its parameter list).
   (b) It is NOT declared int-returning — the emitted source must NOT contain
       ["int <helper>("], the old [elttype_of_typ] placeholder collapse
       ([TTuple] -> [Ir.TInt32]) that produced the int-vs-compound miscompile.
   (c) The [_tup_*] struct DEFINITION is emitted (not merely the name in a
       literal/usage). CUDA and OpenCL emit a C typedef (["} <struct_name>;"]);
       GLSL emits a tagged struct (["struct <struct_name> {"]). Either form
       counts as a definition. *)
let check_codegen name kirc ~struct_name ~helper =
  let ir = ir_of name kirc in
  let types = ir.Sarek_ir_types.kern_types in
  let backends =
    [
      ("CUDA", Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types ir);
      ("OpenCL", Sarek_codegen.Sarek_ir_opencl.generate_with_types ~types ir);
      ("GLSL", Sarek_codegen.Sarek_ir_glsl.generate_with_types ~types ir);
    ]
  in
  List.iter
    (fun (bk, src) ->
      (* (a) helper declared returning the synthesized struct type. *)
      let struct_ret = contains src (struct_name ^ " " ^ helper ^ "(") in
      (* (b) the int-vs-compound miscompile: helper declared "int <helper>(". *)
      let mistyped = contains src ("int " ^ helper ^ "(") in
      (* (c) the struct definition itself (typedef for CUDA/OpenCL, tagged
         struct for GLSL) — not just the mangled name in a literal. *)
      let struct_def =
        contains src ("} " ^ struct_name ^ ";")
        || contains src ("struct " ^ struct_name ^ " {")
      in
      if (not struct_ret) || mistyped || not struct_def then begin
        codegen_ok := false ;
        Printf.printf
          "  codegen[%s/%s]: FAIL (struct-returning helper=%b, int-returning \
           helper=%b, struct defined=%b)\n\
           %s\n\
           %!"
          name
          bk
          struct_ret
          mistyped
          struct_def
          src
      end
      else Printf.printf "  codegen[%s/%s]: OK\n%!" name bk)
    backends

(* ---- Axis 2: behavioural equivalence on every device --------------------- *)

let must_pass fw =
  match fw with
  | "CUDA" | "OpenCL" | "Vulkan" | "Metal" | "Native" | "Interpreter" -> true
  | _ -> false

let any_failure = ref false

let pass_count = ref 0

let run_kernel_on name kirc reff dev n =
  let ir = ir_of name kirc in
  let src = Vector.create Vector.float32 n in
  let dst = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set src i (ref_src i) ;
    Vector.set dst i (-999.0)
  done ;
  let threads = min 64 n in
  let grid_x = (n + threads - 1) / threads in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~block:(Execute.dims1d threads)
    ~grid:(Execute.dims1d grid_x)
    ~args:[Execute.Vec src; Execute.Vec dst; Execute.Int32 (Int32.of_int n)]
    () ;
  Transfer.flush dev ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let got = Vector.get dst i and exp = reff i in
    if abs_float (got -. exp) > 1e-3 then begin
      ok := false ;
      if i < 5 then
        Printf.printf
          "\n    %s mismatch at %d: got %.3f expected %.3f%!"
          name
          i
          got
          exp
    end
  done ;
  !ok

let kernels = [("ff", k_ff, ref_ff); ("fi", k_fi, ref_fi)]

let () =
  print_endline "=== helper-tuple-return E2E ===" ;
  (* Axis 1 — codegen structural check (no device needed). *)
  print_endline "-- emitted-source structural check --" ;
  check_codegen "ff" k_ff ~struct_name:"_tup_float32_float32" ~helper:"mk" ;
  check_codegen "fi" k_fi ~struct_name:"_tup_float32_int32" ~helper:"mkm" ;
  if not !codegen_ok then any_failure := true ;

  (* Axis 2 — behaviour on every available device. *)
  Benchmarks.init () ;
  let devs =
    Device.init
      ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"]
      ()
  in
  if Array.length devs = 0 then
    print_endline "No runtime devices — codegen axis only"
  else begin
    let n = 64 in
    Array.iter
      (fun dev ->
        let fw = dev.Device.framework in
        List.iter
          (fun (name, kirc, reff) ->
            Printf.printf "runtime [%s/%s]: %!" fw name ;
            try
              if run_kernel_on name kirc reff dev n then begin
                incr pass_count ;
                print_endline "PASSED"
              end
              else if must_pass fw then begin
                any_failure := true ;
                print_endline "FAILED"
              end
              else print_endline "skip (non-required)"
            with e ->
              let msg =
                match e with
                | Sarek_backend_error.Backend_error.Backend_error err ->
                    Sarek_backend_error.Backend_error.to_string err
                | e -> Printexc.to_string e
              in
              if must_pass fw then begin
                any_failure := true ;
                Printf.printf "FAIL (%s)\n%!" msg
              end
              else Printf.printf "skip (%s)\n%!" msg)
          kernels)
      devs
  end ;
  Printf.printf
    "\n=== helper-tuple-return: %d runtime checks passed ===\n"
    !pass_count ;
  if !any_failure then exit 1
