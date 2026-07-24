(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for kernel-LOCAL tuple bindings.
 *
 * A local tuple value ([let p = (a, b) in ...], a tuple through an [if]-branch,
 * a [let]/[match] destructure) is lowered to the L13 synthesized positional
 * record ([_tup_*], fields [_0.._n]) — the SAME machinery vector-of-tuple
 * elements already use. Before this fix a local tuple slot was typed by the
 * [elttype_of_typ] placeholder ([int]), so struct backends emitted [int p]
 * where a struct belonged: wrong code / codegen rejection on OpenCL/Vulkan,
 * sometimes tolerated on CUDA. This test locks BOTH:
 *
 *   Axis 1 (codegen): the emitted CUDA-C / OpenCL-C / GLSL for each position
 *     declares the local slot with the synthesized struct type and defines the
 *     struct — never a bare scalar mistyping.
 *   Axis 2 (behaviour): results match a pure-OCaml reference on every available
 *     device — CUDA/PTX (under ZLUDA), OpenCL, Vulkan, Metal, Native AND the
 *     Interpreter (its evaluator carries local tuples by value as positional
 *     records, so no byte-layout host bridge is needed — the tuple never leaves
 *     the kernel).
 *
 * Positions exercised: let-bound literal + match destructure (all-float32);
 * let-pattern destructure (all-float32); tuple through an if-branch
 * (all-float32); mixed (float32 * int32) with an int component consumed via
 * [float_of_int]. All results land in a plain [float32 vector], so verification
 * is a scalar float compare — the tuple is purely kernel-internal.
 ******************************************************************************)

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Benchmarks = Test_helpers.Benchmarks

type float32 = float

(* A: let-bound tuple literal, consumed by a match destructure.
   dst[i] = a + b  with (a, b) = (src[i], src[i] + 1) = 2*src[i] + 1 *)
let k_match =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let open Std in
        let tid = global_thread_id in
        if tid < n then begin
          let p = (src.(tid), src.(tid) +. 1.0) in
          match p with a, b -> dst.(tid) <- a +. b
        end]

(* B: let-PATTERN destructure (desugars to a single-arm tuple match).
   dst[i] = a + b  with (a, b) = (src[i], src[i] * 2) = 3*src[i] *)
let k_letpat =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let open Std in
        let tid = global_thread_id in
        if tid < n then begin
          let a, b = (src.(tid), src.(tid) *. 2.0) in
          dst.(tid) <- a +. b
        end]

(* C: tuple produced by an if-branch, then destructured.
   dst[i] = a + b  with (a, b) = (src[i], 1) if src[i] > 0 else (src[i], 2) *)
let k_ifbranch =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let open Std in
        let tid = global_thread_id in
        if tid < n then begin
          let p =
            if src.(tid) >. 0.0 then (src.(tid), 1.0) else (src.(tid), 2.0)
          in
          match p with a, b -> dst.(tid) <- a +. b
        end]

(* D: mixed (float32 * int32) local tuple; the int component is consumed via
   float_of_int, exercising a mixed-alignment layout.
   dst[i] = a + float_of_int b  with (a, b) = (src[i], tid) *)
let k_mixed =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let open Std in
        let tid = global_thread_id in
        if tid < n then begin
          let p = (src.(tid), tid) in
          match p with a, b -> dst.(tid) <- a +. float b
        end]

(* ---- OCaml references (must mirror the kernels exactly) ------------------ *)

let ref_src i = float_of_int i -. 8.0 (* spans negative and positive for C *)

let ref_match i = ref_src i +. (ref_src i +. 1.0)

let ref_letpat i = ref_src i +. (ref_src i *. 2.0)

let ref_ifbranch i = ref_src i +. if ref_src i > 0.0 then 1.0 else 2.0

let ref_mixed i = ref_src i +. float_of_int i

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

(* The synthesized-record type name for the tuple shape of each kernel. *)
let check_codegen name kirc ~struct_name =
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
      (* The synthesized struct must be defined AND used as the slot type; the
         destructure tmp must never be mistyped as a bare scalar. *)
      let has_struct = contains src struct_name in
      let mistyped =
        contains src ("int " ^ struct_name)
        || contains src "int __sarek_tup"
        || contains src "int p ="
      in
      if (not has_struct) || mistyped then begin
        codegen_ok := false ;
        Printf.printf
          "  codegen[%s/%s]: FAIL (struct present=%b, mistyped=%b)\n%s\n%!"
          name
          bk
          has_struct
          mistyped
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

let kernels =
  [
    ("match", k_match, ref_match);
    ("letpat", k_letpat, ref_letpat);
    ("ifbranch", k_ifbranch, ref_ifbranch);
    ("mixed", k_mixed, ref_mixed);
  ]

let () =
  print_endline "=== local-tuple E2E ===" ;
  (* Axis 1 — codegen structural check (no device needed). *)
  print_endline "-- emitted-source structural check --" ;
  check_codegen "match" k_match ~struct_name:"_tup_float32_float32" ;
  check_codegen "letpat" k_letpat ~struct_name:"_tup_float32_float32" ;
  check_codegen "ifbranch" k_ifbranch ~struct_name:"_tup_float32_float32" ;
  check_codegen "mixed" k_mixed ~struct_name:"_tup_float32_int32" ;
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
  Printf.printf "\n=== local-tuple: %d runtime checks passed ===\n" !pass_count ;
  if !any_failure then exit 1
