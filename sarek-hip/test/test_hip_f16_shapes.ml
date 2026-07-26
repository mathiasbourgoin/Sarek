(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Exhaustive f16 expression-shape audit for AMDGPU fusion demotion (issue #106)
 *
 * WHAT IS KNOWN
 *
 * An AMDGPU ISel combine fuses an f32 multiply into the f32->f16 narrowing that
 * consumes it ([v_fma_mixlo_f16]) and demotes a neighbouring f32 add to
 * binary16 ([v_add_f16]). It sits BELOW the C-level FP controls, so
 * [-ffp-contract=off] does not prevent it. [Sarek_ir_cuda.sarek_f32_barrier]
 * -- [asm volatile("" : "+v"(x))] on the narrowing's argument -- does.
 * That was established on ONE expression shape (the [f16_midround] kernel,
 * 620 of 63488 finite binary16 inputs disagreeing with the interpreter). The
 * [v_add_f16] demotion shows the class is broader than multiply-then-narrow.
 *
 * WHAT THIS TEST DOES
 *
 * It enumerates the f16 expression shapes Sarek can actually emit and sweeps
 * each one over the WHOLE finite binary16 domain -- all 63488 values, not a
 * sample. That matters: a sampled f16 test was green for months while the
 * 620-input defect was live.
 *
 * The enumeration is small, and closed, for a reason that is a property of the
 * language rather than of this test. f16 in Sarek is a STORAGE-ONLY element
 * type: [Sarek_typer] rejects f16 operands for every arithmetic, comparison,
 * bitwise and unary operator; there is no [CFloat16] literal; there is no f16
 * math intrinsic; f16 record/struct fields are rejected at layout; an f16
 * scalar kernel parameter is rejected at lowering. The entire f16 surface is
 * two core primitives, [float16_of_float32] and [float32_of_float16]. So the
 * only value producer is [ECast (TFloat16, e)] for an f32-typed [e], and
 * auditing "every f16 expression shape" reduces to auditing the shapes of [e]
 * that can reach a narrowing, plus the two paths with no narrowing at all
 * (a straight f16->f16 copy, and a widening whose result never narrows).
 * That is what the shape table below covers.
 *
 * THREE COLUMNS, AND WHY
 *
 *   shipping     the ordinary Sarek path ([Execute.run_vectors]) -- the answer
 *                to "is this shape correct today?"
 *   src+barrier  the generated HIP source, run verbatim through [run_source]
 *   src-barrier  the same source with the barrier's [asm volatile] body
 *                deleted -- the answer to "is the barrier what is holding this
 *                shape up?"
 *
 * [src+barrier] exists so that [src-barrier] differs from it in exactly one
 * textual substitution and nothing else; comparing [shipping] against
 * [src+barrier] checks that the [run_source] path is faithful. A shape whose
 * [src-barrier] column is non-zero while [src+barrier] is zero is a shape the
 * barrier is load-bearing for -- and that difference is this harness's own
 * red-on-mutation control: if NO shape goes red when the barrier is removed,
 * the harness is not sensitive to demotion at all and its zeros mean nothing.
 * [main] fails on exactly that condition.
 *
 * ORACLE
 *
 * The interpreter. Its f16 narrowing is [Sarek_interp.Sarek_float16.to_float16]
 * (a [Bigarray.Float16] store), and its f32 rounding is an [Int32] bit
 * round-trip; the reference for each shape below composes those two the way the
 * IR says the expression evaluates. Comparison is bit-exact -- there is no
 * tolerance, because the claim is agreement, not closeness.
 *
 * Transcendentals ([sin], [exp], [log], [pow]) are deliberately NOT in the
 * table. Per docs/fp-contraction-policy.md §1 their rounding is the oracle's
 * own and a device is not required to match it bit-for-bit, so a disagreement
 * there would not be evidence of demotion.
 *
 * Run:  dune exec sarek-hip/test/test_hip_f16_shapes.exe
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

(* [Gpu] must be in scope at the top level: the PPX's native fallback names
   core primitives as [Gpu.<prim>]. *)
module Gpu = Sarek_stdlib.Gpu
module Float32 = Sarek_stdlib.Float32

let () = Sarek_hip.Hip_plugin.register ()

(* ------------------------------------------------------------------ *)
(* Oracle arithmetic                                                   *)
(* ------------------------------------------------------------------ *)

(* The interpreter's binary16 narrowing. *)
let round16 = Sarek_interp.Sarek_float16.to_float16

(* Plain IEEE round-to-nearest-even binary32, as [test_hip_f16] uses: NOT
   [Sarek_float32.to_float32], which additionally flushes subnormals and clamps
   overflow (interpreter policy, not hardware behaviour). *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let fma32 a b c = f32 (Float.fma a b c)

(* Constants as the device sees them: the PPX emits float32 literals, so the
   reference must use the binary32 value of each constant, not its binary64
   one. [1.1] is the whole point -- [f32 1.1 <> 1.1]. *)
let c11 = f32 1.1

let c09 = f32 0.9

let c1000 = 1000.0 (* exact in binary32 *)

let c3 = 3.0 (* exact *)

(* ------------------------------------------------------------------ *)
(* The shapes                                                          *)
(* ------------------------------------------------------------------ *)

(* Every kernel has the same signature -- (out : float16 vector) (inp :
   float16 vector) (n : int32) -- so one runner serves them all. [x] below is
   always [float32_of_float16 inp.(tid)], the only way to get an f32 out of an
   f16. *)

let k_a1_roundtrip =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid))]

let k_a2_mul =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1)]

let k_a3_add =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) +. 1000.0)]

let k_a4_sub =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) -. 1000.0)]

let k_a5_div =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) /. 3.0)]

let k_a6_mul_add =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32 ((float32_of_float16 inp.(tid) *. 1.1) +. 1000.0)]

let k_a7_add_mul =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32 ((float32_of_float16 inp.(tid) +. 1000.0) *. 1.1)]

let k_a8_fma =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (Float32.fma (float32_of_float16 inp.(tid)) 1.1 1000.0)]

let k_a9_sqrt =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        let x = float32_of_float16 inp.(tid) in
        out.(tid) <- float16_of_float32 (Float32.sqrt (x *. x))]

let k_a10_neg =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (0.0 -. float32_of_float16 inp.(tid))]

let k_a11_floor =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (Float32.floor (float32_of_float16 inp.(tid) *. 1.1))]

let k_a12_cond =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        let x = float32_of_float16 inp.(tid) in
        out.(tid) <- float16_of_float32 (if x > 0.0 then x *. 1.1 else x *. 0.9)]

let k_a13_square =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        let x = float32_of_float16 inp.(tid) in
        out.(tid) <- float16_of_float32 (x *. x)]

let k_a14_mul_mul =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1 *. 1.1)]

let k_a15_mul_div_add =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        let x = float32_of_float16 inp.(tid) in
        out.(tid) <- float16_of_float32 ((x *. 1.1) +. (x /. 3.0))]

(* B family: the narrowing is MID-expression, so an unrounded intermediate can
   survive into the result instead of being masked by the f16 store. B1 is the
   shape that produced the original 620. *)

let k_b1_midround =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (float32_of_float16
               (float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1))
            +. 1000.0)]

let k_b2_midround_mul =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (float32_of_float16
               (float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1))
            *. 1.1)]

let k_b3_midround_add_mul =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (float32_of_float16
               (float16_of_float32 (float32_of_float16 inp.(tid) +. 1000.0))
            *. 1.1)]

let k_b4_double_mid =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (float32_of_float16
               (float16_of_float32
                  (float32_of_float16
                     (float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1))
                  +. 1000.0))
            *. 1.1)]

(* C family: no [ECast (TFloat16, _)] at all -- the two f16 paths where no
   barrier is emitted, and none should be needed. *)

let k_c1_copy =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then out.(tid) <- inp.(tid)]

(* ------------------------------------------------------------------ *)
(* Shape table: kernel + its reference semantics                       *)
(* ------------------------------------------------------------------ *)

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with Some ir -> ir | None -> failwith "no IR"

type shape = {
  id : string;
  descr : string;
  ir : Sarek_ir_types.kernel;
  (* [x] is the exact binary16 input value, already representable. *)
  reference : float -> float;
}

let shapes =
  [
    {
      id = "A1";
      descr = "narrow(widen x)              round trip";
      ir = ir_of k_a1_roundtrip;
      reference = (fun x -> round16 x);
    };
    {
      id = "A2";
      descr = "narrow(x *. 1.1)             mul -> narrow";
      ir = ir_of k_a2_mul;
      reference = (fun x -> round16 (f32 (x *. c11)));
    };
    {
      id = "A3";
      descr = "narrow(x +. 1000.)           add -> narrow";
      ir = ir_of k_a3_add;
      reference = (fun x -> round16 (f32 (x +. c1000)));
    };
    {
      id = "A4";
      descr = "narrow(x -. 1000.)           sub -> narrow";
      ir = ir_of k_a4_sub;
      reference = (fun x -> round16 (f32 (x -. c1000)));
    };
    {
      id = "A5";
      descr = "narrow(x /. 3.)              div -> narrow";
      ir = ir_of k_a5_div;
      reference = (fun x -> round16 (f32 (x /. c3)));
    };
    {
      id = "A6";
      descr = "narrow(x *. 1.1 +. 1000.)    mul,add -> narrow";
      ir = ir_of k_a6_mul_add;
      reference = (fun x -> round16 (f32 (f32 (x *. c11) +. c1000)));
    };
    {
      id = "A7";
      descr = "narrow((x +. 1000.) *. 1.1)  add,mul -> narrow";
      ir = ir_of k_a7_add_mul;
      reference = (fun x -> round16 (f32 (f32 (x +. c1000) *. c11)));
    };
    {
      id = "A8";
      descr = "narrow(fma x 1.1 1000.)      fma -> narrow";
      ir = ir_of k_a8_fma;
      reference = (fun x -> round16 (fma32 x c11 c1000));
    };
    {
      id = "A9";
      descr = "narrow(sqrt (x *. x))         mul,sqrt -> narrow";
      ir = ir_of k_a9_sqrt;
      reference = (fun x -> round16 (f32 (sqrt (f32 (x *. x)))));
    };
    {
      id = "A10";
      descr = "narrow(0. -. x)               negation -> narrow";
      ir = ir_of k_a10_neg;
      reference = (fun x -> round16 (f32 (0.0 -. x)));
    };
    {
      id = "A11";
      descr = "narrow(floor (x *. 1.1))     mul,floor -> narrow";
      ir = ir_of k_a11_floor;
      reference = (fun x -> round16 (f32 (Float.floor (f32 (x *. c11)))));
    };
    {
      id = "A12";
      descr = "narrow(if x>0 then .. else)  conditional -> narrow";
      ir = ir_of k_a12_cond;
      reference =
        (fun x -> round16 (f32 (if x > 0.0 then x *. c11 else x *. c09)));
    };
    {
      id = "A13";
      descr = "narrow(x *. x)               square -> narrow";
      ir = ir_of k_a13_square;
      reference = (fun x -> round16 (f32 (x *. x)));
    };
    {
      id = "A14";
      descr = "narrow(x *. 1.1 *. 1.1)      chained mul -> narrow";
      ir = ir_of k_a14_mul_mul;
      reference = (fun x -> round16 (f32 (f32 (x *. c11) *. c11)));
    };
    {
      id = "A15";
      descr = "narrow(x*.1.1 +. x/.3.)      two products -> narrow";
      ir = ir_of k_a15_mul_div_add;
      reference = (fun x -> round16 (f32 (f32 (x *. c11) +. f32 (x /. c3))));
    };
    {
      id = "B1";
      descr = "mid-narrow then +. 1000.     [the original 620]";
      ir = ir_of k_b1_midround;
      reference = (fun x -> round16 (f32 (round16 (f32 (x *. c11)) +. c1000)));
    };
    {
      id = "B2";
      descr = "mid-narrow then *. 1.1";
      ir = ir_of k_b2_midround_mul;
      reference = (fun x -> round16 (f32 (round16 (f32 (x *. c11)) *. c11)));
    };
    {
      id = "B3";
      descr = "mid-narrow of add then *. 1.1";
      ir = ir_of k_b3_midround_add_mul;
      reference = (fun x -> round16 (f32 (round16 (f32 (x +. c1000)) *. c11)));
    };
    {
      id = "B4";
      descr = "two mid-narrows then *. 1.1";
      ir = ir_of k_b4_double_mid;
      reference =
        (fun x ->
          let m1 = round16 (f32 (x *. c11)) in
          let m2 = round16 (f32 (m1 +. c1000)) in
          round16 (f32 (m2 *. c11)));
    };
    {
      id = "C1";
      descr = "out.(i) <- inp.(i)           f16->f16 copy, NO cast";
      ir = ir_of k_c1_copy;
      reference = (fun x -> x);
    };
  ]

(* ------------------------------------------------------------------ *)
(* The exhaustive binary16 domain                                      *)
(* ------------------------------------------------------------------ *)

(* Exact value of a binary16 bit pattern; None for NaN/Inf. Both zeros and the
   whole subnormal range are included. *)
let f16_value_of_bits b =
  let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
  let exp = (b lsr 10) land 0x1f and man = b land 0x3ff in
  if exp = 31 then None
  else if exp = 0 then Some (sign *. float_of_int man *. ldexp 1.0 (-24))
  else Some (sign *. float_of_int (1024 + man) *. ldexp 1.0 (exp - 25))

let sweep_inputs =
  List.init 0x10000 (fun i -> i)
  |> List.filter_map f16_value_of_bits
  |> Array.of_list

let m = Array.length sweep_inputs

(* ------------------------------------------------------------------ *)
(* Barrier neutralisation                                              *)
(* ------------------------------------------------------------------ *)

(* The HIP arm of [Sarek_ir_cuda.sarek_f32_barrier_decl]. Deleting exactly this
   statement turns the barrier into an identity function while leaving every
   call site, and the rest of the generated source, byte-identical. *)
let barrier_stmt = {|asm volatile("" : "+v"(x));|}

let replace_once ~needle ~repl hay =
  let nl = String.length needle and hl = String.length hay in
  let rec go i =
    if i + nl > hl then None
    else if String.sub hay i nl = needle then Some i
    else go (i + 1)
  in
  match go 0 with
  | None -> None
  | Some i ->
      Some (String.sub hay 0 i ^ repl ^ String.sub hay (i + nl) (hl - i - nl))

let count_sub needle hay =
  let nl = String.length needle and hl = String.length hay in
  let rec go i acc =
    if i + nl > hl then acc
    else if String.sub hay i nl = needle then go (i + 1) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

(* ------------------------------------------------------------------ *)
(* Runners                                                             *)
(* ------------------------------------------------------------------ *)

let block = 256

let grid = (m + block - 1) / block

let fresh_io () =
  let inp = Vector.create Vector.float16 m in
  Array.iteri (fun i x -> Vector.set inp i x) sweep_inputs ;
  let out = Vector.create Vector.float16 m in
  for i = 0 to m - 1 do
    Vector.set out i (-999.0)
  done ;
  (inp, out)

let run_shipping dev sh =
  let inp, out = fresh_io () in
  Execute.run_vectors
    ~device:dev
    ~ir:sh.ir
    ~args:[Execute.Vec out; Execute.Vec inp; Execute.Int m]
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d grid)
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let run_from_source dev sh ~source =
  let inp, out = fresh_io () in
  Execute.run_source
    ~device:dev
    ~source
    ~lang:Execute.CUDA_Source
    ~kernel_name:sh.ir.Sarek_ir_types.kern_name
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d grid)
    [Execute.Vec out; Execute.Vec inp; Execute.Int32 (Int32.of_int m)] ;
  Transfer.flush dev ;
  Vector.to_array out

(* Bit equality: OCaml [=] conflates -0.0 and 0.0, and both occur in the sweep
   domain. *)
let same_bits g e = Int64.bits_of_float g = Int64.bits_of_float e

type result = {mismatches : int; first : (float * float * float) option}

let compare_against_reference sh got =
  let bad = ref 0 and first = ref None in
  for i = 0 to m - 1 do
    let g = got.(i) and e = sh.reference sweep_inputs.(i) in
    if not (same_bits g e || (g <> g && e <> e)) then begin
      incr bad ;
      if !first = None then first := Some (sweep_inputs.(i), g, e)
    end
  done ;
  {mismatches = !bad; first = !first}

(* ------------------------------------------------------------------ *)
(* Main                                                                *)
(* ------------------------------------------------------------------ *)

let audit_device (dev : Device.t) =
  Printf.printf
    "\n=== %s (framework %s) ===\n"
    dev.Device.name
    dev.Device.framework ;
  Printf.printf "exhaustive sweep over all %d finite binary16 inputs\n\n" m ;
  Printf.printf
    "  %-4s %-45s %10s %12s %12s\n"
    "id"
    "shape"
    "shipping"
    "src+barrier"
    "src-barrier" ;
  Printf.printf "  %s\n" (String.make 87 '-') ;
  let any_barrier_sensitive = ref false in
  let shipping_failures = ref [] in
  let details = ref [] in
  List.iter
    (fun sh ->
      let source =
        Sarek_codegen.Sarek_ir_cuda.generate_with_types
          ~types:sh.ir.Sarek_ir_types.kern_types
          sh.ir
      in
      let has_barrier = count_sub barrier_stmt source in
      (match Sys.getenv_opt "SAREK_F16_DUMP" with
      | None -> ()
      | Some dir -> (
          let w suffix txt =
            let oc = open_out (Filename.concat dir (sh.id ^ suffix ^ ".hip")) in
            output_string oc txt ;
            close_out oc
          in
          w "_barrier" source ;
          match replace_once ~needle:barrier_stmt ~repl:"" source with
          | Some patched -> w "_nobarrier" patched
          | None -> ())) ;
      let r_ship = compare_against_reference sh (run_shipping dev sh) in
      let r_src =
        compare_against_reference sh (run_from_source dev sh ~source)
      in
      let r_nobar, nobar_label =
        if has_barrier = 0 then
          (* No barrier in this shape's generated code at all: there is nothing
             to neutralise, and that is itself a reportable fact. *)
          ({mismatches = 0; first = None}, "n/a")
        else
          match replace_once ~needle:barrier_stmt ~repl:"" source with
          | None -> assert false
          | Some patched ->
              (* Guard against a vacuous control. *)
              assert (count_sub barrier_stmt patched = has_barrier - 1) ;
              assert (String.length patched < String.length source) ;
              ( compare_against_reference
                  sh
                  (run_from_source dev sh ~source:patched),
                "" )
      in
      let show r lbl =
        if lbl <> "" then lbl
        else if r.mismatches = 0 then "0"
        else string_of_int r.mismatches
      in
      Printf.printf
        "  %-4s %-45s %10s %12s %12s\n"
        sh.id
        sh.descr
        (show r_ship "")
        (show r_src "")
        (show r_nobar nobar_label) ;
      if r_nobar.mismatches > 0 then any_barrier_sensitive := true ;
      if r_ship.mismatches > 0 || r_src.mismatches > 0 then
        shipping_failures := sh.id :: !shipping_failures ;
      details := (sh, has_barrier, r_ship, r_src, r_nobar) :: !details)
    shapes ;
  Printf.printf "\n  witnesses (first disagreeing input per non-zero cell):\n" ;
  List.iter
    (fun (sh, has_barrier, r_ship, r_src, r_nobar) ->
      let w lbl r =
        match r.first with
        | None -> ()
        | Some (x, g, e) ->
            Printf.printf
              "    %-4s %-12s x=%.9g  device=%.9g  interpreter=%.9g  (%d/%d)\n"
              sh.id
              lbl
              x
              g
              e
              r.mismatches
              m
      in
      w "shipping" r_ship ;
      w "src+barrier" r_src ;
      w "src-barrier" r_nobar ;
      if has_barrier = 0 then
        Printf.printf
          "    %-4s no sarek_f32_barrier in generated code (no f16 narrowing \
           to protect)\n"
          sh.id)
    (List.rev !details) ;
  (!shipping_failures, !any_barrier_sensitive)

let () =
  let devs = Device.init () in
  let hip =
    Array.to_list devs |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  if hip = [] then begin
    print_endline "[SKIP] no HIP device available" ;
    exit 0
  end ;
  print_endline
    "=== f16 expression-shape audit for AMDGPU fusion demotion (issue #106) ===" ;
  let results = List.map audit_device hip in
  let failures = List.concat_map fst results in
  let sensitive = List.exists snd results in
  print_endline "" ;
  if not sensitive then begin
    (* Red-on-mutation control. Removing the barrier MUST break at least one
       shape; if it does not, this harness cannot see demotion and every zero
       above is uninformative. *)
    print_endline
      "[FAIL] control broken: neutralising sarek_f32_barrier changed nothing \
       on any shape." ;
    print_endline
      "       This harness has not been shown to detect demotion, so its zeros \
       mean nothing." ;
    exit 1
  end ;
  if failures <> [] then begin
    Printf.printf
      "[FAIL] %d shape(s) disagree with the interpreter as shipped: %s\n"
      (List.length failures)
      (String.concat ", " (List.sort_uniq compare failures)) ;
    exit 1
  end ;
  print_endline
    "[PASS] every shape agrees with the interpreter as shipped, and the \
     barrier is demonstrably load-bearing (removing it breaks at least one \
     shape)."
