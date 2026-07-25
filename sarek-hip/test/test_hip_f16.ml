(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #57 slice 1 — f16 end-to-end gate.
 *
 * Proves the f16 element type works as a WHOLE, not just that it compiles:
 *
 *   1. HOST      — an f16 vector stores/reads at binary16 precision.
 *   2. INTERP    — the same IR run on the interpreter.
 *   3. HIP       — the same IR JIT-compiled by hiprtc and run on the GPU,
 *                  cross-checked against a CPU reference at binary16 tolerance.
 *   4. AGREEMENT — interpreter and GPU must produce the SAME f16 bits. This is
 *                  the real payload of the "store f16, compute f32" decision:
 *                  it is the only discipline under which they can agree.
 *
 * Skip-clean off ROCm: prints [SKIP] for the HIP part and still runs the host
 * and interpreter parts, which need no device.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

let () = Sarek_native.Native_plugin.init ()

(* The f16 kernel. Note the shape the type system FORCES: f16 is a storage type,
   so the element must be widened to f32 before any arithmetic and narrowed back
   on store. `inp.(tid) * 2.0` does not typecheck — f16 is not numeric. *)
let f16_scale =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) *. 2.0)]

(* Second kernel: the narrowing happens MID-EXPRESSION and the result keeps
   being computed on in f32 before the final store. This is the only shape that
   observes ECast (TFloat16, _) rounding on the interpreter/native paths: in
   f16_scale above, the cast feeds the store directly, so the Bigarray.Float16
   store would mask a missing round. Here `x *. 1.1` is narrowed to binary16,
   widened again, and then 1000.0 is added in f32 — an unrounded intermediate
   survives into the result and diverges from the GPU. *)
let f16_midround =
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

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let scale_ir = ir_of f16_scale

let midround_ir = ir_of f16_midround

let n = 1024

(* Inputs chosen so binary16 rounding is OBSERVABLE, not incidental:
   3.14159 -> 3.14062, 0.1 -> 0.099975..., 65504 is max finite binary16. *)
let inputs =
  Array.init n (fun i ->
      match i with
      | 0 -> 3.14159
      | 1 -> 0.1
      | 2 -> 1.0
      | 3 -> 0.0
      | 4 -> -2.5
      | 5 -> 1e-5 (* subnormal territory for binary16 *)
      (* The fusion witness. On this input the f32 product lands EXACTLY on a
         binary16 rounding tie, which is what let the AMDGPU backend's fused
         multiply-and-narrow ([v_fma_mixlo_f16]) be observed: it returned 1006.5
         where every f32-discipline path returns 1006.0. Kept as a permanent
         lane so the fusion cannot come back unnoticed; the exhaustive sweep
         below is the general statement, this is the cheap canary. *)
      | 6 -> 5.68359375
      | _ -> float_of_int (i mod 97) /. 7.0)

let round16 = Sarek_interp.Sarek_float16.to_float16

(* Round to the nearest binary32 value. Every arithmetic step in the references
   below goes through this, because the kernels evaluate in f32 while OCaml
   evaluates in binary64 — and [compare_exact] then demands BIT equality after
   narrowing. Without it the references were exact only by accident.

   MEASURED, not argued: an exhaustive sweep of all 63488 finite binary16 inputs
   finds 373 (~0.6%) on which the old f64-intermediate spelling and this f32 one
   disagree after the final narrowing. Smallest witness x = 5.68359375:

     f32 product = 6.251953125      EXACTLY the binary16 tie -> RNE -> 6.25
                                    -> +1000 -> 1006.25 (tie) -> RNE -> 1006.0
     f64 product = 6.251953125000001  just above the tie     -> 6.25390625
                                    -> +1000 -> 1006.2539    -> 1006.5

   The current fixed input set contains none of the 373, which is the only
   reason the old spelling passed. A bit-exactness test must be exact by
   construction, so the references now follow the kernel's f32 discipline.

   That witness also exposed a real codegen defect, now FIXED (see
   [Sarek_ir_cuda.sarek_f32_barrier_decl]): the AMDGPU backend was fusing the
   f32 multiply into the narrowing ([v_fma_mixlo_f16]) and demoting the f32 add
   to binary16 ([v_add_f16]), skipping the roundings the DSL promises. On this
   witness HIP returned 1006.5 against 1006.0 everywhere else. Across the whole
   binary16 domain the device disagreed on 620 of 63488 inputs; with the barrier
   in place, 0. x = 5.68359375 is now lane 6 of [inputs] as the cheap canary.

   Deliberately NOT [Sarek_interp.Sarek_float32.to_float32]: that helper flushes
   subnormals to zero and clamps overflow to infinity, which is the
   interpreter's policy rather than the hardware's f32 semantics. Importing it
   here would put a second, different rounding discipline inside the very
   reference that defines what "bit-identical" means. This round-trip is plain
   IEEE-754 round-to-nearest binary32.

   Double rounding is safe here: for +, -, * on binary32 operands, binary64 has
   more than the 2p+2 = 50 bits needed, so f64-then-f32 gives the same result as
   the single correctly-rounded f32 operation. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* CPU reference under the SAME discipline the kernel is defined with: the input
   is already a stored f16 value, the multiply happens in f32, the result is
   narrowed on store. ([*. 2.0] is exact, so the [f32] here changes nothing
   today; it is present so the two references cannot drift apart in spelling.) *)
let reference = Array.map (fun x -> round16 (f32 (round16 x *. f32 2.0))) inputs

(* Reference for f16_midround, under the same discipline: every f16-typed value
   -- including the mid-expression one -- is narrowed, everything else is f32.

   Mirrors the kernel step for step:
     t0 = float32_of_float16 inp.(tid)      (widening, exact)
     t1 = t0 *. 1.1                          (f32 multiply, f32 literal)
     t2 = float16_of_float32 t1              (the MID-EXPRESSION narrowing)
     t3 = float32_of_float16 t2 +. 1000.0    (f32 add)
     out = float16_of_float32 t3 *)
let reference_midround =
  Array.map
    (fun x ->
      let mid = round16 (f32 (round16 x *. f32 1.1)) in
      round16 (f32 (mid +. 1000.0)))
    inputs

let make_input () =
  let v = Vector.create Vector.float16 n in
  Array.iteri (fun i x -> Vector.set v i x) inputs ;
  v

let make_output () =
  let v = Vector.create Vector.float16 n in
  for i = 0 to n - 1 do
    Vector.set v i (-999.0)
  done ;
  v

(* --------------------------------------------------------------- *)
(* 1. Host storage round-trip                                      *)
(* --------------------------------------------------------------- *)

let test_host_roundtrip () =
  let v = Vector.create Vector.float16 8 in
  Vector.set v 0 3.14159 ;
  let got = Vector.get v 0 in
  (* The canonical binary16 witness: 3.14159 is NOT representable, and the
     nearest binary16 value is 3.140625. If this reads back as 3.14159 the
     vector is not really f16-backed. *)
  let ok_pi = abs_float (got -. 3.140625) < 1e-9 in
  Printf.printf
    "    host 3.14159 -> %.6f (expect 3.140625) : %s\n"
    got
    (if ok_pi then "OK" else "FAIL") ;
  (* Exactly-representable values must be exact. *)
  Vector.set v 1 1.0 ;
  Vector.set v 2 0.5 ;
  Vector.set v 3 (-2.5) ;
  let ok_exact =
    Vector.get v 1 = 1.0 && Vector.get v 2 = 0.5 && Vector.get v 3 = -2.5
  in
  Printf.printf
    "    host exact values (1.0/0.5/-2.5)       : %s\n"
    (if ok_exact then "OK" else "FAIL") ;
  (* Overflow to infinity above max finite binary16 (65504). *)
  Vector.set v 4 70000.0 ;
  let ok_ovf = Vector.get v 4 = infinity in
  Printf.printf
    "    host 70000.0 -> inf (binary16 overflow): %s\n"
    (if ok_ovf then "OK" else "FAIL") ;
  (* Element size really is 2 bytes. *)
  let ok_size = Vector.elem_size (Vector.kind v) = 2 in
  Printf.printf
    "    host elem_size = 2                     : %s\n"
    (if ok_size then "OK" else "FAIL") ;
  ok_pi && ok_exact && ok_ovf && ok_size

(* --------------------------------------------------------------- *)
(* Comparison at binary16 tolerance                                *)
(* --------------------------------------------------------------- *)

(* Results are f16 values; both sides went through the same narrowing, so this
   should be EXACT. Compare exactly and report the first mismatch. *)
let compare_exact ?(expected = reference) label got =
  let bad = ref 0 and first = ref None in
  Array.iteri
    (fun i g ->
      let e = expected.(i) in
      let same = g = e || (g <> g && e <> e) in
      if not same then (
        incr bad ;
        if !first = None then first := Some (i, g, e)))
    got ;
  (match !first with
  | Some (i, g, e) ->
      Printf.printf
        "    %s: %d/%d mismatch, first at [%d]: got %.8g expected %.8g\n"
        label
        !bad
        (Array.length got)
        i
        g
        e
  | None -> ()) ;
  !bad = 0

(* --------------------------------------------------------------- *)
(* 2. Interpreter                                                  *)
(* --------------------------------------------------------------- *)

let run_interp () =
  let inp = make_input () and out = make_output () in
  Execute.run_interpreter_vectors
    ~ir:scale_ir
    ~args:[Vec out; Vec inp; Int n]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d ((n + 63) / 64))
    ~parallel:false ;
  compare_exact "interp" (Vector.to_array out)

(* --------------------------------------------------------------- *)
(* 3. Native backend (PPX-generated OCaml, no GPU)                 *)
(* --------------------------------------------------------------- *)

(* Devices are enumerated ONCE, in [main], and both backends pick from that one
   list. Calling Device.init ~frameworks:[...] a second time re-initialises the
   registry restricted to those frameworks, which silently made a later
   unrestricted lookup report "no HIP device". *)
let run_native devices =
  match
    Array.to_list devices
    |> List.filter (fun d -> d.Device.framework = "Native")
  with
  | [] ->
      print_endline
        "    Native                                 : [SKIP] no Native device" ;
      (true, [||])
  | dev :: _ ->
      let inp = make_input () and out = make_output () in
      Execute.run_vectors
        ~device:dev
        ~ir:scale_ir
        ~args:[Vec out; Vec inp; Int n]
        ~block:(Execute.dims1d 64)
        ~grid:(Execute.dims1d ((n + 63) / 64))
        () ;
      Transfer.flush dev ;
      let got = Vector.to_array out in
      (compare_exact "native" got, got)

(* --------------------------------------------------------------- *)
(* 4. HIP on real hardware                                         *)
(* --------------------------------------------------------------- *)

let run_hip dev =
  let inp = make_input () and out = make_output () in
  let block_sz = 256 in
  Execute.run_vectors
    ~device:dev
    ~ir:scale_ir
    ~args:[Vec out; Vec inp; Int n]
    ~block:(Execute.dims1d block_sz)
    ~grid:(Execute.dims1d ((n + block_sz - 1) / block_sz))
    () ;
  Transfer.flush dev ;
  let got = Vector.to_array out in
  (compare_exact "hip" got, got)

(* --------------------------------------------------------------- *)
(* 5. Mid-expression narrowing: the ECast-rounding gate             *)
(* --------------------------------------------------------------- *)

let run_midround devices =
  let one label dev_opt =
    let inp = make_input () and out = make_output () in
    (match dev_opt with
    | Some dev ->
        Execute.run_vectors
          ~device:dev
          ~ir:midround_ir
          ~args:[Vec out; Vec inp; Int n]
          ~block:(Execute.dims1d 64)
          ~grid:(Execute.dims1d ((n + 63) / 64))
          () ;
        Transfer.flush dev
    | None ->
        Execute.run_interpreter_vectors
          ~ir:midround_ir
          ~args:[Vec out; Vec inp; Int n]
          ~block:(Execute.dims1d 64)
          ~grid:(Execute.dims1d ((n + 63) / 64))
          ~parallel:false) ;
    let got = Vector.to_array out in
    let ok = compare_exact ~expected:reference_midround label got in
    Printf.printf "    midround %-29s: %s\n" label (if ok then "OK" else "FAIL") ;
    (ok, got)
  in
  let pick fw =
    Array.to_list devices |> List.filter (fun d -> d.Device.framework = fw)
    |> function
    | [] -> None
    | d :: _ -> Some d
  in
  let interp_ok, interp_got = one "interpreter" None in
  let native_ok, _ =
    match pick "Native" with None -> (true, [||]) | dev -> one "native" dev
  in
  let hip_ok, hip_got =
    match pick "HIP" with None -> (true, [||]) | dev -> one "HIP" dev
  in
  (* The agreement assertion that makes the ECast rounding observable: if the
     interpreter skipped narrowing the mid-expression value, these diverge. *)
  let agree =
    Array.length hip_got = 0
    || Array.length hip_got = Array.length interp_got
       &&
       let d = ref true in
       Array.iteri (fun i g -> if g <> interp_got.(i) then d := false) hip_got ;
       !d
  in
  Printf.printf
    "    midround HIP == interpreter            : %s\n"
    (if agree then "OK" else "FAIL") ;
  interp_ok && native_ok && hip_ok && agree

(* --------------------------------------------------------------- *)
(* EXHAUSTIVE domain sweep                                         *)
(* --------------------------------------------------------------- *)

(* "HIP == interpreter, bit-identical" is a claim about the WHOLE f16 domain, so
   state it over the whole domain instead of over a hand-picked sample. binary16
   has only 63488 finite values, so the exhaustive statement is affordable —
   under a second — and it is what actually restores the guarantee. A sampled
   version of this gate passed for months while the backend was fusing away a
   mandated rounding on 620 of those inputs.

   Sensitivity is not assumed: with the codegen barrier removed this reports
   620/63488 on both gfx1100 and the integrated gfx1036, first at 5.68359375. *)

(* Exact value of a binary16 bit pattern; None for NaN/Inf. *)
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

let sweep_reference =
  Array.map
    (fun x ->
      let mid = round16 (f32 (round16 x *. f32 1.1)) in
      round16 (f32 (mid +. 1000.0)))
    sweep_inputs

let run_sweep dev =
  let m = Array.length sweep_inputs in
  let inp = Vector.create Vector.float16 m in
  Array.iteri (fun i x -> Vector.set inp i x) sweep_inputs ;
  let out = Vector.create Vector.float16 m in
  let block = 256 in
  Execute.run_vectors
    ~device:dev
    ~ir:midround_ir
    ~args:[Vec out; Vec inp; Int m]
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d ((m + block - 1) / block))
    () ;
  Transfer.flush dev ;
  let bad = ref 0 and first = ref None in
  for i = 0 to m - 1 do
    let g = Vector.get out i and e = sweep_reference.(i) in
    if not (g = e || (g <> g && e <> e)) then begin
      incr bad ;
      if !first = None then first := Some (sweep_inputs.(i), g, e)
    end
  done ;
  Printf.printf
    "    exhaustive sweep (%d inputs)          : %s\n"
    m
    (match !first with
    | None -> "OK"
    | Some (x, g, e) ->
        Printf.sprintf
          "FAIL — %d mismatches, first x=%.9g got=%.9g expected=%.9g"
          !bad
          x
          g
          e) ;
  !bad = 0

(* --------------------------------------------------------------- *)

let () =
  Printf.printf "test_hip_f16 (#57 slice 1 f16 end-to-end)\n" ;
  let host_ok = test_host_roundtrip () in
  let interp_ok =
    try run_interp ()
    with e ->
      Printf.printf "    interp EXN: %s\n" (Printexc.to_string e) ;
      false
  in
  Printf.printf
    "    interpreter vs CPU reference           : %s\n"
    (if interp_ok then "OK" else "FAIL") ;
  let devices = Device.init () in
  let native_ok, _native_got =
    try run_native devices
    with e ->
      Printf.printf "    native EXN: %s\n" (Printexc.to_string e) ;
      (false, [||])
  in
  Printf.printf
    "    native vs CPU reference                : %s\n"
    (if native_ok then "OK" else "FAIL") ;
  let hip =
    Array.to_list devices |> List.filter (fun d -> d.Device.framework = "HIP")
  in
  let hip_ok, agree_ok =
    match hip with
    | [] ->
        print_endline
          "    HIP                                    : [SKIP] no HIP device" ;
        (true, true)
    | dev :: _ ->
        Printf.printf "    HIP device: %s\n" dev.Device.name ;
        let ok, got =
          try run_hip dev
          with e ->
            Printf.printf "    hip EXN: %s\n" (Printexc.to_string e) ;
            (false, [||])
        in
        Printf.printf
          "    HIP vs CPU reference                   : %s\n"
          (if ok then "OK" else "FAIL") ;
        (* Interpreter/GPU agreement, the cross-backend consistency claim. *)
        let inp = make_input () and out = make_output () in
        Execute.run_interpreter_vectors
          ~ir:scale_ir
          ~args:[Vec out; Vec inp; Int n]
          ~block:(Execute.dims1d 64)
          ~grid:(Execute.dims1d ((n + 63) / 64))
          ~parallel:false ;
        let iarr = Vector.to_array out in
        let agree =
          Array.length got = Array.length iarr
          &&
          let d = ref true in
          Array.iteri (fun i g -> if g <> iarr.(i) then d := false) got ;
          !d
        in
        Printf.printf
          "    HIP == interpreter (bit-identical f16) : %s\n"
          (if agree then "OK" else "FAIL") ;
        (ok, agree)
  in
  let midround_ok =
    try run_midround devices
    with e ->
      Printf.printf "    midround EXN: %s\n" (Printexc.to_string e) ;
      false
  in
  (* Every HIP device, not just the first: the fusion reproduced on both the
     discrete gfx1100 and the integrated gfx1036, so both must be swept. *)
  let sweep_ok =
    List.fold_left
      (fun acc dev ->
        Printf.printf "    sweeping %s\n" dev.Device.name ;
        let ok =
          try run_sweep dev
          with e ->
            Printf.printf "    sweep EXN: %s\n" (Printexc.to_string e) ;
            false
        in
        acc && ok)
      true
      hip
  in
  if
    host_ok && interp_ok && native_ok && hip_ok && agree_ok && midround_ok
    && sweep_ok
  then print_endline "test_hip_f16 PASSED"
  else (
    print_endline "test_hip_f16 FAILED" ;
    exit 1)
