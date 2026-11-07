(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test for the stdlib bit reinterpretations Int32.float_of_bits and
 * Int32.bits_of_float, on every enumerated device.
 *
 * Four kernels:
 *   - decode:    int32 vector -> float32 vector     (float_of_bits)
 *   - encode:    float32 vector -> int32 vector     (bits_of_float)
 *   - roundtrip: int32 -> float32 -> int32 inside one kernel
 *   - record:    an f32 scale stored as int32 inside an int32 record vector,
 *                decoded and used in arithmetic (the downstream use case)
 *
 * The bit patterns cover +-0, +-1, +-inf, the smallest and largest denormals,
 * the largest finite value, a quiet NaN with a payload, a signalling NaN, and
 * a sweep of ordinary values.
 *
 * Expected results are OCaml's own Int32.float_of_bits / bits_of_float, with
 * one documented exception. A GPU moves the 32 bits through a register, so the
 * in-kernel round trip is the identity for EVERY pattern. Native and the
 * Interpreter carry a float32 as an OCaml double: the widening quiets a
 * signalling NaN (bit 22 set), exactly as OCaml's
 * [Int32.bits_of_float (Int32.float_of_bits x)] does. The same widening
 * happens when the HOST reads a float32 vector, so the decode/encode kernels
 * are compared through the host's view, and NaN rows there are checked for
 * NaN-ness only.
 *
 * Run with the CUDA/PTX device surfaced (ZLUDA):
 *   LD_LIBRARY_PATH=$HOME/opt/zluda dune exec sarek/tests/e2e/test_int32_bitcasts.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let decode_kernel =
  [%kernel
    fun (a : int32 vector) (b : float32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then b.(tid) <- Int32.float_of_bits a.(tid)]

let encode_kernel =
  [%kernel
    fun (b : float32 vector) (c : int32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then c.(tid) <- Int32.bits_of_float b.(tid)]

let roundtrip_kernel =
  [%kernel
    fun (a : int32 vector) (c : int32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then
        c.(tid) <- Int32.bits_of_float (Int32.float_of_bits a.(tid))]

(* A literal operand: Int32.min_int is the -0.0 pattern, and a C-family or
   WGSL backend that printed it as [-2147483648] handed the reinterpretation a
   64-bit (C) or out-of-range (WGSL) operand that does not compile. *)
let literal_kernel =
  [%kernel
    fun (c : int32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then
        c.(tid) <- Int32.bits_of_float (Int32.float_of_bits 0x80000000l)]

(* rec_.(2i) holds an f32 scale as its bit pattern, rec_.(2i+1) a small integer
   quantum; out.(i) = scale *. float quantum. *)
let record_kernel =
  [%kernel
    fun (rec_ : int32 vector) (out : float32 vector) (n : int32) ->
      let open Std in
      let tid = global_thread_id in
      if tid < n then begin
        let scale = Int32.float_of_bits rec_.(2 * tid) in
        let q = rec_.((2 * tid) + 1) in
        out.(tid) <- scale *. float q
      end]

let special =
  [|
    0x00000000l (* +0 *);
    0x80000000l (* -0 *);
    0x3f800000l (* 1.0 *);
    0xbf800000l (* -1.0 *);
    0x7f800000l (* +inf *);
    0xff800000l (* -inf *);
    0x00000001l (* smallest denormal *);
    0x807fffffl (* largest negative denormal *);
    0x007fffffl (* largest denormal *);
    0x7f7fffffl (* largest finite *);
    0x7fc12345l (* quiet NaN, payload *);
    0x7f800001l (* signalling NaN *);
  |]

let snan = 0x7f800001l

let patterns =
  Array.append
    special
    (Array.init 52 (fun i ->
         Int32.bits_of_float ((float_of_int i *. 0.375) -. 7.0)))

let n = Array.length patterns

let is_nan_bits x = Float.is_nan (Int32.float_of_bits x)

let is_host_float (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

let launch dev ir args =
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d ((n + 63) / 64))
    () ;
  Transfer.flush dev

(* Returns the number of wrong rows, printing the first few. *)
let compare_rows label ok show =
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if not (ok i) then begin
      if !bad < 4 then Printf.printf "    %s row %d: %s\n%!" label i (show i) ;
      incr bad
    end
  done ;
  !bad

let run_decode dev ir =
  let a = Vector.create Vector.int32 n in
  let b = Vector.create Vector.float32 n in
  Array.iteri
    (fun i x ->
      Vector.set a i x ;
      Vector.set b i 42.0)
    patterns ;
  launch dev ir [Execute.Vec a; Execute.Vec b; Execute.Int n] ;
  compare_rows
    "decode"
    (fun i ->
      let x = patterns.(i) and got = Vector.get b i in
      if is_nan_bits x then Float.is_nan got else Int32.bits_of_float got = x)
    (fun i ->
      Printf.sprintf
        "bits 0x%08lx -> %h (bits 0x%08lx)"
        patterns.(i)
        (Vector.get b i)
        (Int32.bits_of_float (Vector.get b i)))

let run_encode dev ir =
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.int32 n in
  Array.iteri
    (fun i x ->
      Vector.set b i (Int32.float_of_bits x) ;
      Vector.set c i 0x5a5a5a5al)
    patterns ;
  launch dev ir [Execute.Vec b; Execute.Vec c; Execute.Int n] ;
  (* What the host stored is OCaml's narrowing of the widened value. *)
  let expected i = Int32.bits_of_float (Int32.float_of_bits patterns.(i)) in
  compare_rows
    "encode"
    (fun i ->
      let got = Vector.get c i in
      if is_nan_bits patterns.(i) then is_nan_bits got else got = expected i)
    (fun i ->
      Printf.sprintf
        "got 0x%08lx, expected 0x%08lx"
        (Vector.get c i)
        (expected i))

let run_roundtrip dev ir =
  let a = Vector.create Vector.int32 n in
  let c = Vector.create Vector.int32 n in
  Array.iteri
    (fun i x ->
      Vector.set a i x ;
      Vector.set c i 0x5a5a5a5al)
    patterns ;
  launch dev ir [Execute.Vec a; Execute.Vec c; Execute.Int n] ;
  let expected i =
    let x = patterns.(i) in
    (* Host-float backends follow OCaml exactly, which quiets the sNaN. *)
    if is_host_float dev && x = snan then
      Int32.bits_of_float (Int32.float_of_bits x)
    else x
  in
  compare_rows
    "roundtrip"
    (fun i -> Vector.get c i = expected i)
    (fun i ->
      Printf.sprintf
        "got 0x%08lx, expected 0x%08lx"
        (Vector.get c i)
        (expected i))

(* Through memory: decode into a float32 vector in one launch, encode it back
   in a second. The float32 bits are stored to and reloaded from device memory,
   so the compiler cannot fold the pair away as it may in [roundtrip]. The
   intermediate vector is never read with [Vector.get], whose widening to a
   double would quiet a signalling NaN on the host. *)
let run_memory dec enc dev _ =
  let a = Vector.create Vector.int32 n in
  let b = Vector.create Vector.float32 n in
  let c = Vector.create Vector.int32 n in
  Array.iteri
    (fun i x ->
      Vector.set a i x ;
      Vector.set c i 0x5a5a5a5al)
    patterns ;
  launch dev dec [Execute.Vec a; Execute.Vec b; Execute.Int n] ;
  launch dev enc [Execute.Vec b; Execute.Vec c; Execute.Int n] ;
  let expected i =
    let x = patterns.(i) in
    if is_host_float dev && x = snan then
      Int32.bits_of_float (Int32.float_of_bits x)
    else x
  in
  compare_rows
    "memory"
    (fun i -> Vector.get c i = expected i)
    (fun i ->
      Printf.sprintf
        "got 0x%08lx, expected 0x%08lx"
        (Vector.get c i)
        (expected i))

let run_literal dev ir =
  let c = Vector.create Vector.int32 n in
  for i = 0 to n - 1 do
    Vector.set c i 0x5a5a5a5al
  done ;
  launch dev ir [Execute.Vec c; Execute.Int n] ;
  compare_rows
    "literal"
    (fun i -> Vector.get c i = 0x80000000l)
    (fun i -> Printf.sprintf "got 0x%08lx" (Vector.get c i))

(* The device source each GPU framework would compile, generated up front: a
   codegen exception is a regression of this feature and must FAIL, whereas a
   device that cannot launch at all is an environment matter and is reported. *)
let generator (dev : Device.t) =
  let open Sarek_codegen in
  match dev.Device.framework with
  | "CUDA" | "CUDA/C" | "HIP" -> Some Sarek_ir_cuda.generate
  | "OpenCL" -> Some Sarek_ir_opencl.generate
  | "Metal" -> Some Sarek_ir_metal.generate
  | "Vulkan" -> Some (fun k -> Sarek_ir_glsl.generate k)
  | "WebGPU" -> Some (fun k -> Sarek_ir_wgsl.generate k)
  (* The plugin also passes [~soa_params]; these kernels take no SoA
     vectors, so the default is what it compiles. *)
  | "CUDA/PTX" -> Some (fun k -> Sarek_ir_ptx.generate k)
  | _ -> None

(* Classification of device failures is shared: [Test_helpers.Backend_failure]
   (codegen, compile, load and launch errors FAIL; an unavailable backend
   skips). Self-check before any device. *)
let () =
  Test_helpers.Backend_failure.self_check ~test_name:"test_int32_bitcasts"

let scales =
  Array.init n (fun i -> Int32.bits_of_float (0.5 +. (float_of_int i *. 0.25)))

let quanta = Array.init n (fun i -> Int32.of_int ((i mod 7) - 3))

let run_record dev ir =
  let r = Vector.create Vector.int32 (2 * n) in
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set r (2 * i) scales.(i) ;
    Vector.set r ((2 * i) + 1) quanta.(i) ;
    Vector.set out i 42.0
  done ;
  launch dev ir [Execute.Vec r; Execute.Vec out; Execute.Int n] ;
  (* Every product is exact in float32 (a multiple of 0.25 times a small
     integer), so the comparison is exact on every backend. *)
  let expected i =
    Int32.float_of_bits scales.(i) *. Int32.to_float quanta.(i)
  in
  compare_rows
    "record"
    (fun i -> Vector.get out i = expected i)
    (fun i ->
      Printf.sprintf "got %h, expected %h" (Vector.get out i) (expected i))

let ir_of kern name =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith (name ^ " kernel has no IR")

let () =
  let cases =
    [
      ("decode", run_decode, ir_of decode_kernel "decode");
      ("encode", run_encode, ir_of encode_kernel "encode");
      ("roundtrip", run_roundtrip, ir_of roundtrip_kernel "roundtrip");
      ("record", run_record, ir_of record_kernel "record");
      ( "memory",
        run_memory (ir_of decode_kernel "decode") (ir_of encode_kernel "encode"),
        ir_of decode_kernel "decode" );
      ("literal", run_literal, ir_of literal_kernel "literal");
    ]
  in
  let devs = Device.init () in
  print_endline "=== Int32.float_of_bits / bits_of_float, E2E ===" ;
  let host_ran = ref false in
  let failed = ref false in
  Array.iter
    (fun (dev : Device.t) ->
      let host = is_host_float dev in
      (match generator dev with
      | None -> ()
      | Some gen ->
          List.iter
            (fun (label, _, ir) ->
              match gen ir with
              | _ -> ()
              | exception e ->
                  Printf.printf
                    "  %-11s %-34s %-9s FAIL (codegen raised: %s)\n%!"
                    dev.Device.framework
                    dev.Device.name
                    label
                    (Printexc.to_string e) ;
                  failed := true)
            cases) ;
      List.iter
        (fun (label, runner, ir) ->
          match runner dev ir with
          | bad ->
              Printf.printf
                "  %-11s %-34s %-9s %s (%d/%d ok)\n%!"
                dev.Device.framework
                dev.Device.name
                label
                (if bad = 0 then "PASS" else "FAIL")
                (n - bad)
                n ;
              if bad <> 0 then failed := true ;
              if host then host_ran := true
          | exception e ->
              let regression =
                host || Test_helpers.Backend_failure.verdict e = `Fail
              in
              Printf.printf
                "  %-11s %-34s %-9s %s (%s)\n%!"
                dev.Device.framework
                dev.Device.name
                label
                (if regression then "FAIL"
                 else "SKIP (backend could not launch)")
                (Printexc.to_string e) ;
              (* Native/Interpreter must always run. On a GPU, a device
                 compiler that rejects the emitted source is a regression of
                 this feature (the min_int literal was one); only a backend
                 that is not there or cannot launch is reported. A wrong
                 result is always a failure. *)
              if regression then failed := true)
        cases)
    devs ;
  if not !host_ran then begin
    print_endline
      "test_int32_bitcasts: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_int32_bitcasts: FAILED" ;
    exit 1
  end ;
  print_endline "test_int32_bitcasts: PASSED"
