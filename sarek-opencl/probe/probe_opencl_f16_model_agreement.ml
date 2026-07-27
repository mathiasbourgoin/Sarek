(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #62 slice 1(a) — is rusticl's deviation ELEMENT-WISE a named model?
 *
 * WHAT THIS ANSWERS, AND WHY IT IS NOT ALREADY ANSWERED.
 *
 * docs/design/f16-relaxed-accuracy.md §2 records rusticl at 620/63488 on
 * f16(f16(x*1.1)+1000), with agreement to S_fuse_mul_into_narrowing
 * established as a COUNT and a FIRST DIVERGENCE. §1.2 does not accept a count.
 * It requires the device result to be BIT-IDENTICAL to one member of the
 * named model set, on every input the gate sweeps. Two functions can differ
 * from a third on the same 620 inputs and still not be the same function.
 *
 * So this probe compares element-wise, against SEVEN named models rather than
 * two, and reports the inputs that match none.
 *
 * WHY A PROBE AND NOT A TEST. It measures a driver in order to decide whether
 * a contract is deliverable; it does not defend an invariant. The gate that
 * defends the OpenCL refusal is test_opencl_f16_tripwire.ml and is untouched.
 *
 * Run:
 *   dune exec sarek-opencl/probe/probe_opencl_f16_model_agreement.exe
 *   dune exec sarek-opencl/probe/probe_opencl_f16_model_agreement.exe -- --host-only
 ******************************************************************************)

open Sarek_opencl
module Backend = Opencl_plugin_base.Opencl
module M = F16_model_set

let n_local = 256

(* The two-narrowing shape, exactly as test_opencl_f16_tripwire.ml writes it,
   so the 620 this probe reads is the 620 that document records and not a
   number from a differently-written kernel. *)
let src_plain =
  {|
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void midround(__global ushort *out, __global const ushort *in, int n) {
  int i = get_global_id(0);
  if (i < n) {
    float x = (float)as_half(in[i]);
    half  m = (half)(x * 1.1f);
    out[i]  = as_ushort((half)((float)m + 1000.0f));
  }
}
|}

(* The one-narrowing shape. rusticl has never been swept on it in this
   project; RADV has, and comparing the two ACO front ends on the SAME two
   shapes is what makes the slice-1 verdict a statement about the compiler
   rather than about one front end. *)
let src_one_narrowing =
  {|
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void midround(__global ushort *out, __global const ushort *in, int n) {
  int i = get_global_id(0);
  if (i < n) {
    float x = (float)as_half(in[i]);
    out[i] = as_ushort((half)(x * 1.1f));
  }
}
|}

(* GREEN CONTROL, two-narrowing shape. A volatile __local round-trip on each
   f32 intermediate — measured to restore the discipline on this stack. It must
   report S_strict exactly; until it does, a disagreement elsewhere could be a
   wrong buffer layout rather than a statement about ACO. *)
let src_barriered =
  Printf.sprintf
    {|
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void midround(__global ushort *out, __global const ushort *in, int n) {
  __local volatile float s[%d];
  int i = get_global_id(0);
  int l = get_local_id(0);
  if (i < n) {
    float x = (float)as_half(in[i]);
    s[l] = x * 1.1f;
    half m = (half)s[l];
    s[l] = (float)m + 1000.0f;
    out[i] = as_ushort((half)s[l]);
  }
}
|}
    n_local

(* GREEN CONTROL, one-narrowing shape. Same barrier, one narrowing. *)
let src_barriered_one =
  Printf.sprintf
    {|
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void midround(__global ushort *out, __global const ushort *in, int n) {
  __local volatile float s[%d];
  int i = get_global_id(0);
  int l = get_local_id(0);
  if (i < n) {
    float x = (float)as_half(in[i]);
    s[l] = x * 1.1f;
    out[i] = as_ushort((half)s[l]);
  }
}
|}
    n_local

(* POSITIVE CONTROL. Deliberately performs the fusion, so the harness is shown
   able to report S_fuse_mul_into_narrowing on a stack that does not fuse on
   its own. On rusticl the contrast is free (plain 620, barriered 0); the
   control exists so that this probe keeps discriminating if it is ever pointed
   at pocl or IGC, where every variant returns the same thing and a
   silently-broken sweep is indistinguishable from a clean one.

   NOT built on binary64. tools/probes/opencl_f16_contraction_probe.c's
   `fusedctl` uses `double` and says a device without cl_khr_fp64 should fail
   loudly — and rusticl on this box IS such a device: it does not advertise the
   extension and the build is rejected. So the control is built the way the
   Metal probe had to build it (MSL has no `double` either): the exact product
   is carried as an unevaluated f32 pair via Dekker's twoProd, rounded to odd,
   and then narrowed once.

   Why round-to-odd is exact here rather than approximately right: binary32 has
   24 significand bits and binary16 has 11, and 24 >= 2*11 + 2. That is the
   condition under which a round-to-odd intermediate followed by
   round-to-nearest-even gives the same answer as rounding the exact value
   once. It holds with no margin, which is why it is stated. *)
let src_fusedctl =
  {|
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
static inline float exact_prod_ro(float x, float c) {
  float hi = x * c;
  float lo = fma(x, c, -hi);   /* hi + lo is EXACTLY x*c */
  if (lo == 0.0f) return hi;
  uint u = as_uint(hi);
  if ((u & 1u) == 0u) u = ((lo > 0.0f) == (hi > 0.0f)) ? u + 1u : u - 1u;
  return as_float(u);
}
__kernel void midround(__global ushort *out, __global const ushort *in, int n) {
  int i = get_global_id(0);
  if (i < n) {
    float x = (float)as_half(in[i]);
    half  m = (half)exact_prod_ro(x, 1.1f);
    out[i]  = as_ushort((half)((float)m + 1000.0f));
  }
}
|}

let run_kernel device ~source ~inputs =
  let n = Array.length inputs in
  let host_in = Bigarray.(Array1.create int16_unsigned c_layout n) in
  Array.iteri (fun i b -> host_in.{i} <- b) inputs ;
  let din = Backend.Memory.alloc device n Bigarray.int16_unsigned in
  let dout = Backend.Memory.alloc device n Bigarray.int16_unsigned in
  Backend.Memory.host_to_device ~src:host_in ~dst:din ;
  let compiled = Backend.Kernel.compile device ~name:"midround" ~source in
  let args = Backend.Kernel.create_args () in
  Backend.Kernel.set_arg_buffer args 0 dout ;
  Backend.Kernel.set_arg_buffer args 1 din ;
  Backend.Kernel.set_arg_int32 args 2 (Int32.of_int n) ;
  let block = Spoc_framework.Framework_sig.dims_1d n_local in
  let grid =
    Spoc_framework.Framework_sig.dims_1d ((n + n_local - 1) / n_local)
  in
  Backend.Kernel.launch compiled ~args ~grid ~block ~shared_mem:0 ~stream:None ;
  Backend.Device.synchronize device ;
  let host_out = Bigarray.(Array1.create int16_unsigned c_layout n) in
  Backend.Memory.device_to_host ~src:dout ~dst:host_out ;
  Backend.Memory.free din ;
  Backend.Memory.free dout ;
  Array.init n (fun i -> host_out.{i})

let contains ~needle haystack =
  let n = String.length needle and h = String.length haystack in
  let lower = String.lowercase_ascii in
  let needle = lower needle and haystack = lower haystack in
  let rec go i =
    i + n <= h && (String.sub haystack i n = needle || go (i + 1))
  in
  n = 0 || go 0

let describe device = Backend.Device.name device

(* Same key as the tripwire: the COMPILER's own name in the device string.
   "ACO" is the component that performs the fusion; a device-model substring
   would be the wrong key and is deliberately not used. *)
let is_in_scope device = contains ~needle:"aco" (describe device)

let devices () =
  if not (Backend.is_available ()) then [||]
  else begin
    Backend.Device.init () ;
    Array.init (Backend.Device.count ()) Backend.Device.get
  end

let strict_of models = List.find (fun m -> m.M.name = "S_strict") models

let sweep device ~label ~source ~models =
  let got = run_kernel device ~source ~inputs:M.finite_bits in
  let c = M.classify models got in
  M.print_classification ~label c ;
  (got, c)

let report_ceiling models c =
  let strict = strict_of models in
  match c.M.exact_matches with
  | [] -> ()
  | names ->
      Printf.printf
        "    §1.3 ceiling, evaluated AT THE ELIDED NARROWING (not on the final \
         value):\n" ;
      List.iter
        (fun n ->
          let m = List.find (fun m -> m.M.name = n) models in
          M.ceiling_report ~model:m ~strict)
        names

let probe_device device =
  Printf.printf
    "\n================================================================\n" ;
  Printf.printf "device: %s\n" (describe device) ;
  Printf.printf
    "================================================================\n%!" ;

  (* Controls first. A device number printed before its control is a number
     nobody can read. *)
  let _, cb =
    sweep
      device
      ~label:"GREEN CONTROL — volatile __local barrier, two-narrowing shape"
      ~source:src_barriered
      ~models:M.shape2_models
  in
  let green_ok = List.mem "S_strict" cb.M.exact_matches in
  if not green_ok then
    Printf.printf
      "    *** CONTROL BROKEN: the barriered kernel does not reproduce \
       S_strict. Nothing below is attributable to ACO. ***\n" ;

  let _, cf =
    sweep
      device
      ~label:"POSITIVE CONTROL — deliberate fusion via twoProd + round-to-odd"
      ~source:src_fusedctl
      ~models:M.shape2_models
  in
  if not (List.mem "S_fuse_mul_into_narrowing" cf.M.exact_matches) then
    Printf.printf
      "    *** CONTROL BROKEN: the deliberate-fusion kernel does not reproduce \
       S_fuse_mul_into_narrowing element-wise. ***\n" ;

  let _, cb1 =
    sweep
      device
      ~label:"GREEN CONTROL — volatile __local barrier, one-narrowing shape"
      ~source:src_barriered_one
      ~models:M.shape1_models
  in
  if not (List.mem "S_strict" cb1.M.exact_matches) then
    Printf.printf "    *** CONTROL BROKEN on the one-narrowing shape. ***\n" ;

  (* The measurements. *)
  Printf.printf "\n  --- shape: %s ---\n" M.shape1_name ;
  let _, c1 =
    sweep
      device
      ~label:"plain (naive codegen)"
      ~source:src_one_narrowing
      ~models:M.shape1_models
  in
  report_ceiling M.shape1_models c1 ;

  Printf.printf "\n  --- shape: %s ---\n" M.shape2_name ;
  let _, c2 =
    sweep
      device
      ~label:"plain (naive codegen)"
      ~source:src_plain
      ~models:M.shape2_models
  in
  report_ceiling M.shape2_models c2 ;

  (c1, c2)

let () =
  let host_only = Array.exists (fun a -> a = "--host-only") Sys.argv in
  Printf.printf
    "#62 slice 1(a) — element-wise model agreement, OpenCL / rusticl\n\n" ;
  (try M.calibrate ()
   with M.Calibration_failed s ->
     Printf.printf "CALIBRATION FAILED — read nothing below it:\n  %s\n" s ;
     exit 1) ;
  Printf.printf
    "host calibration PASSED: 63488-value round-trip; the two §1.2 models \
     separate on exactly 620 (%s) and 2912 (%s); §1.3's x = -907.5 case \
     reproduces at 1 ulp at the narrowing and 512 ulp on the final value.\n\n"
    M.shape2_name
    M.shape1_name ;

  Printf.printf "SHAPE 1 — %s\n" M.shape1_name ;
  let sep1 = M.separation_matrix M.shape1_models in
  Printf.printf "\nSHAPE 2 — %s\n" M.shape2_name ;
  let sep2 = M.separation_matrix M.shape2_models in
  if not (sep1 && sep2) then
    Printf.printf
      "\n\
       *** at least one model pair coincides on the whole domain — a device \
       matching one of them is not evidence for either ***\n" ;

  if host_only then exit 0 ;

  let ds = Array.to_list (devices ()) in
  let in_scope, out_of_scope = List.partition is_in_scope ds in
  List.iter
    (fun d -> Printf.printf "\nout of scope: %s\n" (describe d))
    out_of_scope ;
  match in_scope with
  | [] ->
      Printf.printf
        "\n\
         [NO DEVICE] No OpenCL device whose name contains \"ACO\". Saw: %s. \
         Nothing is measured; this is not a null result.\n"
        (match ds with
        | [] -> "no OpenCL devices at all"
        | l -> String.concat "; " (List.map describe l)) ;
      exit 2
  | ds -> List.iter (fun d -> ignore (probe_device d)) ds
