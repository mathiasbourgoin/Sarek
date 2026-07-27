(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #62 slice 1(b) — the RADV two-narrowing shape, and the `precise` puzzle.
 *
 * THE OPEN RISK THIS EXISTS TO SETTLE.
 *
 * docs/design/f16-relaxed-accuracy.md §9.3 names one thing as most likely to
 * break the relaxed-accuracy design: on f16(f16(x*1.1)+1000), RADV disagrees
 * with the discipline on 5075/63488 plain and 4776/63488 with `precise`, while
 * fp-contraction-policy.md §6 shows `precise` producing BYTE-IDENTICAL ISA on
 * the one-narrowing shape. A decoration that changes the answer on one shape
 * and changes nothing on another is not obviously one behaviour, and nobody
 * had reconciled the two facts. If the two-narrowing shape matches NO
 * closed-form model, §1.2's contract cannot be stated per backend.
 *
 * Neither count was ever compared against a model. This probe compares
 * element-wise against seven named models, on both local RADV devices.
 *
 * WHY A PROBE AND NOT A TEST. It measures a driver to decide whether a
 * contract is deliverable; it does not defend an invariant. The gate defending
 * the GLSL refusal is sarek-vulkan/test/test_vulkan_f16_tripwire.ml and is
 * untouched.
 *
 * Run:
 *   dune exec sarek-vulkan/probe/probe_vulkan_f16_model_agreement.exe
 *   RADV_DEBUG=asm dune exec ... 2> isa.txt     (for the machine-code tier)
 *
 * THE shaderFloat16 CAVEAT, inherited verbatim from the tripwire. Sarek's
 * Vulkan device creation chains no feature structs beyond core
 * VkPhysicalDeviceFeatures, so the SPIR-V Float16 capability is used without
 * the feature being enabled; RADV accepts it anyway. That plumbing is §7 slice
 * 2. It does not weaken the finding — the green control below reproduces
 * S_strict bit-exactly on the same un-enabled path.
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Kernel = Vulkan_api_kernel
module M = F16_model_set

let n_local = 256

(* Buffers are `uint` holding one binary16 bit pattern in the low 16 bits,
   rather than 16-bit storage: it keeps VK_KHR_16bit_storage out of the picture
   so the only 16-bit thing in play is the arithmetic under test. Identical
   framing to the tripwire, so the counts are comparable to the recorded ones. *)
let sh ?(prelude = "") body =
  Printf.sprintf
    {|#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = %d) in;
layout(std430, binding = 0) volatile buffer Out { uint outb[]; };
layout(std430, binding = 1) readonly buffer In { uint inb[]; };
uint pack(float16_t r) {
  return packFloat2x16(f16vec2(r, float16_t(0.0))) & 0xFFFFu;
}
%s
void main() {
  uint i = gl_GlobalInvocationID.x;
  float x = float(unpackFloat2x16(inb[i]).x);
%s
}
|}
    n_local
    prelude
    body

(* ---------------------------------------------------------------------- *)
(* SHAPE 1 — f16(x * 1.1)                                                   *)
(* ---------------------------------------------------------------------- *)

let s1_plain = sh "  outb[i] = pack(float16_t(x * 1.1));"

let s1_precise =
  sh "  precise float p = x * 1.1;\n  outb[i] = pack(float16_t(p));"

(* GREEN CONTROL: the f32 product forced through the volatile SSBO, which is
   the one defence measured to work on this backend. *)
let s1_barriered =
  sh
    "  outb[i] = floatBitsToUint(x * 1.1);\n\
    \  outb[i] = pack(float16_t(uintBitsToFloat(outb[i])));"

(* ---------------------------------------------------------------------- *)
(* SHAPE 2 — f16(f16(x * 1.1) + 1000)                                       *)
(* ---------------------------------------------------------------------- *)

let s2_body ~qual =
  Printf.sprintf
    "  %sfloat p = x * 1.1;\n\
    \  float16_t m = float16_t(p);\n\
    \  %sfloat q = float(m) + 1000.0;\n\
    \  outb[i] = pack(float16_t(q));"
    qual
    qual

let s2_plain = sh (s2_body ~qual:"")

(* `precise` on every float local is exactly what Sarek_ir_glsl.gen_var_decl
   already emits, so this variant is the one that matters for the shipped
   codegen — not the plain one. *)
let s2_precise = sh (s2_body ~qual:"precise ")

(* GREEN CONTROL: BOTH the f32 intermediates AND the f16 bit pattern forced
   through the volatile SSBO. Barriering only the f32 intermediates is NOT
   enough here — ACO responds by dropping the intermediate narrowing entirely
   (fp-contraction-policy.md §2, 4774/63488), which is why the f16 value has to
   go through memory too. *)
let s2_barriered =
  sh
    "  outb[i] = floatBitsToUint(x * 1.1);\n\
    \  outb[i] = pack(float16_t(uintBitsToFloat(outb[i])));\n\
    \  outb[i] = floatBitsToUint(float(unpackFloat2x16(outb[i]).x) + 1000.0);\n\
    \  outb[i] = pack(float16_t(uintBitsToFloat(outb[i])));"

(* AMBER CONTROL: barrier the f32 intermediates ONLY. Recorded at 4774/63488,
   the count this document attributes to a dropped intermediate narrowing —
   which has never been checked element-wise against that model either. *)
let s2_barriered_f32_only =
  sh
    "  outb[i] = floatBitsToUint(x * 1.1);\n\
    \  float16_t m = float16_t(uintBitsToFloat(outb[i]));\n\
    \  outb[i] = floatBitsToUint(float(m) + 1000.0);\n\
    \  outb[i] = pack(float16_t(uintBitsToFloat(outb[i])));"

(* POSITIVE CONTROL. Deliberately performs the multiply-into-narrowing fusion,
   so the harness is shown able to report S_fuse_mul_into_narrowing on a shape
   or a driver that does not fuse on its own. Built without `double` — the
   exact product is carried as an unevaluated f32 pair via Dekker's twoProd,
   rounded to odd, then narrowed once. Round-to-odd then round-to-nearest is
   exact here because binary32 has 24 significand bits, binary16 has 11, and
   24 >= 2*11 + 2. That holds with no margin, which is why it is stated. *)
let ro_prelude =
  {|float exact_prod_ro(float a, float c) {
  float hi = a * c;
  float lo = fma(a, c, -hi);
  if (lo == 0.0) return hi;
  uint u = floatBitsToUint(hi);
  if ((u & 1u) == 0u) u = ((lo > 0.0) == (hi > 0.0)) ? u + 1u : u - 1u;
  return uintBitsToFloat(u);
}|}

(* The two-narrowing control needs the SSBO barrier as well as the round-to-odd
   product, and that is a measured requirement rather than caution. Written the
   obvious way —

     float16_t m = float16_t(exact_prod_ro(x, 1.1));
     outb[i] = pack(float16_t(float(m) + 1000.0));

   — it does NOT construct S_fuse_mul_into_narrowing on RADV: ACO elides the
   intermediate narrowing anyway and absorbs the add, so the kernel lands on
   S_absorb_all_into_final_narrowing (63487/63488, the one straggler being the
   round-to-odd product's double rounding, which is innocuous into binary16 but
   not into an f32 add). Measured 2026-07-27 on both local RADV devices. So the
   f16 bit pattern goes through the volatile SSBO here too: the control is
   allowed to be expensive, it is not allowed to be defeated. *)
let s2_fusedctl =
  sh
    ~prelude:ro_prelude
    "  outb[i] = floatBitsToUint(exact_prod_ro(x, 1.1));\n\
    \  outb[i] = pack(float16_t(uintBitsToFloat(outb[i])));\n\
    \  outb[i] = floatBitsToUint(float(unpackFloat2x16(outb[i]).x) + 1000.0);\n\
    \  outb[i] = pack(float16_t(uintBitsToFloat(outb[i])));"

let s1_fusedctl =
  sh ~prelude:ro_prelude "  outb[i] = pack(float16_t(exact_prod_ro(x, 1.1)));"

(* ---------------------------------------------------------------------- *)

let run device ~source ~inputs =
  let n = Array.length inputs in
  let host_in = Bigarray.(Array1.create int32 c_layout n) in
  Array.iteri (fun i b -> host_in.{i} <- Int32.of_int b) inputs ;
  let din = Memory.alloc device n Bigarray.int32 in
  let dout = Memory.alloc device n Bigarray.int32 in
  Memory.host_to_device ~src:host_in ~dst:din ;
  let compiled = Kernel.compile device ~name:"main" ~source in
  let args = Kernel.create_args () in
  Kernel.set_arg_buffer args 0 dout ;
  Kernel.set_arg_buffer args 1 din ;
  let block = Spoc_framework.Framework_sig.dims_1d n_local in
  let grid = Spoc_framework.Framework_sig.dims_1d (n / n_local) in
  Kernel.launch compiled ~args ~grid ~block ~shared_mem:0 ~stream:None ;
  Device.synchronize device ;
  let host_out = Bigarray.(Array1.create int32 c_layout n) in
  Memory.device_to_host ~src:dout ~dst:host_out ;
  Memory.free din ;
  Memory.free dout ;
  Array.init n (fun i -> Int32.to_int host_out.{i} land 0xFFFF)

let contains ~needle haystack =
  let n = String.length needle and h = String.length haystack in
  let lower = String.lowercase_ascii in
  let needle = lower needle and haystack = lower haystack in
  let rec go i =
    i + n <= h && (String.sub haystack i n = needle || go (i + 1))
  in
  n = 0 || go 0

let describe device = device.Device.name

(* Keyed on DRIVER identity, as the tripwire is: RADV compiles with ACO, which
   is the component that performs the combine. Not on a device model. *)
let is_in_scope device = contains ~needle:"radv" (describe device)

let devices () =
  if not (Vulkan_api.is_available ()) then [||]
  else begin
    Device.init () ;
    Array.init (Device.count ()) Device.get
  end

let strict_of models = List.find (fun m -> m.M.name = "S_strict") models

let sweep device ~label ~source ~models =
  let got = run device ~source ~inputs:M.finite_bits in
  let c = M.classify models got in
  M.print_classification ~label c ;
  c

let report_ceiling models c =
  let strict = strict_of models in
  match c.M.exact_matches with
  | [] ->
      Printf.printf
        "    §1.3 ceiling: NOT EVALUATED — no model matched, so there is no \
         admitted deviation to measure.\n"
  | names ->
      Printf.printf
        "    §1.3 ceiling, evaluated AT THE ELIDED NARROWING (not on the final \
         value):\n" ;
      List.iter
        (fun n ->
          let m = List.find (fun m -> m.M.name = n) models in
          M.ceiling_report ~model:m ~strict)
        names

(* One named variant, one device, one shader compiled — so that
   `RADV_DEBUG=asm` produces an ISA dump attributable to a single shader. With
   the full run, several shaders are compiled in sequence and reading the ISA
   means guessing which dump belongs to which, which is not a machine-code
   evidence tier. *)
let variants =
  [
    ("s1_plain", (s1_plain, M.shape1_models));
    ("s1_precise", (s1_precise, M.shape1_models));
    ("s2_plain", (s2_plain, M.shape2_models));
    ("s2_precise", (s2_precise, M.shape2_models));
  ]

let probe_one device name =
  match List.assoc_opt name variants with
  | None ->
      Printf.printf
        "unknown variant %S; known: %s\n"
        name
        (String.concat ", " (List.map fst variants)) ;
      exit 2
  | Some (source, models) ->
      Printf.printf
        "device: %s\nvariant: %s\nGLSL:\n%s\n"
        (describe device)
        name
        source ;
      let c = sweep device ~label:name ~source ~models in
      report_ceiling models c

let probe_device device =
  Printf.printf
    "\n================================================================\n" ;
  Printf.printf "device: %s\n" (describe device) ;
  Printf.printf
    "================================================================\n%!" ;

  Printf.printf "\n  --- controls ---\n" ;
  let cg1 =
    sweep
      device
      ~label:"GREEN CONTROL — f16 bit pattern through the SSBO, one narrowing"
      ~source:s1_barriered
      ~models:M.shape1_models
  in
  if not (List.mem "S_strict" cg1.M.exact_matches) then
    Printf.printf
      "    *** CONTROL BROKEN: nothing below is attributable to ACO ***\n" ;
  let cg2 =
    sweep
      device
      ~label:"GREEN CONTROL — f16 bit pattern through the SSBO, two narrowings"
      ~source:s2_barriered
      ~models:M.shape2_models
  in
  if not (List.mem "S_strict" cg2.M.exact_matches) then
    Printf.printf
      "    *** CONTROL BROKEN: nothing below is attributable to ACO ***\n" ;
  let cp1 =
    sweep
      device
      ~label:"POSITIVE CONTROL — deliberate fusion, one narrowing"
      ~source:s1_fusedctl
      ~models:M.shape1_models
  in
  if not (List.mem "S_fuse_mul_into_narrowing" cp1.M.exact_matches) then
    Printf.printf
      "    *** POSITIVE CONTROL did not reproduce the fused model ***\n" ;
  let cp2 =
    sweep
      device
      ~label:"POSITIVE CONTROL — deliberate fusion, two narrowings"
      ~source:s2_fusedctl
      ~models:M.shape2_models
  in
  if not (List.mem "S_fuse_mul_into_narrowing" cp2.M.exact_matches) then
    Printf.printf
      "    *** POSITIVE CONTROL did not reproduce the fused model ***\n" ;

  Printf.printf "\n  --- shape: %s ---\n" M.shape1_name ;
  let c =
    sweep device ~label:"plain" ~source:s1_plain ~models:M.shape1_models
  in
  report_ceiling M.shape1_models c ;
  let c =
    sweep
      device
      ~label:"precise (what Sarek_ir_glsl emits)"
      ~source:s1_precise
      ~models:M.shape1_models
  in
  report_ceiling M.shape1_models c ;

  Printf.printf "\n  --- shape: %s ---\n" M.shape2_name ;
  let c =
    sweep device ~label:"plain" ~source:s2_plain ~models:M.shape2_models
  in
  report_ceiling M.shape2_models c ;
  let c =
    sweep
      device
      ~label:"precise (what Sarek_ir_glsl emits)"
      ~source:s2_precise
      ~models:M.shape2_models
  in
  report_ceiling M.shape2_models c ;
  let c =
    sweep
      device
      ~label:"volatile SSBO on the f32 intermediates ONLY"
      ~source:s2_barriered_f32_only
      ~models:M.shape2_models
  in
  report_ceiling M.shape2_models c

let () =
  let host_only = Array.exists (fun a -> a = "--host-only") Sys.argv in
  Printf.printf
    "#62 slice 1(b) — element-wise model agreement, Vulkan / RADV\n\n" ;
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
         [NO DEVICE] No Vulkan device whose name contains \"RADV\". Saw: %s. \
         Nothing is measured; this is not a null result.\n"
        (match ds with
        | [] -> "no Vulkan devices at all"
        | l -> String.concat "; " (List.map describe l)) ;
      exit 2
  | ds -> (
      let rec only i =
        if i + 1 >= Array.length Sys.argv then None
        else if Sys.argv.(i) = "--variant" then Some Sys.argv.(i + 1)
        else only (i + 1)
      in
      match only 1 with
      | Some v -> probe_one (List.hd ds) v
      | None -> List.iter probe_device ds)
