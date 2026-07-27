(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #57 slice 2b — GLSL/Vulkan f16 refusal TRIPWIRE.
 *
 * WHAT THIS TEST IS FOR, because it is not what it looks like.
 *
 * [Sarek_ir_glsl] refuses float16. That refusal is not "unimplemented" — the
 * codegen is a small change and the GLSL extension compiles and runs here. It
 * is a REFUSAL BY MEASUREMENT: RADV's ACO backend absorbs the f32->f16
 * narrowing into the arithmetic that feeds it (v_fma_mixlo_f16), rounding once
 * where slice 1's f16 discipline mandates twice, so 2912 of the 63488 finite
 * binary16 inputs disagree with the interpreter on the single-narrowing shape
 * this file measures. See docs/fp-contraction-policy.md, row
 * "Vulkan / RADV (f16 narrowing)".
 *
 * A refusal backed by a measurement can quietly outlive its own justification.
 * If Mesa stops fusing, the refusal becomes wrong, and nothing would tell us —
 * the codegen would keep raising, the golden test would keep passing, and the
 * documentation would keep citing a defect that no longer exists. So this test
 * asserts the REASON, not the behaviour. It does not check that Sarek refuses
 * f16 (test_cuda_f16_golden pins that). It checks that refusing is still
 * WARRANTED. It goes red when the fusion STOPS.
 *
 * WHY THIS ONE USES A HOST REFERENCE, WHERE THE OPENCL TRIPWIRE COMPARES TWO
 * KERNELS.
 *
 * test_opencl_f16_tripwire.ml avoids host binary16 arithmetic entirely by
 * diffing a naive kernel against a barriered one. That trick is not available
 * in full here: on RADV, putting a barrier around the f32 intermediates of the
 * two-narrowing shape does NOT restore the discipline — ACO responds by
 * dropping the intermediate narrowing altogether rather than materialising a
 * binary16 value (4774/63488, measured; see the table in
 * docs/fp-contraction-policy.md). A barriered kernel is therefore not a
 * trustworthy oracle on this backend at that shape.
 *
 * So the oracle is a host implementation of the discipline, and it is
 * CALIBRATED before it is believed, three ways:
 *
 *   1. [host_rounding_round_trips] — [f16_bits] re-encodes every one of the
 *      63488 finite binary16 values to its own bit pattern. Catches an encoder
 *      that is wrong about subnormals, the binade edges or the carry case.
 *   2. [host_models_reproduce_the_620] — the SAME host code, applied to the
 *      HIP/OpenCL kernel shape, must separate the two-roundings model from the
 *      one-rounding model on exactly 620 inputs. 620 is the figure independently
 *      measured on hiprtc/gfx1100 and on rusticl/radeonsi. Reproducing a
 *      known positive with an independent implementation is what makes the
 *      rest of the file's zeros and nonzeros worth reading.
 *   3. [barrier_variant_matches_the_discipline] — a kernel whose f32
 *      intermediate is forced through a volatile SSBO round-trip agrees with
 *      the host model on all 63488 inputs. This is the green control: it proves
 *      the harness CAN report agreement, so "disagrees" below is a statement
 *      about ACO and not about a broken buffer layout, a broken pack/unpack, or
 *      the Float16 caveat in the next paragraph.
 *
 * A CAVEAT THAT WAS HONEST AND HAS NOW BEEN RETIRED BY MEASUREMENT.
 *
 * Vulkan requires the [shaderFloat16] feature to be enabled at device creation
 * before a shader may use the SPIR-V Float16 capability, and until backlog-62
 * slice 2 Sarek chained no feature structs beyond core
 * VkPhysicalDeviceFeatures — so these shaders ran on a path where the feature
 * was never requested, and RADV accepted them anyway. [Vulkan_api_device] now
 * queries VkPhysicalDeviceShaderFloat16Int8Features through the Features2 chain
 * and REQUESTS shaderFloat16 (and storageBuffer16BitAccess) at vkCreateDevice.
 *
 * The numbers below did not move. Run as a controlled A/B on 2026-07-27 — one
 * build, the feature request toggled, this executable run on each arm — both
 * arms report 2912/63488 on the RX 7900 XTX AND on the Raphael iGPU, with the
 * same first divergence (x = 8.94069672e-07, device 0x0011, discipline 0x0010),
 * the same `precise` figures, and all three calibration controls green. So the
 * caveat named a real gap in the plumbing but not a confound in the
 * measurement: ACO's absorption of the f32->f16 narrowing does not depend on
 * whether the feature was requested. The defect remains visible in the emitted
 * ISA too (run this executable under RADV_DEBUG=asm to see v_fma_mixlo_f16 for
 * the plain variant and a separate conversion for the barriered one).
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Kernel = Vulkan_api_kernel

let n_local = 256

(* Buffers are `uint` holding one binary16 bit pattern in the low 16 bits,
   rather than 16-bit storage: it keeps VK_KHR_16bit_storage out of the picture
   so the only 16-bit thing in play is the arithmetic under test. The
   pack/unpack framing is not itself a barrier — the plain variant below still
   fuses with it in place, and the barriered variant still agrees with it in
   place. *)
let shader body =
  Printf.sprintf
    {|#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = %d) in;
layout(std430, binding = 0) volatile buffer Out { uint outb[]; };
layout(std430, binding = 1) readonly buffer In { uint inb[]; };
uint pack(float16_t r) {
  return packFloat2x16(f16vec2(r, float16_t(0.0))) & 0xFFFFu;
}
void main() {
  uint i = gl_GlobalInvocationID.x;
  float x = float(unpackFloat2x16(inb[i]).x);
%s
}
|}
    n_local
    body

(* The narrowing shape that fuses: an f32 multiply consumed by an f32->f16
   narrowing. Sarek's discipline rounds twice here (once to binary32 at the
   multiply, once to binary16 at the narrowing). *)
let src_plain = shader "  outb[i] = pack(float16_t(x * 1.1));"

(* Identical, except the f32 local carries `precise` — which is what
   Sarek_ir_glsl.gen_var_decl already emits on every float local, and which
   glslang lowers to SPIR-V NoContraction. Present as its own variant because
   "we already emit precise, so we are fine" is the inference this measurement
   exists to refute. *)
let src_precise =
  shader "  precise float p = x * 1.1;\n  outb[i] = pack(float16_t(p));"

(* Identical, except the f32 product is forced through a volatile SSBO
   round-trip before being narrowed — measured to be the one defence that works.
   Not shippable as codegen (a global-memory round-trip per narrowing, into a
   scratch buffer this backend does not control) but a perfectly good control. *)
let src_barriered =
  shader
    "  outb[i] = floatBitsToUint(x * 1.1);\n\
    \  outb[i] = pack(float16_t(uintBitsToFloat(outb[i])));"

(* ---------------------------------------------------------------------- *)
(* HOST REFERENCE                                                           *)
(* ---------------------------------------------------------------------- *)

(* Round a binary64 to binary32. Every intermediate below is exactly
   representable in binary64 before this is applied (a binary16 operand times a
   binary32 constant needs at most 35 significand bits), so this is a single
   correct rounding and not a double rounding. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Exact value of a binary16 bit pattern; None for NaN/Inf. Decoding needs no
   rounding, so it cannot itself be a source of mismatch. *)
let f16_decode b =
  let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
  let e = (b lsr 10) land 0x1f and m = b land 0x3ff in
  if e = 31 then None
  else if e = 0 then Some (sign *. float_of_int m *. ldexp 1.0 (-24))
  else Some (sign *. float_of_int (1024 + m) *. ldexp 1.0 (e - 25))

let dec b =
  match f16_decode b with
  | Some v -> v
  | None ->
      if b land 0x3ff <> 0 then Float.nan
      else if b land 0x8000 <> 0 then Float.neg_infinity
      else Float.infinity

let round_even v =
  let f = Float.floor v in
  let r = v -. f in
  if r > 0.5 then f +. 1.0
  else if r < 0.5 then f
  else if Float.rem f 2.0 = 0.0 then f
  else f +. 1.0

(* Round-to-nearest-even of an exactly-represented real to a binary16 bit
   pattern. [round_even] is exact because [a /. ldexp 1.0 k] is a power-of-two
   rescale, and the rescaled value is below 2048. *)
let f16_bits d =
  if Float.is_nan d then 0x7E00
  else
    let s = if d < 0.0 || (d = 0.0 && 1.0 /. d < 0.0) then 0x8000 else 0 in
    let a = Float.abs d in
    if a = Float.infinity then s lor 0x7C00
    else if a = 0.0 then s
    else
      (* [frexp] is exact by construction: it returns [(m, k)] with
         [0.5 <= |m| < 1] and [a = m * 2^k], so the unbiased exponent is
         [k - 1]. Deriving it from [log2] instead would need a correction step
         for the binade edges, and that correction cannot be exercised by any
         input in this domain — an unfalsifiable branch is worse than none. *)
      let e = snd (Float.frexp a) - 1 in
      if e < -14 then
        (* subnormal; a carry out of the 10-bit field lands exactly on the
           smallest normal, which the bit layout already spells correctly *)
        s lor int_of_float (round_even (a /. ldexp 1.0 (-24)))
      else
        let q = round_even (a /. ldexp 1.0 (e - 10)) in
        let e, q = if q >= 2048.0 then (e + 1, 1024.0) else (e, q) in
        if e + 15 >= 31 then s lor 0x7C00
        else s lor ((e + 15) lsl 10) lor (int_of_float q - 1024)

(* All finite binary16 bit patterns. 63488 of them — small enough that the
   exhaustive statement is affordable, which is why the f16 gates in this
   project are exhaustive rather than sampled. Also exactly 248 * 256, so the
   kernels need no bounds check and no scalar argument. *)
let finite_bits =
  let acc = ref [] in
  for b = 0xFFFF downto 0 do
    if f16_decode b <> None then acc := b :: !acc
  done ;
  Array.of_list !acc

let c11 = f32 1.1

(* Sarek's discipline for [f16(x * 1.1)]: two roundings. *)
let ref_discipline b = f16_bits (f32 (dec b *. c11))

(* The multiply absorbed into the narrowing: one rounding. *)
let ref_fused b = f16_bits (dec b *. c11)

(* The HIP/OpenCL kernel shape, [f16(f16(x*1.1) + 1000)], under each model.
   Used only by the 620 calibration — no kernel here computes it. *)
let ref_hipshape_discipline b =
  let m = f16_bits (f32 (dec b *. c11)) in
  f16_bits (f32 (dec m +. 1000.0))

let ref_hipshape_fused b =
  let m = f16_bits (dec b *. c11) in
  f16_bits (f32 (dec m +. 1000.0))

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

(* count of disagreements, and the index of the first *)
let compare_bits got want =
  let c = ref 0 and first = ref (-1) in
  Array.iteri
    (fun i x ->
      if x <> want.(i) then begin
        incr c ;
        if !first < 0 then first := i
      end)
    got ;
  (!c, !first)

(* ---------------------------------------------------------------------- *)
(* SCOPE                                                                    *)
(*                                                                          *)
(* The refusal this test defends is about ONE compiler: the ACO shader      *)
(* backend in Mesa, reached here through Mesa's Vulkan driver (RADV). It is *)
(* NOT a claim about Vulkan or GLSL in general, and asserting it against    *)
(* whatever device happens to be first is how a gate fires for the wrong    *)
(* reason.                                                                  *)
(*                                                                          *)
(* Keyed on DRIVER identity: the device name must contain "RADV". Mesa puts *)
(* its Vulkan driver name in VkPhysicalDeviceProperties::deviceName, e.g.   *)
(* "AMD Radeon RX 7900 XTX (RADV NAVI31)". RADV compiles shaders with ACO,  *)
(* which is the component that performs the fusion, so this is driver       *)
(* identity rather than a device model.                                     *)
(*                                                                          *)
(* Deliberately NOT keyed on a device-model substring ("NAVI31") or on a    *)
(* CPU/GPU distinction. The second local device is named "AMD Ryzen 9 7950X *)
(* 16-Core Processor (RADV RAPHAEL_MENDOCINO)" — an integrated GPU whose    *)
(* NAME reads like a CPU, and it reproduces the defect identically, with    *)
(* identical counts. Any heuristic keyed on the model, or on "looks like a  *)
(* CPU", would silently drop it from scope. The same trap was found on the  *)
(* OpenCL side in slice 2a.                                                 *)
(*                                                                          *)
(* IF THE VENDOR RENAMES: this test SKIPS, it does not silently pass. Mesa  *)
(* renaming RADV, or switching it away from ACO, makes [in_scope] empty,    *)
(* which produces a visible skip naming what it looked for. That is the     *)
(* intended failure direction — an unmeasured platform must never be read   *)
(* as evidence that the refusal still holds — but it does mean a rename     *)
(* converts this gate into a no-op until someone updates the key. The skip  *)
(* message says so, so the silence is at least loud.                        *)
(*                                                                          *)
(* There is deliberately NO "non-RADV Vulkan does not fuse" cross-check     *)
(* here, unlike the OpenCL tripwire's pocl one: no non-RADV Vulkan          *)
(* implementation has been measured for this project, so such a check would *)
(* assert a claim nobody has evidence for.                                  *)
(* ---------------------------------------------------------------------- *)

let contains ~needle haystack =
  let n = String.length needle and h = String.length haystack in
  let lower = String.lowercase_ascii in
  let needle = lower needle and haystack = lower haystack in
  let rec go i =
    i + n <= h && (String.sub haystack i n = needle || go (i + 1))
  in
  n = 0 || go 0

let describe device = device.Device.name

let is_in_scope device = contains ~needle:"radv" (describe device)

let all_devices () =
  if not (Vulkan_api.is_available ()) then [||]
  else begin
    Device.init () ;
    Array.init (Device.count ()) Device.get
  end

(* Partition, report what was seen, and skip visibly when nothing is in scope.
   The listing matters: a skip that does not say which devices it rejected is
   indistinguishable from a skip caused by a broken scope predicate. *)
let with_in_scope_devices f =
  let devices = all_devices () in
  let in_scope, out_of_scope =
    Array.to_list devices |> List.partition is_in_scope
  in
  List.iter
    (fun d -> Printf.printf "    out of scope: %s\n%!" (describe d))
    out_of_scope ;
  match in_scope with
  | [] ->
      let seen =
        match out_of_scope with
        | [] -> "no Vulkan devices at all"
        | l -> String.concat "; " (List.map describe l)
      in
      Printf.printf
        "[SKIP] No in-scope Vulkan device. This tripwire defends a refusal \
         measured on Mesa's ACO backend reached through RADV, so it only \
         asserts on a device whose name contains \"RADV\" (e.g. \"AMD Radeon \
         RX 7900 XTX (RADV NAVI31)\"). Saw: %s. Not asserting: a different \
         Vulkan implementation agreeing here would say nothing about whether \
         ACO still fuses.\n\
         %!"
        seen ;
      Alcotest.skip ()
  | ds -> List.iter f ds

(* ---------------------------------------------------------------------- *)
(* CALIBRATION                                                              *)
(* ---------------------------------------------------------------------- *)

let host_rounding_round_trips () =
  Array.iter
    (fun b ->
      if f16_bits (dec b) <> b then
        Alcotest.failf
          "host binary16 rounding is wrong: re-encoding the exact value of \
           0x%04X gives 0x%04X. Every device verdict in this file is measured \
           against this function, so nothing below means anything until it is \
           fixed."
          b
          (f16_bits (dec b)))
    finite_bits ;
  Alcotest.(check int)
    "the exhaustive finite binary16 domain"
    63488
    (Array.length finite_bits)

let host_models_reproduce_the_620 () =
  let disc = Array.map ref_hipshape_discipline finite_bits in
  let fused = Array.map ref_hipshape_fused finite_bits in
  let diffs, _ = compare_bits disc fused in
  if diffs <> 620 then
    Alcotest.failf
      "CALIBRATION FAILED — do not read any other result in this file.\n\n\
       On the HIP/OpenCL kernel shape f16(f16(x*1.1) + 1000), this file's own \
       host models separate the two-roundings discipline from the \
       fused-first-narrowing behaviour on %d of %d inputs. It must be 620: \
       that is the figure measured independently on hiprtc/gfx1100 \
       (sarek-hip/test/test_hip_f16.ml) and on rusticl/radeonsi \
       (docs/fp-contraction-policy.md).\n\n\
       Reproducing a known positive is what licenses believing this file's \
       other counts. If this number moved, [f16_bits], [f32] or [dec] is wrong \
       — fix the host reference, do not adjust this expectation."
      diffs
      (Array.length finite_bits)

(* The GREEN control. A tripwire that can only ever report disagreement proves
   nothing when it reports disagreement. *)
let barrier_variant_matches_the_discipline () =
  with_in_scope_devices (fun device ->
      let want = Array.map ref_discipline finite_bits in
      let got = run device ~source:src_barriered ~inputs:finite_bits in
      let diffs, first = compare_bits got want in
      if diffs <> 0 then
        Alcotest.failf
          "CONTROL BROKEN on %s: the barriered kernel — whose f32 product is \
           forced through a volatile SSBO round-trip — disagrees with the host \
           model on %d/%d inputs (first at x=%.9g: device 0x%04X, model \
           0x%04X).\n\n\
           This is the control that shows the harness can report AGREEMENT. \
           Until it does, a disagreement reported by the other cases cannot be \
           attributed to ACO: it could equally be a wrong buffer layout, a \
           wrong pack/unpack, or a wrong host reference. Fix this before \
           reading anything else."
          (describe device)
          diffs
          (Array.length finite_bits)
          (dec finite_bits.(first))
          got.(first)
          want.(first))

(* ---------------------------------------------------------------------- *)
(* THE TRIPWIRE                                                             *)
(* ---------------------------------------------------------------------- *)

let obsolete_refusal_message device n =
  Printf.sprintf
    "OBSOLETE REFUSAL, NOT A REGRESSION — READ BEFORE \"FIXING\".\n\n\
     On %s, the naive f32->f16 narrowing now agrees with Sarek's f16 \
     discipline on all %d finite binary16 inputs. The multiply is no longer \
     being absorbed into the narrowing.\n\n\
     This test exists to detect exactly that. Sarek_ir_glsl refuses float16 \
     *because* of that fusion (2912/63488 disagreements when this was \
     measured, 2026-07-26, RADV NAVI31 and RADV RAPHAEL_MENDOCINO, Mesa \
     26.1.4-arch3.1). If the fusion is gone, the refusal has lost its \
     justification and #57 slice 2b should be REVISITED.\n\n\
     Do NOT make this pass by deleting or weakening the assertion. The correct \
     responses are, in order: (1) re-run this executable under RADV_DEBUG=asm \
     and confirm v_fma_mixlo_f16 is gone from the plain variant, (2) \
     re-measure the two-narrowing shapes too — this case covers only the \
     single narrowing, and the two-narrowing shape failed WORSE (5075/63488), \
     (3) record the new measurement in docs/fp-contraction-policy.md, (4) \
     enable f16 in Sarek_ir_glsl behind the usual exhaustive \
     interpreter-agreement gate, and (5) delete this test as part of THAT \
     change, not before it. The shaderFloat16 device-feature plumbing this \
     step used to call for was built in backlog-62 slice 2 and is no longer \
     outstanding."
    device
    n

let refusal_is_still_warranted () =
  with_in_scope_devices (fun device ->
      let want = Array.map ref_discipline finite_bits in
      let got = run device ~source:src_plain ~inputs:finite_bits in
      let n = Array.length finite_bits in
      let diffs, first = compare_bits got want in
      if diffs > 0 then
        Printf.printf
          "    [%s] fusion still present: %d/%d differ; first at x=%.9g \
           (device 0x%04X, discipline 0x%04X)\n\
           %!"
          (describe device)
          diffs
          n
          (dec finite_bits.(first))
          got.(first)
          want.(first) ;
      if diffs = 0 then
        Alcotest.failf "%s" (obsolete_refusal_message (describe device) n) ;
      (* Not asserted as an exact count: it is a Mesa-version-dependent number.
         Asserted as a range, because "everything differs" would mean the kernel
         is no longer computing the intended expression at all. *)
      if diffs = n then
        Alcotest.failf
          "all %d inputs differ on %s — that is not a fusion signature, it \
           means the kernel and the host model are no longer computing the \
           same expression. Fix the tripwire, do not read this as evidence \
           about Mesa."
          n
          (describe device))

(* Separate case, deliberately. The one above is the CI-blocking essential
   ("is fusion still happening at all"). This one pins the MECHANISM, and is
   the specific inference that would otherwise get made from #106/#126: that
   backend already emits `precise` on every float local, #106 measured 0 of 7
   f32 contraction shapes on RADV, so surely f16 is safe. It is not — `precise`
   stops a*b+c from contracting, which is a different combine from a conversion
   absorbing its operand. *)
let precise_does_not_prevent_it () =
  with_in_scope_devices (fun device ->
      let want = Array.map ref_discipline finite_bits in
      let fused = Array.map ref_fused finite_bits in
      let got = run device ~source:src_precise ~inputs:finite_bits in
      let n = Array.length finite_bits in
      let diffs, _ = compare_bits got want in
      let vs_fused, _ = compare_bits got fused in
      Printf.printf
        "    [%s] precise: %d/%d differ from the discipline, %d/%d from the \
         single-rounding model\n\
         %!"
        (describe device)
        diffs
        n
        vs_fused
        n ;
      if diffs = 0 then
        Alcotest.failf
          "`precise` now DOES prevent the f16 narrowing fusion on %s: the \
           precise variant agrees with Sarek's discipline on all %d inputs, \
           where it disagreed on 2912 when this was measured.\n\n\
           That would be a real change and a genuinely good one — \
           Sarek_ir_glsl already emits `precise` on every float local, so it \
           would mean the codegen's existing defence now covers f16 too. It is \
           NOT licence to enable f16 on its own: re-measure the two-narrowing \
           shape as well (it failed worse, 4776/63488 even with `precise`) \
           before touching the refusal. Update docs/fp-contraction-policy.md \
           either way."
          (describe device)
          n ;
      if vs_fused <> 0 then
        Alcotest.failf
          "MECHANISM CHANGED on %s. The precise variant still disagrees with \
           Sarek's discipline (%d/%d), so the refusal stands — but it no \
           longer matches the single-rounding model either (%d/%d), which it \
           did exactly when this was measured.\n\n\
           This is not a regression and not a reason to weaken the assertion. \
           It means RADV is now getting the answer wrong in a way this file \
           does not describe. Re-run under RADV_DEBUG=asm, work out the new \
           combine, and update both this model and \
           docs/fp-contraction-policy.md."
          (describe device)
          diffs
          n
          vs_fused
          n)

let () =
  Alcotest.run
    "Vulkan_f16_tripwire"
    [
      ( "calibration",
        [
          Alcotest.test_case
            "host binary16 rounding round-trips on the whole domain"
            `Quick
            host_rounding_round_trips;
          Alcotest.test_case
            "host models reproduce the independently measured 620"
            `Quick
            host_models_reproduce_the_620;
          Alcotest.test_case
            "the barriered kernel agrees with the discipline (green control)"
            `Quick
            barrier_variant_matches_the_discipline;
        ] );
      ( "refusal_still_warranted",
        [
          Alcotest.test_case
            "the f32 multiply is still absorbed into the f16 narrowing"
            `Quick
            refusal_is_still_warranted;
          Alcotest.test_case
            "`precise` does not prevent it"
            `Quick
            precise_does_not_prevent_it;
        ] );
    ]
