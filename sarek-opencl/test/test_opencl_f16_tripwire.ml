(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #57 slice 2a — OpenCL f16 refusal TRIPWIRE.
 *
 * WHAT THIS TEST IS FOR, because it is not what it looks like.
 *
 * [Sarek_ir_opencl] refuses float16. That refusal is not "unimplemented" — the
 * codegen is a two-line change. It is a REFUSAL BY MEASUREMENT: on
 * rusticl/radeonsi the ACO backend fuses the f32 multiply into the f32->f16
 * narrowing that consumes it, rounding once where Sarek's f16 discipline
 * mandates twice, so 620 of the 63488 finite binary16 inputs disagree with the
 * interpreter. See docs/fp-contraction-policy.md, row
 * "OpenCL / rusticl (f16 narrowing)".
 *
 * A refusal backed by a measurement has a failure mode a normal test does not:
 * it can quietly outlive its own justification. If Mesa stops fusing, the
 * refusal becomes wrong, and nothing would tell us — the codegen would keep
 * raising, the golden test would keep passing, and the documentation would
 * keep citing a defect that no longer exists.
 *
 * So this test asserts the REASON, not the behaviour. It does not check that
 * Sarek refuses f16 (test_cuda_f16_golden already pins that). It checks that
 * refusing is still WARRANTED. It goes red when the fusion STOPS.
 *
 * Its relationship to the C probe is deliberate and neither replaces the other:
 *
 *   tools/probes/opencl_f16_contraction_probe.c
 *       proves the defect EXISTS, and must be able to indict the OpenCL stack
 *       without Sarek's codegen in the loop — since its conclusion is that
 *       Sarek's codegen should keep refusing. It is a documented reproducer,
 *       run by hand, with many variants.
 *
 *   this file
 *       proves the REASON FOR REFUSING STILL HOLDS, on every CI run that has a
 *       device. One variant, no host float16 arithmetic, fails loudly when the
 *       premise expires.
 *
 * HOW IT TRIED TO AVOID A HOST BINARY16 REFERENCE, AND WHY IT NO LONGER DOES.
 *
 * Both kernels below compute the same value. They differ only in whether the
 * f32 intermediate is forced through a `volatile __local` round-trip — measured
 * (slice 2a) to be one of only two barriers that defeat this fusion, and the
 * one that costs no global traffic. So:
 *
 *   plain <> barriered on some input  =>  the fusion is still happening
 *   plain  = barriered on every input =>  the fusion is gone
 *
 * The barriered kernel is therefore simultaneously the control and the oracle,
 * and no OCaml round-to-binary16 is needed anywhere. Decoding binary16 bits to
 * a float IS needed, but decoding is exact and rounding-free (see [f16_decode]).
 *
 * THAT INFERENCE IS SOUND ONLY WHERE THE BARRIER IS KNOWN TO WORK, and the
 * barrier is known to work on exactly one compiler. `plain <> barriered` really
 * says "one of these two kernels is wrong"; reading it as "plain fused" is a
 * step that borrows its warrant from the ACO measurement. The first non-ACO GPU
 * this project ever ran on showed the borrowing was not free: on Intel Arc
 * Graphics (Meteor Lake-P) the barriered kernel is the wrong one (backlog #123,
 * docs/fp-contraction-policy.md §11).
 *
 * So this file now also carries a HOST reference — [ref_discipline], ported
 * from the Vulkan tripwire, which needed one from the start because on RADV no
 * affordable barrier works at all. A host reference is an oracle on every
 * implementation rather than on one, and it is the same discipline the
 * interpreter implements. It is used:
 *
 *   - in [sanity_barriered_computes_the_right_thing], to check over the whole
 *     domain that the barrier really is defeating the fusion on the in-scope
 *     device, instead of trusting a single hand-computed point (Intel passes
 *     that point while being wrong on 4774 inputs);
 *   - in [non_aco_implementations_do_not_fuse], as the thing the plain kernel
 *     is compared against, so the locus cross-check cannot blame the plain
 *     kernel for the barrier's own defects.
 *
 * A REJECTED ALTERNATIVE, recorded because it looks obviously right and is not.
 * Splitting the expression across two kernel launches, with the binary16
 * intermediate materialised in a __global buffer, seems to give an oracle by
 * construction: no compiler can fuse across a dispatch. It does not. The
 * fusion is multiply-into-narrowing, and BOTH of those live in the first
 * kernel, so the dispatch boundary separates the wrong pair. Measured: the
 * two-pass construction reproduces ACO's fused answer exactly, 620/63488 away
 * from the discipline on RX 7900 XTX. A construction argument is not a
 * measurement, and this one was wrong.
 *
 * The in-scope tripwire itself still compares plain against barriered, which is
 * correct there and keeps the assertion pointed at the ACO behaviour it was
 * written for — now with the barrier's validity checked rather than assumed.
 ******************************************************************************)

open Sarek_opencl
module Backend = Opencl_plugin_base.Opencl

let n_local = 256

(* The narrowing shape that fuses. Buffers are raw `ushort` binary16 bit
   patterns rather than `half`, so the host side never needs an f16 Bigarray;
   the as_half/as_ushort bitcasts at the buffer boundary were measured NOT to
   affect the fusion (the "bitcast" probe variant still reports 620/63488), so
   they are safe framing and not an accidental barrier.

   Mirrors sarek-hip/test/test_hip_f16.ml's `f16_midround`: narrow mid-
   expression, widen, then keep computing in f32. That is the shape where an
   elided rounding survives into the result. *)
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

(* Identical, except each f32 intermediate is forced through volatile __local.
   Measured 0/63488 against the interpreter, i.e. this one obeys the DSL's
   double-rounding rule. Not shippable as codegen — it costs LDS traffic per
   narrowing and needs a workgroup-sized allocation the backend does not
   control — but perfectly good as a test oracle. *)
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

(* Exact value of a binary16 bit pattern; None for NaN/Inf. Decoding needs no
   rounding, so this is exact and cannot itself be the source of a mismatch. *)
let f16_decode b =
  let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
  let e = (b lsr 10) land 0x1f and m = b land 0x3ff in
  if e = 31 then None
  else if e = 0 then Some (sign *. float_of_int m *. ldexp 1.0 (-24))
  else Some (sign *. float_of_int (1024 + m) *. ldexp 1.0 (e - 25))

(* All finite binary16 bit patterns. 63488 of them — small enough that the
   exhaustive statement is affordable, which is the whole reason the f16 gates
   in this project are exhaustive rather than sampled. *)
let finite_bits =
  let acc = ref [] in
  for b = 0xFFFF downto 0 do
    if f16_decode b <> None then acc := b :: !acc
  done ;
  Array.of_list !acc

(* ---------------------------------------------------------------------- *)
(* HOST REFERENCE                                                           *)
(*                                                                          *)
(* Ported verbatim from sarek-vulkan/test/test_vulkan_f16_tripwire.ml, which *)
(* needed a host oracle from the start because on RADV NO affordable barrier *)
(* works, so it had no barriered kernel to lean on. The duplication follows  *)
(* the existing precedent in these two files (both already carry their own   *)
(* [f16_decode] and [finite_bits]); they live in different libraries and a   *)
(* shared test lib for sixty lines is not worth the build-graph edge.        *)
(*                                                                          *)
(* Why it is here now (backlog #123). [non_aco_implementations_do_not_fuse]  *)
(* used the barriered kernel as its oracle on devices where the barrier had  *)
(* never been checked. That is only sound on ACO. A host reference is sound  *)
(* everywhere, and it is the same oracle the interpreter implements.        *)
(* ---------------------------------------------------------------------- *)

(* Round a binary64 to binary32. Every intermediate below is exactly
   representable in binary64 before this is applied (a binary16 operand times a
   binary32 constant needs at most 35 significand bits), so this is a single
   correct rounding and not a double rounding. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

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
         [k - 1]. *)
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

let c11 = f32 1.1

(* The kernel shape above, f16(f16(x*1.1) + 1000), under Sarek's discipline:
   the multiply rounds to binary32, then to binary16, then the add rounds to
   binary32 and the result to binary16. Four roundings, all mandated. *)
let ref_discipline b =
  let m = f16_bits (f32 (dec b *. c11)) in
  f16_bits (f32 (dec m +. 1000.0))

(* The same shape with the multiply absorbed into the narrowing that consumes
   it: the first binary32 rounding is skipped. This is what ACO produces, and
   it is the model whose separation from [ref_discipline] is the known 620. *)
let ref_fused b =
  let m = f16_bits (dec b *. c11) in
  f16_bits (f32 (dec m +. 1000.0))

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

let count_diffs a b =
  let d = ref 0 and first = ref None in
  Array.iteri
    (fun i x ->
      if x <> b.(i) then begin
        incr d ;
        if !first = None then first := Some i
      end)
    a ;
  (!d, !first)

(* ---------------------------------------------------------------------- *)
(* SCOPE                                                                    *)
(*                                                                          *)
(* The refusal this test defends is about ONE compiler: the ACO shader      *)
(* backend in Mesa, reached through Mesa's OpenCL front end (rusticl). It   *)
(* is NOT a claim about OpenCL in general, and asserting it against         *)
(* whatever device happens to be first is how a gate fires for the wrong    *)
(* reason. CI proved that concretely: on an AMD EPYC 7763 runner under pocl *)
(* the naive and barriered kernels agree on all 63488 inputs — correctly,   *)
(* because pocl on x86 is a different compiler on a different target — and  *)
(* the tripwire reported the refusal obsolete. A false positive on a        *)
(* CI-blocking gate is the dangerous polarity: it pressures the next person *)
(* to delete an assertion that explicitly tells them not to.                *)
(*                                                                          *)
(* CHOICE OF KEY, and how it fails.                                         *)
(*                                                                          *)
(* Keyed on the COMPILER's own name: the device string must contain "ACO".  *)
(* Mesa reports its shader compiler in the OpenCL device name, e.g.         *)
(*   "AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, DRM 3.64, ...)".       *)
(* "ACO" names the component that performs the fusion, which is exactly the *)
(* thing the refusal is about — so this is driver identity, not a device    *)
(* model. A model substring like "navi31" would be the wrong key and is     *)
(* deliberately not used.                                                   *)
(*                                                                          *)
(* Why not ALSO require the platform to be "rusticl", which would state the *)
(* measured configuration more exactly: Opencl_plugin_base.Opencl is sealed *)
(* with Framework_sig.PLUGIN_BASE, so Device.t is abstract and the platform *)
(* field is unreachable from here. Reaching it would mean re-implementing   *)
(* context/queue/program/buffer handling against the raw Opencl_api just to *)
(* read one string. Not worth it, because the conjunct turns out to be      *)
(* redundant: the reason for wanting it was to exclude rusticl-on-llvmpipe  *)
(* (a CPU target that does not go through ACO), and llvmpipe devices report *)
(* "llvmpipe (LLVM ...)" — no "ACO" — so they are already out of scope.     *)
(*                                                                          *)
(* Deliberately NOT keyed on a device-model substring ("navi31") or on a    *)
(* CPU/GPU distinction. The second local device is named "AMD Ryzen 9 7950X *)
(* 16-Core Processor (radeonsi, raphael_mendocino, ACO, ...)" — it is an    *)
(* integrated GPU whose NAME reads like a CPU, and it reproduces the defect *)
(* identically. Any heuristic keyed on the model name, or on "looks like a  *)
(* CPU", would silently drop it from scope. Driver identity gets it right.  *)
(*                                                                          *)
(* IF THE VENDOR RENAMES: this test SKIPS, it does not silently pass. Mesa  *)
(* renaming rusticl, or switching radeonsi's compiler away from ACO, makes  *)
(* [in_scope] empty, which produces a visible skip naming what it looked    *)
(* for. That is the intended failure direction — an unmeasured platform     *)
(* must never be read as evidence that the refusal still holds — but it     *)
(* does mean a rename converts this gate into a no-op until someone updates *)
(* the key. The skip message says so, so the silence is at least loud.      *)
(* ---------------------------------------------------------------------- *)

let contains ~needle haystack =
  let n = String.length needle and h = String.length haystack in
  let lower = String.lowercase_ascii in
  let needle = lower needle and haystack = lower haystack in
  let rec go i =
    i + n <= h && (String.sub haystack i n = needle || go (i + 1))
  in
  n = 0 || go 0

let describe device = Backend.Device.name device

let is_in_scope device = contains ~needle:"aco" (describe device)

let all_devices () =
  if not (Backend.is_available ()) then [||]
  else begin
    Backend.Device.init () ;
    Array.init (Backend.Device.count ()) Backend.Device.get
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
        | [] -> "no OpenCL devices at all"
        | l -> String.concat "; " (List.map describe l)
      in
      Printf.printf
        "[SKIP] No in-scope OpenCL device. This tripwire defends a refusal \
         measured on Mesa's ACO backend, so it only asserts on a device whose \
         DEVICE name contains \"ACO\" — the Mesa shader compiler that performs \
         the fusion (e.g. \"AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, \
         ...)\"). Saw: %s. Not asserting: a different OpenCL implementation \
         agreeing here would say nothing about whether ACO still fuses.\n\
         %!"
        seen ;
      Alcotest.skip ()
  | ds -> List.iter f ds

(* Guards the "compare two kernels" trick: if both kernels were somehow the same
   or both broken, they would agree and the tripwire would wrongly announce that
   the fusion is gone. Pin the barriered kernel against a hand-computed value.

   For x = 1.0: 1.0 *. 1.1f narrowed to binary16 is 1.099609375; widened and
   added to 1000.0 gives 1001.099609375; narrowed again gives 1001.0, since the
   binary16 spacing at 1001 is 0.5 and 1001.0996 is nearer 1001.0 than 1001.5. *)
let sanity_barriered_computes_the_right_thing () =
  with_in_scope_devices (fun device ->
      let one =
        0x3C00
        (* binary16 1.0 *)
      in
      let inputs = Array.make n_local one in
      let got = run_kernel device ~source:src_barriered ~inputs in
      (match f16_decode got.(0) with
      | None ->
          Alcotest.failf
            "barriered kernel returned a non-finite binary16 (bits 0x%04X) for \
             input 1.0 — the tripwire's oracle is not computing the intended \
             expression, so its agreement/disagreement verdict means nothing"
            got.(0)
      | Some v ->
          Alcotest.(check (float 0.001))
            "barriered midround(1.0) = 1001.0 (oracle sanity)"
            1001.0
            v) ;
      (* The one-point pin above is not enough on its own, and Intel hardware
         proved it: on Intel Arc the barriered kernel drops the intermediate
         narrowing on 4774 inputs, yet still returns 1001.0 for x=1.0, because
         at that operand the dropped rounding does not change the final result.
         A pin that a broken oracle passes is not a check. So sweep the whole
         domain against the host reference. *)
      let inputs = finite_bits in
      let barriered = run_kernel device ~source:src_barriered ~inputs in
      let want = Array.map ref_discipline inputs in
      let diffs, first = count_diffs barriered want in
      if diffs <> 0 then begin
        let d v = match f16_decode v with Some x -> x | None -> nan in
        let i = match first with Some i -> i | None -> 0 in
        Alcotest.failf
          "THE TRIPWIRE'S ORACLE IS NOT VALID ON THIS DEVICE.\n\n\
           On %s the `volatile __local` barriered kernel disagrees with \
           Sarek's f16 discipline on %d of %d finite binary16 inputs; first at \
           x=%.9g (barriered %.9g, discipline %.9g).\n\n\
           The barrier is only known to defeat the fusion on Mesa's ACO \
           backend. Where it does not, [refusal_is_still_warranted]'s \
           plain-vs-barriered comparison is measuring the barrier, not the \
           fusion, and its verdict means nothing on this device.\n\n\
           Do NOT relax this. Either find a barrier that works for this \
           implementation, or re-express the tripwire against [ref_discipline] \
           here too."
          (Backend.Device.name device)
          diffs
          (Array.length inputs)
          (d inputs.(i))
          (d barriered.(i))
          (d want.(i))
      end)

(* CALIBRATION. Host-only, so it runs everywhere including GPU-less CI. *)

let host_rounding_round_trips () =
  Array.iter
    (fun b ->
      if f16_bits (dec b) <> b then
        Alcotest.failf
          "host binary16 rounding is wrong: re-encoding the exact value of \
           0x%04X gives 0x%04X. The locus cross-check below is measured \
           against this function, so nothing there means anything until it is \
           fixed."
          b
          (f16_bits (dec b)))
    finite_bits ;
  Alcotest.(check int)
    "the exhaustive finite binary16 domain"
    63488
    (Array.length finite_bits)

let host_models_reproduce_the_620 () =
  let disc = Array.map ref_discipline finite_bits in
  let fused = Array.map ref_fused finite_bits in
  let diffs, _ = count_diffs disc fused in
  if diffs <> 620 then
    Alcotest.failf
      "CALIBRATION FAILED — do not read any other result in this file.\n\n\
       On the kernel shape f16(f16(x*1.1) + 1000), this file's own host models \
       separate the two-roundings discipline from the fused-first-narrowing \
       behaviour on %d of %d inputs. It must be 620: that is the figure \
       measured independently on hiprtc/gfx1100 \
       (sarek-hip/test/test_hip_f16.ml) and on rusticl/radeonsi \
       (docs/fp-contraction-policy.md), and reproduced on Intel Arc by the \
       `fusedctl` variant of tools/probes/opencl_f16_contraction_probe.c.\n\n\
       Reproducing a known positive is what licenses believing the null that \
       the locus cross-check reports on a non-fusing device. If this number \
       moved, [f16_bits], [f32] or [dec] is wrong — fix the host reference, do \
       not adjust this expectation."
      diffs
      (Array.length finite_bits)

let refusal_is_still_warranted () =
  with_in_scope_devices (fun device ->
      let inputs = finite_bits in
      let n = Array.length inputs in
      let plain = run_kernel device ~source:src_plain ~inputs in
      let barriered = run_kernel device ~source:src_barriered ~inputs in
      let diffs = ref 0 and first = ref None in
      Array.iteri
        (fun i p ->
          if p <> barriered.(i) then begin
            incr diffs ;
            if !first = None then first := Some i
          end)
        plain ;
      let device_name = Backend.Device.name device in
      (match !first with
      | Some i ->
          let d v = match f16_decode v with Some x -> x | None -> nan in
          Printf.printf
            "    [%s] fusion still present: %d/%d differ; first at x=%.9g \
             (fused %.9g, barriered %.9g)\n\
             %!"
            device_name
            !diffs
            n
            (d inputs.(i))
            (d plain.(i))
            (d barriered.(i))
      | None -> ()) ;
      if !diffs = 0 then
        Alcotest.failf
          "OBSOLETE REFUSAL, NOT A REGRESSION — READ BEFORE \"FIXING\".\n\n\
           On %s, the naive f32->f16 narrowing now agrees with the barriered \
           one on all %d finite binary16 inputs. The multiply is no longer \
           being fused into the narrowing.\n\n\
           This test exists to detect exactly that. Sarek_ir_opencl refuses \
           float16 *because* of that fusion (620/63488 disagreements when this \
           was measured, 2026-07-26, rusticl/radeonsi). If the fusion is gone, \
           the refusal has lost its justification and #57 slice 2a should be \
           REVISITED: OpenCL f16 codegen is a small change (\"half\" type \
           string, a narrowing arm, and a cl_khr_fp16 pragma).\n\n\
           Do NOT make this pass by deleting or weakening the assertion. The \
           correct responses are, in order: (1) re-run \
           tools/probes/opencl_f16_contraction_probe.c to confirm \
           independently of Sarek, (2) record the new measurement in \
           docs/fp-contraction-policy.md, (3) enable f16 in Sarek_ir_opencl \
           behind the usual exhaustive interpreter-agreement gate, and (4) \
           delete this test as part of THAT change, not before it."
          device_name
          n ;
      (* Not asserted as an exact count: it is a Mesa-version-dependent number.
         Asserted as a range, because "everything differs" would mean the two
         kernels are computing different expressions, not that one is fused. *)
      if !diffs = n then
        Alcotest.failf
          "all %d inputs differ between the plain and barriered kernels — that \
           is not a fusion signature, it means the two kernels are no longer \
           computing the same expression. Fix the tripwire, do not read this \
           as evidence about Mesa."
          n)

(* The complement of the tripwire, and a real assertion rather than a printout.

   Recording the CI observation was the immediate motive: on an AMD EPYC 7763
   runner under pocl, the naive and barriered kernels agree on all 63488 inputs.
   That is worth keeping, because it is what separates *AMD's GPU compilers*
   from *OpenCL in general* as the locus of the defect.

   It does NOT separate the two AMD compilers from each other, and an earlier
   version of this comment said it did. rusticl/radeonsi compiles through ACO
   and hiprtc compiles through LLVM's AMDGPU backend — two different compilers
   that produce the same 620/63488, which is two compilers agreeing rather than
   one bug seen twice. See docs/fp-contraction-policy.md §2, "Two AMD
   compilers".

   docs/fp-contraction-policy.md states that scoping as a claim. A claim
   deserves a guard, so this asserts it rather than merely printing it: an
   out-of-scope OpenCL implementation that DOES fuse is something we want to be
   told about, whether it widens the claim (a non-AMD fuser) or merely widens
   this test's predicate (an AMD stack outside the "ACO" key). On every
   implementation measured so far it passes trivially.

   The tradeoff is deliberate and stated: this can go red on hardware nobody has
   studied, and that red would be a genuine finding about the doc rather than a
   flake. It must not be silenced by narrowing the predicate — the message says
   what to do instead. Out-of-scope devices that cannot build or run the kernel
   at all (no cl_khr_fp16, say) are reported and skipped over, since a device
   that cannot express the computation says nothing about fusion either way.

   WHAT THIS COMPARES, AND WHY IT CHANGED (backlog #123). It used to compare the
   plain kernel against the BARRIERED kernel, reading any difference as "the
   plain kernel fused". That is a comparison with no oracle: the two kernels
   compute the same expression, so a disagreement proves one of them is wrong
   and says nothing about which. On ACO the barriered kernel is the right one,
   which is where the reading came from; the first non-ACO GPU this project ever
   ran on falsified it. On Intel Arc Graphics (Meteor Lake-P) under the Intel
   Compute Runtime the plain kernel is correct on all 63488 inputs and the
   BARRIERED kernel is wrong on 4774 — so the old comparison went red and
   announced, in its own failure text, that Intel "fuses too" and that the
   documented locus-is-ACO scoping was falsified. Both statements were the exact
   opposite of what the hardware does.

   So it now compares the plain kernel against [run_oracle], which is correct by
   construction rather than by measurement on one vendor's compiler. A red here
   means the plain kernel really does skip a mandated rounding. *)
let non_aco_implementations_do_not_fuse () =
  let devices = all_devices () in
  let out_of_scope =
    Array.to_list devices |> List.filter (fun d -> not (is_in_scope d))
  in
  if out_of_scope = [] then begin
    Printf.printf
      "[SKIP] no out-of-scope OpenCL device present to cross-check the locus \
       claim against\n\
       %!" ;
    Alcotest.skip ()
  end
  else
    List.iter
      (fun device ->
        let name = describe device in
        match
          try
            let inputs = finite_bits in
            let plain = run_kernel device ~source:src_plain ~inputs in
            let barriered = run_kernel device ~source:src_barriered ~inputs in
            let want = Array.map ref_discipline inputs in
            let fused, first = count_diffs plain want in
            let barrier_harm, _ = count_diffs barriered want in
            Ok (fused, first, barrier_harm, inputs, plain, want)
          with e -> Error (Printexc.to_string e)
        with
        | Error msg ->
            Printf.printf
              "    %s: cannot run the f16 comparison (%s) — no evidence either \
               way, not counted\n\
               %!"
              name
              msg
        | Ok (fused, first, barrier_harm, inputs, plain, want) ->
            let n = Array.length inputs in
            (* The barrier count is reported, never asserted. It is not what
               this check is about, and on Intel it is nonzero as shipped: the
               ACO barrier is measured HARMFUL there. Asserting it would pin a
               permanent red with no action available. The place it must not go
               unnoticed is where it is load-bearing, and that IS asserted —
               see [sanity_barriered_computes_the_right_thing]. *)
            Printf.printf
              "    %s: plain vs discipline %d/%d differ (barriered vs \
               discipline %d/%d, reported only)\n\
               %!"
              name
              fused
              n
              barrier_harm
              n ;
            if fused <> 0 then begin
              let d v = match f16_decode v with Some x -> x | None -> nan in
              let i = match first with Some i -> i | None -> 0 in
              Alcotest.failf
                "AN OUT-OF-SCOPE OPENCL IMPLEMENTATION FUSES.\n\n\
                 %s is outside this test's \"ACO\" device-string scope, yet \
                 the naive narrowing disagrees with Sarek's f16 discipline on \
                 %d of %d finite binary16 inputs; first at x=%.9g (naive %.9g, \
                 discipline %.9g). The discipline is the host reference, \
                 calibrated on the same run by \
                 [host_models_reproduce_the_620], so this device really is \
                 skipping a mandated rounding.\n\n\
                 READ THE DEVICE BEFORE READING THE CLAIM. \
                 docs/fp-contraction-policy.md §2 (\"Two AMD compilers\") \
                 attributes this defect to BOTH of AMD's GPU compilers — ACO \
                 and LLVM's AMDGPU backend — not to ACO alone. The \"ACO\" key \
                 above selects Mesa stacks, so an AMD GPU reached through a \
                 non-Mesa OpenCL (ROCm's, which compiles through LLVM/AMDGPU \
                 like HIP) lands here while being an EXPECTED fuser: that is \
                 not a falsification, it is the documented behaviour arriving \
                 through an unkeyed door, and the fix is to widen the scope \
                 predicate. A NON-AMD implementation fusing is the real \
                 falsification, and it would mean the locus is not the vendor \
                 toolchain at all.\n\n\
                 Do NOT fix this by excluding the device from the predicate. \
                 Widen the claim: re-measure with \
                 tools/probes/opencl_f16_contraction_probe.c on this platform \
                 (its `fusedctl` variant calibrates the sweep against the \
                 known 620/63488), correct the OpenCL rows in \
                 docs/fp-contraction-policy.md, and reconsider whether the \
                 refusal in Sarek_ir_opencl should be stated \
                 per-implementation rather than per-backend-compiler."
                name
                fused
                n
                (d inputs.(i))
                (d plain.(i))
                (d want.(i))
            end)
      out_of_scope

let () =
  Alcotest.run
    "Opencl_f16_tripwire"
    [
      ( "calibration",
        [
          Alcotest.test_case
            "host binary16 rounding round-trips every finite input"
            `Quick
            host_rounding_round_trips;
          Alcotest.test_case
            "host models reproduce the known 620"
            `Quick
            host_models_reproduce_the_620;
        ] );
      ( "refusal_still_warranted",
        [
          Alcotest.test_case
            "barriered kernel computes the intended expression"
            `Quick
            sanity_barriered_computes_the_right_thing;
          Alcotest.test_case
            "the f32 multiply is still fused into the f16 narrowing"
            `Quick
            refusal_is_still_warranted;
          Alcotest.test_case
            "out-of-scope OpenCL implementations do not fuse (locus check)"
            `Quick
            non_aco_implementations_do_not_fuse;
        ] );
    ]
