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
 * HOW IT AVOIDS NEEDING A HOST BINARY16 REFERENCE.
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
 * Self-check: comparing two kernels to each other would report "fusion gone" if
 * BOTH were broken into agreement (say, a build that silently produced two
 * copies of the same source). [sanity_barriered_computes_the_right_thing]
 * closes that by pinning the barriered kernel's output for a known input
 * against a value computed by hand, so agreement can only be reported when the
 * kernels are really computing the intended expression.
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
      match f16_decode got.(0) with
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
            v)

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
   That is worth keeping, because it is what separates *the ACO backend* from
   *OpenCL in general* as the locus of the defect — and it is the second
   independent reason to believe rusticl and HIP are ONE bug seen twice rather
   than two coincidentally equal ones.

   docs/fp-contraction-policy.md now states that scoping as a claim. A claim
   deserves a guard, so this asserts it rather than merely printing it: any
   NON-ACO OpenCL implementation that DOES fuse would falsify "the locus is
   ACO", and we want to be told. On every implementation measured so far it
   passes trivially.

   The tradeoff is deliberate and stated: this can go red on hardware nobody has
   studied, and that red would be a genuine finding about the doc rather than a
   flake. It must not be silenced by narrowing the predicate — the message says
   what to do instead. Out-of-scope devices that cannot build or run the kernel
   at all (no cl_khr_fp16, say) are reported and skipped over, since a device
   that cannot express the computation says nothing about fusion either way. *)
let non_aco_implementations_do_not_fuse () =
  let devices = all_devices () in
  let out_of_scope =
    Array.to_list devices |> List.filter (fun d -> not (is_in_scope d))
  in
  if out_of_scope = [] then begin
    Printf.printf
      "[SKIP] no non-ACO OpenCL device present to cross-check the locus-is-ACO \
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
            let diffs = ref 0 in
            Array.iteri (fun i p -> if p <> barriered.(i) then incr diffs) plain ;
            Ok (!diffs, Array.length inputs)
          with e -> Error (Printexc.to_string e)
        with
        | Error msg ->
            Printf.printf
              "    %s: cannot run the f16 comparison (%s) — no evidence either \
               way, not counted\n\
               %!"
              name
              msg
        | Ok (diffs, n) ->
            Printf.printf "    %s: %d/%d differ\n%!" name diffs n ;
            if diffs <> 0 then
              Alcotest.failf
                "THE LOCUS-IS-ACO CLAIM IS NOW TOO NARROW.\n\n\
                 %s is NOT an ACO device, yet %d of %d finite binary16 inputs \
                 differ between the naive and barriered narrowings — it fuses \
                 too.\n\n\
                 docs/fp-contraction-policy.md currently attributes this \
                 defect to Mesa's ACO backend specifically, and cites the fact \
                 that non-ACO OpenCL does not fuse as evidence that rusticl \
                 and HIP are one bug seen twice. This device falsifies that \
                 scoping.\n\n\
                 Do NOT fix this by excluding the device from the predicate. \
                 Widen the claim: re-measure with \
                 tools/probes/opencl_f16_contraction_probe.c on this platform, \
                 correct the OpenCL rows in docs/fp-contraction-policy.md, and \
                 reconsider whether the refusal in Sarek_ir_opencl should be \
                 stated per-implementation rather than per-backend-compiler."
                name
                diffs
                n)
      out_of_scope

let () =
  Alcotest.run
    "Opencl_f16_tripwire"
    [
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
            "non-ACO OpenCL implementations do not fuse (locus check)"
            `Quick
            non_aco_implementations_do_not_fuse;
        ] );
    ]
