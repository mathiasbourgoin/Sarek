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

(******************************************************************************
 * backlog-151 EXTENSION — `--catalogue`, the other 18 shapes.
 *
 * Slice 1 left one candidate GENERATIVE RULE
 * (docs/fp-contraction-policy.md §12.4) marked "unverified as a general rule",
 * and named the remaining 18 of the 20 emittable f16 shapes as what would
 * settle it. `--catalogue` sweeps all 20 — the two already measured included,
 * as the regression anchor — against the five POLICIES of
 * [F16_shape_catalogue], which are slice 1's four named models restated as
 * functions of an arbitrary expression tree.
 *
 * Two hazards this mode is built against, both of which slice 1 hit:
 *
 *  - A COUNT-ONLY sweep would miss ACO re-absorbing a control 18 times over,
 *    exactly as it missed it once on the two-narrowing shape. Every variant
 *    here is classified ELEMENT-WISE and the harness prints WHICH MODEL
 *    matched, never a bare count.
 *  - A shape on which all five policies coincide reports "S_strict, 0/63488"
 *    while measuring nothing. The host-only separation pass runs first and any
 *    such shape is labelled NON-DISCRIMINATING in its own row, so that a table
 *    of zeros cannot be read as twenty confirmations.
 ******************************************************************************)

open Sarek_vulkan
module Device = Vulkan_api_device
module Memory = Vulkan_api_memory
module Kernel = Vulkan_api_kernel
module M = F16_model_set
module C = F16_shape_catalogue

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

(* ---------------------------------------------------------------------- *)
(* backlog-151 — the 20-shape catalogue                                     *)
(* ---------------------------------------------------------------------- *)

(* What the generative rule of §12.4 PREDICTS, per variant. Written down before
   any device is read, so the sweep is a test of a prediction and not a
   description of an outcome. *)
let predicted ~precise =
  if precise then C.rule_precise.C.pname else C.rule_plain.C.pname

(* What the CORRECTED local rule predicts. Stated here, next to the rule it
   replaces, so both predictions are written down before any device is read. *)
let predicted_local ~precise =
  if precise then "R_local_absorb_nocontract" else "R_local_absorb"

type row = {
  r_id : string;
  r_distinct : int;
  r_plain : string;
  r_precise : string;
  r_unexpl : int;
  r_green : string;
  r_rule : string;
  r_local : string;
}

(* §1.2 requires the device result to be bit-identical to ONE member of the set
   on every input. That is strictly stronger than "every input matches some
   member", and the difference is not academic: shape B4 plain matches
   S_absorb_all on 63480 inputs and S_f32_mul_then_absorb_add on 63486, so no
   single member describes it while every individual input is covered. The two
   outcomes are therefore named differently. *)
let name_of c =
  match c.M.exact_matches with
  | [] when c.M.unexplained = 0 -> "NO SINGLE MODEL (mixture)"
  | [] -> Printf.sprintf "NO MODEL (%d unmatched)" c.M.unexplained
  | [n] -> n
  | l -> String.concat " = " l

let catalogue_device device =
  Printf.printf
    "\n================================================================\n" ;
  Printf.printf "backlog-151 CATALOGUE — device: %s\n" (describe device) ;
  Printf.printf
    "================================================================\n%!" ;

  (* REPORTING CONTROL, host-side. The trap slice 1 hit was a control ACO
     re-absorbed, caught only because the harness reported the WRONG MODEL
     rather than an implausible count. So before any device number is read,
     feed the classifier a host-computed NON-STRICT model and require it to
     name that model. A classifier that answers "S_strict" to everything would
     report twenty false confirmations here. *)
  Printf.printf "\n  --- reporting control (host-injected deviation) ---\n" ;
  let control_ok = ref true in
  List.iter
    (fun sh ->
      if C.distinct_model_count sh > 1 then begin
        let models = C.models_of sh in
        let injected =
          Array.map (C.result C.rule_plain sh.C.expr) M.finite_bits
        in
        let c = M.classify models injected in
        let got = name_of c in
        let ok =
          List.mem C.rule_plain.C.pname c.M.exact_matches && c.M.unexplained = 0
        in
        if not ok then control_ok := false ;
        Printf.printf
          "    %-4s injected %-34s -> reported %s%s\n"
          sh.C.id
          C.rule_plain.C.pname
          got
          (if ok then "" else "   *** REPORTING CONTROL BROKEN ***")
      end)
    C.shapes ;
  if not !control_ok then
    Printf.printf
      "    *** the classifier does not name an injected deviation; nothing \
       below is readable ***\n" ;

  let rows = ref [] in
  List.iter
    (fun sh ->
      let models = C.models_with_local sh in
      let distinct = C.distinct_model_count sh in
      Printf.printf "\n  --- %s : %s ---\n" sh.C.id sh.C.descr ;
      if sh.C.discriminating_note <> "" then
        Printf.printf "    NOTE: %s\n" sh.C.discriminating_note ;
      if distinct = 1 then
        Printf.printf
          "    NON-DISCRIMINATING: all five policies are the SAME FUNCTION on \
           this shape over all 63488 inputs. Whatever the device returns, this \
           row is not evidence for or against the rule.\n"
      else
        Printf.printf
          "    the five policies induce %d distinct functions here\n"
          distinct ;
      let nin = C.inner_narrowing_count sh.C.expr in
      if nin > 1 then
        Printf.printf
          "    %d intermediate narrowings: §1.3's ceiling is derived for the \
           elision of ONE rounding, so it is evaluated at the INNERMOST \
           narrowing and covers only that elision on this shape\n"
          nin ;
      (* Same sweep, but the device array is kept so an input matching NO model
         can be shown as a bit pattern next to what each model wanted. A bare
         "N inputs match no model" leaves the reader unable to tell a fusion
         hazard from a sign-of-zero difference, and on this catalogue two shapes
         turn out to be the latter. *)
      let sweep_v ~label ~precise ~barrier =
        let source = C.source ~dialect:C.Glsl ~precise ~barrier sh in
        let got = run device ~source ~inputs:M.finite_bits in
        let c = M.classify models got in
        M.print_classification ~label c ;
        if c.M.unexplained > 0 then begin
          let idx = c.M.first_unexplained in
          let b = M.finite_bits.(idx) in
          Printf.printf
            "      first unmatched input x = %.9g (0x%04X): device 0x%04X"
            (M.dec b)
            b
            got.(idx) ;
          List.iter
            (fun m -> Printf.printf ", %s 0x%04X" m.M.name (m.M.result b))
            models ;
          Printf.printf "\n"
        end ;
        c
      in
      let cg =
        sweep_v
          ~label:"GREEN CONTROL — every temporary through the volatile SSBO"
          ~precise:false
          ~barrier:true
      in
      if not (List.mem "S_strict" cg.M.exact_matches) then
        Printf.printf
          "    *** GREEN CONTROL did not reproduce S_strict: the two rows \
           below are not attributable to ACO ***\n" ;
      let cp = sweep_v ~label:"plain" ~precise:false ~barrier:false in
      report_ceiling models cp ;
      let cq =
        sweep_v
          ~label:"precise (what Sarek_ir_glsl emits)"
          ~precise:true
          ~barrier:false
      in
      report_ceiling models cq ;
      let verdict ~want ~precise c =
        let want = want ~precise in
        if distinct = 1 then "n/a (non-discriminating)"
        else if List.mem want c.M.exact_matches then "HOLDS"
        else if c.M.unexplained > 0 then
          Printf.sprintf "BROKEN — %d inputs match no model" c.M.unexplained
        else Printf.sprintf "BROKEN — matched %s, predicts %s" (name_of c) want
      in
      let v want =
        Printf.sprintf
          "plain -> %s ; precise -> %s"
          (verdict ~want ~precise:false cp)
          (verdict ~want ~precise:true cq)
      in
      Printf.printf "    §12.4 whole-tree rule: %s\n" (v predicted) ;
      Printf.printf "    corrected LOCAL rule : %s\n" (v predicted_local) ;
      rows :=
        {
          r_id = sh.C.id;
          r_distinct = distinct;
          r_plain = name_of cp;
          r_precise = name_of cq;
          r_unexpl = cp.M.unexplained + cq.M.unexplained;
          r_green = name_of cg;
          r_rule = v predicted;
          r_local = v predicted_local;
        }
        :: !rows)
    C.shapes ;

  Printf.printf
    "\n\n  SUMMARY — %s\n  (rule predicts %s for plain and %s for precise)\n\n"
    (describe device)
    C.rule_plain.C.pname
    C.rule_precise.C.pname ;
  List.iter
    (fun r ->
      Printf.printf
        "  %-4s  %d distinct models, %d unmatched inputs\n\
        \        plain   : %s\n\
        \        precise : %s\n\
        \        green   : %s\n\
        \        §12.4 whole-tree rule: %s\n\
        \        corrected LOCAL rule : %s\n"
        r.r_id
        r.r_distinct
        r.r_unexpl
        r.r_plain
        r.r_precise
        r.r_green
        r.r_rule
        r.r_local)
    (List.rev !rows)

let () =
  let host_only = Array.exists (fun a -> a = "--host-only") Sys.argv in
  let catalogue = Array.exists (fun a -> a = "--catalogue") Sys.argv in
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

  if catalogue then begin
    (try C.calibrate ()
     with C.Calibration_failed s ->
       Printf.printf
         "CATALOGUE CALIBRATION FAILED — read nothing below it:\n  %s\n"
         s ;
       exit 1) ;
    Printf.printf
      "catalogue calibration PASSED: the five generic policies reproduce slice \
       1's seven hand-written closed forms bit-for-bit on A2 and B1 over all \
       63488 inputs, and reproduce the recorded separations 2912 / 620 / 5075 \
       / 4776 / 4774 from the generic evaluator.\n\n"
  end ;

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
      let rec flag name i =
        if i + 1 >= Array.length Sys.argv then None
        else if Sys.argv.(i) = name then Some Sys.argv.(i + 1)
        else flag name (i + 1)
      in
      match flag "--shape" 1 with
      | Some id ->
          (* One shape, one variant, one shader compiled — so RADV_DEBUG=asm
             produces an ISA dump attributable to a single shader. With a full
             catalogue run, sixty shaders are compiled in sequence and reading
             the ISA means guessing, which is not a machine-code tier. *)
          let sh = C.shape_by_id id in
          let variant =
            match flag "--variant" 1 with Some v -> v | None -> "plain"
          in
          let precise = variant = "precise" and barrier = variant = "barrier" in
          let source = C.source ~dialect:C.Glsl ~precise ~barrier sh in
          let device = List.hd ds in
          Printf.printf
            "device: %s\nshape: %s (%s)\nvariant: %s\nGLSL:\n%s\n"
            (describe device)
            sh.C.id
            sh.C.descr
            variant
            source ;
          let models = C.models_with_local sh in
          let c = sweep device ~label:variant ~source ~models in
          report_ceiling models c
      | None -> (
          match flag "--variant" 1 with
          | Some v when not catalogue -> probe_one (List.hd ds) v
          | _ ->
              if catalogue then List.iter catalogue_device ds
              else List.iter probe_device ds))
