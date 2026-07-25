(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * f16 SASS conformance gate (#107, following the HIP fusion fix in #290).
 *
 * WHAT THIS PROVES. Sarek's f16 surface promises that arithmetic happens in f32
 * and that narrowing to binary16 is a SEPARATE round-to-nearest-even step. On
 * AMDGPU that promise was broken by an ISel combine that fused the f32 multiply
 * into the narrowing ([v_fma_mixlo_f16]) and demoted an f32 add to binary16
 * ([v_add_f16]) — 620 of 63488 binary16 inputs came back wrong on gfx1100, and
 * -ffp-contract could not reach it. This test asks the same question of the
 * CUDA path and answers it from the MACHINE CODE, not from the C or the PTX:
 *
 *   generated CUDA source -> nvrtc -> PTX -> ptxas -> cubin -> nvdisasm -> SASS
 *
 * and then classifies the SASS arithmetic stream as fused or unfused.
 *
 * WHY THIS AND NOT THE nvrtc GATE. test_cuda_f16_nvrtc asserts on the PTX. PTX
 * is not the machine code: the fusion that bit AMDGPU happened strictly BELOW
 * the equivalent level. Only SASS settles it. No NVIDIA device is needed —
 * ptxas and nvdisasm are host tools.
 *
 * WHAT IT DOES NOT PROVE. That the kernel produces correct values on real
 * hardware. It proves the emitted instruction stream keeps the f32 discipline;
 * it says nothing about the hardware's own conversion rounding, and nothing
 * about the driver's JIT ptxas, which is a different build from the offline one
 * this test drives.
 *
 * NON-VACUITY. A positive control (genuine binary16 arithmetic via __hmul /
 * __hadd) must be classified FUSED. Without that, an assertion that "no fused
 * form appears" would pass for a detector that can see nothing at all.
 *
 * Skips cleanly with no CUDA toolkit, no nvrtc, or no CUDA headers; and skips
 * individual architectures the local ptxas does not know. It does NOT skip an
 * architecture whose SASS the classifier cannot read — see [classifier_min_sm]:
 * that is a hard failure, because a classifier applied to an instruction shape
 * it does not model returns a wrong verdict, not no verdict.
 ******************************************************************************)

open Sarek_ir_types

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let mk_kernel name params body =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = [];
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(** The MIDROUND kernel — the exact shape that exposed the AMDGPU fusion:

    [out.(i) <- f16_of_f32 (f32_of_f16 (f16_of_f32 (f32_of_f16 inp.(i) *. 1.1))
     +. 1000.0)]

    Both halves of the defect are reachable from it: the f32 multiply feeding a
    narrowing (which AMDGPU fused into [v_fma_mixlo_f16]) and the f32 add
    downstream of a widening (which AMDGPU demoted to [v_add_f16]). A kernel
    whose narrowing feeds the store directly cannot show either. *)
let f16_midround_kernel () =
  let out = make_var "out" (TVec TFloat16) in
  let inp = make_var "inp" (TVec TFloat16) in
  let idx = make_var "idx" TInt32 in
  let narrowed_product =
    ECast
      ( TFloat16,
        EBinop
          ( Mul,
            ECast (TFloat32, EArrayRead ("inp", EVar idx)),
            EConst (CFloat32 1.1) ) )
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            ECast
              ( TFloat16,
                EBinop
                  ( Add,
                    ECast (TFloat32, narrowed_product),
                    EConst (CFloat32 1000.0) ) ) ) )
  in
  mk_kernel
    "f16_midround"
    [
      DParam (out, Some {arr_elttype = TFloat16; arr_memspace = Global});
      DParam (inp, Some {arr_elttype = TFloat16; arr_memspace = Global});
    ]
    body

(** The POSITIVE CONTROL, written directly in CUDA C because the DSL cannot
    express it: [__hmul] / [__hadd] are native binary16 arithmetic, so ptxas is
    expected to emit a genuine [HFMA2] on the data path. This is what a
    regression would look like, and the classifier must catch it. *)
let positive_control_source =
  {|#include <cuda_fp16.h>
extern "C" {
__global__ void f16_native_arith(__half* __restrict__ out,
                                 __half* __restrict__ inp) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  __half h = inp[idx];
  out[idx] = __hadd(__hmul(h, __float2half(1.1f)), __float2half(1000.0f));
}
}
|}

let gen k =
  Sarek_codegen.Sarek_ir_cuda.current_framework := None ;
  Sarek_codegen.Sarek_ir_cuda.current_variants := [] ;
  Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:k.kern_types k

(* ------------------------------------------------------------------ *)
(* Toolchain discovery                                                *)
(* ------------------------------------------------------------------ *)

let on_path tool =
  match Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" tool) with
  | Unix.WEXITED 0 -> true
  | _ -> false

let ptxas_available = lazy (on_path "ptxas")

let nvdisasm_available = lazy (on_path "nvdisasm")

let nvrtc_ready () =
  Sarek_cuda.Cuda_nvrtc.is_available ()
  && Lazy.force Sarek_cuda.Cuda_nvrtc.cuda_include_paths <> []

let skip_reason () =
  if not (Sarek_cuda.Cuda_nvrtc.is_available ()) then "libnvrtc not loadable"
  else if Lazy.force Sarek_cuda.Cuda_nvrtc.cuda_include_paths = [] then
    "no CUDA include directory containing cuda_fp16.h"
  else if not (Lazy.force ptxas_available) then "ptxas not on PATH"
  else if not (Lazy.force nvdisasm_available) then "nvdisasm not on PATH"
  else "toolchain ready"

let ready () =
  nvrtc_ready () && Lazy.force ptxas_available && Lazy.force nvdisasm_available

(* ------------------------------------------------------------------ *)
(* PTX -> SASS                                                        *)
(* ------------------------------------------------------------------ *)

(* Architectures to check. compute_75 PTX is assembled to each of them: that is
   exactly what a driver does for forward compatibility, and it was verified
   (see docs/optimization/cuda-f16-fusion-sass-audit.md) to give SASS byte-identical
   to assembling arch-native PTX on every target here. Any name the local ptxas
   does not know is skipped rather than failed, so an older toolkit still runs
   the targets it does support.

   This list is bounded below by [classifier_min_sm]: adding a pre-Turing target
   here (sm_61, say) makes the test fail rather than classify a SASS shape the
   classifier was never written for. *)
let architectures =
  ["sm_75"; "sm_80"; "sm_86"; "sm_89"; "sm_90"; "sm_100"; "sm_120"]

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic ;
  s

let write_file path contents =
  let oc = open_out_bin path in
  output_string oc contents ;
  close_out oc

(** Why this test produced no verdict for an architecture. The distinction is
    load-bearing: [Unknown_arch] is an environment gap and a legitimate skip,
    while [Tool_error] means ptxas or nvdisasm rejected work this test generated
    — a real failure that must not be folded into the skip bucket. Before this
    split every nonzero exit read as "arch not supported", so a ptxas that
    rejected the generated PTX outright made the gate report a green pass having
    checked nothing.

    [Unclassifiable_arch] is the third case and belongs to US, not to the
    toolchain: the local ptxas may assemble the target perfectly well while the
    classifier below has no idea what the resulting instruction stream means. It
    is a hard failure for the same reason [Tool_error] is — a classifier that
    guesses is worse than one that declines. *)
type tool_failure =
  | Unknown_arch of string
  | Unclassifiable_arch of string
  | Tool_error of string

let failure_message = function
  | Unknown_arch m | Unclassifiable_arch m | Tool_error m -> m

(* ptxas reports an unrecognised target as
   [ptxas fatal : Value 'sm_XXX' is not defined for option 'gpu-name'],
   an architecture it was not built for as "Unsupported gpu architecture",
   and an unassemblable .target directive as "Unsupported .target". Anything
   else is a genuine assembler error. If a future toolkit invents new
   wording the effect is a hard failure with the message quoted verbatim,
   which is the safe direction: a new phrase gets noticed and added here,
   rather than joining a silent skip bucket. *)
let contains_sub s sub =
  let n = String.length sub and l = String.length s in
  let rec go i = i + n <= l && (String.sub s i n = sub || go (i + 1)) in
  n = 0 || go 0

let classify_ptxas_error msg =
  if
    contains_sub msg "is not defined for option"
    || contains_sub msg "Unsupported gpu architecture"
    || contains_sub msg "Unsupported .target"
  then Unknown_arch msg
  else Tool_error msg

(* nvdisasm's own "I do not know this SM" wording. *)
let classify_nvdisasm_error msg =
  if
    contains_sub msg "Unrecognized SM version"
    || contains_sub msg "Unsupported SM version"
    || contains_sub msg "unsupported architecture"
  then Unknown_arch msg
  else Tool_error msg

(* ------------------------------------------------------------------ *)
(* What the classifier is allowed to look at                          *)
(* ------------------------------------------------------------------ *)

(** The lowest SM the SASS classifier below is written for.

    Every idiom it matches is Turing-and-later shaped: the
    [HADD2 Rd, -RZ, Rs.H0_H0] widening idiom, the
    [HFMA2.MMA ..., -RZ, RZ, imm, imm] immediate materialisation, and the
    [F2F.F16.F32] / [F2FP.PACK_AB] narrowing family. Pascal (sm_5x/sm_6x) does
    not emit that stream — it converts with [F2F.F32.F16] in BOTH directions and
    has no [HADD2]-based widening idiom at all — so on Pascal {!is_widening}
    would recognise nothing, {!narrowings} would count widenings as narrowings,
    and {!binary16_arithmetic} would flag conversions as the defect. That is
    MISCLASSIFICATION, not detection: the gate would report a defect that is not
    there, or (worse, under a different instruction selection) miss one that is.

    So the classifier declines instead. Adding an architecture below this floor
    to {!architectures} makes the test FAIL with {!Unclassifiable_arch} until
    someone teaches the classifier that architecture's idioms and verifies them
    against real SASS for it. *)
let classifier_min_sm = 75

(* [sm_90a] / [sm_100f] and friends: take the leading digit run after "sm_". *)
let sm_version arch =
  let prefix = "sm_" in
  let pl = String.length prefix and al = String.length arch in
  if al <= pl || String.sub arch 0 pl <> prefix then None
  else
    let i = ref pl in
    while !i < al && arch.[!i] >= '0' && arch.[!i] <= '9' do
      incr i
    done ;
    if !i = pl then None else int_of_string_opt (String.sub arch pl (!i - pl))

(** [Ok ()] only for architectures whose SASS the classifier actually models. An
    unparsable name is refused too: guessing at [gfx1100] or [compute_75] here
    would be the same mistake. *)
let classifier_supports arch =
  match sm_version arch with
  | Some v when v >= classifier_min_sm -> Ok ()
  | Some v ->
      Error
        (Unclassifiable_arch
           (Printf.sprintf
              "the SASS classifier in this test is written for sm_%d and later \
               (Turing/Ampere/Hopper/Blackwell instruction shapes: HADD2 \
               widening idiom, F2F/F2FP narrowing family). sm_%d predates that \
               shape, so classifying its disassembly would produce a wrong \
               verdict rather than no verdict. Refusing. Teach the classifier \
               this architecture's idioms (is_widening, narrowings, \
               binary16_arithmetic) and verify them against real sm_%d SASS \
               before adding it to `architectures`."
              classifier_min_sm
              v
              v))
  | None ->
      Error
        (Unclassifiable_arch
           (Printf.sprintf
              "cannot tell which SASS instruction shape %S has; the classifier \
               only models sm_NN targets with NN >= %d. Refusing rather than \
               guessing."
              arch
              classifier_min_sm))

(** [sass_of_ptx ~arch ptx] assembles [ptx] for [arch] and disassembles the
    result. See {!tool_failure} for the three error kinds. The classifier's own
    competence is checked FIRST: an architecture it cannot read is refused
    before any tool runs, so the refusal does not depend on which CUDA release
    happens to be installed. *)
let sass_of_ptx ~arch ptx =
  match classifier_supports arch with
  | Error _ as e -> e
  | Ok () ->
      let base = Filename.temp_file "sarek_f16_sass_" "" in
      let src = base ^ ".ptx" and obj = base ^ ".cubin" in
      let err = base ^ ".err" and out = base ^ ".sass" in
      let cleanup () =
        List.iter
          (fun f -> try Sys.remove f with _ -> ())
          [base; src; obj; err; out]
      in
      Fun.protect ~finally:cleanup (fun () ->
          write_file src ptx ;
          let rc =
            Unix.system
              (Printf.sprintf
                 "ptxas -arch=%s -o %s %s 2>%s"
                 (Filename.quote arch)
                 (Filename.quote obj)
                 (Filename.quote src)
                 (Filename.quote err))
          in
          match rc with
          | Unix.WEXITED 0 -> (
              let rc =
                Unix.system
                  (Printf.sprintf
                     "nvdisasm -c %s >%s 2>%s"
                     (Filename.quote obj)
                     (Filename.quote out)
                     (Filename.quote err))
              in
              match rc with
              | Unix.WEXITED 0 -> Ok (read_file out)
              | _ ->
                  (* ptxas accepted the target but nvdisasm did not. ptxas and
                 nvdisasm are located independently by [on_path], so they can
                 come from different CUDA installs: a newer ptxas assembling
                 sm_100 whose cubin an older nvdisasm cannot read is an
                 environment gap, not a defect. Classify it the same way. *)
                  Error
                    (classify_nvdisasm_error
                       ("nvdisasm failed: " ^ try read_file err with _ -> "")))
          | _ ->
              Error
                (classify_ptxas_error
                   ("ptxas failed: " ^ try read_file err with _ -> "")))

(* ------------------------------------------------------------------ *)
(* SASS classification                                                *)
(* ------------------------------------------------------------------ *)

(** One disassembled instruction: mnemonic (with its dotted modifiers) and the
    operand text. [nvdisasm -c] emits body instructions as
    ["  /*00b0*/  HADD2.F32 R0, -RZ, R2.H0_H0 ;"]. *)
type insn = {mnemonic : string; operands : string}

let instructions sass =
  let addr = Str.regexp "^[ \t]*/\\*[0-9a-f]+\\*/[ \t]+" in
  let split = Str.regexp "[ \t]+" in
  String.split_on_char '\n' sass
  |> List.filter_map (fun line ->
      if not (Str.string_match addr line 0) then None
      else
        let rest = Str.string_after line (Str.match_end ()) in
        (* Drop a predicate guard, then take mnemonic + operands. *)
        let rest =
          if String.length rest > 0 && rest.[0] = '@' then
            match Str.bounded_split split rest 2 with [_; r] -> r | _ -> rest
          else rest
        in
        match Str.bounded_split split rest 2 with
        | [m] -> Some {mnemonic = m; operands = ""}
        | m :: r :: _ ->
            (* strip the trailing ";" and anything after it *)
            let r = try String.sub r 0 (String.index r ';') with _ -> r in
            Some {mnemonic = m; operands = String.trim r}
        | [] -> None)

let starts_with ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

(** [HFMA2.MMA Rd, -RZ, RZ, imm, imm] is ptxas's idiom for materialising a
    32-bit immediate into a register — the halves are the two 16-bit halves of
    the constant and nothing is computed. It appears on sm_80 carrying the
    integer stride 2, entirely off the floating-point data path. Every other
    HFMA2 is real binary16 arithmetic. *)
let is_immediate_materialisation insn =
  starts_with ~prefix:"HFMA2" insn.mnemonic
  && contains ~needle:"-RZ, RZ," insn.operands

(** [HADD2.F32 Rd, -RZ, Rs.H0_H0] is ptxas's idiom for the f16 -> f32 WIDENING
    (add negative zero, produce f32). Widening is exact, so it is conformant. A
    HADD2 with no f32 marker at all is a binary16 add — the [v_add_f16] defect.
    The exemption is deliberately narrow: only HADD2, and only when the f32
    marker is present. HMUL2 and HFMA2 are never exempt, because there is no
    conversion idiom that uses them.

    The marker is looked for in the operands as well as the mnemonic: nvdisasm
    renders it as [HADD2.F32 R0, -RZ, R2.H0_H0] on the register form but as
    [HADD2 R0, -RZ.H0_H0, c[0x2][0x0].F32] when an operand comes from a constant
    bank. Both are the same widening. *)
let is_widening insn =
  starts_with ~prefix:"HADD2" insn.mnemonic
  && (contains ~needle:".F32" insn.mnemonic
     || contains ~needle:".F32" insn.operands)

(** Binary16 arithmetic on the data path: the CUDA analogue of [v_fma_mixlo_f16]
    / [v_add_f16]. Its presence in a kernel the DSL specified as f32-arithmetic
    is the defect. *)
let binary16_arithmetic insns =
  List.filter
    (fun i ->
      (starts_with ~prefix:"HMUL" i.mnemonic
      || starts_with ~prefix:"HFMA" i.mnemonic
      || starts_with ~prefix:"HADD" i.mnemonic
      || starts_with ~prefix:"HSET" i.mnemonic
      || starts_with ~prefix:"HMNMX" i.mnemonic)
      && (not (is_immediate_materialisation i))
      && not (is_widening i))
    insns

(** An f32 -> f16 narrowing. sm_75 emits [F2F.F16.F32]; sm_80/86 emit
    [F2FP.PACK_AB]; sm_89 and up emit [F2FP.F16.F32.PACK_AB]. All are the same
    single-instruction conversion, so match the [F2F] family. *)
let narrowings insns =
  List.filter (fun i -> starts_with ~prefix:"F2F" i.mnemonic) insns

let count pred insns = List.length (List.filter pred insns)

let is_f32_mul i = starts_with ~prefix:"FMUL" i.mnemonic

let is_f32_add i = starts_with ~prefix:"FADD" i.mnemonic

let render insns =
  String.concat
    "\n"
    (List.map (fun i -> Printf.sprintf "    %s %s" i.mnemonic i.operands) insns)

(* ------------------------------------------------------------------ *)
(* Tests                                                              *)
(* ------------------------------------------------------------------ *)

let ptx_of_source ~name src =
  match Sarek_cuda.Cuda_nvrtc.compile_to_ptx ~name ~arch:"compute_75" src with
  | ptx -> Ok ptx
  | exception e -> Error (Printexc.to_string e)

(** THE GATE. For every architecture the local ptxas knows, the midround
    kernel's SASS must show the f32 discipline intact:

    - a genuine f32 multiply (FMUL) — not folded into the narrowing;
    - a genuine f32 add (FADD) — not demoted to a binary16 add;
    - two separate f32 -> f16 narrowings, one per [float16_of_float32];
    - no binary16 arithmetic anywhere on the data path.

    Each bullet is one half of what went wrong on AMDGPU, plus the count that
    catches a narrowing being folded AWAY rather than fused. *)
let test_midround_sass_unfused () =
  if not (ready ()) then begin
    Printf.printf "  [SKIP] f16 SASS gate: %s\n" (skip_reason ()) ;
    Alcotest.skip ()
  end
  else
    let src = gen (f16_midround_kernel ()) in
    match ptx_of_source ~name:"f16_midround" src with
    | Error e -> Alcotest.failf "nvrtc rejected the generated f16 kernel: %s" e
    | Ok ptx ->
        let checked = ref 0 in
        let unusable = ref [] in
        List.iter
          (fun arch ->
            match sass_of_ptx ~arch ptx with
            | Error (Tool_error e) ->
                (* The toolchain knows this target and still refused the work
                   this test generated. A real defect, not an environment
                   gap: do not fold it into the skip bucket. *)
                Alcotest.failf
                  "%s: the local CUDA toolchain rejected the generated kernel: \
                   %s"
                  arch
                  (String.trim e)
            | Error (Unclassifiable_arch e) ->
                (* Not an environment gap: this architecture is in the list
                   above and the classifier cannot read its SASS. Skipping it
                   would let the gate report green having classified a stream
                   it does not understand. *)
                Alcotest.failf
                  "%s: refusing to classify this architecture's SASS: %s"
                  arch
                  (String.trim e)
            | Error (Unknown_arch e) ->
                (* ptxas does not know this architecture. Legitimate per-arch
                   skip; recorded so the zero-architecture case can report
                   WHY nothing was checked instead of passing silently. *)
                unusable := (arch, String.trim e) :: !unusable ;
                Printf.printf "  [SKIP] %s: %s\n" arch (String.trim e)
            | Ok sass ->
                incr checked ;
                let insns = instructions sass in
                if insns = [] then
                  Alcotest.failf
                    "%s: could not parse any instruction out of the disassembly:\n\
                     %s"
                    arch
                    sass ;
                let fused = binary16_arithmetic insns in
                if fused <> [] then
                  Alcotest.failf
                    "%s: binary16 arithmetic on the data path of a kernel the \
                     DSL specified in f32 — this is the CUDA form of the \
                     AMDGPU v_fma_mixlo_f16 / v_add_f16 defect:\n\
                     %s\n\
                     full stream:\n\
                     %s"
                    arch
                    (render fused)
                    (render insns) ;
                if count is_f32_mul insns <> 1 then
                  Alcotest.failf
                    "%s: expected exactly one f32 multiply (FMUL); the DSL \
                     specifies the product in f32 and a fused form would \
                     absorb it:\n\
                     %s"
                    arch
                    (render insns) ;
                if count is_f32_add insns <> 1 then
                  Alcotest.failf
                    "%s: expected exactly one f32 add (FADD); a demotion to \
                     binary16 would remove it:\n\
                     %s"
                    arch
                    (render insns) ;
                let n_narrow = List.length (narrowings insns) in
                if n_narrow <> 2 then
                  Alcotest.failf
                    "%s: expected 2 separate f32->f16 narrowings (one per \
                     float16_of_float32), found %d — a fused form folds one \
                     away:\n\
                     %s"
                    arch
                    n_narrow
                    (render insns) ;
                (* Both widenings must be present too, otherwise the middle
                   round-trip was elided rather than fused. *)
                if count is_widening insns <> 2 then
                  Alcotest.failf
                    "%s: expected 2 f16->f32 widenings (HADD2.F32):\n%s"
                    arch
                    (render insns))
          architectures ;
        if !checked = 0 then begin
          (* Zero architectures checked: the gate asserted nothing. Report a
             SKIP status, never a pass. *)
          Printf.printf
            "  [SKIP] f16 SASS gate: local ptxas knows none of %s\n"
            (String.concat ", " architectures) ;
          List.iter
            (fun (arch, e) -> Printf.printf "    %s: %s\n" arch e)
            (List.rev !unusable) ;
          Alcotest.skip ()
        end
        else
          Printf.printf
            "  f16 SASS gate: f32 discipline intact on %d/%d architectures\n"
            !checked
            (List.length architectures)

(** NON-VACUITY. Genuine binary16 arithmetic must be REPORTED as such. If this
    fails, the gate above proves nothing: it would be asserting the absence of a
    pattern it cannot recognise. *)
let test_classifier_detects_fusion () =
  if not (ready ()) then begin
    Printf.printf "  [SKIP] f16 SASS positive control: %s\n" (skip_reason ()) ;
    Alcotest.skip ()
  end
  else
    match ptx_of_source ~name:"f16_native_arith" positive_control_source with
    | Error e -> Alcotest.failf "nvrtc rejected the positive control: %s" e
    | Ok ptx -> (
        (* One architecture is enough: the question is whether the classifier
           can see a fused form at all. *)
        let arch =
          List.find_opt
            (fun a ->
              match sass_of_ptx ~arch:a ptx with
              | Ok _ -> true
              | Error (Unknown_arch _) -> false
              | Error (Unclassifiable_arch e) ->
                  Alcotest.failf
                    "%s: refusing to classify this architecture's SASS: %s"
                    a
                    (String.trim e)
              | Error (Tool_error e) ->
                  (* Same rule as the gate: a target the toolchain knows but
                     could not process is a failure, not "no usable arch". *)
                  Alcotest.failf
                    "%s: the local CUDA toolchain rejected the positive \
                     control: %s"
                    a
                    (String.trim e))
            architectures
        in
        match arch with
        | None ->
            Printf.printf "  [SKIP] positive control: no usable arch\n" ;
            Alcotest.skip ()
        | Some arch -> (
            match sass_of_ptx ~arch ptx with
            | Error e -> Alcotest.failf "%s: %s" arch (failure_message e)
            | Ok sass ->
                let insns = instructions sass in
                let fused = binary16_arithmetic insns in
                if fused = [] then
                  Alcotest.failf
                    "%s: the classifier failed to flag native binary16 \
                     arithmetic (__hmul/__hadd). The unfused gate is therefore \
                     vacuous. Stream was:\n\
                     %s"
                    arch
                    (render insns) ;
                Printf.printf
                  "  positive control on %s: fusion detected (%s)\n"
                  arch
                  (String.concat ", " (List.map (fun i -> i.mnemonic) fused))))

let () =
  Alcotest.run
    "cuda_f16_sass"
    [
      ( "sass_fusion_gate",
        [
          Alcotest.test_case
            "midround SASS keeps the f32 discipline on every architecture"
            `Quick
            test_midround_sass_unfused;
          Alcotest.test_case
            "classifier detects genuine binary16 arithmetic (non-vacuity)"
            `Quick
            test_classifier_detects_fusion;
        ] );
    ]
