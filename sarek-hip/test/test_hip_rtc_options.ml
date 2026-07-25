(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * #57 slice 1 review — hiprtc option-array ORDERING regression gate.
 *
 * [Sarek_hip.Hip_rtc.base_options] carries "-ffp-contract=off", which is a CONFORMANCE
 * requirement: with contraction on, RDNA3 fuses the f32 multiply into the
 * f32->f16 narrowing and the device stops matching the interpreter on 373 of
 * the 63488 finite binary16 inputs.
 *
 * hiprtc hands its option array straight to clang, and clang resolves
 * conflicting floating-point options by LAST OCCURRENCE. So the conformance
 * flag has to be the LAST -ffp-* option in the array; if a caller-supplied
 * "-ffp-contract=fast" (or "-ffast-math", or "-ffp-model=fast") can appear
 * after it, the conformance setting is silently undone and the f16 agreement
 * gate breaks — on hardware only, which is the worst place to find out.
 *
 * The original spelling was [base_options @ lst], i.e. conformance FIRST, which
 * is exactly the losing position. This test pins the fix.
 *
 * Pure host test: [Sarek_hip.Hip_rtc.hiprtc_options] touches no HIP entry point, so this
 * runs with or without ROCm and never skips.
 ******************************************************************************)

let failures = ref 0

let report label ok detail =
  if not ok then incr failures ;
  Printf.printf "    %-52s : %s%s\n" label (if ok then "OK" else "FAIL") detail

(* Index of the last element satisfying [p], or -1. *)
let last_index_of p l =
  List.fold_left (fun (i, best) x -> (i + 1, if p x then i else best)) (0, -1) l
  |> snd

let has_prefix ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let is_fp_option s =
  has_prefix ~prefix:"-ffp-" s || has_prefix ~prefix:"-ffast-math" s

let show l = "[" ^ String.concat "; " (List.map (Printf.sprintf "%S") l) ^ "]"

(* ------------------------------------------------------------------ *)
(* 1. The conformance flag is present, and is the LAST fp option.      *)
(* ------------------------------------------------------------------ *)

let check_conformance_wins label caller =
  let got = Sarek_hip.Hip_rtc.hiprtc_options caller in
  let contract_off_at = last_index_of (fun s -> s = "-ffp-contract=off") got in
  let last_fp_at = last_index_of is_fp_option got in
  let ok = contract_off_at >= 0 && contract_off_at = last_fp_at in
  report
    label
    ok
    (if ok then ""
     else
       Printf.sprintf
         " — -ffp-contract=off at %d but last fp option at %d; result = %s"
         contract_off_at
         last_fp_at
         (show got))

let () =
  print_endline "  hiprtc option assembly" ;
  (* No caller options: the conformance flag must still be there. *)
  check_conformance_wins "empty caller options keep -ffp-contract=off" [] ;
  (* The realistic rocWMMA case: include dirs must survive untouched. *)
  let rocwmma = ["-I/opt/rocm/include"; "-DSAREK=1"] in
  let got = Sarek_hip.Hip_rtc.hiprtc_options rocwmma in
  let preserved =
    List.filteri (fun i _ -> i < List.length rocwmma) got = rocwmma
  in
  report
    "caller include/define options are preserved in order"
    preserved
    (if preserved then "" else Printf.sprintf " — got %s" (show got)) ;
  check_conformance_wins
    "rocWMMA-style options keep -ffp-contract=off last"
    rocwmma ;
  (* The adversarial cases: each of these, placed after the conformance flag,
     would re-enable contraction. Each must end up BEFORE it. *)
  List.iter
    (fun opt ->
      check_conformance_wins
        (Printf.sprintf "%s cannot override conformance" opt)
        [opt])
    ["-ffp-contract=fast"; "-ffp-contract=on"; "-ffast-math"; "-ffp-model=fast"] ;
  (* Several at once, and mixed with benign options. *)
  check_conformance_wins
    "mixed benign + relaxing options keep conformance last"
    ["-I/opt/rocm/include"; "-ffast-math"; "-ffp-contract=fast"; "-DX=1"] ;
  (* ---------------------------------------------------------------- *)
  (* 2. RED-ON-MUTATION anchor: assert the ordering property is not    *)
  (*    vacuous, i.e. it really is order-sensitive. If the caller      *)
  (*    option were placed last (the OLD behaviour), the check above   *)
  (*    must fail.                                                     *)
  (* ---------------------------------------------------------------- *)
  let old_behaviour caller = Sarek_hip.Hip_rtc.base_options @ caller in
  let bad = old_behaviour ["-ffp-contract=fast"] in
  let bad_contract_at = last_index_of (fun s -> s = "-ffp-contract=off") bad in
  let bad_last_fp_at = last_index_of is_fp_option bad in
  let mutation_detected =
    bad_contract_at >= 0 && bad_contract_at <> bad_last_fp_at
  in
  report
    "old base_options-first ordering IS detected as broken"
    mutation_detected
    (if mutation_detected then ""
     else Printf.sprintf " — %s did not trip the check" (show bad)) ;
  if !failures = 0 then (
    print_endline "  hiprtc option assembly: PASS" ;
    exit 0)
  else (
    Printf.printf "  hiprtc option assembly: %d FAILURE(S)\n" !failures ;
    exit 1)
