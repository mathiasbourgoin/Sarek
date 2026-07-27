(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-151 — host-only: which of the 20 shapes can discriminate at all.
 *
 * WHY THIS RUNS BEFORE ANY DEVICE IS TOUCHED.
 *
 * A shape on which all five policies of F16_shape_catalogue are the SAME
 * FUNCTION over the whole finite binary16 domain cannot produce evidence for or
 * against the generative rule. A device sweep of such a shape returns "matches
 * S_strict, 0/63488" and that sentence measures nothing — it is the
 * coincident-model trap docs/fp-contraction-policy.md §12.2 already guards
 * against on two shapes, and it is much more dangerous across twenty, because a
 * table of twenty zeros reads like a strong result.
 *
 * So this executable prints, per shape, how many DISTINCT functions the five
 * policies induce, and the full pairwise separation. A shape with one distinct
 * function is marked NON-DISCRIMINATING and its device result must be read as
 * "no information", not as "the rule holds here".
 *
 * It also runs the calibration that pins the generic evaluator to slice 1's
 * hand-written closed forms. Nothing below the calibration line is readable if
 * the calibration fails, and it exits non-zero in that case.
 *
 * Run:
 *   dune exec tools/f16_shape_catalogue/probe/probe_f16_shape_separation.exe
 ******************************************************************************)

module M = F16_model_set
module C = F16_shape_catalogue

let () =
  Printf.printf
    "backlog-151 — the 20-shape f16 catalogue, host-only separation analysis\n\n" ;
  (try C.calibrate () with
  | C.Calibration_failed s ->
      Printf.printf "CALIBRATION FAILED — read nothing below it:\n  %s\n" s ;
      exit 1
  | M.Calibration_failed s ->
      Printf.printf
        "CALIBRATION FAILED (slice 1's own) — read nothing below it:\n  %s\n"
        s ;
      exit 1) ;
  Printf.printf
    "CALIBRATION PASSED.\n\
    \  - slice 1's four host checks (63488 round-trip, 620, 2912, x = -907.5);\n\
    \  - all 20 shapes x 5 policies are exactly computable;\n\
    \  - the GENERIC evaluator reproduces slice 1's seven hand-written closed\n\
    \    forms bit-for-bit on A2 and B1, all 63488 inputs;\n\
    \  - and reproduces the recorded separations 2912 / 620 / 5075 / 4776 /\n\
    \    4774 from the generic evaluator rather than from those closed forms.\n\n" ;

  Printf.printf
    "SHAPE DISCRIMINATION — how many DISTINCT functions the five policies\n\
     induce on each shape over all 63488 finite binary16 inputs.\n\n" ;
  Printf.printf "  %-4s %-46s %s\n" "id" "shape" "distinct models" ;
  List.iter
    (fun sh ->
      let d = C.distinct_model_count sh in
      Printf.printf
        "  %-4s %-46s %d%s\n"
        sh.C.id
        sh.C.descr
        d
        (if d = 1 then
           "   <== NON-DISCRIMINATING: a device sweep of this shape measures \
            nothing"
         else ""))
    C.shapes ;

  Printf.printf "\n\nPER-SHAPE PAIRWISE SEPARATION\n" ;
  List.iter
    (fun sh ->
      Printf.printf "\n--- %s : %s ---\n" sh.C.id sh.C.descr ;
      if sh.C.discriminating_note <> "" then
        Printf.printf "  NOTE: %s\n" sh.C.discriminating_note ;
      ignore (M.separation_matrix (C.models_of sh)))
    C.shapes ;

  Printf.printf "\n\nGENERATED SOURCE — GLSL, plain, one per shape\n" ;
  List.iter
    (fun sh ->
      Printf.printf
        "\n--- %s ---\n%s"
        sh.C.id
        (C.source ~dialect:C.Glsl ~precise:false ~barrier:false sh))
    C.shapes ;

  Printf.printf "\n\nGENERATED SOURCE — OpenCL C, plain, one per shape\n" ;
  List.iter
    (fun sh ->
      Printf.printf
        "\n--- %s ---\n%s"
        sh.C.id
        (C.source ~dialect:C.Opencl ~precise:false ~barrier:false sh))
    C.shapes
