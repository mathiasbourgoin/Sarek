(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cooperative-matrix loads and stores at a start index and a stride that are
 * NOT aligned, against the interpreter, on every device that offers the
 * configuration. [CM_load] / [CM_store] promise any index and any stride
 * ({!Sarek_ir_types}); nvcuda::wmma and the PTX wmma instructions need a
 * 32-byte aligned address and a 16-byte multiple stride, so CUDA/C and CUDA/PTX
 * stage such a tile through a per-warp shared buffer (the other backends have
 * no such precondition). This test is what says the answer is still the
 * interpreter's.
 *
 * For each configuration (f16*f16+f32, s8*s8+s32, u8*u8+s32 where offered),
 * layout combination (row/row/row/row, col/col/col/col, row/col/row/col for A,
 * B, C, D) and (index, stride) case, one tile is run on the device and on the
 * interpreter from the same inputs, in a buffer wider than the tile whose
 * output is pre-filled with a sentinel, and the WHOLE output compared: the
 * tile is right and its neighbours are intact. Inputs are small integers
 * (exact in f16 and f32 whatever the accumulation order), so equality is
 * bitwise.
 *
 * Devices: every one offering the configuration; the environment variable
 * COOPMAT_MISALIGN_FRAMEWORK (e.g. "CUDA/PTX", "HIP", "Vulkan") restricts the
 * run to one framework. The run stops at the first difference or exception.
 * Skips (exit 0) where no device offers a configuration.
 *
 * On an NVIDIA GPU (sm_70+; 8-bit: sm_72+):
 *   dune build sarek/tests/e2e/test_coopmat_misalign_e2e.exe
 *   ./_build/default/sarek/tests/e2e/test_coopmat_misalign_e2e.exe
 * and once more with COOPMAT_MISALIGN_FRAMEWORK=CUDA/C (needs libnvrtc).
 ******************************************************************************)

open Sarek
open Coopmat_hip_support
module Fw = Spoc_framework.Framework_sig

let () =
  Backend_hip.init () ;
  Test_helpers.Benchmarks.init_backends ()

let failures = ref 0

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK  " ^ s)) fmt

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s) ;
      print_endline "stopping at the first failure" ;
      exit 1)
    fmt

let offers dev cfg =
  Sarek_capability.permits
    (C.verdict ~support:dev.Device.capabilities.Fw.coopmat cfg)

let wave = Execute.dims1d 32

let one = Execute.dims1d 1

let layouts =
  C.
    [
      ("row/row/row/row", (Row_major, Row_major, Row_major, Row_major));
      ("col/col/col/col", (Col_major, Col_major, Col_major, Col_major));
      ("row/col/row/col", (Row_major, Col_major, Row_major, Col_major));
    ]

(* (start index, stride): a misaligned index alone, a misaligned stride alone,
   both, an element offset that is a multiple of 8 but not of 16 (f16), and an
   aligned control. *)
let default_cases = [(0, 16); (1, 16); (16, 17); (3, 19); (8, 24)]

(* COOPMAT_MISALIGN_CASES="0:16,1:16" overrides the cases (index:stride). *)
let cases =
  match Sys.getenv_opt "COOPMAT_MISALIGN_CASES" with
  | None -> default_cases
  | Some spec ->
      String.split_on_char ',' spec
      |> List.map (fun c ->
          match String.split_on_char ':' c with
          | [i; s] -> (int_of_string i, int_of_string s)
          | _ -> failwith "COOPMAT_MISALIGN_CASES: expected index:stride,...")

let small k = (k * 5 mod 9) - 4

let sentinel_f32 p = float_of_int ((p mod 13) + 100)

(* Elements the tile touches: index + 15 * stride + 15, plus margin. *)
let buffer_len ~index ~stride = index + (16 * stride) + 32

(* Typed runner: one function per element family, to keep the vector types
   apart. *)
let run_f16 ~gpu ir ~len =
  let a = f16_vec (Array.init len (fun p -> float_of_int (small p))) in
  let b = f16_vec (Array.init len (fun p -> float_of_int (small (p + 3)))) in
  let c = f32_vec (Array.init len (fun p -> float_of_int (small (p + 5)))) in
  let d = f32_vec (Array.init len sentinel_f32) in
  let args = Execute.[Vec a; Vec b; Vec c; Vec d] in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Array.map (fun x -> x) (Vector.to_array d)

let run_int ~gpu ir ~len =
  let a = u8_vec (Array.init len (fun p -> ((p * 7) + 3) land 0xff)) in
  let b = u8_vec (Array.init len (fun p -> ((p * 11) + 5) land 0xff)) in
  let c =
    i32_vec (Array.init len (fun p -> Int32.of_int ((p * 37 mod 1009) - 500)))
  in
  let d = i32_vec (Array.init len (fun p -> Int32.of_int (p + 100000))) in
  let args = Execute.[Vec a; Vec b; Vec c; Vec d] in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Array.map Int32.to_float (Vector.to_array d)

(* f16 results are bounded, not bit-identical, between a matrix unit and the
   interpreter (RDNA3 WMMA returned -0.999999 for an exact -1 on these integer
   inputs): within 1e-4 relative, against differences of at least 1 for any
   misplaced element. Integer configurations are exact. *)
let close_f16 g w = Float.abs (g -. w) <= 1e-4 *. Float.max 1.0 (Float.abs w)

exception Skip_config

let refused_by_measurement e =
  let msg = Printexc.to_string e in
  let needle = "is refused by measurement" in
  let n = String.length needle and l = String.length msg in
  let rec at i = i + n <= l && (String.sub msg i n = needle || at (i + 1)) in
  at 0

let check_config dev cfg =
  let name = C.config_name cfg in
  let f16 = cfg.C.cfg_a = C.Float16 in
  let close g w = if f16 then close_f16 g w else g = w in
  let run ~gpu ir ~len =
    if f16 then run_f16 ~gpu ir ~len else run_int ~gpu ir ~len
  in
  List.iter
    (fun (lname, (la, lb, lc, ld)) ->
      List.iter
        (fun (index, stride) ->
          let len = buffer_len ~index ~stride in
          let ir = tile_ir ~index ~stride ~la ~lb ~lc ~ld cfg in
          let want = run ~gpu:None ir ~len in
          let got =
            try run ~gpu:(Some dev) ir ~len with
            | e when refused_by_measurement e ->
                (* the backend refuses this element type on purpose (Vulkan's
                   f16, docs/fp-contraction-policy.md): not a coopmat result *)
                raise Skip_config
            | e ->
                fail
                  "%s %s [%s] index %d stride %d: %s"
                  dev.Device.name
                  name
                  lname
                  index
                  stride
                  (Printexc.to_string e)
          in
          let init =
            Array.init len (fun p ->
                if f16 then sentinel_f32 p else float_of_int (p + 100000))
          in
          let touched = ref 0 and wrong = ref 0 and first = ref None in
          Array.iteri
            (fun p g ->
              if g <> init.(p) then incr touched ;
              if not (close g want.(p)) then begin
                incr wrong ;
                if !first = None then first := Some (p, g, want.(p))
              end)
            got ;
          if !wrong > 0 then
            fail
              "%s %s [%s] index %d stride %d: %d of %d elements differ from \
               the interpreter (first: element %d, device %g, interpreter %g)"
              dev.Device.name
              name
              lname
              index
              stride
              !wrong
              len
              (match !first with Some (p, _, _) -> p | None -> -1)
              (match !first with Some (_, g, _) -> g | None -> nan)
              (match !first with Some (_, _, w) -> w | None -> nan)
          else if !touched < 100 then
            fail
              "%s %s [%s] index %d stride %d: the tile was not written (%d \
               elements changed)"
              dev.Device.name
              name
              lname
              index
              stride
              !touched
          else
            ok
              "%s [%s] index %d stride %d: tile and neighbours match the \
               interpreter"
              name
              lname
              index
              stride)
        cases)
    layouts

let configs =
  [
    config C.Float16 C.Float16 C.Float32;
    config C.Sint8 C.Sint8 C.Sint32;
    config C.Uint8 C.Uint8 C.Sint32;
  ]

let () =
  C.accept_relaxed_accumulation
    ~reason:"test_coopmat_misalign_e2e: exact small-integer inputs" ;
  let wanted = Sys.getenv_opt "COOPMAT_MISALIGN_FRAMEWORK" in
  let devices =
    Device.init () |> Array.to_list
    |> List.filter (fun d ->
        match wanted with None -> true | Some fw -> d.Device.framework = fw)
  in
  let ran = ref 0 in
  List.iter
    (fun dev ->
      match List.filter (offers dev) configs with
      | [] -> ()
      | offered ->
          incr ran ;
          Printf.printf
            "=== misaligned coopmat: %s [%s] ===\n%!"
            dev.Device.name
            dev.Device.framework ;
          List.iter
            (fun cfg ->
              try check_config dev cfg
              with Skip_config ->
                Printf.printf
                  "  SKIP %s: refused by the backend on purpose\n%!"
                  (C.config_name cfg))
            offered)
    devices ;
  if !ran = 0 then
    print_endline
      "[SKIP] no device offers a cooperative-matrix configuration (or none \
       matches COOPMAT_MISALIGN_FRAMEWORK): the device checks did not run"
  else print_endline "PASS"
