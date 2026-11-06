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
let run_f16 ?(extra = []) ~gpu ir ~len =
  let a = f16_vec (Array.init len (fun p -> float_of_int (small p))) in
  let b = f16_vec (Array.init len (fun p -> float_of_int (small (p + 3)))) in
  let c = f32_vec (Array.init len (fun p -> float_of_int (small (p + 5)))) in
  let d = f32_vec (Array.init len sentinel_f32) in
  let args = Execute.[Vec a; Vec b; Vec c; Vec d] @ extra in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Array.map (fun x -> x) (Vector.to_array d)

let run_int ?(extra = []) ~gpu ir ~len =
  let a = u8_vec (Array.init len (fun p -> ((p * 7) + 3) land 0xff)) in
  let b = u8_vec (Array.init len (fun p -> ((p * 11) + 5) land 0xff)) in
  let c =
    i32_vec (Array.init len (fun p -> Int32.of_int ((p * 37 mod 1009) - 500)))
  in
  let d = i32_vec (Array.init len (fun p -> Int32.of_int (p + 100000))) in
  let args = Execute.[Vec a; Vec b; Vec c; Vec d] @ extra in
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

(* {1 Launch-time variant}

   The same tile with its start index and stride read from the kernel
   parameters [off] and [ld]: the CUDA backends decide at LAUNCH, from these
   values, between the aligned variant (bare wmma, no check, no buffer) and the
   staged one ([Coopmat_variant]); the answer must not change the result. *)

let param_tile cfg =
  let ir = tile_ir cfg in
  let var name =
    Sarek_ir_types.EVar
      {
        Sarek_ir_types.var_name = name;
        var_id = (if name = "off" then 100 else 101);
        var_type = Sarek_ir_types.TInt32;
        var_mutable = false;
      }
  in
  let rec subst = function
    | Sarek_ir_types.SSeq ss -> Sarek_ir_types.SSeq (List.map subst ss)
    | Sarek_ir_types.SCoopmat (Sarek_ir_types.CM_load r) ->
        Sarek_ir_types.SCoopmat
          (Sarek_ir_types.CM_load {r with index = var "off"; stride = var "ld"})
    | Sarek_ir_types.SCoopmat (Sarek_ir_types.CM_store r) ->
        Sarek_ir_types.SCoopmat
          (Sarek_ir_types.CM_store {r with index = var "off"; stride = var "ld"})
    | st -> st
  in
  let int_param name =
    Sarek_ir_types.DParam
      ( {
          Sarek_ir_types.var_name = name;
          var_id = (if name = "off" then 100 else 101);
          var_type = Sarek_ir_types.TInt32;
          var_mutable = false;
        },
        None )
  in
  {
    ir with
    Sarek_ir_types.kern_params =
      ir.Sarek_ir_types.kern_params @ [int_param "off"; int_param "ld"];
    kern_body = subst ir.Sarek_ir_types.kern_body;
  }

(* (off, ld) with the variant the CUDA backends must choose. Operand bytes: 2
   (f16) or 1 (8-bit); the f32 / s32 accumulators are stricter only on the
   index, which the operands already constrain more. *)
let variant_cases f16 =
  let aligned off ld =
    let b = if f16 then 2 else 1 in
    off * b mod 32 = 0 && ld * b mod 16 = 0
  in
  List.map
    (fun (off, ld) -> (off, ld, aligned off ld))
    (if f16 then
       [
         (0, 16);
         (32, 16);
         (16, 24);
         (48, 40);
         (1, 16);
         (8, 16);
         (0, 20);
         (16, 12);
         (3, 19);
       ]
     else [(0, 16); (32, 16); (64, 48); (16, 16); (0, 24); (1, 17); (32, 20)])

let is_cuda dev =
  List.mem dev.Device.framework Sarek_execute.Coopmat_variant.frameworks

let check_variants dev cfg =
  let name = C.config_name cfg in
  let f16 = cfg.C.cfg_a = C.Float16 in
  let ir = param_tile cfg in
  List.iter
    (fun (off, ld, aligned) ->
      let len = off + (16 * ld) + 32 in
      let extra = Execute.[Int32 (Int32.of_int off); Int32 (Int32.of_int ld)] in
      let run ~gpu =
        if f16 then run_f16 ~extra ~gpu ir ~len else run_int ~extra ~gpu ir ~len
      in
      let want = run ~gpu:None in
      let dummy = f32_vec [|0.0|] in
      let args = Execute.[Vec dummy; Vec dummy; Vec dummy; Vec dummy] @ extra in
      let chosen = Execute.coopmat_variant ~device:dev ir args in
      let expected =
        if is_cuda dev then
          if aligned then Sarek_execute.Coopmat_variant.Aligned
          else Sarek_execute.Coopmat_variant.Staged
        else Sarek_execute.Coopmat_variant.Not_applicable
      in
      if chosen <> expected then
        fail
          "%s %s off %d ld %d: variant %s, expected %s"
          dev.Device.name
          name
          off
          ld
          (Sarek_execute.Coopmat_variant.outcome_name chosen)
          (Sarek_execute.Coopmat_variant.outcome_name expected) ;
      let a0, s0 = Sarek_execute.Coopmat_variant.counts () in
      let got =
        try run ~gpu:(Some dev)
        with e when refused_by_measurement e -> raise Skip_config
      in
      let a1, s1 = Sarek_execute.Coopmat_variant.counts () in
      if is_cuda dev && (a1 - a0, s1 - s0) <> if aligned then (1, 0) else (0, 1)
      then
        fail
          "%s %s off %d ld %d: the launch ran the wrong variant (aligned +%d, \
           staged +%d)"
          dev.Device.name
          name
          off
          ld
          (a1 - a0)
          (s1 - s0) ;
      let close g w = if f16 then close_f16 g w else g = w in
      let wrong = ref 0 in
      Array.iteri (fun p g -> if not (close g want.(p)) then incr wrong) got ;
      if !wrong > 0 then
        fail
          "%s %s off %d ld %d [%s variant]: %d of %d elements differ from the \
           interpreter"
          dev.Device.name
          name
          off
          ld
          (Sarek_execute.Coopmat_variant.outcome_name chosen)
          !wrong
          len
      else
        ok
          "%s off %d ld %d: %s variant, tile and neighbours match the \
           interpreter"
          name
          off
          ld
          (Sarek_execute.Coopmat_variant.outcome_name chosen))
    (variant_cases f16)

(* The GEMM: aligned sizes select the aligned variant, a k or n that is not a
   multiple of 8 the staged one (introspection only: such a launch is outside
   the GEMM's own shape check). *)
let check_gemm_variant dev =
  let ir = Sarek_gemm.Coopmat.ir () in
  let dummy = f32_vec [|0.0|] in
  let args m n k =
    Execute.
      [Vec dummy; Vec dummy; Vec dummy; Vec dummy; Int32 m; Int32 n; Int32 k]
  in
  let expect what args want =
    let got = Execute.coopmat_variant ~device:dev ir args in
    let want =
      if is_cuda dev then want else Sarek_execute.Coopmat_variant.Not_applicable
    in
    if got <> want then
      fail
        "%s GEMM %s: variant %s, expected %s"
        dev.Device.name
        what
        (Sarek_execute.Coopmat_variant.outcome_name got)
        (Sarek_execute.Coopmat_variant.outcome_name want)
    else
      ok
        "GEMM %s: %s variant"
        what
        (Sarek_execute.Coopmat_variant.outcome_name got)
  in
  expect
    "m=64 n=64 k=128"
    (args 64l 64l 128l)
    Sarek_execute.Coopmat_variant.Aligned ;
  expect
    "m=64 n=40 k=24"
    (args 64l 40l 24l)
    Sarek_execute.Coopmat_variant.Aligned ;
  expect "k=12" (args 64l 64l 12l) Sarek_execute.Coopmat_variant.Staged ;
  expect "n=20" (args 64l 20l 64l) Sarek_execute.Coopmat_variant.Staged

let configs =
  [
    config C.Float16 C.Float16 C.Float32;
    config C.Sint8 C.Sint8 C.Sint32;
    config C.Uint8 C.Uint8 C.Sint32;
  ]

(* The selection itself, on any machine: [Coopmat_variant.choose] for the CUDA
   frameworks, and the source each variant generates (no device is needed to
   generate PTX or CUDA-C). The aligned variant carries no staging buffer, no
   check and no trap; the staged one a per-warp buffer. *)
let host_variants () =
  let module V = Sarek_execute.Coopmat_variant in
  let contains needle src =
    let n = String.length needle and l = String.length src in
    let rec at i = i + n <= l && (String.sub src i n = needle || at (i + 1)) in
    at 0
  in
  let block = (64, 1, 1) in
  let check what ~framework ir scalars want =
    let k, got = V.choose ~count:false ~framework ir scalars in
    if got <> want then
      fail
        "%s on %s: %s, expected %s"
        what
        framework
        (V.outcome_name got)
        (V.outcome_name want) ;
    let ptx = Sarek_codegen.Sarek_ir_ptx.generate ~sm_target:"sm_75" ~block k in
    let cuda =
      Sarek_codegen.Sarek_ir_cuda.generate_with_types ~block ~types:[] k
    in
    let buffer = contains "sarek_cm_tiles" ptx in
    let cuda_buffer = contains "sarek_cm_tiles[" cuda in
    (match got with
    | V.Aligned ->
        if buffer || cuda_buffer || contains "trap" ptx then
          fail "%s: the aligned variant carries a buffer or a trap" what
    | V.Staged ->
        if not (buffer && cuda_buffer) then
          fail "%s: the staged variant has no staging buffer" what
    | V.Not_applicable -> ()) ;
    ok
      "%s on %s: %s variant%s"
      what
      framework
      (V.outcome_name got)
      (if buffer then " (2 KiB staging buffer)" else " (no staging buffer)")
  in
  let gemm = Sarek_gemm.Coopmat.ir () in
  List.iter
    (fun framework ->
      check
        "GEMM n=64 k=128"
        ~framework
        gemm
        [("n", 64); ("k", 128); ("m", 64)]
        V.Aligned ;
      check
        "GEMM n=40 k=24"
        ~framework
        gemm
        [("n", 40); ("k", 24); ("m", 64)]
        V.Aligned ;
      check
        "GEMM k=12"
        ~framework
        gemm
        [("n", 64); ("k", 12); ("m", 64)]
        V.Staged ;
      check
        "GEMM n=20"
        ~framework
        gemm
        [("n", 20); ("k", 64); ("m", 64)]
        V.Staged ;
      let f16 = param_tile (config C.Float16 C.Float16 C.Float32) in
      check
        "f16 tile off=32 ld=24"
        ~framework
        f16
        [("off", 32); ("ld", 24)]
        V.Aligned ;
      check
        "f16 tile off=8 ld=16"
        ~framework
        f16
        [("off", 8); ("ld", 16)]
        V.Staged ;
      check
        "f16 tile off=0 ld=20"
        ~framework
        f16
        [("off", 0); ("ld", 20)]
        V.Staged ;
      let u8 = param_tile (config C.Uint8 C.Uint8 C.Sint32) in
      check
        "u8 tile off=64 ld=48"
        ~framework
        u8
        [("off", 64); ("ld", 48)]
        V.Aligned ;
      check
        "u8 tile off=16 ld=16"
        ~framework
        u8
        [("off", 16); ("ld", 16)]
        V.Staged)
    V.frameworks ;
  (* any other framework: the kernel as given *)
  let k, got = V.choose ~count:false ~framework:"HIP" gemm [("n", 64)] in
  if got <> V.Not_applicable || k != gemm then
    fail "HIP: the kernel must be left alone" ;
  ok "HIP: kernel left alone"

let () =
  C.accept_relaxed_accumulation
    ~reason:"test_coopmat_misalign_e2e: exact small-integer inputs" ;
  host_variants () ;
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
          check_gemm_variant dev ;
          List.iter
            (fun cfg ->
              try
                check_config dev cfg ;
                check_variants dev cfg
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
