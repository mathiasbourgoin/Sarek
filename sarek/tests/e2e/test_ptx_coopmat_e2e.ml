(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * SCoopmat on CUDA/PTX (wmma instructions, tensor cores) executed from SAREK-GENERATED
 * PTX, differential against the exact result and the interpreter.
 *
 * Host checks (every machine): the PTX of the cooperative-matrix GEMM carries
 * the wmma instructions, and the interpreter evaluates the GEMM within the
 * Regime B bound (docs/design/f16-relaxed-accuracy.md).
 *
 * Device checks (each CUDA/PTX device that offers the f16 configuration, i.e.
 * an NVIDIA GPU of compute capability 7.0 or later):
 *   - a tiled f16 GEMM, B row-major and B transposed, within gamma_k of the
 *     exact result, repeatable bit for bit, same bits under three tilings;
 *   - the C-dropping control (C not loaded) is rejected by the same gate;
 *   - a 16-thread block is refused.
 * A device that does not offer it (no tensor cores, wave64, ZLUDA: it rejects
 * wmma instructions) is reported and SKIPPED; a ZLUDA device whose module fails to load is
 * SKIPPED too. Nothing here fails for lack of hardware.
 *
 * On a machine with an NVIDIA GPU and its driver (a line
 * "=== wmma on CUDA/PTX: <device> ===" must appear, not [SKIP]):
 *   dune build sarek/tests/e2e/test_ptx_coopmat_e2e.exe
 *   ./_build/default/sarek/tests/e2e/test_ptx_coopmat_e2e.exe
 ******************************************************************************)

open Sarek
open Coopmat_hip_support
module Gemm = Sarek_gemm.Coopmat
module Fw = Spoc_framework.Framework_sig

let () = Test_helpers.Benchmarks.init_backends ()

let failures = ref 0

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK  " ^ s)) fmt

let offers dev =
  Sarek_capability.permits
    (C.verdict ~support:dev.Device.capabilities.Fw.coopmat f16_config)

let is_zluda dev = contains ~needle:"ZLUDA" dev.Device.name

(** Run the GEMM on [gpu] or the interpreter; [bt] stores B transposed. *)
let run_gemm ?tiling ?(bt = false) ~gpu ~a ~b ~c ~m ~n ~k () =
  let t = Option.value tiling ~default:Gemm.default_tiling in
  let b_layout = if bt then C.Col_major else C.Row_major in
  let b =
    if bt then Array.init (k * n) (fun p -> b.((p mod k * n) + (p / k))) else b
  in
  let va = f16_vec a and vb = f16_vec b and vc = f32_vec c in
  let vd = f32_vec (Array.make (m * n) 0.0) in
  (match gpu with
  | Some dev ->
      Gemm.run
        ~device:dev
        ~tiling:t
        ~b_layout
        ~a:va
        ~b:vb
        ~c:vc
        ~d:vd
        ~m
        ~n
        ~k
        () ;
      Transfer.flush dev
  | None ->
      Gemm.check_shape t ~m ~n ~k ;
      let block, grid = Gemm.launch_dims t ~m ~n in
      let int x = Execute.Int32 (Int32.of_int x) in
      run_interp
        (Gemm.ir ~tiling:t ~b_layout ())
        Execute.[Vec va; Vec vb; Vec vc; Vec vd; int m; int n; int k]
        ~block
        ~grid) ;
  to_floats vd

let check_bounded ~what ~exact ~absterms ~adds got =
  match bound_check ~adds ~exact ~absterms got with
  | 0, worst ->
      ok "%s: within gamma_%d of exact (worst %.3g of bound)" what adds worst
  | bad, _ ->
      fail "%s: %d of %d outside gamma_%d" what bad (Array.length got) adds

let check_dims dev ~m ~n ~k =
  let a = operand (m * k) and b = operand (k * n) and c = addend (m * n) in
  assert_exact_in_binary64 ~a ~b ~c ~k ;
  let exact, absterms = exact_gemm ~a ~b ~c ~m ~n ~k in
  let what = Printf.sprintf "%dx%dx%d" m n k in
  List.iter
    (fun bt ->
      let tag = if bt then " B^T" else "" in
      let gpu = run_gemm ~bt ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k () in
      check_bounded ~what:(what ^ tag ^ " gpu") ~exact ~absterms ~adds:k gpu ;
      if run_gemm ~bt ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k () <> gpu then
        fail "%s%s: two runs differ" what tag
      else ok "%s%s: two runs bit-identical" what tag ;
      let differing =
        List.filter
          (fun tiling ->
            run_gemm ~tiling ~bt ~gpu:(Some dev) ~a ~b ~c ~m ~n ~k () <> gpu)
          Gemm.
            [
              {frags_m = 1; frags_n = 1; waves = 1};
              {frags_m = 2; frags_n = 1; waves = 2};
            ]
      in
      if differing = [] then ok "%s%s: three tilings bit-identical" what tag
      else fail "%s%s: %d tiling(s) differ" what tag (List.length differing) ;
      (* Control: dropping C must fall outside the bound, or the gate cannot
         see a wrong accumulator. *)
      let dropped =
        run_gemm
          ~bt
          ~gpu:(Some dev)
          ~a
          ~b
          ~c:(Array.make (m * n) 0.0)
          ~m
          ~n
          ~k
          ()
      in
      match bound_check ~adds:k ~exact ~absterms dropped with
      | 0, _ -> fail "%s%s: C-dropping control ACCEPTED" what tag
      | bad, _ ->
          ok "%s%s: C-dropping control rejected (%d outside)" what tag bad)
    [false; true]

let check_block_refusal dev =
  let v n = f32_vec (Array.make n 0.0) in
  let h n = f16_vec (Array.make n 0.0) in
  match
    refusal (fun () ->
        run_gpu
          dev
          (tile_ir f16_config)
          Execute.[Vec (h 256); Vec (h 256); Vec (v 256); Vec (v 256)]
          ~block:(Execute.dims1d 16)
          ~grid:(Execute.dims1d 1))
  with
  | None -> fail "%s launched a 16-thread coopmat block" dev.Device.name
  | Some msg when contains ~needle:"32-wide" msg -> ok "16-thread block refused"
  | Some msg -> fail "16-thread block refused for another reason: %s" msg

let host_part () =
  let src =
    Sarek_codegen.Sarek_ir_ptx.generate ~sm_target:"sm_75" (Gemm.ir ())
  in
  List.iter
    (fun needle ->
      if contains ~needle src then ok "PTX contains %s" needle
      else fail "PTX lacks %s" needle)
    [
      "wmma.load.a.sync.aligned";
      "wmma.mma.sync.aligned";
      "wmma.store.d.sync.aligned";
    ] ;
  let m = 64 and n = 64 and k = 128 in
  let a = operand (m * k) and b = operand (k * n) and c = addend (m * n) in
  let exact, absterms = exact_gemm ~a ~b ~c ~m ~n ~k in
  check_bounded
    ~what:"interpreter 64x64x128"
    ~exact
    ~absterms
    ~adds:k
    (run_gemm ~gpu:None ~a ~b ~c ~m ~n ~k ())

let device_part dev =
  Printf.printf "=== wmma on CUDA/PTX: %s ===\n%!" dev.Device.name ;
  match
    check_block_refusal dev ;
    check_dims dev ~m:64 ~n:64 ~k:128 ;
    check_dims dev ~m:256 ~n:256 ~k:1024
  with
  | () -> ()
  | exception e when is_zluda dev ->
      Printf.printf
        "  SKIP: ZLUDA device failed (%s); it rejects wmma.*\n%!"
        (Printexc.to_string e)

let () =
  let devices = Array.to_list (Device.init ()) in
  let ptx = List.filter (fun d -> d.Device.framework = "CUDA/PTX") devices in
  C.accept_relaxed_accumulation
    ~reason:"test_ptx_coopmat_e2e: Regime B is what is under test" ;
  host_part () ;
  (match List.filter offers ptx with
  | [] ->
      List.iter
        (fun d ->
          Printf.printf
            "  CUDA/PTX %s does not offer f16 coopmat: %s\n%!"
            d.Device.name
            (match d.Device.capabilities.Fw.coopmat with
            | Some s -> s.C.ds_evidence
            | None -> "not probed"))
        ptx ;
      print_endline
        "[SKIP] no CUDA/PTX device with tensor cores: the device checks did \
         not run"
  | devs -> List.iter device_part devs) ;
  if !failures > 0 then begin
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1
  end
  else print_endline "PASS"
