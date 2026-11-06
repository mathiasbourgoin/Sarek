(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * 8-bit SCoopmat on CUDA/PTX (wmma s8/u8 m16n16k16, s32 accumulation, sm_72+),
 * executed from SAREK-GENERATED PTX and compared with the interpreter.
 *
 * Host checks (every machine): each of the 8 accepted configurations (s8*s8 and
 * u8*u8 with an s32 or u32 accumulator) generates PTX naming the s32.<t>.<t>.s32
 * instruction, mixed-sign is refused, and the interpreter distinguishes the
 * C-dropping control (so the comparison below can see a wrong accumulator).
 *
 * Device checks (each CUDA/PTX device that offers 8-bit coopmat: an NVIDIA GPU
 * of compute capability 7.2 or later): for each configuration, 16 tiles that
 * cover all 65536 operand pairs bit-identical to the interpreter, the
 * C-dropping control rejected, an offset tile (index 400, stride 48) inside a
 * wider buffer bit-identical, and a layout sweep (every row/column combination
 * of A, B, C and D) against the interpreter. Exact integers, so equality is
 * bitwise. A device that does not offer it (no tensor cores, wave64, ZLUDA) is
 * reported and SKIPPED; a ZLUDA device whose module fails to load is SKIPPED.
 *
 * On a machine with an NVIDIA GPU (a line "=== 8-bit wmma on CUDA/PTX: ..."
 * must appear, not [SKIP]):
 *   dune build sarek/tests/e2e/test_ptx_coopmat_int8_e2e.exe
 *   ./_build/default/sarek/tests/e2e/test_ptx_coopmat_int8_e2e.exe
 ******************************************************************************)

open Sarek
open Coopmat_hip_support
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

let offers dev cfg =
  Sarek_capability.permits
    (C.verdict ~support:dev.Device.capabilities.Fw.coopmat cfg)

let is_zluda dev = contains ~needle:"ZLUDA" dev.Device.name

let same_sign_configs =
  List.concat_map
    (fun op -> List.map (fun acc -> config op op acc) [C.Sint32; C.Uint32])
    [C.Uint8; C.Sint8]

let wave = Execute.dims1d 32

let one = Execute.dims1d 1

(* A[i][k] = 16k + i, B[k][j] = 16t + j: tile t covers 4096 byte pairs and the
   sixteen tiles cover all 65536 exactly once. *)
let int_inputs t =
  let a = Array.init 256 (fun p -> ((16 * (p mod 16)) + (p / 16)) land 0xff) in
  let b = Array.init 256 (fun p -> ((16 * t) + (p mod 16)) land 0xff) in
  let c = Array.init 256 (fun p -> Int32.of_int ((p * 37 mod 1009) - 500)) in
  (a, b, c)

let run_tile ~gpu ir t =
  let a, b, c = int_inputs t in
  let d = i32_vec (Array.make 256 0l) in
  let args = Execute.[Vec (u8_vec a); Vec (u8_vec b); Vec (i32_vec c); Vec d] in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Vector.to_array d

(* Offset tile: row 8, column 16 of a 32 x 48 matrix (index 400, stride 48). *)
let run_offset ~gpu cfg =
  let len = 32 * 48 in
  let ir = tile_ir ~index:((8 * 48) + 16) ~stride:48 cfg in
  let a = Array.init len (fun p -> ((p * 97) + 13) land 0xff) in
  let b = Array.init len (fun p -> ((p * 61) + 7) land 0xff) in
  let c = Array.init len (fun p -> Int32.of_int ((p * 37 mod 1009) - 500)) in
  let d = i32_vec (Array.make len 0l) in
  let args = Execute.[Vec (u8_vec a); Vec (u8_vec b); Vec (i32_vec c); Vec d] in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Vector.to_array d

let host_part () =
  List.iter
    (fun cfg ->
      let a = cfg.C.cfg_a in
      let t = if a = C.Uint8 then "u8" else "s8" in
      let ptx =
        Sarek_codegen.Sarek_ir_ptx.generate ~sm_target:"sm_75" (tile_ir cfg)
      in
      let needle = Printf.sprintf "m16n16k16.s32.%s.%s.s32" t t in
      if contains ~needle ptx then
        ok "%s: PTX contains %s" (C.config_name cfg) needle
      else fail "%s: PTX lacks %s" (C.config_name cfg) needle ;
      let with_c = run_tile ~gpu:None (tile_ir cfg) 3 in
      let without = run_tile ~gpu:None (tile_ir ~drop_c:true cfg) 3 in
      if with_c = without then
        fail
          "%s: the interpreter cannot tell the C-dropping control"
          (C.config_name cfg))
    same_sign_configs ;
  match
    Sarek_codegen.Sarek_ir_ptx.generate
      ~sm_target:"sm_75"
      (tile_ir (config C.Uint8 C.Sint8 C.Sint32))
  with
  | (_ : string) -> fail "mixed-sign u8*s8 generated PTX"
  | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error _ ->
      ok "mixed-sign u8*s8 refused"

let check_config dev cfg =
  let ir = tile_ir cfg and control = tile_ir ~drop_c:true cfg in
  let diverged = ref 0 and rejected = ref 0 in
  for t = 0 to 15 do
    let g = run_tile ~gpu:(Some dev) ir t in
    if g <> run_tile ~gpu:None ir t then incr diverged ;
    if g <> run_tile ~gpu:None control t then incr rejected
  done ;
  if !diverged > 0 then
    fail
      "%s: %d of 16 tiles differ from the interpreter"
      (C.config_name cfg)
      !diverged
  else if !rejected <> 16 then
    fail
      "%s: the C-dropping control was accepted on %d tiles"
      (C.config_name cfg)
      (16 - !rejected)
  else ok "%s: 65536 pairs bit-identical, control rejected" (C.config_name cfg) ;
  if run_offset ~gpu:(Some dev) cfg = run_offset ~gpu:None cfg then
    ok
      "%s: offset tile (index 400, stride 48) bit-identical"
      (C.config_name cfg)
  else fail "%s: offset tile differs from the interpreter" (C.config_name cfg)

(* Every row/column combination of A, B, C and D, same data on the device and
   the interpreter. *)
let check_layouts dev =
  let cfg = config C.Sint8 C.Sint8 C.Sint32 in
  let layouts = [C.Row_major; C.Col_major] in
  let bad = ref 0 in
  List.iter
    (fun la ->
      List.iter
        (fun lb ->
          List.iter
            (fun lc ->
              List.iter
                (fun ld ->
                  let ir = tile_ir ~la ~lb ~lc ~ld cfg in
                  if run_tile ~gpu:(Some dev) ir 5 <> run_tile ~gpu:None ir 5
                  then incr bad)
                layouts)
            layouts)
        layouts)
    layouts ;
  if !bad = 0 then ok "all 16 layout combinations match the interpreter"
  else fail "%d of 16 layout combinations differ from the interpreter" !bad

let device_part dev =
  Printf.printf "=== 8-bit wmma on CUDA/PTX: %s ===\n%!" dev.Device.name ;
  match
    List.iter (check_config dev) same_sign_configs ;
    check_layouts dev
  with
  | () -> ()
  | exception e when is_zluda dev ->
      Printf.printf
        "  SKIP: ZLUDA device failed (%s); it rejects wmma instructions\n%!"
        (Printexc.to_string e)

let () =
  let ptx =
    Device.init () |> Array.to_list
    |> List.filter (fun d -> d.Device.framework = "CUDA/PTX")
  in
  host_part () ;
  (match
     List.filter (fun d -> offers d (config C.Uint8 C.Uint8 C.Sint32)) ptx
   with
  | [] ->
      List.iter
        (fun d ->
          Printf.printf
            "  CUDA/PTX %s does not offer 8-bit coopmat: %s\n%!"
            d.Device.name
            (match d.Device.capabilities.Fw.coopmat with
            | Some s -> s.C.ds_evidence
            | None -> "not probed"))
        ptx ;
      print_endline
        "[SKIP] no CUDA/PTX device with 8-bit tensor cores: the device checks \
         did not run"
  | devs -> List.iter device_part devs) ;
  if !failures > 0 then begin
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1
  end
  else print_endline "PASS"
