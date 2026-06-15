(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * PR-5a proof: FFI-free stdlib metadata population
 *
 * Asserts that sarek_stdlib_meta populates Sarek_ppx_registry with
 * Float32.sin, Int32.add_int32, Gpu.thread_idx_x, etc. without any
 * Ctypes or Spoc_core dependency.
 *
 * This exe links only sarek_frontend + sarek_stdlib_meta (no FFI).
 * If it compiles and passes, the typer can resolve stdlib intrinsics
 * without loading the FFI sarek_stdlib.
 ******************************************************************************)

(* Force sarek_stdlib_meta initialization *)
let () = Sarek_stdlib_meta.force_init ()

let check name =
  match Sarek_ppx_registry.find_intrinsic name with
  | Some info ->
      Printf.printf
        "[PASS] %s found: qualified=%s type=%s\n"
        name
        info.ii_qualified_name
        (Sarek_types.typ_to_string info.ii_type)
  | None ->
      Printf.printf "[FAIL] %s NOT found in registry\n" name ;
      exit 1

(* The transpile typer keys intrinsics as <last-module-component>.<name>
   (Sarek_env.short_module_name). Assert the module path is recorded so a
   user-written `Float32.sin` resolves — not merely the bare short name. *)
let check_qualified short_name expected_mod =
  match Sarek_ppx_registry.find_intrinsic short_name with
  | Some info ->
      let last_mod =
        match List.rev info.ii_module with m :: _ -> m | [] -> "<none>"
      in
      if last_mod = expected_mod then
        Printf.printf
          "[PASS] %s resolves qualified as %s.%s\n"
          short_name
          expected_mod
          short_name
      else begin
        Printf.printf
          "[FAIL] %s module last component=%s, expected %s\n"
          short_name
          last_mod
          expected_mod ;
        exit 1
      end
  | None ->
      Printf.printf "[FAIL] %s NOT found in registry\n" short_name ;
      exit 1

let check_type name =
  match Sarek_ppx_registry.find_type name with
  | Some info ->
      Printf.printf "[PASS] type %s found: size=%d\n" name info.ti_size
  | None ->
      Printf.printf "[FAIL] type %s NOT found in registry\n" name ;
      exit 1

let check_size name expected =
  match Sarek_ppx_registry.find_type name with
  | Some info when info.ti_size = expected ->
      Printf.printf
        "[PASS] type %s size=%d (expected %d)\n"
        name
        info.ti_size
        expected
  | Some info ->
      Printf.printf
        "[FAIL] type %s size=%d but expected %d\n"
        name
        info.ti_size
        expected ;
      exit 1
  | None ->
      Printf.printf "[FAIL] type %s NOT found in registry\n" name ;
      exit 1

let () =
  Printf.printf "=== PR-5a proof: FFI-free stdlib metadata ===\n" ;

  (* Type registrations — sizes must match Ctypes.sizeof values *)
  check_type "float32" ;
  check_size "float32" 4 ;
  check_type "int32" ;
  check_size "int32" 4 ;
  check_type "int64" ;
  check_size "int64" 8 ;

  (* Float32 intrinsics *)
  check "sin" ;
  check "cos" ;
  check "sqrt" ;
  check "exp" ;
  check "add_float32" ;

  (* Int32 intrinsics *)
  check "add_int32" ;
  check "sub_int32" ;
  check "logand" ;

  (* Int64 intrinsics *)
  check "add_int64" ;

  (* Math intrinsics *)
  check "xor" ;
  check "pow" ;

  (* Gpu intrinsics (registered as zero-arity functions by the PPX) *)
  check "thread_idx_x" ;
  check "block_idx_x" ;
  check "global_thread_id" ;

  (* Qualified resolution — the actual PR-5b contract (Float32.sin, not bare sin) *)
  Printf.printf "\n--- qualified-name resolution (typer key form) ---\n" ;
  check_qualified "sin" "Float32" ;
  check_qualified "add_int32" "Int32" ;
  check_qualified "add_int64" "Int64" ;
  check_qualified "xor" "Math" ;
  check_qualified "thread_idx_x" "Gpu" ;

  let total = List.length (Sarek_ppx_registry.all_intrinsics ()) in
  Printf.printf
    "\n=== PASS: registry populated FFI-free (%d intrinsics registered) ===\n"
    total
