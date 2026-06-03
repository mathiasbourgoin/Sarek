(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * PR-5b proof: FFI-free transpile of Float32.sin kernel to 4 backends
 *
 * This executable depends ONLY on sarek_transpile + sarek_stdlib_meta.
 * No spoc_core, no ctypes in the link closure.
 *
 * Asserts:
 * 1. CUDA output contains "sinf("  (float32 f-suffixed form)
 * 2. OpenCL output contains "sin(" (generic form)
 * 3. Metal output contains "sin("  (generic form)
 * 4. GLSL output contains "sin("   (generic form)
 * 5. [%native] kernel returns Unsupported_native error
 *
 * Run: dune exec sarek/transpile/test/test_transpile_proof.exe
 ******************************************************************************)

let () = Sarek_stdlib_meta.force_init ()

let sin_kernel_src =
  "fun (a : float32 vector) (b : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  b.(i) <- Float32.sin a.(i)"

let native_kernel_src =
  "fun (a : float32 vector) (b : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  b.(i) <- [%native ((fun _dev -> \"native_sin(a[i])\"), a.(i))]"

(** Assert that [output] contains [substr], printing pass/fail *)
let assert_contains ~label ~substr output =
  let len_o = String.length output in
  let len_s = String.length substr in
  let found =
    if len_s = 0 then true
    else
      let result = ref false in
      for i = 0 to len_o - len_s do
        if String.sub output i len_s = substr then result := true
      done;
      !result
  in
  if found then
    Printf.printf "[PASS] %s contains %S\n" label substr
  else begin
    Printf.printf "[FAIL] %s does NOT contain %S\nActual:\n%s\n" label substr output;
    exit 1
  end

let assert_not_contains ~label ~substr output =
  let len_o = String.length output in
  let len_s = String.length substr in
  let found =
    if len_s = 0 then false
    else
      let result = ref false in
      for i = 0 to len_o - len_s do
        if String.sub output i len_s = substr then result := true
      done;
      !result
  in
  if not found then
    Printf.printf "[PASS] %s does NOT contain %S (correct)\n" label substr
  else begin
    Printf.printf "[FAIL] %s unexpectedly contains %S\n" label substr;
    exit 1
  end

let transpile_or_fail backend label src =
  match Sarek_transpile.of_source backend src with
  | Error e ->
      Printf.printf "[FAIL] %s transpile failed: %s\n" label
        (Sarek_transpile.string_of_error e);
      exit 1
  | Ok code -> code

let () =
  Printf.printf "=== PR-5b proof: FFI-free Float32.sin transpile ===\n";

  (* 1. CUDA: Float32.sin must emit sinf( *)
  let cuda_src =
    transpile_or_fail Sarek_transpile.CUDA "CUDA" sin_kernel_src
  in
  Printf.printf "[INFO] CUDA output:\n%s\n"
    (String.sub cuda_src 0 (min 400 (String.length cuda_src)));
  assert_contains ~label:"CUDA" ~substr:"sinf(" cuda_src;
  Printf.printf "[PASS] CUDA: Float32.sin -> sinf()\n";

  (* 2. OpenCL: Float32.sin must emit sin( *)
  let opencl_src =
    transpile_or_fail Sarek_transpile.OpenCL "OpenCL" sin_kernel_src
  in
  assert_contains ~label:"OpenCL" ~substr:"sin(" opencl_src;
  Printf.printf "[PASS] OpenCL: Float32.sin -> sin()\n";

  (* 3. Metal: Float32.sin must emit sin( *)
  let metal_src =
    transpile_or_fail Sarek_transpile.Metal "Metal" sin_kernel_src
  in
  assert_contains ~label:"Metal" ~substr:"sin(" metal_src;
  Printf.printf "[PASS] Metal: Float32.sin -> sin()\n";

  (* 4. GLSL: Float32.sin must emit sin( *)
  let glsl_src =
    transpile_or_fail Sarek_transpile.GLSL "GLSL" sin_kernel_src
  in
  assert_contains ~label:"GLSL" ~substr:"sin(" glsl_src;
  Printf.printf "[PASS] GLSL: Float32.sin -> sin()\n";

  (* 5. [%native] rejection *)
  begin
    match Sarek_transpile.of_source Sarek_transpile.CUDA native_kernel_src with
    | Error (Sarek_transpile.Unsupported_native _) ->
        Printf.printf
          "[PASS] [%%native] kernel correctly rejected with Unsupported_native\n"
    | Error e ->
        Printf.printf
          "[FAIL] [%%native] kernel: expected Unsupported_native, got: %s\n"
          (Sarek_transpile.string_of_error e);
        exit 1
    | Ok _ ->
        Printf.printf
          "[FAIL] [%%native] kernel: expected error, got success\n";
        exit 1
  end;

  (* CUDA must NOT contain bare " sin(" (should be "sinf(") *)
  assert_not_contains ~label:"CUDA" ~substr:" sin(" cuda_src;

  Printf.printf
    "\n=== PASS: all 4 backends transpile Float32.sin FFI-free ===\n"
