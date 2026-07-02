(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Hardware-verified check-funnel unification test (S3b item 4).
 *
 * Forces a real OpenCL FFI error (clCreateKernel with a kernel name that
 * does not exist in the built program) and asserts the exception raised by
 * [Opencl_api.check] is the canonical [Sarek_backend_error.Backend_error],
 * not the deprecated [Opencl_api.Opencl_error] variant. Also proves a
 * legacy handler pattern-matching on the deprecated alias still compiles.
 *
 * Requires a real OpenCL device; skips (does not fail) if none is present.
 ******************************************************************************)

open Sarek_opencl

let valid_source =
  {|
__kernel void real_kernel(__global float *out) {
    out[0] = 1.0f;
}
|}

let test_invalid_kernel_name_raises_canonical_backend_error () =
  if not (Opencl_api.is_available ()) then
    Printf.printf
      "[SKIP] No OpenCL device available - skipping hardware check-funnel test\n\
       %!"
  else begin
    Opencl_api.Device.init () ;
    let device = Opencl_api.Device.get 0 in
    let context = Opencl_api.Context.create device in
    let program = Opencl_api.Program.create_from_source context valid_source in
    Opencl_api.Program.build program () ;
    let raised =
      try
        let (_ : Opencl_api.Kernel.t) =
          Opencl_api.Kernel.create program "does_not_exist"
        in
        None
      with e -> Some e
    in
    match raised with
    | None ->
        Alcotest.fail "expected clCreateKernel with an unknown name to fail"
    | Some (Sarek_backend_error.Backend_error.Backend_error _) -> ()
    | Some e ->
        Alcotest.failf "expected Backend_error, got %s" (Printexc.to_string e)
  end

(** Compile-only: a legacy handler pattern-matching on the deprecated
    [Opencl_api.Opencl_error] alias must still type-check (opam-published
    library, out-of-tree code may still reference it). Never reached at runtime;
    [check] no longer raises it (see test above). *)
let _legacy_handler_still_compiles (f : unit -> unit) : unit =
  (try f () with Opencl_api.Opencl_error _ -> ()) [@alert "-deprecated"]

let () =
  Alcotest.run
    "Opencl_ffi_check_funnel"
    [
      ( "check_funnel_unification",
        [
          Alcotest.test_case
            "invalid kernel name raises canonical Backend_error"
            `Quick
            test_invalid_kernel_name_raises_canonical_backend_error;
        ] );
    ]
