(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Key-level regression test for the CUDA compile-cache fix.
 *
 * NOT hardware-verified: CUDA hardware is commonly absent in CI, and
 * Cuda_api.Kernel.compile_cached requires a real device/context, so this
 * test cannot drive the full compile path here. Instead it exercises the
 * exact key-construction logic Cuda_api.Kernel.compile_cached now uses
 * (Spoc_framework.Compile_cache.make_key keyed on device id + kernel name +
 * source digest), proving the two-kernels-one-source collision described in
 * the bug report can no longer happen at the key level.
 *
 * Pre-fix behavior (not reproduced here since the old code path is gone):
 * the key was built from device id + source digest only
 * (Printf.sprintf "%d:%s" device.Device.id (Digest.string source)), so
 * compiling "kernel_b" after "kernel_a" from the same source string would
 * hit the cache and silently return kernel_a's resolved handle.
 ******************************************************************************)

module CC = Spoc_framework.Compile_cache

let test_cuda_style_key_includes_name () =
  let shared_source =
    "__global__ void kernel_a(){} __global__ void kernel_b(){}"
  in
  let device_id = 0 in
  let key_a =
    CC.make_key
      ~device:(string_of_int device_id)
      ~name:"kernel_a"
      ~source:shared_source
      ()
  in
  let key_b =
    CC.make_key
      ~device:(string_of_int device_id)
      ~name:"kernel_b"
      ~source:shared_source
      ()
  in
  Alcotest.(check bool)
    "kernel_a and kernel_b from the same source must not share a cache key"
    true
    (key_a <> key_b)

let () =
  Alcotest.run
    "Cuda_cache_key"
    [
      ( "compile_cache_key",
        [
          Alcotest.test_case
            "CUDA-style key includes kernel name"
            `Quick
            test_cuda_style_key_includes_name;
        ] );
    ]
