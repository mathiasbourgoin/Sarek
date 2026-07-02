(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Hardware-verified non-vacuous test for the compile-cache fix (S3a item 5).
 *
 * OpenCL naturally supports multiple __kernel entry points inside a single
 * program source (extracted by name via clCreateKernel), which makes it the
 * clearest hardware demonstration of the bug class this change guards
 * against: a cache keyed by device + source digest alone (omitting the
 * kernel name) would return kernel_a's compiled handle when asked for
 * kernel_b, since both come from byte-identical source text.
 *
 * This test drives Opencl_plugin_base.Opencl.Kernel.compile_cached (the
 * actual cache used by the OpenCL plugin) directly, executes both kernels
 * on the real device, and asserts each writes its own distinct value -
 * proving no aliasing occurs. OpenCL already included the kernel name in its
 * key before this change (see Opencl_plugin_base.ml:227-233), so this test
 * is a non-regressing proof of correctness rather than a fix-verification;
 * it also exercises the exact cache path CUDA/Vulkan were fixed to match.
 *
 * Requires a real OpenCL device; skips (does not fail) if none is present so
 * the suite stays green in CI environments without a GPU.
 ******************************************************************************)

open Sarek_opencl
module Backend = Opencl_plugin_base.Opencl

let two_kernel_source =
  {|
__kernel void kernel_a(__global float *out) {
    out[0] = 1.0f;
}
__kernel void kernel_b(__global float *out) {
    out[0] = 2.0f;
}
|}

let run_kernel_and_read_result device ~name =
  let compiled =
    Backend.Kernel.compile_cached device ~name ~source:two_kernel_source
  in
  let buf = Backend.Memory.alloc device 1 Bigarray.float32 in
  let args = Backend.Kernel.create_args () in
  Backend.Kernel.set_arg_buffer args 0 buf ;
  let dims = Spoc_framework.Framework_sig.dims_1d 1 in
  Backend.Kernel.launch
    compiled
    ~args
    ~grid:dims
    ~block:dims
    ~shared_mem:0
    ~stream:None ;
  Backend.Device.synchronize device ;
  let result = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 1 in
  Backend.Memory.device_to_host ~src:buf ~dst:result ;
  Backend.Memory.free buf ;
  result.{0}

let test_two_kernels_one_source_resolve_independently () =
  if not (Backend.is_available ()) then
    Printf.printf
      "[SKIP] No OpenCL device available - skipping hardware cache test\n%!"
  else begin
    Backend.Device.init () ;
    let device = Backend.Device.get 0 in
    let result_a = run_kernel_and_read_result device ~name:"kernel_a" in
    let result_b = run_kernel_and_read_result device ~name:"kernel_b" in
    Alcotest.(check (float 0.0001))
      "kernel_a compiled from the shared source writes 1.0"
      1.0
      result_a ;
    Alcotest.(check (float 0.0001))
      "kernel_b compiled from the same shared source writes its own 2.0, not \
       kernel_a's value"
      2.0
      result_b
  end

let () =
  Alcotest.run
    "Opencl_cache_two_kernels"
    [
      ( "compile_cache",
        [
          Alcotest.test_case
            "two kernels sharing one source resolve independently"
            `Quick
            test_two_kernels_one_source_resolve_independently;
        ] );
    ]
