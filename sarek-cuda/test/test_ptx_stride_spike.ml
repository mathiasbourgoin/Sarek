(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Hardware spike for aggregate-element addressing.

    Validates the exact PTX addressing shape the aggregate emitter produces
    for records stored in global vectors: a non-power-of-2 element stride
    computed with [mul.wide.u32], plus per-field immediate offsets
    ([\[%rd+4\]], [\[%rd+8\]]). NVCC rarely emits this form (it prefers
    shifts for pow2 strides), so translator layers such as ZLUDA may
    under-test it — this pins the behavior on whatever driver is present.

    The kernel reads a 12-byte {x; y; z} float32 element at index [i] and
    stores [x + y + z] into a scalar output vector. Skips cleanly when no
    CUDA driver is available. *)

open Sarek_cuda

let stride12_ptx =
  {|.version 8.0
.target sm_86
.address_size 64

.entry stride12(
    .param .u64 param_in,
    .param .u32 param_sarek_in_length,
    .param .u64 param_out,
    .param .u32 param_sarek_out_length,
    .param .u32 param_n
)
{
    .reg .u32 %r<10>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<6>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [param_in];
    ld.param.u32 %r0, [param_sarek_in_length];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r1, [param_sarek_out_length];
    ld.param.u32 %r2, [param_n];
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r4, %r5;
    add.u32 %r7, %r3, %r6;
    setp.lt.s32 %p0, %r7, %r2;
    @!%p0 bra L0;
    mul.wide.u32 %rd2, %r7, 12;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    ld.global.f32 %f1, [%rd3+4];
    ld.global.f32 %f2, [%rd3+8];
    add.f32 %f3, %f0, %f1;
    add.f32 %f4, %f3, %f2;
    mul.wide.u32 %rd4, %r7, 4;
    add.u64 %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f4;
L0:
    ret;
}
|}

let n = 4096

let block_size = 256

let test_stride12 () =
  if not (Cuda_api.is_driver_available ()) then (
    Printf.printf "  [SKIP] no CUDA device\n%!" ;
    ())
  else begin
    Cuda_api.Device.init () ;
    let dev = Cuda_api.Device.get 0 in

    (* Interleaved {x; y; z} elements: x = i, y = 2i, z = 3i -> sum = 6i *)
    let h_in =
      Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout (3 * n)
    in
    for i = 0 to n - 1 do
      h_in.{3 * i} <- float_of_int i ;
      h_in.{(3 * i) + 1} <- float_of_int (2 * i) ;
      h_in.{(3 * i) + 2} <- float_of_int (3 * i)
    done ;
    let h_out = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
    Bigarray.Array1.fill h_out 0.0 ;

    let d_in = Cuda_api.Memory.alloc dev (3 * n) Bigarray.float32 in
    let d_out = Cuda_api.Memory.alloc dev n Bigarray.float32 in
    Cuda_api.Memory.host_to_device ~src:h_in ~dst:d_in ;

    let kernel =
      Cuda_api.Kernel.load_from_ptx dev ~name:"stride12" ~ptx:stride12_ptx
    in
    let grid = ((n + block_size - 1) / block_size, 1, 1) in
    let block = (block_size, 1, 1) in
    Cuda_api.Kernel.launch
      kernel
      ~args:
        (let len = Cuda_api.Kernel.ArgInt32 (Int32.of_int n) in
         [
           Cuda_api.Kernel.ArgBuffer d_in;
           len;
           Cuda_api.Kernel.ArgBuffer d_out;
           len;
           len (* param_n *);
         ])
      ~grid
      ~block
      ~shared_mem:0
      ~stream:None ;

    Cuda_api.Device.synchronize dev ;
    Cuda_api.Memory.device_to_host ~src:d_out ~dst:h_out ;

    let ok = ref true in
    for i = 0 to n - 1 do
      let expected = float_of_int (6 * i) in
      let got = h_out.{i} in
      if abs_float (got -. expected) > 1e-3 then begin
        if !ok then
          Printf.printf "  FAIL at i=%d: expected %f got %f\n%!" i expected got ;
        ok := false
      end
    done ;
    Alcotest.(check bool) "stride-12 field addressing correct" true !ok ;

    Cuda_api.Memory.free d_in ;
    Cuda_api.Memory.free d_out
  end

let () =
  Alcotest.run
    "ptx_stride_spike"
    [
      ( "addressing",
        [
          Alcotest.test_case
            "mul.wide.u32 stride-12 + immediate field offsets"
            `Quick
            test_stride12;
        ] );
    ]
