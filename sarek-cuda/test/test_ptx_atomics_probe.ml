(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Hardware probe for the wide/exotic atomic forms.

    The emitter now produces [atom.global.add.f64], [atom.global.add.u64],
    [atom.global.inc.u32]/[dec.u32] and [atom.global.cas.b32] — forms that
    NVCC-generated PTX emits rarely, so PTX translators (ZLUDA) may under-test
    them. This pins their runtime behavior on whatever driver is present, in the
    style of test_ptx_stride_spike. Skips cleanly without a CUDA driver.

    Kernel: every thread atomically adds 1.0 to acc_f64[0], adds 1 to acc_u64
    (viewed as u64), increments acc_inc (wrap limit 0xffffffff) and performs one
    CAS attempt on acc_cas (expected 0 -> 42; exactly one thread wins). *)

open Sarek_cuda

let atomics_ptx =
  {|.version 8.0
.target sm_86
.address_size 64

.entry atomics_probe(
    .param .u64 param_acc,
    .param .u32 param_sarek_acc_length,
    .param .u32 param_n
)
{
    .reg .u32 %r<8>;
    .reg .u64 %rd<6>;
    .reg .f64 %fd<3>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [param_acc];
    ld.param.u32 %r0, [param_sarek_acc_length];
    ld.param.u32 %r1, [param_n];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mul.lo.u32 %r5, %r3, %r4;
    add.u32 %r6, %r2, %r5;
    setp.ge.s32 %p0, %r6, %r1;
    @%p0 bra L0;
    mov.f64 %fd0, 0D3FF0000000000000;
    atom.global.add.f64 %fd1, [%rd0], %fd0;
    add.u64 %rd1, %rd0, 8;
    mov.u64 %rd2, 1;
    atom.global.add.u64 %rd3, [%rd1], %rd2;
    add.u64 %rd4, %rd0, 16;
    mov.u32 %r7, 4294967295;
    atom.global.inc.u32 %r7, [%rd4], %r7;
    add.u64 %rd5, %rd0, 24;
    atom.global.cas.b32 %r7, [%rd5], 0, 42;
L0:
    ret;
}
|}

let n = 4096

let block_size = 256

(* The accumulator buffer is 4 x 8 bytes viewed as f64 lanes:
   [0] f64 sum; [1] u64 count (bit pattern); [2] u32 inc count (low word);
   [3] u32 cas cell (low word). *)
let test_atomics () =
  if not (Cuda_api.is_driver_available ()) then (
    Printf.printf "  [SKIP] no CUDA device\n%!" ;
    ())
  else begin
    Cuda_api.Device.init () ;
    let dev = Cuda_api.Device.get 0 in
    let h_acc = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 4 in
    Bigarray.Array1.fill h_acc 0.0 ;
    let d_acc = Cuda_api.Memory.alloc dev 4 Bigarray.float64 in
    Cuda_api.Memory.host_to_device ~src:h_acc ~dst:d_acc ;
    let kernel =
      Cuda_api.Kernel.load_from_ptx dev ~name:"atomics_probe" ~ptx:atomics_ptx
    in
    let grid = ((n + block_size - 1) / block_size, 1, 1) in
    Cuda_api.Kernel.launch
      kernel
      ~args:
        (let len = Cuda_api.Kernel.ArgInt32 (Int32.of_int n) in
         [Cuda_api.Kernel.ArgBuffer d_acc; len; len (* param_n *)])
      ~grid
      ~block:(block_size, 1, 1)
      ~shared_mem:0
      ~stream:None ;
    Cuda_api.Device.synchronize dev ;
    Cuda_api.Memory.device_to_host ~src:d_acc ~dst:h_acc ;
    let f64_sum = h_acc.{0} in
    let u64_count = Int64.bits_of_float h_acc.{1} in
    let low_word x =
      Int64.to_int (Int64.logand (Int64.bits_of_float x) 0xffffffffL)
    in
    let inc_count = low_word h_acc.{2} in
    let cas_cell = low_word h_acc.{3} in
    Alcotest.(check (float 0.5))
      "atom.add.f64 accumulates n"
      (float_of_int n)
      f64_sum ;
    Alcotest.(check int64) "atom.add.u64 counts n" (Int64.of_int n) u64_count ;
    Alcotest.(check int) "atom.inc.u32 counts n" n inc_count ;
    Alcotest.(check int) "atom.cas.b32 exactly-one winner" 42 cas_cell
  end

let () =
  Alcotest.run
    "ptx_atomics_probe"
    [
      ( "atomics",
        [
          Alcotest.test_case
            "wide/exotic atom forms execute correctly"
            `Quick
            test_atomics;
        ] );
    ]
