(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Hardware probe for tensor-core [mma.sync] PTX (GO/NO-GO evidence for L15b).

    L15b would optimize the L15a tiled GEMM with warp-level tensor-core
    matrix-multiply-accumulate. That is only implementable if the runtime stack
    both (a) ASSEMBLES the [mma.sync] PTX form (ptxas, static) and (b) EXECUTES
    it correctly on this GPU. On this machine the driver is ZLUDA (CUDA-on-ROCm)
    on an RX 7900 XTX (RDNA3, which has WMMA hardware) — so whether ZLUDA's PTX
    translator implements [mma.sync] is an open, empirical question. This probe
    answers (b), in the style of [test_ptx_atomics_probe]. Skips cleanly with no
    CUDA driver.

    Kernel: one warp (32 threads) computes a single
    [mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32]: D = A * B + C with A
    (16x8) and B (8x8) all-ones f16, C all-2.0 f32. Because every A/B element is
    1.0 the K=8 dot product is 8.0 for EVERY output element regardless of the
    (thread,register)->(row,col) fragment mapping, so D[i,j] = 8.0 + 2.0 = 10.0
    everywhere. All-ones inputs make the check robust to fragment-layout
    details; the nonzero C discriminates a translator that drops the accumulator
    (would give 8.0) from one that drops the product (would give 2.0). Each
    thread stores its four D registers to out[lane*4..], and the host asserts
    all 128 lanes equal 10.0.

    Legacy fallback: if the [mma.sync] module fails to LOAD (ZLUDA rejects the
    instruction) the probe retries the pre-Volta [wmma.mma.sync] form and
    reports which, if either, is viable. *)

open Sarek_cuda

(* f16 1.0 = 0x3C00; packed x2 into a .b32 = 0x3C003C00 = 1010449920. *)
let mma_ptx =
  {|.version 8.0
.target sm_80
.address_size 64

.visible .entry mma_probe(
    .param .u64 param_out
)
{
    .reg .u64 %ptr<3>;
    .reg .u32 %t<2>;
    .reg .b32 %a<2>;
    .reg .b32 %b<1>;
    .reg .f32 %c<4>;
    .reg .f32 %d<4>;

    ld.param.u64 %ptr0, [param_out];
    mov.b32 %a0, 1010449920;
    mov.b32 %a1, 1010449920;
    mov.b32 %b0, 1010449920;
    mov.f32 %c0, 0f40000000;
    mov.f32 %c1, 0f40000000;
    mov.f32 %c2, 0f40000000;
    mov.f32 %c3, 0f40000000;

    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
        {%d0, %d1, %d2, %d3},
        {%a0, %a1},
        {%b0},
        {%c0, %c1, %c2, %c3};

    mov.u32 %t0, %tid.x;
    mul.wide.u32 %ptr1, %t0, 16;
    add.u64 %ptr2, %ptr0, %ptr1;
    st.global.f32 [%ptr2], %d0;
    st.global.f32 [%ptr2+4], %d1;
    st.global.f32 [%ptr2+8], %d2;
    st.global.f32 [%ptr2+12], %d3;
    ret;
}
|}

(* Legacy Volta-era wmma path (16x16x16 f16->f32). Load-only smoke: if this
   module loads where mma.sync did not, that is the fallback lead for L15b. *)
let wmma_ptx =
  {|.version 8.0
.target sm_80
.address_size 64

.visible .entry wmma_probe(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_out
)
{
    .reg .u64 %pa, %pb, %po;
    .reg .b32 %a<8>;
    .reg .b32 %b<8>;
    .reg .f32 %c<8>;

    ld.param.u64 %pa, [param_a];
    ld.param.u64 %pb, [param_b];
    ld.param.u64 %po, [param_out];
    wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7}, [%pa];
    wmma.load.b.sync.aligned.col.m16n16k16.global.f16 {%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7}, [%pb];
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
        {%c0,%c1,%c2,%c3,%c4,%c5,%c6,%c7},
        {%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7},
        {%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7},
        {%c0,%c1,%c2,%c3,%c4,%c5,%c6,%c7};
    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%po], {%c0,%c1,%c2,%c3,%c4,%c5,%c6,%c7};
    ret;
}
|}

let warp = 32

let n_out = warp * 4 (* four f32 D-fragment registers per lane = 128 *)

let expected = 10.0 (* 8 (=sum of eight 1*1) + 2 (accumulator C) *)

(* Try to load+launch the mma kernel. Returns [Ok values] on success or
   [Error msg] if ZLUDA rejects the module or the launch. *)
let run_mma dev =
  try
    let h = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n_out in
    Bigarray.Array1.fill h (-1.0) ;
    let d = Cuda_api.Memory.alloc dev n_out Bigarray.float32 in
    Cuda_api.Memory.host_to_device ~src:h ~dst:d ;
    let kernel =
      Cuda_api.Kernel.load_from_ptx dev ~name:"mma_probe" ~ptx:mma_ptx
    in
    Cuda_api.Kernel.launch
      kernel
      ~args:[Cuda_api.Kernel.ArgBuffer d]
      ~grid:(1, 1, 1)
      ~block:(warp, 1, 1)
      ~shared_mem:0
      ~stream:None ;
    Cuda_api.Device.synchronize dev ;
    Cuda_api.Memory.device_to_host ~src:d ~dst:h ;
    Ok h
  with e -> Error (Printexc.to_string e)

let run_wmma_load dev =
  try
    let a = Cuda_api.Memory.alloc dev 256 Bigarray.float32 in
    let b = Cuda_api.Memory.alloc dev 256 Bigarray.float32 in
    let o = Cuda_api.Memory.alloc dev 256 Bigarray.float32 in
    let kernel =
      Cuda_api.Kernel.load_from_ptx dev ~name:"wmma_probe" ~ptx:wmma_ptx
    in
    Cuda_api.Kernel.launch
      kernel
      ~args:
        [
          Cuda_api.Kernel.ArgBuffer a;
          Cuda_api.Kernel.ArgBuffer b;
          Cuda_api.Kernel.ArgBuffer o;
        ]
      ~grid:(1, 1, 1)
      ~block:(warp, 1, 1)
      ~shared_mem:0
      ~stream:None ;
    Cuda_api.Device.synchronize dev ;
    Ok ()
  with e -> Error (Printexc.to_string e)

let test_mma () =
  if not (Cuda_api.is_driver_available ()) then
    Printf.printf "  [SKIP] no CUDA device\n%!"
  else begin
    Cuda_api.Device.init () ;
    let dev = Cuda_api.Device.get 0 in
    let major, minor = dev.Cuda_api.Device.compute_capability in
    Printf.printf "  [INFO] device compute capability sm_%d%d\n%!" major minor ;
    match run_mma dev with
    | Ok h ->
        (* mma.sync loaded and launched. Assert the arithmetic is correct on
           EVERY lane — this is the GO claim, so it is a hard assertion. *)
        Printf.printf
          "  [INFO] mma.sync loaded+launched; out[0]=%g out[127]=%g\n%!"
          h.{0}
          h.{n_out - 1} ;
        for i = 0 to n_out - 1 do
          Alcotest.(check (float 0.01))
            (Printf.sprintf "mma D[lane %d]=A*B+C" i)
            expected
            h.{i}
        done
    | Error msg ->
        (* mma.sync is not viable on this stack. Record the exact rejection and
           probe the legacy wmma path so the verdict doc has both data points.
           This is the documented PARTIAL/NO-GO branch; do not fail the suite —
           the probe's job is to report capability, not to require it. *)
        Printf.printf "  [INFO] mma.sync NOT viable: %s\n%!" msg ;
        (match run_wmma_load dev with
        | Ok () ->
            Printf.printf "  [INFO] legacy wmma.mma.sync DID load+launch\n%!"
        | Error m ->
            Printf.printf "  [INFO] legacy wmma.mma.sync also failed: %s\n%!" m) ;
        Printf.printf
          "  [SKIP] tensor-core mma not available on this driver (see \
           docs/optimization/l15b-mma-probe.md)\n\
           %!"
  end

let () =
  Alcotest.run
    "ptx_mma_probe"
    [
      ( "mma",
        [
          Alcotest.test_case
            "mma.sync tensor-core matmul executes correctly"
            `Quick
            test_mma;
        ] );
    ]
