(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Snapshot test for the PTX code generator.

    Verifies that [Sarek_ir_ptx.demo_vector_add_ptx ()] produces the exact PTX
    string that was validated by ptxas --gpu-name sm_86 on 2026-06-13. This test
    is CPU-only: it exercises the emitter pipeline (types → mem → expr → stmt →
    kernel) without requiring a CUDA device. *)

(** Expected PTX output — register allocation traced from the emitter and
    validated by ptxas --gpu-name sm_86 (2026-06-13). *)
let expected_ptx =
  {|.version 8.0
.target sm_86
.address_size 64

.entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .u32 %r<7>;
    .reg .u64 %rd<12>;
    .reg .f32 %f<3>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_c];
    ld.param.u32 %r0, [param_n];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r4, %r2, %r3;
    add.u32 %r5, %r1, %r4;
    setp.lt.s32 %p0, %r5, %r0;
    selp.u32 %r6, 1, 0, %p0;
    setp.ne.u32 %p1, %r6, 0;
    @!%p1 bra L0;
    cvt.u64.u32 %rd3, %r5;
    shl.b64 %rd4, %rd3, 2;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];
    cvt.u64.u32 %rd6, %r5;
    shl.b64 %rd7, %rd6, 2;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.f32 %f1, [%rd8];
    add.f32 %f2, %f0, %f1;
    cvt.u64.u32 %rd9, %r5;
    shl.b64 %rd10, %rd9, 2;
    add.u64 %rd11, %rd2, %rd10;
    st.global.f32 [%rd11], %f2;
L0:
    ret;
}
|}

let test_vector_add_snapshot () =
  let got = Sarek_codegen.Sarek_ir_ptx.demo_vector_add_ptx () in
  Alcotest.(check string) "vector_add PTX snapshot" expected_ptx got

let () =
  Alcotest.run
    "ptx_snapshot"
    [
      ( "codegen",
        [
          Alcotest.test_case
            "demo_vector_add_ptx snapshot"
            `Quick
            test_vector_add_snapshot;
        ] );
    ]
