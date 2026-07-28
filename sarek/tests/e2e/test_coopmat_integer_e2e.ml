(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-62 slice 3 — the integer cooperative-matrix path END TO END.
 *
 * WHAT THIS ADDS OVER sarek-vulkan/test/test_vulkan_coopmat_integer.ml.
 *
 * That test compiles a HAND-WRITTEN GLSL coopmat shader and measures the
 * driver. This one builds a Sarek IR kernel containing SCoopmat statements,
 * hands it to Execute, and lets Sarek_ir_glsl EMIT the shader — so what is
 * under test here is the codegen, the launch gate and the calling convention,
 * not RADV. docs/fp-contraction-policy.md §7(c) is explicit that a hand-written
 * shader does not substitute for a codegen gate, and the two halves are kept
 * separate for exactly that reason: if the codegen were wrong in a way the
 * driver happened to cancel, a single combined test would still be green.
 *
 * THE ORACLE IS THE INTERPRETER, AND THE COMPARISON IS BIT FOR BIT.
 *
 * SPV_KHR_cooperative_matrix states that integer accumulation is performed at
 * the precision of the result type and is EXACT. So there is no tolerance, no
 * ulp band and no allowlist here: the SAME IR value is executed on the
 * interpreter and on the GPU, and the two int32 arrays must be equal. That is
 * only a legitimate demand because the configuration is integer; the float
 * configurations are refused by both the codegen and the interpreter, and
 * deliberately (design document §5.1 — the order of the k+1 additions is
 * implementation-defined, so no oracle exists).
 *
 * PROVING THE COMPARISON CAN FAIL.
 *
 * Bit-identity against an oracle is worth nothing if the oracle is the same
 * code path. Two positive controls run on the SAME measured GPU output: a
 * C-dropping reference and a stride-1-B reference, each computed by the
 * interpreter from a MUTATED IR kernel, and each must be REJECTED. If either
 * is accepted the test fails.
 *
 * Run with:
 *   dune exec sarek/tests/e2e/test_coopmat_integer_e2e.exe -- --vulkan
 ******************************************************************************)

open Sarek_ir_types
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Test_helpers.Benchmarks.init_backends ()

let mnk = 16

let elems = mnk * mnk

let shape = {Sarek_coopmat.m = mnk; n = mnk; k = mnk}

(* u8 x u8 + s32 -> s32, non-saturating: advertised as configuration [1] of 14
   on the RX 7900 XTX. The accumulator is s32 rather than u32 because s32 is
   what Sarek's TInt32 vector already is — no new host element type is needed on
   the accumulator side, only on the 8-bit operand side. *)
let cfg =
  {
    Sarek_coopmat.cfg_shape = shape;
    cfg_a = Sarek_coopmat.Uint8;
    cfg_b = Sarek_coopmat.Uint8;
    cfg_c = Sarek_coopmat.Sint32;
    cfg_result = Sarek_coopmat.Sint32;
    cfg_saturating = false;
    cfg_scope = Sarek_coopmat.Subgroup;
  }

let frag_a, frag_b, frag_c, frag_d = Sarek_coopmat.fragments_of_config cfg

(** [D = A x B + C], as a Sarek IR kernel.

    [stride1_b] and [drop_c] are the two positive controls, built into the IR
    generator rather than into a separate host reference: a control that takes a
    different path through the code than the thing it controls does not control
    it. Dropping C is expressed by never loading it, leaving the [CM_decl]'s
    zero; [stride1_b] loads B with a stride of 1 — see below.

    No [global_thread_id] anywhere. One subgroup performs one 16x16x16
    multiply-add cooperatively, and the buffer, index and stride arguments of a
    [coopMatLoad] must be dynamically uniform across that subgroup. A kernel
    that indexed them by thread id would be undefined behaviour on the device
    and would silently differ from the interpreter, which has no subgroup. *)
let make_ir ?(drop_c = false) ?(stride1_b = false) () : kernel =
  let v id name ty =
    {var_name = name; var_id = id; var_type = ty; var_mutable = false}
  in
  let a = v 0 "a" (TVec TUint8) in
  let b = v 1 "b" (TVec TUint8) in
  let c = v 2 "c" (TVec TInt32) in
  let d = v 3 "d" (TVec TInt32) in
  let k n = EConst (CInt32 (Int32.of_int n)) in
  let load name frag src stride =
    SCoopmat (CM_load {dst = name; frag; src; index = k 0; stride = k stride})
  in
  let stmts =
    [
      SCoopmat (CM_decl {name = "fa"; frag = frag_a});
      SCoopmat (CM_decl {name = "fb"; frag = frag_b});
      SCoopmat (CM_decl {name = "fc"; frag = frag_c});
      SCoopmat (CM_decl {name = "fd"; frag = frag_d});
      load "fa" frag_a "a" mnk;
      (* [CM_load] with stride [s] reads [m.(r).(c) = buf.(base + r*s + c)], so
         a stride of 1 gives [buf.(r + c)]: every row is the previous row
         shifted by one, and the whole 16x16 fragment is drawn from the first 31
         elements of the buffer.

         That is a Hankel read and NOT the transpose, which would be
         [buf.(c*16 + r)] and is not expressible through the row-major stride at
         all — reaching it needs gl_CooperativeMatrixLayoutColumnMajor, which
         this slice deliberately does not emit. An earlier revision of this
         comment called it "B-transposed"; it was wrong, and a mislabelled
         control is worse than none because the label is what a later reader
         audits against.

         What matters for a control is only that it is a real, emittable IR
         kernel computing a genuinely DIFFERENT function from [D = A*B + C], so
         that a comparison accepting it would be a comparison accepting
         anything. A Hankel read is that. *)
      load "fb" frag_b "b" (if stride1_b then 1 else mnk);
    ]
    @ (if drop_c then [] else [load "fc" frag_c "c" mnk])
    @ [
        SCoopmat (CM_muladd {dst = "fd"; a = "fa"; b = "fb"; c = "fc"; cfg});
        SCoopmat
          (CM_store
             {src = "fd"; frag = frag_d; dst = "d"; index = k 0; stride = k mnk});
      ]
  in
  {
    default_kernel with
    kern_name = "coopmat_u8_s32";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TUint8; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TUint8; arr_memspace = Global});
        DParam (c, Some {arr_elttype = TInt32; arr_memspace = Global});
        DParam (d, Some {arr_elttype = TInt32; arr_memspace = Global});
      ];
    kern_body = SSeq stmts;
  }

(* ------------------------------------------------------------------ *)
(* inputs                                                              *)
(* ------------------------------------------------------------------ *)

(* The same tiling as the driver-side test, so the two halves exercise the same
   arithmetic: A[i][k] = 16k+i and B[k][j] = 16t+j make dispatch t cover sixteen
   disjoint 16x16 blocks of the 256x256 u8 operand-pair space, and t = 0..15
   tiles all 65536 pairs exactly once. *)
let input_a =
  Array.init elems (fun p -> ((16 * (p mod mnk)) + (p / mnk)) land 0xff)

let input_b t = Array.init elems (fun p -> ((16 * t) + (p mod mnk)) land 0xff)

let input_c = Array.init elems (fun p -> Int32.of_int ((p * 37 mod 1009) - 500))

let fill_u8 arr =
  let v = Vector.create Vector.char elems in
  Array.iteri (fun i x -> Vector.set v i (Char.chr x)) arr ;
  v

let fill_i32 arr =
  let v = Vector.create Vector.int32 elems in
  Array.iteri (fun i x -> Vector.set v i x) arr ;
  v

let zeros_i32 () =
  let v = Vector.create Vector.int32 elems in
  for i = 0 to elems - 1 do
    Vector.set v i 0l
  done ;
  v

let args_for ~t =
  let a = fill_u8 input_a
  and b = fill_u8 (input_b t)
  and c = fill_i32 input_c
  and d = zeros_i32 () in
  ([Execute.Vec a; Execute.Vec b; Execute.Vec c; Execute.Vec d], d)

(* ------------------------------------------------------------------ *)
(* runners                                                             *)
(* ------------------------------------------------------------------ *)

(** The device dispatch. One workgroup of exactly one subgroup.

    The block size is the device's PROBED subgroup size, not a constant: a 16x16
    fragment over 64 invocations is 4 components each and over 32 is 8, and
    slice 2 measured 64 on both local devices where the code had been
    hard-coding 32. *)
let run_gpu dev ir ~t =
  let args, d = args_for ~t in
  let sg = dev.Device.capabilities.Spoc_framework.Framework_sig.warp_size in
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d sg)
    ~grid:(Execute.dims1d 1)
    () ;
  Transfer.flush dev ;
  Vector.to_array d

let run_interp ir ~t =
  let args, d = args_for ~t in
  Execute.run_interpreter_vectors
    ~ir
    ~args
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    ~parallel:false ;
  Vector.to_array d

let equal x y =
  Array.length x = Array.length y
  &&
  let ok = ref true in
  Array.iteri (fun i v -> if v <> y.(i) then ok := false) x ;
  !ok

let first_divergence x y =
  let rec go i =
    if i >= Array.length x then None
    else if x.(i) <> y.(i) then Some (i, x.(i), y.(i))
    else go (i + 1)
  in
  go 0

(* ------------------------------------------------------------------ *)

let failures = ref 0

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let () =
  let devices = Array.to_list (Device.init ()) in
  let coopmat_devices =
    List.filter
      (fun dev ->
        Sarek_capability.permits
          (Sarek_coopmat.verdict
             ~support:
               dev.Device.capabilities.Spoc_framework.Framework_sig.coopmat
             cfg))
      devices
  in
  let ir = make_ir () in

  (* The emitted shader, printed unconditionally and asserted on. A codegen test
     whose output nobody can read is a test of a black box, and these four
     substrings are the ones a wrong emission drops silently: without the
     extension lines the shader does not parse, and without the derived
     coopmat type the wrong fragment dimensions compile and compute nonsense. *)
  (match Sarek_codegen.Sarek_ir_glsl.generate ~block:(64, 1, 1) ir with
  | src ->
      print_endline "--- emitted GLSL ---" ;
      print_string src ;
      print_endline "--- end ---" ;
      List.iter
        (fun needle ->
          if not (Test_helpers.string_contains ~needle src) then
            fail "the emitted GLSL does not contain %S" needle)
        [
          "#extension GL_KHR_cooperative_matrix : require";
          "#extension GL_KHR_memory_scope_semantics : require";
          "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require";
          "coopmat<uint8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>";
          "coopmat<int32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>";
          "coopMatLoad(";
          "coopMatMulAdd(";
          "coopMatStore(";
        ]
  | exception e -> fail "GLSL generation raised: %s" (Printexc.to_string e)) ;

  (* The launch gate, on every device that does NOT advertise the
     configuration — checked whether or not any device does, because that is the
     half that can be vacuous. A gate observed only permitting is
     indistinguishable from one that returns Available unconditionally, and this
     workstation has a free negative device (the Raphael iGPU) under the same
     driver build as the positive one. *)
  let refusing =
    List.filter (fun d -> not (List.memq d coopmat_devices)) devices
  in
  if refusing = [] then
    print_endline
      "  note: every device advertises the configuration, so the refusal arm \
       is not exercised on this machine"
  else
    List.iter
      (fun dev ->
        match Execute.check_device_capabilities ~device:dev ir with
        | () ->
            fail
              "%s advertises no such cooperative-matrix configuration, yet the \
               launch gate PERMITTED the kernel"
              dev.Device.name
        | exception _ ->
            Printf.printf "  OK  launch gate refuses on %s\n%!" dev.Device.name)
      refusing ;

  if coopmat_devices = [] then
    print_endline
      "[SKIP] no device advertises the u8*u8+s32->s32 cooperative-matrix \
       configuration; the numeric comparison did not run"
  else
    List.iter
      (fun dev ->
        Printf.printf "device: %s\n%!" dev.Device.name ;
        let controls_c = ref 0 and controls_t = ref 0 in
        for t = 0 to mnk - 1 do
          let gpu = run_gpu dev ir ~t in
          let oracle = run_interp ir ~t in
          (match first_divergence gpu oracle with
          | None -> ()
          | Some (i, g, o) ->
              fail
                "%s tile %d: GPU and interpreter diverge at %d: %ld vs %ld"
                dev.Device.name
                t
                i
                g
                o) ;
          (* Positive controls, computed by the interpreter from MUTATED IR. *)
          if not (equal gpu (run_interp (make_ir ~drop_c:true ()) ~t)) then
            incr controls_c ;
          if not (equal gpu (run_interp (make_ir ~stride1_b:true ()) ~t)) then
            incr controls_t
        done ;
        if !controls_c <> mnk then
          fail
            "%s: the C-dropping control was ACCEPTED on %d of %d tiles"
            dev.Device.name
            (mnk - !controls_c)
            mnk ;
        if !controls_t <> mnk then
          fail
            "%s: the stride-1-B control was ACCEPTED on %d of %d tiles"
            dev.Device.name
            (mnk - !controls_t)
            mnk ;
        if !failures = 0 then
          Printf.printf
            "  OK  16 tiles, all 65536 u8 operand pairs, bit-identical to the \
             interpreter; both controls rejected\n\
             %!")
      coopmat_devices ;

  if !failures > 0 then begin
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1
  end
  else print_endline "PASS"
