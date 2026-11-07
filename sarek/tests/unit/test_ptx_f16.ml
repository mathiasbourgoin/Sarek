(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** float16 in the PTX backend.

    f16 is a STORAGE type in the DSL: it has no arithmetic, values are widened
    to f32, computed on, and narrowed back. So the emitted PTX may contain only
    [ld]/[st]/[mov]/[selp] on [.b16] registers and exactly two conversions,
    [cvt.f32.f16] (exact) and [cvt.rn.f16.f32] (round to nearest even). None of
    those needs sm_53: native [.f16] arithmetic would, which is why the test
    below forbids it, and why an f16 kernel is valid on every target the backend
    can emit (sm_50 and up).

    Also pins the register-class decoder (a [%h] register must never be taken
    for an integer or f32 one) and the loud failure of every IR shape that would
    need f16 arithmetic. CPU only; the ptxas assembly leg is in
    test_ptx_snapshot.ml (it shares the kernels in ptx_f16_kernels.ml). *)

open Sarek_ir_types
open Sarek_codegen
module K = Ptx_f16_kernels

let contains ~needle s =
  let n = String.length needle and l = String.length s in
  let rec go i = i + n <= l && (String.sub s i n = needle || go (i + 1)) in
  n = 0 || go 0

let check_has ptx needle =
  if not (contains ~needle ptx) then
    Alcotest.failf "expected %S in:\n%s" needle ptx

let check_absent ptx needle =
  if contains ~needle ptx then
    Alcotest.failf "expected no %S in:\n%s" needle ptx

let gen ?sm_target k = Sarek_ir_ptx.generate ?sm_target k

let lines s = String.split_on_char '\n' s |> List.map String.trim

(* The two conversions are the only instructions allowed to name an f16 type. *)
let allowed_f16_mnemonics = ["cvt.f32.f16"; "cvt.rn.f16.f32"]

let mnemonic line =
  match String.index_opt line ' ' with
  | Some i -> String.sub line 0 i
  | None -> line

(** No instruction other than the two conversions mentions [.f16] / [.f16x2]:
    that is exactly "no native f16 arithmetic, compare, min/max, fma, neg, abs",
    all of which would need sm_53. *)
let assert_no_native_f16 name ptx =
  List.iter
    (fun line ->
      let is_instr = line <> "" && line.[0] <> '.' && line.[0] <> '{' in
      if
        is_instr
        && (contains ~needle:".f16" line || contains ~needle:"f16x2" line)
        && not (List.mem (mnemonic line) allowed_f16_mnemonics)
      then Alcotest.failf "%s: native f16 instruction %S" name line)
    (lines ptx)

let test_scale_markers () =
  let ptx = gen (K.scale ()) in
  check_has ptx ".reg .b16 %h<" ;
  check_has ptx "ld.global.nc.b16 %h0" ;
  check_has ptx "cvt.f32.f16 %f0, %h0;" ;
  check_has ptx "cvt.rn.f16.f32" ;
  check_has ptx "st.global.b16" ;
  check_has ptx "mul.wide.s32" ;
  (* 2-byte stride: the element size operand of the address *)
  check_has ptx ", 2;" ;
  check_absent ptx "ld.global.f16" ;
  check_absent ptx "mul.f16" ;
  (* The product stays an f32 op, unchanged from any other f32 kernel: the
     fp-contraction behaviour of f32 code is not touched by f16. *)
  check_has ptx "mul.f32"

let test_chain_two_narrowings () =
  let ptx = gen (K.chain ()) in
  let count needle =
    let n = String.length needle in
    let rec go i acc =
      if i + n > String.length ptx then acc
      else go (i + 1) (if String.sub ptx i n = needle then acc + 1 else acc)
    in
    go 0 0
  in
  Alcotest.(check int)
    "two narrowings: the intermediate f16 is not elided"
    2
    (count "cvt.rn.f16.f32") ;
  Alcotest.(check int) "two widenings" 2 (count "cvt.f32.f16")

let test_roundtrip_has_no_conversion () =
  let ptx = gen (K.roundtrip ()) in
  check_has ptx "ld.global.nc.b16" ;
  check_has ptx "st.global.b16" ;
  check_absent ptx "cvt.f32.f16" ;
  check_absent ptx "cvt.rn.f16.f32"

let test_local_uses_mov_b16 () =
  let ptx = gen (K.local_var ()) in
  check_has ptx "mov.b16 %h" ;
  check_absent ptx "mov.u32 %h" ;
  check_absent ptx "mov.f32 %h"

let test_shared_array () =
  let ptx = gen (K.shared_array ()) in
  check_has ptx ".shared .align 2 .b16 sh[256];" ;
  check_has ptx "st.shared.b16" ;
  check_has ptx "ld.shared.b16" ;
  (* shared arrays use 32-bit addresses: shl.b32 by log2(2) *)
  check_has ptx "shl.b32"

let test_local_array () =
  let ptx = gen (K.local_array ()) in
  check_has ptx ".local .align 2 .b16 la[4];" ;
  check_has ptx "st.local.b16" ;
  check_has ptx "ld.local.b16"

let test_casts () =
  let ptx = gen (K.int_and_double_casts ()) in
  (* f16 -> int32: widen exactly, then truncate toward zero like the C cast. *)
  check_has ptx "cvt.f32.f16" ;
  check_has ptx "cvt.rzi.s32.f32" ;
  (* int32 -> f16: the f32 conversion (RN), then the f16 narrowing (RN). *)
  check_has ptx "cvt.rn.f32.s32" ;
  check_has ptx "cvt.rn.f16.f32" ;
  (* f16 <-> f64 go through f32 as well: f64 -> f32 -> f16 is the interpreter's
     definition (Sarek_float16.to_float16 of the f32 value). *)
  check_has ptx "cvt.f64.f32" ;
  check_has ptx "cvt.rn.f32.f64" ;
  check_absent ptx "cvt.rn.f16.f64" ;
  check_absent ptx "cvt.f64.f16"

let test_select () =
  let ptx = gen (K.select_values ()) in
  check_has ptx "selp.b16 %h" ;
  let ptx = gen (K.select_loads ()) in
  check_has ptx "mov.b16 %h" ;
  check_absent ptx "selp.u32 %h" ;
  check_absent ptx "mov.u32 %h"

let test_helper () =
  let ptx = gen (K.helper_roundtrip ()) in
  check_has ptx "mov.b16 %h" ;
  check_has ptx "cvt.rn.f16.f32" ;
  check_absent ptx ".func"

let test_param_is_a_plain_pointer () =
  let ptx = gen (K.scale ()) in
  check_has ptx ".param .u64 param_inp" ;
  check_absent ptx ".param .b16" ;
  check_absent ptx ".param .f16"

let test_no_native_f16_arithmetic () =
  List.iter (fun (name, k) -> assert_no_native_f16 name (gen k)) (K.all ())

(** The same kernel text is valid on every target: only the [.target] line
    changes, so nothing in an f16 kernel depends on sm_53+. *)
let test_target_independent () =
  let targets = ["sm_50"; "sm_52"; "sm_61"; "sm_75"; "sm_86"; "sm_90"] in
  let drop_target ptx =
    String.concat
      "\n"
      (List.filter (fun l -> not (contains ~needle:".target" l)) (lines ptx))
  in
  List.iter
    (fun (name, k) ->
      let reference = drop_target (gen ~sm_target:"sm_86" k) in
      List.iter
        (fun t ->
          let ptx = gen ~sm_target:t k in
          check_has ptx (".target " ^ t) ;
          assert_no_native_f16 (name ^ "@" ^ t) ptx ;
          Alcotest.(check string)
            (name ^ " body is the same at " ^ t)
            reference
            (drop_target ptx))
        targets)
    (K.all ())

(* ---- the register-class decoder ------------------------------------------ *)

let test_reg_class () =
  let open Sarek_ir_ptx_types in
  let cls = function
    | RU32 -> "u32"
    | RU64 -> "u64"
    | RF16 -> "f16"
    | RF32 -> "f32"
    | RF64 -> "f64"
  in
  List.iter
    (fun (r, want) -> Alcotest.(check string) r want (cls (reg_class r)))
    [
      ("%r0", "u32");
      ("%r17", "u32");
      ("%rd3", "u64");
      ("%h0", "f16");
      ("%h123", "f16");
      ("%f2", "f32");
      ("%fd9", "f64");
    ] ;
  let alloc = make_alloc () in
  Alcotest.(check string) "fresh f16 register" "%h0" (new_f16 alloc) ;
  Alcotest.(check string) "next" "%h1" (new_f16 alloc) ;
  List.iter
    (fun bad ->
      match reg_class bad with
      | exception Ptx_codegen_error _ -> ()
      | c -> Alcotest.failf "%S must not classify (got %s)" bad (cls c))
    ["%p0"; "%x1"; "r1"; "0"; ""; "%"]

(* ---- loud failures: IR that needs f16 arithmetic -------------------------- *)

let expect_error what ~needle k =
  match gen k with
  | ptx ->
      Alcotest.failf "%s: expected a Ptx_codegen_error, emitted:\n%s" what ptx
  | exception Sarek_ir_ptx_types.Ptx_codegen_error msg ->
      if not (contains ~needle msg) then
        Alcotest.failf "%s: unexpected diagnostic: %s" what msg

let f16_read i = EArrayRead ("inp", i)

let store_into_out e = K.with_gid (fun i -> SAssign (LArrayElem ("out", i), e))

let h_params = [K.vec_param "out" TFloat16; K.vec_param "inp" TFloat16]

let test_f16_arithmetic_refused () =
  List.iter
    (fun (name, op) ->
      expect_error
        ("f16 " ^ name)
        ~needle:"float16"
        (K.kernel
           "f16_arith"
           h_params
           (store_into_out (op (f16_read (EConst (CInt32 0l)))))))
    [
      ("add", fun a -> EBinop (Add, a, a));
      ("mul", fun a -> EBinop (Mul, a, a));
      ("compare", fun a -> ECast (TFloat16, EBinop (Lt, a, a)));
      ("neg", fun a -> EUnop (Neg, a));
      ( "intrinsic",
        fun a -> EIntrinsic (["Sarek_stdlib"; "Float32"], "sqrt", [a]) );
    ]

let test_mismatched_stores_refused () =
  (* f32 value into an f16 array: needs the explicit narrowing. *)
  expect_error
    "f32 into f16 array"
    ~needle:"not a float16 value"
    (K.kernel "bad_store" h_params (store_into_out (EConst (CFloat32 1.0)))) ;
  (* f16 value into an f32 array: needs the explicit widening. *)
  expect_error
    "f16 into f32 array"
    ~needle:"float16"
    (K.kernel
       "bad_store2"
       [K.vec_param "out" TFloat32; K.vec_param "inp" TFloat16]
       (store_into_out (f16_read (EConst (CInt32 0l)))))

(** An f16 value in a condition or loop-bound position would reach [setp.*.u32]
    on a 16-bit register; each is a located refusal. *)
let test_f16_control_positions_refused () =
  let h = f16_read (EConst (CInt32 0l)) in
  let nop = SAssign (LArrayElem ("out", EConst (CInt32 0l)), h) in
  let k body = K.kernel "f16_ctl" h_params body in
  let loop_var = K.var "i" TInt32 in
  List.iter
    (fun (what, needle, body) -> expect_error what ~needle (k body))
    [
      ("SIf condition", "if condition", SIf (h, nop, None));
      ("SWhile condition", "while condition", SWhile (h, nop));
      ( "SFor lower bound",
        "for-loop lower bound",
        SFor (loop_var, h, EConst (CInt32 4l), Upto, SEmpty) );
      ( "SFor upper bound",
        "for-loop upper bound",
        SFor (loop_var, EConst (CInt32 0l), h, Upto, SEmpty) );
      ( "EIf condition",
        "if condition",
        K.with_gid (fun i ->
            SAssign
              ( LArrayElem ("out", i),
                EIf (h, f16_read i, f16_read (EConst (CInt32 1l))) )) );
    ]

(** A select whose branches disagree on f16-ness must not emit [selp.b16] on a
    32-bit operand. Both orders, with the select in the eager and in the
    branching (array read in a branch) form. *)
let test_f16_select_mismatch_refused () =
  let f32_const = EConst (CFloat32 1.0) in
  let c = EBinop (Lt, EConst (CInt32 0l), EConst (CInt32 1l)) in
  List.iter
    (fun (what, e) ->
      expect_error
        what
        ~needle:"float16"
        (K.kernel
           "f16_sel"
           [K.vec_param "out" TFloat16; K.vec_param "inp" TFloat16]
           (store_into_out e)))
    [
      ("f16 then, f32 else", EIf (c, K.narrow f32_const, f32_const));
      ("f32 then, f16 else", EIf (c, f32_const, K.narrow f32_const));
      ("branching form", EIf (c, f16_read (EConst (CInt32 0l)), f32_const));
    ]

(** Aggregate field stores take no f16 value, and an f16 leaf is named as such
    rather than as a "non-scalar" type. *)
let test_field_store_f16 () =
  let buf = Buffer.create 16 in
  let refuses what ~needle f =
    match f () with
    | () -> Alcotest.failf "%s: expected a refusal" what
    | exception Sarek_ir_ptx_types.Ptx_codegen_error msg ->
        if not (contains ~needle msg) then
          Alcotest.failf "%s: unexpected diagnostic: %s" what msg
  in
  refuses "f16 value into f32 field" ~needle:"float16" (fun () ->
      Sarek_ir_ptx_mem.emit_field_store buf "%rd0" ~offset:0 TFloat32 "%h0") ;
  refuses "f16 leaf type" ~needle:"float16 vector" (fun () ->
      Sarek_ir_ptx_mem.emit_field_store buf "%rd0" ~offset:0 TFloat16 "%h0")

let test_f16_index_refused () =
  expect_error
    "f16 as index"
    ~needle:"index"
    (K.kernel
       "f16_index"
       h_params
       (store_into_out (EArrayRead ("inp", f16_read (EConst (CInt32 0l))))))

let test_scalar_f16_param_refused () =
  expect_error
    "scalar f16 parameter"
    ~needle:"scalar float16"
    (K.kernel "scalar_f16" [DParam (K.var "h" TFloat16, None)] SEmpty)

let test_ptx_cast_totality () =
  (* Every (source, destination) pair of scalar types casts without raising: no
     pair can fall into a class default branch, and every f16 pair names the
     conversion. *)
  let alloc = Sarek_ir_ptx_types.make_alloc () in
  let buf = Buffer.create 64 in
  let reg_of = function
    | TInt32 | TBool -> Sarek_ir_ptx_types.new_u32 alloc
    | TInt64 -> Sarek_ir_ptx_types.new_u64 alloc
    | TFloat16 -> Sarek_ir_ptx_types.new_f16 alloc
    | TFloat32 -> Sarek_ir_ptx_types.new_f32 alloc
    | TFloat64 -> Sarek_ir_ptx_types.new_f64 alloc
    | _ -> assert false
  in
  let tys = [TInt32; TInt64; TFloat16; TFloat32; TFloat64; TBool] in
  List.iter
    (fun src ->
      List.iter
        (fun dst ->
          let r_src = reg_of src in
          let r = Sarek_ir_ptx_expr.emit_cast buf alloc r_src dst in
          let want_f16 = dst = TFloat16 in
          Alcotest.(check bool)
            (Printf.sprintf
               "%s -> %s result class"
               (Sarek_ir_pp.string_of_elttype src)
               (Sarek_ir_pp.string_of_elttype dst))
            want_f16
            (Sarek_ir_ptx_types.is_f16_reg r))
        tys)
    tys ;
  assert_no_native_f16 "cast matrix" (Buffer.contents buf)

let () =
  Alcotest.run
    "ptx_f16"
    [
      ( "emission",
        [
          Alcotest.test_case
            "scale: load, widen, f32 mul, narrow, store"
            `Quick
            test_scale_markers;
          Alcotest.test_case
            "chain: two narrowings kept"
            `Quick
            test_chain_two_narrowings;
          Alcotest.test_case
            "roundtrip: raw 16-bit copy"
            `Quick
            test_roundtrip_has_no_conversion;
          Alcotest.test_case "local: mov.b16" `Quick test_local_uses_mov_b16;
          Alcotest.test_case "shared array" `Quick test_shared_array;
          Alcotest.test_case "local array" `Quick test_local_array;
          Alcotest.test_case "int and double casts" `Quick test_casts;
          Alcotest.test_case "selects" `Quick test_select;
          Alcotest.test_case "inlined helper" `Quick test_helper;
          Alcotest.test_case
            "pointer parameter"
            `Quick
            test_param_is_a_plain_pointer;
          Alcotest.test_case
            "cast matrix is total"
            `Quick
            test_ptx_cast_totality;
        ] );
      ( "sm support",
        [
          Alcotest.test_case
            "no native f16 arithmetic"
            `Quick
            test_no_native_f16_arithmetic;
          Alcotest.test_case
            "identical body on sm_50..sm_90"
            `Quick
            test_target_independent;
        ] );
      ("register classes", [Alcotest.test_case "decoder" `Quick test_reg_class]);
      ( "refusals",
        [
          Alcotest.test_case "f16 arithmetic" `Quick test_f16_arithmetic_refused;
          Alcotest.test_case
            "mismatched stores"
            `Quick
            test_mismatched_stores_refused;
          Alcotest.test_case "f16 index" `Quick test_f16_index_refused;
          Alcotest.test_case
            "f16 control positions"
            `Quick
            test_f16_control_positions_refused;
          Alcotest.test_case
            "f16 select mismatch"
            `Quick
            test_f16_select_mismatch_refused;
          Alcotest.test_case "f16 aggregate field" `Quick test_field_store_f16;
          Alcotest.test_case
            "scalar f16 parameter"
            `Quick
            test_scalar_f16_param_refused;
        ] );
    ]
