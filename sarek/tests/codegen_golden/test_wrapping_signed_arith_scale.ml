(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Wrapping signed integer arithmetic (Sarek_ir_c_arith): typing scopes and
 * cost.
 *
 * 1. Scopes: array element types are those of the function being emitted
 *    (a helper's [buf] does not hide the kernel's); variant payloads read
 *    through [v.data.C_v] are typed; the [~types] table the struct
 *    declarations come from types record fields.
 * 2. Cost, counted in typed nodes (Sarek_ir_c_arith.typing_steps), not in
 *    time: emitting a chain, many statements and a long chain of lets types a
 *    number of nodes linear in their size.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
open Wrapping_support

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

(* ---- 1. scopes ---- *)

(* The kernel's [buf] is int32, the helper's float32. *)
let test_helper_scope () =
  let buf_k = var "buf" (TVec TInt32) and buf_h = var "buf" (TVec TFloat32) in
  let helper =
    {
      hf_name = "h";
      hf_params = [buf_h];
      hf_ret_type = TFloat32;
      hf_body =
        SReturn (EBinop (Add, EArrayRead ("buf", i32 0), EConst (CFloat32 1.0)));
    }
  in
  let k =
    {
      default_kernel with
      kern_name = "scope";
      kern_params =
        [
          DParam (buf_k, Some {arr_elttype = TInt32; arr_memspace = Global});
          vec "out" TInt32;
        ];
      kern_funcs = [helper];
      kern_body =
        SAssign
          ( LArrayElem ("out", i32 0),
            EBinop (Add, EArrayRead ("buf", i32 0), i32 1) );
    }
  in
  List.iter
    (fun d ->
      let src = generate d k in
      check_has
        d.name
        src
        (Printf.sprintf
           "out[0] = %s;"
           (d.w32 (Printf.sprintf "((%s)buf[0] + 1u)" d.u32))) ;
      check_has d.name src "return (buf[0] + 1.0f);")
    dialects

let variant =
  ("shape", [("A", [TInt32]); ("B", [TFloat32]); ("C", [TInt32; TInt64])])

let payload constr fields =
  let scrut = EVar (var "s" (TVariant (fst variant, []))) in
  List.fold_left
    (fun e f -> ERecordField (e, f))
    scrut
    ("data" :: (constr ^ "_v") :: fields)

let test_variant_payloads () =
  let k = {default_kernel with kern_variants = [variant]} in
  let env = C_arith.env_of_kernel k in
  let check label expected e =
    Alcotest.(check bool) label true (C_arith.int_width env e = expected)
  in
  check "one int32 payload" (Some C_arith.W32) (payload "A" []) ;
  check "one float payload" None (payload "B" []) ;
  check "second of two payloads" (Some C_arith.W64) (payload "C" ["_1"]) ;
  Alcotest.(check bool)
    "wrapping width of A + 1"
    true
    (C_arith.binop_width env Add (payload "A" []) (i32 1) = Some C_arith.W32)

(* The struct is declared from [~types] (px : float32); a conflicting
   [kern_types] entry (px : int32) must not make [r.px + r.px] wrap. *)
let test_types_table () =
  let r = var "r" (TRecord ("pt", [])) in
  let k =
    {
      default_kernel with
      kern_name = "records";
      kern_params = [vec "out" TFloat32; DParam (r, None)];
      kern_types = [("pt", [("px", TInt32)])];
      kern_body =
        SAssign
          ( LArrayElem ("out", i32 0),
            EBinop (Add, ERecordField (ev r, "px"), ERecordField (ev r, "px"))
          );
    }
  in
  let src =
    Sarek_ir_opencl.generate_with_types ~types:[("pt", [("px", TFloat32)])] k
  in
  check_has "OpenCL" src "out[0] = (r.px + r.px);"

(* ---- 2. cost ---- *)

(* Typed nodes for one emission of [e] with a fresh environment. *)
let steps_of_expr e =
  let env = C_arith.empty () in
  ignore (cuda.expr env e) ;
  C_arith.typing_steps env

(* Linear: doubling the size at most doubles the count (plus a constant). *)
let check_linear label steps n =
  let s1 = steps n and s2 = steps (2 * n) in
  if s2 > (2 * s1) + 16 then
    Alcotest.failf
      "%s: %d typed nodes at size %d, %d at %d"
      label
      s1
      n
      s2
      (2 * n)

let test_chain_cost () =
  let rec chain n e =
    if n = 0 then e
    else
      chain (n - 1) (EUnop (Neg, EBinop (Shl, EBinop (Add, e, i32 1), i32 1)))
  in
  check_linear "nested -, lsl, +" (fun n -> steps_of_expr (chain n (ev x))) 500 ;
  let rec sum n e = if n = 0 then e else sum (n - 1) (EBinop (Add, e, i32 1)) in
  check_linear "left-deep sum" (fun n -> steps_of_expr (sum n (ev x))) 1000

(* Arithmetic under two non-arithmetic levels at every step: the emitter
   prints the levels in between with its own arms, and each node must still
   be typed once (the typing travels with C_arith.visit). *)
let test_nested_wrappers_cost () =
  let nest wrap n =
    let rec go n e =
      if n = 0 then e else go (n - 1) (EBinop (Add, ev x, wrap e))
    in
    go n (ev x)
  in
  let cast_of_cast e = ECast (TInt32, ECast (TFloat32, e)) in
  let field_of_field e =
    ERecordField
      ( ERecordField
          ( ERecord ("outer", [("inner", ERecord ("inner", [("v", e)]))]),
            "inner" ),
        "v" )
  in
  let call_in_field e =
    ERecordField
      (ERecord ("box", [("v", EApp (EVar (var "id" TInt32), [e]))]), "v")
  in
  List.iter
    (fun (label, wrap) ->
      check_linear label (fun n -> steps_of_expr (nest wrap n)) 400)
    [
      ("cast of a cast", cast_of_cast);
      ("field of a field", field_of_field);
      ("call in a field", call_in_field);
    ]

let kernel_of body =
  {
    default_kernel with
    kern_name = "k";
    kern_params = [vec "out" TInt32; DParam (x, None)];
    kern_body = body;
  }

(* Typed nodes for building the environment and emitting the body. *)
let steps_of_kernel k =
  let env = C_arith.env_of_kernel k in
  let st = {(Sarek_ir_cuda.state k) with arith = env} in
  Sarek_ir_cuda.gen_stmt st (Buffer.create 4096) "" k.kern_body ;
  (env, C_arith.typing_steps env)

let test_statements_cost () =
  let same_shape n =
    kernel_of
      (SSeq
         (List.init n (fun i ->
              SAssign (LArrayElem ("out", i32 i), EBinop (Add, ev x, i32 1)))))
  in
  check_linear
    "repeated statements"
    (fun n -> snd (steps_of_kernel (same_shape n)))
    2000

(* [a1 = x + 1; a2 = a1 + 1; ...; out.(...) <- an]: as a stored value, every
   let is excluded (one by one, through the worklist); as a subscript, every
   let is an index let. *)
let let_chain ~as_index n =
  let a i = {(var (Printf.sprintf "a%d" i) TInt32) with var_id = i} in
  let last = EVar (a n) in
  let rec build i =
    if i > n then
      if as_index then SAssign (LArrayElem ("out", last), i32 0)
      else SAssign (LArrayElem ("out", i32 0), last)
    else
      let prev = if i = 1 then ev x else EVar (a (i - 1)) in
      SLet (a i, EBinop (Add, prev, i32 1), build (i + 1))
  in
  (kernel_of (build 1), a)

let test_let_chain_cost () =
  List.iter
    (fun as_index ->
      let k, a = let_chain ~as_index 50 in
      let env, _ = steps_of_kernel k in
      Alcotest.(check bool)
        (Printf.sprintf "a1 is an index let = %b" as_index)
        as_index
        (C_arith.is_index_let env (a 1)) ;
      check_linear
        (Printf.sprintf "let chain (index %b)" as_index)
        (fun n -> snd (steps_of_kernel (fst (let_chain ~as_index n))))
        800)
    [false; true]

(* The propagation itself: worklist pops and dependency visits grow linearly
   with the number of lets (a chain whose end is a stored value excludes every
   let, one through the other). *)
let test_index_propagation () =
  let steps n =
    let env = C_arith.env_of_kernel (fst (let_chain ~as_index:false n)) in
    C_arith.index_steps env
  in
  Alcotest.(check bool) "some propagation" true (steps 100 >= 100) ;
  check_linear "index propagation" steps 1000

let () =
  Alcotest.run
    "wrapping_signed_arith_scale"
    [
      ( "scopes",
        [
          Alcotest.test_case "helper arrays" `Quick test_helper_scope;
          Alcotest.test_case "variant payloads" `Quick test_variant_payloads;
          Alcotest.test_case "types table" `Quick test_types_table;
        ] );
      ( "cost",
        [
          Alcotest.test_case "chains" `Quick test_chain_cost;
          Alcotest.test_case "nested wrappers" `Quick test_nested_wrappers_cost;
          Alcotest.test_case "statements" `Quick test_statements_cost;
          Alcotest.test_case "let chains" `Quick test_let_chain_cost;
          Alcotest.test_case "index propagation" `Quick test_index_propagation;
        ] );
    ]
