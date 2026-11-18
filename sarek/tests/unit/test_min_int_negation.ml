(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Negating min_int, and the int32 min_int spelling, without a GPU.
 *
 * OCaml negation wraps: [Int32.neg Int32.min_int = Int32.min_int] and
 * [Int64.neg Int64.min_int = Int64.min_int]. The Interpreter and Native follow
 * OCaml. The source backends used to print [-(min_int)] as unary minus applied
 * to the min_int spelling, [(-(-9223372036854775807LL - 1LL))], a signed
 * overflow in a constant expression (undefined in C, CUDA C, OpenCL C and
 * MSL). They now fold a negated integer literal to the wrapped constant; GLSL
 * and WGSL, which wrap, fold the same way so that all printers agree.
 *
 * 1. Interpreter and Native: [-(min_int)] is min_int for int32 and int64, and
 *    the kernel IR does hold [EUnop (Neg, EConst min_int)].
 * 2. CUDA, OpenCL, Metal, GLSL and WGSL print [-(min_int)] and [-(-(min_int))]
 *    as the min_int literal, with no negation of it.
 * 3. GLSL spells [Int32.min_int] like the other C-family printers.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
open Sarek
module Vector = Spoc_core.Vector
module Device = Spoc_core.Device

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

(* The type constraint keeps the literal a separate node: OCaml folds
   [-<literal>] (with or without parentheses) into one negative constant. *)
let kernel64 =
  [%kernel
    fun (out : int64 vector) ->
      let i = thread_idx_x in
      if i = 0l then out.(0) <- -(0x8000000000000000L : int64)]

let kernel32 =
  [%kernel
    fun (out : int32 vector) ->
      let i = thread_idx_x in
      if i = 0l then out.(0) <- -(-2147483648l : int32)]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(** {1 IR} *)

let rec has_neg_stmt target = function
  | SAssign (_, e) -> has_neg_expr target e
  | SSeq l -> List.exists (has_neg_stmt target) l
  | SIf (c, t, f) ->
      has_neg_expr target c || has_neg_stmt target t
      || Option.fold ~none:false ~some:(has_neg_stmt target) f
  | SLet (_, e, s) | SLetMut (_, e, s) ->
      has_neg_expr target e || has_neg_stmt target s
  | _ -> false

and has_neg_expr target = function
  | EUnop (Neg, EConst c) when c = target -> true
  | EBinop (_, a, b) -> has_neg_expr target a || has_neg_expr target b
  | EUnop (_, a) -> has_neg_expr target a
  | _ -> false

let test_ir () =
  Alcotest.(check bool)
    "int64 kernel negates the min_int literal"
    true
    (has_neg_stmt (CInt64 Int64.min_int) (ir_of kernel64).kern_body) ;
  Alcotest.(check bool)
    "int32 kernel negates the min_int literal"
    true
    (has_neg_stmt (CInt32 Int32.min_int) (ir_of kernel32).kern_body)

(** {1 CPU backends} *)

let cpu_devices =
  lazy
    (Sarek_interpreter.Interpreter_plugin.init () ;
     Sarek_native.Native_plugin.init () ;
     Device.init ~frameworks:["Interpreter"; "Native"] ())

let device framework =
  lazy
    (match
       Array.find_opt
         (fun d -> d.Device.framework = framework)
         (Lazy.force cpu_devices)
     with
    | Some d -> d
    | None -> Alcotest.failf "no %s device" framework)

let run_kernel device kernel out =
  Execute.run_vectors
    ~device
    ~ir:(ir_of kernel)
    ~args:[Execute.Vec out]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    () ;
  Spoc_core.Transfer.flush device

let run_on framework () =
  let device = Lazy.force (device framework) in
  let out64 = Vector.create Vector.int64 1 in
  Vector.set out64 0 0L ;
  run_kernel device kernel64 out64 ;
  Alcotest.(check int64)
    "int64: -(min_int) wraps like Int64.neg"
    (Int64.neg Int64.min_int)
    (Vector.get out64 0) ;
  let out32 = Vector.create Vector.int32 1 in
  Vector.set out32 0 0l ;
  run_kernel device kernel32 out32 ;
  Alcotest.(check int32)
    "int32: -(min_int) wraps like Int32.neg"
    (Int32.neg Int32.min_int)
    (Vector.get out32 0)

(** {1 Generated source} *)

let var name id ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

(* [out[0] = value] for an [out] of element type [ty]. *)
let store_kernel ty value : kernel =
  {
    default_kernel with
    kern_name = "min_int_negation";
    kern_params =
      [
        DParam
          (var "out" 0 (TVec ty), Some {arr_elttype = ty; arr_memspace = Global});
      ];
    kern_body = SAssign (LArrayElem ("out", EConst (CInt32 0l)), value);
  }

let neg e = EUnop (Neg, e)

let min32 = EConst (CInt32 Int32.min_int)

let min64 = EConst (CInt64 Int64.min_int)

(* Each case: backend, printer, kernel, the whole right-hand side the store
   must have. *)
let cases : (string * (kernel -> string) * kernel * string) list =
  let c_family name gen suffix =
    let lit64 = Printf.sprintf "= (-9223372036854775807%s - 1%s);" suffix suffix
    and lit32 = "= (-2147483647 - 1);" in
    [
      (name ^ " int64 -min", gen, store_kernel TInt64 (neg min64), lit64);
      (name ^ " int64 --min", gen, store_kernel TInt64 (neg (neg min64)), lit64);
      (name ^ " int32 -min", gen, store_kernel TInt32 (neg min32), lit32);
      (name ^ " int32 --min", gen, store_kernel TInt32 (neg (neg min32)), lit32);
      ( name ^ " int32 -5",
        gen,
        store_kernel TInt32 (neg (EConst (CInt32 5l))),
        "= -5;" );
    ]
  in
  c_family "CUDA" Sarek_ir_cuda.generate "LL"
  @ c_family "OpenCL" Sarek_ir_opencl.generate "L"
  @ c_family "Metal" Sarek_ir_metal.generate "L"
  @ c_family "GLSL" (fun k -> Sarek_ir_glsl.generate k) "L"
  @
  let wgsl k = Sarek_ir_wgsl.generate k in
  [
    ( "WGSL int32 -min",
      wgsl,
      store_kernel TInt32 (neg min32),
      "(-2147483647i - 1i);" );
    ( "WGSL int32 --min",
      wgsl,
      store_kernel TInt32 (neg (neg min32)),
      "(-2147483647i - 1i);" );
    ( "WGSL int32 -5",
      wgsl,
      store_kernel TInt32 (neg (EConst (CInt32 5l))),
      "-5i;" );
  ]

let test_case (name, gen, k, spelling) () =
  let src = gen k in
  Alcotest.(check bool)
    (Printf.sprintf "%s emits %s" name spelling)
    true
    (contains src spelling) ;
  List.iter
    (fun negated ->
      Alcotest.(check bool)
        (Printf.sprintf "%s does not negate a literal (%s)" name negated)
        false
        (contains src negated))
    ["(-("; "(- ("; "(-5"; "(- -"]

(* GLSL used [Int32.to_string] for int32 literals, printing min_int as
   [-2147483648] while CUDA, OpenCL and Metal print [(-2147483647 - 1)]. *)
let test_glsl_int32_min () =
  let src = Sarek_ir_glsl.generate (store_kernel TInt32 min32) in
  Alcotest.(check bool)
    "GLSL spells int32 min_int as (-2147483647 - 1)"
    true
    (contains src "= (-2147483647 - 1);") ;
  Alcotest.(check bool)
    "GLSL never spells int32 min_int as -2147483648"
    false
    (contains src "-2147483648")

let () =
  Alcotest.run
    "min_int_negation"
    [
      ("ir", [Alcotest.test_case "Neg of min_int literal" `Quick test_ir]);
      ( "cpu",
        [
          Alcotest.test_case "Interpreter" `Quick (run_on "Interpreter");
          Alcotest.test_case "Native" `Quick (run_on "Native");
        ] );
      ( "source",
        Alcotest.test_case "GLSL int32 min_int" `Quick test_glsl_int32_min
        :: List.map
             (fun ((name, _, _, _) as c) ->
               Alcotest.test_case name `Quick (test_case c))
             cases );
    ]
