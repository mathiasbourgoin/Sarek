(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * int64 literals keep all 64 bits (issue #550), without a GPU.
 *
 * The PPX used to quote an int64 constant as [Int64.of_int <int literal>],
 * through a 63-bit OCaml int: a literal whose two top bits differ lost bit 63
 * (0x4000000000000000L became 0xc000000000000000L). Both the IR quoted for the
 * device backends and the OCaml code generated for the Native backend went
 * through that path.
 *
 * 1. The kernel IR carries the literals unchanged.
 * 2. The kernel writes the literals unchanged on the Interpreter (quoted IR)
 *    and on Native (PPX-generated OCaml).
 * 3. Each C-family source backend, GLSL and PTX print a constant that their
 *    language reads back as the same 64-bit value: [Int64.min_int] is not
 *    spelled [-9223372036854775808], which C, OpenCL C and MSL read as unary
 *    minus applied to an unsigned literal.
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

let kernel =
  [%kernel
    fun (out : int64 vector) ->
      let i = thread_idx_x in
      if i = 0l then begin
        out.(0) <- 0x3fffffffffffffffL ;
        out.(1) <- 0x4000000000000000L ;
        out.(2) <- 0x5be0cd19137e2179L ;
        out.(3) <- 0x7fffffffffffffffL ;
        out.(4) <- 0x8000000000000000L ;
        out.(5) <- 0xbfffffffffffffffL ;
        out.(6) <- 0xc000000000000000L ;
        out.(7) <- 0xffffffffffffffffL ;
        out.(8) <- (0x5be0cd19L lsl 32L) lor 0x137e2179L
      end]

let expected =
  [|
    0x3fffffffffffffffL;
    0x4000000000000000L;
    0x5be0cd19137e2179L;
    0x7fffffffffffffffL;
    0x8000000000000000L;
    0xbfffffffffffffffL;
    0xc000000000000000L;
    0xffffffffffffffffL;
    0x5be0cd19137e2179L;
  |]

let ir =
  let _, kirc = kernel in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let hex = Alcotest.testable (fun fmt v -> Format.fprintf fmt "0x%016Lx" v) ( = )

(** {1 IR} *)

let rec int64_consts_stmt acc = function
  | SAssign (_, e) -> int64_consts_expr acc e
  | SSeq l -> List.fold_left int64_consts_stmt acc l
  | SIf (c, t, f) ->
      let acc = int64_consts_stmt (int64_consts_expr acc c) t in
      Option.fold ~none:acc ~some:(int64_consts_stmt acc) f
  | SLet (_, e, s) | SLetMut (_, e, s) ->
      int64_consts_stmt (int64_consts_expr acc e) s
  | _ -> acc

and int64_consts_expr acc = function
  | EConst (CInt64 n) -> n :: acc
  | EBinop (_, a, b) -> int64_consts_expr (int64_consts_expr acc a) b
  | EUnop (_, a) -> int64_consts_expr acc a
  | _ -> acc

let test_ir_constants () =
  let consts = int64_consts_stmt [] ir.kern_body in
  Array.iteri
    (fun i v ->
      if i < 8 then
        Alcotest.(check bool)
          (Printf.sprintf "IR holds 0x%016Lx" v)
          true
          (List.mem v consts))
    expected

(** {1 CPU backends} *)

(* [Device.init] enumerates once per process, so both CPU frameworks are
   registered and enumerated together. *)
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

let interpreter = device "Interpreter"

let native = device "Native"

let run_on dev () =
  let device = Lazy.force dev in
  let out = Vector.create Vector.int64 9 in
  for i = 0 to 8 do
    Vector.set out i 0L
  done ;
  Execute.run_vectors
    ~device
    ~ir
    ~args:[Execute.Vec out]
    ~block:(Execute.dims1d 1)
    ~grid:(Execute.dims1d 1)
    () ;
  Spoc_core.Transfer.flush device ;
  Array.iteri
    (fun i v ->
      Alcotest.check hex (Printf.sprintf "out.(%d)" i) v (Vector.get out i))
    expected

(** {1 Generated source} *)

let var name id ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

let literal_kernel values : kernel =
  {
    default_kernel with
    kern_name = "int64_literals";
    kern_params =
      [
        DParam
          ( var "out" 0 (TVec TInt64),
            Some {arr_elttype = TInt64; arr_memspace = Global} );
      ];
    kern_body =
      SSeq
        (List.mapi
           (fun i v ->
             SAssign
               ( LArrayElem ("out", EConst (CInt32 (Int32.of_int i))),
                 EConst (CInt64 v) ))
           values);
  }

let source_values =
  [
    0x4000000000000000L;
    0x7fffffffffffffffL;
    0x8000000000000000L;
    0xbfffffffffffffffL;
  ]

(* The spelling each backend must emit for [source_values], in order. *)
let backends : (string * (kernel -> string) * string list) list =
  let c suffix =
    [
      "4611686018427387904" ^ suffix;
      "9223372036854775807" ^ suffix;
      Printf.sprintf "(-9223372036854775807%s - 1%s)" suffix suffix;
      "-4611686018427387905" ^ suffix;
    ]
  in
  [
    ("CUDA", Sarek_ir_cuda.generate, c "LL");
    ("OpenCL", Sarek_ir_opencl.generate, c "L");
    ("Metal", Sarek_ir_metal.generate, c "L");
    ("GLSL", (fun k -> Sarek_ir_glsl.generate k), c "L");
    ( "PTX",
      (fun k -> Sarek_ir_ptx.generate k),
      [
        ", 4611686018427387904;";
        ", 9223372036854775807;";
        ", 0x8000000000000000;";
        ", -4611686018427387905;";
      ] );
  ]

let test_source (name, gen, spellings) () =
  let src = gen (literal_kernel source_values) in
  List.iter2
    (fun v s ->
      Alcotest.(check bool)
        (Printf.sprintf "%s spells 0x%016Lx as %s" name v s)
        true
        (contains src s))
    source_values
    spellings ;
  Alcotest.(check bool)
    (name ^ " never spells min_int as one negative literal")
    false
    (contains src "-9223372036854775808")

let () =
  Alcotest.run
    "int64_literals"
    [
      ("ir", [Alcotest.test_case "constants" `Quick test_ir_constants]);
      ( "cpu",
        [
          Alcotest.test_case "Interpreter" `Quick (run_on interpreter);
          Alcotest.test_case "Native" `Quick (run_on native);
        ] );
      ( "source",
        List.map
          (fun ((name, _, _) as b) ->
            Alcotest.test_case name `Quick (test_source b))
          backends );
    ]
