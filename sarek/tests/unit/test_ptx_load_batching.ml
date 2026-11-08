(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The PTX generator's memory-level parallelism: unrolling of innermost
    runtime-trip loops ([Sarek_ir_ptx_unroll]), [ld.global.nc] for the vector
    parameters a kernel never writes ([Sarek_ir_ptx_readonly]) and one
    [mul.wide.s32] per global address. Device-free: text checks on the PTX, and
    the Interpreter as the oracle that the unrolled IR computes the very same
    values as the loop it replaces. *)

open Sarek_ir_types
open Sarek_codegen
module Interp = Sarek.Sarek_ir_interp

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let count s sub =
  let n = String.length s and m = String.length sub in
  let rec go i acc =
    if i + m > n then acc
    else if String.sub s i m = sub then go (i + m) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let contains s sub = count s sub > 0

let ptx_of ir = Sarek_ir_ptx.generate ~sm_target:"sm_75" ~block:(256, 1, 1) ir

(******************************************************************************
 * Kernels
 ******************************************************************************)

(* The streaming part of a matvec: runtime trip count, two independent global
   loads per iteration. *)
let dot_while =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      let k = mut tid in
      while k < n do
        acc := acc +. (w.((block_idx_x * n) + k) *. x.(k)) ;
        k := k + 256l
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

(* The same with [<=]. *)
let dot_while_le =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      let k = mut tid in
      while k <= n do
        acc := acc +. (w.((block_idx_x * (n + 1l)) + k) *. x.(k)) ;
        k := k + 3l
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

(* A quantised-weight matvec shape: per iteration one packed word of four 8-bit
   quants, one scale and four activations (six global loads). *)
let quantised =
  [%kernel
    fun (wq : int32 vector)
        (sc : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (nb : int32) ->
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      let k = mut tid in
      while k < nb do
        let w = wq.((block_idx_x * nb) + k) in
        let s = sc.((block_idx_x * nb) + k) in
        let q0 = float ((w land 255l) - 128l) in
        let q1 = float ((w / 256l land 255l) - 128l) in
        let q2 = float ((w / 65536l land 255l) - 128l) in
        let q3 = float ((w / 16777216l land 255l) - 128l) in
        acc :=
          acc
          +. s
             *. ((q0 *. x.(k * 4l))
                +. (q1 *. x.((k * 4l) + 1l))
                +. (q2 *. x.((k * 4l) + 2l))
                +. (q3 *. x.((k * 4l) + 3l))) ;
        k := k + 256l
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

let for_up =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (lo : int32) (hi : int32) ->
      let acc = mut 0.0 in
      for i = lo to hi do
        acc := acc +. (a.(i + 64l) *. b.(i + 64l))
      done ;
      b.(0l) <- acc]

let for_down =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (hi : int32) (lo : int32) ->
      let acc = mut 0.0 in
      for i = hi downto lo do
        acc := (acc *. 0.5) +. (a.(i + 64l) -. b.(i + 64l))
      done ;
      b.(0l) <- acc]

(* Neither loop below may change. *)
let constant_trip =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) ->
      let acc = mut 0.0 in
      for i = 0 to 31 do
        acc := acc +. a.(i)
      done ;
      b.(0l) <- acc]

let no_global_load =
  [%kernel
    fun (b : float32 vector) (n : int32) ->
      let acc = mut 0.0 in
      let k = mut 0l in
      while k < n do
        acc := acc +. 1.0 ;
        k := k + 1l
      done ;
      b.(0l) <- acc]

let variable_step =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) (step : int32) ->
      let acc = mut 0.0 in
      let k = mut 0l in
      while k < n do
        acc := acc +. a.(k) ;
        k := k + step
      done ;
      b.(0l) <- acc]

let barrier_in_loop =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) ->
      let acc = mut 0.0 in
      let k = mut 0l in
      while k < n do
        acc := acc +. a.(k) ;
        block_barrier () ;
        k := k + 1l
      done ;
      b.(0l) <- acc]

let bound_assigned_in_body =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) ->
      let acc = mut 0.0 in
      let m = mut n in
      let k = mut 0l in
      while k < m do
        acc := acc +. a.(k) ;
        m := m - 1l ;
        k := k + 1l
      done ;
      b.(0l) <- acc]

let for_bound_assigned_in_body =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n0 : int32) ->
      let acc = mut 0.0 in
      let n = mut n0 in
      for i = 0 to n do
        acc := acc +. a.(i) ;
        n := n - 1l
      done ;
      b.(0l) <- acc]

let nested_loops =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) (m : int32) ->
      let acc = mut 0.0 in
      for i = 0 to n do
        for j = 0 to m do
          acc := acc +. a.(i + j)
        done
      done ;
      b.(0l) <- acc]

(* A shared tile read in the loop, a global tile staged in. *)
let staged =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (n : int32) ->
      let%shared (stage : float32) = 256l in
      let tid = thread_idx_x in
      stage.(tid) <- w.(tid) ;
      block_barrier () ;
      let acc = mut 0.0 in
      let k = mut tid in
      while k < n do
        acc := acc +. (stage.(k land 255l) *. x.(k)) ;
        k := k + 256l
      done ;
      y.(tid) <- acc]

(* Constant-trip loops: expanded, the iteration's value an immediate offset. *)
let const_up =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) ->
      let acc = mut 0.0 in
      for i = 0 to 3 do
        acc := (acc *. 0.5) +. a.(n + i)
      done ;
      b.(0l) <- acc]

let const_down =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) ->
      let acc = mut 0.0 in
      for i = 5 downto 2 do
        acc := (acc *. 0.5) +. a.(n + (i * 2l))
      done ;
      b.(0l) <- acc]

let offset_plus =
  [%kernel
    fun (a : float32 vector) (b : float32 vector) (n : int32) ->
      b.(n + 2l) <- a.(n + 1l) +. a.(n - 2l) +. a.(n)]

(* Division and remainder by powers of two, on a value of unknown sign and on
   bounded non-negative ones. *)
let pow2_kernel =
  [%kernel
    fun (x : int32) (o : int32 vector) ->
      o.(0l) <- x / 8l ;
      o.(1l) <- x mod 8l ;
      o.(2l) <- x land 1023l / 32l ;
      o.(3l) <- x land 1023l mod 16l ;
      o.(4l) <- x / 2l ;
      o.(5l) <- x mod 1024l ;
      o.(6l) <- (thread_idx_x + 5l) / 4l]

(* [j] is [i + 1] for the first [i]; the second [i] must not redirect it. *)
let shadowed_atom =
  [%kernel
    fun (o : float32 vector) ->
      let i = thread_idx_x * 2l in
      let j = i + 1l in
      let i = i * 4l in
      o.(j + i - i) <- 1.0]

(******************************************************************************
 * ld.global.nc and mul.wide
 ******************************************************************************)

let test_nc_on_read_only_params () =
  let ptx = ptx_of (ir_of dot_while) in
  Alcotest.(check bool)
    "w and x load through ld.global.nc"
    true
    (contains ptx "ld.global.nc.f32") ;
  Alcotest.(check bool)
    "no plain ld.global.f32 remains: y is only stored"
    false
    (contains ptx "ld.global.f32") ;
  Alcotest.(check bool) "y is stored" true (contains ptx "st.global.f32")

let test_no_nc_on_written_param () =
  (* [b] is read and written; [a] is only read. *)
  let ptx = ptx_of (ir_of for_up) in
  Alcotest.(check bool)
    "the read-only a is ld.global.nc"
    true
    (contains ptx "ld.global.nc.f32") ;
  Alcotest.(check bool)
    "the written b keeps a plain ld.global"
    true
    (contains ptx "ld.global.f32")

let test_shared_reads_are_not_nc () =
  let ptx = ptx_of (ir_of staged) in
  Alcotest.(check bool) "shared read" true (contains ptx "ld.shared.f32") ;
  Alcotest.(check bool)
    "no nc qualifier on the shared space"
    false
    (contains ptx "ld.shared.nc") ;
  Alcotest.(check bool) "global w is nc" true (contains ptx "ld.global.nc.f32")

let test_mad_wide_addresses () =
  let ptx = ptx_of (ir_of dot_while) in
  Alcotest.(check bool) "mul.wide.s32" true (contains ptx "mul.wide.s32") ;
  Alcotest.(check bool)
    "no cvt.u64.u32/shl.b64 address recomputation"
    false
    (contains ptx "cvt.u64.u32" || contains ptx "shl.b64")

(* The analysis on hand-built IR: every way a parameter can be written. *)
let test_readonly_analysis () =
  let var name ty =
    {var_name = name; var_id = 0; var_type = ty; var_mutable = false}
  in
  let vec name = var name (TVec TInt32) in
  let a = vec "a" and b = vec "b" and c = vec "c" and d = vec "d" in
  let e = vec "e" and f = vec "f" and g = vec "g" in
  let i = var "i" TInt32 in
  let param v =
    DParam (v, Some {arr_elttype = TInt32; arr_memspace = Global})
  in
  let body =
    SSeq
      [
        (* b: stored *)
        SAssign (LArrayElem ("b", EConst (CInt32 0l)), EArrayRead ("a", EVar i));
        (* c: passed to a helper *)
        SExpr (EApp (EVar (var "helper" TUnit), [EVar c]));
        (* d: argument of an atomic *)
        SExpr
          (EIntrinsic
             ([], "atomic_add_int32", [EVar d; EConst (CInt32 0l); EVar i]));
        (* f: reached by an indirect store *)
        SAssign (LArrayElemExpr (EVar f, EConst (CInt32 0l)), EConst (CInt32 1l));
        (* e and g: only read *)
        SAssign
          ( LVar i,
            EBinop
              ( Add,
                EArrayRead ("e", EConst (CInt32 0l)),
                EArrayReadExpr (EVar g, EConst (CInt32 0l)) ) );
      ]
  in
  let k =
    {
      default_kernel with
      kern_name = "analysis";
      kern_params = List.map param [a; b; c; d; e; f; g];
      kern_body = body;
    }
  in
  Alcotest.(check (list string))
    "read-only parameters"
    ["a"; "e"; "g"]
    (Sarek_ir_ptx_readonly.params k)

(* A helper parameter that shares its name with a read-only kernel parameter is
   another array: here the helper's [x] is the kernel's written [y]. *)
let test_helper_param_shadows_read_only_name () =
  let var name ty =
    {var_name = name; var_id = 0; var_type = ty; var_mutable = false}
  in
  let x = var "x" (TVec TFloat32) and y = var "y" (TVec TFloat32) in
  let helper_x = var "x" (TVec TFloat32) and i = var "i" TInt32 in
  let helper =
    {
      hf_name = "h";
      hf_params = [helper_x; i];
      hf_ret_type = TFloat32;
      hf_body = SReturn (EArrayRead ("x", EVar i));
    }
  in
  let zero = EConst (CInt32 0l) in
  let param v =
    DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Global})
  in
  let k =
    {
      default_kernel with
      kern_name = "shadow";
      kern_params = [param x; param y];
      kern_funcs = [helper];
      kern_body =
        SSeq
          [
            SAssign (LArrayElem ("y", EConst (CInt32 1l)), EConst (CFloat32 0.));
            SAssign
              ( LArrayElem ("y", zero),
                EBinop
                  ( Add,
                    EApp (EVar (var "h" TFloat32), [EVar y; zero]),
                    EArrayRead ("x", zero) ) );
          ];
    }
  in
  Alcotest.(check (list string))
    "only the kernel's x is read-only"
    ["x"]
    (Sarek_ir_ptx_readonly.params k) ;
  let ptx = ptx_of k in
  Alcotest.(check int)
    "the kernel's own read of x is nc"
    1
    (count ptx "ld.global.nc.f32") ;
  Alcotest.(check int)
    "the helper's read of its x (the written y) is a plain load"
    1
    (count ptx "ld.global.f32")

(* A literal part of an index is the access's immediate offset. *)
let test_offsets_fold () =
  let ptx = ptx_of (ir_of offset_plus) in
  Alcotest.(check bool)
    "n + 1: +4"
    true
    (contains ptx "ld.global.nc.f32 %f0, [") ;
  Alcotest.(check bool) "offset +4" true (contains ptx "+4];") ;
  Alcotest.(check bool) "offset -8" true (contains ptx "+-8];") ;
  Alcotest.(check bool) "offset +8 on the store" true (contains ptx "+8], ") ;
  Alcotest.(check bool)
    "a signed multiply-add for the address"
    true
    (contains ptx "mul.wide.s32")

let test_constant_index_needs_no_address () =
  let ptx = ptx_of (ir_of const_up) in
  Alcotest.(check bool)
    "b.(0) is a store at the base register"
    true
    (contains ptx "st.global.f32 [%rd1]")

let test_const_loops_expand () =
  let ptx = ptx_of (ir_of const_up) in
  Alcotest.(check int) "4 iterations, 4 loads" 4 (count ptx "ld.global.nc.f32") ;
  Alcotest.(check bool) "no loop left" false (contains ptx "bra L")

(* A caller's literal binding must not fold into a helper's own variable of the
   same name. *)
let test_helper_does_not_see_caller_literals () =
  let var name ty =
    {var_name = name; var_id = 0; var_type = ty; var_mutable = false}
  in
  let c_caller = var "c" TInt32 and c_param = var "c" TInt32 in
  let p = var "p" (TVec TFloat32) and y = var "y" (TVec TFloat32) in
  let n = var "n" TInt32 in
  let helper =
    {
      hf_name = "h";
      hf_params = [var "p" (TVec TFloat32); c_param];
      hf_ret_type = TFloat32;
      hf_body = SReturn (EArrayRead ("p", EVar c_param));
    }
  in
  let param v =
    DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Global})
  in
  let k =
    {
      default_kernel with
      kern_name = "scope";
      kern_params = [param p; param y; DParam (n, None)];
      kern_funcs = [helper];
      kern_body =
        SLet
          ( c_caller,
            EConst (CInt32 5l),
            SAssign
              ( LArrayElem ("y", EConst (CInt32 0l)),
                EApp (EVar (var "h" TFloat32), [EVar p; EVar n]) ) );
    }
  in
  let ptx = ptx_of k in
  Alcotest.(check bool)
    "the helper indexes with its own c, not the caller's literal 5"
    false
    (contains ptx "+20]")

(* A straight-line evaluator for the few integer instructions the division by a
   power of two lowers to: runs the kernel's PTX on one input and returns the
   values it stores, in order. *)
let run_int_ptx ptx ~x ~tid =
  let regs = Hashtbl.create 32 in
  let get r = match Hashtbl.find_opt regs r with Some v -> v | None -> 0 in
  let wrap v = v land 0xffffffff in
  let signed v = if v >= 0x80000000 then v - 0x100000000 else v in
  let value tok =
    let tok = String.trim tok in
    if tok <> "" && tok.[0] = '%' then get tok else wrap (int_of_string tok)
  in
  let stores = ref [] in
  List.iter
    (fun line ->
      let line = String.trim line in
      let line =
        if line <> "" && line.[String.length line - 1] = ';' then
          String.sub line 0 (String.length line - 1)
        else line
      in
      match String.index_opt line ' ' with
      | None -> ()
      | Some i -> (
          let op = String.sub line 0 i in
          let args =
            List.map
              String.trim
              (String.split_on_char
                 ','
                 (String.sub line i (String.length line - i)))
          in
          match (op, args) with
          | "ld.param.u32", [d; p] when contains p "param_x" ->
              Hashtbl.replace regs d (wrap x)
          | "mov.u32", [d; "%tid.x"] -> Hashtbl.replace regs d tid
          | "mov.u32", [d; s] when s <> "" && s.[0] <> '%' ->
              Hashtbl.replace regs d (value s)
          | "mov.u32", [d; s] -> Hashtbl.replace regs d (get s)
          | "shr.s32", [d; a; n] ->
              Hashtbl.replace regs d (wrap (signed (value a) asr value n))
          | "shr.u32", [d; a; n] -> Hashtbl.replace regs d (value a lsr value n)
          | "add.u32", [d; a; b] ->
              Hashtbl.replace regs d (wrap (value a + value b))
          | "sub.u32", [d; a; b] ->
              Hashtbl.replace regs d (wrap (value a - value b))
          | "and.b32", [d; a; b] -> Hashtbl.replace regs d (value a land value b)
          | "st.global.s32", [_; v] -> stores := signed (value v) :: !stores
          | ("div.s32" | "rem.s32"), _ -> Alcotest.fail "a division was left"
          | _ -> ()))
    (String.split_on_char '\n' ptx) ;
  List.rev !stores

let test_pow2_division_exact () =
  let ptx = ptx_of (ir_of pow2_kernel) in
  Alcotest.(check bool) "no div.s32" false (contains ptx "div.s32") ;
  Alcotest.(check bool) "no rem.s32" false (contains ptx "rem.s32") ;
  let m = Int32.to_int Int32.min_int and mx = Int32.to_int Int32.max_int in
  List.iter
    (fun x ->
      List.iter
        (fun tid ->
          let r32 f = Int32.to_int (f (Int32.of_int x)) in
          let expect =
            [
              r32 (fun v -> Int32.div v 8l);
              r32 (fun v -> Int32.rem v 8l);
              r32 (fun v -> Int32.div (Int32.logand v 1023l) 32l);
              r32 (fun v -> Int32.rem (Int32.logand v 1023l) 16l);
              r32 (fun v -> Int32.div v 2l);
              r32 (fun v -> Int32.rem v 1024l);
              (tid + 5) / 4;
            ]
          in
          Alcotest.(check (list int))
            (Printf.sprintf "x=%d tid=%d" x tid)
            expect
            (run_int_ptx ptx ~x ~tid))
        [0; 1; 3; 1023])
    [
      m;
      m + 1;
      -1025;
      -1024;
      -1000;
      -9;
      -8;
      -7;
      -2;
      -1;
      0;
      1;
      2;
      7;
      8;
      9;
      1023;
      1024;
      1000;
      99999;
      mx - 1;
      mx;
    ]

let test_rebound_atom_is_not_folded () =
  let ptx = ptx_of (ir_of shadowed_atom) in
  Alcotest.(check bool)
    "j keeps its own register: no +1 element offset on the old i"
    false
    (contains ptx "+4]")

let ir_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let float_out_param () =
  DParam
    ( ir_var "o" (TVec TFloat32),
      Some {arr_elttype = TFloat32; arr_memspace = Global} )

(* A match arm whose pattern variable is named like an earlier literal [let]:
   inside the arm the index is the payload, not the literal 3. *)
let test_match_binder_hides_literal () =
  let i = ir_var "i" TInt32 and n = ir_var "n" TInt32 in
  let k =
    {
      default_kernel with
      kern_name = "match_binder";
      kern_params = [float_out_param (); DParam (n, None)];
      kern_variants = [("opt", [("Some", [TInt32]); ("None", [])])];
      kern_body =
        SLet
          ( i,
            EConst (CInt32 3l),
            SMatch
              ( EVariant ("opt", "Some", [EVar n]),
                [
                  ( PConstr ("Some", ["i"]),
                    SAssign (LArrayElem ("o", EVar i), EConst (CFloat32 1.0)) );
                  (PConstr ("None", []), SEmpty);
                ] ) );
    }
  in
  Alcotest.(check bool)
    "the arm's i is the payload: the literal 3 (offset 12) is not folded"
    false
    (contains (ptx_of k) "+12]")

(* [a] is an intrinsic cast of a mutable variable: its register is the
   variable's own, so [b = a + 1] must not be read through it after [m] moves. *)
let test_cast_alias_is_not_stable () =
  let m = {(ir_var "m" TInt32) with var_mutable = true} in
  let a = ir_var "a" TInt32 and b = ir_var "b" TInt32 in
  let k =
    {
      default_kernel with
      kern_name = "cast_alias";
      kern_params = [float_out_param ()];
      kern_body =
        SLetMut
          ( m,
            EIntrinsic (["Sarek_stdlib"; "Gpu"], "thread_idx_x", []),
            SLet
              ( a,
                EIntrinsic ([], "to_int32", [EVar m]),
                SLet
                  ( b,
                    EBinop (Add, EVar a, EConst (CInt32 1l)),
                    SSeq
                      [
                        SAssign
                          (LVar m, EBinop (Add, EVar m, EConst (CInt32 8l)));
                        SAssign (LArrayElem ("o", EVar b), EConst (CFloat32 0.0));
                      ] ) ) );
    }
  in
  Alcotest.(check bool)
    "b keeps its own register: no +1 element offset on a's register"
    false
    (contains (ptx_of k) "+4]")

(******************************************************************************
 * Which loops are unrolled
 ******************************************************************************)

(* Kernels carry closures (the native fallback): compare the code only. *)
let same_body (a : kernel) (b : kernel) =
  a.kern_body = b.kern_body && a.kern_funcs = b.kern_funcs

let rewritten k = Sarek_ir_ptx_unroll.rewrite (ir_of k)

let unchanged name k =
  Alcotest.test_case name `Quick (fun () ->
      let ir = ir_of k in
      Alcotest.(check bool)
        "loop left as written"
        true
        (same_body (Sarek_ir_ptx_unroll.rewrite ir) ir))

let changed name k =
  Alcotest.test_case name `Quick (fun () ->
      let ir = ir_of k in
      Alcotest.(check bool)
        "loop unrolled"
        true
        (not (same_body (Sarek_ir_ptx_unroll.rewrite ir) ir)))

let test_unroll_factor_follows_loads () =
  (* two loads per iteration: 16 copies, plus the remainder loop *)
  let ptx = ptx_of (ir_of dot_while) in
  Alcotest.(check int)
    "loads of w and x, 16 + 1 copies"
    34
    (count ptx "ld.global.nc.f32") ;
  (* six loads per iteration: 32 / 6 = 5 copies, plus the remainder loop *)
  let ptx = ptx_of (ir_of quantised) in
  Alcotest.(check int)
    "quantised: 5 + 1 copies of 6 loads"
    36
    (count ptx "ld.global.nc") ;
  Alcotest.(check bool)
    "the loop condition is tested in 64 bits"
    true
    (contains ptx "setp.lt.s64")

(* The kernels of the f16 corpus have no runtime-trip loop with global loads: the
   unroll pass leaves every one of them as it found it. *)
let test_corpus_unchanged () =
  List.iter
    (fun (name, k) ->
      Alcotest.(check bool)
        (name ^ " unchanged")
        true
        (same_body (Sarek_ir_ptx_unroll.rewrite k) k))
    (Ptx_f16_kernels.all ())

let test_explicit_pragma_wins () =
  let ir = ir_of dot_while in
  let rec under_pragma = function
    | SSeq l -> SSeq (List.map under_pragma l)
    | SLet (v, e, b) -> SLet (v, e, under_pragma b)
    | SLetMut (v, e, b) -> SLetMut (v, e, under_pragma b)
    | SWhile _ as w -> SPragma (["unroll 1"], w)
    | s -> s
  in
  let pragmaed = {ir with kern_body = under_pragma ir.kern_body} in
  Alcotest.(check bool)
    "a loop under an explicit pragma is left as the author wrote it"
    true
    (same_body (Sarek_ir_ptx_unroll.rewrite pragmaed) pragmaed)

(******************************************************************************
 * The unrolled IR computes what the loop computes (Interpreter oracle)
 ******************************************************************************)

let f32s values = Array.map (fun x -> Interp.VFloat32 x) values

let i32 n = Interp.VInt32 (Int32.of_int n)

let floats len seed =
  Array.init len (fun i ->
      (* values with different magnitudes: any reassociation shows *)
      let v = float_of_int (((i * 7919) + seed) mod 1009) /. 37.0 in
      if i mod 3 = 0 then v *. 1024.0 else v /. 64.0)

let run_args ir ~block ~grid args =
  let args =
    List.map
      (function
        | name, `Floats a -> (name, Interp.ArgArray (f32s a))
        | name, `Ints a ->
            (name, Interp.ArgArray (Array.map (fun x -> Interp.VInt32 x) a))
        | name, `Int n -> (name, Interp.ArgScalar (i32 n)))
      args
  in
  Interp.run_kernel ir ~block ~grid args ;
  List.filter_map
    (function
      | name, Interp.ArgArray a -> Some (name, Array.copy a) | _ -> None)
    args

let same_outputs label ir_a ir_b ~block ~grid args =
  let ra = run_args ir_a ~block ~grid args in
  let rb = run_args ir_b ~block ~grid args in
  List.iter2
    (fun (na, va) (_, vb) ->
      let bits = function
        | Interp.VFloat32 x -> Int64.bits_of_float x
        | Interp.VInt32 n -> Int64.of_int32 n
        | _ -> 0L
      in
      Alcotest.(check (array int64))
        (Printf.sprintf "%s: %s identical bits" label na)
        (Array.map bits va)
        (Array.map bits vb))
    ra
    rb

(* Trip counts around every boundary of the factor: 0, 1, u - 1, u, u + 1,
   several of them, with a remainder, and a negative bound. *)
let trip_counts = [-3; 0; 1; 2; 4; 5; 6; 7; 15; 16; 17; 31; 32; 33; 100]

let test_while_oracle () =
  let ir = ir_of dot_while in
  let unrolled = Sarek_ir_ptx_unroll.rewrite ir in
  Alcotest.(check bool) "rewritten" true (not (same_body unrolled ir)) ;
  List.iter
    (fun trips ->
      (* thread tid runs k = tid, tid + 256, ... while k < n *)
      let n = trips * 256 in
      same_outputs
        (Printf.sprintf "dot_while n=%d" n)
        ir
        unrolled
        ~block:(4, 1, 1)
        ~grid:(2, 1, 1)
        [
          ("w", `Floats (floats (max 1 (2 * n)) 1));
          ("x", `Floats (floats (max 1 n) 2));
          ("y", `Floats (Array.make 512 0.0));
          ("n", `Int n);
        ])
    trip_counts

let test_while_le_oracle () =
  let ir = ir_of dot_while_le in
  let unrolled = Sarek_ir_ptx_unroll.rewrite ir in
  Alcotest.(check bool) "rewritten" true (not (same_body unrolled ir)) ;
  List.iter
    (fun n ->
      same_outputs
        (Printf.sprintf "dot_while_le n=%d" n)
        ir
        unrolled
        ~block:(5, 1, 1)
        ~grid:(2, 1, 1)
        [
          ("w", `Floats (floats (max 1 (2 * (n + 1))) 3));
          ("x", `Floats (floats (max 1 (n + 1)) 4));
          ("y", `Floats (Array.make 512 0.0));
          ("n", `Int n);
        ])
    [-5; 0; 1; 2; 3; 47; 48; 49; 50; 51; 52; 53; 54; 100; 247]

let test_quantised_oracle () =
  let ir = ir_of quantised in
  let unrolled = Sarek_ir_ptx_unroll.rewrite ir in
  Alcotest.(check bool) "rewritten" true (not (same_body unrolled ir)) ;
  List.iter
    (fun trips ->
      let nb = trips * 8 in
      let words =
        Array.init
          (max 1 (2 * nb))
          (fun i -> Int32.of_int (i * 2654435761 land 0x7fffffff))
      in
      same_outputs
        (Printf.sprintf "quantised nb=%d" nb)
        ir
        unrolled
        ~block:(8, 1, 1)
        ~grid:(2, 1, 1)
        [
          ("wq", `Ints words);
          ("sc", `Floats (floats (max 1 (2 * nb)) 5));
          ("x", `Floats (floats (max 4 (4 * nb)) 6));
          ("y", `Floats (Array.make 512 0.0));
          ("nb", `Int nb);
        ])
    [0; 1; 2; 3; 4; 5; 6; 7; 8; 9; 12; 25]

let test_const_oracle () =
  List.iter
    (fun (name, kernel) ->
      let ir = ir_of kernel in
      let unrolled = Sarek_ir_ptx_unroll.rewrite ir in
      Alcotest.(check bool)
        (name ^ " rewritten")
        true
        (not (same_body unrolled ir)) ;
      List.iter
        (fun n ->
          same_outputs
            (Printf.sprintf "%s n=%d" name n)
            ir
            unrolled
            ~block:(1, 1, 1)
            ~grid:(1, 1, 1)
            [
              ("a", `Floats (floats 64 11));
              ("b", `Floats (floats 8 12));
              ("n", `Int n);
            ])
        [0; 1; 7; 20])
    [("const_up", const_up); ("const_down", const_down)]

let test_for_oracle () =
  List.iter
    (fun (name, kernel, (first, second)) ->
      let ir = ir_of kernel in
      let unrolled = Sarek_ir_ptx_unroll.rewrite ir in
      Alcotest.(check bool)
        (name ^ " rewritten")
        true
        (not (same_body unrolled ir)) ;
      List.iter
        (fun (p, q) ->
          same_outputs
            (Printf.sprintf "%s %d %d" name p q)
            ir
            unrolled
            ~block:(1, 1, 1)
            ~grid:(1, 1, 1)
            [
              ("a", `Floats (floats 256 7));
              ("b", `Floats (floats 256 8));
              (first, `Int p);
              (second, `Int q);
            ])
        [
          (0, -1);
          (0, 0);
          (0, 1);
          (-3, 12);
          (2, 3);
          (0, 14);
          (0, 15);
          (0, 16);
          (1, 16);
          (-10, 40);
          (5, 4);
          (30, 100);
          (0, 150);
          (-60, -50);
          (40, 0);
          (150, 0);
          (-1, -60);
        ])
    [("for_up", for_up, ("lo", "hi")); ("for_down", for_down, ("hi", "lo"))]

(******************************************************************************
 * The policy
 ******************************************************************************)

let test_policy_restriction_is_identity () =
  let ir = ir_of dot_while in
  Alcotest.(check bool)
    "enabled by default"
    true
    (not (same_body (rewritten dot_while) ir)) ;
  Sarek_policy.restrict Sarek_policy.Auto_unroll ;
  Alcotest.(check bool)
    "SAREK_OPT_DISABLE=unroll leaves the kernel untouched"
    true
    (Sarek_ir_ptx_unroll.rewrite ir == ir) ;
  let ptx = ptx_of ir in
  Alcotest.(check int)
    "one copy of the two loads"
    2
    (count ptx "ld.global.nc.f32")

let () =
  Alcotest.run
    "ptx_load_batching"
    [
      ( "nc_and_addresses",
        [
          Alcotest.test_case
            "nc on read-only params"
            `Quick
            test_nc_on_read_only_params;
          Alcotest.test_case
            "no nc on a written param"
            `Quick
            test_no_nc_on_written_param;
          Alcotest.test_case
            "shared reads are not nc"
            `Quick
            test_shared_reads_are_not_nc;
          Alcotest.test_case "mul.wide addresses" `Quick test_mad_wide_addresses;
          Alcotest.test_case "read-only analysis" `Quick test_readonly_analysis;
          Alcotest.test_case "literal offsets fold" `Quick test_offsets_fold;
          Alcotest.test_case
            "a match binder hides a literal"
            `Quick
            test_match_binder_hides_literal;
          Alcotest.test_case
            "a cast alias is not stable"
            `Quick
            test_cast_alias_is_not_stable;
          Alcotest.test_case
            "a rebound atom is not folded"
            `Quick
            test_rebound_atom_is_not_folded;
          Alcotest.test_case
            "division by a power of two is exact"
            `Quick
            test_pow2_division_exact;
          Alcotest.test_case
            "constant index needs no address"
            `Quick
            test_constant_index_needs_no_address;
          Alcotest.test_case
            "constant-trip loops expand"
            `Quick
            test_const_loops_expand;
          Alcotest.test_case
            "helper does not see the caller's literals"
            `Quick
            test_helper_does_not_see_caller_literals;
          Alcotest.test_case
            "helper parameter shadowing a read-only name"
            `Quick
            test_helper_param_shadows_read_only_name;
        ] );
      ( "which_loops",
        [
          changed "runtime while" dot_while;
          changed "runtime while <=" dot_while_le;
          changed "runtime for up" for_up;
          changed "runtime for down" for_down;
          changed "quantised matvec" quantised;
          unchanged "constant trip" constant_trip;
          unchanged "no global load" no_global_load;
          unchanged "variable step" variable_step;
          unchanged "barrier in the body" barrier_in_loop;
          unchanged "bound assigned in the body" bound_assigned_in_body;
          unchanged "for bound assigned in the body" for_bound_assigned_in_body;
          changed "inner loop of a nest" nested_loops;
          Alcotest.test_case
            "factor follows the loads"
            `Quick
            test_unroll_factor_follows_loads;
          Alcotest.test_case "f16 corpus unchanged" `Quick test_corpus_unchanged;
          Alcotest.test_case
            "explicit pragma wins"
            `Quick
            test_explicit_pragma_wins;
        ] );
      ( "oracle",
        [
          Alcotest.test_case "while" `Quick test_while_oracle;
          Alcotest.test_case "while <=" `Quick test_while_le_oracle;
          Alcotest.test_case "quantised" `Quick test_quantised_oracle;
          Alcotest.test_case "for up and down" `Quick test_for_oracle;
          Alcotest.test_case "constant-trip loops" `Quick test_const_oracle;
        ] );
      ( "policy",
        [
          Alcotest.test_case
            "restriction is the identity"
            `Quick
            test_policy_restriction_is_identity;
        ] );
    ]
