(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** How a kernel uses its vector parameters, and whether it can be fused
    horizontally (specs/launch-window.md).

    Both answers are read from the IR only and are conservative: a use this
    module cannot classify counts as a read AND a write, and a construct it does
    not understand makes the kernel non-fusable. Wrong in the other direction
    would let the launch window reorder or merge launches that conflict. *)

open Sarek_ir_types

(** {1 Read/write sets} *)

type access = {read : bool; write : bool}

let no_access = {read = false; write = false}

let both = {read = true; write = true}

let join a b = {read = a.read || b.read; write = a.write || b.write}

(** Vector parameter names of [k], in declaration order. *)
let vector_params (k : kernel) : string list =
  List.filter_map
    (function DParam (v, Some _) -> Some v.var_name | _ -> None)
    k.kern_params

(* The accumulator maps a parameter name to its access so far. Only names in
   the initial table are recorded: shared arrays and locals are not launch
   arguments. *)
type acc = (string, access) Hashtbl.t

let note (tbl : acc) name a =
  match Hashtbl.find_opt tbl name with
  | Some prev -> Hashtbl.replace tbl name (join prev a)
  | None -> ()

let read_ = {read = true; write = false}

let write_ = {read = false; write = true}

(* An array base given as an expression: an [EVar] naming a parameter is an
   access of that kind; anything else is visited as an ordinary expression,
   where a bare parameter counts as [both]. *)
let rec base_access tbl a = function
  | EVar v -> note tbl v.var_name a
  | e -> expr tbl e

and expr tbl e =
  match e with
  | EConst _ | EArrayLen _ -> ()
  | EVar v ->
      (* A vector passed whole (to an intrinsic, a helper, a record...) may be
         read or written by the callee. *)
      note tbl v.var_name both
  | EArrayRead (name, idx) ->
      note tbl name read_ ;
      expr tbl idx
  | EArrayReadExpr (base, idx) ->
      base_access tbl read_ base ;
      expr tbl idx
  | EBinop (_, a, b) ->
      expr tbl a ;
      expr tbl b
  | EUnop (_, a) | ECast (_, a) | ERecordField (a, _) -> expr tbl a
  | EIntrinsic (_, name, args)
    when Sarek_ir_analysis.is_atomic_intrinsic_name name ->
      (* An atomic may take its target as an element ([&a[i]]): that element
         is read AND written. *)
      List.iter
        (function
          | EArrayRead (a, i) ->
              note tbl a both ;
              expr tbl i
          | x -> expr tbl x)
        args
  | EIntrinsic (_, _, args) | ETuple args | EVariant (_, _, args) ->
      List.iter (expr tbl) args
  | EApp (f, args) -> List.iter (expr tbl) (f :: args)
  | ERecord (_, fields) -> List.iter (fun (_, e) -> expr tbl e) fields
  | EArrayCreate (_, size, _) -> expr tbl size
  | EIf (c, t, f) -> List.iter (expr tbl) [c; t; f]
  | EMatch (s, cases) ->
      expr tbl s ;
      List.iter (fun (_, e) -> expr tbl e) cases

(* [a.(i).f <- v] writes element [i] of [a]: the field case recurses. *)
let rec lvalue tbl = function
  | LVar _ -> ()
  | LArrayElem (name, idx) ->
      note tbl name write_ ;
      expr tbl idx
  | LArrayElemExpr (base, idx) ->
      base_access tbl write_ base ;
      expr tbl idx
  | LRecordField (lv, _) -> lvalue tbl lv

let coopmat tbl = function
  | CM_decl _ | CM_muladd _ -> ()
  | CM_load {src; index; stride; _} ->
      note tbl src read_ ;
      expr tbl index ;
      expr tbl stride
  | CM_store {dst; index; stride; _} ->
      note tbl dst write_ ;
      expr tbl index ;
      expr tbl stride
  | CM_scale_add _ as op ->
      (* The scales are read, never written; the fragments are not buffers. *)
      List.iter (fun buf -> note tbl buf read_) (Sarek_coopmat_scale.buffers op) ;
      List.iter (expr tbl) (Sarek_coopmat_scale.exprs op)

let rec stmt tbl s =
  match s with
  | SAssign (lv, e) ->
      lvalue tbl lv ;
      expr tbl e
  | SSeq ss -> List.iter (stmt tbl) ss
  | SIf (c, t, f) ->
      expr tbl c ;
      stmt tbl t ;
      Option.iter (stmt tbl) f
  | SWhile (c, b) ->
      expr tbl c ;
      stmt tbl b
  | SFor (_, lo, hi, _, b) ->
      expr tbl lo ;
      expr tbl hi ;
      stmt tbl b
  | SMatch (e, cases) ->
      expr tbl e ;
      List.iter (fun (_, b) -> stmt tbl b) cases
  | SReturn e | SExpr e -> expr tbl e
  | SLet (_, e, b) | SLetMut (_, e, b) ->
      expr tbl e ;
      stmt tbl b
  | SPragma (_, b) | SBlock b -> stmt tbl b
  | SCoopmat op -> coopmat tbl op
  | SNative _ ->
      (* Inline backend code may touch any parameter. *)
      Hashtbl.filter_map_inplace (fun _ _ -> Some both) tbl
  | SBarrier | SWarpBarrier | SMemFence | SEmpty -> ()

(** Access of each vector parameter of [k], in declaration order. Helper
    functions receive vectors only as arguments, which {!expr} already counts as
    [both]. *)
let param_accesses (k : kernel) : (string * access) list =
  let names = vector_params k in
  let tbl : acc = Hashtbl.create 8 in
  List.iter (fun n -> Hashtbl.replace tbl n no_access) names ;
  stmt tbl k.kern_body ;
  List.iter
    (function
      | DLocal (_, Some e) -> expr tbl e | DLocal _ | DParam _ | DShared _ -> ())
    k.kern_locals ;
  List.map (fun n -> (n, Hashtbl.find tbl n)) names

(** {1 Horizontal fusability} *)

(** Position intrinsics that horizontal fusion rewrites, under the exact path
    [Sarek_stdlib.Gpu] only. Each rewritten form keeps the type the backends
    give the original: [block_idx_x - off] and [gdx * block_dim_x] stay in the
    backend's own unsigned type where it has one. *)
let rewritten_intrinsics =
  [
    "block_idx_x";
    "global_thread_id";
    "global_idx";
    "global_idx_x";
    "global_size";
    "global_size_x";
  ]

(** Position intrinsics a member may not use at all. [grid_dim_x] would have to
    become an [int] parameter, while CUDA/HIP, OpenCL and Metal emit it as an
    unsigned value, so comparisons and divisions with negative operands would
    change meaning. [block_id_x] is an alias the backends accept for
    [block_idx_x]; it is refused rather than rewritten under a second name. *)
let refused_intrinsics = ["grid_dim_x"; "block_id_x"]

let gpu_path = ["Sarek_stdlib"; "Gpu"]

let is_stdlib_path = function "Sarek_stdlib" :: _ -> true | _ -> false

(* The leading [let a = create_array ... Shared in] bindings of a body, and the
   rest. Shared arrays must be declared at kernel scope on OpenCL and Metal, so
   only this prefix can be hoisted above the dispatch branch. *)
let rec shared_prefix = function
  | SLet (v, (EArrayCreate (_, _, Shared) as e), rest) ->
      let binds, body = shared_prefix rest in
      ((v, e) :: binds, body)
  | s -> ([], s)

exception Refuse of string

let refuse fmt = Printf.ksprintf (fun s -> raise (Refuse s)) fmt

module Ids = Set.Make (Int)
module Names = Set.Make (String)

(* Variables in scope. A free variable (an unbound global reference) would be
   renamed by the fusion rewrite and stop resolving, so it is refused. *)
type scope = {ids : Ids.t; names : Names.t; helpers : Names.t; in_helper : bool}

let bind sc (v : var) =
  {sc with ids = Ids.add v.var_id sc.ids; names = Names.add v.var_name sc.names}

let bind_pattern sc = function
  | PConstr (_, vs) ->
      {sc with names = List.fold_left (fun n v -> Names.add v n) sc.names vs}
  | PWild -> sc

let check_intrinsic sc path name =
  if not (is_stdlib_path path) then
    refuse
      "intrinsic %s.%s is not from Sarek_stdlib"
      (String.concat "." path)
      name ;
  if List.mem name refused_intrinsics then refuse "uses %s" name ;
  if List.mem name rewritten_intrinsics then begin
    if path <> gpu_path then
      refuse "%s under %s, not Sarek_stdlib.Gpu" name (String.concat "." path) ;
    if sc.in_helper then refuse "helper function reads %s" name
  end

let rec check_expr sc e =
  let go = check_expr sc in
  match e with
  | EIntrinsic (path, name, args) ->
      check_intrinsic sc path name ;
      List.iter go args
  | EVar v ->
      if not (Ids.mem v.var_id sc.ids || Names.mem v.var_name sc.names) then
        refuse "free variable %s" v.var_name
  | EApp (EVar f, es) when Names.mem f.var_name sc.helpers -> List.iter go es
  | EArrayCreate (_, _, Shared) ->
      refuse "shared array not in the body's leading let prefix"
  | EArrayCreate (_, size, _) -> go size
  | EConst _ | EArrayLen _ -> ()
  | EArrayRead (_, i) -> go i
  | EArrayReadExpr (a, b) | EBinop (_, a, b) ->
      go a ;
      go b
  | EUnop (_, a) | ECast (_, a) | ERecordField (a, _) -> go a
  | ETuple es | EVariant (_, _, es) -> List.iter go es
  | EApp (f, es) -> List.iter go (f :: es)
  | ERecord (_, fs) -> List.iter (fun (_, e) -> go e) fs
  | EIf (a, b, c) -> List.iter go [a; b; c]
  | EMatch (x, cs) ->
      go x ;
      List.iter (fun (p, e) -> check_expr (bind_pattern sc p) e) cs

let rec check_lvalue sc = function
  | LVar v -> check_expr sc (EVar v)
  | LArrayElem (_, i) -> check_expr sc i
  | LArrayElemExpr (a, i) ->
      check_expr sc a ;
      check_expr sc i
  | LRecordField (lv, _) -> check_lvalue sc lv

let rec check_stmt sc s =
  let e = check_expr sc and st = check_stmt sc in
  match s with
  | SAssign (lv, x) ->
      check_lvalue sc lv ;
      e x
  | SSeq ss -> List.iter st ss
  | SIf (c, t, f) ->
      e c ;
      st t ;
      Option.iter st f
  | SWhile (c, b) ->
      e c ;
      st b
  | SFor (v, lo, hi, _, b) ->
      e lo ;
      e hi ;
      check_stmt (bind sc v) b
  | SMatch (x, cs) ->
      e x ;
      List.iter (fun (p, b) -> check_stmt (bind_pattern sc p) b) cs
  | SReturn x | SExpr x -> e x
  | SLet (v, x, b) | SLetMut (v, x, b) ->
      e x ;
      check_stmt (bind sc v) b
  | SPragma (_, b) | SBlock b -> st b
  | SNative _ -> refuse "inline native code"
  | SCoopmat _ -> refuse "cooperative-matrix statement"
  | SBarrier | SWarpBarrier | SMemFence | SEmpty -> ()

(* Bytes of one shared declaration, when its size is a constant. *)
let shared_bytes_of elt size =
  match (size, Sarek_ir_layout.scalar_size elt) with
  | EConst (CInt32 n), w -> Int32.to_int n * w
  | EConst (CInt64 n), w -> Int64.to_int n * w
  | _, _ -> refuse "shared array of non-constant size"
  | exception Invalid_argument _ -> refuse "shared array of aggregate elements"

let kernel_scope (k : kernel) =
  let helpers =
    List.fold_left (fun n h -> Names.add h.hf_name n) Names.empty k.kern_funcs
  in
  List.fold_left
    (fun sc -> function
      | DParam (v, _) | DLocal (v, _) -> bind sc v
      | DShared (name, _, _) -> {sc with names = Names.add name sc.names})
    {ids = Ids.empty; names = Names.empty; helpers; in_helper = false}
    k.kern_params
  |> fun sc ->
  List.fold_left
    (fun sc -> function DLocal (v, _) -> bind sc v | _ -> sc)
    sc
    k.kern_locals

let check_helper (k : kernel) hf =
  let sc = kernel_scope k in
  let sc =
    List.fold_left
      bind
      {sc with ids = Ids.empty; names = Names.empty; in_helper = true}
      hf.hf_params
  in
  check_stmt sc hf.hf_body

(* Under [Refuse]: the shared memory [k] declares, in bytes.

   A kernel that uses a warp operation (a shuffle, [warp_size], [lane_id]) is
   never merged, fused or re-launched with another: a shuffle is a convergent
   operation whose result depends on which threads share its warp, and every
   transformation built on this check changes that membership - horizontal
   fusion re-numbers the blocks of its members and guards them, an epilogue
   adds threads and a guard around the producer. Refused here, at the one gate
   all of them go through ([horizontal_verdict], which Launch_window,
   Launch_epilogue and Fusion_epilogue's consumer shape read). *)
let check_kernel (k : kernel) : int =
  if Sarek_ir_analysis.kernel_uses_warp_ops k then
    refuse "uses a warp operation (shuffle, warp_size or lane_id)" ;
  let sc = kernel_scope k in
  let binds, body = shared_prefix k.kern_body in
  let sc, bytes =
    List.fold_left
      (fun (sc, bytes) (v, x) ->
        match x with
        | EArrayCreate (elt, size, _) ->
            check_expr sc size ;
            (bind sc v, bytes + shared_bytes_of elt size)
        | _ -> (sc, bytes))
      (sc, 0)
      binds
  in
  check_stmt sc body ;
  let locals =
    List.fold_left
      (fun bytes -> function
        | DLocal (_, Some x) ->
            check_expr sc x ;
            bytes
        | DShared (_, elt, Some x) ->
            check_expr sc x ;
            bytes + shared_bytes_of elt x
        | DShared (_, _, None) -> refuse "shared declaration without a size"
        | DLocal _ | DParam _ -> bytes)
      0
      k.kern_locals
  in
  List.iter (check_helper k) k.kern_funcs ;
  bytes + locals

(** [Ok bytes] when [k] can be a member of a horizontally fused kernel, with the
    shared memory it declares; otherwise the reason it cannot. *)
let horizontal_verdict (k : kernel) : (int, string) result =
  match check_kernel k with
  | bytes -> Ok bytes
  | exception Refuse why -> Error why
