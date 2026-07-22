(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX codegen shared types: register allocator, environment, error helpers,
    and buffer emit primitives. *)

open Sarek_ir_types

(** {1 Error handling} *)

exception Ptx_codegen_error of string

let fail msg = raise (Ptx_codegen_error msg)

let unsupported what = fail ("PTX codegen: unsupported construct: " ^ what)

(** {1 Register allocator} *)

(** Counter-based register allocator. Each PTX type has an independent counter
    so that register names stay readable (e.g. %r0, %f0, %rd0). *)
type reg_alloc = {
  mutable u32 : int;
  mutable u64 : int;
  mutable f32 : int;
  mutable f64 : int;
  mutable pred : int;
  mutable label : int;
  arr_elt_types : (string, elttype) Hashtbl.t;
  arr_memspaces : (string, unit) Hashtbl.t;
  shared_decls : Buffer.t;
      (** [.shared] declarations discovered while emitting the body (SLet-bound
          shared arrays); spliced into the kernel prologue by [generate]. *)
  funcs : (string, helper_func) Hashtbl.t;
      (** Kernel helper functions ([kern_funcs]), inlined at EApp sites. *)
  mutable inline_stack : string list;
      (** Helper names currently being inlined — recursion guard. *)
  mutable inline_ret : (string option * string) list;
      (** Per-inline (result register, end label); SReturn inside an inlined
          body writes the register (if any) and branches to the label instead of
          emitting [ret]. *)
}

let make_alloc () =
  {
    u32 = 0;
    u64 = 0;
    f32 = 0;
    f64 = 0;
    pred = 0;
    label = 0;
    arr_elt_types = Hashtbl.create 8;
    arr_memspaces = Hashtbl.create 8;
    shared_decls = Buffer.create 128;
    funcs = Hashtbl.create 4;
    inline_stack = [];
    inline_ret = [];
  }

let new_u32 a =
  let n = a.u32 in
  a.u32 <- n + 1 ;
  Printf.sprintf "%%r%d" n

let new_u64 a =
  let n = a.u64 in
  a.u64 <- n + 1 ;
  Printf.sprintf "%%rd%d" n

let new_f32 a =
  let n = a.f32 in
  a.f32 <- n + 1 ;
  Printf.sprintf "%%f%d" n

let new_f64 a =
  let n = a.f64 in
  a.f64 <- n + 1 ;
  Printf.sprintf "%%fd%d" n

let new_pred a =
  let n = a.pred in
  a.pred <- n + 1 ;
  Printf.sprintf "%%p%d" n

let new_label a =
  let n = a.label in
  a.label <- n + 1 ;
  Printf.sprintf "L%d" n

(** {1 Type mapping} *)

let ptx_reg_type_of = function
  | TInt32 | TBool -> ".u32"
  | TInt64 -> ".u64"
  | TFloat32 -> ".f32"
  | TFloat64 -> ".f64"
  | TUnit -> ".u32"
  | TVec _ -> ".u64"
  | TArray _ -> ".u64"
  | TRecord _ -> unsupported "TRecord register type"
  | TVariant _ -> unsupported "TVariant register type"

let new_reg_for_type alloc = function
  | TInt32 | TBool | TUnit -> new_u32 alloc
  | TInt64 -> new_u64 alloc
  | TFloat32 -> new_f32 alloc
  | TFloat64 -> new_f64 alloc
  | TVec _ | TArray _ -> new_u64 alloc
  | TRecord _ -> unsupported "TRecord new_reg"
  | TVariant _ -> unsupported "TVariant new_reg"

(** {1 Environment: variable name -> PTX binding}

    Scalars bind to a single register name. Aggregates (records/variants,
    SROA-decomposed) bind to a structured register set. Nested records appear as
    nested [ARecord] values under their field name. *)

type agg_value =
  | ARecord of (string * binding) list
      (** Record value: one binding per field, in declaration order. *)
  | AVariant of {tag_reg : string; payloads : (int * binding list) list}
      (** Variant value: u32 tag register + per-(constructor index) payload
          bindings in payload-argument order. *)

and binding = Scalar of string | Agg of agg_value

type env = (string, binding) Hashtbl.t

let make_env () : env = Hashtbl.create 32

let env_bind (env : env) name reg = Hashtbl.replace env name (Scalar reg)

let env_bind_binding (env : env) name b = Hashtbl.replace env name b

let env_lookup (env : env) name =
  match Hashtbl.find_opt env name with
  | Some (Scalar r) -> r
  | Some (Agg _) ->
      fail
        ("PTX codegen: variable " ^ name
       ^ " is an aggregate; internal error: scalar lookup on aggregate binding"
        )
  | None -> fail ("PTX codegen: unbound variable: " ^ name)

let env_lookup_binding (env : env) name =
  match Hashtbl.find_opt env name with
  | Some b -> b
  | None -> fail ("PTX codegen: unbound variable: " ^ name)

(** Name of the implicit length parameter paired with a vector/array parameter.
    Single definition for both the producer (emit_params declares and binds it)
    and the consumer (EArrayLen looks it up) — it must also match the
    "sarek_<name>_length" convention used by Execute.expand_to_run_source_args
    and the C-family code generators. *)
let length_param_name arr_name = Printf.sprintf "sarek_%s_length" arr_name

(** {1 Emit helpers} *)

let emit buf fmt = Printf.bprintf buf ("    " ^^ fmt ^^ "\n")

let emit_label buf lbl = Printf.bprintf buf "%s:\n" lbl

(** [copy_reg buf alloc r] allocates a fresh register of [r]'s class and emits a
    mov from [r] into it. Used wherever a binding must not alias the source
    register (mutable lets, inlined helper parameters). *)
let copy_reg buf a r_src =
  if String.length r_src >= 3 && r_src.[1] = 'f' && r_src.[2] = 'd' then begin
    let r = new_f64 a in
    emit buf "mov.f64 %s, %s;" r r_src ;
    r
  end
  else if String.length r_src >= 2 && r_src.[1] = 'f' then begin
    let r = new_f32 a in
    emit buf "mov.f32 %s, %s;" r r_src ;
    r
  end
  else if String.length r_src >= 3 && r_src.[1] = 'r' && r_src.[2] = 'd' then begin
    let r = new_u64 a in
    emit buf "mov.u64 %s, %s;" r r_src ;
    r
  end
  else begin
    let r = new_u32 a in
    emit buf "mov.u32 %s, %s;" r r_src ;
    r
  end

(** {1 Shared-memory declaration helpers} *)

let ptx_align_of_elttype = function
  | TFloat32 | TInt32 | TBool -> 4
  | TFloat64 | TInt64 -> 8
  | TUnit -> 4
  | TVec _ | TArray _ -> 8
  | TRecord _ | TVariant _ -> unsupported "align of custom type"

let ptx_btype_of_elttype = function
  | TFloat32 | TInt32 | TBool | TUnit -> "b32"
  | TFloat64 | TInt64 -> "b64"
  | TVec _ | TArray _ -> "b64"
  | TRecord _ | TVariant _ -> unsupported "btype of custom type"

(** {1 Statement-emitter hook}

    EApp inlining in the expression emitter must emit the helper body, a
    statement — but Sarek_ir_ptx_stmt depends on Sarek_ir_ptx_expr. This hook
    breaks the cycle: Sarek_ir_ptx_stmt installs [emit_stmt] here at load time,
    and the expression emitter calls through it. *)
let stmt_emitter :
    (Buffer.t -> reg_alloc -> env -> Sarek_ir_types.stmt -> unit) ref =
  ref (fun _ _ _ _ -> fail "PTX codegen: stmt_emitter not initialized")
