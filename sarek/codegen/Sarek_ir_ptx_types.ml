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

(** {1 Environment: variable name -> PTX register name} *)

type env = (string, string) Hashtbl.t

let make_env () : env = Hashtbl.create 32

let env_bind (env : env) name reg = Hashtbl.replace env name reg

let env_lookup (env : env) name =
  match Hashtbl.find_opt env name with
  | Some r -> r
  | None -> fail ("PTX codegen: unbound variable: " ^ name)

(** {1 Emit helpers} *)

let emit buf fmt = Printf.bprintf buf ("    " ^^ fmt ^^ "\n")

let emit_label buf lbl = Printf.bprintf buf "%s:\n" lbl
