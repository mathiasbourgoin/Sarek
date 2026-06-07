(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Sarek_ir_types
module F32 = Sarek_float32

(** Re-export value type from Sarek_value for convenience *)
type value = Sarek_value.value =
  | VInt32 of int32
  | VInt64 of int64
  | VFloat32 of float
  | VFloat64 of float
  | VBool of bool
  | VUnit
  | VArray of value array
  | VRecord of string * value array
  | VVariant of string * int * value list

(** {1 BSP Barrier Effect}
    Used for synchronizing threads at barriers. Each thread is suspended when it
    hits a barrier, and all threads are resumed together. *)

type _ Effect.t += Barrier : unit Effect.t

(** {1 Thread State} *)

type thread_state = {
  thread_idx : int * int * int;
  block_idx : int * int * int;
  block_dim : int * int * int;
  grid_dim : int * int * int;
}

(** {1 Environment} *)

type env = {
  vars : (int, value) Hashtbl.t;  (** var_id -> value *)
  vars_by_name : (string, value) Hashtbl.t;  (** var_name -> value (fallback) *)
  arrays : (string, value array) Hashtbl.t;  (** array_name -> data *)
  shared : (string, value array) Hashtbl.t;  (** shared arrays for block *)
  funcs : (string, helper_func) Hashtbl.t;  (** helper functions *)
}

let create_env () =
  {
    vars = Hashtbl.create 32;
    vars_by_name = Hashtbl.create 32;
    arrays = Hashtbl.create 16;
    shared = Hashtbl.create 8;
    funcs = Hashtbl.create 8;
  }

let copy_env env =
  {
    vars = Hashtbl.copy env.vars;
    vars_by_name = Hashtbl.copy env.vars_by_name;
    arrays = env.arrays;
    (* shared across threads *)
    shared = env.shared;
    (* shared within block *)
    funcs = env.funcs;
    (* shared *)
  }

(** Bind a variable in the environment (both by id and name) *)
let bind_var env (v : var) value =
  Hashtbl.replace env.vars v.var_id value ;
  Hashtbl.replace env.vars_by_name v.var_name value

(** Look up a variable (try id first, then name as fallback) *)
let lookup_var env (v : var) =
  match Hashtbl.find_opt env.vars v.var_id with
  | Some value -> value
  | None -> (
      match Hashtbl.find_opt env.vars_by_name v.var_name with
      | Some value -> value
      | None ->
          Interp_error.raise_error
            (Unbound_variable {name = v.var_name; context = "eval_expr"}))

(** {1 Value Operations} *)

let to_int32 = function
  | VInt32 n -> n
  | VInt64 n -> Int64.to_int32 n
  | VFloat32 f -> Int32.of_float f
  | VFloat64 f -> Int32.of_float f
  | VBool b -> if b then 1l else 0l
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "int32";
             context = "to_int32";
           })

let to_int64 = function
  | VInt64 n -> n
  | VInt32 n -> Int64.of_int32 n
  | VFloat32 f -> Int64.of_float f
  | VFloat64 f -> Int64.of_float f
  | VBool b -> if b then 1L else 0L
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "int64";
             context = "to_int64";
           })

let to_int v = Int32.to_int (to_int32 v)

let to_float32 = function
  | VFloat32 f -> f
  | VFloat64 f -> F32.to_float32 f
  | VInt32 n -> F32.to_float32 (Int32.to_float n)
  | VInt64 n -> F32.to_float32 (Int64.to_float n)
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "float32";
             context = "to_float32";
           })

let to_float64 = function
  | VFloat64 f -> f
  | VFloat32 f -> f
  | VInt32 n -> Int32.to_float n
  | VInt64 n -> Int64.to_float n
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "float64";
             context = "to_float64";
           })

let to_bool = function
  | VBool b -> b
  | VInt32 n -> n <> 0l
  | VInt64 n -> n <> 0L
  | VFloat32 f -> f <> 0.0
  | VFloat64 f -> f <> 0.0
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "bool";
             context = "to_bool";
           })

(** {1 Binary Operations} *)

let eval_binop op v1 v2 =
  match op with
  | Add -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.add a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a +. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.add (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 +. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.add (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.add (to_int32 v1) (to_int32 v2)))
  | Sub -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.sub a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a -. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.sub (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 -. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.sub (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.sub (to_int32 v1) (to_int32 v2)))
  | Mul -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.mul a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a *. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.mul (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 *. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.mul (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.mul (to_int32 v1) (to_int32 v2)))
  | Div -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.div a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a /. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.div (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 /. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.div (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.div (to_int32 v1) (to_int32 v2)))
  | Mod -> (
      match (v1, v2) with
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.rem (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.rem (to_int32 v1) (to_int32 v2)))
  | Eq -> VBool (v1 = v2)
  | Ne -> VBool (v1 <> v2)
  | Lt -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a < b)
      | VFloat64 a, VFloat64 b -> VBool (a < b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 < to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 < to_float64 v2)
      | _ -> VBool (to_int32 v1 < to_int32 v2))
  | Le -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a <= b)
      | VFloat64 a, VFloat64 b -> VBool (a <= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 <= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 <= to_float64 v2)
      | _ -> VBool (to_int32 v1 <= to_int32 v2))
  | Gt -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a > b)
      | VFloat64 a, VFloat64 b -> VBool (a > b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 > to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 > to_float64 v2)
      | _ -> VBool (to_int32 v1 > to_int32 v2))
  | Ge -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a >= b)
      | VFloat64 a, VFloat64 b -> VBool (a >= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 >= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 >= to_float64 v2)
      | _ -> VBool (to_int32 v1 >= to_int32 v2))
  | And -> VBool (to_bool v1 && to_bool v2)
  | Or -> VBool (to_bool v1 || to_bool v2)
  | Shl -> VInt32 (Int32.shift_left (to_int32 v1) (to_int v2))
  | Shr -> VInt32 (Int32.shift_right_logical (to_int32 v1) (to_int v2))
  | BitAnd -> VInt32 (Int32.logand (to_int32 v1) (to_int32 v2))
  | BitOr -> VInt32 (Int32.logor (to_int32 v1) (to_int32 v2))
  | BitXor -> VInt32 (Int32.logxor (to_int32 v1) (to_int32 v2))

let eval_unop op v =
  match op with
  | Neg -> (
      match v with
      | VFloat32 f -> VFloat32 (-.f)
      | VFloat64 f -> VFloat64 (-.f)
      | VInt64 n -> VInt64 (Int64.neg n)
      | _ -> VInt32 (Int32.neg (to_int32 v)))
  | Not -> VBool (not (to_bool v))
  | BitNot -> VInt32 (Int32.lognot (to_int32 v))

(** {1 Intrinsics} *)

let is_gpu_path = function
  | ["Gpu"] | [] | ["Std"] | ["Sarek_stdlib"; "Gpu"] | ["Sarek_stdlib"; "Std"]
    ->
      true
  | _ -> false

let is_float32_path = function
  | ["Float32"] | ["Sarek_stdlib"; "Float32"] -> true
  | _ -> false

let is_float64_path = function
  | ["Float64"] | ["Sarek_stdlib"; "Float64"] -> true
  | _ -> false

let is_int32_path = function
  | ["Int32"] | ["Sarek_stdlib"; "Int32"] -> true
  | _ -> false
