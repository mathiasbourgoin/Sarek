(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek_ir_interp - CPU interpreter for Sarek V2 IR kernels
 *
 * Executes Sarek_ir.kernel on CPU for debugging and testing without GPU.
 * Supports BSP-style barrier synchronization using OCaml 5 effects.
 *
 * This is the V2 counterpart to Sarek_interp (which works on Kirc_Ast).
 ******************************************************************************)

open Sarek_ir_types
open Spoc_framework.Typed_value
open Sarek_ir_interp_eval

(** Re-export value type and constructors for external callers *)
type value = Sarek_ir_interp_value.value =
  | VInt32 of int32
  | VInt64 of int64
  | VFloat32 of float
  | VFloat64 of float
  | VBool of bool
  | VUnit
  | VArray of value array
  | VRecord of string * value array
  | VVariant of string * int * value list

let to_int = Sarek_ir_interp_value.to_int

let to_int32 = Sarek_ir_interp_value.to_int32

let to_int64 = Sarek_ir_interp_value.to_int64

let to_float32 = Sarek_ir_interp_value.to_float32

let to_float64 = Sarek_ir_interp_value.to_float64

let to_bool = Sarek_ir_interp_value.to_bool

(** Re-export thread_state type for external callers *)
type thread_state = Sarek_ir_interp_value.thread_state = {
  thread_idx : int * int * int;
  block_idx : int * int * int;
  block_dim : int * int * int;
  grid_dim : int * int * int;
}

(** Re-export env type for external callers *)
type env = Sarek_ir_interp_value.env = {
  vars : (int, value) Hashtbl.t;
  vars_by_name : (string, value) Hashtbl.t;
  arrays : (string, value array) Hashtbl.t;
  shared : (string, value array) Hashtbl.t;
  funcs : (string, Sarek_ir_types.helper_func) Hashtbl.t;
  policy : Sarek_policy.Budget.t;
  coopmats : (int * string, value array) Hashtbl.t;
}

let create_env = Sarek_ir_interp_value.create_env

let copy_env = Sarek_ir_interp_value.copy_env

let bind_var = Sarek_ir_interp_value.bind_var

let lookup_var = Sarek_ir_interp_value.lookup_var

let eval_binop = Sarek_ir_interp_value.eval_binop

let eval_unop = Sarek_ir_interp_value.eval_unop

let eval_expr = Sarek_ir_interp_eval.eval_expr

let exec_stmt = Sarek_ir_interp_eval.exec_stmt

(** {1 Parallel Execution}

    Blocks run on {!Cpu_domain_pool}, shared with the Native barrier path,
    created on the first parallel launch and retired when idle. *)

(** Run all blocks in a grid (parallel - distributes blocks across domain pool)
*)
let run_grid_parallel env body block_dim grid_dim =
  let gx, gy, gz = grid_dim in
  (* Blocks in x-fastest order, one task each. *)
  let run_block_task b () =
    let bx = b mod gx and by = b / gx mod gy and bz = b / (gx * gy) in
    (* Each block gets its own NEW shared memory hashtable. Don't use
       copy_env.shared because it's shared by reference. *)
    let block_env =
      {
        (copy_env env) with
        shared = Hashtbl.create 8;
        coopmats = Hashtbl.create 4;
      }
    in
    run_block block_env body (bx, by, bz) block_dim grid_dim
  in
  (* As nested per-axis loops: no block when any extent is not positive. *)
  let blocks = if gx > 0 && gy > 0 && gz > 0 then gx * gy * gz else 0 in
  Cpu_domain_pool.run blocks (fun b -> run_block_task b ())

(** Parallel execution mode flag *)
let parallel_mode = ref true

(** Run all blocks in a grid (uses parallel or sequential based on flag) *)
let run_grid env body block_dim grid_dim =
  if !parallel_mode then run_grid_parallel env body block_dim grid_dim
  else run_grid_sequential env body block_dim grid_dim

(** {1 Public API} *)

(** Argument for kernel execution *)
type arg = ArgArray of value array | ArgScalar of value

(** Run a kernel on CPU *)
let run_kernel (k : kernel) ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (args : (string * arg) list) =
  (* 8-bit storage types: refuse what no backend accepts, then make the
     narrowing of every 8-bit store and the extension of every 8-bit read
     explicit, so the evaluator implements them through its [ECast] arm alone.
     A store into a [Vector.char] buffer would otherwise keep a value outside
     0..255 that the host write-back cannot represent. *)
  (match Sarek_ir_int8.refusal k with
  | None -> ()
  | Some reason ->
      Interp_error.raise_error
        (Unsupported_operation {operation = "8-bit storage type"; reason})) ;
  let k = Sarek_ir_int8.normalise k in
  let env =
    create_env ~policy:(Sarek_policy.budget ~kernels:[k.kern_name]) ()
  in

  (* Register helper functions *)
  List.iter (fun hf -> Hashtbl.add env.funcs hf.hf_name hf) k.kern_funcs ;

  (* Bind parameters *)
  List.iter2
    (fun decl (name, arg) ->
      match (decl, arg) with
      | DParam (v, Some _), ArgArray arr ->
          Hashtbl.add env.arrays name arr ;
          bind_var env v (VArray arr)
      | DParam (v, None), ArgScalar value -> bind_var env v value
      | DShared (name, ty, Some size_expr), _ ->
          let dummy_state =
            {
              thread_idx = (0, 0, 0);
              block_idx = (0, 0, 0);
              block_dim = (bx, by, bz);
              grid_dim = (gx, gy, gz);
            }
          in
          let size = to_int (eval_expr dummy_state env size_expr) in
          Hashtbl.add
            env.shared
            name
            (Sarek_ir_interp_value.alloc_kernel_array ty size)
      | _ -> ())
    k.kern_params
    args ;

  run_grid env k.kern_body (bx, by, bz) (gx, gy, gz)

(** {1 V2 Vector Support}

    These functions work with typed Kernel_arg.t values. This is the preferred
    interface for Native/Interpreter backends. *)

(** Convert V2 Vector to interpreter value array. Uses the vector's element type
    to create properly typed values. *)
let vector_to_array : type a b. (a, b) Spoc_core.Vector.t -> value array =
 fun vec ->
  let len = Spoc_core.Vector.length vec in
  match Spoc_core.Vector.kind vec with
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int32 ->
      Array.init len (fun i -> VInt32 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int64 ->
      Array.init len (fun i -> VInt64 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float16 ->
      (* f16 is a STORAGE type: there is no VFloat16. [get] already returns the
         binary16-rounded value as an OCaml float, and the interpreter computes
         in f32 — so an f16 element enters the interpreter as VFloat32. This is
         the "reuse the f32 accessor, let the storage kind do the rounding"
         decision: rounding is a property of the Bigarray.Float16 cell, not of
         the accessor. *)
      Array.init len (fun i -> VFloat32 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float32 ->
      Array.init len (fun i -> VFloat32 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float64 ->
      Array.init len (fun i -> VFloat64 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Custom custom -> (
      (* Custom types: use helpers to convert to VRecord *)
      let type_name = custom.Spoc_core.Vector.name in
      match Sarek_type_helpers.lookup_typed custom.Spoc_core.Vector.type_id with
      | Some (module H) ->
          Array.init len (fun i ->
              let native_record = Spoc_core.Vector.get vec i in
              H.to_value native_record)
      | None ->
          (* Fallback: wrap in VRecord with empty fields - shouldn't happen *)
          Array.init len (fun _i -> VRecord (type_name, [||])))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Char ->
      Array.init len (fun i ->
          VInt32 (Int32.of_int (Char.code (Spoc_core.Vector.get vec i))))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int8 ->
      (* [Bigarray.int8_signed] reads back -128..127: the sign-extended int32 an
         8-bit read yields (Sarek_ir_int8). *)
      Array.init len (fun i ->
          VInt32 (Int32.of_int (Spoc_core.Vector.get vec i)))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Complex32 ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "vector_to_array";
             reason = "Complex32 vectors are not supported by the interpreter";
           })

(** Write interpreter value array back to V2 Vector *)
let array_to_vector : type a b. value array -> (a, b) Spoc_core.Vector.t -> unit
    =
 fun arr vec ->
  let len = min (Array.length arr) (Spoc_core.Vector.length vec) in
  match Spoc_core.Vector.kind vec with
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int32 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_int32 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int64 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_int64 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float16 ->
      (* Writing back through a Bigarray.Float16 cell rounds to binary16 for
         free — this is where "round on store" actually happens on the
         interpreter and native paths. *)
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_float32 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float32 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_float32 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float64 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_float64 arr.(i))
      done
  | Spoc_core.Vector.Custom _ ->
      (* Custom types: convert VRecord to native OCaml values using helpers *)
      for i = 0 to len - 1 do
        match arr.(i) with
        | VRecord (type_name, _fields) as vrec -> (
            (* All [@@sarek.type] records have helpers - this should always succeed *)
            match
              Sarek_type_helpers.lookup_typed
                (Spoc_core.Vector.type_id (Spoc_core.Vector.kind vec))
            with
            | Some (module H) ->
                (* Use generated helper for type-safe conversion *)
                let native_record = H.from_value vrec in
                Spoc_core.Vector.set vec i native_record
            | None ->
                Interp_error.raise_error
                  (Unsupported_operation
                     {
                       operation = "vector_to_array";
                       reason =
                         Printf.sprintf
                           "No helper found for type '%s'. Did you forget \
                            [@@sarek.type]?"
                           type_name;
                     }))
        | _ -> () (* Skip other values *)
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Char ->
      (* The low byte, as a device store keeps it: every store into an 8-bit
         array is already narrowed (Sarek_ir_int8.normalise), and [Char.chr]
         would raise on anything else rather than wrap. *)
      for i = 0 to len - 1 do
        Spoc_core.Vector.set
          vec
          i
          (Char.chr (Int32.to_int (to_int32 arr.(i)) land 0xff))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int8 ->
      (* [Bigarray.int8_signed] keeps the low 8 bits of the [int] it is given. *)
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (Int32.to_int (to_int32 arr.(i)))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Complex32 ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "array_to_vector";
             reason = "Complex32 vectors are not supported by the interpreter";
           })

(** Existential wrapper to track V2 Vector + its interpreter array for writeback
*)
type writeback =
  | Writeback : (('a, 'b) Spoc_core.Vector.t * value array) -> writeback

type exec_writeback =
  | Exec_writeback :
      (module Spoc_framework.Typed_value.EXEC_VECTOR) * value array
      -> exec_writeback

(** Whether two vectors are the same vector.

    Physical identity, taken on the vector's own [device_buffers] table: it is
    allocated once per vector and never replaced, it has one type whatever the
    vectors' element types (so two vectors of different types can be compared
    without [Obj]), and a comparison of the records themselves would need both
    to have the same type. The vector id is deliberately NOT used: ids come from
    a non-atomic counter and can collide across domains (R-1). *)
let same_vector : type a b c d.
    (a, b) Spoc_core.Vector.t -> (c, d) Spoc_core.Vector.t -> bool =
 fun x y ->
  x.Spoc_core.Vector.device_buffers == y.Spoc_core.Vector.device_buffers

(** The value array bound to [vec]: the one already bound among [writebacks] if
    [vec] is a vector passed before, otherwise a fresh one from [to_array],
    which is recorded in [writebacks].

    Two kernel parameters bound to the same vector must share ONE value array: a
    private copy each would make a write through one invisible to a read through
    the other, and the writebacks would then overwrite each other in an order
    that depends on the entry point. Binding by identity gives the semantics of
    the device, where both parameters are the same buffer. *)
let bind_vector (type a b) (writebacks : writeback list ref)
    (vec : (a, b) Spoc_core.Vector.t)
    ~(to_array : (a, b) Spoc_core.Vector.t -> value array) : value array =
  let bound =
    List.find_map
      (fun (Writeback (other, arr)) ->
        if same_vector vec other then Some arr else None)
      !writebacks
  in
  match bound with
  | Some arr -> arr
  | None ->
      let arr = to_array vec in
      writebacks := Writeback (vec, arr) :: !writebacks ;
      arr

(* Tuple vector elements (L13) reach the interpreter as a [COMPOSITE_TYPE] whose
   [fields] descriptor list is empty but whose [to_bytes]/[of_bytes] marshal the
   raw device-layout element bytes. The tuple-shape byte layout (per-field type
   and offset) is resolved from [Sarek_tuple_vec], the single authority that
   built the host vector, and used to decode those bytes into a positional
   [VRecord] (fields [_0], [_1], ...) and re-encode them on writeback. *)
let value_of_bytes (b : bytes) (fl : Sarek_tuple_vec.field_layout) : value =
  let off = fl.Sarek_tuple_vec.fl_offset in
  match fl.Sarek_tuple_vec.fl_elttype with
  | Sarek_ir_types.TFloat32 ->
      VFloat32 (Int32.float_of_bits (Bytes.get_int32_le b off))
  | Sarek_ir_types.TFloat64 ->
      VFloat64 (Int64.float_of_bits (Bytes.get_int64_le b off))
  | Sarek_ir_types.TInt32 -> VInt32 (Bytes.get_int32_le b off)
  | Sarek_ir_types.TInt64 -> VInt64 (Bytes.get_int64_le b off)
  | Sarek_ir_types.TBool -> VBool (Bytes.get_int8 b off <> 0)
  | _ ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "value_of_bytes";
             reason = "unsupported tuple component type";
           })

let bytes_of_shape (shape : Sarek_tuple_vec.shape) (fields : value array) :
    bytes =
  let b = Bytes.make shape.Sarek_tuple_vec.sh_size '\000' in
  List.iteri
    (fun idx fl ->
      let off = fl.Sarek_tuple_vec.fl_offset in
      let v = if idx < Array.length fields then fields.(idx) else VUnit in
      match fl.Sarek_tuple_vec.fl_elttype with
      | Sarek_ir_types.TFloat32 ->
          Bytes.set_int32_le b off (Int32.bits_of_float (to_float32 v))
      | Sarek_ir_types.TFloat64 ->
          Bytes.set_int64_le b off (Int64.bits_of_float (to_float64 v))
      | Sarek_ir_types.TInt32 -> Bytes.set_int32_le b off (to_int32 v)
      | Sarek_ir_types.TInt64 -> Bytes.set_int64_le b off (to_int64 v)
      | Sarek_ir_types.TBool ->
          Bytes.set_int8 b off (if to_bool v then 1 else 0)
      | _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {
                 operation = "bytes_of_shape";
                 reason = "unsupported tuple component type";
               }))
    shape.Sarek_tuple_vec.sh_fields ;
  b

let value_of_typed_value (tv : Spoc_framework.Typed_value.typed_value) : value =
  match tv with
  | TV_Scalar (SV ((module S), x)) -> (
      match S.to_primitive x with
      | PInt32 n -> VInt32 n
      | PInt64 n -> VInt64 n
      | PFloat f -> if S.name = "float64" then VFloat64 f else VFloat32 f
      | PBool b -> VBool b
      | PBytes _ ->
          Interp_error.raise_error
            (Unsupported_operation
               {operation = "value_of_typed_value"; reason = "PBytes scalar"}))
  | TV_Composite (CV ((module C), x)) -> (
      match Sarek_tuple_vec.lookup_shape C.name with
      | Some shape ->
          let b = C.to_bytes x in
          VRecord
            ( C.name,
              Array.of_list
                (List.map (value_of_bytes b) shape.Sarek_tuple_vec.sh_fields) )
      | None -> VRecord (C.name, [||]))

let typed_value_of_value (type a)
    (module V : Spoc_framework.Typed_value.EXEC_VECTOR with type elt = a)
    (v : value) : Spoc_framework.Typed_value.typed_value option =
  match v with
  | VInt32 n ->
      Some (TV_Scalar (SV ((module Spoc_framework.Typed_value.Int32_type), n)))
  | VInt64 n ->
      Some (TV_Scalar (SV ((module Spoc_framework.Typed_value.Int64_type), n)))
  | VFloat32 f ->
      Some
        (TV_Scalar (SV ((module Spoc_framework.Typed_value.Float32_type), f)))
  | VFloat64 f ->
      Some
        (TV_Scalar (SV ((module Spoc_framework.Typed_value.Float64_type), f)))
  | VBool b ->
      Some (TV_Scalar (SV ((module Spoc_framework.Typed_value.Bool_type), b)))
  | VUnit | VArray _ | VRecord _ | VVariant _ -> None

let exec_vector_to_array (module V : Spoc_framework.Typed_value.EXEC_VECTOR) :
    value array =
  Array.init V.length (fun i ->
      match Sarek_type_helpers.lookup_typed V.type_id with
      | Some (module H) -> H.to_value (V.get_typed i)
      | None -> value_of_typed_value (V.get i))

let array_to_exec_vector (module V : Spoc_framework.Typed_value.EXEC_VECTOR)
    (arr : value array) : unit =
  let len = min (Array.length arr) V.length in
  for i = 0 to len - 1 do
    match arr.(i) with
    | VVariant (type_name, _, _) as vv -> (
        (* Standalone variant vector element: the [@@sarek.type] variant helper
           decodes the [VVariant] back to the native constructor, then the
           composite's typed setter writes the [tag|payload] bytes. *)
        match Sarek_type_helpers.lookup_typed V.type_id with
        | Some (module H) -> V.set_typed i (H.from_value vv)
        | None ->
            (* A [@@sarek.type] variant element always has a registered helper;
               a missing one is a registration bug. Surface it (mirrors
               Execute.interp_array_to_vector) rather than silently dropping the
               write and producing wrong output. Unlike the VRecord arm there is
               no tuple-shape fallback for variants. *)
            Interp_error.raise_error
              (Unsupported_operation
                 {
                   operation = "variant vector writeback";
                   reason =
                     "no registered [@@sarek.type] helper for variant element \
                      type '" ^ type_name ^ "'";
                 }))
    | VRecord (name, fields) as vrec -> (
        match Sarek_type_helpers.lookup_typed V.type_id with
        | Some (module H) -> V.set_typed i (H.from_value vrec)
        | None -> (
            (* Tuple vector element (L13): re-encode the positional record to
               raw element bytes via the resolved shape layout, then round-trip
               through the composite's [of_bytes] to set the element. *)
            match Sarek_tuple_vec.lookup_shape name with
            | Some shape -> (
                match V.get i with
                | Spoc_framework.Typed_value.TV_Composite (CV ((module C), _))
                  ->
                    let b = bytes_of_shape shape fields in
                    V.set
                      i
                      (Spoc_framework.Typed_value.TV_Composite
                         (Spoc_framework.Typed_value.CV
                            ((module C), C.of_bytes b)))
                | _ -> ())
            | None -> ()))
    | scalar -> (
        match typed_value_of_value (module V) scalar with
        | Some tv -> V.set i tv
        | None -> ())
  done

(** Whether two [EXEC_VECTOR] wrappers stand for the same vector.

    Their [underlying] values can only be compared once their types are known to
    be equal, which the type ids witness. [Execute.exec_arg_of_vector] gives
    every wrapper of a [Vector.t] the SAME stable [underlying_type_id] (per
    element kind) and [underlying = v], so two arguments that are one vector
    compare physically equal. A wrapper that mints a fresh id per instance (the
    interpreter plugin's own buffer wrapper) never matches: it wraps a device
    buffer, not a host vector, and no launch built from host vectors reaches it.
*)
let same_exec_vector (module V : Spoc_framework.Typed_value.EXEC_VECTOR)
    (module W : Spoc_framework.Typed_value.EXEC_VECTOR) : bool =
  match
    Sarek_ir_types.Type_id.equal V.underlying_type_id W.underlying_type_id
  with
  | Some Sarek_ir_types.Type_id.Refl -> V.underlying == W.underlying
  | None -> false

(** {!bind_vector} for an [EXEC_VECTOR] argument. *)
let bind_exec_vector (writebacks : exec_writeback list ref)
    (vec : (module Spoc_framework.Typed_value.EXEC_VECTOR)) : value array =
  let bound =
    List.find_map
      (fun (Exec_writeback (other, arr)) ->
        if same_exec_vector vec other then Some arr else None)
      !writebacks
  in
  match bound with
  | Some arr -> arr
  | None ->
      let arr = exec_vector_to_array vec in
      writebacks := Exec_writeback (vec, arr) :: !writebacks ;
      arr

let args_from_exec_args (k : kernel)
    (exec_args : Spoc_framework.Framework_sig.exec_arg list) :
    (string * arg) list * exec_writeback list =
  let writebacks = ref [] in
  let idx = ref 0 in
  let arg_at i =
    if i >= List.length exec_args then None else Some (List.nth exec_args i)
  in
  let args =
    List.filter_map
      (fun decl ->
        match decl with
        | DParam (v, Some _arr_info) -> (
            match arg_at !idx with
            | None -> None
            | Some karg -> (
                incr idx ;
                match karg with
                | Spoc_framework.Framework_sig.EA_Vec vec ->
                    let arr = bind_exec_vector writebacks vec in
                    Some (v.var_name, ArgArray arr)
                | _ ->
                    Interp_error.raise_error
                      (Type_conversion_error
                         {
                           from_type = "scalar";
                           to_type = "Vec";
                           context = "param " ^ v.var_name;
                         })))
        | DParam (v, None) -> (
            match arg_at !idx with
            | None -> None
            | Some karg ->
                incr idx ;
                let value =
                  match karg with
                  | Spoc_framework.Framework_sig.EA_Int32 n -> VInt32 n
                  | Spoc_framework.Framework_sig.EA_Int64 n -> VInt64 n
                  | Spoc_framework.Framework_sig.EA_Float32 f -> VFloat32 f
                  | Spoc_framework.Framework_sig.EA_Float64 f -> VFloat64 f
                  | Spoc_framework.Framework_sig.EA_Scalar ((module S), x) ->
                      value_of_typed_value (TV_Scalar (SV ((module S), x)))
                  | Spoc_framework.Framework_sig.EA_Composite ((module C), x) ->
                      value_of_typed_value (TV_Composite (CV ((module C), x)))
                  | Spoc_framework.Framework_sig.EA_Vec _ ->
                      Interp_error.raise_error
                        (Type_conversion_error
                           {
                             from_type = "non-scalar";
                             to_type = "scalar";
                             context = "param " ^ v.var_name;
                           })
                in
                Some (v.var_name, ArgScalar value))
        | DShared _ | DLocal _ -> None)
      k.kern_params
  in
  (args, List.rev !writebacks)

let run_kernel_with_exec_args (k : kernel) ~(block : int * int * int)
    ~(grid : int * int * int)
    (exec_args : Spoc_framework.Framework_sig.exec_arg list) : unit =
  let args, writebacks = args_from_exec_args k exec_args in
  run_kernel k ~block ~grid args ;
  List.iter
    (fun (Exec_writeback (vec, arr)) -> array_to_exec_vector vec arr)
    writebacks

(** Convert Kernel_arg.t list to interpreter args, tracking vectors for
    writeback *)
let args_from_kernel_args (k : kernel) (kargs : Spoc_core.Kernel_arg.t list) :
    (string * arg) list * writeback list =
  let writebacks = ref [] in
  let idx = ref 0 in
  let args =
    List.filter_map
      (fun decl ->
        match decl with
        | DParam (v, Some _arr_info) ->
            (* Vector parameter: expects a Vec in Kernel_arg *)
            if !idx >= List.length kargs then None
            else begin
              let karg = List.nth kargs !idx in
              incr idx ;
              match karg with
              | Spoc_core.Kernel_arg.Vec vec ->
                  let arr =
                    bind_vector writebacks vec ~to_array:vector_to_array
                  in
                  Some (v.var_name, ArgArray arr)
              | _ ->
                  Interp_error.raise_error
                    (Type_conversion_error
                       {
                         from_type = "scalar";
                         to_type = "Vec";
                         context = "param " ^ v.var_name;
                       })
            end
        | DParam (v, None) ->
            (* Scalar parameter *)
            if !idx >= List.length kargs then None
            else begin
              let karg = List.nth kargs !idx in
              incr idx ;
              let value =
                match karg with
                | Spoc_core.Kernel_arg.Int n -> VInt32 (Int32.of_int n)
                | Spoc_core.Kernel_arg.Int32 n -> VInt32 n
                | Spoc_core.Kernel_arg.Int64 n -> VInt64 n
                | Spoc_core.Kernel_arg.Float32 f -> VFloat32 f
                | Spoc_core.Kernel_arg.Float64 f -> VFloat64 f
                | Spoc_core.Kernel_arg.Vec _ ->
                    Interp_error.raise_error
                      (Type_conversion_error
                         {
                           from_type = "non-scalar";
                           to_type = "scalar";
                           context = "param " ^ v.var_name;
                         })
              in
              Some (v.var_name, ArgScalar value)
            end
        | DShared _ -> None
        | DLocal _ -> None)
      k.kern_params
  in
  (args, List.rev !writebacks)

(** Run kernel with V2 Vector arguments (Kernel_arg.t list). This is the
    preferred entry point for Native/Interpreter backends. Handles conversion
    to/from interpreter format with proper writeback. *)
let run_kernel_with_args (k : kernel) ~(block : int * int * int)
    ~(grid : int * int * int) (kargs : Spoc_core.Kernel_arg.t list) : unit =
  let args, writebacks = args_from_kernel_args k kargs in
  run_kernel k ~block ~grid args ;
  (* Write modified arrays back to V2 Vectors *)
  List.iter (fun (Writeback (vec, arr)) -> array_to_vector arr vec) writebacks
