(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_ir_types - Pure type definitions for GPU kernel IR

    This module contains only type definitions with no external dependencies.
    Used by spoc_framework for typed generate_source signature. *)

(** Runtime type identities with equality proofs.

    This uses generative extensible GADT constructors, so equality can be
    checked without unsafe casts. *)
module Type_id = struct
  type _ witness = ..

  module type ID = sig
    type t

    type _ witness += Id : t witness
  end

  type 'a t = (module ID with type t = 'a)

  type (_, _) eq = Refl : ('a, 'a) eq

  let create (type a) () : a t =
    let module M = struct
      type t = a

      type _ witness += Id : t witness
    end in
    (module M)

  let equal (type a b) (id_a : a t) (id_b : b t) : (a, b) eq option =
    let module A = (val id_a : ID with type t = a) in
    let module B = (val id_b : ID with type t = b) in
    match A.Id with B.Id -> Some Refl | _ -> None
end

(** Memory spaces *)
type memspace = Global | Shared | Local

(** Element types *)
type elttype =
  | TInt32
  | TInt64
  | TFloat16
      (** IEEE binary16 {e storage} type. Values are stored/loaded as binary16;
          arithmetic promotes to [TFloat32], computes there, and rounds back on
          store. There is deliberately no [CFloat16] constant: f16 values are
          produced by conversion ([ECast (TFloat16, _)]), never by a literal. *)
  | TFloat32
  | TFloat64
  | TUint8
      (** Unsigned 8-bit integer {e storage} type, on the [TFloat16] model: an
          array element occupies one byte, but there is no 8-bit value at run
          time and no 8-bit arithmetic. Reading an element gives an int32
          (zero-extended); storing an int32 keeps its low 8 bits;
          [ECast (TUint8, e)] is the int32 [e land 0xff]. Valid as the element
          type of a global ([TVec]) or shared ([DShared],
          [EArrayCreate (_, _, Shared)]) array and as a cast target; an 8-bit
          scalar binder, record field or private array is refused (see
          {!Sarek_ir_int8}, which holds the shared semantics).

          It is also the element type of an 8-bit cooperative-matrix operand
          buffer, of either signedness: [coopMatLoad] requires the backing
          array's element type to match the fragment's component type, and the
          s8 fragments of this IR read a [TUint8] buffer as signed. *)
  | TInt8
      (** Signed 8-bit integer storage type: as [TUint8], with the read
          sign-extended and [ECast (TInt8, e)] the low byte of [e]
          sign-extended. *)
  | TBool
  | TUnit
  | TRecord of string * (string * elttype) list
      (** Record type: name and field list *)
  | TVariant of string * (string * elttype list) list
      (** Variant type: name and constructor list with arg types *)
  | TArray of elttype * memspace
      (** Array type with element type and memory space *)
  | TVec of elttype  (** Vector (GPU array parameter) *)

(** Variables with type info *)
type var = {
  var_name : string;
  var_id : int;
  var_type : elttype;
  var_mutable : bool;
}

(** Constants *)
type const =
  | CInt32 of int32
  | CInt64 of int64
  | CFloat32 of float
  | CFloat64 of float
  | CBool of bool
  | CUnit

(** Binary operators *)
type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | Eq
  | Ne
  | Lt
  | Le
  | Gt
  | Ge
  | And
  | Or
  | Shl
  | Shr  (** Arithmetic (sign-extending) shift right: OCaml [asr]. *)
  | Lshr32
      (** Logical (zero-filling) shift right of a 32-bit operand: OCaml [lsr] on
          [int]/[int32]. Its own operator because the C-family emitters must
          spell the unsigned type and cannot infer it from the operands. *)
  | Lshr64  (** Logical shift right of a 64-bit operand: [lsr] on [int64]. *)
  | BitAnd
  | BitOr
  | BitXor

(** Unary operators *)
type unop = Neg | Not | BitNot

(** Loop direction *)
type for_dir = Upto | Downto

(** Match pattern *)
type pattern =
  | PConstr of string * string list (* Constructor name, bound vars *)
  | PWild

(** Expressions (pure, no side effects) *)
type expr =
  | EConst of const
  | EVar of var
  | EBinop of binop * expr * expr
  | EUnop of unop * expr
  | EArrayRead of string * expr  (** arr[idx] *)
  | EArrayReadExpr of expr * expr  (** base_expr[idx] for complex bases *)
  | ERecordField of expr * string  (** r.field *)
  | EIntrinsic of string list * string * expr list
      (** module path, name, args *)
  | ECast of elttype * expr
  | ETuple of expr list
  | EApp of expr * expr list
  | ERecord of string * (string * expr) list
      (** Record construction: type name, field values *)
  | EVariant of string * string * expr list
      (** Variant construction: type name, constructor, args *)
  | EArrayLen of string  (** Array length intrinsic *)
  | EArrayCreate of elttype * expr * memspace  (** elem type, size, memspace *)
  | EIf of expr * expr * expr  (** condition, then, else - value-returning if *)
  | EMatch of expr * (pattern * expr) list
      (** scrutinee, cases - value-returning match *)

(** L-values (assignable locations) *)
type lvalue =
  | LVar of var
  | LArrayElem of string * expr (* arr[idx] *)
  | LArrayElemExpr of expr * expr (* base_expr[idx] for complex bases *)
  | LRecordField of lvalue * string (* r.field *)

(** Statements (imperative, side effects) *)
type stmt =
  | SAssign of lvalue * expr
  | SSeq of stmt list
  | SIf of expr * stmt * stmt option
  | SWhile of expr * stmt
  | SFor of var * expr * expr * for_dir * stmt
  | SMatch of expr * (pattern * stmt) list
  | SReturn of expr
  | SBarrier  (** Block-level barrier (__syncthreads) *)
  | SWarpBarrier  (** Warp-level sync (__syncwarp) *)
  | SExpr of expr  (** Side-effecting expression *)
  | SEmpty
  | SLet of var * expr * stmt  (** Let binding: let v = e in body *)
  | SLetMut of var * expr * stmt  (** Mutable let: let v = ref e in body *)
  | SPragma of string list * stmt  (** Pragma hints wrapping a statement *)
  | SMemFence  (** Memory fence (threadfence) *)
  | SBlock of stmt
      (** Scoped block - creates a C scope for variable isolation *)
  | SNative of {
      gpu : framework:string -> string;  (** Generate GPU code for framework *)
      ocaml : ocaml_closure;  (** Typed OCaml fallback *)
    }  (** Inline native GPU code with OCaml fallback *)
  | SCoopmat of coopmat_op
      (** A cooperative-matrix (tensor-core) operation — backlog-62 slice 3.

          {b Why ONE statement constructor carrying an operation family, rather
             than four constructors.} Seventeen places in this repository match
          exhaustively on {!stmt}, and most of them are backends whose only
          correct response to any of these operations is the same refusal. Four
          constructors would be sixty-eight arms to write and to keep in
          agreement; one is seventeen, and a backend that handles [SCoopmat] at
          all is then forced by the compiler to consider every member of
          {!coopmat_op} in one place where the four cases sit next to each
          other.

          {b Why fragments are NOT [var]s and NOT an [elttype].} A fragment is a
          subgroup-cooperative value: the whole subgroup collectively holds
          [rows * columns] components and each invocation holds a few of them at
          an implementation-defined position. It cannot be indexed, assigned to,
          added, cast, passed to a helper, or stored in an array. Giving it an
          {!elttype} would make every one of those spellable in the IR and would
          oblige ~36 exhaustive [elttype] matches to invent an answer for a type
          none of them can represent. Fragments therefore live in their own
          namespace, addressed by name, and the only things that can be done to
          one are the four below. *)

(** The four things that can be done with a cooperative-matrix fragment.

    Fragment names live in a namespace of their own, separate from {!var}. They
    are plain strings for the same reason an [SShared] array name is: a fragment
    is not an l-value, cannot be captured, and cannot escape the kernel body, so
    there is nothing for a [var]'s mutability or type field to carry that
    {!Sarek_coopmat_types.fragment} does not already say. *)
and coopmat_op =
  | CM_decl of {name : string; frag : Sarek_coopmat_types.fragment}
      (** Bring a fragment into scope for the rest of the enclosing block.

          {b The fragment is zero-initialised} (every component, normative on
          every backend): a kernel may [CM_muladd] into a declared accumulator
          without loading it, and may leave a [CM_load] out entirely. The
          Interpreter zero-fills, CUDA emits [fill_fragment(f, 0)], HIP [= {}],
          GLSL [coopmat<..>(0)]. A backend that cannot guarantee this must
          refuse the declaration, never emit an uninitialised fragment.

          Statement-level rather than a scoping form like {!SLet}, because GLSL,
          MSL and C all admit a declaration in the middle of a block and because
          [D = A * B + C] wants four fragments live at once — nesting four
          [SLet]-shaped binders to express that is noise with no invariant
          behind it. *)
  | CM_load of {
      dst : string;
      frag : Sarek_coopmat_types.fragment;
      src : string;
      index : expr;
      stride : expr;
      layout : Sarek_coopmat_types.layout;
    }
      (** Fill [dst] from the buffer [src], element (0, 0) at [index]. With
          [layout = Row_major], [stride] elements separate consecutive rows
          (element (r, c) is at [index + r * stride + c]); with [Col_major],
          they separate consecutive columns ([index + c * stride + r]).

          [frag] is repeated here rather than looked up from the [CM_decl]: a
          codegen backend must be able to emit this statement without carrying a
          fragment environment, and an interpreter must be able to CHECK the two
          agree. A single source of truth that every consumer has to reconstruct
          is not a single source of truth.

          [layout] is required, not defaulted, so every construction site states
          which of the two it means. Column-major is what lets an operand stored
          transposed (a B kept as [n][k], token-major activations) be loaded
          with contiguous reads: on HIP gfx11 a column-major B fragment is one
          contiguous 16-element run per lane, a row-major one a stride-[stride]
          gather (specs/coopmat-hip-wmma.md). A backend that cannot lower a
          layout refuses the statement. *)
  | CM_store of {
      src : string;
      frag : Sarek_coopmat_types.fragment;
      dst : string;
      index : expr;
      stride : expr;
      layout : Sarek_coopmat_types.layout;
    }  (** The inverse of {!CM_load}, with the same [layout] semantics. *)
  | CM_muladd of {
      dst : string;
      a : string;
      b : string;
      c : string;
      cfg : Sarek_coopmat_types.config;
    }
      (** [dst = a * b + c], the tensor-core instruction itself.

          [cfg] is the whole point of carrying a configuration rather than four
          fragments: it is what the device gate is keyed on, it is what says
          whether the accumulation SATURATES (a property of the operation and
          not of any operand), and it is what
          {!Sarek_coopmat_types.accumulation_is_exact} reads to decide whether
          this statement is under the strict contract or needs the relaxation of
          docs/design/f16-relaxed-accuracy.md §1.6. *)

(** Declarations *)
and decl =
  | DParam of
      var * array_info option (* kernel parameter, optional array info *)
  | DLocal of var * expr option (* local variable, optional init *)
  | DShared of
      string * elttype * expr option (* shared array: name, elem type, size *)

and array_info = {arr_elttype : elttype; arr_memspace : memspace}

(** Helper function (device function called from kernel) *)
and helper_func = {
  hf_name : string;
  hf_params : var list;
  hf_ret_type : elttype;
  hf_body : stmt;
}

(** Native argument type for kernel execution. Typed arguments with runtime type
    witnesses - used by PPX-generated native functions. *)
and native_arg =
  | NA_Int32 of int32
  | NA_Int64 of int64
  | NA_Float32 of float
  | NA_Float64 of float
  | NA_Vec of native_vec

and native_vec = NV : ('elt, 'underlying) native_vec_ops -> native_vec

and ('elt, 'underlying) native_vec_ops = {
  length : int;
  elem_size : int;
  type_name : string;
  type_id : 'elt Type_id.t;
  get_f32 : int -> float;
  set_f32 : int -> float -> unit;
  get_f64 : int -> float;
  set_f64 : int -> float -> unit;
  get_i32 : int -> int32;
  set_i32 : int -> int32 -> unit;
  get_i64 : int -> int64;
  set_i64 : int -> int64 -> unit;
  get_typed : int -> 'elt;
  set_typed : int -> 'elt -> unit;
  underlying_type_id : 'underlying Type_id.t;
  underlying : 'underlying;
}

and ocaml_closure = {
  run :
    block:int * int * int -> grid:int * int * int -> native_arg array -> unit;
}

(** {2 Typed Helpers for Custom Types} *)

(** Get element from NA_Vec as a type checked custom value. *)
let vec_get_custom : type a. a Type_id.t -> native_arg -> int -> a =
 fun type_id arg i ->
  match arg with
  | NA_Vec (NV v) -> (
      match Type_id.equal type_id v.type_id with
      | Some Type_id.Refl -> v.get_typed i
      | None -> failwith "vec_get_custom: vector element type mismatch")
  | _ -> failwith "vec_get_custom: expected NA_Vec"

(** Set element in NA_Vec from a type checked custom value. *)
let vec_set_custom : type a. a Type_id.t -> native_arg -> int -> a -> unit =
 fun type_id arg i x ->
  match arg with
  | NA_Vec (NV v) -> (
      match Type_id.equal type_id v.type_id with
      | Some Type_id.Refl -> v.set_typed i x
      | None -> failwith "vec_set_custom: vector element type mismatch")
  | _ -> failwith "vec_set_custom: expected NA_Vec"

(** Get length from NA_Vec *)
let vec_length : native_arg -> int =
 fun arg ->
  match arg with
  | NA_Vec (NV v) -> v.length
  | _ -> failwith "vec_length: expected NA_Vec"

(** Get the checked underlying vector/buffer value. *)
let vec_as_vector : type a. a Type_id.t -> native_arg -> a =
 fun type_id arg ->
  match arg with
  | NA_Vec (NV v) -> (
      match Type_id.equal type_id v.underlying_type_id with
      | Some Type_id.Refl -> v.underlying
      | None -> failwith "vec_as_vector: underlying type mismatch")
  | _ -> failwith "vec_as_vector: expected NA_Vec"

(** Native function type for V2 execution. Uses typed native_arg. *)
type native_fn_t =
  | NativeFn of
      (parallel:bool ->
      block:int * int * int ->
      grid:int * int * int ->
      native_arg array ->
      unit)

(** Upper bound on the work-group size a kernel may be launched with.

    Declared in the kernel body with [pragma ["sarek.max_threads N"]]. It lets
    backends that compile for the worst case (1024 threads) lift the register
    cap. A kernel without the hint is generated exactly as before. *)
type launch_bounds = {max_threads : int}

(** Kernel representation *)
type kernel = {
  kern_name : string;
  kern_params : decl list;
  kern_locals : decl list;
  kern_body : stmt;
  kern_types : (string * (string * elttype) list) list;
      (** Record type definitions: (type_name, [(field_name, field_type); ...])
      *)
  kern_variants : (string * (string * elttype list) list) list;
      (** Variant type definitions: (type_name,
          [(constructor_name, payload_types); ...]) *)
  kern_funcs : helper_func list;
      (** Helper functions defined in kernel scope *)
  kern_native_fn : native_fn_t option;
      (** Optional pre-compiled native function for CPU execution *)
  kern_launch_bounds : launch_bounds option;
      (** Optional work-group size bound ([sarek.max_threads]); [None] means the
          backend default. *)
}

(** Bound of a kernel built from several kernels (fusion): the smallest declared
    bound, since the fused kernel must satisfy every member. [None] when no
    member declares one. *)
let combine_launch_bounds (bounds : launch_bounds option list) :
    launch_bounds option =
  List.fold_left
    (fun acc b ->
      match (acc, b) with
      | None, b -> b
      | a, None -> a
      | Some x, Some y -> Some {max_threads = min x.max_threads y.max_threads})
    None
    bounds

(** A kernel with every field at its empty value, for use as the base of a
    record update: [{default_kernel with kern_name = "k"; kern_body = b}].

    WHY THIS EXISTS. OCaml requires every field at every record literal, so
    adding one field to {!kernel} used to mean editing all 119 construction
    sites in the tree — which is why the type has been avoided rather than
    extended. A record UPDATE names only the fields it sets, so once a site is
    written this way a new field costs it nothing.

    Prefer this over spelling out the empty fields. {!make_kernel} is the same
    thing with labels, for new code that would otherwise set most fields. *)
let default_kernel =
  {
    kern_name = "";
    kern_params = [];
    kern_locals = [];
    kern_body = SEmpty;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
    kern_launch_bounds = None;
  }

(** {!default_kernel} with labels. [~name] and [~body] are required because a
    kernel with neither is not a kernel; everything else defaults to empty. *)
let make_kernel ?(params = []) ?(locals = []) ?(types = []) ?(variants = [])
    ?(funcs = []) ?native_fn ?launch_bounds ~name ~body () =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = locals;
    kern_body = body;
    kern_types = types;
    kern_variants = variants;
    kern_funcs = funcs;
    kern_native_fn = native_fn;
    kern_launch_bounds = launch_bounds;
  }
