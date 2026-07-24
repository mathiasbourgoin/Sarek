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

(** {1 Value bindings}

    Scalars bind to a single register name. Aggregates (records/variants,
    SROA-decomposed) bind to a structured register set. Nested records appear as
    nested [ARecord] values under their field name. *)

type agg_value =
  | ARecord of (string * binding) list
      (** Record value: one binding per field, in declaration order. *)
  | AVariant of {
      vname : string;  (** Variant type name (for error messages). *)
      tag_reg : string;  (** u32 register holding the constructor tag. *)
      ctors : (string * binding list) list;
          (** One payload slot list per constructor, ALL constructors in
              declaration order (a constructor's tag value = its position in
              this list). Slots of constructors other than the one a value was
              built with are freshly-allocated, never-written registers: their
              contents are undefined but never read dynamically (the tag never
              selects them); they exist so every value of a given variant type
              has a uniform full shape for leaf-wise copy/mov merging. *)
    }

and binding = Scalar of string | Agg of agg_value

(** {1 Register allocator} *)

(** Non-global state space of an array known to the emitter. Arrays absent from
    [arr_memspaces] are global (vector parameters). *)
type arr_space = SpaceShared | SpaceLocal

(** One scalar leaf of a Structure-of-Arrays (SoA) custom-vector parameter: the
    record field it comes from, its scalar type/byte size, and the u64 register
    holding its own device base pointer (each leaf lives in its own contiguous
    device buffer, so field access is a plain coalesced scalar-array access at
    that base). Populated by [emit_params] for parameters selected via
    [~soa_params]; consumed by the aggregate load/store paths. *)
type soa_leaf = {sl_field : string; sl_type : elttype; sl_base : string}

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
  arr_memspaces : (string, arr_space) Hashtbl.t;
  arr_soa : (string, soa_leaf list) Hashtbl.t;
      (** Custom-vector parameters lowered as Structure-of-Arrays: name -> its
          scalar leaves (in record declaration order), each carrying its own
          device base-pointer register. A name present here is SoA; absent means
          AoS (packed, single base pointer). Populated by [emit_params] from
          [~soa_params]; empty for every kernel compiled without SoA. *)
  shared_decls : Buffer.t;
      (** [.shared] declarations discovered while emitting the body (SLet-bound
          shared arrays); spliced into the kernel prologue by [generate]. *)
  local_decls : Buffer.t;
      (** [.local] declarations discovered while emitting the body (SLet-bound
          per-thread local arrays); spliced into the kernel prologue by
          [generate]. *)
  funcs : (string, helper_func) Hashtbl.t;
      (** Kernel helper functions ([kern_funcs]), inlined at EApp sites. *)
  variant_decls : (string, (string * elttype list) list) Hashtbl.t;
      (** Variant type declarations ([kern_variants]): type name -> constructors
          in declaration order. EVariant construction needs the full declaration
          (it only carries the type name) to allocate every constructor's
          payload slots and compute the tag index. *)
  mutable inline_stack : string list;
      (** Helper names currently being inlined — recursion guard. *)
  inline_budget : (string, int) Hashtbl.t;
      (** Remaining recursive-inline depth per helper, seeded from the helper's
          [pragma ["sarek.inline N"]] at first entry; consulted (and
          decremented) only on recursive re-entry. Helpers without the pragma
          have no entry and recursive re-entry stays an error. *)
  mutable inline_ret : (binding option * string) list;
      (** Per-inline (result binding, end label); SReturn inside an inlined body
          writes the binding (if any) and branches to the label instead of
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
    arr_soa = Hashtbl.create 4;
    shared_decls = Buffer.create 128;
    local_decls = Buffer.create 128;
    funcs = Hashtbl.create 4;
    variant_decls = Hashtbl.create 4;
    inline_stack = [];
    inline_budget = Hashtbl.create 4;
    inline_ret = [];
  }

(** [arr_space_of alloc name] is the registered non-global state space of array
    [name], or [None] for global arrays (vector parameters). *)
let arr_space_of a name = Hashtbl.find_opt a.arr_memspaces name

(** [is_soa alloc name] is true when the vector parameter [name] was lowered as
    Structure-of-Arrays (N per-leaf base pointers) rather than packed AoS. *)
let is_soa a name = Hashtbl.mem a.arr_soa name

(** [soa_leaves alloc name] are the SoA leaves of [name] in record declaration
    order. Raises if [name] is not SoA (callers guard with [is_soa]). *)
let soa_leaves a name = Hashtbl.find a.arr_soa name

(** [soa_leaf_of_field alloc name field] is the leaf of SoA vector [name] whose
    record field is [field], or [None] if there is no such leaf. *)
let soa_leaf_of_field a name field =
  match Hashtbl.find_opt a.arr_soa name with
  | None -> None
  | Some leaves -> List.find_opt (fun l -> l.sl_field = field) leaves

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

(** [binding_of_elttype alloc t] allocates a fresh binding with the register
    shape of type [t], without emitting any instruction: the registers are
    declared (they count toward the [.reg] block) but unwritten. Used to
    pre-allocate aggregate results (inlined helper returns) and the payload
    slots of not-constructed variant constructors. *)
let rec binding_of_elttype alloc = function
  | TRecord (_, fields) ->
      Agg
        (ARecord
           (List.map (fun (n, t) -> (n, binding_of_elttype alloc t)) fields))
  | TVariant (name, ctors) ->
      Agg
        (AVariant
           {
             vname = name;
             tag_reg = new_u32 alloc;
             ctors =
               List.map
                 (fun (cn, tys) ->
                   (cn, List.map (binding_of_elttype alloc) tys))
                 ctors;
           })
  | t -> Scalar (new_reg_for_type alloc t)

(** {1 Environment: variable name -> PTX binding} *)

type env = (string, binding) Hashtbl.t

let make_env () : env = Hashtbl.create 32

let env_bind (env : env) name reg = Hashtbl.replace env name (Scalar reg)

let env_bind_binding (env : env) name b = Hashtbl.replace env name b

let env_lookup (env : env) name =
  match Hashtbl.find_opt env name with
  | Some (Scalar r) -> r
  | Some (Agg _) ->
      fail
        ("PTX codegen: variable '" ^ name
       ^ "' is an aggregate (record/variant) and cannot be used in a scalar \
          context; read one of its scalar fields instead")
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

(** {1 Register-class helpers}

    A register's class is recovered from its PTX name prefix: [%fdN] = f64,
    [%fN] = f32, [%rdN] = u64, [%rN] = u32. *)

type reg_class = RU32 | RU64 | RF32 | RF64

let reg_class r =
  if String.length r >= 3 && r.[1] = 'f' && r.[2] = 'd' then RF64
  else if String.length r >= 2 && r.[1] = 'f' then RF32
  else if String.length r >= 3 && r.[1] = 'r' && r.[2] = 'd' then RU64
  else RU32

let mov_op_of_class = function
  | RU32 -> "mov.u32"
  | RU64 -> "mov.u64"
  | RF32 -> "mov.f32"
  | RF64 -> "mov.f64"

(** [new_reg_like alloc r] allocates a fresh register of [r]'s class. *)
let new_reg_like a r =
  match reg_class r with
  | RU32 -> new_u32 a
  | RU64 -> new_u64 a
  | RF32 -> new_f32 a
  | RF64 -> new_f64 a

(** [mov_scalar buf ~dst ~src] emits a typed mov of [src] into [dst]; the mov
    type is selected from [dst]'s register class. *)
let mov_scalar buf ~dst ~src =
  emit buf "%s %s, %s;" (mov_op_of_class (reg_class dst)) dst src

(** [copy_reg buf alloc r] allocates a fresh register of [r]'s class and emits a
    mov from [r] into it. Used wherever a binding must not alias the source
    register (mutable lets, inlined helper parameters). *)
let copy_reg buf a r_src =
  let r = new_reg_like a r_src in
  mov_scalar buf ~dst:r ~src:r_src ;
  r

(** [copy_binding buf alloc b] copies every scalar leaf of [b] into fresh
    registers (leaf-wise {!copy_reg}), preserving the aggregate shape. Used for
    mutation isolation (mutable lets, inlined helper parameters bound to
    aggregates). *)
let rec copy_binding buf a = function
  | Scalar r -> Scalar (copy_reg buf a r)
  | Agg (ARecord fields) ->
      Agg (ARecord (List.map (fun (n, b) -> (n, copy_binding buf a b)) fields))
  | Agg (AVariant {vname; tag_reg; ctors}) ->
      Agg
        (AVariant
           {
             vname;
             tag_reg = copy_reg buf a tag_reg;
             ctors =
               List.map
                 (fun (cn, bs) -> (cn, List.map (copy_binding buf a) bs))
                 ctors;
           })

(** [mov_binding buf ~src ~dst] emits leaf-wise typed movs of [src]'s registers
    into [dst]'s registers. The two bindings must have compatible shapes
    (records matched by field name; variant payload slots matched by constructor
    tag — [src] slots absent from [dst] are an error, [dst] slots absent from
    [src] are left untouched). *)
let rec mov_binding buf ~src ~dst =
  match (src, dst) with
  | Scalar s, Scalar d -> mov_scalar buf ~dst:d ~src:s
  | Agg (ARecord fs), Agg (ARecord fd) ->
      List.iter
        (fun (name, db) ->
          match List.assoc_opt name fs with
          | Some sb -> mov_binding buf ~src:sb ~dst:db
          | None ->
              fail
                ("PTX codegen: internal error: record shape mismatch in \
                  aggregate mov (missing field '" ^ name ^ "')"))
        fd
  | Agg (AVariant vs), Agg (AVariant vd) ->
      mov_scalar buf ~dst:vd.tag_reg ~src:vs.tag_reg ;
      List.iter
        (fun (cn, sbs) ->
          match List.assoc_opt cn vd.ctors with
          | Some dbs when List.length dbs = List.length sbs ->
              List.iter2 (fun sb db -> mov_binding buf ~src:sb ~dst:db) sbs dbs
          | _ ->
              fail
                (Printf.sprintf
                   "PTX codegen: internal error: variant shape mismatch in \
                    aggregate mov (constructor '%s' of type '%s')"
                   cn
                   vs.vname))
        vs.ctors
  | _ ->
      fail
        "PTX codegen: internal error: aggregate shape mismatch in mov \
         (scalar/record/variant kinds differ)"

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
