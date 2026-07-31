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
  coopmats : (string, value array) Hashtbl.t;
      (** Cooperative-matrix fragments, by fragment name — backlog-62 slice 3.

          A SEPARATE table from {!vars} and {!arrays} because fragment names are
          a separate namespace in the IR: a fragment is not a variable and not
          an array, and merging it into either would make a name collision
          between a fragment and a variable silently resolve to one of them.

          {b The model.} A subgroup-scope fragment is held collectively by the
          whole subgroup, with each invocation holding a few components at an
          implementation-defined position. The interpreter has no subgroup, so
          it holds the WHOLE matrix redundantly in every invocation and has
          every invocation perform the whole operation. That is observationally
          equivalent to the device — a [coopMatStore] then writes the same value
          to the same location once per invocation instead of once per subgroup
          — precisely BECAUSE GL_KHR_cooperative_matrix requires the buffer,
          index and stride arguments to be dynamically uniform across the scope.
          The redundancy is not an approximation; it is the same function
          computed the only way a scalar interpreter can compute it. *)
}

let create_env () =
  {
    vars = Hashtbl.create 32;
    vars_by_name = Hashtbl.create 32;
    arrays = Hashtbl.create 16;
    shared = Hashtbl.create 8;
    funcs = Hashtbl.create 8;
    coopmats = Hashtbl.create 4;
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
    coopmats = Hashtbl.copy env.coopmats;
    (* per-invocation, like [vars]: a fragment does not outlive the block that
       declared it and is never shared between threads. *)
  }

(** Environment for a HELPER CALL: the callee's own scope, not a copy of the
    caller's.

    [copy_env] duplicates [vars]/[vars_by_name], which is right for a nested
    block — a block sees its enclosing locals — and wrong for a function call:
    carrying the caller's bindings in makes the callee's lookups depend on ids
    and names not in its scope, and since [lookup_var] resolves by id before
    name, a caller binding can answer a callee reference.

    "A helper sees only its parameters" would be the clean statement and it is
    NOT true of this language: a module-level constant ([MConst]) is declared
    before the helpers and IS lexically visible in their bodies, yet lowering
    emits it as an [SLet] at the head of the KERNEL body — into [vars], the one
    table this does not alias. Such a kernel is broken on both sides of this
    change today (it fails on the base revision too, for an unrelated
    positional-id reason), so nothing regresses here — but the scope model below
    is narrower than the surface language, and the two must be reconciled BEFORE
    the [hf_params] follow-up lands: that fix would turn a masked failure into a
    hard [Unbound_variable] which [copy_env] was accidentally covering.

    No test distinguishes this from copy semantics: reverting it leaves the
    whole suite green, measured by two independent review passes. That is not
    because it is decoration — it is the PRECONDITION that makes [get_array]'s
    new precedence sound. That lookup consults [vars_by_name] BEFORE [arrays] so
    a helper's formal shadows a kernel array of the same name; under copy
    semantics [vars_by_name] also holds the CALLER's locals, so the same
    precedence would let a caller binding answer a lookup that should reach the
    kernel's array. The two changes are one change and land together.

    No test separates them because a caller local holding an array value is not
    expressible today: kernel vectors live in [arrays], and a [let rec] in the
    kernel body — which could close over one — is rejected at parse time. So the
    dependency is argued, not measured, and this says which of the two it is.

    [coopmats] is fresh where [copy_env] copied it. A fragment belongs to the
    block that declared it, which is right for one the helper declares itself;
    if coopmat ops in helper bodies ever become expressible against a
    KERNEL-scope fragment, this must alias [coopmats] too.

    [arrays], [shared] and [funcs] stay ALIASED, deliberately: kernel buffers,
    block-shared memory and the helper table are genuinely global to the
    invocation, and [arrays] in particular is shared across threads by design,
    so it must not be copied. *)
let callee_env env =
  {
    vars = Hashtbl.create 8;
    vars_by_name = Hashtbl.create 8;
    arrays = env.arrays;
    shared = env.shared;
    funcs = env.funcs;
    coopmats = Hashtbl.create 4;
  }

(** Detach a record value from whatever container it was read out of, so that a
    LOCAL holding it cannot write back through it.

    A [VRecord] carries a MUTABLE [value array], and reading an element out of a
    vector hands back that same array rather than a copy — which is precisely
    what makes [v.(i).f <- e] land in storage
    ([Sarek_ir_interp_eval.assign_lvalue]'s [LRecordField] arm). A local binding
    must NOT inherit that sharing: on every other backend

    let e = v.(tid) in e.p <- 42.0

    stores into a copy (the C-family emits a struct-copy local, Native marshals
    a fresh record out through [Vector.get]) and leaves vector storage alone.
    Measured with 4 elements: Native / OpenCL x2 / Vulkan x2 / CUDA-PTX x2 all
    read back [0 1 2 3], while the sharing interpreter read back [42 42 42 42].
    Pinned by test_record_local_alias_agreement.ml.

    The copy is DEEP through records — a shallow one leaves [e.sub.p <- 42.0]
    writing into an inner record still shared with storage — and through variant
    payloads, for the same reason.

    It deliberately stops at [VArray]. An array binding must keep aliasing: a
    kernel buffer is shared across threads by design and block-shared memory is
    shared within the block, so copying one would break both. Arrays are
    reference-like on every backend; records are values.

    {b Depth bound.} The recursion is bounded by {!detach_max_depth} and raises
    past it. Two things are true at once here and the guard exists because of
    the gap between them.

    A cyclic [value] is not constructible through the DSL. A back-edge needs a
    type whose field (or variant payload) type is the type itself, and
    [[@@sarek.type]] refuses that at declaration:

    {[
      type rec_r = {here : float32; next : rec_r} [@@sarek.type]

      type rec_v = A | B of rec_v [@@sarek.type]
    ]}

    both fail to compile with "sarek: unknown alignment for field type 'rec_r' -
    register it with [%ktype] before using it as a record/variant field",
    because a field type must already be registered and registration happens at
    the end of the very declaration that would close the loop. Neither arm of
    the recursion below has a source-level way in. (Verified by compiling both.)

    But that argument is a property of the PPX, in another module, and this
    function's parameter is a bare [value]. The interpreter is dynamically typed
    over [value], and the in-place field store this whole file exists for writes
    [fields.(i) <- e] with no check that [e]'s shape is the field's declared
    type; a future emitter change, a relaxed layout rule, or any other producer
    of [value] could close a loop that no declaration did. So the argument is
    recorded rather than relied on.

    What the guard buys, measured rather than asserted — and re-measured, since
    the two previous numbers written here were both wrong. It is not "an infinite
    loop" (round 4's claim), and it is not "about a second" (the correction to
    it, off by roughly 36x). The recursion is not tail recursive, so an unguarded
    cyclic value does not hang; it dies with [Stack_overflow], but slowly,
    because every level runs the [Array.map] below and so ALLOCATES an array per
    frame. The cost is dominated by that allocation and the resulting GC work, it
    is not a fast stack walk.

    Measured on one host — Linux x86-64, OCaml 5.3.0, [ulimit -s] 8192,
    [OCAMLRUNPARAM] unset — by replacing the [depth > detach_max_depth] test
    below with [false && depth > detach_max_depth]:

    - a standalone probe building ONE self-referential [VRecord] (the [value
      array] is mutable, so a field can hold its own record) and calling
      [detach_record] on it raised [Stack_overflow] after 36.6s of wall clock:
      36.80s, 36.48s, 36.60s over three runs, CPU time within 2% of wall, about
      84M minor words allocated on the way down.
    - the three-case test_detach_record_depth.ml suite, guard disabled, exited 1
      reporting [exception] Stack overflow, in 62.4s / 63.5s / 71.4s over three
      runs — noisier than the probe because it makes two such descents and the
      machine was not quiet. With the guard restored the same suite exits 0 in
      0.003s.

    Only that one configuration was measured, and the seconds are a property of
    it, not of the code: a larger [ulimit -s] buys more levels before the
    overflow and therefore more time. So the load-bearing claim here is the
    DIRECTION, not the figure — tens of seconds of allocation and GC thrash
    ending in an untyped crash, rather than a hang, and equally not a fast
    failure someone could shrug at. It is worth replacing on either reading. A
    [Stack_overflow] escaping through the interpreter is an untyped crash naming
    neither the value nor the binding; this raises [Unsupported_operation] with
    the operation and the bound in it. A diagnosable error is better than a
    crash.

    The bound is on DEPTH, not on visited identity, because pointer-identity
    tracking would cost an allocation per bind on this hot path, while a
    legitimate nesting depth is small — the deepest nesting any [@@sarek.type]
    declaration in this repository reaches is two levels ([colored_point] over
    [point] in tests/e2e/test_nested_types.ml, [l2] over [l1] in the
    record-field-store tests).

    "Small" there is a claim about plausible types, not a limit the compiler
    imposes, and the wording that used to sit here — 64 is "far above anything
    expressible", the nesting being one "which the layout rules already bound" —
    asserted the second. Checked, and it is false. A chain of 65 DISTINCT
    declarations,

      type float32 = float
      type t64 = {v : float32} [@@sarek.type]
      type t63 = {f : t64} [@@sarek.type]
      (* ... one per level, down to ... *)
      type t0 = {f : t1} [@@sarek.type]

    compiles and links clean through the ppx (built as an executable under
    tests/e2e/ and run). Nothing caps chain length: each link's size and
    alignment are resolved by one lookup of the already-registered field type in
    the ppx's size/alignment tables, so a link costs the same whether it is the
    2nd or the 65th. The unreachability argument above is about a SELF-referential
    field type, and a finite chain of DISTINCT types is a different thing — it
    registers cleanly, innermost first, in declaration order.

    Bound to a LOCAL, that [t0] is 65 nested [VRecord]s (the ppx's marshaller
    emits one per link rather than flattening), so [t64] is reached at depth 64
    and its [float32] at depth 65, which trips the test below. A legal type,
    refused. Two things narrow it: the chain must be RECORDS, since the layout
    rules refuse a variant nested below the top level, and 64 links pass — 65 is
    the first that does not.

    What was and was not observed, because the paragraph above is prose and
    NOTHING IN THE TREE GOES RED IF IT STOPS BEING TRUE. The 65-link chain was
    generated, compiled and run once by hand, in round 5; the probe was reverted
    rather than committed, so no test covers it. That the refusal then FIRES at
    depth 65 is weaker still — it is read off the marshaller and [bind_var], not
    observed, since no kernel binding a [t0] local was written.

    Be precise about which way that fails, because the two halves fail
    differently. If the ppx later grows a chain-length cap, or the marshaller
    starts flattening, then the CONCLUSION drawn below stays safe: the guard would
    merely be conservative about a false positive that can no longer happen. But
    the FACTUAL sentences above ("compiles and links clean", "nothing caps chain
    length") would simply be false — stale, and stated with more confidence than
    an unchecked claim earns. That is a documentation rot risk, not a correctness
    risk, and it is the honest characterisation. It is the same shape as the
    defect this round exists to fix, one order weaker.

    Left uncovered on purpose. A committed fixture here would pin the ABSENCE of a
    ppx limitation, so it would fire at anyone who later adds a chain-length cap —
    a change nobody has argued against — and read as a promise that deep chains
    are supported, which is the opposite of this note's point. If coverage is ever
    wanted, pin the boundary on THIS side instead, where the behaviour actually
    matters and the ppx is not involved: [nest 64] is copied and [nest 65] is
    refused, six lines in test_detach_record_depth.ml, no ppx compile.

    So the honest bound is: 64 is far above anything PLAUSIBLE, not above
    anything expressible, and far below a stack overflow. The trade is kept
    deliberately. The false positive needs a hand-written 65-link chain of
    distinct types, and it fails LOUDLY — the refusal below names the operation
    and the bound, so whoever wrote that chain is told what to raise — whereas
    identity tracking would tax every record bind in every kernel. If such a type
    ever shows up in earnest, raise [detach_max_depth]; do not read this comment
    as a promise that it cannot. *)

let detach_max_depth = 64

let detach_record (v : value) : value =
  let rec go depth v =
    if depth > detach_max_depth then
      Interp_error.raise_error
        (Interp_error.Unsupported_operation
           {
             operation = "record copy-at-bind";
             reason =
               Printf.sprintf
                 "nesting deeper than %d while detaching a record for a local \
                  binding. The likely cause is a CYCLIC value: a \
                  self-referential [@@sarek.type] field type is refused for \
                  lack of a layout, so no declaration produces one, but the \
                  interpreter's [value] is dynamically typed and an in-place \
                  field store can close a loop no declaration did. The other \
                  possibility is a genuine chain of more than %d distinct \
                  nested record types, which the PPX does accept — if that is \
                  what this is, raise detach_max_depth"
                 detach_max_depth
                 detach_max_depth;
           })
    else
      match v with
      | VRecord (name, fields) ->
          VRecord (name, Array.map (go (depth + 1)) fields)
      | VVariant (ty, tag, args) ->
          VVariant (ty, tag, List.map (go (depth + 1)) args)
      | VArray _ | VInt32 _ | VInt64 _ | VFloat32 _ | VFloat64 _ | VBool _
      | VUnit ->
          v
  in
  go 0 v

(** The interpreter's variant tag for a constructor NAME.

    THE definition, and the only one: {!Sarek_ir_interp_eval}'s [EVariant] arm
    and both of its matchers ([EMatch], [SMatch]) call this rather than
    repeating [Hashtbl.hash _ mod 256], so a value built here and an arm
    selected there cannot drift apart. They were three separate copies of the
    expression, which is how a fourth copy — a literal [0] — went unnoticed.

    [mod 256] means the encoding is NOT injective: two constructors of the same
    type whose names collide modulo 256 select each other's arms. That is a
    pre-existing property of the interpreter's variant representation, not
    something introduced here, and it is left alone; naming the function at
    least gives it one place to be fixed. *)
let variant_tag_of_ctor (ctor : string) : int = Hashtbl.hash ctor mod 256

(** {1 Array allocation — backlog-206}

    Zero value of an element type, used to fill a freshly declared kernel array
    ([EArrayCreate] / [DShared]).

    Records and variants used to fall through to [VUnit] here, which is why
    [let%shared (s : tri) = 4l] followed by [s.(i).a <- e] raised "assignment
    target of .a (got unit)" on the Interpreter while Native accepted the same
    store — the divergence backlog-206 is about. A record element type now gets
    a [VRecord] whose fields are themselves zeroed, and a variant with at least
    one constructor gets a real constructor with zeroed payloads. A [TVariant]
    carrying an EMPTY constructor list still gives [VUnit]: there is no
    constructor to pick, the PPX produces no such type, and inventing a tag for
    it would be worse than the [VUnit].

    WHICH constructor, and WHAT TAG, are both load-bearing, and the first
    version of this got the tag wrong.

    The tag is {b not} a positional index in this interpreter. [EVariant]
    encodes [Hashtbl.hash ctor_name mod 256] ({!Sarek_ir_interp_eval}), and
    [EMatch]/[SMatch] select an arm with exactly that predicate. A literal [0]
    therefore matches the first constructor only if its name happens to hash to
    zero — so a default slot matched NO arm and the interpreter raised "Pattern
    match failure in SMatch". Measured on a registered
    [type c2 = A of float32 | B] whose unwritten shared slot was read:
    Interpreter x2 RAISED where Native answered [B]. The tag now goes through
    {!variant_tag_of_ctor}, the same encoding the evaluator uses, so a default
    slot decodes to the constructor this function chose.

    The constructor is the first NULLARY one if there is one, else the first
    constructor with zeroed payloads. That is not an aesthetic preference: it is
    what the Native backend's {!Sarek_native_gen_expr} default does
    ([Sarek_native_helpers.default_value_for_type] searches for a nullary
    constructor first), and a CPU-backend disagreement about the contents of a
    freshly declared shared array is the exact shape of divergence backlog-206
    was filed as. The two CPU backends now agree.

    None of this is a promise that reading an unwritten slot is DEFINED.
    [__local]/[shared] memory is uninitialised storage on every device, so any
    value read before a write is arbitrary there; what these two paragraphs buy
    is that the two backends which do have to put something in the slot put the
    same thing, and that the something is a value the interpreter can match.

    [TUint8] gets [VInt32 0l], and that is a GUESS rather than a convention:
    [TUint8] appears nowhere else in this interpreter, so unlike [TFloat16] —
    whose [VFloat32] carrier is established by [Sarek_ir_interp_eval] and
    [Sarek_ir_interp] — nothing here fixes its representation. It exists only as
    a cooperative-matrix operand element type, which does not reach a kernel
    array declaration, so the arm is unreachable in practice; it is written this
    way so that the match is exhaustive rather than because a zero of that type
    has been agreed. If [TUint8] ever becomes a real element type, decide its
    carrier before trusting this.

    [TArray] and [TVec] elements stay [VUnit]: an array of arrays is not
    expressible as a kernel array element type on any backend, and inventing a
    nested [VArray] of an unknown length here would be a guess.

    Plain recursion, no depth guard, unlike {!detach_record}. The argument is
    different in kind: this walks an [elttype], a finite tree the PPX builds
    bottom-up, where {!detach_record} walks a [value], which the interpreter's
    in-place field store can shape at run time. *)
let rec default_value_of_elttype (ty : elttype) : value =
  match ty with
  | TInt32 -> VInt32 0l
  | TInt64 -> VInt64 0L
  | TFloat16 | TFloat32 -> VFloat32 0.0
  | TFloat64 -> VFloat64 0.0
  | TUint8 -> VInt32 0l
  | TBool -> VBool false
  | TUnit -> VUnit
  | TRecord (name, fields) ->
      VRecord
        ( name,
          Array.of_list
            (List.map (fun (_, fty) -> default_value_of_elttype fty) fields) )
  | TVariant (name, constrs) -> (
      let nullary = List.find_opt (fun (_, payload) -> payload = []) constrs in
      match (nullary, constrs) with
      | Some (ctor, _), _ -> VVariant (name, variant_tag_of_ctor ctor, [])
      | None, (ctor, payload) :: _ ->
          VVariant
            ( name,
              variant_tag_of_ctor ctor,
              List.map default_value_of_elttype payload )
      | None, [] -> VUnit)
  | TArray _ | TVec _ -> VUnit

(** Allocate a kernel array of [size] elements of type [ty].

    [Array.init], never [Array.make]: for a boxed element type (a record, or a
    variant with a payload) [Array.make] stores ONE value in every slot, so
    [s.(0).f <- e] is visible through every index. That is exactly the Native
    half of backlog-206, whose CPU-runtime allocator had the same
    [Array.make size default] shape. Slots must be independent allocations. *)
let alloc_kernel_array (ty : elttype) (size : int) : value array =
  Array.init size (fun _ -> default_value_of_elttype ty)

(** Bind a variable in the environment (both by id and name).

    Records are detached on the way in (see {!detach_record}): binding is what
    creates a LOCAL, and a local record is a value, not a window onto the
    container it was read from. Every binding site goes through here — [SLet],
    [SLetMut], the [SFor] loop variable, helper parameters, and a whole-variable
    assignment — so the rule holds uniformly rather than at whichever site
    someone remembered.

    NOT covered: the variant-pattern binders in [SMatch]/[EMatch], which write
    [vars_by_name] directly. Their payloads come from a scrutinee that is itself
    already a local in every shape expressible today, so there is no container
    to write back into; if a variant read straight out of a vector ever becomes
    a match scrutinee, those two sites need this too. *)
let bind_var env (v : var) value =
  let value = detach_record value in
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
      (* Float Mod is C fmod on every backend; Float.rem is C fmod. The
         float arms were missing entirely (audit follow-up to M1/M4): float
         operands were truncated to ints before the rem. *)
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.to_float32 (Float.rem a b))
      | VFloat64 a, VFloat64 b -> VFloat64 (Float.rem a b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.to_float32 (Float.rem (to_float32 v1) (to_float32 v2)))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (Float.rem (to_float64 v1) (to_float64 v2))
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
      | VInt64 _, _ | _, VInt64 _ ->
          VBool (Int64.compare (to_int64 v1) (to_int64 v2) < 0)
      | _ -> VBool (to_int32 v1 < to_int32 v2))
  | Le -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a <= b)
      | VFloat64 a, VFloat64 b -> VBool (a <= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 <= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 <= to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VBool (Int64.compare (to_int64 v1) (to_int64 v2) <= 0)
      | _ -> VBool (to_int32 v1 <= to_int32 v2))
  | Gt -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a > b)
      | VFloat64 a, VFloat64 b -> VBool (a > b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 > to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 > to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VBool (Int64.compare (to_int64 v1) (to_int64 v2) > 0)
      | _ -> VBool (to_int32 v1 > to_int32 v2))
  | Ge -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a >= b)
      | VFloat64 a, VFloat64 b -> VBool (a >= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 >= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 >= to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VBool (Int64.compare (to_int64 v1) (to_int64 v2) >= 0)
      | _ -> VBool (to_int32 v1 >= to_int32 v2))
  | And -> VBool (to_bool v1 && to_bool v2)
  | Or -> VBool (to_bool v1 || to_bool v2)
  | Shl -> (
      match v1 with
      | VInt64 a -> VInt64 (Int64.shift_left a (to_int v2))
      | _ -> VInt32 (Int32.shift_left (to_int32 v1) (to_int v2)))
  | Shr -> (
      (* Arithmetic (sign-extending) shift, matching every codegen backend:
         CUDA/OpenCL/Metal/GLSL/WGSL emit plain [>>] on a signed int type,
         and PTX emits shr.s32/shr.s64. [lsr] is lowered to a separate
         expression tree in Sarek_lower_ir.ml precisely because this node is
         arithmetic - see G phase 1 in
         the unpublished fix-critical-semantics-evidence.md note. *)
      match v1 with
      | VInt64 a -> VInt64 (Int64.shift_right a (to_int v2))
      | _ -> VInt32 (Int32.shift_right (to_int32 v1) (to_int v2)))
  | BitAnd -> (
      match (v1, v2) with
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.logand (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.logand (to_int32 v1) (to_int32 v2)))
  | BitOr -> (
      match (v1, v2) with
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.logor (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.logor (to_int32 v1) (to_int32 v2)))
  | BitXor -> (
      match (v1, v2) with
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.logxor (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.logxor (to_int32 v1) (to_int32 v2)))

let eval_unop op v =
  match op with
  | Neg -> (
      match v with
      | VFloat32 f -> VFloat32 (-.f)
      | VFloat64 f -> VFloat64 (-.f)
      | VInt64 n -> VInt64 (Int64.neg n)
      | _ -> VInt32 (Int32.neg (to_int32 v)))
  | Not -> VBool (not (to_bool v))
  | BitNot -> (
      match v with
      | VInt64 a -> VInt64 (Int64.lognot a)
      | _ -> VInt32 (Int32.lognot (to_int32 v)))

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
  | ["Float64"]
  | ["Sarek_stdlib"; "Float64"]
  (* Float64 intrinsics live in the standalone [sarek_float64] library
     (module [Sarek_float64.Float64]), so a kernel that opens it lowers
     transcendentals under this module path. GPU backends resolve them
     via the intrinsic's embedded device-ref, but the interpreter
     dispatches on the path, so it must recognise it here too. *)
  | ["Sarek_float64"; "Float64"] ->
      true
  | _ -> false

let is_int32_path = function
  | ["Int32"] | ["Sarek_stdlib"; "Int32"] -> true
  | _ -> false
