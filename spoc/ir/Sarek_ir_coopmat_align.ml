(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Static alignment proof for cooperative-matrix loads and stores on NVIDIA
    ([wmma]), shared by the CUDA/C and PTX generators.

    [wmma.load] / [wmma.store] need a 32-byte aligned address and a leading
    dimension that is a multiple of 16 bytes. The IR promises neither (any start
    index, any stride: {!Sarek_ir_types.CM_load}), so a generator either checks
    at run time and stages a misaligned tile through shared memory, or proves
    the access aligned and emits the bare instruction. This module is the proof.

    {b What is proved.} A power-of-two {e factor} of an int32 expression: [f]
    such that [f] divides the expression's value, whatever the run. It is
    computed from the expression alone, by rules that hold under wrapping 32-bit
    arithmetic (divisibility by a power of two is preserved modulo 2^32):
    - an integer constant [c]: its lowest set bit ([0]: unbounded);
    - [a + b], [a - b]: the minimum of the factors;
    - [a * b]: the product;
    - [a << c] with a constant [c]: the factor of [a] times [2^c];
    - a variable bound by an immutable [let]: the factor of its definition;
    - anything else (kernel parameters, loop variables, mutable variables,
      division, intrinsics, array reads): 1.

    An access is {e proven} when [factor index * element bytes >= 32] and
    [factor stride * element bytes >= 16] (both sides are powers of two, so [>=]
    is divisibility). {b The base pointer is assumed aligned}: a global buffer
    is a device allocation (256-byte aligned) and a [.shared] array a
    cooperative-matrix statement accesses is declared 32-byte aligned by the
    generators. A cooperative-matrix kernel that does not rely on either (a
    pointer into the middle of an allocation) must not pass a view to the
    launch.

    The walk is by name with lexical scoping, so a shadowing binder (a [let], a
    [for] variable, a match payload) hides an outer fact. The result is the list
    of proven operations, identified {e physically}: a generator asks about the
    very node it is about to emit, and an operation it cannot find (a rewritten
    tree, a helper body) is unproven, which is the safe answer. *)

open Sarek_ir_types
module C = Sarek_coopmat_types

(** Cap of a factor: [2^30], standing for "divisible by every power of two that
    matters". *)
let unbounded = 1 lsl 30

(** The edge of a cooperative-matrix tile: a store with a stride below it
    overlaps its own rows (columns), and is written serially (the answer is the
    last write of the interpreter's loop; a matrix-unit store does not promise
    an order). *)
let tile_edge = 16

let lowest_bit n = if n = 0 then unbounded else min unbounded (n land -n)

type env = (string * int) list

let rec factor (env : env) (e : expr) : int =
  match e with
  | EConst (CInt32 c) -> lowest_bit (Int32.to_int c)
  | EVar v when v.var_mutable -> 1
  | EVar v -> Option.value (List.assoc_opt v.var_name env) ~default:1
  | EBinop ((Add | Sub), a, b) -> min (factor env a) (factor env b)
  | EBinop (Mul, a, b) -> min unbounded (factor env a * factor env b)
  | EBinop (Shl, a, EConst (CInt32 s)) when s >= 0l && s < 31l ->
      min unbounded (factor env a lsl Int32.to_int s)
  | _ -> 1

(** Bytes per element of the buffer a fragment is loaded from or stored to. *)
let element_bytes (f : C.fragment) =
  match f.C.frag_component with
  | C.Uint8 | C.Sint8 -> 1
  | C.Float16 -> 2
  | C.BFloat16 | C.Float32 | C.Uint32 | C.Sint32 -> 4

let access_is_proven env ~(frag : C.fragment) ~index ~stride =
  let bytes = element_bytes frag in
  factor env index * bytes >= 32 && factor env stride * bytes >= 16

(** Bytes of the 16x16 staging tile of a fragment. *)
let tile_bytes (f : C.fragment) = 256 * element_bytes f

(** Pragma hint carried by the body of the aligned variant of a kernel. The
    generators read it as "every access is aligned at run time" and emit the
    bare instruction; every other consumer ignores a pragma. *)
let aligned_hint = "sarek.coopmat_aligned"

(** Pragma hint of the {e checked} variant: no static proof is used, every
    access is tested at run time (and staged when misaligned). Selected when a
    launch argument's base pointer is not 32-byte aligned, which every static
    proof otherwise assumes. *)
let checked_hint = "sarek.coopmat_checked"

let is_checked (body : stmt) =
  let rec go = function
    | SPragma (hints, _) when List.mem checked_hint hints -> true
    | SPragma (_, s) | SBlock s -> go s
    | _ -> false
  in
  go body

let is_marked (body : stmt) =
  let rec go = function
    | SPragma (hints, _) when List.mem aligned_hint hints -> true
    | SPragma (_, s) | SBlock s -> go s
    | _ -> false
  in
  go body

(** The proven loads and stores of a kernel body; all of them in the aligned
    variant ({!mark_aligned}). *)
let proven_ops (body : stmt) : coopmat_op list =
  let assume = is_marked body in
  let found = ref [] and refuted = ref [] in
  if is_checked body then []
  else begin
    (* a store is provable only with a constant stride of at least the tile
       edge: below it its rows overlap and it is written serially *)
    let stride_ok op stride =
      match op with
      | CM_store _ -> (
          match stride with
          | EConst (CInt32 c) -> abs (Int32.to_int c) >= tile_edge
          | _ -> false)
      | CM_load _ | CM_decl _ | CM_muladd _ | CM_scale_add _ -> true
    in
    let rec walk env = function
      | SCoopmat (CM_load {frag; index; stride; _} as op)
      | SCoopmat (CM_store {frag; index; stride; _} as op) ->
          if
            assume
            || (access_is_proven env ~frag ~index ~stride && stride_ok op stride)
          then found := op :: !found
          else refuted := op :: !refuted
      | SCoopmat (CM_decl _ | CM_muladd _ | CM_scale_add _) -> ()
      | SSeq ss -> List.iter (walk env) ss
      | SIf (_, a, b) ->
          walk env a ;
          Option.iter (walk env) b
      | SWhile (_, s) | SPragma (_, s) | SBlock s -> walk env s
      | SFor (v, _, _, _, s) -> walk ((v.var_name, 1) :: env) s
      | SLet (v, e, s) -> walk ((v.var_name, factor env e) :: env) s
      | SLetMut (v, _, s) -> walk ((v.var_name, 1) :: env) s
      | SMatch (_, arms) ->
          List.iter
            (fun (pat, s) ->
              let env =
                match pat with
                | PConstr (_, names) ->
                    List.fold_left (fun env n -> (n, 1) :: env) env names
                | PWild -> env
              in
              walk env s)
            arms
      | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
      | SMemFence | SNative _ ->
          ()
    in
    walk [] body ;
    (* One node can sit under several bindings (a shared IR value): it is proven
     only if every occurrence is. *)
    List.filter (fun op -> not (List.exists (fun r -> r == op) !refuted)) !found
  end

(** Whether [op] is one of the [proven] operations (physical identity). *)
let is_proven proven op = List.exists (fun p -> p == op) proven

(** {1 Launch-time variant: a predicate over the scalar arguments}

    An access whose index or stride depends on a kernel parameter (a GEMM's [k]
    and [n]) cannot be proven aligned from the IR alone, but the host knows the
    parameter's value at launch. {!derive} reduces "every cooperative-matrix
    access of the kernel is aligned" to a condition on the kernel's scalar int32
    parameters, a disjunction of conjunctions of atoms "parameter [p] is a
    multiple of [n]" (a {e predicate}). The launch evaluates it ({!holds}) on
    the actual arguments: true selects the variant of the kernel generated with
    {!mark_aligned} (bare [wmma], no check, no buffer), false the staged
    variant.

    [need env e n] is the condition under which [e] is a multiple of [n] (a
    power of two), by the rules of {!factor} extended with parameters as atoms,
    and with a product [a * b] allowed any split [a] multiple of [x], [b]
    multiple of [n / x]. A kernel parameter that is not shadowed is the atom
    [(name, n)]; anything the rules cannot reduce makes the condition [false]
    and {!derive} answer [None] (the kernel always takes the staged variant). *)

type atom =
  | Multiple of string * int  (** the parameter is a multiple of [n] *)
  | At_least of string * int
      (** [|parameter| >= n]: a store whose stride is below the tile edge
          overlaps its own rows, and is not left to the matrix unit *)

(** Disjunction of conjunctions; [[[]]] is true, [[]] false. *)
type predicate = atom list list

let p_true : predicate = [[]]

let p_false : predicate = []

let max_terms = 64

(* An atom [(p, a)] implies [(p, b)] when [a] is a multiple of [b]. *)
let implies a b =
  match (a, b) with
  | Multiple (p1, x), Multiple (p2, y) -> String.equal p1 p2 && x mod y = 0
  | At_least (p1, x), At_least (p2, y) -> String.equal p1 p2 && x >= y
  | Multiple _, At_least _ | At_least _, Multiple _ -> false

(* Drop an atom another atom of the conjunction implies. *)
let reduce_conj c =
  let c = List.sort_uniq compare c in
  List.filter (fun a -> not (List.exists (fun b -> b <> a && implies b a) c)) c

(* Drop a conjunction that implies another (the other is weaker, so the
   disjunction is unchanged). *)
let normalise (p : predicate) : predicate =
  let p = List.sort_uniq compare (List.map reduce_conj p) in
  List.filter
    (fun a ->
      not
        (List.exists
           (fun b ->
             b <> a
             && List.for_all (fun x -> List.exists (fun y -> implies y x) a) b)
           p))
    p

let p_and (a : predicate) (b : predicate) : predicate =
  if a = [] || b = [] then p_false
  else
    let r = normalise (List.concat_map (fun x -> List.map (( @ ) x) b) a) in
    if List.length r > max_terms then p_false else r

let p_or (a : predicate) (b : predicate) : predicate =
  let r = normalise (a @ b) in
  if List.length r > max_terms then p_false else r

type binding =
  | Bound of expr * (string * binding) list
      (** an immutable [let]: its definition and the scope it was made in *)
  | Opaque  (** a loop, mutable or match variable: unknown *)

type symbolic_env = (string * binding) list

let rec need ~(params : string list) (env : symbolic_env) (e : expr) (n : int) :
    predicate =
  if n <= 1 then p_true
  else
    match e with
    | EConst (CInt32 c) ->
        if Int32.rem c (Int32.of_int n) = 0l then p_true else p_false
    | EVar v when v.var_mutable -> p_false
    | EVar v -> (
        match List.assoc_opt v.var_name env with
        | Some (Bound (def, def_env)) -> need ~params def_env def n
        | Some Opaque -> p_false
        | None ->
            if List.mem v.var_name params then [[Multiple (v.var_name, n)]]
            else p_false)
    | EBinop ((Add | Sub), a, b) ->
        p_and (need ~params env a n) (need ~params env b n)
    | EBinop (Mul, a, b) ->
        let rec splits x acc =
          if x > n then acc
          else
            splits
              (x * 2)
              (p_or
                 acc
                 (p_and (need ~params env a x) (need ~params env b (n / x))))
        in
        splits 1 p_false
    | EBinop (Shl, a, EConst (CInt32 s)) when s >= 0l && s < 31l ->
        let m = 1 lsl Int32.to_int s in
        if m >= n then p_true else need ~params env a (n / m)
    | _ -> p_false

(** The condition under which [|e| >= n]: a constant, a parameter (or a [let]
    bound to one), nothing else (anything the rules cannot reduce is [false]).
*)
let rec at_least ~(params : string list) (env : symbolic_env) (e : expr)
    (n : int) : predicate =
  match e with
  | EConst (CInt32 c) -> if abs (Int32.to_int c) >= n then p_true else p_false
  | EVar v when v.var_mutable -> p_false
  | EVar v -> (
      match List.assoc_opt v.var_name env with
      | Some (Bound (def, def_env)) -> at_least ~params def_env def n
      | Some Opaque -> p_false
      | None ->
          if List.mem v.var_name params then [[At_least (v.var_name, n)]]
          else p_false)
  | _ -> p_false

(** The scalar int32 parameters of a kernel. *)
let scalar_params (k : kernel) =
  List.filter_map
    (function
      | DParam (({var_type = TInt32; _} as v), None) -> Some v.var_name
      | DParam _ | DLocal _ | DShared _ -> None)
    k.kern_params

(** The condition under which every cooperative-matrix access of [k]'s body is
    aligned, [None] when it cannot be derived (a coopmat statement in a helper
    function, or an access the rules cannot reduce). A kernel with no access
    that needs a condition gives [Some p_true]. *)
let derive (k : kernel) : predicate option =
  let params = scalar_params k in
  let acc = ref p_true in
  let rec walk (env : symbolic_env) = function
    | SCoopmat (CM_load {frag; index; stride; _}) ->
        let bytes = element_bytes frag in
        let want n = max 1 (n / bytes) in
        acc :=
          p_and
            !acc
            (p_and
               (need ~params env index (want 32))
               (need ~params env stride (want 16)))
    | SCoopmat (CM_store {frag; index; stride; _}) ->
        let bytes = element_bytes frag in
        let want n = max 1 (n / bytes) in
        acc :=
          p_and
            !acc
            (p_and
               (p_and
                  (need ~params env index (want 32))
                  (need ~params env stride (want 16)))
               (at_least ~params env stride tile_edge))
    | SCoopmat (CM_decl _ | CM_muladd _ | CM_scale_add _) -> ()
    | SSeq ss -> List.iter (walk env) ss
    | SIf (_, a, b) ->
        walk env a ;
        Option.iter (walk env) b
    | SWhile (_, s) | SPragma (_, s) | SBlock s -> walk env s
    | SFor (v, _, _, _, s) -> walk ((v.var_name, Opaque) :: env) s
    | SLet (v, e, s) -> walk ((v.var_name, Bound (e, env)) :: env) s
    | SLetMut (v, _, s) -> walk ((v.var_name, Opaque) :: env) s
    | SMatch (_, arms) ->
        List.iter
          (fun (pat, s) ->
            let env =
              match pat with
              | PConstr (_, names) ->
                  List.fold_left (fun env n -> (n, Opaque) :: env) env names
              | PWild -> env
            in
            walk env s)
          arms
    | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
    | SMemFence | SNative _ ->
        ()
  in
  walk [] k.kern_body ;
  let in_helper =
    List.exists
      (fun hf ->
        Sarek_ir_analysis.kernel_has_coopmat_op
          {default_kernel with kern_body = hf.hf_body})
      k.kern_funcs
  in
  if in_helper || !acc = p_false then None else Some !acc

(** Whether [p] holds for the scalar arguments [args] (parameter name, value). A
    parameter without a value makes its atom false. *)
let holds (p : predicate) (args : (string * int) list) =
  List.exists
    (List.for_all (fun atom ->
         let name = match atom with Multiple (p, _) | At_least (p, _) -> p in
         match List.assoc_opt name args with
         | None -> false
         | Some v -> (
             match atom with
             | Multiple (_, n) -> v mod n = 0
             | At_least (_, n) -> abs v >= n)))
    p

(** The aligned variant of [k]: the same kernel with its body marked. *)
let mark_aligned (k : kernel) : kernel =
  if is_marked k.kern_body then k
  else {k with kern_body = SPragma ([aligned_hint], k.kern_body)}

(** The kernel with every static proof switched off: the checked variant. *)
let mark_checked (k : kernel) : kernel =
  if is_checked k.kern_body then k
  else {k with kern_body = SPragma ([checked_hint], k.kern_body)}

(** The arrays the kernel's cooperative-matrix loads and stores read or write,
    in order of first appearance. *)
let arrays_accessed (k : kernel) : string list =
  let seen = ref [] in
  let note name = if not (List.mem name !seen) then seen := name :: !seen in
  List.iter
    (function
      | CM_load {src; _} -> note src
      | CM_store {dst; _} -> note dst
      | CM_decl _ | CM_muladd _ | CM_scale_add _ -> ())
    (Sarek_ir_analysis.kernel_coopmat_ops k) ;
  List.rev !seen
