(** val fst : ('a1 * 'a2) -> 'a1 **)

let fst = function | (x, _) -> x

(** val snd : ('a1 * 'a2) -> 'a2 **)

let snd = function | (_, y) -> y

(** val app : 'a1 list -> 'a1 list -> 'a1 list **)

let rec app l m = match l with | [] -> m | a :: l1 -> a :: (app l1 m)

(** val add : int -> int -> int **)

let rec add = (+)

(** val mul : int -> int -> int **)

let rec mul = ( * )

(** val sub : int -> int -> int **)

let rec sub = fun n m -> Stdlib.max 0 (n-m)

module Nat = struct (** val sub : int -> int -> int **)

let rec sub n m = (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun _ ->
  n) (fun k -> (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun _ -> n)
  (fun l -> sub k l) m) n

(** val max : int -> int -> int **)

let rec max n m = (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun _ ->
  m) (fun n' -> (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun _ -> n)
  (fun m' -> Stdlib.Int.succ (max n' m')) m) n

(** val divmod : int -> int -> int -> int -> int * int **)

let rec divmod x y q u = (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun
  _ -> (q, u)) (fun x' -> (fun fO fS n -> if n=0 then fO () else fS (n-1))
  (fun _ -> divmod x' y (Stdlib.Int.succ q) y) (fun u' -> divmod x' y q u') u)
  x

(** val div : int -> int -> int **)

let div x y = (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun _ -> y)
  (fun y' -> fst (divmod x y' 0 y')) y

(** val modulo : int -> int -> int **)

let modulo x y = (fun fO fS n -> if n=0 then fO () else fS (n-1)) (fun _ -> x)
  (fun y' -> sub y' (snd (divmod x y' 0 y'))) y end

(** val fold_right : ('a2 -> 'a1 -> 'a1) -> 'a1 -> 'a2 list -> 'a1 **)

let rec fold_right f a0 = function | [] -> a0 | b :: l0 -> f b (fold_right f
  a0 l0)

(** val forallb : ('a1 -> bool) -> 'a1 list -> bool **)

let rec forallb f = function | [] -> true | a :: l0 -> (&&) (f a) (forallb f
  l0)

(** val align_up : int -> int -> int **)

let align_up off a = mul a (Nat.div (sub (add off a) (Stdlib.Int.succ 0)) a)

type lty = | L32 | L64

(** val scalar_size : lty -> int **)

let scalar_size = function | L32 -> Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ 0))) | L64 -> Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0)))))))

(** val scalar_align : lty -> int **)

let scalar_align = scalar_size

type lfield = | LLeaf of lty | LRec of lfields and lfields = | LNil | LCons of
  int * lfield * lfields

(** val falign : lfield -> int **)

let rec falign = function | LLeaf t -> scalar_align t | LRec fs -> fsalign fs

(** val fsalign : lfields -> int **)

and fsalign = function | LNil -> Stdlib.Int.succ 0 | LCons (_, f, r) ->
  Nat.max (falign f) (fsalign r)

(** val fsize : lfield -> int **)

let rec fsize = function | LLeaf t -> scalar_size t | LRec fs -> align_up
  (fsend 0 fs) (fsalign fs)

(** val fsend : int -> lfields -> int **)

and fsend off = function | LNil -> off | LCons (_, f, r) -> fsend (add
  (align_up off (falign f)) (fsize f)) r

type leaf = { lf_path : int list; lf_ty : lty; lf_off : int }

(** val leaf_size : leaf -> int **)

let leaf_size l = scalar_size l.lf_ty

(** val flatten : int list -> int -> lfield -> leaf list **)

let rec flatten p off = function | LLeaf t -> { lf_path = p; lf_ty = t; lf_off
  = off } :: [] | LRec fs -> flattens p off fs

(** val flattens : int list -> int -> lfields -> leaf list **)

and flattens p off = function | LNil -> [] | LCons (n, f, r) -> let o =
  align_up off (falign f) in app (flatten (app p (n :: [])) o f) (flattens p
  (add o (fsize f)) r)

(** val record_leaves : lfields -> leaf list **)

let record_leaves fs = flattens [] 0 fs

(** val record_size : lfields -> int **)

let record_size fs = align_up (fsend 0 fs) (fsalign fs)

(** val field_offsets : int -> lfields -> (int * int) list **)

let rec field_offsets off = function | LNil -> [] | LCons (n, f, r) -> let o =
  align_up off (falign f) in (n, o) :: (field_offsets (add o (fsize f)) r)

(** val record_field_offsets : lfields -> (int * int) list **)

let record_field_offsets fs = field_offsets 0 fs

(** val tag_offset : int **)

let tag_offset = 0

(** val tag_size : int **)

let tag_size = Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ 0)))

(** val number_args : int -> lfield list -> lfields **)

let rec number_args i = function | [] -> LNil | a :: r -> LCons (i, a,
  (number_args (Stdlib.Int.succ i) r))

(** val ctor_align : lfield list -> int **)

let ctor_align args = fsalign (number_args 0 args)

(** val variant_payload_offset : lfield list list -> int **)

let variant_payload_offset ctors = fold_right (fun c acc -> Nat.max
  (ctor_align c) acc) (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
  (Stdlib.Int.succ 0)))) ctors

(** val payload_struct_size : lfield list -> int **)

let payload_struct_size args = align_up (fsend 0 (number_args 0 args))
  (fsalign (number_args 0 args))

type ctor_layout = { cl_tag : int; cl_leaves : leaf list; cl_payload_size :
  int }

(** val ctor_layouts : int -> int -> lfield list list -> ctor_layout list **)

let rec ctor_layouts payoff tag = function | [] -> [] | c :: r -> { cl_tag =
  tag; cl_leaves = (flattens (tag :: []) payoff (number_args 0 c));
  cl_payload_size = (payload_struct_size c) } :: (ctor_layouts payoff
  (Stdlib.Int.succ tag) r)

(** val max_payload : ctor_layout list -> int **)

let max_payload cls = fold_right (fun c acc -> Nat.max c.cl_payload_size acc)
  0 cls

(** val variant_size : lfield list list -> int **)

let variant_size ctors = let p = variant_payload_offset ctors in align_up (add
  p (max_payload (ctor_layouts p 0 ctors))) p

(** val leaf_aligned : leaf -> bool **)

let leaf_aligned l = (=) (Nat.modulo l.lf_off (scalar_align l.lf_ty)) 0

(** val record_accepted : lfields -> bool **)

let record_accepted fs = forallb leaf_aligned (record_leaves fs)

(** val variant_accepted : lfield list list -> bool **)

let variant_accepted ctors = forallb (fun c -> forallb leaf_aligned
  c.cl_leaves) (ctor_layouts (variant_payload_offset ctors) 0 ctors)
