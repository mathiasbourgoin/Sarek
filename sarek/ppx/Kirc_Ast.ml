(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2013)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
 *******************************************************************************)
type kernel

(*and kint =*)
and kvect = IntVect of int | Floatvect of int

type intrinsics = string * string

type elttype = EInt32 | EInt64 | EFloat32 | EFloat64

type memspace = LocalSpace | Global | Shared

type k_ext =
  | Kern of k_ext * k_ext
  | Block of k_ext
  | Params of k_ext
  | Plus of k_ext * k_ext
  | Plusf of k_ext * k_ext
  | Min of k_ext * k_ext
  | Minf of k_ext * k_ext
  | Mul of k_ext * k_ext
  | Mulf of k_ext * k_ext
  | Div of k_ext * k_ext
  | Divf of k_ext * k_ext
  | Mod of k_ext * k_ext
  | Id of string
  | IdName of string
  | GlobalFun of k_ext * string * string
  | IntVar of int * string * bool
  | FloatVar of int * string * bool
  | UnitVar of int * string * bool
  | CastDoubleVar of int * string
  | DoubleVar of int * string * bool
  | BoolVar of int * string * bool
  | Arr of string * k_ext * elttype * memspace
  | VecVar of k_ext * int * string
  | Concat of k_ext * k_ext
  | Constr of string * string * k_ext list
  | Record of string * k_ext list
  | RecGet of k_ext * string
  | RecSet of k_ext * k_ext
  | Empty
  | Seq of k_ext * k_ext
  | Return of k_ext
  | Set of k_ext * k_ext
  | Decl of k_ext
  | SetV of k_ext * k_ext
  | SetLocalVar of k_ext * k_ext * k_ext
  | Intrinsics of intrinsics
  | IntrinsicRef of string list * string
      (** module_path * name: looked up in registry at JIT *)
  | IntId of string * int
  | Int of int
  | Float of float
  | Double of float
  | Custom of string * int * string
  | CustomVar of string * string * string
  | IntVecAcc of k_ext * k_ext
  | Local of k_ext * k_ext
  | Acc of k_ext * k_ext
  | Ife of k_ext * k_ext * k_ext
  | If of k_ext * k_ext
  | Match of string * k_ext * case array
  | Or of k_ext * k_ext
  | And of k_ext * k_ext
  | Not of k_ext
  | EqCustom of string * k_ext * k_ext
  | EqBool of k_ext * k_ext
  | LtBool of k_ext * k_ext
  | GtBool of k_ext * k_ext
  | LtEBool of k_ext * k_ext
  | GtEBool of k_ext * k_ext
  | DoLoop of k_ext * k_ext * k_ext * k_ext
  | While of k_ext * k_ext
  | App of k_ext * k_ext array
  | GInt of (unit -> int32)
  | GFloat of (unit -> float)
  | GFloat64 of (unit -> float)
  | GIntVar of string  (** Global int32 variable by name - for PPX quoting *)
  | GFloatVar of string
      (** Global float32 variable by name - for PPX quoting *)
  | GFloat64Var of string
      (** Global float64 variable by name - for PPX quoting *)
  | Native of (string -> string)
  | NativeWithFallback of {
      gpu : Ppxlib.expression;  (** fun dev -> "cuda/opencl code" *)
      ocaml : Ppxlib.expression;  (** OCaml fallback for interpreter/native *)
    }
      (** Native code with GPU and OCaml fallback expressions - for PPX quoting
      *)
  | Pragma of string list * k_ext
  | Map of (k_ext * k_ext * k_ext)
  | Unit

and case = int * (string * string * int * string) option * k_ext

type kfun = KernFun of k_ext * k_ext

let string_of_ast a =
  let open Printf in
  let rec soa i s =
    if i = 0 then sprintf "%s\n" s else soa (i - 1) (sprintf "  %s" s)
  in
  let rec aux i = function
    | Kern (a, b) ->
        sprintf "%s%s%s" (soa i "Kern") (aux (i + 1) a) (aux (i + 1) b)
    | Block b -> sprintf "%s%s" (soa i "Kern") (aux (i + 1) b)
    | Params p -> sprintf "%s%s" (soa i "Params") (aux (i + 1) p)
    | Plus (a, b) ->
        sprintf "%s%s%s" (soa i "Plus") (aux (i + 1) a) (aux (i + 1) b)
    | Plusf (a, b) ->
        sprintf "%s%s%s" (soa i "Plusf") (aux (i + 1) a) (aux (i + 1) b)
    | Min (a, b) ->
        sprintf "%s%s%s" (soa i "Min") (aux (i + 1) a) (aux (i + 1) b)
    | Minf (a, b) ->
        sprintf "%s%s%s" (soa i "Minf") (aux (i + 1) a) (aux (i + 1) b)
    | Mul (a, b) ->
        sprintf "%s%s%s" (soa i "Mul") (aux (i + 1) a) (aux (i + 1) b)
    | Mulf (a, b) ->
        sprintf "%s%s%s" (soa i "Mulf") (aux (i + 1) a) (aux (i + 1) b)
    | Div (a, b) ->
        sprintf "%s%s%s" (soa i "Div") (aux (i + 1) a) (aux (i + 1) b)
    | Divf (a, b) ->
        sprintf "%s%s%s" (soa i "Divf") (aux (i + 1) a) (aux (i + 1) b)
    | Mod (a, b) ->
        sprintf "%s%s%s" (soa i "Mod") (aux (i + 1) a) (aux (i + 1) b)
    | Id s -> soa i ("Id " ^ s)
    | IdName s -> soa i ("IdName " ^ s)
    | IntVar (ii, s, _m) -> soa i ("IntVar " ^ string_of_int ii ^ " -> " ^ s)
    | FloatVar (ii, s, _m) -> soa i ("FloatVar " ^ string_of_int ii ^ " -> " ^ s)
    | CastDoubleVar (ii, s) ->
        soa i ("CastDoubleVar " ^ string_of_int ii ^ " ->" ^ s)
    | DoubleVar (ii, s, _m) ->
        soa i ("DoubleVar " ^ string_of_int ii ^ " ->" ^ s)
    | BoolVar (ii, s, _m) -> soa i ("BoolVar " ^ string_of_int ii ^ " ->" ^ s)
    | UnitVar (ii, s, _m) -> soa i ("UnitVar " ^ string_of_int ii ^ " ->" ^ s)
    | VecVar (_t, ii, s) -> soa i ("VecVar " ^ string_of_int ii ^ " ->" ^ s)
    | Concat (a, b) ->
        sprintf "%s%s%s" (soa i "Concat") (aux (i + 1) a) (aux (i + 1) b)
    | Empty -> soa i "Empty"
    | Seq (a, b) ->
        sprintf "%s%s%s" (soa i "Seq") (aux (i + 1) a) (aux (i + 1) b)
    | Return a -> sprintf "%s%s" (soa i "Return") (aux (i + 1) a)
    | Set (a, b) ->
        sprintf "%s%s%s" (soa i "Set") (aux (i + 1) a) (aux (i + 1) b)
    | Decl a -> sprintf "%s%s" (soa i "Decl") (aux (i + 1) a)
    | Acc (a, b) ->
        sprintf "%s%s%s" (soa i "Acc") (aux (i + 1) a) (aux (i + 1) b)
    | SetV (a, b) ->
        sprintf "%s%s%s" (soa i "SetV") (aux (i + 1) a) (aux (i + 1) b)
    | SetLocalVar (a, b, c) ->
        sprintf
          "%s%s%s%s"
          (soa i "SetLocalVar")
          (aux (i + 1) a)
          (aux (i + 1) b)
          (aux (i + 1) c)
    | Intrinsics _ -> soa i "Intrinsics"
    | IntId (s, ii) -> soa i ("IntId " ^ s ^ " " ^ string_of_int ii)
    | Int ii -> soa i ("Int " ^ string_of_int ii)
    | Float f | Double f -> soa i ("Float " ^ string_of_float f)
    | IntVecAcc (a, b) ->
        sprintf "%s%s%s" (soa i "IntVecAcc") (aux (i + 1) a) (aux (i + 1) b)
    | Local (a, b) ->
        sprintf "%s%s%s" (soa i "Local") (aux (i + 1) a) (aux (i + 1) b)
    | Ife (a, b, c) ->
        sprintf
          "%s%s%s%s"
          (soa i "Ife")
          (aux (i + 1) a)
          (aux (i + 1) b)
          (aux (i + 1) c)
    | If (a, b) -> sprintf "%s%s%s" (soa i "If") (aux (i + 1) a) (aux (i + 1) b)
    | EqBool (a, b) ->
        sprintf "%s%s%s" (soa i "EqBool") (aux (i + 1) a) (aux (i + 1) b)
    | EqCustom (_n, a, b) ->
        sprintf "%s%s%s" (soa i "EqSum") (aux (i + 1) a) (aux (i + 1) b)
    | Or (a, b) -> sprintf "%s%s%s" (soa i "Or") (aux (i + 1) a) (aux (i + 1) b)
    | And (a, b) ->
        sprintf "%s%s%s" (soa i "And") (aux (i + 1) a) (aux (i + 1) b)
    | Not a -> sprintf "%s%s" (soa i "Or") (aux (i + 1) a)
    | LtBool (a, b) ->
        sprintf "%s%s%s" (soa i "LtBool") (aux (i + 1) a) (aux (i + 1) b)
    | GtBool (a, b) ->
        sprintf "%s%s%s" (soa i "GtBool") (aux (i + 1) a) (aux (i + 1) b)
    | LtEBool (a, b) ->
        sprintf "%s%s%s" (soa i "LtEBool") (aux (i + 1) a) (aux (i + 1) b)
    | GtEBool (a, b) ->
        sprintf "%s%s%s" (soa i "GtEBool") (aux (i + 1) a) (aux (i + 1) b)
    | DoLoop (a, b, c, d) ->
        sprintf
          "%s%s%s%s%s"
          (soa i "DoLoop")
          (aux (i + 1) a)
          (aux (i + 1) b)
          (aux (i + 1) c)
          (aux (i + 1) d)
    | While (a, b) ->
        sprintf "%s%s%s" (soa i "While") (aux (i + 1) a) (aux (i + 1) b)
    | Arr (s, _l, t, m) ->
        let memspace =
          match m with
          | LocalSpace -> "__private"
          | Shared -> "__local"
          | Global -> "__global"
        and elttype =
          match t with
          | EInt32 -> "int"
          | EInt64 -> "long"
          | EFloat32 -> "float"
          | EFloat64 -> "double"
        in
        soa i ("Arr" ^ s ^ " " ^ memspace ^ " " ^ elttype)
    | App (a, b) ->
        sprintf
          "%s%s%s"
          (soa i "App")
          (aux (i + 1) a)
          (Array.fold_left (fun a b -> a ^ aux (i + 1) b) "" b)
    | GInt _a -> soa i "GInt"
    | GFloat _a -> soa i "GFloat"
    | Unit -> soa i "Unit"
    | GlobalFun (e, s, n) ->
        sprintf "%s%s" (soa i ("Global Fun " ^ s ^ " " ^ n)) (aux (i + 1) e)
    | Constr (s1, s2, l) ->
        sprintf
          "%s%s"
          (soa i ("Constr " ^ s1 ^ " " ^ s2))
          (List.fold_left (fun a b -> a ^ aux (i + 1) b) "" l)
    | Record (s, l) ->
        sprintf
          "%s%s"
          (soa i ("Record " ^ s))
          (List.fold_left (fun a b -> a ^ aux (i + 1) b) "" l)
    | RecGet (r, _s) -> sprintf "%s%s" (soa i "RecGet") (aux (i + 1) r)
    | RecSet (r, v) ->
        sprintf "%s%s%s" (soa i "RecGet") (aux (i + 1) r) (aux (i + 1) v)
    | Custom (s, _, _ss) -> soa i ("Custom " ^ s)
    | Native _f -> soa i "Native "
    | Match (s, e1, l) ->
        sprintf
          "%s%s%s"
          (soa i ("Match " ^ s))
          (aux (i + 1) e1)
          (Array.fold_left (fun _a (_, _, b) -> aux (i + 1) b) "" l)
    | CustomVar _ -> soa i "CustomVar"
    | GFloat64 _ -> soa i "GFloat64"
    | GIntVar n -> soa i ("GIntVar " ^ n)
    | GFloatVar n -> soa i ("GFloatVar " ^ n)
    | GFloat64Var n -> soa i ("GFloat64Var " ^ n)
    | NativeWithFallback _ -> soa i "NativeWithFallback ~gpu:... ~ocaml:..."
    | IntrinsicRef (path, name) ->
        soa i ("IntrinsicRef " ^ String.concat "." (path @ [name]))
    | Pragma _ -> soa i "Pragma"
    | Map _ -> soa i "Map"
  in
  aux 0 a

let print_ast a = Printf.printf "%s\n" (string_of_ast a)
