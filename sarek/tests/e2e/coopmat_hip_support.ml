(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Kernels, inputs and the Regime B gate for test_coopmat_hip_e2e
   (specs/coopmat-hip-wmma.md). *)

open Sarek_ir_types
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module C = Sarek_coopmat

let shape = {C.m = 16; n = 16; k = 16}

let config a b acc =
  {
    C.cfg_shape = shape;
    cfg_a = a;
    cfg_b = b;
    cfg_c = acc;
    cfg_result = acc;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

let f16_config = config C.Float16 C.Float16 C.Float32

let buffer_elt = function
  | C.Float16 -> TFloat16
  | C.BFloat16 | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

(** One 16x16x16 multiply-add [d = a * b + c] for [cfg], row-major, stride 16.
    [drop_c] never loads C, leaving the declared zero: the C-dropping control
    built from the same IR generator, so it takes the same path as the code it
    controls. *)
let tile_ir ?(drop_c = false) ?(index = 0) ?(stride = 16)
    ?(la = Sarek_coopmat_types.Row_major) ?(lb = Sarek_coopmat_types.Row_major)
    ?(lc = Sarek_coopmat_types.Row_major) ?(ld = Sarek_coopmat_types.Row_major)
    cfg : kernel =
  let fa, fb, fc, fd = C.fragments_of_config cfg in
  let v id name elt =
    ( {var_name = name; var_id = id; var_type = TVec elt; var_mutable = false},
      elt )
  in
  let params =
    [
      v 0 "a" (buffer_elt cfg.C.cfg_a);
      v 1 "b" (buffer_elt cfg.C.cfg_b);
      v 2 "c" (buffer_elt cfg.C.cfg_c);
      v 3 "d" (buffer_elt cfg.C.cfg_result);
    ]
  in
  let k0 = EConst (CInt32 (Int32.of_int index))
  and k16 = EConst (CInt32 (Int32.of_int stride)) in
  let load dst frag src layout =
    SCoopmat (CM_load {dst; frag; src; index = k0; stride = k16; layout})
  in
  let body =
    [
      SCoopmat (CM_decl {name = "fa"; frag = fa});
      SCoopmat (CM_decl {name = "fb"; frag = fb});
      SCoopmat (CM_decl {name = "fc"; frag = fc});
      SCoopmat (CM_decl {name = "fd"; frag = fd});
      load "fa" fa "a" la;
      load "fb" fb "b" lb;
    ]
    @ (if drop_c then [] else [load "fc" fc "c" lc])
    @ [
        SCoopmat (CM_muladd {dst = "fd"; a = "fa"; b = "fb"; c = "fc"; cfg});
        SCoopmat
          (CM_store
             {
               src = "fd";
               frag = fd;
               dst = "d";
               index = k0;
               stride = k16;
               layout = ld;
             });
      ]
  in
  {
    default_kernel with
    kern_name = "coopmat_tile";
    kern_params =
      List.map
        (fun (var, elt) ->
          DParam (var, Some {arr_elttype = elt; arr_memspace = Global}))
        params;
    kern_body = SSeq body;
  }

(** {1 Vectors} *)

let f16_vec arr =
  let v = Vector.create Vector.float16 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

let f32_vec arr =
  let v = Vector.create Vector.float32 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

let u8_vec arr =
  let v = Vector.create Vector.char (Array.length arr) in
  Array.iteri (fun i x -> Vector.set v i (Char.chr x)) arr ;
  v

let i32_vec arr =
  let v = Vector.create Vector.int32 (Array.length arr) in
  Array.iteri (Vector.set v) arr ;
  v

(** {1 f16 inputs whose exact products sum exactly in binary64}

    Every value is [+-(1024 + m) / 1024 * 2^e] with [m] in [0, 1023], so it is a
    binary16 value by construction. Operands take [e] in [-4, 0] and C takes [e]
    in [-1, 2]: product exponents lie in [-8, 1] and the term exponent span is
    at most 10 binades, which {!assert_exact_in_binary64} turns into a checked
    invariant. *)
let rng = Random.State.make [|0x5a4e; 0x7e1c|]

let f16_value ~emin ~emax =
  let m = Random.State.int rng 1024 in
  let e = emin + Random.State.int rng (emax - emin + 1) in
  let s = if Random.State.bool rng then 1.0 else -1.0 in
  s *. Float.ldexp (float_of_int (1024 + m) /. 1024.0) e

let operand n = Array.init n (fun _ -> f16_value ~emin:(-4) ~emax:0)

let addend n = Array.init n (fun _ -> f16_value ~emin:(-1) ~emax:2)

(** {1 The Regime B gate (docs/design/f16-relaxed-accuracy.md §5)} *)

let u = Float.ldexp 1.0 (-24)

(** [gamma n]: the bound factor for [n] binary32 additions in any order. *)
let gamma n =
  let nu = float_of_int n *. u in
  nu /. (1.0 -. nu)

(** Exact [D = A x B + C] and [sum |terms|] for row-major [m x k], [k x n],
    [m x n]. Exact because of {!assert_exact_in_binary64}. *)
let exact_gemm ~a ~b ~c ~m ~n ~k =
  let d = Array.make (m * n) 0.0 and s = Array.make (m * n) 0.0 in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let acc = ref c.((i * n) + j) and abs = ref (Float.abs c.((i * n) + j)) in
      for x = 0 to k - 1 do
        let p = a.((i * k) + x) *. b.((x * n) + j) in
        acc := !acc +. p ;
        abs := !abs +. Float.abs p
      done ;
      d.((i * n) + j) <- !acc ;
      s.((i * n) + j) <- !abs
    done
  done ;
  (d, s)

(** §5.3: binary64 sums [k + 1] terms exactly when the terms' exponent span plus
    [25 + ceil (log2 (k + 1))] is at most 53. The generator above keeps the span
    at 10; this checks the generated data rather than trusting it. *)
let assert_exact_in_binary64 ~a ~b ~c ~k =
  let exps arr = Array.map (fun x -> snd (Float.frexp x)) arr in
  let range arr =
    Array.fold_left
      (fun (lo, hi) e -> (min lo e, max hi e))
      (max_int, min_int)
      arr
  in
  let alo, ahi = range (exps a) and blo, bhi = range (exps b) in
  let clo, chi = range (exps c) in
  let lo = min (alo + blo) clo and hi = max (ahi + bhi) chi in
  let log2k = int_of_float (Float.ceil (Float.log2 (float_of_int (k + 1)))) in
  if hi - lo + 25 + log2k > 53 then
    failwith
      (Printf.sprintf
         "input generator broke the binary64 exactness invariant: span %d"
         (hi - lo))

(** [(violations, worst)]: elements of [got] outside [gamma_adds * sum|terms|]
    of [exact], and the largest [|got - exact| / (gamma_adds * sum|terms|)] (at
    most 1 when none). *)
let bound_check ~adds ~exact ~absterms got =
  let g = gamma adds in
  let bad = ref 0 and worst = ref 0.0 in
  Array.iteri
    (fun i x ->
      let r = Float.abs (x -. exact.(i)) /. (g *. absterms.(i)) in
      if r > 1.0 then incr bad ;
      if r > !worst then worst := r)
    got ;
  (!bad, !worst)

(** The binary16-accumulating control of §5.4: the same products summed in order
    with every partial sum rounded to binary16. *)
let f16_accumulating ~a ~b ~c ~m ~n ~k =
  let r = Sarek_interp.Sarek_float16.to_float16 in
  Array.init (m * n) (fun idx ->
      let i = idx / n and j = idx mod n in
      let acc = ref (r c.(idx)) in
      for x = 0 to k - 1 do
        acc := r (!acc +. (a.((i * k) + x) *. b.((x * n) + j)))
      done ;
      !acc)

(** {1 Runners} *)

let run_gpu dev ir args ~block ~grid =
  Execute.run_vectors ~device:dev ~ir ~args ~block ~grid () ;
  Transfer.flush dev

let run_interp ir args ~block ~grid =
  Execute.run_interpreter_vectors ~ir ~args ~block ~grid ~parallel:false

let to_floats v = Array.map (fun x -> x) (Vector.to_array v)

(** [refusal f] is [Some message] if [f ()] raises, [None] if it returns. *)
let refusal f =
  match f () with () -> None | exception e -> Some (Printexc.to_string e)

let contains ~needle s = Test_helpers.string_contains ~needle s
