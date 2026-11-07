(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** A tiled f16 GEMM built from cooperative-matrix statements
    (specs/coopmat-hip-wmma.md).

    [D = A x B + C] for row-major matrices: [A] is [m x k] float16, [B] is
    [k x n] float16, [C] and [D] are [m x n] float32. With
    [~b_layout:Col_major], [B] is stored transposed ([n x k], element [(r, c)]
    of [B] at [c * k + r]), which is how token-major activations are laid out
    and what a fragment loads with contiguous reads on HIP gfx11. Each subgroup
    (wavefront / warp) computes a [frag*frags_m x frag*frags_n] tile of [D]
    ([frag] is 16 unless given, 8 on Metal), stepping through [k] [frag] at a
    time with one [frag x frag x frag] multiply-add per fragment pair; a block
    holds [waves] subgroups of 32 invocations.

    The kernel is IR, not a [%kernel]: the DSL has no cooperative-matrix
    surface. It runs where the configuration [f16 * f16 + f32 -> f32] is offered
    — today HIP on gfx11 — and on the interpreter, and in both cases only after
    {!Sarek_coopmat.accept_relaxed_accumulation}: the matrix unit chooses its
    addition order, so the result is within [gamma_k * (sum |A*B| + |C|)] of the
    exact value ([u = 2^-24]), not bit-identical to a scalar loop. *)

open Sarek_ir_types

type tiling = {frags_m : int; frags_n : int; waves : int}

(** 2x2 fragments (a 32x32 tile) per wave, 2 waves per block: the best single
    choice across the downstream LLM prefill shapes measured on the RX 7900 XTX
    (specs/coopmat-hip-wmma.md, Benchmark). It is not the best per shape: at
    [N = 1024] one fragment per wave fills more of the 96 compute units, and at
    [N = 12288] 4x2 is faster. *)
let default_tiling = {frags_m = 2; frags_n = 2; waves = 2}

let subgroup = 32

(** The multiply-add configuration for a [frag x frag x frag] fragment: 16 for
    the HIP/CUDA/Vulkan matrix units, 8 for Metal's [simdgroup_matrix]
    (specs/coopmat-metal-simdgroup.md). *)
let config_for ~frag =
  if frag < 1 || 16 mod frag <> 0 then
    invalid_arg
      (Printf.sprintf
         "Sarek_gemm.Coopmat: frag = %d must be positive and divide 16"
         frag) ;
  {
    Sarek_coopmat.cfg_shape = {Sarek_coopmat.m = frag; n = frag; k = frag};
    cfg_a = Sarek_coopmat.Float16;
    cfg_b = Sarek_coopmat.Float16;
    cfg_c = Sarek_coopmat.Float32;
    cfg_result = Sarek_coopmat.Float32;
    cfg_saturating = false;
    cfg_scope = Sarek_coopmat.Subgroup;
  }

let config = config_for ~frag:16

let tile_m ?(frag = 16) t = frag * t.frags_m

let tile_n ?(frag = 16) t = frag * t.frags_n

(** [check_shape ?frag t ~m ~n ~k] raises [Invalid_argument] unless the problem
    tiles exactly: [m] a multiple of the wave tile height, [n] of its width, [k]
    of [frag] (16 unless given), and the number of wave tiles a multiple of
    [t.waves]. *)
let check_shape ?(frag = 16) t ~m ~n ~k =
  ignore (config_for ~frag) ;
  if t.frags_m < 1 || t.frags_n < 1 || t.waves < 1 then
    invalid_arg "Sarek_gemm.Coopmat: tiling fields must be positive" ;
  let fail what = invalid_arg ("Sarek_gemm.Coopmat: " ^ what) in
  if m <= 0 || n <= 0 || k <= 0 then fail "m, n, k must be positive" ;
  if m mod tile_m ~frag t <> 0 then
    fail (Printf.sprintf "m = %d is not a multiple of %d" m (tile_m ~frag t)) ;
  if n mod tile_n ~frag t <> 0 then
    fail (Printf.sprintf "n = %d is not a multiple of %d" n (tile_n ~frag t)) ;
  if k mod frag <> 0 then
    fail (Printf.sprintf "k = %d is not a multiple of %d" k frag) ;
  let tiles = m / tile_m ~frag t * (n / tile_n ~frag t) in
  if tiles mod t.waves <> 0 then
    fail
      (Printf.sprintf
         "%d wave tiles is not a multiple of %d waves per block"
         tiles
         t.waves)

(** [(block, grid)] for an [m x n] output; call {!check_shape} first. *)
let launch_dims ?(frag = 16) t ~m ~n =
  let tiles = m / tile_m ~frag t * (n / tile_n ~frag t) in
  ( Sarek.Execute.dims1d (t.waves * subgroup),
    Sarek.Execute.dims1d (tiles / t.waves) )

(** {1 The IR} *)

let var id name ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = false}

let i32 n = EConst (CInt32 (Int32.of_int n))

let ( +: ) a b = EBinop (Add, a, b)

let ( *: ) a b = EBinop (Mul, a, b)

let ( /: ) a b = EBinop (Div, a, b)

let ( %: ) a b = EBinop (Mod, a, b)

let intrinsic name = EIntrinsic ([], name, [])

let fragments ~frag =
  let a, b, _, acc = Sarek_coopmat.fragments_of_config (config_for ~frag) in
  (a, b, acc)

let cm op = SCoopmat op

let a_name i = Printf.sprintf "fa%d" i

let b_name j = Printf.sprintf "fb%d" j

let acc_name i j = Printf.sprintf "facc%d_%d" i j

let range n = List.init n Fun.id

let pairs t =
  List.concat_map
    (fun i -> List.map (fun j -> (i, j)) (range t.frags_n))
    (range t.frags_m)

(** One step of the [k] loop: load the wave's A and B fragments for columns
    [16*kk .. 16*kk+15] of A, then every multiply-add. *)
let k_step t ~frag ~b_layout ~row0 ~col0 ~kk ~k ~n =
  let frag_a, frag_b, _ = fragments ~frag in
  let config = config_for ~frag in
  let load_a i =
    cm
      (CM_load
         {
           dst = a_name i;
           frag = frag_a;
           src = "a";
           index = ((row0 +: i32 (frag * i)) *: k) +: (kk *: i32 frag);
           stride = k;
           layout = Sarek_coopmat.Row_major;
         })
  and load_b j =
    cm
      (CM_load
         {
           dst = b_name j;
           frag = frag_b;
           src = "b";
           index =
             (match b_layout with
             | Sarek_coopmat.Row_major ->
                 (kk *: i32 frag *: n) +: (col0 +: i32 (frag * j))
             | Sarek_coopmat.Col_major ->
                 ((col0 +: i32 (frag * j)) *: k) +: (kk *: i32 frag));
           stride =
             (match b_layout with
             | Sarek_coopmat.Row_major -> n
             | Sarek_coopmat.Col_major -> k);
           layout = b_layout;
         })
  and muladd (i, j) =
    cm
      (CM_muladd
         {
           dst = acc_name i j;
           a = a_name i;
           b = b_name j;
           c = acc_name i j;
           cfg = config;
         })
  in
  List.map load_a (range t.frags_m)
  @ List.map load_b (range t.frags_n)
  @ List.map muladd (pairs t)

(** The accumulator tile at [(row0 + 16i, col0 + 16j)] in an [n]-wide buffer. *)
let acc_index ~frag ~row0 ~col0 ~n (i, j) =
  ((row0 +: i32 (frag * i)) *: n) +: (col0 +: i32 (frag * j))

let wave_body t ~frag ~b_layout ~row0 ~col0 ~k ~n ~kk =
  let frag_a, frag_b, frag_acc = fragments ~frag in
  let decl name frag = cm (CM_decl {name; frag}) in
  let decls =
    List.map (fun i -> decl (a_name i) frag_a) (range t.frags_m)
    @ List.map (fun j -> decl (b_name j) frag_b) (range t.frags_n)
    @ List.map (fun (i, j) -> decl (acc_name i j) frag_acc) (pairs t)
  in
  let load_c ij =
    let i, j = ij in
    cm
      (CM_load
         {
           dst = acc_name i j;
           frag = frag_acc;
           src = "c";
           index = acc_index ~frag ~row0 ~col0 ~n ij;
           stride = n;
           layout = Sarek_coopmat.Row_major;
         })
  and store_d ij =
    let i, j = ij in
    cm
      (CM_store
         {
           src = acc_name i j;
           frag = frag_acc;
           dst = "d";
           index = acc_index ~frag ~row0 ~col0 ~n ij;
           stride = n;
           layout = Sarek_coopmat.Row_major;
         })
  in
  let loop =
    SFor
      ( kk,
        i32 0,
        (k /: i32 frag) +: i32 (-1),
        Upto,
        SSeq (k_step t ~frag ~b_layout ~row0 ~col0 ~kk:(EVar kk) ~k ~n) )
  in
  SSeq (decls @ List.map load_c (pairs t) @ [loop] @ List.map store_d (pairs t))

(** [ir ?tiling ?b_layout ()] is the kernel [(a, b, c, d, m, n, k)]:
    [d = a * b + c]. [a], [b] are float16 vectors, [c], [d] float32 vectors,
    [m], [n], [k] int32; [b] is [k x n] row-major, or [n x k] with
    [~b_layout:Col_major]. Launch with {!launch_dims} after {!check_shape}. *)
let ir ?(tiling = default_tiling) ?(frag = 16)
    ?(b_layout = Sarek_coopmat.Row_major) () : kernel =
  let t = tiling in
  let vec id name elt = (var id name (TVec elt), elt) in
  let a = vec 0 "a" TFloat16 and b = vec 1 "b" TFloat16 in
  let c = vec 2 "c" TFloat32 and d = vec 3 "d" TFloat32 in
  let m = var 4 "m" TInt32 and n = var 5 "n" TInt32 and k = var 6 "k" TInt32 in
  let wave = var 7 "wave" TInt32 and kk = var 8 "kk" TInt32 in
  let row0 = var 9 "row0" TInt32 and col0 = var 10 "col0" TInt32 in
  let param (v, elt) =
    DParam (v, Some {arr_elttype = elt; arr_memspace = Global})
  in
  let tiles_n = EVar n /: i32 (tile_n ~frag t) in
  let body =
    SLet
      ( wave,
        (intrinsic "block_idx_x" *: i32 t.waves)
        +: (intrinsic "thread_idx_x" /: i32 subgroup),
        SLet
          ( row0,
            EVar wave /: tiles_n *: i32 (tile_m ~frag t),
            SLet
              ( col0,
                EVar wave %: tiles_n *: i32 (tile_n ~frag t),
                SIf
                  ( EBinop (Lt, EVar row0, EVar m),
                    wave_body
                      t
                      ~frag
                      ~b_layout
                      ~row0:(EVar row0)
                      ~col0:(EVar col0)
                      ~k:(EVar k)
                      ~n:(EVar n)
                      ~kk,
                    None ) ) ) )
  in
  {
    default_kernel with
    kern_name =
      Printf.sprintf
        "sarek_coopmat_gemm_f16_%dx%d_w%d%s%s"
        t.frags_m
        t.frags_n
        t.waves
        (if frag = 16 then "" else Printf.sprintf "_f%d" frag)
        (match b_layout with
        | Sarek_coopmat.Row_major -> ""
        | Sarek_coopmat.Col_major -> "_bt");
    kern_params =
      [param a; param b; param c; param d]
      @ List.map (fun v -> DParam (v, None)) [m; n; k];
    kern_body = body;
  }

(** {1 Host launch} *)

(* One IR value per tiling and B layout, so repeated launches hit Execute's
   launch memo, which keys on the IR's physical identity. *)
let ir_cache : (tiling * int * Sarek_coopmat.layout, kernel) Hashtbl.t =
  Hashtbl.create 4

let ir_cache_lock = Mutex.create ()

let cached_ir tiling frag b_layout =
  Mutex.protect ir_cache_lock (fun () ->
      match Hashtbl.find_opt ir_cache (tiling, frag, b_layout) with
      | Some k -> k
      | None ->
          let k = ir ~tiling ~frag ~b_layout () in
          Hashtbl.add ir_cache (tiling, frag, b_layout) k ;
          k)

(** [run ~device ?tiling ?b_layout ~a ~b ~c ~d ~m ~n ~k ()] launches {!ir} on
    [device]. Raises [Invalid_argument] on a shape {!check_shape} rejects or a
    vector shorter than the shape needs, and the launch gate's capability error
    where the device does not offer the configuration or the program has not
    called {!Sarek_coopmat.accept_relaxed_accumulation}. Asynchronous like any
    launch: read [d] after the usual transfer. *)
let check_lengths ~a ~b ~c ~d ~m ~n ~k =
  let check what v want =
    let got = Spoc_core.Vector.length v in
    if got < want then
      invalid_arg
        (Printf.sprintf
           "Sarek_gemm.Coopmat: %s has %d elements, needs %d"
           what
           got
           want)
  in
  check "a" a (m * k) ;
  check "b" b (k * n) ;
  check "c" c (m * n) ;
  check "d" d (m * n)

let run ~device ?(tiling = default_tiling) ?(frag = 16)
    ?(b_layout = Sarek_coopmat.Row_major) ~a ~b ~c ~d ~m ~n ~k () =
  check_shape ~frag tiling ~m ~n ~k ;
  (* The kernel has no bounds checks: a vector shorter than its shape would be
     read or written out of bounds on the device. *)
  check_lengths ~a ~b ~c ~d ~m ~n ~k ;
  let block, grid = launch_dims ~frag tiling ~m ~n in
  let int x = Sarek.Execute.Int32 (Int32.of_int x) in
  Sarek.Execute.run_vectors
    ~device
    ~ir:(cached_ir tiling frag b_layout)
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Vec c;
        Sarek.Execute.Vec d;
        int m;
        int n;
        int k;
      ]
    ~block
    ~grid
    ()
