(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cooperative matrices in the PTX backend ([wmma.*], Sarek_ir_ptx_coopmat):
    snapshots, refusals, parity of what is accepted with the CUDA/C lowering,
    and byte-identity of kernels that use no cooperative matrix. CPU-only: no
    CUDA device and no toolkit needed. The ptxas assembly gate skips when ptxas
    is not on PATH or does not accept a target. *)

open Sarek_ir_types
open Sarek_codegen
module C = Sarek_coopmat_types

let contains ~needle s =
  let n = String.length needle and l = String.length s in
  let rec go i = i + n <= l && (String.sub s i n = needle || go (i + 1)) in
  go 0

let check_has what ptx needles =
  List.iter
    (fun needle ->
      if not (contains ~needle ptx) then
        Alcotest.failf "%s: expected %S in:\n%s" what needle ptx)
    needles

let check_lacks what ptx needles =
  List.iter
    (fun needle ->
      if contains ~needle ptx then
        Alcotest.failf "%s: unexpected %S in:\n%s" what needle ptx)
    needles

let count ~needle s =
  let n = String.length needle and l = String.length s in
  let rec go i acc =
    if i + n > l then acc
    else go (i + 1) (if String.sub s i n = needle then acc + 1 else acc)
  in
  go 0 0

(** {1 Kernels} *)

let shape16 = {C.m = 16; n = 16; k = 16}

let cfg ?(shape = shape16) ?(sat = false) ?(scope = C.Subgroup) a b c r =
  {
    C.cfg_shape = shape;
    cfg_a = a;
    cfg_b = b;
    cfg_c = c;
    cfg_result = r;
    cfg_saturating = sat;
    cfg_scope = scope;
  }

let f16_cfg = cfg C.Float16 C.Float16 C.Float32 C.Float32

let u8_cfg = cfg C.Uint8 C.Uint8 C.Sint32 C.Sint32

let s8_cfg = cfg C.Sint8 C.Sint8 C.Uint32 C.Uint32

let elt = function
  | C.Float16 -> TFloat16
  | C.BFloat16 | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

let param name e =
  DParam
    ( {var_name = name; var_id = 0; var_type = TVec e; var_mutable = false},
      Some {arr_elttype = e; arr_memspace = Global} )

let row = C.Row_major

let col = C.Col_major

let k_int n = EConst (CInt32 (Int32.of_int n))

(** One tile [d = a * b + c]; [la]/[lb]/[lc]/[ld] are the layouts of the loads
    and of the store. *)
let tile ?(index = 0) ?(stride = 16) ?(la = row) ?(lb = row) ?(lc = row)
    ?(ld = row) ?(a_buf = None) (c : C.config) : kernel =
  let fa, fb, fc, fd = C.fragments_of_config c in
  let load dst frag src layout =
    SCoopmat
      (CM_load
         {dst; frag; src; index = k_int index; stride = k_int stride; layout})
  in
  {
    default_kernel with
    kern_name = "cm";
    kern_params =
      [
        param "a" (Option.value a_buf ~default:(elt c.C.cfg_a));
        param "b" (elt c.C.cfg_b);
        param "c" (elt c.C.cfg_c);
        param "d" (elt c.C.cfg_result);
      ];
    kern_body =
      SSeq
        [
          SCoopmat (CM_decl {name = "fa"; frag = fa});
          SCoopmat (CM_decl {name = "fb"; frag = fb});
          SCoopmat (CM_decl {name = "fc"; frag = fc});
          SCoopmat (CM_decl {name = "fd"; frag = fd});
          load "fa" fa "a" la;
          load "fb" fb "b" lb;
          load "fc" fc "c" lc;
          SCoopmat
            (CM_muladd {dst = "fd"; a = "fa"; b = "fb"; c = "fc"; cfg = c});
          SCoopmat
            (CM_store
               {
                 src = "fd";
                 frag = fd;
                 dst = "d";
                 index = k_int index;
                 stride = k_int stride;
                 layout = ld;
               });
        ];
  }

let ptx ?block ?(sm_target = "sm_75") k =
  Sarek_ir_ptx.generate ~sm_target ?block k

let refused what f =
  match f () with
  | (_ : string) -> Alcotest.failf "%s: generated, expected a refusal" what
  | exception Sarek_ir_ptx_types.Ptx_codegen_error _ -> ()

(** {1 Snapshots} *)

(* The register group of [wmma.load] / [wmma.store] / [wmma.mma] for one
   fragment is the same text everywhere; pull the first group after [prefix]. *)
let group_after ptx prefix =
  let re = Str.regexp (Str.quote prefix ^ "[^{]*{\\([^}]*\\)}") in
  ignore (Str.search_forward re ptx 0) ;
  Str.matched_group 1 ptx

let test_tile_text () =
  let src = ptx (tile f16_cfg) in
  check_has
    "f16 tile"
    src
    [
      ".version 8.0";
      ".target sm_75";
      "wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {";
      "wmma.load.b.sync.aligned.row.m16n16k16.global.f16 {";
      "wmma.load.c.sync.aligned.row.m16n16k16.global.f32 {";
      "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {";
      "wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [";
    ] ;
  (* constants 0 and 16: proven aligned, so the bare instructions *)
  check_lacks "proven tile" src ["trap"; "bra "; "sarek_cm_tiles"; ".shared"] ;
  (* Fragment register groups: the mma consumes exactly the groups the loads
     filled, and stores what it produced. *)
  let a = group_after src "wmma.load.a" and b = group_after src "wmma.load.b" in
  let c = group_after src "wmma.load.c" in
  let mma =
    Str.regexp
      "wmma.mma[^{]*{\\([^}]*\\)}, {\\([^}]*\\)}, {\\([^}]*\\)}, {\\([^}]*\\)}"
  in
  ignore (Str.search_forward mma src 0) ;
  let d = Str.matched_group 1 src in
  Alcotest.(check string) "mma A" a (Str.matched_group 2 src) ;
  Alcotest.(check string) "mma B" b (Str.matched_group 3 src) ;
  Alcotest.(check string) "mma C" c (Str.matched_group 4 src) ;
  let stored =
    let re = Str.regexp "wmma.store[^{]*{\\([^}]*\\)}" in
    ignore (Str.search_forward re src 0) ;
    Str.matched_group 1 src
  in
  Alcotest.(check string) "store D" d stored ;
  let regs g = List.length (String.split_on_char ',' g) in
  Alcotest.(check (list int))
    "8 registers each"
    [8; 8; 8; 8]
    [regs a; regs b; regs c; regs d]

let test_layouts () =
  let src = ptx (tile ~la:col ~lb:col ~lc:col ~ld:col f16_cfg) in
  check_has
    "col-major tile"
    src
    [
      "wmma.load.a.sync.aligned.col.";
      "wmma.load.b.sync.aligned.col.";
      "wmma.load.c.sync.aligned.col.";
      "wmma.mma.sync.aligned.col.col.m16n16k16.f32.f32";
      "wmma.store.d.sync.aligned.col.";
    ] ;
  let src = ptx (tile ~lb:col f16_cfg) in
  check_has "B transposed" src ["wmma.mma.sync.aligned.row.col."]

let test_int8_text () =
  let src = ptx (tile u8_cfg) in
  check_has
    "u8 tile"
    src
    [
      "wmma.load.a.sync.aligned.row.m16n16k16.global.u8 {%fr";
      "wmma.load.b.sync.aligned.row.m16n16k16.global.u8 {%fr";
      "wmma.load.c.sync.aligned.row.m16n16k16.global.s32 {%fr";
      "wmma.mma.sync.aligned.row.row.m16n16k16.s32.u8.u8.s32 {";
      "wmma.store.d.sync.aligned.row.m16n16k16.global.s32 [";
      ".reg .b32 %fr<";
    ] ;
  let regs g = List.length (String.split_on_char ',' g) in
  Alcotest.(check (list int))
    "2 registers per 8-bit operand, 8 per accumulator"
    [2; 2; 8]
    [
      regs (group_after src "wmma.load.a");
      regs (group_after src "wmma.load.b");
      regs (group_after src "wmma.load.c");
    ] ;
  let s8 = ptx (tile ~la:col ~lb:col ~lc:col ~ld:col s8_cfg) in
  check_has
    "s8 col-major tile"
    s8
    [
      "wmma.load.a.sync.aligned.col.m16n16k16.global.s8 {";
      "wmma.mma.sync.aligned.col.col.m16n16k16.s32.s8.s8.s32 {";
      "wmma.store.d.sync.aligned.col.m16n16k16.global.s32 [";
    ] ;
  (* 1-byte elements: the stride check multiplies by 1, the address is not
     scaled. *)
  check_lacks "proven 8-bit tile" src ["trap"; "bra "; "sarek_cm_tiles"] ;
  (* 1-byte elements: the unproven access multiplies the stride by 1 *)
  check_has
    "u8 stride check"
    (ptx ~block:(32, 1, 1) (tile ~index:1 ~stride:17 u8_cfg))
    ["mul.lo.u32 %r"; ", 1;"; "ld.global.u8"; "st.shared.u8"]

let test_header_and_targets () =
  (* sm_70 is the first target with wmma; a suffix is ignored. *)
  List.iter
    (fun t -> check_has t (ptx ~sm_target:t (tile f16_cfg)) ["wmma.mma"])
    ["sm_70"; "sm_75"; "sm_80"; "sm_86"; "sm_90a"] ;
  let default = Sarek_ir_ptx.generate (tile f16_cfg) in
  check_has "default header" default [".version 8.0"; ".target sm_86"] ;
  List.iter
    (fun t -> refused t (fun () -> ptx ~sm_target:t (tile f16_cfg)))
    ["sm_61"; "sm_52"]

let test_gemm_snapshot () =
  let t = Sarek_gemm.Coopmat.default_tiling in
  let per_wave = t.frags_m * t.frags_n in
  let src = ptx (Sarek_gemm.Coopmat.ir ()) in
  Alcotest.(check int)
    "mma per k step"
    per_wave
    (count ~needle:"wmma.mma.sync" src) ;
  Alcotest.(check int)
    "A fragments"
    t.frags_m
    (count ~needle:"wmma.load.a.sync" src) ;
  Alcotest.(check int)
    "B fragments"
    t.frags_n
    (count ~needle:"wmma.load.b.sync" src) ;
  Alcotest.(check int) "C loads" per_wave (count ~needle:"wmma.load.c.sync" src) ;
  Alcotest.(check int)
    "D stores"
    per_wave
    (count ~needle:"wmma.store.d.sync" src) ;
  check_has "gemm" src ["wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32"] ;
  let bt = ptx (Sarek_gemm.Coopmat.ir ~b_layout:C.Col_major ()) in
  check_has
    "gemm B transposed"
    bt
    ["wmma.mma.sync.aligned.row.col."; "wmma.load.b.sync.aligned.col."]

let shared_acc_kernel ?(index = 0) () =
  let frag =
    {
      C.frag_use = C.Accumulator;
      frag_shape = shape16;
      frag_component = C.Float32;
      frag_scope = C.Subgroup;
    }
  in
  let s =
    {
      var_name = "s";
      var_id = 1;
      var_type = TArray (TFloat32, Shared);
      var_mutable = false;
    }
  in
  {
    default_kernel with
    kern_name = "cm_shared";
    kern_params = [param "d" TFloat32];
    kern_body =
      SLet
        ( s,
          EArrayCreate (TFloat32, k_int 256, Shared),
          SSeq
            [
              SCoopmat (CM_decl {name = "f"; frag});
              SCoopmat
                (CM_load
                   {
                     dst = "f";
                     frag;
                     src = "s";
                     index = k_int index;
                     stride = k_int 16;
                     layout = row;
                   });
              SCoopmat
                (CM_store
                   {
                     src = "f";
                     frag;
                     dst = "d";
                     index = k_int 0;
                     stride = k_int 16;
                     layout = row;
                   });
            ] );
  }

let test_shared () =
  (* index 1 is not provably aligned: the check is emitted *)
  let src = ptx (shared_acc_kernel ~index:1 ()) in
  check_has
    "shared source"
    src
    [
      "wmma.load.c.sync.aligned.row.m16n16k16.shared.f32 {";
      "and.b32";
      "wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [";
    ] ;
  (* A shared array a coopmat statement accesses is declared 32-byte aligned
     (wmma's requirement), on both declaration paths (SLet-bound, and DShared);
     one it does not access keeps its element alignment. *)
  check_has "SLet-bound shared" src [".shared .align 32 .b32 s[256];"] ;
  let k = shared_acc_kernel () in
  let declared =
    {
      k with
      kern_locals =
        [
          DShared ("s", TFloat32, Some (k_int 256));
          DShared ("t", TFloat32, Some (k_int 64));
        ];
      kern_body = (match k.kern_body with SLet (_, _, body) -> body | b -> b);
    }
  in
  check_has
    "DShared"
    (ptx declared)
    [".shared .align 32 .b32 s[256];"; ".shared .align 4 .b32 t[64];"]

(** {1 Refusals} *)

let test_refusals () =
  let gen what c = refused what (fun () -> ptx (tile c)) in
  gen
    "shape 16x16x32"
    (cfg
       ~shape:{C.m = 16; n = 16; k = 32}
       C.Float16
       C.Float16
       C.Float32
       C.Float32) ;
  gen
    "workgroup scope"
    (cfg ~scope:C.Workgroup C.Float16 C.Float16 C.Float32 C.Float32) ;
  gen "saturating" (cfg ~sat:true C.Float16 C.Float16 C.Float32 C.Float32) ;
  gen "bf16" (cfg C.BFloat16 C.BFloat16 C.Float32 C.Float32) ;
  gen "mixed-sign u8*s8" (cfg C.Uint8 C.Sint8 C.Sint32 C.Sint32) ;
  gen "mixed-sign s8*u8" (cfg C.Sint8 C.Uint8 C.Uint32 C.Uint32) ;
  gen "8-bit with an f32 accumulator" (cfg C.Uint8 C.Uint8 C.Float32 C.Float32) ;
  gen "8-bit, addend and result differ" (cfg C.Uint8 C.Uint8 C.Sint32 C.Uint32) ;
  gen "saturating 8-bit" (cfg ~sat:true C.Sint8 C.Sint8 C.Sint32 C.Sint32) ;
  refused "8-bit below sm_72" (fun () ->
      ptx ~sm_target:"sm_70" (tile (cfg C.Uint8 C.Uint8 C.Sint32 C.Sint32))) ;
  gen "f16 accumulator" (cfg C.Float16 C.Float16 C.Float16 C.Float16) ;
  gen "mixed f16*bf16" (cfg C.Float16 C.BFloat16 C.Float32 C.Float32) ;
  refused "A from a float32 buffer" (fun () ->
      ptx (tile ~a_buf:(Some TFloat32) f16_cfg)) ;
  refused "partial warp" (fun () -> ptx ~block:(16, 1, 1) (tile f16_cfg)) ;
  refused "partial warp 2-D" (fun () -> ptx ~block:(24, 2, 1) (tile f16_cfg)) ;
  ignore (ptx ~block:(32, 2, 1) (tile f16_cfg)) ;
  ignore (ptx ~block:(16, 2, 1) (tile f16_cfg)) ;
  (* an operand loaded in both layouts has no wmma.mma spelling *)
  let k = tile f16_cfg in
  let k2 =
    match k.kern_body with
    | SSeq ss ->
        let fa, _, _, _ = C.fragments_of_config f16_cfg in
        {
          k with
          kern_body =
            SSeq
              (ss
              @ [
                  SCoopmat
                    (CM_load
                       {
                         dst = "fa";
                         frag = fa;
                         src = "a";
                         index = k_int 0;
                         stride = k_int 16;
                         layout = col;
                       });
                ]);
        }
    | _ -> assert false
  in
  refused "operand in both layouts" (fun () -> ptx k2)

(** {1 Parity with CUDA/C} *)

let cuda_accepts k =
  match Sarek_ir_cuda.generate_with_types ~types:[] k with
  | exception Sarek_backend_error.Backend_error.Backend_error _ -> false
  | src ->
      (* The preamble carries one #error of its own (the HIP gfx11 guard); any
         other is the CUDA arm refusing in the compiler. *)
      count ~needle:"#error \"Sarek cooperative matrix:" src <= 1

let ptx_accepts k =
  match ptx k with
  | (_ : string) -> true
  | exception Sarek_ir_ptx_types.Ptx_codegen_error _ -> false

(* PTX accepts exactly the configurations CUDA/C accepts, over every operand
   pair, every addend/result pair, with and without saturation. *)
let test_parity () =
  let operands = [C.Float16; C.BFloat16; C.Uint8; C.Sint8] in
  let accs = [C.Float32; C.Sint32; C.Uint32; C.Float16] in
  let n = ref 0 and accepted = ref 0 in
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          List.iter
            (fun c ->
              List.iter
                (fun r ->
                  List.iter
                    (fun sat ->
                      let config = cfg ~sat a b c r in
                      let k = tile config in
                      let cu = cuda_accepts k and pt = ptx_accepts k in
                      incr n ;
                      if cu then incr accepted ;
                      if cu <> pt then
                        Alcotest.failf
                          "CUDA/C %s, PTX %s: %s"
                          (if cu then "accepts" else "refuses")
                          (if pt then "accepts" else "refuses")
                          (C.config_name config))
                    [false; true])
                accs)
            accs)
        operands)
    operands ;
  (* f16*f16+f32 and the 8 same-sign 8-bit configurations (2 signs x 2 x 2
     accumulator signednesses, addend = result) *)
  Alcotest.(check int) "grid size" 512 !n ;
  Alcotest.(check int) "accepted configurations" (1 + 4) !accepted

let index_after_ret src =
  let needle = "    ret;\n" in
  let n = String.length needle and l = String.length src in
  let rec last i best =
    if i + n > l then best
    else last (i + 1) (if String.sub src i n = needle then i + n else best)
  in
  last 0 0

(** {1 Misaligned accesses: static proof, trap, staging} *)

module Align = Sarek_ir_coopmat_align

let ivar ?(mutable_ = false) name =
  {var_name = name; var_id = 0; var_type = TInt32; var_mutable = mutable_}

let test_factor () =
  let f ?(env = []) e = Align.factor env e in
  let v name = EVar (ivar name) in
  Alcotest.(check int) "16" 16 (f (k_int 16)) ;
  Alcotest.(check int) "48 has factor 16" 16 (f (k_int 48)) ;
  Alcotest.(check int) "1" 1 (f (k_int 1)) ;
  Alcotest.(check bool) "0 is unbounded" true (f (k_int 0) >= 1 lsl 30) ;
  Alcotest.(check int) "a parameter" 1 (f (v "k")) ;
  Alcotest.(check int) "k * 16" 16 (f (EBinop (Mul, v "k", k_int 16))) ;
  Alcotest.(check int)
    "(k * 16) + 8"
    8
    (f (EBinop (Add, EBinop (Mul, v "k", k_int 16), k_int 8))) ;
  Alcotest.(check int) "k << 5" 32 (f (EBinop (Shl, v "k", k_int 5))) ;
  Alcotest.(check int) "k / 16 is unknown" 1 (f (EBinop (Div, v "k", k_int 16))) ;
  Alcotest.(check int)
    "a let-bound variable takes its definition's factor"
    32
    (f ~env:[("row0", 32)] (v "row0")) ;
  Alcotest.(check int)
    "a mutable variable is unknown"
    1
    (f ~env:[("m", 32)] (EVar (ivar ~mutable_:true "m")))

let accessed_at ~index ~stride body_of =
  let frag =
    {
      C.frag_use = C.Accumulator;
      frag_shape = shape16;
      frag_component = C.Float32;
      frag_scope = C.Subgroup;
    }
  in
  let load =
    SCoopmat (CM_load {dst = "f"; frag; src = "c"; index; stride; layout = row})
  in
  body_of load

let proven body = Align.proven_ops body <> []

let test_proof_walk () =
  let v name = EVar (ivar name) in
  let base = EBinop (Mul, v "row0", k_int 4) in
  let at body_of =
    proven (accessed_at ~index:base ~stride:(k_int 16) body_of)
  in
  (* row0 bound by an immutable let to a multiple of 32: proven *)
  Alcotest.(check bool)
    "let-bound multiple"
    true
    (at (fun load -> SLet (ivar "row0", EBinop (Mul, v "w", k_int 32), load))) ;
  (* the same name as a kernel parameter: unknown *)
  Alcotest.(check bool) "parameter" false (at (fun load -> load)) ;
  (* a for variable shadows the let-bound fact *)
  Alcotest.(check bool)
    "shadowed by a for variable"
    false
    (at (fun load ->
         SLet
           ( ivar "row0",
             k_int 32,
             SFor (ivar "row0", k_int 0, k_int 4, Upto, load) ))) ;
  (* a mutable let is unknown *)
  Alcotest.(check bool)
    "mutable let"
    false
    (at (fun load -> SLetMut (ivar "row0", k_int 32, load))) ;
  (* the stride decides too: a parameter stride is unproven *)
  Alcotest.(check bool)
    "parameter stride"
    false
    (proven (accessed_at ~index:(k_int 0) ~stride:(v "ld") (fun load -> load))) ;
  (* f32 elements: stride 4 is 16 bytes, stride 2 is not *)
  Alcotest.(check bool)
    "f32 stride of 4 elements"
    true
    (proven (accessed_at ~index:(k_int 8) ~stride:(k_int 4) (fun l -> l))) ;
  Alcotest.(check bool)
    "f32 stride of 2 elements"
    false
    (proven (accessed_at ~index:(k_int 8) ~stride:(k_int 2) (fun l -> l))) ;
  Alcotest.(check bool)
    "f32 index of 4 elements is 16 bytes, not 32"
    false
    (proven (accessed_at ~index:(k_int 4) ~stride:(k_int 4) (fun l -> l)))

let slow_blocks src =
  (* everything after the kernel's final ret *)
  let i = index_after_ret src in
  String.sub src i (String.length src - i)

let test_staging () =
  let k = tile ~index:1 ~stride:17 f16_cfg in
  let src = ptx ~block:(32, 1, 1) k in
  check_has
    "staged"
    src
    [
      ".shared .align 32 .b8 sarek_cm_tiles[1024];";
      "bra L";
      "mov.u32 %r";
      "bar.warp.sync 0xffffffff;";
      "wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {";
      "wmma.load.c.sync.aligned.row.m16n16k16.shared.f32 {";
      "wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [";
      "ld.global.b16";
      "st.shared.b16";
      "ld.shared.b32";
      "st.global.b32";
    ] ;
  check_lacks "no trap with a known block" src ["trap"] ;
  (* the direct instruction stays on the fall-through path, the staged one
     lives after the final ret *)
  let slow = slow_blocks src in
  check_has "slow path out of line" slow ["wmma.load.a"; "bar.warp.sync"] ;
  Alcotest.(check bool)
    "no staging copy before the ret"
    false
    (contains
       ~needle:"st.shared.b16"
       (String.sub src 0 (String.length src - String.length slow))) ;
  (* the buffer is sized from the block: 2 x 2 x 1 / 32 ... here 4 warps *)
  check_has "four warps" (ptx ~block:(64, 2, 1) k) [".b8 sarek_cm_tiles[4096];"] ;
  (* the 1-byte and 4-byte copies *)
  let u8 = ptx ~block:(32, 1, 1) (tile ~index:1 ~stride:17 u8_cfg) in
  check_has
    "u8 copies"
    u8
    ["ld.global.u8"; "st.shared.u8"; "ld.shared.b32"; "st.global.b32"] ;
  (* without a block no buffer can be sized: the trap *)
  let trap = ptx k in
  check_has "no block: trap" trap ["@%p"; " trap;"] ;
  check_lacks "no block: no buffer" trap ["sarek_cm_tiles"; "bar.warp.sync"]

let test_gemm_footprint () =
  let t = Sarek_gemm.Coopmat.default_tiling in
  let src = ptx ~block:(32 * t.waves, 1, 1) (Sarek_gemm.Coopmat.ir ()) in
  (* every access of the GEMM takes its stride from a kernel parameter, so none
     is provable: a staging buffer of one tile per warp *)
  check_has
    "GEMM staging buffer"
    src
    [Printf.sprintf ".b8 sarek_cm_tiles[%d];" (t.waves * 1024)] ;
  Alcotest.(check int)
    "one slow path per load and store"
    (t.frags_m + t.frags_n + (2 * t.frags_m * t.frags_n))
    (count ~needle:", sarek_cm_tiles;" src) ;
  Printf.printf
    "  GEMM %dx%d, %d warps: %d bytes of staging shared memory\n%!"
    t.frags_m
    t.frags_n
    t.waves
    (t.waves * 1024)

(** {1 Kernels without cooperative matrices are unchanged} *)

(* MD5 of the PTX of the f16 regression kernels. Recorded from the generator
   before the wmma lowering existed (origin/main 1ad791d5), and re-recorded
   deliberately when the global loads became [ld.global.nc] for the parameters a
   kernel never writes and each global address one [mad.wide.u32]: those two
   change the text of every kernel with a global access. The unrolling of
   runtime-trip loops changes none of them (test_ptx_load_batching checks that
   it is the identity on these kernels). *)
let baseline_digests =
  [
    ("f16_scale", "b4e558634ac84aab2952bfdc1f709f99");
    ("f16_chain", "ee7599296b9d53e6243eb5f53589636a");
    ("f16_roundtrip", "61e2440f65449e822d06ec7f298df11c");
    ("f16_local", "ea2a41f7fe6cfaf8291c6ceba3e52154");
    ("f16_shared", "9b1c2748f6f84ea245cbe14269426e68");
    ("f16_local_array", "16e83234612deae26a60a13cebc34478");
    ("f16_casts", "c200991a523b533e95c7368124b77534");
    ("f16_select", "2baa4bc86eed7f9cc9294ac94a64c516");
    ("f16_select_loads", "c18cf1e2b218baa041c8a2488b22ff57");
    ("f16_helper", "d2aab675141d70900558358f928ebe80");
    ("f16_direct_trunc", "ea0b65d2554d85e8fd89281921223246");
    ("f16_direct_fromint", "45e520b9d59b3e1d5e2b97f13197112b");
  ]

let test_unchanged () =
  List.iter
    (fun (name, k) ->
      let got = Digest.to_hex (Digest.string (Sarek_ir_ptx.generate k)) in
      match List.assoc_opt name baseline_digests with
      | Some want -> Alcotest.(check string) name want got
      | None -> Alcotest.failf "no baseline digest for %s" name)
    (Ptx_f16_kernels.all ()) ;
  let vadd = Sarek_ir_ptx.demo_vector_add_ptx () in
  Alcotest.(check string)
    "vector_add"
    "dca73ce555a14d8c612de34c5845dee6"
    (Digest.to_hex (Digest.string vadd)) ;
  Alcotest.(check bool) "no wmma" false (contains ~needle:"wmma" vadd)

(** {1 ptxas} *)

let ptxas_available =
  lazy
    (match Unix.system "command -v ptxas >/dev/null 2>&1" with
    | Unix.WEXITED 0 -> true
    | _ -> false)

let assemble ptx =
  let base = Filename.temp_file "sarek_ptx_wmma_" "" in
  let src = base ^ ".ptx" and obj = base ^ ".cubin" in
  let oc = open_out src in
  output_string oc ptx ;
  close_out oc ;
  let target =
    ignore
      (Str.search_forward
         (Str.regexp "\\.target[ \t]+\\(sm_[0-9a-z]+\\)")
         ptx
         0) ;
    Str.matched_group 1 ptx
  in
  let rc =
    Unix.system
      (Printf.sprintf
         "ptxas --compile-only --gpu-name %s -o %s %s 2>%s.err"
         (Filename.quote target)
         (Filename.quote obj)
         (Filename.quote src)
         (Filename.quote base))
  in
  let err =
    try
      let ic = open_in (base ^ ".err") in
      let s = really_input_string ic (in_channel_length ic) in
      close_in ic ;
      s
    with _ -> ""
  in
  List.iter
    (fun f -> try Sys.remove f with _ -> ())
    [src; obj; base; base ^ ".err"] ;
  match rc with Unix.WEXITED 0 -> Ok () | _ -> Error err

let test_ptxas () =
  if not (Lazy.force ptxas_available) then
    Printf.printf "  SKIP: ptxas not on PATH; wmma kernels not assembled\n%!"
  else
    let probe target =
      assemble
        (Printf.sprintf
           ".version 8.0\n\
            .target %s\n\
            .address_size 64\n\
            .entry probe()\n\
            {\n\
           \    ret;\n\
            }\n"
           target)
    in
    let kernels =
      [
        ("tile row/row", tile f16_cfg);
        ( "tile staged f16 (index 1, stride 17)",
          tile ~index:1 ~stride:17 f16_cfg );
        ( "tile staged col/col f16",
          tile ~index:3 ~stride:19 ~la:col ~lb:col ~lc:col ~ld:col f16_cfg );
        ("shared staged accumulator", shared_acc_kernel ~index:1 ());
        ("tile u8*u8+s32", tile u8_cfg);
        ("tile u8 staged", tile ~index:1 ~stride:17 u8_cfg);
        ( "tile s8 staged col/col",
          tile ~index:5 ~stride:23 ~la:col ~lb:col ~lc:col ~ld:col s8_cfg );
        ("tile s8*s8+u32", tile s8_cfg);
        ("tile s8 col/col", tile ~la:col ~lb:col ~lc:col ~ld:col s8_cfg);
        ("tile col/col", tile ~la:col ~lb:col ~lc:col ~ld:col f16_cfg);
        ("tile row/col", tile ~lb:col f16_cfg);
        ("shared accumulator", shared_acc_kernel ());
        ("gemm", Sarek_gemm.Coopmat.ir ());
        ("gemm B^T", Sarek_gemm.Coopmat.ir ~b_layout:C.Col_major ());
      ]
    in
    List.iter
      (fun target ->
        match probe target with
        | Error _ ->
            Printf.printf "  SKIP: local ptxas does not accept %s\n%!" target
        | Ok () ->
            let sm =
              Option.value (Sarek_ir_ptx_coopmat.sm_number target) ~default:0
            in
            List.iter
              (fun (name, k) ->
                (* 8-bit wmma starts at sm_72: the generator refuses it below *)
                let is_8bit =
                  (String.length name >= 7 && String.sub name 0 7 = "tile u8")
                  || (String.length name >= 7 && String.sub name 0 7 = "tile s8")
                in
                if is_8bit && sm < 72 then ()
                else
                  match
                    assemble (ptx ~block:(64, 1, 1) ~sm_target:target k)
                  with
                  | Ok () -> Printf.printf "  ptxas OK: %s (%s)\n%!" name target
                  | Error err ->
                      Alcotest.failf
                        "ptxas rejected %s at %s:\n%s"
                        name
                        target
                        err)
              kernels)
      ["sm_70"; "sm_75"; "sm_80"; "sm_86"; "sm_90"]

let () =
  Alcotest.run
    "ptx_coopmat"
    [
      ( "snapshot",
        [
          Alcotest.test_case "f16 tile" `Quick test_tile_text;
          Alcotest.test_case "layouts" `Quick test_layouts;
          Alcotest.test_case "8-bit tile" `Quick test_int8_text;
          Alcotest.test_case "header and targets" `Quick test_header_and_targets;
          Alcotest.test_case "GEMM" `Quick test_gemm_snapshot;
          Alcotest.test_case "shared source" `Quick test_shared;
        ] );
      ("refusals", [Alcotest.test_case "unsupported" `Quick test_refusals]);
      ( "misalignment",
        [
          Alcotest.test_case "alignment factor" `Quick test_factor;
          Alcotest.test_case "proof walk" `Quick test_proof_walk;
          Alcotest.test_case "staging" `Quick test_staging;
          Alcotest.test_case "GEMM footprint" `Quick test_gemm_footprint;
        ] );
      ("parity", [Alcotest.test_case "same cases as CUDA/C" `Quick test_parity]);
      ( "unchanged",
        [Alcotest.test_case "no coopmat: byte-identical" `Quick test_unchanged]
      );
      ( "ptxas",
        [Alcotest.test_case "assembles (skips without ptxas)" `Quick test_ptxas]
      );
    ]
