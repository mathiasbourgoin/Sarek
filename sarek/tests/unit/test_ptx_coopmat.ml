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
let tile ?(la = row) ?(lb = row) ?(lc = row) ?(ld = row) ?(a_buf = None)
    (c : C.config) : kernel =
  let fa, fb, fc, fd = C.fragments_of_config c in
  let load dst frag src layout =
    SCoopmat
      (CM_load {dst; frag; src; index = k_int 0; stride = k_int 16; layout})
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
                 index = k_int 0;
                 stride = k_int 16;
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
      "@%p0 trap;";
    ] ;
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

let shared_acc_kernel () =
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
                     index = k_int 0;
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
  let src = ptx (shared_acc_kernel ()) in
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
  gen "u8*u8+s32 (phase 2)" (cfg C.Uint8 C.Uint8 C.Sint32 C.Sint32) ;
  gen "s8*s8+s32 (phase 2)" (cfg C.Sint8 C.Sint8 C.Sint32 C.Sint32) ;
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

let is_8bit = function C.Uint8 | C.Sint8 -> true | _ -> false

(* PTX accepts nothing CUDA/C refuses, and refuses what CUDA/C accepts only in
   the 8-bit family (phase 2). Every f16-operand configuration agrees. *)
let test_parity () =
  let operands = [C.Float16; C.BFloat16; C.Uint8; C.Sint8] in
  let accs = [C.Float32; C.Sint32; C.Uint32; C.Float16] in
  let n = ref 0 and both = ref 0 in
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          List.iter
            (fun c ->
              List.iter
                (fun r ->
                  let config = cfg a b c r in
                  let k = tile config in
                  let cu = cuda_accepts k and pt = ptx_accepts k in
                  incr n ;
                  if cu && pt then incr both ;
                  let what = C.config_name config in
                  if pt && not cu then
                    Alcotest.failf "PTX accepts what CUDA/C refuses: %s" what ;
                  if cu && (not pt) && not (is_8bit a && is_8bit b) then
                    Alcotest.failf
                      "PTX refuses a non-8-bit configuration CUDA/C accepts: %s"
                      what)
                [c; (if c = C.Float32 then C.Sint32 else C.Float32)])
            accs)
        operands)
    operands ;
  Alcotest.(check bool) "grid is non-trivial" true (!n > 100 && !both >= 1)

(** {1 Kernels without cooperative matrices are unchanged} *)

(* MD5 of the PTX of the f16 regression kernels, recorded from the generator
   before this lowering existed (origin/main 1ad791d5). *)
let baseline_digests =
  [
    ("f16_scale", "78598803fb925e3cc4b201bddbace1df");
    ("f16_chain", "08fe3f98a98a6f17c0c023b61ced98f9");
    ("f16_roundtrip", "d19a79111587c499b24ba208936ca4e7");
    ("f16_local", "d553c156c6fe90bf670f2da7e444783e");
    ("f16_shared", "2de97ee9ac17c726b4c8f2658bda3f7e");
    ("f16_local_array", "a8e58ce1e433b2d6a7371b76efcb6f34");
    ("f16_casts", "a2652c8259a520cade66f8ce8dd43aba");
    ("f16_select", "60c76d9420e7e799477846bcfa4d8ccc");
    ("f16_select_loads", "4280b98c78c653d2e4235c1aa5cd77c1");
    ("f16_helper", "51e76a485b8637b1bf8731953b93161a");
    ("f16_direct_trunc", "63c93e1b4355697a0ce251c7184fa870");
    ("f16_direct_fromint", "d950ef58174c259126b31a1616795089");
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
    "488a54635ad61b356af3ca7cd1b47678"
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
            List.iter
              (fun (name, k) ->
                match assemble (ptx ~sm_target:target k) with
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
          Alcotest.test_case "header and targets" `Quick test_header_and_targets;
          Alcotest.test_case "GEMM" `Quick test_gemm_snapshot;
          Alcotest.test_case "shared source" `Quick test_shared;
        ] );
      ("refusals", [Alcotest.test_case "unsupported" `Quick test_refusals]);
      ("parity", [Alcotest.test_case "same cases as CUDA/C" `Quick test_parity]);
      ( "unchanged",
        [Alcotest.test_case "no coopmat: byte-identical" `Quick test_unchanged]
      );
      ( "ptxas",
        [Alcotest.test_case "assembles (skips without ptxas)" `Quick test_ptxas]
      );
    ]
