(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CM_scale_add end to end (docs/design/coopmat-scaled-accumulate.md).
 *
 *   1. the launch gate on every enumerated device: a gfx11 HIP device permits
 *      the statement; every other device refuses it, naming CM_scale_add when
 *      its backend probed cooperative matrices; a u32 source is refused by
 *      name everywhere;
 *   2. on each gfx11 HIP device: bit-identical to the interpreter over random
 *      int32 and f32 source fragments and random scales (full int32 range, so
 *      |src| > 2^24; one tile in four with scales small enough for subnormal
 *      results), for every combination of col / offset / accumulate, the
 *      in-place f32 rescale, and row scales read from a shared array;
 *   3. the ISA of the generated source for gfx1100 (hipcc -S): with
 *      -ffp-contract=off, which hiprtc is always given, the scale step has no
 *      fused multiply-add; the same source with -ffp-contract=fast has one, so
 *      the check can fail and non-contraction is shown to rest on the flag.
 *
 * Skips (exit 0) the device and ISA parts when no gfx11 HIP device or no hipcc
 * is present.
 ******************************************************************************)

open Sarek_ir_types
open Sarek
open Coopmat_hip_support
module Fw = Spoc_framework.Framework_sig

let () =
  Backend_hip.init () ;
  Test_helpers.Benchmarks.init_backends ()

let failures = ref 0

let is_hip dev = dev.Device.framework = "HIP"

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK  " ^ s)) fmt

(** {1 The kernel} *)

let acc component =
  {
    C.frag_use = C.Accumulator;
    frag_shape = shape;
    frag_component = component;
    frag_scope = C.Subgroup;
  }

let f32_acc = acc C.Float32

let waves = 2 (* per block *)

let tiles = 64

let block = Execute.dims1d (32 * waves)

let grid = Execute.dims1d (tiles / waves)

(* Each scale buffer holds 16 values per tile, plus a margin, and each scale
   sits at its own offset inside its tile's 16 + margin, so an ignored or
   swapped index reads other values. *)
let margin = 8

let scale_len = (tiles * (16 + margin)) + 16

let i32 n = EConst (CInt32 (Int32.of_int n))

let ( +: ) a b = EBinop (Add, a, b)

let ( *: ) a b = EBinop (Mul, a, b)

let ( /: ) a b = EBinop (Div, a, b)

let intrinsic name = EIntrinsic ([], name, [])

let lane_in_block = intrinsic "thread_idx_x"

let wave_in_block = lane_in_block /: i32 32

(* The tile this wave owns, uniform across the wave. *)
let tile = (intrinsic "block_idx_x" *: i32 waves) +: wave_in_block

let scale buf at =
  {scale_buf = buf; scale_index = (tile *: i32 (16 + margin)) +: i32 at}

type variant = {
  float_src : bool;
  col : bool;
  offset : bool;
  accumulate : bool;
  in_place : bool;
  shared_row : bool;
}

let variant_name v =
  Printf.sprintf
    "src %s%s%s%s%s%s"
    (if v.float_src then "f32" else "s32")
    (if v.col then ", col" else "")
    (if v.offset then ", offset" else "")
    (if v.accumulate then ", accumulate" else ", overwrite")
    (if v.in_place then ", in place" else "")
    (if v.shared_row then ", shared row scales" else "")

let vec_param id name elt =
  DParam
    ( {var_name = name; var_id = id; var_type = TVec elt; var_mutable = false},
      Some {arr_elttype = elt; arr_memspace = Global} )

(* Row scales staged in a shared array: lane t < 16 * waves of the block copies
   the row scales of wave t / 16 to rs[t]; then a barrier. *)
let shared_rows =
  let t = lane_in_block in
  let w = t /: i32 16 in
  let src_index =
    (((intrinsic "block_idx_x" *: i32 waves) +: w) *: i32 (16 + margin))
    +: i32 1
    +: EBinop (Mod, t, i32 16)
  in
  SSeq
    [
      SIf
        ( EBinop (Lt, t, i32 (16 * waves)),
          SAssign (LArrayElem ("rs", t), EArrayRead ("r", src_index)),
          None );
      SBarrier;
    ]

let kernel ?(src_component = C.Sint32) v =
  let src_frag =
    if v.float_src || v.in_place then f32_acc else acc src_component
  in
  let cm op = SCoopmat op in
  let base = tile *: i32 256 in
  let load dst frag src =
    cm
      (CM_load
         {dst; frag; src; index = base; stride = i32 16; layout = C.Row_major})
  in
  let src_name = if v.in_place then "fd" else "fs" in
  let row =
    if v.shared_row then
      {scale_buf = "rs"; scale_index = wave_in_block *: i32 16}
    else scale "r" 1
  in
  let body =
    (if v.in_place then [] else [cm (CM_decl {name = "fs"; frag = src_frag})])
    @ [cm (CM_decl {name = "fd"; frag = f32_acc})]
    @ (if v.in_place then [] else [load "fs" src_frag "src"])
    @ [
        load "fd" f32_acc "acc";
        cm
          (CM_scale_add
             {
               dst = "fd";
               dst_frag = f32_acc;
               src = src_name;
               src_frag;
               row;
               col = (if v.col then Some (scale "c" 3) else None);
               offset =
                 (if v.offset then
                    Some {off_row = scale "p" 5; off_col = scale "q" 7}
                  else None);
               accumulate = v.accumulate;
             });
        cm
          (CM_store
             {
               src = "fd";
               frag = f32_acc;
               dst = "out";
               index = base;
               stride = i32 16;
               layout = C.Row_major;
             });
      ]
  in
  {
    default_kernel with
    kern_name = "coopmat_scale_add";
    kern_params =
      [
        vec_param 0 "src" (if v.float_src then TFloat32 else TInt32);
        vec_param 1 "acc" TFloat32;
        vec_param 2 "r" TFloat32;
        vec_param 3 "c" TFloat32;
        vec_param 4 "p" TFloat32;
        vec_param 5 "q" TFloat32;
        vec_param 6 "out" TFloat32;
      ];
    kern_body =
      (if v.shared_row then
         (* Declared in the body, as the matrix rewrite declares its staging
            arrays. *)
         SLet
           ( {
               var_name = "rs";
               var_id = 7;
               var_type = TArray (TFloat32, Shared);
               var_mutable = false;
             },
             EArrayCreate (TFloat32, i32 (16 * waves), Shared),
             SSeq (shared_rows :: body) )
       else SSeq body);
  }

let all_variants =
  let bools = [true; false] in
  List.concat_map
    (fun float_src ->
      List.concat_map
        (fun col ->
          List.concat_map
            (fun offset ->
              List.map
                (fun accumulate ->
                  {
                    float_src;
                    col;
                    offset;
                    accumulate;
                    in_place = false;
                    shared_row = false;
                  })
                bools)
            bools)
        bools)
    bools
  @ [
      {
        float_src = true;
        col = false;
        offset = false;
        accumulate = false;
        in_place = true;
        shared_row = false;
      };
      {
        float_src = false;
        col = true;
        offset = true;
        accumulate = true;
        in_place = false;
        shared_row = true;
      };
    ]

(** {1 Inputs} *)

let rng = Random.State.make [|0x5ca1e; 0xe2e|]

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let random_f32 ~emin ~emax =
  let m = 1.0 +. Random.State.float rng 1.0 in
  let e = emin + Random.State.int rng (emax - emin + 1) in
  let s = if Random.State.bool rng then 1.0 else -1.0 in
  f32 (s *. Float.ldexp m e)

(* Half the source elements over the whole int32 range, half small. *)
let random_int32 () =
  if Random.State.bool rng then Random.State.bits32 rng
  else Int32.of_int (Random.State.int rng 2001 - 1000)

(* One tile in four gets scales near 2^-70, so t3 and t5 are subnormal. *)
let random_scales () =
  Array.init scale_len (fun i ->
      if i / (16 + margin) mod 4 = 0 then random_f32 ~emin:(-72) ~emax:(-66)
      else random_f32 ~emin:(-12) ~emax:4)

type inputs = {
  src_i : int32 array;
  src_f : float array;
  acc_v : float array;
  r : float array;
  c : float array;
  p : float array;
  q : float array;
}

let inputs () =
  let n = tiles * 256 in
  {
    src_i = Array.init n (fun _ -> random_int32 ());
    src_f = Array.init n (fun _ -> random_f32 ~emin:(-20) ~emax:20);
    acc_v = Array.init n (fun _ -> random_f32 ~emin:(-30) ~emax:30);
    r = random_scales ();
    c = random_scales ();
    p = random_scales ();
    q = random_scales ();
  }

let args v i out =
  let src =
    if v.float_src then Execute.Vec (f32_vec i.src_f)
    else Execute.Vec (i32_vec i.src_i)
  in
  Execute.
    [
      src;
      Vec (f32_vec i.acc_v);
      Vec (f32_vec i.r);
      Vec (f32_vec i.c);
      Vec (f32_vec i.p);
      Vec (f32_vec i.q);
      Vec out;
    ]

let run ?gpu v i =
  let out = f32_vec (Array.make (tiles * 256) 0.0) in
  let ir = kernel v in
  (match gpu with
  | Some dev -> run_gpu dev ir (args v i out) ~block ~grid
  | None -> run_interp ir (args v i out) ~block ~grid) ;
  to_floats out

let bits = Int32.bits_of_float

(* NaN payloads are not compared (design §2.4); both NaN is agreement. *)
let same a b = bits a = bits b || (Float.is_nan a && Float.is_nan b)

(** {1 1. The gate} *)

let gate dev ir () = Execute.check_device_capabilities ~device:dev ir

let is_gfx11 dev =
  is_hip dev
  &&
  match dev.Device.capabilities.Fw.coopmat with
  | Some s -> s.C.ds_scale_add
  | None -> false

let check_gate devices =
  let v = List.find (fun v -> not v.float_src) all_variants in
  let ir = kernel v and u32 = kernel ~src_component:C.Uint32 v in
  List.iter
    (fun dev ->
      let probed = dev.Device.capabilities.Fw.coopmat <> None in
      (match (is_gfx11 dev, refusal (gate dev ir)) with
      | true, None -> ok "%s permits CM_scale_add" dev.Device.name
      | true, Some msg -> fail "%s refuses CM_scale_add: %s" dev.Device.name msg
      | false, None -> fail "%s permits CM_scale_add" dev.Device.name
      | false, Some msg when probed && not (contains ~needle:"CM_scale_add" msg)
        ->
          fail "%s refuses without naming CM_scale_add: %s" dev.Device.name msg
      | false, Some _ ->
          ok "%s [%s] refuses CM_scale_add" dev.Device.name dev.Device.framework) ;
      match refusal (gate dev u32) with
      | Some msg when contains ~needle:"u32" msg ->
          ok "%s refuses a u32 source by name" dev.Device.name
      | Some msg when not (is_gfx11 dev) ->
          ok "%s refuses the u32 kernel (%s)" dev.Device.name msg
      | Some msg ->
          fail "%s refuses u32 for another reason: %s" dev.Device.name msg
      | None -> fail "%s permits a u32 source" dev.Device.name)
    devices ;
  let out = f32_vec (Array.make (tiles * 256) 0.0) in
  match
    refusal (fun () -> run_interp u32 (args v (inputs ()) out) ~block ~grid)
  with
  | Some msg when contains ~needle:"u32" msg ->
      ok "the interpreter refuses a u32 source by name"
  | Some msg -> fail "the interpreter refuses u32 for another reason: %s" msg
  | None -> fail "the interpreter evaluated a u32 source"

(** {1 2. Bit identity} *)

let check_variant dev v =
  let i = inputs () in
  let gpu = run ~gpu:dev v i and ref_ = run v i in
  let bad = ref 0 and first = ref None and subnormal = ref 0 in
  Array.iteri
    (fun k x ->
      let y = ref_.(k) in
      if Float.abs y > 0.0 && Float.abs y < Float.ldexp 1.0 (-126) then
        incr subnormal ;
      if not (same x y) then begin
        incr bad ;
        if !first = None then first := Some (k, x, y)
      end)
    gpu ;
  match !first with
  | None ->
      ok
        "%s: %d elements bit-identical to the interpreter (%d subnormal)"
        (variant_name v)
        (Array.length gpu)
        !subnormal
  | Some (k, x, y) ->
      fail
        "%s: %d of %d differ; first at %d: device %h (%08lx), interpreter %h \
         (%08lx)"
        (variant_name v)
        !bad
        (Array.length gpu)
        k
        x
        (bits x)
        y
        (bits y)

(** {1 3. ISA} *)

let hipcc =
  let rocm = Option.value (Sys.getenv_opt "ROCM_PATH") ~default:"/opt/rocm" in
  Filename.concat rocm "bin/hipcc"

let read_file path =
  let ic = open_in_bin path in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic ;
  s

let write_file path s =
  let oc = open_out_bin path in
  output_string oc s ;
  close_out oc

(* hiprtc includes the HIP runtime and fp16 headers implicitly; hipcc does
   not. Same flags as scripts/f16_shape_isa_audit.sh, device code only. *)
let isa ~contract source =
  let src = Filename.temp_file "sarek_scale_add" ".hip" in
  let asm = Filename.chop_suffix src ".hip" ^ ".s" in
  write_file src source ;
  let cmd =
    Printf.sprintf
      "%s -x hip -include hip/hip_runtime.h -include hip/hip_fp16.h \
       --offload-arch=gfx1100 --cuda-device-only -O3 -ffp-contract=%s \
       -fno-gpu-flush-denormals-to-zero -S -o %s %s"
      (Filename.quote hipcc)
      contract
      (Filename.quote asm)
      (Filename.quote src)
  in
  let status = Sys.command cmd in
  let text = if status = 0 && Sys.file_exists asm then read_file asm else "" in
  (try Sys.remove src with Sys_error _ -> ()) ;
  (try Sys.remove asm with Sys_error _ -> ()) ;
  if status = 0 then Some text else None

let fused_ops = ["v_fma"; "v_fmac"; "v_mac_f32"; "v_mad_f32"; "v_pk_fma"]

let has_fused text = List.exists (fun needle -> contains ~needle text) fused_ops

let check_isa () =
  if not (Sys.file_exists hipcc) then print_endline "[SKIP] ISA: no hipcc"
  else
    (* The s32 variant with every step: col, offset and accumulate. *)
    let v = List.find (fun v -> not v.float_src) all_variants in
    let source = Sarek_codegen.Sarek_ir_cuda.generate (kernel v) in
    let check ~what ~contract ~source ~want_fused =
      match isa ~contract source with
      | None -> fail "ISA %s: hipcc failed" what
      | Some text -> (
          (* Steps a contraction would remove are only required when none is
             expected. *)
          let needed =
            if want_fused then ["v_cvt_f32_i32"; "v_mul_f32"]
            else ["v_cvt_f32_i32"; "v_mul_f32"; "v_add_f32"; "v_sub_f32"]
          in
          List.iter
            (fun op ->
              if not (contains ~needle:op text) then
                fail
                  "ISA %s: no %s, so the check is not looking at the step"
                  what
                  op)
            needed ;
          match (has_fused text, want_fused) with
          | false, false -> ok "ISA %s: no fused multiply-add" what
          | true, true -> ok "ISA %s: fused multiply-add present (control)" what
          | true, false -> fail "ISA %s: a fused multiply-add was emitted" what
          | false, true ->
              fail
                "ISA %s: control shows no fused multiply-add, check vacuous"
                what)
    in
    check ~what:"-ffp-contract=off" ~contract:"off" ~source ~want_fused:false ;
    check
      ~what:"-ffp-contract=fast (control)"
      ~contract:"fast"
      ~source
      ~want_fused:true

let () =
  let devices = Array.to_list (Device.init ()) in
  List.iter
    (fun d ->
      Printf.printf "device: %s [%s]\n%!" d.Device.name d.Device.framework)
    devices ;
  check_gate devices ;
  (match List.filter is_gfx11 devices with
  | [] ->
      print_endline
        "[SKIP] no HIP device lowering CM_scale_add: the device checks did not \
         run"
  | gfx11 ->
      List.iter (fun dev -> List.iter (check_variant dev) all_variants) gfx11 ;
      check_isa ()) ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
