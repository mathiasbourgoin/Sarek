(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** [CM_scale_add] (docs/design/coopmat-scaled-accumulate.md) without a device:
    the interpreter, which is the oracle, on chosen values whose bits change if
    a step is fused, reordered or flushed; its refusals; the launch-gate
    verdict; and the code generators' lowering or named refusal.

    Every chosen case states the bits the order of §2.3 gives AND checks that
    the alternative it guards against (a fused multiply-add, another order, a
    flushed subnormal) gives different bits, so the case can fail. *)

open Sarek_ir_types
open Alcotest
module Interp = Sarek.Sarek_ir_interp
module C = Sarek_coopmat

let shape = {C.m = 16; n = 16; k = 16}

let acc component =
  {
    C.frag_use = C.Accumulator;
    frag_shape = shape;
    frag_component = component;
    frag_scope = C.Subgroup;
  }

let f32_acc = acc C.Float32

let s32_acc = acc C.Sint32

let i32 n = EConst (CInt32 (Int32.of_int n))

let var id name elt =
  {var_name = name; var_id = id; var_type = TVec elt; var_mutable = false}

let vec_param id name elt =
  DParam (var id name elt, Some {arr_elttype = elt; arr_memspace = Global})

(* The scales sit at these offsets in their buffers, so a lowering that ignored
   an index, or swapped two, reads the wrong values. *)
let row_at = 1

let col_at = 2

let p_at = 3

let q_at = 5

let scale buf at = {scale_buf = buf; scale_index = i32 at}

(** One 16x16 tile: [fs] loaded from [src] (unless [in_place], where the
    destination is also the source), [fd] loaded from [acc], the statement, [fd]
    stored to [out]. *)
let kernel ?(src_frag = s32_acc) ?(dst_frag = f32_acc) ?(src_elt = TInt32)
    ?(scale_elt = TFloat32) ?(in_place = false) ?(col = true) ?(offset = true)
    ?(accumulate = true) () =
  let cm op = SCoopmat op in
  let load dst frag src =
    cm
      (CM_load
         {dst; frag; src; index = i32 0; stride = i32 16; layout = C.Row_major})
  in
  let src_name = if in_place then "fd" else "fs" in
  let body =
    (if in_place then [] else [cm (CM_decl {name = "fs"; frag = src_frag})])
    @ [cm (CM_decl {name = "fd"; frag = dst_frag})]
    @ (if in_place then [] else [load "fs" src_frag "src"])
    @ [
        load "fd" dst_frag "acc";
        cm
          (CM_scale_add
             {
               dst = "fd";
               dst_frag;
               src = src_name;
               src_frag;
               row = scale "r" row_at;
               col = (if col then Some (scale "c" col_at) else None);
               offset =
                 (if offset then
                    Some {off_row = scale "p" p_at; off_col = scale "q" q_at}
                  else None);
               accumulate;
             });
        cm
          (CM_store
             {
               src = "fd";
               frag = dst_frag;
               dst = "out";
               index = i32 0;
               stride = i32 16;
               layout = C.Row_major;
             });
      ]
  in
  {
    default_kernel with
    kern_name = "scale_add_tile";
    kern_params =
      [
        vec_param 0 "src" src_elt;
        vec_param 1 "acc" TFloat32;
        vec_param 2 "r" scale_elt;
        vec_param 3 "c" TFloat32;
        vec_param 4 "p" TFloat32;
        vec_param 5 "q" TFloat32;
        vec_param 6 "out" TFloat32;
      ];
    kern_body = SSeq body;
  }

(** {1 Running the interpreter} *)

type inputs = {
  src : Interp.value array;
  acc : float array;
  r : float array;
  c : float array;
  p : float array;
  q : float array;
}

let f32s = Array.map (fun x -> Interp.VFloat32 x)

let run_tile ?(k = kernel ()) (i : inputs) =
  let out = f32s (Array.make 256 0.0) in
  Interp.run_kernel
    k
    ~block:(32, 1, 1)
    ~grid:(1, 1, 1)
    [
      ("src", Interp.ArgArray i.src);
      ("acc", Interp.ArgArray (f32s i.acc));
      ("r", Interp.ArgArray (f32s i.r));
      ("c", Interp.ArgArray (f32s i.c));
      ("p", Interp.ArgArray (f32s i.p));
      ("q", Interp.ArgArray (f32s i.q));
      ("out", Interp.ArgArray out);
    ] ;
  Array.map (function Interp.VFloat32 x -> x | _ -> fail "out") out

(* Every element and every scale the same: the expected value is one number. *)
let uniform ~src ~dst ~r ~c ~p ~q =
  {
    src = Array.make 256 src;
    acc = Array.make 256 dst;
    r = Array.make 24 r;
    c = Array.make 24 c;
    p = Array.make 24 p;
    q = Array.make 24 q;
  }

let s32 n = Interp.VInt32 (Int32.of_int n)

let bits x = Int32.bits_of_float x

let check_all what expected got =
  Array.iteri
    (fun idx x ->
      if bits x <> bits expected then
        failf
          "%s: element %d is %h (bits %08lx), expected %h (bits %08lx)"
          what
          idx
          x
          (bits x)
          expected
          (bits expected))
    got

(* Binary32 rounding, written out here rather than taken from the module under
   test. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let two n = Float.ldexp 1.0 n

(** {1 Chosen values} *)

(* t2 = 1 + 2^-12, t2 * c = 1 + 2^-11 + 2^-24 is a tie, rounded to 1 + 2^-11;
   adding dst = -1 leaves 2^-11. A fused multiply-add keeps 2^-11 + 2^-24. *)
let test_no_fused_accumulate () =
  let rc = 1.0 +. two (-12) in
  let got =
    run_tile
      ~k:(kernel ~offset:false ())
      (uniform ~src:(s32 1) ~dst:(-1.0) ~r:rc ~c:rc ~p:0.0 ~q:0.0)
  in
  let fused = f32 (Float.fma rc rc (-1.0)) in
  check bool "the fused form differs" true (bits fused <> bits (two (-11))) ;
  check_all "dst + (src * r) * c, unfused" (two (-11)) got

(* t5 = p * q = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11; t4 = 1 - t5 = -2^-11. A
   fused multiply-subtract keeps -(2^-11 + 2^-24). *)
let test_no_fused_offset () =
  let pq = 1.0 +. two (-12) in
  let got =
    run_tile
      ~k:(kernel ~col:false ())
      (uniform ~src:(s32 1) ~dst:0.0 ~r:1.0 ~c:0.0 ~p:pq ~q:pq)
  in
  let fused = f32 (Float.fma (-.pq) pq 1.0) in
  check bool "the fused form differs" true (bits fused <> bits (-.two (-11))) ;
  check_all "t4 - p * q, unfused" (-.two (-11)) got

(* (t1 * r) * c, not t1 * (r * c): with t1 = 3, r = 1 + 2^-23 and
   c = 1 - 2^-24 the two orders round differently. *)
let test_order_of_products () =
  let r = 1.0 +. two (-23) and c = 1.0 -. two (-24) in
  let left = f32 (f32 (3.0 *. r) *. c) and right = f32 (3.0 *. f32 (r *. c)) in
  check bool "the two orders differ" true (bits left <> bits right) ;
  let got =
    run_tile
      ~k:(kernel ~offset:false ~accumulate:false ())
      (uniform ~src:(s32 3) ~dst:0.0 ~r ~c ~p:0.0 ~q:0.0)
  in
  check_all "(t1 * r) * c" left got

(* |src| > 2^24: the conversion rounds to nearest even. *)
let test_large_source () =
  let case n expected =
    let got =
      run_tile
        ~k:(kernel ~col:false ~offset:false ~accumulate:false ())
        (uniform ~src:(s32 n) ~dst:0.0 ~r:1.0 ~c:0.0 ~p:0.0 ~q:0.0)
    in
    check_all (Printf.sprintf "f32 (%d)" n) expected got
  in
  case 16777217 16777216.0 ;
  case (-16777219) (-16777220.0) ;
  case 2147483647 2147483648.0 ;
  case (-2147483648) (-2147483648.0)

(* Subnormal results are kept and rounded as IEEE 754 says: 2^-140 exactly, and
   3 * 2^-150, a tie between 2^-149 and 2^-148, to 2^-148. Sarek_float32 would
   flush both to zero. *)
let test_subnormal_kept () =
  let got =
    run_tile
      ~k:(kernel ~offset:false ~accumulate:false ())
      (uniform
         ~src:(s32 1)
         ~dst:0.0
         ~r:(two (-100))
         ~c:(two (-40))
         ~p:0.0
         ~q:0.0)
  in
  check
    bool
    "flushed by Sarek_float32"
    true
    (Sarek_interp.Sarek_float32.mul (two (-100)) (two (-40)) = 0.0) ;
  check_all "2^-140" (two (-140)) got ;
  let got =
    run_tile
      ~k:(kernel ~offset:false ~accumulate:false ())
      (uniform
         ~src:(s32 3)
         ~dst:0.0
         ~r:(two (-75))
         ~c:(two (-75))
         ~p:0.0
         ~q:0.0)
  in
  check_all "3 * 2^-150 rounds to 2^-148" (two (-148)) got

(* accumulate = false does not read the destination: a NaN there is ignored. *)
let test_overwrite_ignores_dst () =
  let got =
    run_tile
      ~k:(kernel ~accumulate:false ())
      (uniform ~src:(s32 5) ~dst:Float.nan ~r:0.5 ~c:4.0 ~p:1.0 ~q:2.0)
  in
  check_all "5 * 0.5 * 4 - 1 * 2" 8.0 got

(* dst = src, f32, row scale only, overwrite: the per-row rescale. *)
let test_in_place_rescale () =
  let k =
    kernel
      ~src_frag:f32_acc
      ~in_place:true
      ~col:false
      ~offset:false
      ~accumulate:false
      ()
  in
  let acc = Array.init 256 (fun i -> float_of_int (i - 100) *. 0.25) in
  let r = Array.init 24 (fun i -> float_of_int (i + 1) *. 0.5) in
  let got =
    run_tile
      ~k
      {
        src = f32s (Array.make 256 0.0);
        acc;
        r;
        c = Array.make 24 0.0;
        p = Array.make 24 0.0;
        q = Array.make 24 0.0;
      }
  in
  Array.iteri
    (fun idx x ->
      let expected = f32 (acc.(idx) *. r.(row_at + (idx / 16))) in
      if bits x <> bits expected then
        failf "element %d: %h, expected %h" idx x expected)
    got

(** {1 Positions: each element takes its own row's and column's scales} *)

let rng = Random.State.make [|0x5ca1e; 0xadd|]

let random_f32 () =
  let m = 1.0 +. Random.State.float rng 1.0 in
  let e = Random.State.int rng 12 - 6 in
  let s = if Random.State.bool rng then 1.0 else -1.0 in
  f32 (s *. Float.ldexp m e)

let reference ~accumulate ~col ~offset (i : inputs) =
  Array.init 256 (fun idx ->
      let row = idx / 16 and column = idx mod 16 in
      let t1 =
        match i.src.(idx) with
        | Interp.VInt32 n -> f32 (Int32.to_float n)
        | Interp.VFloat32 x -> x
        | _ -> fail "src"
      in
      let t2 = f32 (t1 *. i.r.(row_at + row)) in
      let t3 = if col then f32 (t2 *. i.c.(col_at + column)) else t2 in
      let t4 = if accumulate then f32 (i.acc.(idx) +. t3) else t3 in
      if offset then f32 (t4 -. f32 (i.p.(p_at + row) *. i.q.(q_at + column)))
      else t4)

let random_inputs ~float_src =
  {
    src =
      Array.init 256 (fun _ ->
          if float_src then Interp.VFloat32 (random_f32 ())
          else Interp.VInt32 (Random.State.bits32 rng));
    acc = Array.init 256 (fun _ -> random_f32 ());
    r = Array.init 24 (fun _ -> random_f32 ());
    c = Array.init 24 (fun _ -> random_f32 ());
    p = Array.init 24 (fun _ -> random_f32 ());
    q = Array.init 24 (fun _ -> random_f32 ());
  }

let test_positions () =
  List.iter
    (fun (float_src, col, offset, accumulate) ->
      let i = random_inputs ~float_src in
      let src_frag = if float_src then f32_acc else s32_acc in
      let src_elt = if float_src then TFloat32 else TInt32 in
      let got =
        run_tile ~k:(kernel ~src_frag ~src_elt ~col ~offset ~accumulate ()) i
      in
      let expected = reference ~accumulate ~col ~offset i in
      Array.iteri
        (fun idx x ->
          if bits x <> bits expected.(idx) then
            failf
              "src %s, col %b, offset %b, accumulate %b: element %d is %h, \
               expected %h"
              (if float_src then "f32" else "s32")
              col
              offset
              accumulate
              idx
              x
              expected.(idx))
        got)
    (List.concat_map
       (fun f ->
         List.concat_map
           (fun c ->
             List.concat_map
               (fun o -> List.map (fun a -> (f, c, o, a)) [true; false])
               [true; false])
           [true; false])
       [true; false])

(** {1 Refusals of the interpreter} *)

let contains ~needle s =
  let n = String.length needle and l = String.length s in
  let rec at i = i + n <= l && (String.sub s i n = needle || at (i + 1)) in
  at 0

let refusal f =
  match f () with _ -> None | exception e -> Some (Printexc.to_string e)

let expect_refusal what needle f =
  match refusal f with
  | None -> failf "%s: evaluated, expected a refusal" what
  | Some msg ->
      if not (contains ~needle msg) then
        failf "%s: refusal does not mention %S: %s" what needle msg

let some_inputs () = random_inputs ~float_src:false

let test_refuse_u32 () =
  expect_refusal "u32 source" "u32" (fun () ->
      run_tile ~k:(kernel ~src_frag:(acc C.Uint32) ()) (some_inputs ()))

let test_refuse_destination () =
  expect_refusal "s32 destination" "f32 accumulator" (fun () ->
      run_tile ~k:(kernel ~dst_frag:s32_acc ()) (some_inputs ()))

let test_refuse_shape () =
  let small = {(acc C.Sint32) with C.frag_shape = {C.m = 8; n = 8; k = 8}} in
  expect_refusal "shape mismatch" "same shape" (fun () ->
      run_tile ~k:(kernel ~src_frag:small ()) (some_inputs ()))

let test_refuse_int_scale () =
  let i = some_inputs () in
  let k = kernel ~scale_elt:TInt32 () in
  expect_refusal "int32 scale buffer" "float32" (fun () ->
      let out = f32s (Array.make 256 0.0) in
      Interp.run_kernel
        k
        ~block:(32, 1, 1)
        ~grid:(1, 1, 1)
        [
          ("src", Interp.ArgArray i.src);
          ("acc", Interp.ArgArray (f32s i.acc));
          ("r", Interp.ArgArray (Array.make 24 (s32 1)));
          ("c", Interp.ArgArray (f32s i.c));
          ("p", Interp.ArgArray (f32s i.p));
          ("q", Interp.ArgArray (f32s i.q));
          ("out", Interp.ArgArray out);
        ])

let test_refuse_out_of_bounds () =
  let i = some_inputs () in
  expect_refusal "scale past the end" "q" (fun () ->
      run_tile {i with q = Array.make (q_at + 15) 1.0})

(** {1 The launch-gate verdict} *)

let support ~scale_add =
  Some
    {
      C.ds_configs = [];
      ds_robust_buffer_access = false;
      ds_subgroup_size = 32;
      ds_advertised_count = 0;
      ds_evidence = "test";
      ds_scale_add = scale_add;
    }

let verdict_name = function
  | Sarek_capability.Available -> "available"
  | Sarek_capability.Unavailable cap -> cap.Sarek_capability.cap_name
  | Sarek_capability.Unknown _ -> "unknown"

let test_verdict () =
  let v ?(src_frag = s32_acc) ?(dst_frag = f32_acc) s =
    verdict_name (C.scale_add_verdict ~support:s ~dst_frag ~src_frag)
  in
  check string "lowered" "available" (v (support ~scale_add:true)) ;
  check
    string
    "f32 source"
    "available"
    (v ~src_frag:f32_acc (support ~scale_add:true)) ;
  check string "not lowered" "coopmat-scale-add" (v (support ~scale_add:false)) ;
  check string "not probed" "unknown" (v None) ;
  check
    string
    "u32 source"
    "coopmat-scale-add-u32-source"
    (v ~src_frag:(acc C.Uint32) (support ~scale_add:true)) ;
  check
    string
    "s32 destination"
    "coopmat-scale-add-destination"
    (v ~dst_frag:s32_acc (support ~scale_add:true)) ;
  check
    string
    "operand source"
    "coopmat-scale-add-source"
    (v
       ~src_frag:{s32_acc with C.frag_use = C.Matrix_a}
       (support ~scale_add:true))

(** {1 Code generation} *)

let test_hip_lowering () =
  let src = Sarek_codegen.Sarek_ir_cuda.generate (kernel ()) in
  List.iter
    (fun needle ->
      if not (contains ~needle src) then
        failf "the CUDA/HIP source lacks %S:\n%s" needle src)
    [
      "sarek_cm_scale_add<true, true, true>(fd, fs, r, (1), c, (2), p, (3), q, \
       (5));";
      "#error \"Sarek cooperative matrix: CUDA: CM_scale_add is not lowered";
    ]

let refused what needle gen =
  match refusal gen with
  | None -> failf "%s: generated, expected a refusal" what
  | Some msg ->
      if not (contains ~needle msg) then
        failf "%s: refusal does not mention %S: %s" what needle msg

let only_scale_add frag_component =
  let src_frag = acc frag_component in
  let k = kernel ~src_frag () in
  {
    k with
    kern_body =
      SCoopmat
        (CM_scale_add
           {
             dst = "fd";
             dst_frag = f32_acc;
             src = "fs";
             src_frag;
             row = scale "r" 0;
             col = None;
             offset = None;
             accumulate = true;
           });
  }

let test_refusing_generators () =
  refused "CUDA/PTX" "CM_scale_add" (fun () ->
      Sarek_codegen.Sarek_ir_ptx.generate ~sm_target:"sm_80" (kernel ())) ;
  refused "GLSL" "CM_scale_add" (fun () ->
      Sarek_codegen.Sarek_ir_glsl.generate (only_scale_add C.Sint32)) ;
  refused "OpenCL" "cooperative" (fun () ->
      Sarek_codegen.Sarek_ir_opencl.generate (kernel ())) ;
  refused "WGSL" "cooperative" (fun () ->
      Sarek_codegen.Sarek_ir_wgsl.generate (kernel ())) ;
  refused "Metal, s32 source" "no integer simdgroup_matrix" (fun () ->
      Sarek_codegen.Sarek_ir_metal.generate (only_scale_add C.Sint32)) ;
  refused "Metal, f32 source" "CM_scale_add with an f32 source" (fun () ->
      Sarek_codegen.Sarek_ir_metal.generate (only_scale_add C.Float32))

let () =
  run
    "coopmat_scale_add"
    [
      ( "chosen values",
        [
          test_case "no fused accumulate" `Quick test_no_fused_accumulate;
          test_case "no fused offset" `Quick test_no_fused_offset;
          test_case "order of the products" `Quick test_order_of_products;
          test_case "|src| > 2^24" `Quick test_large_source;
          test_case "subnormals kept" `Quick test_subnormal_kept;
          test_case "overwrite ignores dst" `Quick test_overwrite_ignores_dst;
          test_case "in-place rescale" `Quick test_in_place_rescale;
        ] );
      ("positions", [test_case "every flag combination" `Quick test_positions]);
      ( "interpreter refusals",
        [
          test_case "u32 source" `Quick test_refuse_u32;
          test_case "destination" `Quick test_refuse_destination;
          test_case "shape" `Quick test_refuse_shape;
          test_case "int32 scale" `Quick test_refuse_int_scale;
          test_case "out of bounds" `Quick test_refuse_out_of_bounds;
        ] );
      ("gate", [test_case "scale_add_verdict" `Quick test_verdict]);
      ( "codegen",
        [
          test_case "HIP lowering, CUDA #error" `Quick test_hip_lowering;
          test_case "refusing generators" `Quick test_refusing_generators;
        ] );
    ]
