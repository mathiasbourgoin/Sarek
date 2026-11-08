(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Cooperative-matrix loads and stores at a start index and a stride that are
 * NOT aligned, against the interpreter, on every device that offers the
 * configuration. [CM_load] / [CM_store] promise any index and any stride
 * ({!Sarek_ir_types}); nvcuda::wmma and the PTX wmma instructions need a
 * 32-byte aligned address and a 16-byte multiple stride, so CUDA/C and CUDA/PTX
 * stage such a tile through a per-warp shared buffer (the other backends have
 * no such precondition). This test is what says the answer is still the
 * interpreter's.
 *
 * For each configuration (f16*f16+f32, s8*s8+s32, u8*u8+s32 where offered),
 * layout combination (row/row/row/row, col/col/col/col, row/col/row/col for A,
 * B, C, D) and (index, stride) case, one tile is run on the device and on the
 * interpreter from the same inputs, in a buffer wider than the tile whose
 * output is pre-filled with a sentinel, and the WHOLE output compared: the
 * tile is right and its neighbours are intact. Inputs are small integers
 * (exact in f16 and f32 whatever the accumulation order), so equality is
 * bitwise.
 *
 * Devices: every one offering the configuration; the environment variable
 * COOPMAT_MISALIGN_FRAMEWORK (e.g. "CUDA/PTX", "HIP", "Vulkan") restricts the
 * run to one framework. The run stops at the first difference or exception.
 * Skips (exit 0) where no device offers a configuration.
 *
 * On an NVIDIA GPU (sm_70+; 8-bit: sm_72+):
 *   dune build sarek/tests/e2e/test_coopmat_misalign_e2e.exe
 *   ./_build/default/sarek/tests/e2e/test_coopmat_misalign_e2e.exe
 * and once more with COOPMAT_MISALIGN_FRAMEWORK=CUDA/C (needs libnvrtc).
 ******************************************************************************)

open Sarek
open Coopmat_hip_support
module Fw = Spoc_framework.Framework_sig

let () =
  Backend_hip.init () ;
  Backend_cuda_c.register () ;
  Test_helpers.Benchmarks.init_backends ()

let failures = ref 0

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK  " ^ s)) fmt

(* A failed check is reported and counted and the next case still runs: every
   failure of a device run is wanted, not the first. *)
exception Case_failed

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s) ;
      raise Case_failed)
    fmt

let guarded f = try f () with Case_failed -> ()

let offers dev cfg =
  Sarek_capability.permits
    (C.verdict ~support:dev.Device.capabilities.Fw.coopmat cfg)

let wave = Execute.dims1d 32

let one = Execute.dims1d 1

let layouts =
  C.
    [
      ("row/row/row/row", (Row_major, Row_major, Row_major, Row_major));
      ("col/col/col/col", (Col_major, Col_major, Col_major, Col_major));
      ("row/col/row/col", (Row_major, Col_major, Row_major, Col_major));
    ]

(* (start index, stride): a misaligned index alone, a misaligned stride alone,
   both, an element offset that is a multiple of 8 but not of 16 (f16), and an
   aligned control. *)
let default_cases = [(0, 16); (1, 16); (16, 17); (3, 19); (8, 24)]

(* COOPMAT_MISALIGN_CASES="0:16,1:16" overrides the cases (index:stride). *)
let cases =
  match Sys.getenv_opt "COOPMAT_MISALIGN_CASES" with
  | None -> default_cases
  | Some spec ->
      String.split_on_char ',' spec
      |> List.map (fun c ->
          match String.split_on_char ':' c with
          | [i; s] -> (int_of_string i, int_of_string s)
          | _ -> failwith "COOPMAT_MISALIGN_CASES: expected index:stride,...")

let small k = (k * 5 mod 9) - 4

let sentinel_f32 p = float_of_int ((p mod 13) + 100)

(* Elements the tile touches: index + 15 * stride + 15, plus margin. *)
let buffer_len ~index ~stride = index + (16 * stride) + 32

(* Typed runner: one function per element family, to keep the vector types
   apart. *)
let run_f16 ?(extra = []) ~gpu ir ~len =
  let a = f16_vec (Array.init len (fun p -> float_of_int (small p))) in
  let b = f16_vec (Array.init len (fun p -> float_of_int (small (p + 3)))) in
  let c = f32_vec (Array.init len (fun p -> float_of_int (small (p + 5)))) in
  let d = f32_vec (Array.init len sentinel_f32) in
  let args = Execute.[Vec a; Vec b; Vec c; Vec d] @ extra in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Array.map (fun x -> x) (Vector.to_array d)

let run_int ?(extra = []) ~gpu ir ~len =
  let a = u8_vec (Array.init len (fun p -> ((p * 7) + 3) land 0xff)) in
  let b = u8_vec (Array.init len (fun p -> ((p * 11) + 5) land 0xff)) in
  let c =
    i32_vec (Array.init len (fun p -> Int32.of_int ((p * 37 mod 1009) - 500)))
  in
  let d = i32_vec (Array.init len (fun p -> Int32.of_int (p + 100000))) in
  let args = Execute.[Vec a; Vec b; Vec c; Vec d] @ extra in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Array.map Int32.to_float (Vector.to_array d)

(* f16 results are bounded, not bit-identical, between a matrix unit and the
   interpreter (RDNA3 WMMA returned -0.999999 for an exact -1 on these integer
   inputs): within 1e-4 relative, against differences of at least 1 for any
   misplaced element. Integer configurations are exact. *)
let close_f16 g w = Float.abs (g -. w) <= 1e-4 *. Float.max 1.0 (Float.abs w)

exception Skip_config

let refused_by_measurement e =
  let msg = Printexc.to_string e in
  let needle = "is refused by measurement" in
  let n = String.length needle and l = String.length msg in
  let rec at i = i + n <= l && (String.sub msg i n = needle || at (i + 1)) in
  at 0

let check_config dev cfg =
  let name = C.config_name cfg in
  let f16 = cfg.C.cfg_a = C.Float16 in
  let close g w = if f16 then close_f16 g w else g = w in
  let run ~gpu ir ~len =
    if f16 then run_f16 ~gpu ir ~len else run_int ~gpu ir ~len
  in
  List.iter
    (fun (lname, (la, lb, lc, ld)) ->
      List.iter
        (fun (index, stride) ->
          guarded @@ fun () ->
          let len = buffer_len ~index ~stride in
          let ir = tile_ir ~index ~stride ~la ~lb ~lc ~ld cfg in
          let want = run ~gpu:None ir ~len in
          let got =
            try run ~gpu:(Some dev) ir ~len with
            | e when refused_by_measurement e ->
                (* the backend refuses this element type on purpose (Vulkan's
                   f16, docs/fp-contraction-policy.md): not a coopmat result *)
                raise Skip_config
            | e ->
                fail
                  "%s %s [%s] index %d stride %d: %s"
                  dev.Device.name
                  name
                  lname
                  index
                  stride
                  (Printexc.to_string e)
          in
          let init =
            Array.init len (fun p ->
                if f16 then sentinel_f32 p else float_of_int (p + 100000))
          in
          let touched = ref 0 and wrong = ref 0 and first = ref None in
          Array.iteri
            (fun p g ->
              if g <> init.(p) then incr touched ;
              if not (close g want.(p)) then begin
                incr wrong ;
                if !first = None then first := Some (p, g, want.(p))
              end)
            got ;
          if !wrong > 0 then
            fail
              "%s %s [%s] index %d stride %d: %d of %d elements differ from \
               the interpreter (first: element %d, device %g, interpreter %g)"
              dev.Device.name
              name
              lname
              index
              stride
              !wrong
              len
              (match !first with Some (p, _, _) -> p | None -> -1)
              (match !first with Some (_, g, _) -> g | None -> nan)
              (match !first with Some (_, _, w) -> w | None -> nan)
          else if !touched < 100 then
            fail
              "%s %s [%s] index %d stride %d: the tile was not written (%d \
               elements changed)"
              dev.Device.name
              name
              lname
              index
              stride
              !touched
          else
            ok
              "%s [%s] index %d stride %d: tile and neighbours match the \
               interpreter"
              name
              lname
              index
              stride)
        cases)
    layouts

(* {1 Overlapping tiles}

   A leading dimension below 16 makes the rows (columns) of a tile overlap in
   memory. A LOAD then reads shared cells, which is well defined and exact on
   every backend (A and B at off 16, ld 12 are run above and below). A STORE
   writes some cells twice with different values, and which write wins is NOT
   specified: the interpreter keeps the last write of its loop (rows outer,
   columns inner, in either layout), but the matrix-unit store of a device
   writes them in an order of its own. The first NVIDIA run of the device test
   failed on exactly that: f16, off 16, ld 12, a tile of A * B + C stored with
   overlapping rows differed from the interpreter in the cells written twice
   (55 of 240), and nowhere else.

   So the comparison skips the cells a store writes twice (checked just above
   for the tiles, below for the copy kernel), and the copy kernel makes the
   semantics explicit: it LOADS a tile of distinct values at a stride of 16 and
   STORES it with an overlapping stride. The interpreter must equal a model of
   its own loop on the host; a device must write every other cell exactly and
   each doubly written cell with one of the values written to it. (An earlier
   version of this kernel loaded and stored with the same overlapping stride,
   so both writers carried the same value and the order could not show.) *)

let overlap_kernel ~index ~stride ~layout =
  let frag =
    {
      C.frag_use = C.Accumulator;
      frag_shape = shape;
      frag_component = C.Float32;
      frag_scope = C.Subgroup;
    }
  in
  let vec id name =
    Sarek_ir_types.DParam
      ( {
          Sarek_ir_types.var_name = name;
          var_id = id;
          var_type = Sarek_ir_types.TVec Sarek_ir_types.TFloat32;
          var_mutable = false;
        },
        Some
          {
            Sarek_ir_types.arr_elttype = Sarek_ir_types.TFloat32;
            arr_memspace = Sarek_ir_types.Global;
          } )
  in
  let k n = Sarek_ir_types.EConst (Sarek_ir_types.CInt32 (Int32.of_int n)) in
  {
    Sarek_ir_types.default_kernel with
    kern_name = "coopmat_overlap";
    kern_params = [vec 0 "c"; vec 1 "d"];
    kern_body =
      Sarek_ir_types.SSeq
        [
          Sarek_ir_types.SCoopmat (Sarek_ir_types.CM_decl {name = "f"; frag});
          (* the source tile is contiguous: distinct values, no overlap *)
          Sarek_ir_types.SCoopmat
            (Sarek_ir_types.CM_load
               {dst = "f"; frag; src = "c"; index = k 0; stride = k 16; layout});
          Sarek_ir_types.SCoopmat
            (Sarek_ir_types.CM_store
               {
                 src = "f";
                 frag;
                 dst = "d";
                 index = k index;
                 stride = k stride;
                 layout;
               });
        ];
  }

let overlap_len ~index = index + (16 * 15) + 40

let overlap_input p = float_of_int (p + 1)

let overlap_init p = -.float_of_int (p + 1)

(* Memory element of tile element (r, c), and of the contiguous source. *)
let element ~index ~stride ~layout r c =
  match layout with
  | C.Row_major -> index + (r * stride) + c
  | C.Col_major -> index + (c * stride) + r

let source_value ~layout r c =
  overlap_input (element ~index:0 ~stride:16 ~layout r c)

(* The cells a store writes, with the values written to each, in the
   interpreter's order (rows outer, columns inner). *)
let writers ~index ~stride ~layout =
  let cells = Array.make (overlap_len ~index) [] in
  for r = 0 to 15 do
    for c = 0 to 15 do
      let e = element ~index ~stride ~layout r c in
      cells.(e) <- cells.(e) @ [source_value ~layout r c]
    done
  done ;
  cells

(* The interpreter's loop, on arrays: the last writer wins. *)
let overlap_model ~index ~stride ~layout =
  let cells = writers ~index ~stride ~layout in
  Array.mapi
    (fun p ws ->
      match List.rev ws with last :: _ -> last | [] -> overlap_init p)
    cells

(* Cells written more than once. *)
let doubly_written ~index ~stride ~layout =
  Array.map (fun ws -> List.length ws > 1) (writers ~index ~stride ~layout)

let run_overlap ~gpu ~index ~stride ~layout =
  let len = overlap_len ~index in
  let c = f32_vec (Array.init len overlap_input) in
  let d = f32_vec (Array.init len overlap_init) in
  let ir = overlap_kernel ~index ~stride ~layout in
  let args = Execute.[Vec c; Vec d] in
  (match gpu with
  | Some dev -> run_gpu dev ir args ~block:wave ~grid:one
  | None -> run_interp ir args ~block:wave ~grid:one) ;
  Array.map (fun x -> x) (Vector.to_array d)

let overlap_cases =
  [(0, 0); (0, 1); (3, 4); (16, 12); (16, 15); (8, 8); (64, -3); (0, 16)]

let layout_name = function
  | C.Row_major -> "row-major"
  | C.Col_major -> "col-major"

let check_overlap what ~gpu =
  List.iter
    (fun layout ->
      List.iter
        (fun (index, stride) ->
          guarded @@ fun () ->
          let model = overlap_model ~index ~stride ~layout in
          let cells = writers ~index ~stride ~layout in
          let got =
            try run_overlap ~gpu ~index ~stride ~layout with
            | e when gpu <> None && refused_by_measurement e ->
                raise Skip_config
            | e when gpu <> None ->
                fail
                  "%s overlap %s index %d stride %d: %s"
                  what
                  (layout_name layout)
                  index
                  stride
                  (Printexc.to_string e)
          in
          (* the interpreter: exactly the model; a device: exact on the cells
             written at most once, one of the written values on the others *)
          let acceptable p g =
            if gpu = None then g = model.(p)
            else
              match cells.(p) with
              | [] -> g = overlap_init p
              | ws -> List.mem g ws
          in
          let wrong = ref 0 and first = ref (-1) in
          Array.iteri
            (fun p g ->
              if not (acceptable p g) then begin
                incr wrong ;
                if !first < 0 then first := p
              end)
            got ;
          if !wrong > 0 then
            fail
              "%s overlap %s index %d stride %d: %d of %d elements are not \
               what the loop writes (first: element %d, got %g, model %g)"
              what
              (layout_name layout)
              index
              stride
              !wrong
              (Array.length got)
              !first
              got.(!first)
              model.(!first)
          else
            ok
              "%s overlap %s index %d stride %d: %s"
              what
              (layout_name layout)
              index
              stride
              (if gpu = None then "the last write of the loop wins"
               else "single writes exact, double writes one of the values"))
        overlap_cases)
    [C.Row_major; C.Col_major]

(* {1 Launch-time variant}

   The same tile with its start index and stride read from the kernel
   parameters [off] and [ld]: the CUDA backends decide at LAUNCH, from these
   values, between the aligned variant (bare wmma, no check, no buffer) and the
   staged one ([Coopmat_variant]); the answer must not change the result. *)

let param_tile cfg =
  let ir = tile_ir cfg in
  let var name =
    Sarek_ir_types.EVar
      {
        Sarek_ir_types.var_name = name;
        var_id = (if name = "off" then 100 else 101);
        var_type = Sarek_ir_types.TInt32;
        var_mutable = false;
      }
  in
  let rec subst = function
    | Sarek_ir_types.SSeq ss -> Sarek_ir_types.SSeq (List.map subst ss)
    | Sarek_ir_types.SCoopmat (Sarek_ir_types.CM_load r) ->
        Sarek_ir_types.SCoopmat
          (Sarek_ir_types.CM_load {r with index = var "off"; stride = var "ld"})
    | Sarek_ir_types.SCoopmat (Sarek_ir_types.CM_store r) ->
        Sarek_ir_types.SCoopmat
          (Sarek_ir_types.CM_store {r with index = var "off"; stride = var "ld"})
    | st -> st
  in
  let int_param name =
    Sarek_ir_types.DParam
      ( {
          Sarek_ir_types.var_name = name;
          var_id = (if name = "off" then 100 else 101);
          var_type = Sarek_ir_types.TInt32;
          var_mutable = false;
        },
        None )
  in
  {
    ir with
    Sarek_ir_types.kern_params =
      ir.Sarek_ir_types.kern_params @ [int_param "off"; int_param "ld"];
    kern_body = subst ir.Sarek_ir_types.kern_body;
  }

(* (off, ld) with the variant the CUDA backends must choose. Operand bytes: 2
   (f16) or 1 (8-bit); the f32 / s32 accumulators are stricter only on the
   index, which the operands already constrain more. *)
let variant_cases f16 =
  let aligned off ld =
    let b = if f16 then 2 else 1 in
    off * b mod 32 = 0 && ld * b mod 16 = 0
  in
  List.map
    (fun (off, ld) -> (off, ld, aligned off ld))
    (if f16 then
       [
         (0, 16);
         (32, 16);
         (16, 24);
         (48, 40);
         (1, 16);
         (8, 16);
         (0, 20);
         (16, 12);
         (3, 19);
       ]
     else [(0, 16); (32, 16); (64, 48); (16, 16); (0, 24); (1, 17); (32, 20)])

let is_cuda dev =
  List.mem dev.Device.framework Sarek_execute.Coopmat_variant.frameworks

let check_variants dev cfg =
  let name = C.config_name cfg in
  let f16 = cfg.C.cfg_a = C.Float16 in
  let ir = param_tile cfg in
  List.iter
    (fun (off, ld, aligned) ->
      guarded @@ fun () ->
      let len = off + (16 * ld) + 32 in
      let extra = Execute.[Int32 (Int32.of_int off); Int32 (Int32.of_int ld)] in
      let run ~gpu =
        if f16 then run_f16 ~extra ~gpu ir ~len else run_int ~extra ~gpu ir ~len
      in
      let want = run ~gpu:None in
      let dummy = f32_vec [|0.0|] in
      let args = Execute.[Vec dummy; Vec dummy; Vec dummy; Vec dummy] @ extra in
      let chosen = Execute.coopmat_variant ~device:dev ir args in
      let expected =
        if is_cuda dev then
          if aligned then Sarek_execute.Coopmat_variant.Aligned
          else Sarek_execute.Coopmat_variant.Staged
        else Sarek_execute.Coopmat_variant.Not_applicable
      in
      if chosen <> expected then
        fail
          "%s %s off %d ld %d: variant %s, expected %s"
          dev.Device.name
          name
          off
          ld
          (Sarek_execute.Coopmat_variant.outcome_name chosen)
          (Sarek_execute.Coopmat_variant.outcome_name expected) ;
      let a0, s0 = Sarek_execute.Coopmat_variant.counts () in
      let got =
        try run ~gpu:(Some dev)
        with e when refused_by_measurement e -> raise Skip_config
      in
      let a1, s1 = Sarek_execute.Coopmat_variant.counts () in
      if is_cuda dev && (a1 - a0, s1 - s0) <> if aligned then (1, 0) else (0, 1)
      then
        fail
          "%s %s off %d ld %d: the launch ran the wrong variant (aligned +%d, \
           staged +%d)"
          dev.Device.name
          name
          off
          ld
          (a1 - a0)
          (s1 - s0) ;
      let close g w = if f16 then close_f16 g w else g = w in
      (* a store with ld < 16 writes cells twice, in an unspecified order on a
         device (see "Overlapping tiles"): those cells are not compared *)
      let twice =
        if ld < 16 then doubly_written ~index:off ~stride:ld ~layout:C.Row_major
        else [||]
      in
      let wrong = ref 0 in
      Array.iteri
        (fun p g ->
          if p < Array.length twice && twice.(p) then ()
          else if not (close g want.(p)) then incr wrong)
        got ;
      if !wrong > 0 then
        fail
          "%s %s off %d ld %d [%s variant]: %d of %d elements differ from the \
           interpreter"
          dev.Device.name
          name
          off
          ld
          (Sarek_execute.Coopmat_variant.outcome_name chosen)
          !wrong
          len
      else
        ok
          "%s off %d ld %d: %s variant, tile and neighbours match the \
           interpreter"
          name
          off
          ld
          (Sarek_execute.Coopmat_variant.outcome_name chosen))
    (variant_cases f16)

(* The GEMM: aligned sizes select the aligned variant, a k or n that is not a
   multiple of 8 the staged one (introspection only: such a launch is outside
   the GEMM's own shape check). *)
let check_gemm_variant dev =
  let ir = Sarek_gemm.Coopmat.ir () in
  let dummy = f32_vec [|0.0|] in
  let args m n k =
    Execute.
      [Vec dummy; Vec dummy; Vec dummy; Vec dummy; Int32 m; Int32 n; Int32 k]
  in
  let expect what args want =
    guarded @@ fun () ->
    let got = Execute.coopmat_variant ~device:dev ir args in
    let want =
      if is_cuda dev then want else Sarek_execute.Coopmat_variant.Not_applicable
    in
    if got <> want then
      fail
        "%s GEMM %s: variant %s, expected %s"
        dev.Device.name
        what
        (Sarek_execute.Coopmat_variant.outcome_name got)
        (Sarek_execute.Coopmat_variant.outcome_name want)
    else
      ok
        "GEMM %s: %s variant"
        what
        (Sarek_execute.Coopmat_variant.outcome_name got)
  in
  expect
    "m=64 n=64 k=128"
    (args 64l 64l 128l)
    Sarek_execute.Coopmat_variant.Aligned ;
  expect
    "m=64 n=40 k=24"
    (args 64l 40l 24l)
    Sarek_execute.Coopmat_variant.Aligned ;
  expect "k=12" (args 64l 64l 12l) Sarek_execute.Coopmat_variant.Staged ;
  expect "n=20" (args 64l 20l 64l) Sarek_execute.Coopmat_variant.Staged

let configs =
  [
    config C.Float16 C.Float16 C.Float32;
    config C.Sint8 C.Sint8 C.Sint32;
    config C.Uint8 C.Uint8 C.Sint32;
  ]

(* The selection itself, on any machine: [Coopmat_variant.choose] for the CUDA
   frameworks, and the source each variant generates (no device is needed to
   generate PTX or CUDA-C). The aligned variant carries no staging buffer, no
   check and no trap; the staged one a per-warp buffer. *)
let host_variants () =
  let module V = Sarek_execute.Coopmat_variant in
  let contains needle src =
    let n = String.length needle and l = String.length src in
    let rec at i = i + n <= l && (String.sub src i n = needle || at (i + 1)) in
    at 0
  in
  let block = (64, 1, 1) in
  let check ?bases what ~framework ir scalars want =
    guarded @@ fun () ->
    let k, got = V.choose ~count:false ?bases ~framework ir scalars in
    if got <> want then
      fail
        "%s on %s: %s, expected %s"
        what
        framework
        (V.outcome_name got)
        (V.outcome_name want) ;
    let ptx = Sarek_codegen.Sarek_ir_ptx.generate ~sm_target:"sm_75" ~block k in
    let cuda =
      Sarek_codegen.Sarek_ir_cuda.generate_with_types ~block ~types:[] k
    in
    let buffer = contains "sarek_cm_tiles" ptx in
    let cuda_buffer = contains "sarek_cm_tiles[" cuda in
    (match got with
    | V.Aligned ->
        if buffer || cuda_buffer || contains "trap" ptx then
          fail "%s: the aligned variant carries a buffer or a trap" what
    | V.Staged ->
        if not (buffer && cuda_buffer) then
          fail "%s: the staged variant has no staging buffer" what
    | V.Checked ->
        (* no static proof: even the constant accesses are tested *)
        if not (buffer && cuda_buffer) then
          fail "%s: the checked variant has no staging buffer" what
    | V.Not_applicable -> ()) ;
    ok
      "%s on %s: %s variant%s"
      what
      framework
      (V.outcome_name got)
      (if buffer then " (2 KiB staging buffer)" else " (no staging buffer)")
  in
  let gemm = Sarek_gemm.Coopmat.ir () in
  List.iter
    (fun framework ->
      check
        "GEMM n=64 k=128"
        ~framework
        gemm
        [("n", 64); ("k", 128); ("m", 64)]
        V.Aligned ;
      check
        "GEMM n=40 k=24"
        ~framework
        gemm
        [("n", 40); ("k", 24); ("m", 64)]
        V.Aligned ;
      check
        "GEMM k=12"
        ~framework
        gemm
        [("n", 64); ("k", 12); ("m", 64)]
        V.Staged ;
      check
        "GEMM n=20"
        ~framework
        gemm
        [("n", 20); ("k", 64); ("m", 64)]
        V.Staged ;
      let f16 = param_tile (config C.Float16 C.Float16 C.Float32) in
      check
        "f16 tile off=32 ld=24"
        ~framework
        f16
        [("off", 32); ("ld", 24)]
        V.Aligned ;
      check
        "f16 tile off=8 ld=16"
        ~framework
        f16
        [("off", 8); ("ld", 16)]
        V.Staged ;
      check
        "f16 tile off=0 ld=20"
        ~framework
        f16
        [("off", 0); ("ld", 20)]
        V.Staged ;
      let u8 = param_tile (config C.Uint8 C.Uint8 C.Sint32) in
      check
        "u8 tile off=64 ld=48"
        ~framework
        u8
        [("off", 64); ("ld", 48)]
        V.Aligned ;
      check
        "u8 tile off=16 ld=16"
        ~framework
        u8
        [("off", 16); ("ld", 16)]
        V.Staged)
    V.frameworks ;
  (* the base pointers: a launch argument that is not 32-byte aligned (a view
     into the middle of an allocation) selects the checked variant, whatever
     the scalar arguments say; an aligned one does not *)
  let by_pointer p = V.alignment_of_pointer (Nativeint.of_int p) in
  let check_int what want got =
    guarded @@ fun () ->
    if want <> got then fail "%s: %d, expected %d" what got want
    else ok "%s: %d" what got
  in
  check_int "pointer 0x1000" 4096 (by_pointer 0x1000) ;
  check_int "pointer 0x10000" 4096 (by_pointer 0x10000) ;
  check_int "pointer 0x1100" 256 (by_pointer 0x1100) ;
  check_int "pointer 0x1010" 16 (by_pointer 0x1010) ;
  check_int "pointer 0x1002" 2 (by_pointer 0x1002) ;
  check_int "pointer 0" 4096 (by_pointer 0) ;
  let tile = param_tile (config C.Float16 C.Float16 C.Float32) in
  let scalars = [("off", 32); ("ld", 24)] in
  let bases a b c d = [("a", a); ("b", b); ("c", c); ("d", d)] in
  List.iter
    (fun framework ->
      let run what bases want =
        guarded @@ fun () ->
        let _, got = V.choose ~count:false ~bases ~framework tile scalars in
        if got <> want then
          fail
            "%s on %s: %s, expected %s"
            what
            framework
            (V.outcome_name got)
            (V.outcome_name want)
        else ok "%s on %s: %s variant" what framework (V.outcome_name got)
      in
      run "bases 256/256/256/256" (bases 256 256 256 256) V.Aligned ;
      run "bases 32/32/32/32" (bases 32 32 32 32) V.Aligned ;
      run "A at +16 bytes" (bases 16 256 256 256) V.Checked ;
      run "D at +4 bytes" (bases 256 256 256 4) V.Checked ;
      run "an argument without a pointer" [("a", 256)] V.Checked ;
      (* misaligned scalars and aligned bases: staged, as before *)
      let _, got =
        V.choose
          ~count:false
          ~bases:(bases 256 256 256 256)
          ~framework
          tile
          [("off", 1); ("ld", 24)]
      in
      guarded (fun () ->
          if got <> V.Staged then fail "off=1 with aligned bases must be staged"))
    V.frameworks ;
  (* the checked variant: constants that are proven everywhere else are tested
     here, and a buffer is declared *)
  let proven_tile = tile_ir (config C.Float16 C.Float16 C.Float32) in
  List.iter
    (fun framework ->
      check
        "constant tile, A misaligned"
        ~framework
        ~bases:(bases 16 256 256 256)
        proven_tile
        []
        V.Checked ;
      check
        "constant tile, aligned bases"
        ~framework
        ~bases:(bases 256 256 256 256)
        proven_tile
        []
        V.Aligned)
    V.frameworks ;
  (* any other framework: the kernel as given *)
  let k, got = V.choose ~count:false ~framework:"HIP" gemm [("n", 64)] in
  guarded (fun () ->
      if got <> V.Not_applicable || k != gemm then
        fail "HIP: the kernel must be left alone" ;
      ok "HIP: kernel left alone")

let () =
  C.accept_relaxed_accumulation
    ~reason:"test_coopmat_misalign_e2e: exact small-integer inputs" ;
  host_variants () ;
  check_overlap "interpreter" ~gpu:None ;
  (* the kernel is not vacuous: at off 16, ld 12 sixty cells are written twice,
     with different values *)
  ( guarded @@ fun () ->
    let cells = writers ~index:16 ~stride:12 ~layout:C.Row_major in
    let twice =
      Array.to_list cells |> List.filter (fun ws -> List.length ws > 1)
    in
    if List.length twice <> 60 then
      fail
        "overlap kernel: %d cells written twice, expected 60"
        (List.length twice)
    else if List.exists (function [a; b] -> a = b | _ -> false) twice then
      fail
        "overlap kernel: two writers carry the same value, the order cannot \
         show"
    else ok "overlap kernel: 60 cells written twice, with different values" ) ;
  let wanted = Sys.getenv_opt "COOPMAT_MISALIGN_FRAMEWORK" in
  let devices =
    Device.init () |> Array.to_list
    |> List.filter (fun d ->
        match wanted with None -> true | Some fw -> d.Device.framework = fw)
  in
  let ran = ref 0 in
  List.iter
    (fun dev ->
      match List.filter (offers dev) configs with
      | [] -> ()
      | offered ->
          incr ran ;
          Printf.printf
            "=== misaligned coopmat: %s [%s] ===\n%!"
            dev.Device.name
            dev.Device.framework ;
          check_gemm_variant dev ;
          (if offers dev f16_config then
             try
               check_overlap
                 (Printf.sprintf "%s [%s]" dev.Device.name dev.Device.framework)
                 ~gpu:(Some dev)
             with Skip_config ->
               Printf.printf "  SKIP overlap: refused by the backend\n%!") ;
          List.iter
            (fun cfg ->
              try
                check_config dev cfg ;
                check_variants dev cfg
              with Skip_config ->
                Printf.printf
                  "  SKIP %s: refused by the backend on purpose\n%!"
                  (C.config_name cfg))
            offered)
    devices ;
  if !failures > 0 then begin
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1
  end ;
  if !ran = 0 then
    print_endline
      "[SKIP] no device offers a cooperative-matrix configuration (or none \
       matches COOPMAT_MISALIGN_FRAMEWORK): the device checks did not run"
  else print_endline "PASS"
