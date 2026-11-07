(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The 8-bit storage types [TUint8]/[TInt8] without a device:

    - the interpreter (the oracle every device is compared with) against an
      independent host model, on reads, stores, casts, global and shared arrays,
      over every byte value and int32 inputs that are negative and above 255;
    - the spelling each backend emits for them, which is what a device run then
      measures (test_int8_storage_e2e), and WGSL's named refusal;
    - the refusals every backend and the interpreter share: an 8-bit scalar
      binder, record field, private array or ambiguous array name;
    - the capability: a device that does not report [Int8] is refused. *)

open Sarek_ir_types
open Sarek_codegen
module K = Int8_kernels
module Execute = Sarek.Execute
module Vector = Spoc_core.Vector

let contains ~needle s =
  let n = String.length needle and l = String.length s in
  let rec go i = i + n <= l && (String.sub s i n = needle || go (i + 1)) in
  n = 0 || go 0

let check_has what src needle =
  if not (contains ~needle src) then
    Alcotest.failf "%s: expected %S in:\n%s" what needle src

let check_absent what src needle =
  if contains ~needle src then
    Alcotest.failf "%s: expected no %S in:\n%s" what needle src

(** {1 Host vectors} *)

let vec_i32 a =
  let v = Vector.create Vector.int32 (Array.length a) in
  Array.iteri (Vector.set v) a ;
  v

let vec_u8 a =
  let v = Vector.create Vector.char (Array.length a) in
  Array.iteri (fun i x -> Vector.set v i (Char.chr x)) a ;
  v

let vec_s8 a =
  let v = Vector.create Vector.int8 (Array.length a) in
  Array.iteri (Vector.set v) a ;
  v

let u8_contents v = Array.map Char.code (Vector.to_array v)

let check_i32_array what expected got =
  Array.iteri
    (fun i e ->
      if got.(i) <> e then
        Alcotest.failf "%s: element %d is %ld, expected %ld" what i got.(i) e)
    expected

let check_int_array what expected got =
  Array.iteri
    (fun i e ->
      if got.(i) <> e then
        Alcotest.failf "%s: element %d is %d, expected %d" what i got.(i) e)
    expected

let interp ir args ~threads =
  Execute.run_interpreter_vectors
    ~ir
    ~args
    ~block:(Execute.dims1d K.block)
    ~grid:(Execute.dims1d (threads / K.block))
    ~parallel:false

(** {1 Interpreter against the host model} *)

let test_interp_rw () =
  let out = vec_i32 (Array.make (K.rw_slots * K.n) 0l)
  and dst_u8 = vec_u8 (Array.make K.n 0)
  and dst_s8 = vec_s8 (Array.make K.n 0) in
  interp
    (K.rw_kernel ())
    [
      Execute.Vec (vec_u8 K.src_u8);
      Execute.Vec (vec_s8 K.src_s8);
      Execute.Vec (vec_i32 K.vals);
      Execute.Vec out;
      Execute.Vec dst_u8;
      Execute.Vec dst_s8;
    ]
    ~threads:K.n ;
  let e_out, e_u8, e_s8 =
    K.rw_model ~src_u8:K.src_u8 ~src_s8:K.src_s8 ~vals:K.vals ()
  in
  check_i32_array "out" e_out (Vector.to_array out) ;
  check_int_array "dst_u8" e_u8 (u8_contents dst_u8) ;
  check_int_array "dst_s8" e_s8 (Vector.to_array dst_s8)

let test_interp_shared () =
  let out = vec_i32 (Array.make (2 * K.n) 0l) in
  interp
    (K.shared_kernel ())
    [Execute.Vec (vec_i32 K.vals); Execute.Vec out]
    ~threads:K.n ;
  check_i32_array "out" (K.shared_model ~vals:K.vals ()) (Vector.to_array out)

let test_interp_scalar () =
  let out = vec_i32 (Array.make (K.scalar_slots * K.n) 0l) in
  interp
    (K.scalar_kernel ())
    [Execute.Vec (vec_i32 K.vals); Execute.Vec out]
    ~threads:K.n ;
  check_i32_array "out" (K.scalar_model ~vals:K.vals ()) (Vector.to_array out)

(** The inputs reach what the tests claim: values above 255 and below -128, and
    every byte value on both host kinds. *)
let test_inputs_cover_the_edges () =
  let has p = Array.exists p K.vals in
  if not (has (fun v -> v > 255l) && has (fun v -> v < -128l)) then
    Alcotest.fail "vals has no value outside the 8-bit ranges" ;
  for b = 0 to 255 do
    if not (Array.mem b K.src_u8) then Alcotest.failf "src_u8 misses %d" b ;
    if not (Array.mem (K.s8 b) K.src_s8) then
      Alcotest.failf "src_s8 misses %d" (K.s8 b)
  done

(** A [char] vector cannot stand in for an int8 parameter: its bytes would read
    back unsigned on the CPU backends and signed on a device. *)
let test_char_vector_refused_for_int8 () =
  let k = K.rw_kernel () in
  match
    interp
      k
      [
        Execute.Vec (vec_u8 K.src_u8);
        Execute.Vec (vec_u8 K.src_u8);
        Execute.Vec (vec_i32 K.vals);
        Execute.Vec (vec_i32 (Array.make (K.rw_slots * K.n) 0l));
        Execute.Vec (vec_u8 (Array.make K.n 0));
        Execute.Vec (vec_s8 (Array.make K.n 0));
      ]
      ~threads:K.n
  with
  | () -> Alcotest.fail "a char vector was accepted for an int8 parameter"
  | exception e ->
      (* The launch check names both element types; [uint8 vector] is the
         argument's, so this cannot match the kernel's name by accident. *)
      let msg = Printexc.to_string e in
      check_has "refusal" msg "uint8 vector" ;
      (* and the parameter's: " int8 vector" with the space, which
         "uint8 vector" does not contain *)
      check_has "refusal" msg " int8 vector"

(** {1 Emitted spelling} *)

let all_kernels = [K.rw_kernel (); K.shared_kernel (); K.scalar_kernel ()]

let gen_all f = String.concat "\n" (List.map f all_kernels)

(** Every conversion of an int to a signed 8-bit type is the explicit
    sign-extension of the low byte, never a C conversion: converting an
    out-of-range [int] to [signed char] is implementation-defined in C, C++
    before C++20, OpenCL C and MSL. Checked for the cast, the global store, the
    shared store, and an int8 local's initialiser and reassignment; and no
    signed-char cast remains. *)
let signed_narrowing_sites =
  [
    "out[((i * 6) + 3)] = ((((vals[i]) & 0xFF) ^ 0x80) - 0x80);";
    "dst_s8[i] = (((((vals[i] ^ 90)) & 0xFF) ^ 0x80) - 0x80);";
    "ss[t] = (((((vals[g] ^ 51)) & 0xFF) ^ 0x80) - 0x80);";
    "int y = (((((vals[i] ^ 17)) & 0xFF) ^ 0x80) - 0x80);";
    "y = (((((y + 200)) & 0xFF) ^ 0x80) - 0x80);";
  ]

let check_c_family what src ~u8 ~s8 ~global ~restrict ~shared =
  List.iter
    (check_has what src)
    ([
       Printf.sprintf "%s%s* %ssrc_u8" global u8 restrict;
       Printf.sprintf "%s%s* %ssrc_s8" global s8 restrict;
       Printf.sprintf "(%s)vals[i]" u8;
       Printf.sprintf "%s%s su[64]" shared u8;
       Printf.sprintf "%s%s ss[64]" shared s8;
     ]
    @ signed_narrowing_sites) ;
  check_absent what src (Printf.sprintf "(%s)" s8)

let test_cuda_spelling () =
  (* CUDA C and HIP share this emitter. *)
  let src = gen_all Sarek_ir_cuda.generate in
  check_c_family
    "CUDA/HIP"
    src
    ~u8:"unsigned char"
    ~s8:"signed char"
    ~global:""
    ~restrict:"__restrict__ "
    ~shared:"__shared__ "

let test_opencl_spelling () =
  let src = gen_all Sarek_ir_opencl.generate in
  check_c_family
    "OpenCL"
    src
    ~u8:"uchar"
    ~s8:"char"
    ~global:"__global "
    ~restrict:"restrict "
    ~shared:"__local "

let test_metal_spelling () =
  let src = gen_all Sarek_ir_metal.generate in
  check_c_family
    "Metal"
    src
    ~u8:"uchar"
    ~s8:"char"
    ~global:"device "
    ~restrict:""
    ~shared:"threadgroup "

let test_glsl_spelling () =
  let rw = Sarek_ir_glsl.generate ~block:(K.block, 1, 1) (K.rw_kernel ()) in
  let sh = Sarek_ir_glsl.generate ~block:(K.block, 1, 1) (K.shared_kernel ()) in
  List.iter
    (check_has "GLSL rw" rw)
    [
      "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require";
      "uint8_t src_u8[]";
      "int8_t src_s8[]";
      (* reads widen, stores narrow once, casts round-trip *)
      "int(src_u8[i])";
      "int(src_s8[i])";
      "dst_u8[i] = uint8_t(vals[i]);";
      "dst_s8[i] = int8_t((vals[i] ^ 90));";
      "int(uint8_t(vals[i]))";
      "int(int8_t(int(uint8_t(vals[i]))))";
    ] ;
  (* No cooperative-matrix requirement for a plain 8-bit buffer. *)
  check_absent "GLSL rw" rw "GL_KHR_cooperative_matrix" ;
  List.iter
    (check_has "GLSL shared" sh)
    [
      "shared uint8_t su[64]";
      "shared int8_t ss[64]";
      "su[t] = uint8_t(vals[g]);";
      "int(su[(63 - t)])";
    ]

let test_ptx_spelling () =
  let rw = Sarek_ir_ptx.generate (K.rw_kernel ()) in
  let sh = Sarek_ir_ptx.generate (K.shared_kernel ()) in
  List.iter
    (check_has "PTX rw" rw)
    [
      "ld.global.nc.u8 %r";
      "ld.global.nc.s8 %r";
      "st.global.u8 [";
      "and.b32 %r";
      "bfe.s32 %r";
    ] ;
  List.iter
    (check_has "PTX shared" sh)
    [
      ".shared .align 1 .b8";
      "st.shared.u8 [";
      "ld.shared.u8 %r";
      "ld.shared.s8 %r";
    ]

let expect_refusal what f needle =
  match f () with
  | (_ : string) -> Alcotest.failf "%s: generated instead of refusing" what
  | exception e ->
      let msg = Printexc.to_string e in
      if not (contains ~needle msg) then
        Alcotest.failf "%s: refused, but not with %S: %s" what needle msg

let test_wgsl_refuses () =
  List.iter
    (fun k ->
      expect_refusal
        ("WGSL " ^ k.kern_name)
        (fun () -> Sarek_ir_wgsl.generate k)
        "WebGPU has no 8-bit integer type")
    all_kernels

(** {1 Shared refusals} *)

let refused_kernels = Int8_refused_kernels.all

let test_backends_refuse_misuse () =
  List.iter
    (fun (k, needle) ->
      List.iter
        (fun (name, gen) ->
          expect_refusal (name ^ " " ^ k.kern_name) gen needle)
        [
          ("CUDA", fun () -> Sarek_ir_cuda.generate k);
          ("OpenCL", fun () -> Sarek_ir_opencl.generate k);
          ("Metal", fun () -> Sarek_ir_metal.generate k);
          ("GLSL", fun () -> Sarek_ir_glsl.generate k);
          ("PTX", fun () -> Sarek_ir_ptx.generate k);
        ])
    refused_kernels

let test_interpreter_refuses_misuse () =
  List.iter
    (fun (k, needle) ->
      (* Straight to the evaluator: the refusal is its first act, before any
         argument is bound, so no arguments are needed. *)
      match
        Sarek.Sarek_ir_interp.run_kernel
          k
          ~block:(K.block, 1, 1)
          ~grid:(1, 1, 1)
          []
      with
      | () -> Alcotest.failf "interpreter ran %s" k.kern_name
      | exception e ->
          let msg = Printexc.to_string e in
          if not (contains ~needle msg) then
            Alcotest.failf
              "interpreter refused %s, but not with %S: %s"
              k.kern_name
              needle
              msg)
    refused_kernels

(** {1 Capability} *)

let test_capability () =
  let k = K.rw_kernel () in
  Alcotest.(check bool)
    "an 8-bit kernel requires Int8"
    true
    (List.mem Sarek_ir_analysis.Int8 (Sarek_ir_analysis.kernel_requirements k)) ;
  Alcotest.(check bool)
    "and not a cooperative matrix"
    false
    (Sarek_ir_analysis.kernel_uses Sarek_ir_analysis.Coopmat k) ;
  (match
     Sarek_capability.device_verdict
       ~provided:(Some [Sarek_ir_analysis.Float64])
       Sarek_ir_analysis.Int8
   with
  | Sarek_capability.Unavailable cap ->
      check_has "refusal" cap.Sarek_capability.cap_why "storageBuffer8BitAccess"
  | Sarek_capability.Available | Sarek_capability.Unknown _ ->
      Alcotest.fail "a device without Int8 was not refused") ;
  Alcotest.(check bool)
    "a device with Int8 is permitted"
    true
    (Sarek_capability.permits
       (Sarek_capability.device_verdict
          ~provided:(Some [Sarek_ir_analysis.Int8])
          Sarek_ir_analysis.Int8))

(** An integer cooperative-matrix kernel with u8 operand buffers needs both
    features: the matrix operation and the 8-bit storage of its operands. *)
let test_coopmat_kernel_needs_both () =
  let cfg =
    {
      Sarek_coopmat.cfg_shape = {Sarek_coopmat.m = 16; n = 16; k = 16};
      cfg_a = Sarek_coopmat.Uint8;
      cfg_b = Sarek_coopmat.Uint8;
      cfg_c = Sarek_coopmat.Sint32;
      cfg_result = Sarek_coopmat.Sint32;
      cfg_saturating = false;
      cfg_scope = Sarek_coopmat.Subgroup;
    }
  in
  let k =
    Int8_refused_kernels.one_u8_buffer
      "coopmat_u8"
      (SCoopmat (CM_muladd {dst = "d"; a = "a"; b = "b"; c = "c"; cfg}))
  in
  Alcotest.(check (list string))
    "requirements"
    ["int8-storage"; "cooperative-matrix"]
    (List.map
       Sarek_ir_analysis.feature_name
       (Sarek_ir_analysis.kernel_requirements k))

(** {1 s8 cooperative-matrix fragments from an int8 buffer} *)

let run_coopmat_interp ~buf_elt =
  let operand a =
    match buf_elt with
    | TInt8 -> Execute.Vec (vec_s8 a)
    | _ -> Execute.Vec (vec_u8 (Array.map K.u8 a))
  in
  let d = vec_i32 (Array.make (K.mnk * K.mnk) 0l) in
  Execute.run_interpreter_vectors
    ~ir:(K.coopmat_s8_kernel ~buf_elt)
    ~args:
      [
        operand K.coop_a;
        operand K.coop_b;
        Execute.Vec (vec_i32 K.coop_c);
        Execute.Vec d;
      ]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    ~parallel:false ;
  Vector.to_array d

let test_coopmat_int8_buffer_interp () =
  let model = K.coopmat_s8_model () in
  check_i32_array "int8 buffers" model (run_coopmat_interp ~buf_elt:TInt8) ;
  check_i32_array "uint8 buffers" model (run_coopmat_interp ~buf_elt:TUint8)

let test_coopmat_int8_buffer_codegen () =
  let k8 = K.coopmat_s8_kernel ~buf_elt:TInt8
  and ku = K.coopmat_s8_kernel ~buf_elt:TUint8 in
  let c8 = Sarek_ir_cuda.generate k8 and cu = Sarek_ir_cuda.generate ku in
  (* HIP's load helper takes const unsigned char *: the int8 buffer is cast,
     the uint8 one passed as before. *)
  check_has "CUDA/HIP int8" c8 "(const unsigned char *)(a)" ;
  check_absent "CUDA/HIP uint8" cu "(const unsigned char *)(a)" ;
  ignore (Sarek_ir_ptx.generate ~sm_target:"sm_86" k8 : string) ;
  let g8 = Sarek_ir_glsl.generate ~block:(64, 1, 1) k8 in
  check_has "GLSL int8" g8 "int8_t a[]" ;
  check_has "GLSL int8" g8 "coopmat<int8_t"

(** Kernel fusion substitutes a producer's stored expression for the consumer's
    read, which would drop the narrowing of an 8-bit store. *)
let fusion_pair elt =
  let inter = K.var 0 "inter" (TVec elt)
  and src = K.var 1 "src" (TVec TInt32)
  and out = K.var 2 "out" (TVec TInt32)
  and i = K.var 3 "i" TInt32 in
  let iv = EVar i in
  let producer =
    {
      default_kernel with
      kern_name = "p";
      kern_params = [K.vec_param src TInt32; K.vec_param inter elt];
      kern_body = SLet (i, K.gid, K.store "inter" iv (K.read "src" iv));
    }
  in
  let consumer =
    {
      default_kernel with
      kern_name = "c";
      kern_params = [K.vec_param inter elt; K.vec_param out TInt32];
      kern_body = SLet (i, K.gid, K.store "out" iv (K.read "inter" iv));
    }
  in
  (producer, consumer)

(** Both the element-wise and the stencil fusion admit the int32 pair (the
    control) and decline the same pair through a uint8 intermediate. *)
let test_fusion_declines () =
  let check name fuse =
    let p32, c32 = fusion_pair TInt32 and p8, c8 = fusion_pair TUint8 in
    Alcotest.(check bool) (name ^ ": int32 control fuses") true (fuse p32 c32) ;
    Alcotest.(check bool) (name ^ ": uint8 declined") false (fuse p8 c8)
  in
  check "can_fuse" (fun p c -> Sarek.Sarek_fusion.can_fuse p c "inter") ;
  check "can_fuse_stencil" (fun p c ->
      Sarek.Sarek_fusion.can_fuse_stencil p c "inter")

(** Lowering an 8-bit local matches its binder by id AND name: an int32 local
    with the same id (a pass may number a fresh temporary from 0) keeps its
    value. Checked on the emitted GLSL; the interpreter keys variables by id and
    so requires distinct ids anyway. *)
let test_lowering_matches_binder () =
  let x = K.var 0 "x" TUint8
  and y = K.var 0 "y" TInt32
  and out = K.var 1 "out" (TVec TInt32) in
  let k =
    {
      default_kernel with
      kern_name = "int8_same_id";
      kern_params = [K.vec_param out TInt32];
      kern_body =
        SLet
          ( x,
            K.i32 300,
            SLet
              ( y,
                K.i32 1000,
                SSeq
                  [
                    K.store "out" (K.i32 0) (EVar x);
                    K.store "out" (K.i32 1) (EVar y);
                  ] ) );
    }
  in
  let glsl = Sarek_ir_glsl.generate ~block:(1, 1, 1) k in
  check_has "GLSL" glsl "int x = int(uint8_t(300));" ;
  check_has "GLSL" glsl "int y = 1000;"

let () =
  Alcotest.run
    "int8_storage"
    [
      ( "interpreter",
        [
          Alcotest.test_case
            "inputs cover the edges"
            `Quick
            test_inputs_cover_the_edges;
          Alcotest.test_case "global reads, stores, casts" `Quick test_interp_rw;
          Alcotest.test_case "shared arrays" `Quick test_interp_shared;
          Alcotest.test_case "local scalars" `Quick test_interp_scalar;
          Alcotest.test_case
            "a char vector is refused for an int8 parameter"
            `Quick
            test_char_vector_refused_for_int8;
          Alcotest.test_case
            "refuses 8-bit misuse"
            `Quick
            test_interpreter_refuses_misuse;
        ] );
      ( "codegen",
        [
          Alcotest.test_case "CUDA/HIP spelling" `Quick test_cuda_spelling;
          Alcotest.test_case "OpenCL spelling" `Quick test_opencl_spelling;
          Alcotest.test_case "Metal spelling" `Quick test_metal_spelling;
          Alcotest.test_case "GLSL spelling" `Quick test_glsl_spelling;
          Alcotest.test_case "PTX spelling" `Quick test_ptx_spelling;
          Alcotest.test_case "WGSL refuses" `Quick test_wgsl_refuses;
          Alcotest.test_case
            "every backend refuses 8-bit misuse"
            `Quick
            test_backends_refuse_misuse;
        ] );
      ( "capability",
        [
          Alcotest.test_case "Int8 gate" `Quick test_capability;
          Alcotest.test_case "fusion declines" `Quick test_fusion_declines;
          Alcotest.test_case
            "lowering matches the binder, not only its id"
            `Quick
            test_lowering_matches_binder;
          Alcotest.test_case
            "s8 coopmat from an int8 buffer (interpreter)"
            `Quick
            test_coopmat_int8_buffer_interp;
          Alcotest.test_case
            "s8 coopmat from an int8 buffer (codegen)"
            `Quick
            test_coopmat_int8_buffer_codegen;
          Alcotest.test_case
            "a u8 coopmat kernel needs Int8 and Coopmat"
            `Quick
            test_coopmat_kernel_needs_both;
        ] );
    ]
