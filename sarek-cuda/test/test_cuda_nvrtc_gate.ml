(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** NVRTC compile gate for the CUDA-C backend.

    The CUDA-C emitter had no executable validation anywhere: the golden
    snapshots in [sarek/tests/codegen_golden] pin the emitted bytes, and
    [sarek/tests/unit/test_ptx_snapshot.ml] assembles the *PTX* backend with
    [ptxas], but nothing ever fed the emitted CUDA *C* to a compiler. The
    consequence is a whole class of "generated source is byte-exact and
    completely uncompilable" bugs that only a reviewer pasting the output into a
    real compiler would catch — the historical example being an emitted
    [#include <cuda_fp16.h>] that NVRTC could not resolve, because NVRTC has no
    filesystem include path unless one is passed explicitly.

    This gate closes that hole with the same NVRTC entry point the runtime
    actually uses ([Cuda_nvrtc.compile_to_ptx]), so the gate and production
    share the include/option behaviour. Two properties matter:

    - NVRTC is a {b host-side} compiler: [nvrtcCompileProgram] needs no CUDA
      device, no driver and no [/dev/nvidia*]. This gate therefore runs on a
      CPU-only CI machine, which is why it is worth having in CI at all.
    - It skips cleanly when [libnvrtc] cannot be dlopen'd, because a developer
      machine without a CUDA toolkit is a legitimate configuration. The CI-side
      guarantee that the skip does not silently return is
      [ci/assert-toolchain.sh], which fails the build when [libnvrtc] is missing
      from the image. *)

open Sarek_cuda
open Sarek_ir_types

let make_var ?(mut = false) name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = mut}

let base_kernel name params body =
  {default_kernel with kern_name = name; kern_params = params; kern_body = body}

let vec name ty = DParam (make_var name (TVec ty), None)

let scalar name ty = DParam (make_var name ty, None)

(** [tid] bound from [thread_idx_x], then [body tid]. *)
let with_tid body =
  let tid = make_var "tid" TInt32 in
  SLet (tid, EIntrinsic ([], "thread_idx_x", []), body tid)

(** {1 Corpus}

    Deliberately spans the axes where an emitter can produce well-formed-looking
    but uncompilable C: element types (i32/i64/f32/f64), intrinsic lowering
    (native vs software helper families), shared memory, control flow and
    locally-declared arrays. Each entry is a whole kernel module, so the
    generated preamble (typedefs, helper functions, [#include] lines) is
    compiled too — that preamble is where the historical f16 include bug lived,
    and no per-expression test would have reached it. *)
let corpus () =
  let f32_binop op name =
    ( name,
      base_kernel
        name
        [vec "a" TFloat32; vec "b" TFloat32; vec "out" TFloat32]
        (with_tid (fun tid ->
             SAssign
               ( LArrayElem ("out", EVar tid),
                 EBinop
                   (op, EArrayRead ("a", EVar tid), EArrayRead ("b", EVar tid))
               ))) )
  in
  (* [path] is the intrinsic's module qualifier, e.g. ["Float32"]. The
     dispatcher keys on it (see Sarek_ir_intrinsic_dispatch), so an unqualified
     name only resolves for the framework-wide intrinsics. *)
  let unary_intrinsic ty tyname path fn =
    let name = Printf.sprintf "%s_%s" tyname fn in
    ( name,
      base_kernel
        name
        [vec "a" ty; vec "out" ty]
        (with_tid (fun tid ->
             SAssign
               ( LArrayElem ("out", EVar tid),
                 EIntrinsic (path, fn, [EArrayRead ("a", EVar tid)]) ))) )
  in
  let binary_intrinsic ty tyname path fn =
    let name = Printf.sprintf "%s_%s" tyname fn in
    ( name,
      base_kernel
        name
        [vec "a" ty; vec "b" ty; vec "out" ty]
        (with_tid (fun tid ->
             SAssign
               ( LArrayElem ("out", EVar tid),
                 EIntrinsic
                   ( path,
                     fn,
                     [EArrayRead ("a", EVar tid); EArrayRead ("b", EVar tid)] )
               ))) )
  in
  [
    f32_binop Add "f32_add";
    f32_binop Mul "f32_mul";
    f32_binop Div "f32_div";
    (* Integer division/remainder: the signed-division lowering is the exact
       shape the ptxas gate covers on the PTX side. *)
    ( "i32_div_rem",
      base_kernel
        "i32_div_rem"
        [vec "a" TInt32; vec "b" TInt32; vec "out" TInt32]
        (with_tid (fun tid ->
             SAssign
               ( LArrayElem ("out", EVar tid),
                 EBinop
                   ( Add,
                     EBinop
                       ( Div,
                         EArrayRead ("a", EVar tid),
                         EArrayRead ("b", EVar tid) ),
                     EBinop
                       ( Mod,
                         EArrayRead ("a", EVar tid),
                         EArrayRead ("b", EVar tid) ) ) ))) );
    ( "i64_compare",
      base_kernel
        "i64_compare"
        [vec "a" TInt64; vec "out" TInt32]
        (with_tid (fun tid ->
             SIf
               ( EBinop (Lt, EArrayRead ("a", EVar tid), EConst (CInt64 7L)),
                 SAssign (LArrayElem ("out", EVar tid), EConst (CInt32 1l)),
                 Some
                   (SAssign (LArrayElem ("out", EVar tid), EConst (CInt32 0l)))
               ))) );
    (* f64 transcendentals: these are the ones that go through the software
       helper family on shader targets, so the CUDA path emits a different
       preamble and is worth compiling. *)
    unary_intrinsic TFloat64 "f64" ["Float64"] "sqrt";
    unary_intrinsic TFloat64 "f64" ["Float64"] "sin";
    unary_intrinsic TFloat64 "f64" ["Float64"] "exp";
    unary_intrinsic TFloat64 "f64" ["Float64"] "log";
    unary_intrinsic TFloat32 "f32" ["Float32"] "sqrt";
    unary_intrinsic TFloat32 "f32" ["Float32"] "rsqrt";
    unary_intrinsic TFloat32 "f32" ["Float32"] "abs_float";
    binary_intrinsic TFloat32 "f32" ["Float32"] "fmod";
    binary_intrinsic TFloat32 "f32" ["Float32"] "copysign";
    binary_intrinsic TFloat64 "f64" ["Float64"] "fmod";
    (* Shared memory + barrier: exercises the __shared__ declaration path. *)
    ( "shared_reduce",
      base_kernel
        "shared_reduce"
        [vec "inp" TFloat32; vec "out" TFloat32]
        (with_tid (fun tid ->
             let sdata = make_var "sdata" (TArray (TFloat32, Shared)) in
             SLet
               ( sdata,
                 EArrayCreate (TFloat32, EConst (CInt32 256l), Shared),
                 SSeq
                   [
                     SAssign
                       ( LArrayElem ("sdata", EVar tid),
                         EArrayRead ("inp", EVar tid) );
                     SBarrier;
                     SAssign
                       ( LArrayElem ("out", EVar tid),
                         EArrayRead ("sdata", EVar tid) );
                   ] ))) );
    (* Mutable local + while loop: the shape tail-recursion elimination
       produces. *)
    ( "while_accumulate",
      base_kernel
        "while_accumulate"
        [vec "data" TFloat32; vec "out" TFloat32; scalar "n" TInt32]
        (with_tid (fun tid ->
             let i = make_var ~mut:true "i" TInt32 in
             let acc = make_var ~mut:true "acc" TFloat32 in
             SLet
               ( i,
                 EConst (CInt32 0l),
                 SLet
                   ( acc,
                     EConst (CFloat32 0.0),
                     SSeq
                       [
                         SWhile
                           ( EBinop (Lt, EVar i, EVar (make_var "n" TInt32)),
                             SSeq
                               [
                                 SAssign
                                   ( LVar acc,
                                     EBinop
                                       ( Add,
                                         EVar acc,
                                         EArrayRead ("data", EVar i) ) );
                                 SAssign
                                   ( LVar i,
                                     EBinop (Add, EVar i, EConst (CInt32 1l)) );
                               ] );
                         SAssign (LArrayElem ("out", EVar tid), EVar acc);
                       ] ) ))) );
  ]

(** {1 Gate} *)

let nvrtc_available = lazy (Cuda_nvrtc.is_available ())

(* compute_75 is the floor Sarek's own NVRTC path falls back through, and needs
   no device to target. *)
let arch = "compute_75"

(** [true] iff [needle] occurs anywhere in [hay]. *)
let contains hay needle =
  let n = String.length needle and h = String.length hay in
  n <= h
  &&
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  go 0

let compile_ok ~kernel_name name cuda =
  match Cuda_nvrtc.compile_to_ptx ~name ~arch cuda with
  | ptx ->
      (* NVRTC can return successfully with PTX that contains no kernel at all —
         e.g. if the emitter dropped the __global__ qualifier, or emitted the
         body into a device function nobody calls. Assert the entry point is
         really there, keyed on the kernel's own name, so a compiling-but-empty
         module cannot pass. A length threshold cannot do this: NVRTC's PTX
         header alone clears any plausible bound. *)
      let entry = ".entry " ^ kernel_name in
      if not (contains ptx entry) then
        Alcotest.failf
          "NVRTC compiled %s but the PTX has no '%s' — the module carries no \
           kernel:\n\
           %s"
          kernel_name
          entry
          ptx ;
      Ok ()
  | exception e -> Error (Printexc.to_string e)

let gate_tests () =
  List.map
    (fun (name, k) ->
      Alcotest.test_case (Printf.sprintf "nvrtc/%s" name) `Quick (fun () ->
          let cuda = Sarek_ir_cuda.generate_with_types ~types:k.kern_types k in
          (* Generation is unconditional: with no toolkit we still exercise the
             emitter and would catch a codegen exception. Only the compile step
             is conditional. *)
          if String.length cuda = 0 then
            Alcotest.failf "CUDA-C generation produced empty source for %s" name ;
          if not (Lazy.force nvrtc_available) then
            Printf.printf
              "  SKIP: libnvrtc not loadable (no CUDA toolkit) — %s generated \
               only\n\
               %!"
              name
          else
            match compile_ok ~kernel_name:k.kern_name name cuda with
            | Ok () -> Printf.printf "  nvrtc OK: %s\n%!" name
            | Error e ->
                Alcotest.failf
                  "NVRTC rejected generated CUDA-C for %s:\n\
                   %s\n\
                   --- source ---\n\
                   %s"
                  name
                  e
                  cuda))
    (corpus ())

let () = Alcotest.run "cuda_nvrtc_gate" [("nvrtc-compile-gate", gate_tests ())]
