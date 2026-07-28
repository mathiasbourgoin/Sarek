(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A backend's entry points must agree (backlog-155).
 *
 * WHAT THIS IS FOR
 *
 * Each of the five source-emitting backends exposes two entry points that
 * SHOULD produce the same text for the same kernel:
 *
 *   generate k                             (* the plain one *)
 *   generate_with_types ~types:k.kern_types k
 *
 * They should agree because [~types] has exactly the type of the [kern_types]
 * FIELD of the kernel it travels with ([Sarek_ir_types.kernel]), and every
 * production caller only ever passed that field. The parameter was redundant
 * with the record.
 *
 * They did NOT agree. [generate] was a separate 34-84 line copy of the emit
 * sequence in each backend, and each copy omitted the type-declaration step:
 * no record typedefs, no variant typedefs, and (on four of the five) no
 * [current_variants] assignment either. A kernel with a record type emitted
 * through [generate] therefore produced source that USES the struct and never
 * DECLARES it — the C-family backends emit `Point2 p = pts[idx];` against no
 * `typedef struct ... Point2`. That is not a diagnostic; it is source the device
 * compiler rejects, or worse on WGSL, where [generate] set [current_variants]
 * (arming SMatch payload extraction) for types it then never declared.
 *
 * The public transpiler ([Sarek_transpile.emit_backend]) routes ALL FIVE
 * backends through the plain [generate], so `sarek-transpile` on any kernel
 * carrying a record or a variant emitted undeclared-type source — on every
 * backend, for every user.
 *
 * HOW IT CHECKS, AND WHY THIS WAY
 *
 * Byte equality of the two outputs. Not a substring probe for `typedef`, not a
 * singleton-identifier diff: one entry point is DEFINED as a special case of the
 * other, so their outputs must be identical, and that is both the strongest
 * available assertion and the cheapest. It needs no per-backend list of language
 * builtins and no external validator, and it cannot be satisfied by a partial
 * fix — emitting the record typedef but not the variant typedef still differs.
 *
 * The kernels below carry a record AND a variant, because the omission was
 * per-step: a backend could restore one and not the other. The scalar kernel is
 * the control — it carries neither, the two entry points agreed on it even
 * before the fix, so a red there means this test broke rather than the backends.
 *
 * WHAT IT DOES NOT COVER
 *
 * Agreement, not correctness. If both entry points emit the same WRONG typedef,
 * this is green — the golden suite and the device-compiler gates own that.
 * PTX is absent: [Sarek_ir_ptx_kernel.generate_with_types] takes [~types:_] and
 * ignores it (types reach it through [Sarek_ir_ptx_types] instead), so the two
 * entry points are equal there by construction and the check would be vacuous.
 * Stating that is the point: a green PTX row would have asserted nothing.
 *
 * Run with: dune exec sarek/tests/unit/test_entry_point_agreement.exe
 ******************************************************************************)

open Sarek_ir_types
module Wgsl = Sarek_codegen.Sarek_ir_wgsl
module Metal = Sarek_codegen.Sarek_ir_metal
module Cuda = Sarek_codegen.Sarek_ir_cuda
module Opencl = Sarek_codegen.Sarek_ir_opencl
module Glsl = Sarek_codegen.Sarek_ir_glsl

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let empty_kernel name params locals body =
  {
    kern_name = name;
    kern_params = params;
    kern_locals = locals;
    kern_body = body;
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(* ------------------------------------------------------------------------ *)
(* Kernels                                                                   *)

(** Carries a record type: reads a Point2, doubles both fields, writes it back.
    [kern_types] is non-empty, which is the whole point — it is the input the
    plain entry point used to drop. *)
let record_kernel () =
  let fields = [("x", TFloat32); ("y", TFloat32)] in
  let point = TRecord ("Point2", fields) in
  let pts = make_var "pts" (TVec point) in
  let idx = make_var "idx" TInt32 in
  let p = make_var "p" point in
  let scaled f =
    EBinop (Mul, ERecordField (EVar p, f), EConst (CFloat32 2.0))
  in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SLet
          ( p,
            EArrayRead ("pts", EVar idx),
            SAssign
              ( LArrayElem ("pts", EVar idx),
                ERecord ("Point2", [("x", scaled "x"); ("y", scaled "y")]) ) )
      )
  in
  let k = empty_kernel "record_kernel" [DParam (pts, None)] [] body in
  {k with kern_types = [("Point2", fields)]}

(** Carries a variant type. Separate from the record kernel because the omission
    was per-step: restoring record typedefs alone would leave this red. *)
let variant_kernel () =
  let constrs = [("OptNone", []); ("OptSome", [TFloat32])] in
  let opt = TVariant ("Opt", constrs) in
  let out = make_var "out" (TVec opt) in
  let idx = make_var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            EVariant ("Opt", "OptSome", [EConst (CFloat32 1.0)]) ) )
  in
  let k = empty_kernel "variant_kernel" [DParam (out, None)] [] body in
  {k with kern_variants = [("Opt", constrs)]}

(** Control: neither a record nor a variant, so the two entry points agreed even
    before the fix. A red here indicts the test, not the backends. *)
let scalar_kernel () =
  let a = make_var "a" (TVec TFloat32) in
  let c = make_var "c" (TVec TFloat32) in
  let arr v =
    DParam (v, Some {arr_elttype = TFloat32; arr_memspace = Global})
  in
  empty_kernel
    "scalar_kernel"
    [arr a; arr c]
    []
    (SAssign
       ( LArrayElem ("c", EConst (CInt32 0l)),
         EBinop
           (Mul, EArrayRead ("a", EConst (CInt32 0l)), EConst (CFloat32 2.0)) ))

(* ------------------------------------------------------------------------ *)
(* The matrix                                                                *)

type backend = {
  label : string;
  plain : kernel -> string;
  with_types : kernel -> string;
}

let backends =
  [
    {
      label = "CUDA";
      plain = (fun k -> Cuda.generate k);
      with_types = (fun k -> Cuda.generate_with_types ~types:k.kern_types k);
    };
    {
      label = "OpenCL";
      plain = (fun k -> Opencl.generate k);
      with_types = (fun k -> Opencl.generate_with_types ~types:k.kern_types k);
    };
    {
      label = "Metal";
      plain = (fun k -> Metal.generate k);
      with_types = (fun k -> Metal.generate_with_types ~types:k.kern_types k);
    };
    {
      label = "WGSL";
      plain = (fun k -> Wgsl.generate k);
      with_types = (fun k -> Wgsl.generate_with_types ~types:k.kern_types k);
    };
    {
      label = "GLSL";
      plain = (fun k -> Glsl.generate k);
      with_types = (fun k -> Glsl.generate_with_types ~types:k.kern_types k);
    };
  ]

let kernels =
  [
    ("record", record_kernel);
    ("variant", variant_kernel);
    ("scalar (control)", scalar_kernel);
  ]

(* A backend may legitimately REFUSE a kernel (Metal has no f64, GLSL no f16).
   A refusal is agreement as long as both entry points refuse identically, so
   compare the outcome rather than only the success path — otherwise a backend
   that raises on both sides would silently drop out of the matrix. *)
let outcome f k = try Ok (f k) with e -> Error (Printexc.to_string e)

let check b (kname, mk) () =
  let k = mk () in
  match (outcome b.plain k, outcome b.with_types k) with
  | Ok plain, Ok with_types ->
      if plain <> with_types then begin
        (* Report the first differing line: the whole shader is unreadable in an
           assertion message, and the interesting delta is a missing
           declaration near the top. *)
        let split s = String.split_on_char '\n' s in
        let pl = split plain and wl = split with_types in
        let rec first_diff i a b =
          match (a, b) with
          | x :: xs, y :: ys when x = y -> first_diff (i + 1) xs ys
          | x :: _, y :: _ -> Printf.sprintf "line %d: %S vs %S" i x y
          | [], y :: _ -> Printf.sprintf "line %d: <absent> vs %S" i y
          | x :: _, [] -> Printf.sprintf "line %d: %S vs <absent>" i x
          | [], [] -> "identical line-wise but not byte-wise"
        in
        Alcotest.failf
          "%s/%s: generate and generate_with_types ~types:k.kern_types \
           disagree (%d vs %d bytes) — %s"
          b.label
          kname
          (String.length plain)
          (String.length with_types)
          (first_diff 1 pl wl)
      end
  | Error e1, Error e2 ->
      (* Both refused: agreement, as long as it is the SAME refusal. *)
      Alcotest.check Alcotest.string (b.label ^ "/" ^ kname ^ " refusal") e1 e2
  | Ok _, Error e ->
      Alcotest.failf
        "%s/%s: generate succeeded but generate_with_types raised %s"
        b.label
        kname
        e
  | Error e, Ok _ ->
      Alcotest.failf
        "%s/%s: generate raised %s but generate_with_types succeeded"
        b.label
        kname
        e

let () =
  Alcotest.run
    "entry-point agreement"
    [
      ( "generate = generate_with_types ~types:k.kern_types",
        List.concat_map
          (fun b ->
            List.map
              (fun (kname, mk) ->
                Alcotest.test_case
                  (b.label ^ ": " ^ kname)
                  `Quick
                  (check b (kname, mk)))
              kernels)
          backends );
    ]
