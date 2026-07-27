(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CLASS VALIDATOR for silent width changes on the BACKEND side of the type
 * mapping surface.
 *
 * sarek/tests/unit/test_type_width_totality.ml is the same validator for the
 * FRONT half of the pipeline: source type -> IR element type. It stops at the
 * IR. Nothing held the second half — IR element type -> device type string —
 * to the same rule, and that is exactly where #141 lived:
 *
 *   Sarek_ir_metal.metal_type_of_elttype mapped [TFloat64] to "float", with a
 *   comment saying Metal has no double and no refusal anywhere. A float64
 *   kernel compiled, ran, and read an 8-byte-per-element host buffer at a
 *   4-byte stride.
 *
 * The invariant enforced here is the one that arm broke:
 *
 *   for every backend B and every scalar IR element type T,
 *     either B's type mapper RAISES a diagnostic for T,
 *     or     the device type it names occupies exactly
 *            [Sarek_ir_layout.scalar_size T] bytes,
 *     or     that device type has no memory form at all on B (and B's own
 *            validator refuses it in a buffer — see [No_memory_form]).
 *
 * [Sarek_ir_layout.scalar_size] is the right denominator and not a restatement:
 * it IS the width the host marshaller uses (its own comment pins it to
 * Sarek_ppx's field-size mapping), so a device type that disagrees with it
 * disagrees with the bytes the host actually wrote.
 *
 * WHY THIS GATE CANNOT SILENTLY PASS (see the "gates that cannot fail" list):
 *
 *  - the element-type enumeration is not a hand-written list that can go
 *    stale: it is unfolded from a total successor chain ([next_elt]) whose
 *    match has no wildcard, so a new constructor on [Sarek_ir_types.elttype]
 *    fails to COMPILE here (warning 8 is an error in this test's flags);
 *  - the backend table is likewise checked against a pinned count, so deleting
 *    a backend from the sweep is a failure rather than a smaller sweep;
 *  - a device type name this test has never seen is a FAILURE, not a skip: the
 *    per-backend width tables are closed, so a backend that starts emitting a
 *    new spelling must have its width recorded before the sweep goes green;
 *  - [No_memory_form] is the only escape hatch, and its full set is pinned by
 *    [test_no_memory_form_set_is_exactly_as_recorded]: adding one is a
 *    deliberate edit to an expected list, not a quiet widening;
 *  - each backend must exercise at least one NON-refusing, memory-form mapping
 *    ([test_every_backend_checks_something]), so a backend that starts
 *    refusing everything cannot pass this file vacuously;
 *  - a refusal only counts if it is a DIAGNOSTIC. [Match_failure],
 *    [Not_found], [Invalid_argument] and [Failure] are internal errors, not
 *    refusals, and are rejected as such.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

(* ------------------------------------------------------------------ *)
(* Compiler-enforced enumeration of the scalar IR element types.       *)
(*                                                                     *)
(* Total match, NO wildcard: adding a constructor to                   *)
(* [Sarek_ir_types.elttype] fails to compile here. The aggregate arms  *)
(* end the chain because they have no scalar width (they are laid out  *)
(* field-by-field from the scalars below, which is what this sweep     *)
(* pins).                                                              *)
(* ------------------------------------------------------------------ *)

let next_elt : elttype -> elttype option = function
  | TInt32 -> Some TInt64
  | TInt64 -> Some TFloat16
  | TFloat16 -> Some TFloat32
  | TFloat32 -> Some TFloat64
  | TFloat64 -> Some TBool
  | TBool -> Some TUnit
  | TUnit -> None
  | TRecord _ | TVariant _ | TArray _ | TVec _ -> None

(* NB the accumulator carries [x], not [y]. The version of this helper in
   sarek/tests/unit/test_type_width_totality.ml pushed [y] and so dropped the
   FIRST element of every chain while duplicating the last — its length check
   still read 7, because one duplicate exactly replaced one omission. Fixed
   there too in this change; see that file's [unfold]. *)
let unfold next first =
  let rec go acc x =
    match next x with Some y -> go (x :: acc) y | None -> List.rev (x :: acc)
  in
  go [] first

let all_scalar_elts = unfold next_elt TInt32

let name_of_elt = function
  | TInt32 -> "TInt32"
  | TInt64 -> "TInt64"
  | TFloat16 -> "TFloat16"
  | TFloat32 -> "TFloat32"
  | TFloat64 -> "TFloat64"
  | TBool -> "TBool"
  | TUnit -> "TUnit"
  | TRecord (n, _) -> "TRecord " ^ n
  | TVariant (n, _) -> "TVariant " ^ n
  | TArray _ -> "TArray"
  | TVec _ -> "TVec"

(* ------------------------------------------------------------------ *)
(* Device-side widths                                                  *)
(* ------------------------------------------------------------------ *)

(** What a device type name occupies in a buffer on a given target.

    [No_memory_form reason] is for a spelling that cannot appear in a memory
    slot at ALL — C's [void], WGSL's [bool]. It is not "we do not know": each
    one carries the evidence that the target's OWN validator refuses it there,
    so the failure mode is loud rather than a wrong stride. The complete set is
    pinned below. *)
type device_width = Bytes of int | No_memory_form of string

(** Closed per-backend tables. [None] means "this test has never seen that
    spelling", which the sweep reports as a failure — a backend that starts
    emitting a new device type must record its width here first. That is what
    keeps the sweep from being outrun by the code it checks. *)

(* C-family scalar spellings shared by CUDA and OpenCL. Both target 64-bit
   hosts through their own C dialects; OpenCL C fixes `long` at exactly 64 bits
   (OpenCL C spec, table of built-in scalar data types) and CUDA reaches 64 bits
   through `long long`. *)
let c_family_width = function
  | "int" -> Some (Bytes 4)
  | "long" -> Some (Bytes 8)
  | "long long" -> Some (Bytes 8)
  | "__half" -> Some (Bytes 2)
  | "float" -> Some (Bytes 4)
  | "double" -> Some (Bytes 8)
  | "void" -> Some (No_memory_form "C `void` has no object representation")
  | _ -> None

(* Metal Shading Language. Same spellings as C above EXCEPT `bool`, which MSL
   fixes at ONE byte (MSL spec, size and alignment of scalar data types) where
   the host uses a 4-byte slot — so `bool` must never be what this backend
   emits for [TBool], and is deliberately absent from this table: if it comes
   back, the sweep reports an unrecorded spelling. *)
let metal_width = function
  | "int" -> Some (Bytes 4)
  | "long" -> Some (Bytes 8)
  | "half" -> Some (Bytes 2)
  | "float" -> Some (Bytes 4)
  | "void" -> Some (No_memory_form "MSL `void` has no object representation")
  | _ -> None

(* GLSL. `bool` is 4 bytes here and that is MEASURED, not assumed: glslang
   lowers a bool member of an std430 storage buffer to a 32-bit uint. Verified
   on the emitted bool-record shader with glslangValidator -V --target-env
   vulkan1.2 (exit 0) and spirv-dis:

     %Flagged_0 = OpTypeStruct %uint %int
     OpMemberDecorate %Flagged_0 0 Offset 0
     OpMemberDecorate %Flagged_0 1 Offset 4
     OpDecorate %_runtimearr_Flagged_0 ArrayStride 8

   i.e. the same 4-byte slot the host writes. *)
let glsl_width = function
  | "int" -> Some (Bytes 4)
  | "int64_t" -> Some (Bytes 8)
  | "float16_t" -> Some (Bytes 2)
  | "float" -> Some (Bytes 4)
  | "double" -> Some (Bytes 8)
  | "bool" -> Some (Bytes 4)
  | "void" -> Some (No_memory_form "GLSL `void` cannot be a buffer member")
  | _ -> None

(* WGSL. `bool` genuinely has no memory form: it is not host-shareable, and
   naga refuses it in a storage binding rather than picking a width. Verified on
   the emitted bool-record shader:

     error: Global variable [0] 'pts' is invalid
     = Alignment requirements for address space Storage ... are not met by [0]
     = The type is not host-shareable

   so a bool that reaches a buffer is a hard validation failure at shader-load
   time, never a wrong stride. *)
let wgsl_width = function
  | "i32" -> Some (Bytes 4)
  | "f16" -> Some (Bytes 2)
  | "f32" -> Some (Bytes 4)
  | "bool" ->
      Some
        (No_memory_form
           "WGSL `bool` is not host-shareable; naga refuses it in a storage \
            binding")
  | "/* unit */" ->
      Some
        (No_memory_form "WGSL has no unit type; the emitter writes a comment")
  | _ -> None

(* PTX register classes. These are the widths ptxas gives each class. *)
let ptx_width = function
  | ".u32" -> Some (Bytes 4)
  | ".u64" -> Some (Bytes 8)
  | ".f32" -> Some (Bytes 4)
  | ".f64" -> Some (Bytes 8)
  | _ -> None

type backend = {
  bk_name : string;
  bk_map : elttype -> string;
  bk_width : string -> device_width option;
}

let backends =
  [
    {
      bk_name = "Metal";
      bk_map = Sarek_ir_metal.metal_type_of_elttype;
      bk_width = metal_width;
    };
    {
      bk_name = "CUDA";
      bk_map = Sarek_ir_cuda.cuda_type_of_elttype;
      bk_width = c_family_width;
    };
    {
      bk_name = "OpenCL";
      bk_map = Sarek_ir_opencl.opencl_type_of_elttype;
      bk_width = c_family_width;
    };
    {
      bk_name = "GLSL";
      bk_map = Sarek_ir_glsl.glsl_type_of_elttype;
      bk_width = glsl_width;
    };
    {
      bk_name = "WGSL";
      bk_map = Sarek_ir_wgsl.wgsl_type_of_elttype;
      bk_width = wgsl_width;
    };
    {
      bk_name = "PTX";
      bk_map = Sarek_ir_ptx_types.ptx_reg_type_of;
      bk_width = ptx_width;
    };
  ]

(* ------------------------------------------------------------------ *)
(* What counts as a refusal                                            *)
(* ------------------------------------------------------------------ *)

(** A refusal is admissible only if it is a DIAGNOSTIC. These four are how OCaml
    reports that the code fell off its own map, and accepting them as "the
    backend refused" would let an incomplete match masquerade as a policy. *)
let is_internal_error = function
  | Match_failure _ | Not_found | Invalid_argument _ | Failure _ -> true
  | _ -> false

type outcome = Refused of string | Emitted of string

let run_mapper b t =
  match b.bk_map t with
  | s -> Emitted s
  | exception e ->
      if is_internal_error e then
        Alcotest.failf
          "%s/%s: the mapper did not refuse, it CRASHED (%s). An internal \
           error is not a diagnostic — the caller learns nothing about why the \
           type is unavailable."
          b.bk_name
          (name_of_elt t)
          (Printexc.to_string e)
      else Refused (Printexc.to_string e)

(* ------------------------------------------------------------------ *)
(* Anti-vacuity                                                        *)
(* ------------------------------------------------------------------ *)

let test_enumeration_is_complete () =
  Alcotest.(check int)
    "scalar elttype constructors swept"
    7
    (List.length all_scalar_elts) ;
  Alcotest.(check int) "backends swept" 6 (List.length backends) ;
  (* A count alone does not prove the chain was walked correctly — see the
     [unfold] note above. Distinctness is what rules out an off-by-one walk that
     omits one constructor and duplicates another. *)
  let labels = List.map name_of_elt all_scalar_elts in
  if List.length (List.sort_uniq compare labels) <> List.length labels then
    Alcotest.failf
      "the element-type enumeration contains duplicates, so some type is going \
       unswept: %s"
      (String.concat ", " labels) ;
  let bnames = List.map (fun b -> b.bk_name) backends in
  if List.length (List.sort_uniq compare bnames) <> List.length bnames then
    Alcotest.failf
      "the backend table contains duplicates: %s"
      (String.concat ", " bnames)

(** A backend that refused every element type would satisfy the invariant while
    checking nothing. Each must have at least one type it maps to a real,
    width-checked memory form. *)
let test_every_backend_checks_something () =
  List.iter
    (fun b ->
      let n =
        List.length
          (List.filter
             (fun t ->
               match run_mapper b t with
               | Refused _ -> false
               | Emitted s -> (
                   match b.bk_width s with
                   | Some (Bytes _) -> true
                   | Some (No_memory_form _) | None -> false))
             all_scalar_elts)
      in
      if n = 0 then
        Alcotest.failf
          "%s maps NO scalar element type to a width-checked device type, so \
           every assertion about it in this file is vacuous"
          b.bk_name)
    backends

(** The [No_memory_form] escape hatch, pinned exactly. Widening it is how this
    validator would be defeated, so it is a list someone has to edit on purpose.
    Each entry is (backend, element type). *)
let expected_no_memory_form =
  [
    ("Metal", "TUnit");
    ("CUDA", "TUnit");
    ("OpenCL", "TUnit");
    ("GLSL", "TUnit");
    ("WGSL", "TBool");
    ("WGSL", "TUnit");
  ]

let test_no_memory_form_set_is_exactly_as_recorded () =
  let actual =
    List.concat_map
      (fun b ->
        List.filter_map
          (fun t ->
            match run_mapper b t with
            | Emitted s -> (
                match b.bk_width s with
                | Some (No_memory_form _) -> Some (b.bk_name, name_of_elt t)
                | _ -> None)
            | Refused _ -> None)
          all_scalar_elts)
      backends
  in
  let show l =
    String.concat
      ", "
      (List.map (fun (b, t) -> b ^ "/" ^ t) (List.sort compare l))
  in
  if List.sort compare actual <> List.sort compare expected_no_memory_form then
    Alcotest.failf
      "the set of device types claiming NO memory form changed.\n\
       expected: %s\n\
       actual:   %s\n\
       This is the one escape hatch in this validator: a type claiming no \
       memory form is exempt from the width check, so growing the set must be \
       a deliberate edit backed by evidence that the target's own validator \
       refuses that type in a buffer."
      (show expected_no_memory_form)
      (show actual)

(* ------------------------------------------------------------------ *)
(* THE INVARIANT                                                       *)
(* ------------------------------------------------------------------ *)

let test_no_backend_silently_changes_width () =
  List.iter
    (fun b ->
      List.iter
        (fun t ->
          let label = Printf.sprintf "%s/%s" b.bk_name (name_of_elt t) in
          match run_mapper b t with
          | Refused _ ->
              (* Loud is always admissible. *)
              ()
          | Emitted device_type -> (
              let host = Sarek_ir_layout.scalar_size t in
              match b.bk_width device_type with
              | None ->
                  Alcotest.failf
                    "%s emits the device type %S, whose byte width this \
                     validator has never been told. Record it in this file's \
                     %s table — until then nothing is checking that a %s value \
                     occupies the %d bytes the host wrote."
                    label
                    device_type
                    b.bk_name
                    (name_of_elt t)
                    host
              | Some (No_memory_form _) ->
                  (* Exempt, and the exemption set is pinned above. *)
                  ()
              | Some (Bytes w) ->
                  if w <> host then
                    Alcotest.failf
                      "%s maps to the device type %S, which is %d byte(s), but \
                       the host lays a %s out in %d byte(s) \
                       (Sarek_ir_layout.scalar_size). A kernel using it would \
                       compile, run, and stride the buffer wrong with no \
                       diagnostic. Either map it at the host's width or refuse \
                       it."
                      label
                      device_type
                      w
                      (name_of_elt t)
                      host))
        all_scalar_elts)
    backends

(* ------------------------------------------------------------------ *)
(* #141 point regressions — the captured red                           *)
(* ------------------------------------------------------------------ *)

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

(** The exact arm from #141. Pre-fix this returned "float". *)
let test_metal_refuses_f64_element_type () =
  match Sarek_ir_metal.metal_type_of_elttype TFloat64 with
  | s ->
      Alcotest.failf
        "Metal mapped TFloat64 to %S instead of refusing. This is #141: the \
         host writes 8 bytes per element and the device would read 4."
        s
  | exception e ->
      let msg = Printexc.to_string e in
      if not (contains ~needle:"Sarek_real64" msg) then
        Alcotest.failf
          "Metal's f64 refusal must name Sarek_real64 as the supported route \
           (it is what the runtime already selects — Metal_api reports \
           supports_fp64 = false). Got: %s"
          msg

(** The whole-kernel gate. The element-type arm alone does not see a kernel
    whose f64 only ever appears as a [CFloat64] literal. Both [generate] entry
    points must refuse. *)
let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let f64_scale_kernel () =
  let out = make_var "out" (TVec TFloat64) in
  let inp = make_var "inp" (TVec TFloat64) in
  let idx = make_var "idx" TInt32 in
  {
    kern_name = "f64_scale";
    kern_params =
      [
        DParam (out, Some {arr_elttype = TFloat64; arr_memspace = Global});
        DParam (inp, Some {arr_elttype = TFloat64; arr_memspace = Global});
      ];
    kern_locals = [];
    kern_body =
      SLet
        ( idx,
          EIntrinsic ([], "global_thread_id", []),
          SAssign
            ( LArrayElem ("out", EVar idx),
              EBinop (Mul, EArrayRead ("inp", EVar idx), EConst (CFloat64 2.0))
            ) );
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

(** f64 reachable ONLY through a constant and an f64-typed local — the shape the
    per-element-type arm would not necessarily see. Pre-fix this emitted
    ["float x = 0.10000000000000001;"], captured verbatim. *)
let f64_local_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let x = make_var "x" TFloat64 in
  let idx = make_var "idx" TInt32 in
  {
    kern_name = "f64_local";
    kern_params =
      [DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global})];
    kern_locals = [];
    kern_body =
      SLet
        ( idx,
          EIntrinsic ([], "global_thread_id", []),
          SLet
            ( x,
              EConst (CFloat64 0.1),
              SAssign (LArrayElem ("out", EVar idx), ECast (TFloat32, EVar x))
            ) );
    kern_types = [];
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let expect_metal_kernel_refused label gen =
  match gen () with
  | (src : string) ->
      Alcotest.failf
        "Metal accepted the f64 kernel %s and emitted:\n%s"
        label
        src
  | exception e ->
      let msg = Printexc.to_string e in
      if not (contains ~needle:"float64" msg) then
        Alcotest.failf
          "Metal/%s: diagnostic does not name float64: %s"
          label
          msg

let test_metal_refuses_f64_kernels () =
  List.iter
    (fun (label, k) ->
      expect_metal_kernel_refused (label ^ "/generate") (fun () ->
          Sarek_ir_metal.generate k) ;
      expect_metal_kernel_refused (label ^ "/generate_with_types") (fun () ->
          Sarek_ir_metal.generate_with_types ~types:k.kern_types k))
    [("f64_scale", f64_scale_kernel ()); ("f64_local", f64_local_kernel ())]

(** The refusal must be f64-specific: it must not have broken f32 on Metal. *)
let test_metal_still_accepts_f32 () =
  let out = make_var "out" (TVec TFloat32) in
  let inp = make_var "inp" (TVec TFloat32) in
  let idx = make_var "idx" TInt32 in
  let k =
    {
      kern_name = "f32_scale";
      kern_params =
        [
          DParam (out, Some {arr_elttype = TFloat32; arr_memspace = Global});
          DParam (inp, Some {arr_elttype = TFloat32; arr_memspace = Global});
        ];
      kern_locals = [];
      kern_body =
        SLet
          ( idx,
            EIntrinsic ([], "global_thread_id", []),
            SAssign
              ( LArrayElem ("out", EVar idx),
                EBinop (Mul, EArrayRead ("inp", EVar idx), EConst (CFloat32 2.0))
              ) );
      kern_types = [];
      kern_variants = [];
      kern_funcs = [];
      kern_native_fn = None;
    }
  in
  ignore (Sarek_ir_metal.generate k) ;
  ignore (Sarek_ir_metal.generate_with_types ~types:k.kern_types k)

(** Both refusal sites must render the SAME capability entry, so the two cannot
    drift apart the way the OpenCL f16 pair did before #138.

    Since #64 slice 1 the single source is not a string constant but a row in
    the capability table — [Sarek_capability.float64_absent_metal], rendered by
    [explain]. Asserting against [explain] rather than against a literal is what
    makes this a check on the TABLE being the source: a site that reconstructed
    an equivalent sentence by hand would fail here. *)
let test_metal_f64_refusal_is_single_sourced () =
  let msg_of f =
    match f () with
    | (_ : string) -> Alcotest.fail "expected a refusal"
    | exception e -> Printexc.to_string e
  in
  let arm = msg_of (fun () -> Sarek_ir_metal.metal_type_of_elttype TFloat64) in
  let kern = msg_of (fun () -> Sarek_ir_metal.generate (f64_scale_kernel ())) in
  let expected =
    Sarek_capability.explain
      ~target:"Metal"
      Sarek_capability.float64_absent_metal
  in
  if not (contains ~needle:expected arm) then
    Alcotest.failf
      "the element-type arm does not render the capability entry: %s"
      arm ;
  if not (contains ~needle:expected kern) then
    Alcotest.failf
      "the whole-kernel gate does not render the capability entry: %s"
      kern ;
  (* And the entry must be the RIGHT kind. A refusal decided at codegen is only
     legitimate for a kind that needs no device — filing this as, say,
     Device_optional would make a static refusal over-refuse a device that could
     have supplied it. *)
  Alcotest.(check bool)
    "Metal f64 is decidable without a device"
    false
    (Sarek_capability.kind_needs_device
       Sarek_capability.float64_absent_metal.cap_kind)

let () =
  Alcotest.run
    "backend_type_width_totality"
    [
      ( "gate is not vacuous",
        [
          Alcotest.test_case
            "enumeration and backend table are complete"
            `Quick
            test_enumeration_is_complete;
          Alcotest.test_case
            "every backend checks at least one width"
            `Quick
            test_every_backend_checks_something;
          Alcotest.test_case
            "the no-memory-form exemption set is exactly as recorded"
            `Quick
            test_no_memory_form_set_is_exactly_as_recorded;
        ] );
      ( "silent-narrowing class validator",
        [
          Alcotest.test_case
            "no backend maps an element type to a different width without \
             refusing"
            `Quick
            test_no_backend_silently_changes_width;
        ] );
      ( "#141 Metal f64",
        [
          Alcotest.test_case
            "the element-type arm refuses and names Sarek_real64"
            `Quick
            test_metal_refuses_f64_element_type;
          Alcotest.test_case
            "both generate entry points refuse an f64 kernel"
            `Quick
            test_metal_refuses_f64_kernels;
          Alcotest.test_case
            "f32 on Metal is untouched"
            `Quick
            test_metal_still_accepts_f32;
          Alcotest.test_case
            "both refusal sites quote the single-sourced sentence"
            `Quick
            test_metal_f64_refusal_is_single_sourced;
        ] );
    ]
