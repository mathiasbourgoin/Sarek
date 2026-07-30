(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * DEVICE-INDEPENDENT pin on the record/variant declaration EMISSION ORDER, for
 * every backend that emits struct declarations (backlog-211).
 *
 * WHY THIS FILE EXISTS AND THE E2E TEST IS NOT ENOUGH.
 * sarek/tests/e2e/test_record_variant_decl_order.ml reproduces both halves of
 * the gap on real hardware, and that is the measurement — but it can only speak
 * for the devices the host has. This host has OpenCL and Vulkan; it has no
 * NVIDIA, no HIP and no Metal hardware, and there is no WGSL device at all. On
 * top of that, the e2e test EXITS 0 when it finds no device, so on CI — which
 * has no GPU — it asserts nothing whatsoever.
 *
 * This file closes both holes: it calls the generators directly, so it needs no
 * device, and it sweeps all five of them, so the four families this host cannot
 * execute are covered by the same assertion as the two it can.
 *
 * THE INVARIANT. For each backend and each of the two cross-kind shapes, the
 * declaration that is REFERENCED must be emitted before the declaration that
 * REFERENCES it:
 *
 *   shape A  variant with a record payload   — the C family's half; before the
 *            fix, OpenCL reported `unknown type name '<record>'`
 *   shape B  record with a variant-typed field — GLSL/WGSL's half; before the
 *            fix, Vulkan reported `syntax error, unexpected IDENTIFIER`
 *
 * Each family was green on the shape the other was red on, so both shapes are
 * swept over every backend rather than each shape over the family it broke.
 *
 * WHY THIS TEST CANNOT SILENTLY PASS (see the "gates that cannot fail" list in
 * kb/properties.md):
 *   - the backend table is checked against a pinned count, so deleting a
 *     backend from the sweep is a failure, not a smaller sweep;
 *   - a missing declaration ANCHOR is a failure, not a skip: if a backend stops
 *     emitting one of the two declarations the assertion has nothing to compare
 *     and says so, rather than comparing two -1s and finding them ordered;
 *   - the referenced name must be named INSIDE the referencing declaration's own
 *     body. A backend that emitted both declarations but dropped the
 *     field/payload that creates the edge would otherwise satisfy the ordering
 *     assertion vacuously — and counting the name across the WHOLE source would
 *     not catch that either, because a variant's emission repeats its own type
 *     name in every constructor signature;
 *   - the anchors are per-family spellings, not a generic "first occurrence of
 *     the name". In the C family a typedef names its type only at the END, so
 *     "first occurrence" is satisfied by the WRONG order too — the very
 *     measurement this file exists to make would have read green.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen

(* Distinctive names: the vacuity guard counts occurrences of a name as a
   substring, so no name here may be a substring of another. *)
let dep_record_name = "pt211"

let user_variant_name = "probe211"

let dep_variant_name = "flag211"

let user_record_name = "gauge211"

let dep_record_fields = [("px", TFloat32); ("py", TFloat32)]

let dep_record = TRecord (dep_record_name, dep_record_fields)

let dep_variant_constrs = [("Off211", []); ("Level211", [TFloat32])]

let dep_variant = TVariant (dep_variant_name, dep_variant_constrs)

(* A kernel skeleton every generator accepts: one float32 vector parameter, an
   empty body. The declarations under test come from [~types] and
   [kern_variants], not from the body — this file pins the DECLARATION order,
   and a body would only add ways for a generator to refuse the input. *)
let kernel_with ~variants =
  {
    default_kernel with
    kern_name = "decl_order_probe";
    kern_params =
      [
        DParam
          ( {
              var_name = "dst";
              var_id = 0;
              var_type = TVec TFloat32;
              var_mutable = false;
            },
            Some {arr_elttype = TFloat32; arr_memspace = Global} );
      ];
    kern_body = SEmpty;
    kern_variants = variants;
  }

(* Where a name becomes usable in each family's syntax, and where its own
   declaration BODY starts and ends.

   C family: a typedef names its type on the CLOSING line, so the anchor is the
   point after which the name exists, and the body runs back to the matching
   [typedef struct {]. GLSL and WGSL open with the name, and declarations do not
   nest, so the opening line orders them just as exactly and the body runs
   forward to the first closing brace. *)
type family = {
  anchor : string -> string;
  (* [body src name] is [name]'s own declaration text, or [None] if the
     declaration is not there at all. *)
  body : string -> string -> string option;
}

let find_sub_from hay needle start =
  let nl = String.length needle and hl = String.length hay in
  let rec go i =
    if i + nl > hl then -1
    else if String.sub hay i nl = needle then i
    else go (i + 1)
  in
  if start < 0 then -1 else go start

let find_sub hay needle = find_sub_from hay needle 0

let rfind_sub_before hay needle limit =
  let nl = String.length needle in
  let rec go i =
    if i < 0 then -1
    else if i + nl <= String.length hay && String.sub hay i nl = needle then i
    else go (i - 1)
  in
  go (min limit (String.length hay - nl))

let c_family =
  {
    anchor = (fun name -> "} " ^ name ^ ";");
    body =
      (fun src name ->
        let e = find_sub src ("} " ^ name ^ ";") in
        if e < 0 then None
        else
          let s = rfind_sub_before src "typedef struct {" e in
          if s < 0 then None else Some (String.sub src s (e - s)));
  }

let struct_family =
  {
    anchor = (fun name -> "struct " ^ name ^ " {");
    body =
      (fun src name ->
        let s = find_sub src ("struct " ^ name ^ " {") in
        if s < 0 then None
        else
          let e = find_sub_from src "}" s in
          if e < 0 then None else Some (String.sub src s (e - s)));
  }

type backend = {
  bname : string;
  generate : types:(string * (string * elttype) list) list -> kernel -> string;
  family : family;
}

(* All five generators that emit struct declarations. PTX is absent on purpose:
   Sarek_ir_ptx_kernel.generate_with_types ignores [~types] and declares no
   struct types at all. HIP has no generator of its own — Hip_plugin calls the
   CUDA one, so the CUDA row covers it. *)
let backends =
  [
    {
      bname = "CUDA (also HIP, via Hip_plugin)";
      generate = (fun ~types k -> Sarek_ir_cuda.generate_with_types ~types k);
      family = c_family;
    };
    {
      bname = "OpenCL";
      generate = (fun ~types k -> Sarek_ir_opencl.generate_with_types ~types k);
      family = c_family;
    };
    {
      bname = "Metal";
      generate = (fun ~types k -> Sarek_ir_metal.generate_with_types ~types k);
      family = c_family;
    };
    {
      bname = "GLSL (Vulkan)";
      generate = (fun ~types k -> Sarek_ir_glsl.generate_with_types ~types k);
      family = struct_family;
    };
    {
      bname = "WGSL";
      generate = (fun ~types k -> Sarek_ir_wgsl.generate_with_types ~types k);
      family = struct_family;
    };
  ]

(* Deleting a backend from the sweep must be a failure, not a quieter run. *)
let expected_backends = 5

(* One (backend, shape) cell. RETURNS its problems, tagged with the backend,
   rather than raising: the sweep must report EVERY failing cell, not only the
   first one. Alcotest's [failf] aborts the case, and the two halves of this gap
   are one backend apart in the table — a fail-fast sweep would have named CUDA
   and said nothing about the other four, which is the shape of report that made
   the gap look like one family's problem in the first place. *)
let check_cell ~b ~shape ~dep ~user ~src =
  let where = Printf.sprintf "%s / %s" b.bname shape in
  let anchor = b.family.anchor in
  let dep_at = find_sub src (anchor dep) in
  let user_at = find_sub src (anchor user) in
  let problems = ref [] in
  let problem fmt =
    Printf.ksprintf (fun s -> problems := (b.bname, s) :: !problems) fmt
  in
  if dep_at < 0 then
    problem
      "%s: no declaration of %S found (looked for %S) in:\n%s"
      where
      dep
      (anchor dep)
      src ;
  if user_at < 0 then
    problem
      "%s: no declaration of %S found (looked for %S) in:\n%s"
      where
      user
      (anchor user)
      src ;
  (* Vacuity guard. [dep] must be named INSIDE [user]'s own declaration body,
     which is the reference that creates the edge under test. Counting [dep]
     across the whole source instead would be vacuous for shape B: a variant's
     emission repeats its own type name in every constructor signature, so the
     count stays high after the record field naming it is dropped, and on the C
     family — which declares variants first anyway — the order assertion below
     would then pass with no edge left to order. *)
  (match b.family.body src user with
  | None ->
      (* Already reported as a missing declaration above; no second problem. *)
      ()
  | Some body ->
      if find_sub body dep < 0 then
        problem
          "%s: %S is not named inside %S's own declaration body, so the edge \
           under test is gone and the order assertion would be vacuous. Body \
           was:\n\
           %s\n\
           Full source:\n\
           %s"
          where
          dep
          user
          body
          src) ;
  (* Only compare positions that were actually found: two -1s compare as
     ordered, which is how a sweep over a backend that emitted nothing at all
     reads green. *)
  if dep_at >= 0 && user_at >= 0 && dep_at > user_at then
    problem
      "%s: %S is declared at %d, AFTER %S at %d, which references it:\n%s"
      where
      dep
      dep_at
      user
      user_at
      src ;
  List.rev !problems

(* Report every failing cell of one shape's sweep in one message. One backend can
   contribute more than one problem, so the two counts are reported separately —
   "N problems" is not "N backends". *)
let report shape problems =
  if problems <> [] then
    let failed_backends =
      List.sort_uniq String.compare (List.map fst problems)
    in
    Alcotest.failf
      "%s: %d problem(s) across %d of %d backend(s) (%s):\n\n%s"
      shape
      (List.length problems)
      (List.length failed_backends)
      (List.length backends)
      (String.concat ", " failed_backends)
      (String.concat "\n\n" (List.map snd problems))

(* Shape A — a variant whose payload is a record. The C family emitted variants
   before records, so this was its red half. *)
let test_variant_with_record_payload () =
  List.concat_map
    (fun b ->
      let k =
        kernel_with
          ~variants:
            [(user_variant_name, [("Nowhere211", []); ("At211", [dep_record])])]
      in
      let src = b.generate ~types:[(dep_record_name, dep_record_fields)] k in
      check_cell
        ~b
        ~shape:"shape A: variant with a record payload"
        ~dep:dep_record_name
        ~user:user_variant_name
        ~src)
    backends
  |> report "shape A: variant with a record payload"

(* Shape B — a record whose field type is a variant. GLSL and WGSL emitted
   records before variants, so this was their red half. *)
let test_record_with_variant_field () =
  List.concat_map
    (fun b ->
      let k = kernel_with ~variants:[(dep_variant_name, dep_variant_constrs)] in
      let src =
        b.generate
          ~types:[(user_record_name, [("gk", dep_variant); ("gv", TFloat32)])]
          k
      in
      check_cell
        ~b
        ~shape:"shape B: record with a variant field"
        ~dep:dep_variant_name
        ~user:user_record_name
        ~src)
    backends
  |> report "shape B: record with a variant field"

let test_backend_count () =
  Alcotest.(check int)
    "every declaration-emitting backend is in the sweep"
    expected_backends
    (List.length backends)

let () =
  Alcotest.run
    "declaration order, all backends (backlog-211)"
    [
      ( "cross-kind declaration order",
        [
          Alcotest.test_case
            "backend sweep is complete"
            `Quick
            test_backend_count;
          Alcotest.test_case
            "variant with a record payload"
            `Quick
            test_variant_with_record_payload;
          Alcotest.test_case
            "record with a variant field"
            `Quick
            test_record_with_variant_field;
        ] );
    ]
