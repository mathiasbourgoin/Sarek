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
 *   - the referenced name must occur at least TWICE in the emitted source (its
 *     own declaration, plus at least one reference to it). A backend that
 *     emitted the declarations but dropped the field/payload that creates the
 *     edge would otherwise satisfy the ordering assertion vacuously;
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

(* Where a name becomes usable in each family's syntax.

   C family: a typedef names its type on the CLOSING line, so this is the point
   after which the name exists. GLSL and WGSL open with the name, and
   declarations do not nest, so the opening line orders them just as exactly. *)
let c_anchor name = "} " ^ name ^ ";"

let struct_anchor name = "struct " ^ name ^ " {"

type backend = {
  bname : string;
  generate : types:(string * (string * elttype) list) list -> kernel -> string;
  anchor : string -> string;
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
      anchor = c_anchor;
    };
    {
      bname = "OpenCL";
      generate = (fun ~types k -> Sarek_ir_opencl.generate_with_types ~types k);
      anchor = c_anchor;
    };
    {
      bname = "Metal";
      generate = (fun ~types k -> Sarek_ir_metal.generate_with_types ~types k);
      anchor = c_anchor;
    };
    {
      bname = "GLSL (Vulkan)";
      generate = (fun ~types k -> Sarek_ir_glsl.generate_with_types ~types k);
      anchor = struct_anchor;
    };
    {
      bname = "WGSL";
      generate = (fun ~types k -> Sarek_ir_wgsl.generate_with_types ~types k);
      anchor = struct_anchor;
    };
  ]

(* Deleting a backend from the sweep must be a failure, not a quieter run. *)
let expected_backends = 5

let find_sub hay needle =
  let nl = String.length needle and hl = String.length hay in
  let rec go i =
    if i + nl > hl then -1
    else if String.sub hay i nl = needle then i
    else go (i + 1)
  in
  go 0

let count_sub hay needle =
  let nl = String.length needle and hl = String.length hay in
  if nl = 0 then 0
  else
    let rec go i acc =
      if i + nl > hl then acc
      else if String.sub hay i nl = needle then go (i + 1) (acc + 1)
      else go (i + 1) acc
    in
    go 0 0

(* One (backend, shape) cell: [dep] must be declared before [user], the
   reference that creates the edge must still be present, and both declarations
   must exist at all. *)
let check_cell ~b ~shape ~dep ~user ~src =
  let where = Printf.sprintf "%s / %s" b.bname shape in
  let dep_at = find_sub src (b.anchor dep) in
  let user_at = find_sub src (b.anchor user) in
  if dep_at < 0 then
    Alcotest.failf
      "%s: no declaration of %S found (looked for %S) in:\n%s"
      where
      dep
      (b.anchor dep)
      src ;
  if user_at < 0 then
    Alcotest.failf
      "%s: no declaration of %S found (looked for %S) in:\n%s"
      where
      user
      (b.anchor user)
      src ;
  (* Vacuity guard: [dep] must appear as its own declaration AND at least once
     more, as the reference inside [user]. Without this, a generator that
     emitted two independent declarations would satisfy the order assertion
     while the edge under test had disappeared. *)
  let occurrences = count_sub src dep in
  if occurrences < 2 then
    Alcotest.failf
      "%s: %S occurs %d time(s) — the reference that creates the edge is gone, \
       so the order assertion below would be vacuous:\n\
       %s"
      where
      dep
      occurrences
      src ;
  if dep_at > user_at then
    Alcotest.failf
      "%s: %S is declared at %d, AFTER %S at %d, which references it:\n%s"
      where
      dep
      dep_at
      user
      user_at
      src

(* Shape A — a variant whose payload is a record. The C family emitted variants
   before records, so this was its red half. *)
let test_variant_with_record_payload () =
  List.iter
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

(* Shape B — a record whose field type is a variant. GLSL and WGSL emitted
   records before variants, so this was their red half. *)
let test_record_with_variant_field () =
  List.iter
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
