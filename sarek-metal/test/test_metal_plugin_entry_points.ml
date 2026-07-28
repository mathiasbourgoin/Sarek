(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** [Metal_plugin.generate_source] must declare the record types it uses
    (backlog-155).

    This is the Metal half of [sarek/tests/unit/test_typedef_entry_points.ml],
    and it lives here rather than there for a mechanical reason: [sarek_metal]
    is an [(optional)] library and the only place in the tree that links it is
    [sarek-metal/test], so this is the only directory from which the entry point
    can be CALLED.

    That mattered more than it sounds. [Metal_plugin.ml:287] used to be
    [let generate_source = Sarek_ir_metal.generate] — a top-level alias of the
    typedef-less emitter, under the same name as [Backend.generate_source] in
    the module above it, which does route through [generate_with_types]. When
    the alias was replaced by a [~types]-taking function, the change compiled
    with nothing to check it: [generate_source] at that line has ZERO callers
    outside its own file, so neither the new signature nor the new routing was
    exercised by anything. A signature change that no call site type-checks
    against, and a routing change that no test executes, are both free.

    Same shape as the OpenCL/transpiler file: the declaration marker must appear
    BEFORE the use marker (a typedef emitted below the code that needs it is
    invalid Metal all the same), and each assertion is paired with a control
    against [Sarek_ir_metal.generate] so it cannot pass vacuously. *)

open Sarek_metal
open Sarek_ir_types

let index_of haystack needle =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    if i + nl > hl then None
    else if String.sub haystack i nl = needle then Some i
    else go (i + 1)
  in
  if nl = 0 then Some 0 else go 0

let contains haystack needle = index_of haystack needle <> None

let check_declared_before_use ~ctx ~decl ~use src =
  match (index_of src decl, index_of src use) with
  | _, None ->
      Alcotest.failf
        "%s: the body never uses the type (marker %S), so this case proves \
         nothing. Emitted source:\n\
         %s"
        ctx
        use
        src
  | None, Some _ ->
      Alcotest.failf
        "%s: emitted source uses the type but never declares it (missing \
         marker %S). Emitted source:\n\
         %s"
        ctx
        decl
        src
  | Some d, Some u ->
      Alcotest.(check bool)
        (Printf.sprintf
           "%s: declaration %S is at offset %d but the use %S is at offset %d \
            — the declaration must come FIRST, this source is rejected by any \
            compiler. Emitted source:\n\
            %s"
           ctx
           decl
           d
           use
           u
           src)
        true
        (d < u)

let make_var name ty =
  {var_id = 0; var_name = name; var_type = ty; var_mutable = false}

let point_types = [("point", [("x", TFloat32); ("y", TFloat32)])]

let point_kernel () =
  let out = make_var "out" (TVec TFloat32) in
  let p = make_var "p" (TRecord ("point", List.assoc "point" point_types)) in
  {
    kern_name = "k";
    kern_params = [DParam (out, None)];
    kern_locals = [DLocal (p, None)];
    kern_body =
      SSeq
        [
          SAssign
            ( LArrayElem ("out", EConst (CInt32 0l)),
              EBinop
                (Add, ERecordField (EVar p, "x"), ERecordField (EVar p, "y")) );
        ];
    kern_types = point_types;
    kern_variants = [];
    kern_funcs = [];
    kern_native_fn = None;
  }

let typedef_line = "} point;"

let record_use = "point p"

let test_generate_source_declares_records () =
  let src = Metal_plugin.generate_source ~types:point_types (point_kernel ()) in
  check_declared_before_use
    ~ctx:"Metal_plugin.generate_source"
    ~decl:typedef_line
    ~use:record_use
    src

(** POSITIVE CONTROL: [Sarek_ir_metal.generate] is what [generate_source] used
    to alias. The same declaration check must come out FALSE against it, while
    the USE marker is still there — that pairing is what shows the emitted
    source really did name an undeclared struct type. *)
let test_plain_generate_omits_records () =
  let src = Sarek_ir_metal.generate (point_kernel ()) in
  Alcotest.(check bool)
    "control: bare `generate` uses the record"
    true
    (contains src record_use) ;
  Alcotest.(check bool)
    "control: bare `generate` emits no typedef, so the check discriminates"
    false
    (contains src typedef_line)

let () =
  Alcotest.run
    "metal_plugin_entry_points"
    [
      ( "generate_source",
        [
          Alcotest.test_case
            "declares record typedefs, above the use"
            `Quick
            test_generate_source_declares_records;
          Alcotest.test_case
            "control: bare generate does not"
            `Quick
            test_plain_generate_omits_records;
        ] );
    ]
