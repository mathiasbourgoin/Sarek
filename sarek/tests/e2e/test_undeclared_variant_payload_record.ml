(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E test: a type referenced but never DECLARED produces a refusal, not
 * silence (backlog-212).
 *
 * PR #397/backlog-211 made record and variant declarations emit from one
 * interleaved dependency-ordered pass, so a cross-kind edge between a
 * DECLARED record and a DECLARED variant is now ordered correctly. That pass
 * only orders declarations that EXIST in [kern_types]/[kern_variants]. If a
 * type is referenced by a field or a constructor payload but was never
 * registered as a declaration at all, there is no node for it in either list,
 * so there is no edge and (before this fix) no error: [gen_type_decls] wrote
 * the referencing declaration anyway, and the emitted source names a
 * struct/union member type with no [typedef]/[struct] anywhere above it.
 *
 * THIS IS REACHABLE FROM ORDINARY [@@sarek.type] SOURCE, established BY
 * EXECUTION (running the ACTUAL [%kernel] PPX output through
 * [Sarek_ir_codegen], not a hand-built [Sarek_ir_types.kernel] value) — see
 * [require_undeclared_reference] below, which fails loudly if a future PPX
 * change closes this gap and the kernel below stops carrying it.
 *
 * WHY THIS SHAPE, MEASURED: [Sarek_lower_ir.ml]'s [TEConstr] case registers a
 * variant's constructor payload from the CONSTRUCTOR'S OWN declared type
 * (via the typer's [repr te.ty]), independent of how the value passed to the
 * constructor was obtained. [register_types_from_typ] (also in
 * [Sarek_lower_ir.ml]) registers a record type when it is a literal
 * ([TERecord]), a parameter type, or a local array element type — but it
 * does NOT recurse through a [TVariant] at all (documented at length on that
 * function: "VARIANTS are deliberately NOT handled here"). So a record whose
 * ONLY occurrence in a kernel is a value extracted from a variant match arm
 * and immediately re-wrapped in a DIFFERENT constructor never enters
 * [kern_types]: [probe2] (the target variant) is registered via [TEConstr],
 * but [probe_pt] (its constructor's payload record) is not, because it is
 * never literal-constructed, never a parameter type, and never reached
 * through the [probe]-typed source parameter (a [TVariant], which
 * [register_types_from_typ] skips).
 *
 * WHAT THIS KERNEL DOES NOT ALSO CLAIM TO FIX. [probe] itself (the SOURCE
 * parameter's type) is never registered either — no [TEConstr] ever applies
 * IT, only matches against it — which is the separate, already-documented
 * "variant-typed kernel parameter" gap [register_types_from_typ]'s doc
 * comment calls out (a kernel whose only variant occurrence is a parameter
 * "already fails today, identically"). That gap is NOT backlog-212 and is not
 * touched by this fix; this kernel is not claimed to compile end-to-end even
 * after it, and no device is run over it here — see the deviceless-only
 * rationale below. What backlog-212 fixes, and what this test isolates, is
 * that [probe2] (which DOES reach [kern_variants]) can no longer silently
 * reference an undeclared [probe_pt].
 *
 * DEVICE-INDEPENDENT ON PURPOSE. [Sarek_ir_codegen.gen_type_decls] (and the
 * per-backend [generate_with_types] built on it) is pure string generation —
 * no device is needed to observe the refusal it raises, and CI has none. This
 * file therefore never opens a [Device.init] and never runs a kernel; it
 * calls [generate_with_types] directly (OpenCL, chosen arbitrarily — the
 * check lives in the shared [spoc/ir] module every struct-emitting backend
 * goes through) and asserts on the raised exception's MESSAGE, not just that
 * something raised: a crash or an unrelated exception would also make an
 * exit-code-only assertion pass.
 ******************************************************************************)

[@@@warning "-32"]

let () = Sarek_native.Native_plugin.init ()

type float32 = float

(* The record that must end up referenced but undeclared. *)
type probe_pt = {px : float32; py : float32} [@@sarek.type]

(* A SOURCE variant, matched but never constructed in this kernel — [probe]
   itself is the separate, out-of-scope gap described above. *)
type probe = Nowhere | At of probe_pt [@@sarek.type]

(* The TARGET variant: constructed via [At2 q], so [TEConstr] registers it in
   [kern_variants] with a payload type of [TRecord ("..probe_pt", ...)] — the
   undeclared reference this file exists to catch. *)
type probe2 = Nowhere2 | At2 of probe_pt [@@sarek.type]

let k =
  snd
    [%kernel
      fun (p : probe) (dst : probe2 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          match p with
          | Nowhere -> dst.(tid) <- Nowhere2
          | At q -> dst.(tid) <- At2 q
        end]

let ir =
  match k.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* THE GAP UNDER TEST MUST BE IN THE LOWERED IR, checked rather than assumed —
   the same discipline test_record_variant_decl_order.ml's
   [require_cross_edge] uses for backlog-211. If a future PPX change starts
   registering [probe_pt] here (closing this specific reachability path), this
   kernel would no longer exercise the codegen refusal at all, and every
   assertion below would vacuously pass by never reaching the exception. *)
let require_undeclared_reference () =
  let open Sarek_ir_types in
  (* [vname] is module-qualified as the PPX registers it (e.g.
     "Test_undeclared_variant_payload_record.probe2"), so match the
     unqualified suffix rather than an exact/mangled name. *)
  let is_probe2 vname =
    let suffix = ".probe2" in
    let sl = String.length suffix and vl = String.length vname in
    vname = "probe2" || (vl >= sl && String.sub vname (vl - sl) sl = suffix)
  in
  let payload_record_name =
    List.find_map
      (fun (vname, constrs) ->
        if not (is_probe2 vname) then None
        else
          List.find_map
            (fun (cname, args) ->
              if cname <> "At2" then None
              else
                List.find_map
                  (function TRecord (rn, _) -> Some rn | _ -> None)
                  args)
            constrs)
      ir.kern_variants
  in
  match payload_record_name with
  | None ->
      Printf.printf
        "NOTHING TO VERIFY: kern_variants=[%s] carries no \"probe2\"/\"At2\" \
         entry with a record payload, so the codegen refusal below would not \
         be exercised. kern_types=[%s]\n\
         %!"
        (String.concat "; " (List.map fst ir.kern_variants))
        (String.concat "; " (List.map fst ir.kern_types)) ;
      exit 1
  | Some rn ->
      if List.exists (fun (n, _) -> n = rn) ir.kern_types then begin
        Printf.printf
          "NOTHING TO VERIFY: %S is present in kern_types=[%s] — the PPX now \
           registers it, so this kernel no longer carries an undeclared \
           reference and the refusal below is not exercised.\n\
           %!"
          rn
          (String.concat "; " (List.map fst ir.kern_types)) ;
        exit 1
      end ;
      Printf.printf
        "confirmed: kern_variants has \"probe2\"'s \"At2\" payload naming %S, \
         absent from kern_types=[%s] (reached from ordinary [@@sarek.type] + \
         [%%kernel] source, not hand-built IR)\n\
         %!"
        rn
        (String.concat "; " (List.map fst ir.kern_types))

(* The refusal itself, on the shared [spoc/ir] entry point every
   struct-emitting backend goes through. OpenCL is arbitrary — the check lives
   in [Sarek_ir_codegen.sort_type_decls_by_dependency], shared by all five. *)
let require_refusal () =
  match
    Sarek_codegen.Sarek_ir_opencl.generate_with_types
      ~types:ir.Sarek_ir_types.kern_types
      ir
  with
  | src ->
      Printf.printf
        "FAILED: generate_with_types produced %d bytes with NO error — the \
         undeclared reference silently reached the emitted source (the \
         pre-backlog-212 defect). First 200 chars:\n\
         %s\n"
        (String.length src)
        (String.sub src 0 (min 200 (String.length src))) ;
      exit 1
  | exception Sarek_ir_codegen.Undeclared_type_ref msgs ->
      let joined = String.concat "; " msgs in
      let has substr =
        let sl = String.length substr and jl = String.length joined in
        let rec go i =
          i + sl <= jl && (String.sub joined i sl = substr || go (i + 1))
        in
        go 0
      in
      (* The names in the message are module-qualified and then mangled
         (e.g. ["Test_..._probe_pt"]), so check unqualified substrings rather
         than exact/mangled equality. *)
      if not (has "probe2" && has {|"At2"|} && has "probe_pt") then begin
        Printf.printf
          "FAILED: Undeclared_type_ref raised but its message does not name \
           the variant, the constructor, and the missing record: %s\n"
          joined ;
        exit 1
      end ;
      Printf.printf "PASSED: Undeclared_type_ref, named: %s\n" joined
  | exception e ->
      Printf.printf
        "FAILED: expected Undeclared_type_ref, got %s\n"
        (Printexc.to_string e) ;
      exit 1

let () =
  print_endline
    "=== a type referenced but never declared is refused, by name \
     (backlog-212) ===" ;
  require_undeclared_reference () ;
  require_refusal () ;
  print_endline "ALL PASSED (deviceless: no device is needed to observe this)"
