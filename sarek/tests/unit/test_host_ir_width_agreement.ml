(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * CLASS VALIDATOR for the HOST side of the element-type surface — the third
 * and last segment of the pipeline, and the one that had no sweep.
 *
 * The two existing sweeps cover the ends and meet nowhere:
 *
 *   sarek/tests/unit/test_type_width_totality.ml
 *       source type  ->  IR element type          (stops AT the IR)
 *   sarek/tests/codegen_golden/test_backend_type_width_totality.ml
 *       IR element type  ->  device type string   (starts AT the IR)
 *
 * Neither says anything about the third edge:
 *
 *       IR element type  <->  HOST scalar_kind  ->  exec-arg dispatch
 *
 * That edge is where the bytes actually live. It is also where measurement
 * says the expensive defects are. Adding a constructor to
 * [Sarek_ir_types.elttype] and to [Sarek_types.registered_type] makes 21
 * production files fail to compile; the f16 width needed 40, and the other
 * half — [Spoc_core_base_scalar], [Vector], [Memory], [Ctypes_ops],
 * [Execute], the interpreter narrowing, the plugin bases — is invisible to
 * the compiler, because none of it matches on the IR type. Two of the three
 * f16 soundness bugs lived in that invisible half:
 *
 *   - [Execute]'s exec-arg [get]/[set] had no [Float16] arm, so an f16 kernel
 *     could not run on the Interpreter DEVICE at all. The dispatch is a match
 *     on a (typed_value, kind) PAIR with a catch-all, so no constructor
 *     addition anywhere can force that arm to exist. Nothing failed to
 *     compile; the feature was simply absent.
 *   - the host width and the IR width can disagree with no diagnostic — the
 *     [char] case (Bigarray char is ONE byte, [Ir.TInt32] declares four) is
 *     the recorded instance.
 *
 * This file closes that edge by the same discipline the other two use: a
 * total successor chain with NO wildcard, so a new host scalar kind fails to
 * COMPILE here (warning 8 is an error in this test's flags), and a pinned
 * refusal set, so a kind that stops round-tripping is a deliberate edit
 * rather than a quiet narrowing.
 *
 * WHY THIS GATE CANNOT SILENTLY PASS (see the "gates that cannot fail" list):
 *
 *  - the kind enumeration is unfolded from [next_scalar], a total match on
 *    the [scalar_kind] GADT with no wildcard, and its length is pinned;
 *  - the labels are checked for DISTINCTNESS, so an off-by-one walk that
 *    omits one kind while duplicating another cannot pass on the count;
 *  - every probe carries a value that is NOT the zero/default of its type, so
 *    a round trip that silently returns a fresh cell fails;
 *  - the set of kinds with no IR counterpart, and the set whose exec-arg
 *    dispatch refuses, are each pinned EXACTLY — growing either is an edit to
 *    a list, not a smaller sweep;
 *  - [test_sweep_is_not_vacuous] requires a floor of kinds that actually
 *    complete a round trip, so a change that made everything refuse would
 *    fail here rather than pass trivially.
 ******************************************************************************)

module Ir = Sarek_ir_types
module Vector = Spoc_core.Vector
module Execute = Sarek.Execute
module Framework_sig = Spoc_framework.Framework_sig

(* ------------------------------------------------------------------ *)
(* Compiler-enforced enumeration of the HOST scalar kinds              *)
(* ------------------------------------------------------------------ *)

type packed = Pack : ('a, 'b) Vector.scalar_kind -> packed

(** Total match, NO wildcard: adding a constructor to
    [Spoc_core_base_scalar.scalar_kind] fails to compile here. *)
let next_scalar : type a b. (a, b) Vector.scalar_kind -> packed option =
  function
  | Vector.Float16 -> Some (Pack Vector.Float32)
  | Vector.Float32 -> Some (Pack Vector.Float64)
  | Vector.Float64 -> Some (Pack Vector.Int32)
  | Vector.Int32 -> Some (Pack Vector.Int64)
  | Vector.Int64 -> Some (Pack Vector.Char)
  | Vector.Char -> Some (Pack Vector.Complex32)
  | Vector.Complex32 -> None

let unfold_kinds () =
  let rec go acc (Pack k as p) =
    match next_scalar k with
    | Some q -> go (p :: acc) q
    | None -> List.rev (p :: acc)
  in
  go [] (Pack Vector.Float16)

let all_kinds = unfold_kinds ()

(* ------------------------------------------------------------------ *)
(* What each host kind must satisfy                                    *)
(* ------------------------------------------------------------------ *)

(** A probe carries a host kind together with a representative value, so the
    exec-arg round trip below can be run without knowing the element type
    statically.

    [ir] is the IR element type this host kind stores. [None] means the host
    kind has no IR counterpart at all, which is a real state and is pinned below
    — it must never be reached by forgetting to add one. *)
type probe =
  | Probe : {
      kind : ('a, 'b) Vector.kind;
      label : string;
      ir : Ir.elttype option;
      sample : 'a;
          (** deliberately not the zero value of the type: a round trip that
              returns a fresh, untouched cell must not be able to pass *)
      expect : 'a;  (** [sample] after a store-and-load through this kind *)
      equal : 'a -> 'a -> bool;
      show : 'a -> string;
    }
      -> probe

let cx_equal (a : Complex.t) (b : Complex.t) = a.re = b.re && a.im = b.im

let cx_show (c : Complex.t) = Printf.sprintf "%g+%gi" c.re c.im

(** Total match, NO wildcard: a new host scalar kind fails to compile here too,
    and — unlike a table keyed by name — it cannot be satisfied by a stub,
    because the arm has to produce a value of the kind's own element type. *)
let probe_of : type a b. (a, b) Vector.scalar_kind -> probe = function
  | Vector.Float16 ->
      Probe
        {
          kind = Vector.float16;
          label = "Float16";
          ir = Some Ir.TFloat16;
          sample = 3.14159;
          (* Nearest binary16 to 3.14159. Reading back 3.14159 would prove the
             store is not actually 2-byte, which is the width claim this file
             checks from the other direction. *)
          expect = 3.140625;
          equal = Float.equal;
          show = string_of_float;
        }
  | Vector.Float32 ->
      Probe
        {
          kind = Vector.float32;
          label = "Float32";
          ir = Some Ir.TFloat32;
          sample = 2.5;
          expect = 2.5;
          equal = Float.equal;
          show = string_of_float;
        }
  | Vector.Float64 ->
      Probe
        {
          kind = Vector.float64;
          label = "Float64";
          ir = Some Ir.TFloat64;
          sample = 2.5;
          expect = 2.5;
          equal = Float.equal;
          show = string_of_float;
        }
  | Vector.Int32 ->
      Probe
        {
          kind = Vector.int32;
          label = "Int32";
          ir = Some Ir.TInt32;
          sample = 7l;
          expect = 7l;
          equal = Int32.equal;
          show = Int32.to_string;
        }
  | Vector.Int64 ->
      Probe
        {
          kind = Vector.int64;
          label = "Int64";
          ir = Some Ir.TInt64;
          sample = 9L;
          expect = 9L;
          equal = Int64.equal;
          show = Int64.to_string;
        }
  | Vector.Char ->
      Probe
        {
          kind = Vector.char;
          label = "Char";
          (* No IR counterpart, and this is the recorded wrong-width defect:
             [Spoc_core.Vector.char] is a Bigarray of OCaml chars, ONE byte per
             element, and the IR has no 1-byte element type. [char] used to
             lower to [Ir.TInt32] — declaring `int*` on the device, four bytes
             — so a `char vector` kernel strode the buffer at 4x the host's
             element size with no diagnostic. [Sarek_lower_ir] now rejects it
             rather than mapping it, which is why this is [None] and not a
             width this sweep could check. *)
          ir = None;
          sample = 'A';
          expect = 'A';
          equal = Char.equal;
          show = (fun c -> String.make 1 c);
        }
  | Vector.Complex32 ->
      Probe
        {
          kind = Vector.complex32;
          label = "Complex32";
          (* No IR counterpart: the IR has no complex element type at all. *)
          ir = None;
          sample = {Complex.re = 1.5; im = -2.5};
          expect = {Complex.re = 1.5; im = -2.5};
          equal = cx_equal;
          show = cx_show;
        }

let all_probes = List.map (fun (Pack k) -> probe_of k) all_kinds

let label_of (Probe p) = p.label

(* ------------------------------------------------------------------ *)
(* Anti-vacuity                                                        *)
(* ------------------------------------------------------------------ *)

let test_enumeration_is_complete () =
  Alcotest.(check int) "host scalar kinds swept" 7 (List.length all_kinds) ;
  (* A count alone does not prove the chain was walked correctly: an off-by-one
     walk that omits one kind and duplicates another keeps the count. *)
  let labels = List.map label_of all_probes in
  if List.length (List.sort_uniq compare labels) <> List.length labels then
    Alcotest.failf
      "the host-kind enumeration contains duplicates, so some kind is going \
       unswept: %s"
      (String.concat ", " labels)

(** Pinned exactly. A host scalar kind with no IR element type cannot be used as
    a kernel element type at all, so growing this set silently is how a width
    would get dropped from the DSL without anything going red. *)
let expected_no_ir_counterpart = ["Char"; "Complex32"]

let test_no_ir_counterpart_set_is_exactly_as_recorded () =
  let actual =
    List.filter_map
      (fun (Probe p) -> if p.ir = None then Some p.label else None)
      all_probes
  in
  if List.sort compare actual <> List.sort compare expected_no_ir_counterpart
  then
    Alcotest.failf
      "the set of host scalar kinds with NO IR element type changed.\n\
       expected: %s\n\
       actual:   %s\n\
       A kind in this set cannot be a kernel element type. Adding one must be \
       a deliberate edit backed by a reason, not a consequence of forgetting \
       to map it."
      (String.concat ", " (List.sort compare expected_no_ir_counterpart))
      (String.concat ", " (List.sort compare actual))

(* ------------------------------------------------------------------ *)
(* INVARIANT 1 — the host width IS the IR width                        *)
(* ------------------------------------------------------------------ *)

(** [Sarek_ir_layout.scalar_size] is what the device-side sweep uses as its
    denominator, and [Vector.elem_size] is what the host allocator and every
    transfer actually stride by. If they disagree, the two sweeps are each
    internally consistent and the pipeline is still wrong — which is precisely
    the shape of the [char] defect. *)
let test_host_width_equals_ir_width () =
  List.iter
    (fun (Probe p) ->
      match p.ir with
      | None -> ()
      | Some t ->
          let host = Vector.elem_size p.kind in
          let ir = Sarek_ir_layout.scalar_size t in
          if host <> ir then
            Alcotest.failf
              "%s: the host stores %d byte(s) per element (Vector.elem_size) \
               but the IR lays the same element out in %d byte(s) \
               (Sarek_ir_layout.scalar_size, which is the denominator the \
               backend width sweep checks device types against). A kernel \
               using it would compile, run, and stride the buffer wrong with \
               no diagnostic."
              p.label
              host
              ir)
    all_probes

(* ------------------------------------------------------------------ *)
(* INVARIANT 2 — every host kind survives the exec-arg dispatch        *)
(* ------------------------------------------------------------------ *)

(* The dispatch this exercises is [Execute.exec_arg_of_vector]'s [get]/[set],
    which is a match on a (typed_value, Vector.kind) PAIR ending in a
    catch-all. No constructor addition can force an arm into it, so nothing
    but a sweep can tell you an arm is missing — and a missing arm is not a
    compile error, it is a feature that silently does not exist on the
    Interpreter device.

    Run as a store-through-the-framework / load-through-the-framework round
   trip so that a [set] arm that accepts a value and drops it fails here too. *)

(** Outcome of one round trip. The element type is existential inside [Probe],
    so the comparison must happen in the scope that unpacks it — hence a verdict
    rather than a returned value. *)
type outcome = Completed | Lost of string | Refused

let roundtrip_through_exec_arg (Probe p) : outcome =
  let v = Vector.create p.kind 4 in
  Vector.set v 0 p.sample ;
  match Execute.exec_arg_of_vector v with
  | Framework_sig.EA_Vec (module EV) -> (
      match
        let tv = EV.get 0 in
        EV.set 1 tv ;
        Vector.get v 1
      with
      | got ->
          if p.equal got p.expect then Completed
          else Lost (Printf.sprintf "%s, not %s" (p.show got) (p.show p.expect))
      | exception _ -> Refused)
  | EA_Int32 _ | EA_Int64 _ | EA_Float32 _ | EA_Float64 _ | EA_Scalar _
  | EA_Composite _ ->
      Alcotest.failf "%s: exec_arg_of_vector did not produce an EA_Vec" p.label

(** Pinned exactly, and every entry is a kind with no IR counterpart — i.e. one
    that cannot be a kernel element type anyway. A kind that CAN be a kernel
    element type appearing here would mean that element type does not work on
    the Interpreter device, which is exactly the f16 hole. *)
let expected_exec_arg_refusals = ["Char"; "Complex32"]

let test_exec_arg_roundtrip_totality () =
  let refused = ref [] in
  List.iter
    (fun (Probe p as probe) ->
      match roundtrip_through_exec_arg probe with
      | Completed -> ()
      | Refused -> refused := p.label :: !refused
      | Lost detail ->
          Alcotest.failf
            "%s: a value written and read back through the exec-arg interface \
             came out as %s. The dispatch accepted the element and lost it."
            p.label
            detail)
    all_probes ;
  if List.sort compare !refused <> List.sort compare expected_exec_arg_refusals
  then
    Alcotest.failf
      "the set of host scalar kinds the exec-arg dispatch REFUSES changed.\n\
       expected: %s\n\
       actual:   %s\n\
       A kind that is a legal kernel element type but is refused here cannot \
       run on the Interpreter device — the framework's get/set dispatch ends \
       in a catch-all, so this sweep is the only thing that can notice."
      (String.concat ", " (List.sort compare expected_exec_arg_refusals))
      (String.concat ", " (List.sort compare !refused))

(** A change that made every kind refuse would satisfy a pinned-set check that
    was written loosely. Require a floor of kinds that actually complete. *)
let test_sweep_is_not_vacuous () =
  let n =
    List.length
      (List.filter
         (fun probe -> roundtrip_through_exec_arg probe = Completed)
         all_probes)
  in
  if n < 5 then
    Alcotest.failf
      "only %d host scalar kind(s) complete an exec-arg round trip; every \
       assertion in this file about the rest is vacuous"
      n

let () =
  Alcotest.run
    "host_ir_width_agreement"
    [
      ( "gate is not vacuous",
        [
          Alcotest.test_case
            "the host-kind enumeration is complete and distinct"
            `Quick
            test_enumeration_is_complete;
          Alcotest.test_case
            "the no-IR-counterpart set is exactly as recorded"
            `Quick
            test_no_ir_counterpart_set_is_exactly_as_recorded;
          Alcotest.test_case
            "enough kinds complete a round trip for the sweep to mean something"
            `Quick
            test_sweep_is_not_vacuous;
        ] );
      ( "host/IR width agreement",
        [
          Alcotest.test_case
            "the host element size equals Sarek_ir_layout.scalar_size"
            `Quick
            test_host_width_equals_ir_width;
        ] );
      ( "exec-arg dispatch totality",
        [
          Alcotest.test_case
            "every host scalar kind round-trips, or is a pinned refusal"
            `Quick
            test_exec_arg_roundtrip_totality;
        ] );
    ]
