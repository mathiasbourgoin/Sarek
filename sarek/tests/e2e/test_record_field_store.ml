(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * In-place record-field store on a vector element: `v.(i).f <- e`.
 *
 * backlog-172. The construct is documented, is used by a shipped kernel
 * (p3_scale_y_kernel in test_soa_emitter_equiv), and worked on CUDA/PTX — while
 * the two CPU backends did something else entirely:
 *
 *   Interpreter: REFUSED, raising Unsupported_operation "record field
 *                assignment" / "not fully supported".
 *   Native:      TWO failure modes, split by whether the field is mutable. On a
 *                MUTABLE field it ACCEPTED the store and silently dropped it:
 *                the generated OCaml was a setfield on the fresh record
 *                Vector.get had just marshalled out of storage, so the write hit
 *                a temporary, no error was raised on any path, and the vector
 *                simply kept its old values. On an IMMUTABLE field the same
 *                setfield did not COMPILE ("The record field b is not mutable")
 *                — loud, but misdiagnosed, since mutability was never the
 *                problem. Both are pinned below ([triple] and [mtriple]); "no
 *                error on any path" holds of the mutable half and not of the
 *                construct, and the mutable half is the dangerous one because
 *                the immutable error is what pushed users into it.
 *
 * Native's silent half is the one that shapes this test: it asserts on EVERY
 * available device rather than only on the backend that broke loudly. A
 * silently-dropped store is indistinguishable from a kernel that did not run,
 * so the only thing that catches it is reading the values back and comparing.
 *
 * What is checked, per device:
 *   1. The written field holds the new value.
 *   2. The OTHER fields are untouched — a read-modify-write that rebuilt the
 *      record from defaults would satisfy (1) and destroy the rest.
 *   3. A store into the SECOND field of a mixed record lands in that field and
 *      not in the first, which is what a wrong field index looks like.
 *   4. A CHAINED target, v.(i).mid.b <- e, lands — with witnesses at BOTH levels
 *      — on EVERY enumerated device, with no tolerated failure anywhere. The
 *      depth-1 fix did not cover this: with depth 1 fixed, the interpreter still
 *      read the intermediate record through the registry's copying get_field and
 *      Native still matched only the depth-1 shape, so both silently dropped the
 *      chained store. Nesting is where "it works now" and "the shape I fixed
 *      works now" come apart.
 *
 *      NO TOLERANCE, and it took a second PR to get there. One of the two nested
 *      cases used to fail to COMPILE on OpenCL and Vulkan for a reason that had
 *      nothing to do with the store: record typedefs were emitted in [kern_types]
 *      order with no dependency sort, so an enclosing struct could be written out
 *      BEFORE the struct it names as a field type. It affected a READ-ONLY nested
 *      kernel identically, so it was a struct-ordering gap and not a store gap
 *      (backlog-203). This file carried a per-kernel predictor and a two-dialect
 *      message predicate to tolerate exactly that failure and nothing else.
 *
 *      backlog-203 is now FIXED on main
 *      ([Sarek_ir_codegen.sort_record_types_by_dependency], #394), and the whole
 *      tolerance apparatus is deleted rather than kept: a tolerance for a gap that
 *      no longer exists is a hole the size of the gap, and it would have absorbed
 *      a regression of the very fix that closed it. Measured on this host before
 *      and after — 4 rows reported "known gap" (OpenCL ×2, Vulkan ×2) before, 0
 *      after, 28 rows green across 7 devices.
 *
 *      CUDA/PTX never had the gap: the direct PTX emitter declares no C structs,
 *      so there is no declaration order to get wrong. Measured under ZLUDA on an
 *      RX 7900 XTX and on a Ryzen 9 7950X (ZLUDA-on-AMD both times; there is no
 *      NVIDIA hardware on this machine) — both CUDA/PTX devices compile BOTH
 *      nested kernels and produce correct values at both levels. That measurement
 *      is what the dune rule's LD_LIBRARY_PATH exists for: the gate once
 *      enumerated 7 devices and printed no CUDA/PTX row at all, so the sentence
 *      above was not reproducible by the test that states it. The run now prints
 *      a named NOT-MEASURED-HERE line for every framework this header makes a
 *      claim about and the run did not enumerate — see [claimed_frameworks], and
 *      the note there on what the ABSENCE of such a line does and does not mean.
 *      Metal and HIP are untested here (no such device on this machine), so
 *      nothing is claimed about them; a failure on either is a failure.
 *
 * Every device failure makes the process exit non-zero.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

(* Explicit registration: linking a plugin does not enumerate its devices. *)
let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

(* Three same-width fields: the store target plus two witnesses on either side,
   so a store that overruns in either direction is visible. *)
type triple = {a : float32; b : float32; c : float32} [@@sarek.type]

(* The SAME record with mutable fields, because Native's pre-fix behaviour was
   TWO different failures and only this one is silent.

   With immutable fields the old codegen did not compile at all: the emitted
   setfield produced "The record field b is not mutable" — loud, but
   misdiagnosed, since the problem was never mutability. That error is what
   pushed a user to add [mutable], and THEN the store compiled and was silently
   discarded (point3d in test_soa_emitter_equiv carries exactly that [mutable]
   with a comment describing it as necessary "to write a leaf in place").

   So the immutable case alone would prove-red as a build failure and never
   exercise the silent path. Both are pinned, and after the fix [mutable] is not
   required for either. *)
type mtriple = {
  mutable ma : float32;
  mutable mb : float32;
  mutable mc : float32;
}
[@@sarek.type]

(* Nested target. [outer] holds a whole [triple], so v.(i).mid.b <- e has to
   rebuild TWO levels on the way back into vector storage. The [tag] and the
   sibling fields of [mid] are witnesses: rebuilding either level from defaults
   satisfies "the target changed" and destroys something else. *)
type outer = {tag : float32; mid : triple} [@@sarek.type]

(* The nested target with a MUTABLE leaf, for exactly the reason [mtriple] exists
   at depth 1 — and it was missing, so the committed nested case only ever
   exercised the LOUD half.

   Measured: revert only the nested half of the fix and the [outer] kernel does
   not build ("The record field \"b\" is not mutable"), because the chained
   lvalue falls through to a setfield on the record [Vector.get] marshalled out.
   That is a build failure, not the silent drop the header describes. With a
   mutable leaf the same revert COMPILES and silently drops the store — measured
   "got 1 want 2" on Native at both levels. Both halves are now committed. *)
type mouter = {mtag : float32; mmid : mtriple} [@@sarek.type]

let scale_b =
  snd
    [%kernel
      fun (v : triple vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).b <- v.(tid).b *. 2.0]

let scale_mb =
  snd
    [%kernel
      fun (v : mtriple vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).mb <- v.(tid).mb *. 2.0]

let scale_nested =
  snd
    [%kernel
      fun (v : outer vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).mid.b <- v.(tid).mid.b *. 2.0]

let scale_nested_mut =
  snd
    [%kernel
      fun (v : mouter vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then v.(tid).mmid.mb <- v.(tid).mmid.mb *. 2.0]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let n = 64

let orig i =
  {a = float_of_int i; b = float_of_int (i + 1); c = float_of_int (i + 2)}

let morig i =
  {ma = float_of_int i; mb = float_of_int (i + 1); mc = float_of_int (i + 2)}

(* One checker over both record shapes. [read] returns the three field values in
   (target, witness1, witness2) order so the assertions below are shape-agnostic
   and cannot drift apart between the two cases. *)
let run_case (dev : Device.t) ~(label : string) ~kernel ~make ~read : bool =
  Printf.printf
    "field-store %-9s [%s] %s: %!"
    label
    dev.Device.framework
    dev.Device.name ;
  let v = make () in
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of kernel)
      ~args:[Vec v; Int n]
      ~block:(Sarek.Execute.dims1d (min 64 n))
      ~grid:(Sarek.Execute.dims1d ((n + 63) / 64))
      ()
  with
  | exception e ->
      (* A refusal is a FAILURE here, not a skip. The construct is part of the
         DSL and every backend in this list executes ordinary custom-vector
         kernels; a backend that cannot do this one is the defect. *)
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok = ref true in
      let reported = ref 0 in
      for i = 0 to n - 1 do
        let tgt, w1, w2 = read v i in
        let bad name got want =
          if Float.abs (got -. want) > 1e-4 then begin
            ok := false ;
            if !reported < 3 then begin
              incr reported ;
              Printf.printf "\n  @%d field %s: got %g want %g" i name got want
            end
          end
        in
        (* The written field doubled; the two witnesses untouched. Checking the
           witnesses is not padding: a read-modify-write that rebuilt the record
           from defaults would satisfy the first assertion and destroy the
           rest. *)
        bad "target" tgt (float_of_int (i + 1) *. 2.0) ;
        bad "witness 1 (must be untouched)" w1 (float_of_int i) ;
        bad "witness 2 (must be untouched)" w2 (float_of_int (i + 2))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

(* One nested case. [read] returns (target, same-level witness 1, same-level
   witness 2, outer-level witness) so the two shapes cannot drift apart in what
   they assert.

   NO TOLERATED FAILURE. Every device on the list must compile and pass both
   nested shapes, and any exception is a failure.

   This file used to carry the opposite: a [predict_struct_gap] predictor, an
   allowlist of struct-emitting backends, a two-dialect source inspector and a
   two-clause message predicate, all so that a compile failure could be accepted
   from OpenCL and Vulkan when the emitted source used a nested record's struct
   before declaring it — backlog-203, which was a codegen ordering bug and not a
   store bug. That bug is fixed on `main`
   ([Sarek_ir_codegen.sort_record_types_by_dependency], #394), so the whole
   apparatus is deleted rather than left in place: a tolerance for a gap that no
   longer exists is a hole exactly the size of the gap, and it would have swallowed
   a regression of the fix that closed it.

   Measured on this host after the deletion: 28 rows green across 7 devices, 0
   tolerated. Before #394 the same file reported 4 rows as "known gap" — OpenCL
   ×2 and Vulkan ×2 — so the deletion is what turns those four into assertions. *)
let nested_case (dev : Device.t) ~(label : string) ~kernel ~make ~read : bool =
  Printf.printf
    "field-store %-10s [%s] %s: %!"
    label
    dev.Device.framework
    dev.Device.name ;
  let v = make () in
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of kernel)
      ~args:[Vec v; Int n]
      ~block:(Sarek.Execute.dims1d (min 64 n))
      ~grid:(Sarek.Execute.dims1d ((n + 63) / 64))
      ()
  with
  | exception e ->
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok = ref true in
      let reported = ref 0 in
      for i = 0 to n - 1 do
        let tgt, w1, w2, outer_w = read v i in
        let bad name got want =
          if Float.abs (got -. want) > 1e-4 then begin
            ok := false ;
            if !reported < 3 then begin
              incr reported ;
              Printf.printf "\n  @%d %s: got %g want %g" i name got want
            end
          end
        in
        bad "leaf target" tgt (float_of_int (i + 1) *. 2.0) ;
        (* Same level as the target. *)
        bad "leaf witness 1 (untouched)" w1 (float_of_int i) ;
        bad "leaf witness 2 (untouched)" w2 (float_of_int (i + 2)) ;
        (* OUTER level: a rebuild that dropped the enclosing record would show up
           here and nowhere else. *)
        bad "outer witness (untouched)" outer_w (float_of_int (1000 + i))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

let run_on (dev : Device.t) : bool =
  let immutable_ok =
    run_case
      dev
      ~label:"immutable"
      ~kernel:scale_b
      ~make:(fun () ->
        let v = Vector.create_custom triple_custom n in
        for i = 0 to n - 1 do
          Vector.set v i (orig i)
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.b, p.a, p.c))
  in
  let mutable_ok =
    run_case
      dev
      ~label:"mutable"
      ~kernel:scale_mb
      ~make:(fun () ->
        let v = Vector.create_custom mtriple_custom n in
        for i = 0 to n - 1 do
          Vector.set v i (morig i)
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.mb, p.ma, p.mc))
  in
  (* The nested cases carry their own checker for ONE reason: [run_case] reports
     one target and two witnesses, and this needs three witnesses across two
     levels. They used to have a second reason — a per-backend tolerated failure
     — and that is gone with backlog-203, so both checkers now share the same
     "a refusal is always a failure" rule. *)
  let nested_ok =
    nested_case
      dev
      ~label:"nested"
      ~kernel:scale_nested
      ~make:(fun () ->
        let v = Vector.create_custom outer_custom n in
        for i = 0 to n - 1 do
          Vector.set v i {tag = float_of_int (1000 + i); mid = orig i}
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.mid.b, p.mid.a, p.mid.c, p.tag))
  in
  (* The same shape with a MUTABLE leaf. Depth 1 needed both polarities
     ([triple] and [mtriple]) and nesting needs them for the same reason: with an
     immutable leaf the pre-fix Native codegen does not BUILD, so the immutable
     case alone can never exercise the silent drop. *)
  let nested_mut_ok =
    nested_case
      dev
      ~label:"nested-mut"
      ~kernel:scale_nested_mut
      ~make:(fun () ->
        let v = Vector.create_custom mouter_custom n in
        for i = 0 to n - 1 do
          Vector.set v i {mtag = float_of_int (1000 + i); mmid = morig i}
        done ;
        v)
      ~read:(fun v i ->
        let p = Vector.get v i in
        (p.mmid.mb, p.mmid.ma, p.mmid.mc, p.mtag))
  in
  immutable_ok && mutable_ok && nested_ok && nested_mut_ok

let () =
  let devs =
    Device.init
      ~frameworks:
        ["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"; "HIP"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found — nothing asserted, and that is a gap" ;
    (* Exit non-zero: a run that asserted nothing must not read as a pass. *)
    exit 1
  end ;
  let any_failure = ref false in
  Array.iter (fun dev -> if not (run_on dev) then any_failure := true) devs ;
  (* A device class this file's header makes a claim ABOUT must not be able to
     go missing quietly.

     The header states that CUDA/PTX compiles BOTH nested kernels and produces
     correct values at both levels, "measured under ZLUDA on an RX 7900 XTX and
     on a Ryzen 9 7950X". Under `dune runtest` without ZLUDA on the loader path,
     this file enumerated 7 devices, printed ZERO CUDA/PTX rows, and exited 0 —
     the claim was not reproduced and nothing said so. The dune rule now sets
     LD_LIBRARY_PATH so the device is present where it exists; this line is what
     makes its absence visible where it does not.

     NOT a failure: a machine with no ZLUDA and no CUDA driver legitimately has
     no such device, and failing there would make the suite unrunnable rather
     than honest. It is a loud named skip, which is the difference between "the
     header's measurement was not reproduced here" and a false green.

     EVERY framework the header makes a claim about, not CUDA/PTX alone. The list
     held only CUDA/PTX while the sentence introducing it said "any framework
     this header makes a claim about", and the header makes measured claims about
     five — so on a CPU-only host the OpenCL and Vulkan claims went unreproduced
     in exactly the silence this mechanism exists to break, and the sentence was
     wider than the list under it. Each framework carries its OWN claim text,
     because one generic sentence stretched over five different measurements
     would be that same defect again, one level down.

     WHAT THE ABSENCE OF A LINE DOES AND DOES NOT MEAN. This tracks ENUMERATION
     only. No line for framework F means a device of that framework ran and every
     case asserted on it; it is not a statement about which internal path the
     backend took to get there. The claim texts are therefore written as what this
     framework is expected to DO, which since backlog-203 was fixed is the same
     thing for all five — compile both nested kernels and land the store at both
     levels, with no tolerated failure — rather than as a per-backend split that no
     longer exists. *)
  let frameworks_seen =
    Array.to_list devs |> List.map (fun (d : Device.t) -> d.Device.framework)
  in
  let claimed_frameworks =
    [
      ( "Interpreter",
        "the store lands at depth 1 and nested, the pre-fix \
         Unsupported_operation \"record field assignment\" refusal being gone"
      );
      ( "Native",
        "the store lands at depth 1 and nested rather than being dropped into \
         the temporary record Vector.get marshalled out" );
      ( "CUDA/PTX",
        "both nested kernels compile and land at both levels — put ZLUDA on \
         LD_LIBRARY_PATH, or run on a CUDA host, to exercise it" );
      ( "OpenCL",
        "both nested kernels compile and land at both levels, [outer] \
         included, which is what the backlog-203 declaration-ordering fix \
         bought here — before it, [outer] did not compile at all" );
      ( "Vulkan",
        "both nested kernels compile and land at both levels, [outer] \
         included, which is what the backlog-203 declaration-ordering fix \
         bought here — before it, [outer] did not compile at all" );
    ]
  in
  (* EVERY KEY ABOVE MUST BE A NAME A BACKEND ACTUALLY REGISTERS UNDER.

     [frameworks_seen] is built from [d.Device.framework], which is the resolved
     BACKEND name — "CUDA/PTX" — not the family name "CUDA" that
     [Device.init ~frameworks] accepts as a request. [Device.resolve_framework]
     expands a family to its registered variants ("CUDA" -> "CUDA/PTX",
     "CUDA/C"), so the two vocabularies are different and only one of them can
     appear on the left of the [List.mem] below.

     Get that wrong in either direction and this whole mechanism inverts into the
     thing it exists to prevent: a key no device can ever report makes
     [List.mem] unconditionally false, so the loud named skip fires on a run
     where the leg DID execute and pass. A skip predicate that cannot be false is
     the same defect as an assertion that cannot fail, and it is not detectable by
     reading the output — the line looks exactly like an honest skip.

     HOW IT IS CHECKED, and the version of this check that was wrong. The first
     attempt compared each key against
     [Framework_registry.all_backend_names ()], on the assumption that
     registration is independent of hardware. Measured: it is not.
     [Cuda_plugin.init ()] registers nothing when no CUDA driver loads, so
     without ZLUDA on the loader path "CUDA/PTX" is absent from the registry and
     that check turned this file's intended loud skip into a hard failure on
     every driver-less host. It replaced a false green with a false red, which is
     not an improvement.

     So the comparison is against the devices actually enumerated, in the one
     direction that carries information:

       - a key that matches an enumerated framework exactly  -> fine;
       - a key that matches NOTHING but whose FAMILY is enumerated -> FAIL, the
         key is the wrong vocabulary for a device that is right here ("CUDA" when
         the device reports "CUDA/PTX");
       - a key that matches nothing and whose family is absent -> the honest loud
         skip, which is the whole point of the list.

     Stated plainly because it bounds the check: this can only fire on a host that
     HAS the device. It cannot fire in CI, which enumerates none. It is a guard
     against the vocabulary mistake being reintroduced on a developer machine or a
     GPU runner, not a CI gate. *)
  let family fw =
    match String.index_opt fw '/' with
    | Some i -> String.sub fw 0 i
    | None -> fw
  in
  let bad_keys =
    List.filter
      (fun (fw, _) ->
        (not (List.mem fw frameworks_seen))
        && List.exists
             (fun seen -> String.equal (family seen) (family fw))
             frameworks_seen)
      claimed_frameworks
  in
  if bad_keys <> [] then begin
    List.iter
      (fun (fw, _) ->
        Printf.printf
          "claimed_frameworks key %S matches no enumerated framework, but its \
           family %S IS enumerated — the key is the wrong vocabulary (a device \
           reports the BACKEND name, not the family name Device.init accepts). \
           Enumerated: %s\n\
           %!"
          fw
          (family fw)
          (String.concat ", " (List.sort_uniq String.compare frameworks_seen)))
      bad_keys ;
    exit 1
  end ;
  List.iter
    (fun (fw, claim) ->
      if not (List.mem fw frameworks_seen) then
        Printf.printf
          "NOT MEASURED HERE: no %s device was enumerated, so the header's %s \
           claim (%s) is NOT reproduced by this run.\n\
           %!"
          fw
          fw
          claim)
    claimed_frameworks ;
  Printf.printf
    "%d device(s) exercised (frameworks: %s)\n%!"
    (Array.length devs)
    (String.concat ", " (List.sort_uniq String.compare frameworks_seen)) ;
  if !any_failure then exit 1
