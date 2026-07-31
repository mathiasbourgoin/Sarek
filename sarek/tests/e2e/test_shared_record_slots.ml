(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * backlog-206. A shared-memory array of a record type: `let%shared (s : tri)`
 * then a FIELD store through one slot, `s.(i).a <- e`.
 *
 * MEASURED BEFORE, on this host, 9 devices, the 4-thread kernel below
 * (`if tid = 0l then s.(tid).a <- 7.0`, want 7 0 0 0):
 *
 *   Interpreter x2  RAISED  "assignment target of .a (got unit)"
 *   Native          7 7 7 7 -- accepted the store and wrote EVERY slot
 *   CUDA/PTX x2     RAISED  "PTX codegen: unsupported construct: btype of
 *                            custom type"
 *   OpenCL x2       device compiler: "unknown type name
 *                            'Test_shared_record_slots_tri'"
 *   Vulkan x2       glslang: syntax error at the shared declaration
 *
 * Nine devices, four different answers, none of them right. Three separate
 * causes, and only the first two are the ones the backlog item named:
 *
 *  1. Native. [Sarek_cpu_runtime_types.alloc_shared_with_key] filled the array
 *     with [Array.make size default] -- ONE record in every slot -- so a field
 *     store through any index was visible through all of them. It now takes a
 *     per-slot thunk and calls [Array.init].
 *
 *  2. Interpreter. Its [EArrayCreate]/[DShared] arms mapped a record element
 *     type to [VUnit], so the store had nothing to write into. It now builds a
 *     zeroed [VRecord] per slot ([Sarek_ir_interp_value.alloc_kernel_array]).
 *
 *  3. OpenCL and Vulkan. NOT a shared-memory gap at all: the record type was
 *     never registered in the codegen types table, so the kernel declared
 *     `__local Test_..._tri s[4];` with no such struct defined above it.
 *     [Sarek_lower_ir.register_types_from_typ] ran over PARAMETER types only,
 *     and in this kernel [tri] appears nowhere but the shared declaration. The
 *     tell is that the SAME kernel with a whole-slot store (`s.(tid) <- {...}`)
 *     compiled and ran correctly on both, all along -- the record literal
 *     registered the type. That is why this file exercises BOTH shapes: a
 *     whole-slot-store-only test would have been green throughout the defect.
 *
 * CUDA/PTX still refuses, deliberately, and this file asserts the refusal
 * rather than tolerating it. PTX has no struct type, and this backend addresses
 * aggregates only in global vectors ([Sarek_ir_ptx_mem.emit_agg_elem_addr]
 * refuses shared/local aggregates on purpose). What changed there is the
 * message: it now names the array, the element type and the state space
 * instead of "unsupported construct: btype of custom type". The check below is
 * on the message content, so deleting the refusal, or replacing it with the old
 * anonymous one, fails.
 *
 * WHAT THIS FILE DOES NOT COVER, stated rather than left to be found:
 *   - Metal, HIP and WGSL. Nothing is measured and nothing is claimed, and the
 *     reason is stronger than "no such hardware here": this file initialises
 *     the Native, Interpreter, CUDA, OpenCL and Vulkan plugins ONLY, and WGSL
 *     is not even on the [Device.init] list. So no Metal, HIP or WGSL device
 *     can enumerate here on ANY host, and saying "a failure on those is a
 *     failure" would be a promise about legs this file cannot run. Naming them
 *     in [Device.init] is forward compatibility, not coverage. Covering them
 *     means linking their plugins and initialising them, which is a change to
 *     make when such hardware exists to check the result against.
 *   - Reading a shared slot that no thread wrote. Shared/__local/threadgroup
 *     memory is UNINITIALISED on every device, so an unwritten slot has no
 *     defined value. An earlier draft of this test asserted 0 for the untouched
 *     slots and read back, on real hardware, [7; 0.973611; 1210; 1] (OpenCL,
 *     RX 7900 XTX) -- correct behaviour failing a wrong assertion. Every slot
 *     this file reads is written first, by the thread that reads it or by its
 *     neighbour across a barrier. The Interpreter and Native DO zero their
 *     slots (an [Array.init] has to put something there); that is an
 *     implementation detail of the CPU backends and deliberately not asserted.
 *
 * A THIRD case covers [create_array n Local] with the same record element type.
 * That is per-thread private memory, not shared memory, so it is not the
 * backlog-206 defect -- but the Native generator built it with the same
 * [Array.make size default] one line away, so it had the same all-slots-alias
 * bug, and it is fixed and covered here rather than left for the next report.
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

(* MUTABLE fields, and it is load-bearing: the Native code generator emits a
   [setfield] for a store into a record held in a local or a shared slot, so
   with immutable fields the pre-fix build failed with "The record field a is
   not mutable" -- loud, but a build failure, not the silent all-slots write
   this file is about. This is the same split [test_record_field_store.ml] pins
   at depth 1 with [triple]/[mtriple]. The immutable half is NOT reproduced here
   because it cannot be observed in the same build as the silent half, and the
   silent half is the dangerous one. *)
type tri = {mutable a : float32; mutable b : float32; mutable c : float32}
[@@sarek.type]

let n = 4

(* FIELD stores only, no record literal anywhere: that is what makes this kernel
   the one that broke, and it is also what left [tri] unregistered for the
   struct-emitting backends.

   Every thread writes its OWN slot and then, after a barrier, reads its
   NEIGHBOUR's. So the values are defined without relying on any initial content
   of shared memory, and the read proves two things at once: the slots are
   distinct (each carries its own writer's value) and they really are shared
   across the block (a thread sees another thread's write).

   Under the pre-fix Native aliasing all four threads wrote the SAME record, so
   after the barrier every thread read the same value and the four outputs were
   equal. They cannot be equal and also be the four distinct expected values, so
   the separation is deterministic rather than a race between writers. *)
let k =
  snd
    [%kernel
      fun (src : float32 vector)
          (out_own : float32 vector)
          (out_nb : float32 vector) ->
        let%shared (s : tri) = 4l in
        let tid = thread_idx_x in
        s.(tid).a <- src.(tid) ;
        s.(tid).b <- src.(tid) +. 100.0 ;
        block_barrier () ;
        let nb = (tid + 1l) mod 4l in
        out_own.(tid) <- s.(tid).a ;
        out_nb.(tid) <- s.(nb).b]

(* The whole-slot shape, which worked on 7 of 9 devices throughout the defect.
   It is here as a REGRESSION guard, not as evidence of the fix: the thunked
   allocator and the interpreter's new per-slot record both sit on this path
   too, and a wrong per-slot default would show up here first. *)
let k_slot =
  snd
    [%kernel
      fun (src : float32 vector) (out : float32 vector) ->
        let%shared (s : tri) = 4l in
        let tid = thread_idx_x in
        s.(tid) <- {a = src.(tid); b = 0.0; c = 0.0} ;
        block_barrier () ;
        let nb = (tid + 1l) mod 4l in
        out.(tid) <- s.(nb).a]

(* PER-THREAD local array of the same record type, [create_array n Local]. It
   is not shared memory, so it is not the backlog-206 defect, but the Native
   generator built it with the same [Array.make size default] and therefore had
   the same all-slots-alias bug one line away. Thread 0's slots are the ones
   asserted; every slot read is field-stored first, for the same
   uninitialised-storage reason as above. *)
let k_local =
  snd
    [%kernel
      fun (src : float32 vector)
          (out0 : float32 vector)
          (out1 : float32 vector) ->
        let open Sarek_stdlib.Std in
        let tid = thread_idx_x in
        let arr = create_array 4l Local in
        (* Slot 2 is written whole and never read again. Its only job is to give
           the element type, which a field store does not infer
           ("Expected a record type, got 't16[0]"). It does NOT break the
           aliasing this case is looking for: [Array.make] put one record in all
           four slots, and replacing slot 2's pointer leaves 0, 1 and 3 sharing
           the original. *)
        arr.(2l) <- {a = 0.0; b = 0.0; c = 0.0} ;
        arr.(0l).a <- src.(tid) ;
        arr.(1l).a <- src.(tid) +. 10.0 ;
        out0.(tid) <- arr.(0l).a ;
        out1.(tid) <- arr.(1l).a]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* The PTX refusal, asserted on CONTENT. Three independent substrings — the
   state space, the array name and the element type name — because a message
   that merely mentions PTX would also be produced by a dozen unrelated
   failures, and because the point of the backlog-206 change to it is precisely
   that it names those three things.

   PARAMETERISED over the state space and the array name rather than fixed on
   the shared case. The first version was fixed, and the local-array case
   therefore accepted ANY CUDA/PTX exception: the named local-state-space
   refusal could have regressed to the old anonymous "btype of custom type", or
   to something unrelated, and the test would have stayed green. A refusal
   expectation that accepts any exception is a check that cannot fail. *)
let ptx_refusal_is_named ~(what : string) ~(arr : string) (msg : string) : bool
    =
  let has sub =
    let n = String.length sub and m = String.length msg in
    let rec go i = i + n <= m && (String.sub msg i n = sub || go (i + 1)) in
    n = 0 || go 0
  in
  has (what ^ "-memory array") && has ("'" ^ arr ^ "'") && has "tri"

(* [Float.abs (got -. want) > 1e-4] is FALSE when [got] is NaN, so a tolerance
   comparison on its own accepts NaN — and NaN is exactly what a read of
   uninitialised device memory can hand back, which is the failure mode this
   file exists to catch. Every value check goes through here, so the guard
   cannot be present at some call sites and missing at others. *)
let close_enough ~(got : float) ~(want : float) : bool =
  Float.is_finite got && Float.abs (got -. want) <= 1e-4

let fail_count = ref 0

let failf fmt =
  Printf.ksprintf
    (fun s ->
      incr fail_count ;
      Printf.printf "  FAILED: %s\n%!" s)
    fmt

(* CUDA/PTX must refuse, everything else must run. Keyed on the BACKEND name
   [Device.framework] reports ("CUDA/PTX"), never the family name "CUDA" that
   [Device.init ~frameworks] accepts -- a predicate written against the wrong
   vocabulary is silently always-false, which is the same defect as an assertion
   that cannot fail.

   NOT asserted anywhere in the device loop, and the earlier wording here said
   it was. What the run does is REPORT: the NOT-MEASURED-HERE list near the
   bottom prints a named line when no device reports exactly "CUDA/PTX", and
   that list is explicitly not a failure. The exactness cuts both ways — a host
   running the "CUDA/C" backend (sarek-cuda/Cuda_c_plugin.ml, not linked by this
   file) would get the printed line rather than an assertion about a device that
   did run.

   The thing that IS asserted, with no device at all, is [check_deviceless]
   below: the PTX generator is invoked directly on this kernel's IR and its
   refusal message is checked. So the PTX half of backlog-206 has a gate on
   every host, and this predicate only decides how an enumerated device is
   scored. *)
let expects_refusal (dev : Device.t) = dev.Device.framework = "CUDA/PTX"

let run_field_case (dev : Device.t) =
  Printf.printf
    "shared-record field-store [%s] %s: %!"
    dev.Device.framework
    dev.Device.name ;
  let src = Vector.create Vector.float32 n in
  let out_own = Vector.create Vector.float32 n in
  let out_nb = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set src i (float_of_int i) ;
    Vector.set out_own i (-1.0) ;
    Vector.set out_nb i (-1.0)
  done ;
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of k)
      ~args:[Vec src; Vec out_own; Vec out_nb]
      ~block:(Sarek.Execute.dims1d n)
      ~grid:(Sarek.Execute.dims1d 1)
      ()
  with
  | exception e ->
      let msg = Printexc.to_string e in
      if expects_refusal dev then
        if ptx_refusal_is_named ~what:"shared" ~arr:"s" msg then
          Printf.printf "refused, named (expected)\n%!"
        else begin
          Printf.printf "refused\n%!" ;
          failf
            "CUDA/PTX refused, as expected, but the message does not name the \
             shared array and its record type. Got: %s"
            msg
        end
      else begin
        Printf.printf "RAISED\n%!" ;
        failf "%s raised: %s" dev.Device.framework msg
      end
  | () ->
      Transfer.flush dev ;
      if expects_refusal dev then begin
        Printf.printf "ran\n%!" ;
        failf
          "CUDA/PTX accepted an aggregate shared array. If the PTX backend \
           gained aggregate state-space addressing, delete this expectation \
           and assert the values instead."
      end
      else begin
        let bad = ref false in
        for i = 0 to n - 1 do
          let got_own = Vector.get out_own i in
          let want_own = float_of_int i in
          let got_nb = Vector.get out_nb i in
          let want_nb = float_of_int ((i + 1) mod n) +. 100.0 in
          if not (close_enough ~got:got_own ~want:want_own) then begin
            bad := true ;
            failf "@%d own slot .a: got %g want %g" i got_own want_own
          end ;
          if not (close_enough ~got:got_nb ~want:want_nb) then begin
            bad := true ;
            failf "@%d neighbour slot .b: got %g want %g" i got_nb want_nb
          end
        done ;
        if !bad then Printf.printf "\n%!" else Printf.printf "OK\n%!"
      end

let run_slot_case (dev : Device.t) =
  Printf.printf
    "shared-record slot-store  [%s] %s: %!"
    dev.Device.framework
    dev.Device.name ;
  let src = Vector.create Vector.float32 n in
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set src i (float_of_int i) ;
    Vector.set out i (-1.0)
  done ;
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of k_slot)
      ~args:[Vec src; Vec out]
      ~block:(Sarek.Execute.dims1d n)
      ~grid:(Sarek.Execute.dims1d 1)
      ()
  with
  | exception e ->
      let msg = Printexc.to_string e in
      if expects_refusal dev then
        if ptx_refusal_is_named ~what:"shared" ~arr:"s" msg then
          Printf.printf "refused, named (expected)\n%!"
        else begin
          Printf.printf "refused\n%!" ;
          failf
            "CUDA/PTX refused the slot-store shape, as expected, but the \
             message does not name the shared array and its record type. Got: \
             %s"
            msg
        end
      else begin
        Printf.printf "RAISED\n%!" ;
        failf "%s raised: %s" dev.Device.framework msg
      end
  | () ->
      Transfer.flush dev ;
      if expects_refusal dev then begin
        Printf.printf "ran\n%!" ;
        failf "CUDA/PTX accepted an aggregate shared array (slot-store shape)."
      end
      else begin
        let bad = ref false in
        for i = 0 to n - 1 do
          let got = Vector.get out i in
          let want = float_of_int ((i + 1) mod n) in
          if not (close_enough ~got ~want) then begin
            bad := true ;
            failf "@%d neighbour slot .a: got %g want %g" i got want
          end
        done ;
        if !bad then Printf.printf "\n%!" else Printf.printf "OK\n%!"
      end

let run_local_case (dev : Device.t) =
  Printf.printf
    "local-array field  [%s] %s: %!"
    dev.Device.framework
    dev.Device.name ;
  let src = Vector.create Vector.float32 n in
  let out0 = Vector.create Vector.float32 n in
  let out1 = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set src i (float_of_int i) ;
    Vector.set out0 i (-1.0) ;
    Vector.set out1 i (-1.0)
  done ;
  match
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of k_local)
      ~args:[Vec src; Vec out0; Vec out1]
      ~block:(Sarek.Execute.dims1d n)
      ~grid:(Sarek.Execute.dims1d 1)
      ()
  with
  | exception e ->
      let msg = Printexc.to_string e in
      if expects_refusal dev then
        (* "local", not "shared": the local declaration site passes its own
           state-space word, and a predicate that asked for "shared" here would
           be a check that can never pass on a correct refusal. *)
        if ptx_refusal_is_named ~what:"local" ~arr:"arr" msg then
          Printf.printf "refused, named (expected)\n%!"
        else begin
          Printf.printf "refused\n%!" ;
          failf
            "CUDA/PTX refused the local array, as expected, but the message \
             does not name the local array and its record type. Got: %s"
            msg
        end
      else begin
        Printf.printf "RAISED\n%!" ;
        failf "%s raised: %s" dev.Device.framework msg
      end
  | () ->
      Transfer.flush dev ;
      if expects_refusal dev then begin
        Printf.printf "ran\n%!" ;
        failf "CUDA/PTX accepted an aggregate local array."
      end
      else begin
        let bad = ref false in
        for i = 0 to n - 1 do
          let g0 = Vector.get out0 i and g1 = Vector.get out1 i in
          let w0 = float_of_int i and w1 = float_of_int i +. 10.0 in
          if not (close_enough ~got:g0 ~want:w0) then begin
            bad := true ;
            failf "@%d local slot 0 .a: got %g want %g" i g0 w0
          end ;
          if not (close_enough ~got:g1 ~want:w1) then begin
            bad := true ;
            failf "@%d local slot 1 .a: got %g want %g" i g1 w1
          end
        done ;
        if !bad then Printf.printf "\n%!" else Printf.printf "OK\n%!"
      end

(* DEVICE-INDEPENDENT SECTION, and it runs FIRST.

   Everything below this point needs a device. CI has no GPU, so on CI the
   device loop exercises the Interpreter and Native and then PRINTS a
   NOT-MEASURED-HERE line for OpenCL, Vulkan and CUDA/PTX without failing. That
   is honest about what ran, but it means the two halves of this fix that
   affect only those backends — the record-type registration that made OpenCL
   and Vulkan compile at all, which is 4 of the 9 devices, and the named PTX
   refusal — have no gate on a GPU-less host: revert either and CI stays green.

   Both are checkable with no device, because both are properties of GENERATED
   SOURCE. The record type is either in [kern_types] or it is not; the OpenCL
   and GLSL text either declares the struct before the shared array that names
   it or it does not; the PTX generator either raises a message naming the array
   and the type or it does not. Asserted here on the same [k] the device loop
   runs, so the two cannot drift apart.

   Failures here [exit 1] immediately rather than accumulating: they mean the
   generated source is wrong, so every device row after them would be reporting
   on a kernel this file already knows is broken. *)
let contains (hay : string) (needle : string) : bool =
  let n = String.length needle and m = String.length hay in
  let rec go i = i + n <= m && (String.sub hay i n = needle || go (i + 1)) in
  n = 0 || go 0

(* Index of the first occurrence, [-1] if absent. Used to check DECLARATION
   ORDER, which a mere "both substrings are present" check cannot see — the
   struct being emitted after the array that names it is precisely how
   backlog-203 failed, and it would satisfy a presence-only assertion. *)
let index_of (hay : string) (needle : string) : int =
  let n = String.length needle and m = String.length hay in
  let rec go i =
    if i + n > m then -1
    else if String.sub hay i n = needle then i
    else go (i + 1)
  in
  go 0

let deviceless_failures = ref 0

let dfail fmt =
  Printf.ksprintf
    (fun s ->
      incr deviceless_failures ;
      Printf.printf "DEVICELESS FAILED: %s\n%!" s)
    fmt

let record_type_name = "Test_shared_record_slots.tri"

let check_deviceless () =
  let ir = ir_of k in
  (* 1. The lowerer registered the record type from the shared declaration
        alone. [tri] appears in no parameter and in no record literal in [k], so
        before the fix this list did not contain it. *)
  let names = List.map fst ir.Sarek_ir_types.kern_types in
  Printf.printf "deviceless: kern_types = [%s]\n%!" (String.concat "; " names) ;
  if not (List.mem record_type_name names) then
    dfail
      "the record type %S is not in kern_types, so no struct-emitting backend \
       can declare it. This is the OpenCL/Vulkan half of backlog-206."
      record_type_name ;
  (* 2. OpenCL and GLSL declare the struct BEFORE the shared array names it.
        The struct tag is the type name with dots replaced; rather than
        reimplementing that mangling, look for the array declaration and require
        SOME earlier occurrence of the mangled name. *)
  let mangled =
    String.map (fun c -> if c = '.' then '_' else c) record_type_name
  in
  (* STRICTLY BEFORE THE DECLARATION LINE, not merely before the array name.

     The first version of this check compared the first occurrence of the
     mangled type name against the offset of "s[4]" — and it could not fail.
     The array declaration is `__local Test_..._tri s[4];`, so the type name
     occurs ON THAT LINE, a few characters before "s[4]", whether or not any
     struct was ever defined. Measured: with the registration reverted, the
     kern_types assertion went red and this one stayed green on both dialects.

     So the anchor is the START OF THE LINE that declares the array, and the
     requirement is an occurrence of the type name before it. Deliberately not a
     dialect-specific anchor ("typedef struct" in OpenCL, "struct X {" in GLSL):
     two spellings would be two things to keep in step with two emitters, and
     the property — the name is introduced before the line that uses it — is the
     same in both. *)
  List.iter
    (fun (label, src) ->
      let arr = index_of src "s[4]" in
      if arr < 0 then
        dfail "%s source has no shared array declaration for [s]:\n%s" label src
      else begin
        let line_start =
          match String.rindex_from_opt src arr '\n' with
          | Some i -> i + 1
          | None -> 0
        in
        let before = String.sub src 0 line_start in
        let decl = index_of before mangled in
        if decl < 0 then
          dfail
            "%s uses the type %S on the shared-array declaration line but \
             nothing above that line introduces it — the struct definition is \
             missing, which is the OpenCL/Vulkan half of backlog-206:\n\
             %s"
            label
            mangled
            src
      end)
    [
      ("OpenCL", Sarek_codegen.Sarek_ir_opencl.generate ir);
      ("GLSL", Sarek_codegen.Sarek_ir_glsl.generate ir);
    ] ;
  (* 3. PTX refuses, and the refusal names the array, the state space and the
        type. Same predicate the device rows use, so the two cannot disagree
        about what "named" means. *)
  (match Sarek_codegen.Sarek_ir_ptx.generate ir with
  | exception e ->
      let msg = Printexc.to_string e in
      if ptx_refusal_is_named ~what:"shared" ~arr:"s" msg then
        Printf.printf "deviceless: PTX refused, named\n%!"
      else
        dfail
          "PTX refused but the message names neither the shared array nor its \
           record type: %s"
          msg
  | _src ->
      dfail
        "PTX generated code for an aggregate shared array. If the PTX backend \
         gained aggregate state-space addressing, delete this expectation and \
         assert the emitted code instead.") ;
  if !deviceless_failures > 0 then begin
    Printf.printf
      "%d deviceless failure(s) — not running the device cases on a kernel \
       whose generated source is already wrong\n\
       %!"
      !deviceless_failures ;
    exit 1
  end ;
  Printf.printf "deviceless: OK (no device needed for any of the above)\n%!"

let () =
  check_deviceless () ;
  let devs =
    Device.init
      ~frameworks:
        ["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"; "Metal"; "HIP"]
      ()
  in
  if Array.length devs = 0 then begin
    print_endline "No devices found — nothing asserted, and that is a gap" ;
    exit 1
  end ;
  Array.iter run_field_case devs ;
  Array.iter run_slot_case devs ;
  Array.iter run_local_case devs ;
  let frameworks_seen =
    Array.to_list devs |> List.map (fun (d : Device.t) -> d.Device.framework)
  in
  (* The claims this header makes that a deviceless (or driver-less) run cannot
     reproduce, named one by one. Absence of a line means a device of that
     framework was enumerated and every case above asserted on it. Nothing here
     is a failure: a host with no CUDA driver legitimately has no CUDA/PTX
     device, and failing there would make the suite unrunnable rather than
     honest. Keys are BACKEND names, matching [Device.framework]. *)
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
    [
      ( "Interpreter",
        "the field store lands per-slot, the pre-fix \"assignment target of .a \
         (got unit)\" refusal being gone" );
      ("Native", "the field store lands in ONE slot rather than all of them");
      ( "CUDA/PTX",
        "the aggregate shared array is refused with a message naming the array \
         and its record type — put ZLUDA on LD_LIBRARY_PATH, or run on a CUDA \
         host, to exercise it" );
      ( "OpenCL",
        "the kernel COMPILES (the record type is now registered from the \
         shared declaration) and the field store lands per-slot" );
      ( "Vulkan",
        "the kernel COMPILES (the record type is now registered from the \
         shared declaration) and the field store lands per-slot" );
    ] ;
  Printf.printf
    "%d device(s) exercised (frameworks: %s)\n%!"
    (Array.length devs)
    (String.concat ", " (List.sort_uniq String.compare frameworks_seen)) ;
  if !fail_count > 0 then begin
    Printf.printf "%d failure(s)\n%!" !fail_count ;
    exit 1
  end
