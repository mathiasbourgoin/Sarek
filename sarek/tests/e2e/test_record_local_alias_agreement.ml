(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A record bound to a LOCAL is a COPY, on every backend.
 *
 *   let e = v.(tid) in e.p <- 42.0
 *
 * The store lands in the local — it is readable back through [e] — and vector
 * storage is NOT touched. That is what all five device backends do, because
 * that is what the construct MEANS on each of them: the C-family emits a
 * struct-copy local (`l1 e = v[tid]; e.p = 42.0f;`) and Native marshals a fresh
 * OCaml record out of storage through [Vector.get]. Writing back is
 * `v.(tid) <- e`, and nothing here does that.
 *
 * This file exists because the backlog-172 field-store fix broke exactly this
 * on the INTERPRETER and no test noticed. Measured on the pre-fix tip
 * (1b18e090), with 4 elements and `let e = v.(tid) in e.p <- 42.0`:
 *
 *   Native 0 1 2 3 | OpenCL x2 0 1 2 3 | Vulkan x2 0 1 2 3 |
 *   CUDA-PTX x2 0 1 2 3 | Interpreter 42 42 42 42
 *
 * The CUDA-PTX pair is the one this file's gate could not reproduce on its own:
 * the dune rule now puts ZLUDA on LD_LIBRARY_PATH so the two CUDA/PTX devices
 * are enumerated, and the run prints a named NOT-MEASURED-HERE line for that
 * framework when they are not — rather than exiting 0 having exercised 7 of 9
 * devices with nothing to say about it. Every CUDA/PTX row here is
 * ZLUDA-on-AMD; there is no NVIDIA hardware on this machine.
 *
 * The interpreter wrote THROUGH the local into vector storage: [read_lvalue]'s
 * [LVar] arm hands back the [VRecord]'s [value array] itself, and for the
 * `v.(i).f <- e` target that sharing is exactly what makes the store land — but
 * a local binding must not inherit it. Before the fix landed, the interpreter
 * REFUSED this shape outright (`Unsupported operation 'record field
 * assignment'`), so the regression traded a loud refusal for a silent write
 * into memory five devices leave alone.
 *
 * The fix is in the binding, not in the read: [SLet]/[SLetMut] deep-copy a
 * [VRecord] as they bind it. Copying at the READ instead would make the store
 * vanish entirely, and `e.p` would read back the old value — which is NOT what
 * the other backends do either.
 *
 * Two depths, because a shallow copy passes the first and fails the second:
 *
 *   A. depth 1 — `let e = v.(tid) in e.p <- 42.0`. Runs on EVERY device.
 *   B. nested  — `let e = v.(tid) in e.sub.p <- 42.0`. Runs on every device
 *      whose emitted source can be COMPILED, which here is the two CPU backends
 *      plus CUDA/PTX. Not a per-backend exclusion: [l2] names [l1] as a field
 *      type and the record typedefs are emitted in [kern_types] order with no
 *      dependency sort (backlog-203), and for THIS kernel that order puts l2
 *      before l1 — measured from the generated source, per backend, not assumed.
 *      OpenCL and Vulkan therefore cannot build it; the same defect does not bite
 *      the sibling shapes in test_record_field_store.ml, and CUDA/PTX declares no
 *      structs at all. The wording of that compile failure is pinned in both
 *      directions by test_record_field_store.ml, which owns the predicate; this
 *      file does not duplicate it. What IS pinned here is that the nested case
 *      actually RAN on both CPU frameworks — a nested case that quietly
 *      exercised nothing would read the same as one that agreed.
 *
 * Every disagreement makes the process exit non-zero.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer
module Execute = Sarek.Execute

(* Explicit registration: linking a plugin does not enumerate its devices. *)
let () =
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  Sarek_cuda.Cuda_plugin.init () ;
  Sarek_opencl.Opencl_plugin.init () ;
  Sarek_vulkan.Vulkan_plugin.init ()

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

(* [p] is mutable because Native lowers a store into a LOCAL record to an OCaml
   setfield, which an immutable field rejects at compile time ("The record field
   p is not mutable"). The mutability is a Native codegen requirement, not part
   of what is being asserted. *)
type l1 = {mutable p : float32; q : float32} [@@sarek.type]

(* Nested: the local copy must be deep, or a store through [e.sub] reaches the
   inner record still shared with vector storage. *)
type l2 = {tag : float32; sub : l1} [@@sarek.type]

let n = 64

(* A: depth-1 local store. [out] carries the local's own field back so the store
   is checked to have LANDED, not merely to have missed storage. *)
let k_local_depth1 =
  snd
    [%kernel
      fun (v : l1 vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then begin
          let e = v.(tid) in
          e.p <- 42.0 ;
          out.(tid) <- e.p
        end]

(* B: nested local store, one level down. *)
let k_local_nested =
  snd
    [%kernel
      fun (v : l2 vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then begin
          let e = v.(tid) in
          e.sub.p <- 42.0 ;
          out.(tid) <- e.sub.p
        end]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let launch dev ir args =
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d (min 64 n))
    ~grid:(Execute.dims1d ((n + 63) / 64))
    ()

(* Reports at most three mismatches per case, then keeps counting. *)
let make_reporter () =
  let ok = ref true in
  let reported = ref 0 in
  let bad i name got want =
    if Float.abs (got -. want) > 1e-4 then begin
      ok := false ;
      if !reported < 3 then begin
        incr reported ;
        Printf.printf "\n  @%d %s: got %g want %g" i name got want
      end
    end
  in
  (ok, bad)

let case_depth1 (dev : Device.t) : bool =
  Printf.printf
    "local-alias depth1 [%s] %s: %!"
    dev.Device.framework
    dev.Device.name ;
  let v = Vector.create_custom l1_custom n in
  for i = 0 to n - 1 do
    Vector.set v i {p = float_of_int i; q = float_of_int (i + 100)}
  done ;
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set out i 0.0
  done ;
  match
    launch
      dev
      (ir_of k_local_depth1)
      [Execute.Vec v; Execute.Vec out; Execute.Int n]
  with
  | exception e ->
      (* A refusal is a FAILURE: every backend in this list runs ordinary
         custom-vector kernels, and this shape is plain DSL. *)
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok, bad = make_reporter () in
      for i = 0 to n - 1 do
        let e = Vector.get v i in
        (* The store landed IN THE LOCAL. *)
        bad i "out (local's own field after the store)" (Vector.get out i) 42.0 ;
        (* And vector storage is untouched — the regression this file exists
           for wrote 42 here. *)
        bad i "v.(i).p (storage, must be untouched)" e.p (float_of_int i) ;
        bad
          i
          "v.(i).q (storage, must be untouched)"
          e.q
          (float_of_int (i + 100))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

let case_nested (dev : Device.t) : bool =
  Printf.printf
    "local-alias nested [%s] %s: %!"
    dev.Device.framework
    dev.Device.name ;
  let v = Vector.create_custom l2_custom n in
  for i = 0 to n - 1 do
    Vector.set
      v
      i
      {
        tag = float_of_int (1000 + i);
        sub = {p = float_of_int i; q = float_of_int (i + 100)};
      }
  done ;
  let out = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set out i 0.0
  done ;
  match
    launch
      dev
      (ir_of k_local_nested)
      [Execute.Vec v; Execute.Vec out; Execute.Int n]
  with
  | exception e ->
      Printf.printf "FAILED (raised: %s)\n%!" (Printexc.to_string e) ;
      false
  | () ->
      Transfer.flush dev ;
      let ok, bad = make_reporter () in
      for i = 0 to n - 1 do
        let e = Vector.get v i in
        bad i "out (local's own nested field)" (Vector.get out i) 42.0 ;
        bad
          i
          "v.(i).sub.p (storage, must be untouched)"
          e.sub.p
          (float_of_int i) ;
        bad
          i
          "v.(i).sub.q (storage, must be untouched)"
          e.sub.q
          (float_of_int (i + 100)) ;
        bad
          i
          "v.(i).tag (storage, must be untouched)"
          e.tag
          (float_of_int (1000 + i))
      done ;
      if !ok then Printf.printf "OK\n%!" else Printf.printf "\n  FAILED\n%!" ;
      !ok

(* Whether the nested case can even be COMPILED on a device, decided from the
   emitted source rather than from the framework name.

   [l2] names [l1] as a field type, and the record typedefs are emitted in
   [kern_types] order with no dependency sort (backlog-203), so a backend that
   declares C-like structs may see the enclosing struct before the inner one.
   Measured for THIS kernel: the emitted OpenCL declares
   Test_record_local_alias_agreement_l2 before ..._l1, so OpenCL and Vulkan
   cannot compile it. That polarity — a compile failure with exactly the
   backlog-203 wording — is pinned in both directions by
   test_record_field_store.ml, which owns the compiler-wording predicate; this
   file does not duplicate it and simply does not launch what cannot build.

   Not a per-backend claim: the same file measured [mouter]/[mtriple] emitted in
   dependency order and compiling fine on both. And CUDA/PTX declares no structs
   at all, so it is NOT excused here — the nested case runs there.

   The two frameworks the nested case must have actually run on, so that a
   file-wide "not exercised" cannot read as a pass, are asserted below. *)
let struct_emitting_frameworks :
    (string
    * (types:(string * (string * Sarek_ir_types.elttype) list) list ->
      Sarek_ir_types.kernel ->
      string))
    list =
  [
    ("OpenCL", Sarek_codegen.Sarek_ir_opencl.generate_with_types);
    ( "Vulkan",
      fun ~types ir -> Sarek_codegen.Sarek_ir_glsl.generate_with_types ~types ir
    );
  ]

let nested_must_run_on = ["Interpreter"; "Native"]

let find_sub (hay : string) (needle : string) : int option =
  let nh = String.length needle and hl = String.length hay in
  if nh = 0 then Some 0
  else
    let rec go i =
      if i + nh > hl then None
      else if String.equal (String.sub hay i nh) needle then Some i
      else go (i + 1)
    in
    go 0

(* Declared as [struct Ty {...}] (GLSL) or as [typedef struct {...} Ty;]
   (OpenCL C) — both spellings, because looking only for the first reports the
   OpenCL output as in-order when it is not. *)
let uses_type_before_declaring (src : string) (ty : string) : bool =
  let decl =
    match (find_sub src ("struct " ^ ty), find_sub src ("} " ^ ty ^ ";")) with
    | Some a, Some b -> Some (min a b)
    | Some a, None -> Some a
    | None, Some b -> Some b
    | None, None -> None
  in
  match (find_sub src ty, decl) with Some use, Some d -> use < d | _ -> false

(* Both polarities of the above, over synthetic sources, with no device.

   This is a second COPY of the predicate (the two tests are separate [modules]
   stanzas and cannot share it), so it carries the same hazard the comment above
   names, and the same one the sibling file was found to have left uncovered:
   with only the GLSL arm, [nested_can_compile] answers "can compile" for an
   out-of-order OpenCL source and the nested case is launched into a compile
   failure that this file reports as a hard failure. Nothing without a live
   OpenCL device would have caught that. Case 3 is the load-bearing one, and it
   is the only one: measured by removing the typedef arm, cases 1, 2 AND 4 still
   pass and only case 3 goes red ("opencl: used before declared should be true
   and is not"). Case 4 cannot constrain that arm — with the arm gone [decl] is
   [None], the match falls to [| _ -> false], and [false] is the answer case 4
   wanted. It still rules out a constant-true predicate on this dialect, which
   is all it is for.

   Runs before any device, so a broken predicate is not discovered mid-sweep. *)
let () =
  let inner = "Test_record_local_alias_agreement_l1" in
  let outer = "Test_record_local_alias_agreement_l2" in
  let cases =
    [
      (* GLSL spelling: `struct Ty { ... };` *)
      ( true,
        "glsl: used before declared",
        Printf.sprintf
          "#version 450\n\
           struct %s {\n\
          \  float tag;\n\
          \  %s sub;\n\
           };\n\
           struct %s {\n\
          \  float p;\n\
           };\n"
          outer
          inner
          inner );
      ( false,
        "glsl: declared before used",
        Printf.sprintf
          "#version 450\n\
           struct %s {\n\
          \  float p;\n\
           };\n\
           struct %s {\n\
          \  float tag;\n\
          \  %s sub;\n\
           };\n"
          inner
          outer
          inner );
      (* OpenCL C spelling: `typedef struct { ... } Ty;` — "struct Ty" never
         appears, which is why the GLSL-only form finds no declaration and
         wrongly answers false. *)
      ( true,
        "opencl: used before declared",
        Printf.sprintf
          "typedef struct {\n\
          \  float tag;\n\
          \  %s sub;\n\
           } %s;\n\
           typedef struct {\n\
          \  float p;\n\
           } %s;\n"
          inner
          outer
          inner );
      ( false,
        "opencl: declared before used",
        Printf.sprintf
          "typedef struct {\n\
          \  float p;\n\
           } %s;\n\
           typedef struct {\n\
          \  float tag;\n\
          \  %s sub;\n\
           } %s;\n"
          inner
          inner
          outer );
      (* Used and never declared: false, and fail-closed on purpose — a dropped
         typedef is a different defect from a mis-ordered one, and
         [nested_can_compile] must not silently decline to launch because of
         it. *)
      ( false,
        "used, never declared",
        Printf.sprintf "#version 450\nstruct %s {\n  %s sub;\n};\n" outer inner
      );
      (* No struct declarations at all — the CUDA/PTX shape. *)
      (false, "no structs at all", "#version 450\nvoid main() {}\n");
    ]
  in
  let bad =
    List.filter
      (fun (want, _, src) ->
        not (Bool.equal (uses_type_before_declaring src inner) want))
      cases
  in
  if bad <> [] then begin
    List.iter
      (fun (want, label, _) ->
        Printf.printf
          "uses_type_before_declaring self-check: %s should be %b and is not\n\
           %!"
          label
          want)
      bad ;
    exit 1
  end ;
  Printf.printf
    "uses_type_before_declaring self-check: %d case(s) OK (%d gap, %d no-gap)\n\
     %!"
    (List.length cases)
    (List.length (List.filter (fun (w, _, _) -> w) cases))
    (List.length (List.filter (fun (w, _, _) -> not w) cases))

let nested_can_compile (dev : Device.t) : bool =
  match List.assoc_opt dev.Device.framework struct_emitting_frameworks with
  | None -> true
  | Some generate ->
      let ir = ir_of k_local_nested in
      let types = ir.Sarek_ir_types.kern_types in
      not
        (uses_type_before_declaring
           (generate ~types ir)
           "Test_record_local_alias_agreement_l1")

let () =
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
  let any_failure = ref false in
  let nested_ran = ref [] in
  Array.iter
    (fun (dev : Device.t) ->
      if not (case_depth1 dev) then any_failure := true ;
      if nested_can_compile dev then begin
        nested_ran := dev.Device.framework :: !nested_ran ;
        if not (case_nested dev) then any_failure := true
      end
      else
        Printf.printf
          "local-alias nested [%s] %s: not launched — the emitted source \
           declares l2 before l1, so this backend cannot compile it \
           (backlog-203, measured from the generated source; the wording of \
           that failure is pinned by test_record_field_store.ml)\n\
           %!"
          dev.Device.framework
          dev.Device.name)
    devs ;
  (* A nested case that ran on neither CPU framework would print nothing but
     "not exercised" lines and still exit 0. Require both. *)
  List.iter
    (fun fw ->
      if not (List.mem fw !nested_ran) then begin
        Printf.printf
          "nested local-alias case never ran on %s — the deep-copy half of \
           this file asserted nothing\n\
           %!"
          fw ;
        any_failure := true
      end)
    nested_must_run_on ;
  (* Same rule as test_record_field_store.ml: a device class this file's header
     makes a claim ABOUT must not be able to go missing quietly.

     The header's measurement line names CUDA-PTX x2 among the backends that
     read back [0 1 2 3], and the nested case's scope is stated as "the two CPU
     backends plus CUDA/PTX". Without ZLUDA on the loader path this file
     enumerated 7 devices, printed no CUDA/PTX row at all, and exited 0. The
     dune rule now sets LD_LIBRARY_PATH; this line reports the absence where
     there is genuinely no such device.

     A loud named skip rather than a failure, for the reason given in the
     sibling file: a host with neither ZLUDA nor a CUDA driver has no such
     device, and the honest report is "not reproduced here", not a hard stop. *)
  let frameworks_seen =
    Array.to_list devs |> List.map (fun (d : Device.t) -> d.Device.framework)
  in
  List.iter
    (fun fw ->
      if not (List.mem fw frameworks_seen) then
        Printf.printf
          "NOT MEASURED HERE: no %s device was enumerated, so the header's %s \
           rows (storage untouched at both depths, nested case included) are \
           NOT reproduced by this run. Put ZLUDA on LD_LIBRARY_PATH, or run on \
           a CUDA host, to exercise it.\n\
           %!"
          fw
          fw)
    ["CUDA/PTX"] ;
  Printf.printf
    "%d device(s) exercised (frameworks: %s)\n%!"
    (Array.length devs)
    (String.concat ", " (List.sort_uniq String.compare frameworks_seen)) ;
  if !any_failure then exit 1
