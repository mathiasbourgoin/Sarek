(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * L14 — E2E test for static tag erasure (scoped tier, S1).
 *
 * A variant-typed kernel-local slot written exactly once by a literal
 * constructor and read only as a `match` scrutinee has its tag and branch
 * chain erased at code-generation time (Sarek_tag_erasure), on the device
 * path only. This test is self-verifying on two axes:
 *
 *   1. OBSERVABLE erasure: the emitted device source (CUDA-C, generated
 *      directly from the lowered IR — no device required) for the erasable
 *      kernel contains no variant tag, `switch`, or `enum`, whereas a
 *      NEGATIVE-CONTROL kernel whose constructor is genuinely runtime-selected
 *      still carries the tag/branch. This proves the pass erases when safe and
 *      stays conservative otherwise (it is not blindly stripping variants).
 *
 *   2. BEHAVIOURAL equivalence: on every available device the erasable kernels
 *      (a nullary and a unary constructor slot) compute exactly the pure-OCaml
 *      reference. The un-erased Native path (Sarek keeps the tag for native
 *      codegen) acts as an independent oracle alongside the hand-written
 *      reference.
 ******************************************************************************)

module Vector = Spoc_core.Vector
module Device = Spoc_core.Device
module Transfer = Spoc_core.Transfer

[@@@warning "-32"]

(* -11 (redundant-case): the arm-order-shadowed control kernel below is a
   deliberately redundant match (`_ -> .. | Circle r -> ..`) - exactly the
   shape that must NOT be tag-erased. OCaml's own redundancy checker is the
   first line of defense against it; suppressing the warning here lets the
   test drive the erasure pass's own guard (audit finding M7) directly. *)
[@@@warning "-11"]

let () = Test_helpers.Benchmarks.init ()

type float32 = float

(* A variant that is morally an init-time tag in these kernels: constructed
   once, never a live dispatch. *)
type shape = Circle of float32 | Square of float32 [@@sarek.type]

type flag = On | Off [@@sarek.type]

(* Erasable, unary: `let s = Circle src.(tid)` — literal constructor, matched
   twice, never used as a whole value. Result: v^2 + v with v = src.(tid). *)
let unary_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let s = Circle src.(tid) in
          let sq = match s with Circle r -> r *. r | Square x -> x in
          dst.(tid) <- (match s with Circle r -> sq +. r | Square _ -> sq)
        end]

(* Erasable, nullary: `let f = On` — the tag and branch vanish entirely.
   Result: 1.0 for On, 2.0 for Off (always On here). [src] is unused. *)
let nullary_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let f = On in
          match f with On -> dst.(tid) <- 1.0 | Off -> dst.(tid) <- 2.0
        end]

(* NEGATIVE CONTROL (compound payload): a multi-arg constructor's payload
   pattern is a tuple of binders, which the S1 reduction does not substitute —
   the arm is ineligible and the tag must be RETAINED (erasing it would drop
   the tuple binders and emit malformed code). Result: a + 2a with a = src.(tid). *)
type pair_pt = MkPair of float32 * float32 | MkOne of float32 [@@sarek.type]

let multiarg_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let s = MkPair (src.(tid), src.(tid) +. src.(tid)) in
          match s with
          | MkPair (x, y) -> dst.(tid) <- x +. y
          | MkOne x -> dst.(tid) <- x
        end]

(* NEGATIVE CONTROL (arm-order shadowing): the wildcard arm precedes the
   Circle arm, so first-match-wins makes the runtime result the WILDCARD arm
   (v +. 100.), not the Circle arm. Erasing to the Circle arm would miscompile
   to v *. v (audit finding M7). The slot must stay ineligible and the tag be
   retained; behaviour must equal the wildcard arm. *)
let shadowed_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let s = Circle src.(tid) in
          dst.(tid) <-
            (match s with _ -> src.(tid) +. 100.0 | Circle r -> r *. r)
        end]

(* NEGATIVE CONTROL: the live constructor is chosen at runtime, so the slot is
   NOT erasable and the tag/branch must be retained. *)
let retained_kirc =
  snd
    [%kernel
      fun (src : float32 vector) (dst : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_dim_x * block_idx_x) in
        if tid < n then begin
          let s =
            if tid mod 2 = 0 then Circle src.(tid) else Square src.(tid)
          in
          match s with
          | Circle r -> dst.(tid) <- r *. r
          | Square x -> dst.(tid) <- x
        end]

let ir_of name kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith ("no IR for " ^ name)

(* ---- Axis 1: observable erasure in the emitted device source ------------- *)

let contains hay needle =
  let nh = String.length needle and h = String.length hay in
  let rec go i =
    if i + nh > h then false
    else if String.sub hay i nh = needle then true
    else go (i + 1)
  in
  go 0

let mentions_tag src =
  contains src "switch (" || contains src ".tag" || contains src "enum "

let observable_ok = ref true

let check_emitted name kirc ~expect_tag =
  let src = Sarek_codegen.Sarek_ir_cuda.generate (ir_of name kirc) in
  let has = mentions_tag src in
  let ok = has = expect_tag in
  if not ok then observable_ok := false ;
  Printf.printf
    "  emitted[%s]: tag/branch %s (expected %s) -> %s\n%!"
    name
    (if has then "present" else "absent")
    (if expect_tag then "present" else "absent")
    (if ok then "OK" else "MISMATCH") ;
  if not ok then begin
    print_endline "  ---- emitted CUDA-C ----" ;
    print_endline src
  end

(* ---- Axis 2: behavioural equivalence on every device --------------------- *)

let must_pass fw =
  match fw with
  | "CUDA" | "OpenCL" | "Vulkan" | "Metal" | "Native" | "Interpreter" -> true
  | _ -> false

let any_failure = ref false

let run_behaviour ?(must = must_pass) name kirc ~reference =
  Printf.printf "runtime[%s]:\n%!" name ;
  let devs =
    Device.init
      ~frameworks:["Interpreter"; "Native"; "CUDA"; "OpenCL"; "Vulkan"]
      ()
  in
  if Array.length devs = 0 then
    print_endline "  no devices — codegen-only (behaviour skipped)"
  else
    Array.iter
      (fun dev ->
        let fw = dev.Device.framework in
        Printf.printf "  [%s] %s: %!" fw dev.Device.name ;
        try
          let n = 64 in
          let src = Vector.create Vector.float32 n in
          let dst = Vector.create Vector.float32 n in
          for i = 0 to n - 1 do
            Vector.set src i (float_of_int (i + 1)) ;
            Vector.set dst i 0.0
          done ;
          let threads = min 64 n in
          let grid_x = (n + threads - 1) / threads in
          Sarek.Execute.run_vectors
            ~device:dev
            ~block:(Sarek.Execute.dims1d threads)
            ~grid:(Sarek.Execute.dims1d grid_x)
            ~ir:(ir_of name kirc)
            ~args:
              [
                Sarek.Execute.Vec src;
                Sarek.Execute.Vec dst;
                Sarek.Execute.Int32 (Int32.of_int n);
              ]
            () ;
          Transfer.flush dev ;
          let ok = ref true in
          for i = 0 to n - 1 do
            let got = Vector.get dst i and exp = reference i in
            if abs_float (got -. exp) > 1e-2 then begin
              ok := false ;
              if i < 4 then
                Printf.printf "\n    mismatch@%d got %.3f exp %.3f%!" i got exp
            end
          done ;
          if !ok then print_endline "PASSED"
          else begin
            if must fw then any_failure := true ;
            print_endline "FAILED"
          end
        with e ->
          if must fw then any_failure := true ;
          Printf.printf
            "%s %s\n%!"
            (if must fw then "ERROR" else "SKIP (backend lacks the construct)")
            (Printexc.to_string e))
      devs

let () =
  print_endline "=== L14 static tag erasure E2E ===" ;
  print_endline "-- emitted-code snapshots --" ;
  check_emitted "unary(erasable)" unary_kirc ~expect_tag:false ;
  check_emitted "nullary(erasable)" nullary_kirc ~expect_tag:false ;
  check_emitted "runtime-selected(control)" retained_kirc ~expect_tag:true ;
  check_emitted "multiarg-payload(control)" multiarg_kirc ~expect_tag:true ;
  (* Arm-order shadowing (M7): a wildcard-first match compiles to a ternary,
     not a switch, so the crude tag heuristic does not apply. The specific
     regression to guard is erasure rewriting the match to the (unreachable)
     Circle arm [r *. r]; assert instead that the emitted device code keeps
     the wildcard arm [+ 100]. *)
  let shadowed_src =
    Sarek_codegen.Sarek_ir_cuda.generate (ir_of "shadowed" shadowed_kirc)
  in
  let shadowed_ok = contains shadowed_src "100" in
  if not shadowed_ok then observable_ok := false ;
  Printf.printf
    "  emitted[arm-order-shadowed(control)]: wildcard arm %s -> %s\n%!"
    (if shadowed_ok then "kept" else "ERASED to Circle arm")
    (if shadowed_ok then "OK" else "MISMATCH") ;
  print_endline "-- behaviour vs pure-OCaml reference --" ;
  run_behaviour "unary(erasable)" unary_kirc ~reference:(fun i ->
      let r = float_of_int (i + 1) in
      (r *. r) +. r) ;
  run_behaviour "nullary(erasable)" nullary_kirc ~reference:(fun _ -> 1.0) ;
  (* Gated to Native/Interpreter: the retained tagged match on a variant with a
     wildcard arm is a pre-existing device-codegen gap (OpenCL/Vulkan return
     generate_source None), unrelated to erasure. Native+Interpreter are the
     oracle proving erasure did not miscompile to the Circle arm. *)
  run_behaviour
    "arm-order-shadowed(control)"
    shadowed_kirc
    ~must:(fun fw -> fw = "Native" || fw = "Interpreter")
    ~reference:(fun i -> float_of_int (i + 1) +. 100.0) ;
  (* Multi-arg constructor payloads: the constructor's tuple payload is
     FLATTENED to one field per component end-to-end (Sarek_lower_ir), so it
     lines up with the multi-binder pattern and the flat [_0/_1] tagged-union
     payload every backend emits. Now executes on every backend, so the gate
     is the default (all frameworks must pass). *)
  run_behaviour "multiarg-payload(control)" multiarg_kirc ~reference:(fun i ->
      let a = float_of_int (i + 1) in
      a +. (a +. a)) ;
  Printf.printf
    "\n=== observable=%s behaviour=%s ===\n"
    (if !observable_ok then "OK" else "FAIL")
    (if !any_failure then "FAIL" else "OK") ;
  if !observable_ok && not !any_failure then
    print_endline "test_static_tag_erasure PASSED"
  else begin
    print_endline "test_static_tag_erasure FAILED" ;
    exit 1
  end
