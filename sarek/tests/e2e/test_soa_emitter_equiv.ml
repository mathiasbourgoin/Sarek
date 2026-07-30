(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tier 1b device-side SoA emitter equivalence.
 *
 * Tier 1a proved the host SoA transpose + scalar transfer path. This test
 * proves the Tier 1b EMITTER: a single custom (record) vector kernel parameter
 * lowered as Structure-of-Arrays (Sarek_ir_ptx.generate ~soa_params) — N
 * per-leaf base pointers + coalesced per-leaf scalar loads — computes the same
 * result on CUDA/PTX as the default AoS lowering of the very same kernel IR,
 * and as a pure-OCaml reference.
 *
 * Mechanics: the same [%kernel] IR is compiled twice.
 *   - AoS: run via Execute.run_vectors with the single custom vector argument
 *     (backend generate_source, default packed layout).
 *   - SoA: Sarek_ir_ptx.generate ~soa_params:[<the custom vector param>] emits
 *     N pointer params + one length; the AoS host buffer is transposed into N
 *     contiguous leaf vectors (Spoc_core.Soa.scatter) and fed positionally via
 *     Execute.run_source ~inject_lengths:false — exactly the N-base-pointer ABI
 *     the emitter now produces. (The user-facing Vector.create ~layout:SoA +
 *     automatic launch expansion is Tier 1c; this drives the emitter directly.)
 *
 * SoA is PTX-only in this deliverable, so the SoA leg runs on CUDA/PTX devices
 * only; the AoS leg + reference run everywhere and are always checked. f32 and
 * f64 leaves are exercised end-to-end here; i32/i64 leaf codegen is covered at
 * the PTX-instruction + ptxas-assembly level in tests/unit/test_ptx_snapshot.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Gpu_memory = Spoc_core.Gpu_memory
module Soa = Spoc_core.Soa
module Soa_vector = Spoc_core.Soa_vector
module Soa_launch = Sarek.Soa_launch
module Benchmarks = Test_helpers.Benchmarks
open Sarek_codegen

type ('a, 'b) vector = ('a, 'b) Vector.t

type float32 = float

type float64 = float

(* Fields mutable so the round-trip kernel can write a leaf in place
   (pts.(i).y <- ...); the read legs are unaffected. *)
type point3d = {mutable x : float32; mutable y : float32; mutable z : float32}
[@@sarek.type]

type dpair = {u : float64; v : float64} [@@sarek.type]

(* f32 headline case: reads three fields of a custom vector and sums them. *)
let p3_kernel =
  snd
    [%kernel
      fun (pts : point3d vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = pts.(tid) in
          out.(tid) <- p.x +. p.y +. p.z]

(* f64 case: two 8-byte leaves. *)
let dpair_kernel =
  snd
    [%kernel
      fun (pv : dpair vector) (out : float64 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then out.(tid) <- pv.(tid).u +. pv.(tid).v]

(* Write case: scales the y leaf in place. Exercises the SoA field STORE path
   (D2H leaf readback + gather round-trip on the host side). *)
let p3_scale_y_kernel =
  snd
    [%kernel
      fun (pts : point3d vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then pts.(tid).y <- pts.(tid).y *. 2.0]

(* INTEGER leaf combos, executed on device. The Tier 1b handoff shipped these two
   at the PTX-instruction and ptxas-assembly level only and marked device
   execution "(Tier 1c)" — the row it could not fill, because it needs a CUDA/PTX
   device. ZLUDA provides one.

   Not redundant with point3d/dpair: those are uniform-width (3 x 4B, 2 x 8B),
   while these MIX widths, which is what makes the per-leaf stride and the AoS
   padding distinguishable. [mixed] is 4B then 8B (pad after the i32),
   [longpair] is 8B then 4B (trailing pad) — the two orders put the padding in
   different places. *)
type mixed = {i : int32; d : float64} [@@sarek.type]

type longpair = {p : int64; q : int32} [@@sarek.type]

(* Each leaf goes to its OWN output array, at its own width. Deliberately no
   int<->float conversion: a conversion folds both leaves into one number, and a
   per-leaf stride error could then be masked by a compensating error in the
   other leaf. Separate outputs make each leaf independently falsifiable. *)
let mixed_kernel =
  snd
    [%kernel
      fun (mv : mixed vector)
          (oi : int32 vector)
          (od : float64 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then begin
          oi.(tid) <- mv.(tid).i ;
          od.(tid) <- mv.(tid).d
        end]

let longpair_kernel =
  snd
    [%kernel
      fun (lv : longpair vector)
          (op : int64 vector)
          (oq : int32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then begin
          op.(tid) <- lv.(tid).p ;
          oq.(tid) <- lv.(tid).q
        end]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* Name of the first (custom vector) kernel parameter — what we lower as SoA. *)
let first_param_name (ir : Sarek_ir_types.kernel) =
  match ir.Sarek_ir_types.kern_params with
  | Sarek_ir_types.DParam (v, _) :: _ -> v.Sarek_ir_types.var_name
  | _ -> failwith "kernel has no parameters"

let is_ptx (dev : Device.t) = dev.Device.framework = "CUDA/PTX"

let dims threads = Sarek.Execute.dims1d threads

(* ── Guarded cases: the reason a case is skipped, and the devices it is true of ──

   A skip line is a CLAIM about a class of devices, and this file has now got
   that claim wrong twice in a row, each time while fixing the previous attempt:

     round 2: the guarded cases printed NOTHING on a non-PTX device. Silent
              no-op and pass are the same observation — skip-as-pass.
     round 3: every guarded case got a NAMED skip line. But the name was wired to
              the guard [is_ptx], and one of the two reasons — backlog-172 — is
              true only of the two CPU backends. So four cases printed
              "blocked on backlog-172" on OpenCL x2 and Vulkan x2, four devices
              where the construct works fine. 16 real assertions withheld under a
              false reason, among them the ONLY non-PTX coverage of the
              Vector.unsafe_set / Vector.fill Stale_CPU write-loss fix, which is
              host-side and not PTX-specific at all.

   The shape of both failures is the same: the guard and the reason were two
   independent statements, and nothing checked that they agreed. So a reason is
   no longer a string handed to an else-branch. It is a record that carries its
   own predicate, and the predicate is what SELECTS it — a case is skipped
   because some blocker applies, and the line printed is that blocker's. A reason
   printed on a device its predicate does not describe is now unrepresentable.

   [applies] takes the framework NAME rather than a [Device.t] so that
   {!check_skip_reason_scope} can evaluate it over a declared, host-independent
   set of framework names (see {!declared_frameworks} for what that set is and,
   as importantly, what it does not claim to be) rather than merely the backends
   plugged into the host running the suite. Both real blockers are
   framework-level facts, so nothing is lost. *)
type blocker = {
  reason : string;
      (** Printed verbatim as the skip line. Must be true of every framework in
          [describes] and of no other. *)
  applies : string -> bool;
      (** [applies framework]: is a case carrying this blocker unrunnable there?
      *)
  describes : string list;
      (** The frameworks [reason] claims to be about. Compared against [applies]
          over the whole framework universe by {!check_skip_reason_scope} — this
          is the second, independent statement, deliberately written out rather
          than derived, because a check that derives its expectation from the
          thing it checks cannot fail. *)
}

(* The universe {!check_skip_reason_scope} quantifies over: a blocker predicate
   that fires outside its [describes] on ANY member is a false claim waiting for
   that device to appear, whether or not this host has one.

   Round 4 wrote this as a hand-maintained list of six names under a comment
   asserting it was "every framework Sarek can enumerate". It was not, and the
   comment is what made that unfalsifiable. Three names were missing:

     - "CUDA/C"  — sarek-cuda/Cuda_c_plugin.ml, [let name = "CUDA/C"] ([:26]),
                   entered into the table by
                   [Framework_registry.register_backend] ([:89]).
                   [Device.ml:30-34] documents "<family>/<variant>" names as
                   first class, so "CUDA/PTX" and "CUDA/C" are two frameworks,
                   not two spellings of one.
     - "WebGPU"  — sarek/plugins/webgpu/Webgpu_plugin.ml, [let name = "WebGPU"]
                   ([:43]), likewise a [register_backend] backend ([:191]) that
                   produces devices whose [framework] field carries that name.
     - "HIP"     — sarek-hip/Hip_plugin.ml, likewise a [register_backend]
                   backend ([:86]) under [let name = "HIP"]
                   ([Hip_plugin_base.ml:17]), forced at module init on [:93]. It
                   is additionally requested by [Device.init]'s default framework
                   list ([Device.ml:41-43]) and special-cased by [Device.is_gpu]
                   ([:203]). Round 5 of this comment (and its commit message,
                   which cannot be rewritten) said HIP had "no registering
                   backend today"; that was false, and this bullet supersedes it.

   All three sit in one category, and it is the category the union below exists
   for: a [register_backend] backend whose name is absent from THIS host's table
   only because nothing forced its registration or [is_available ()] returned
   false. None of them is a name Sarek cannot produce a [Device.t] for.

   One name in [Device.init]'s default list is deliberately NOT in the floor: the
   bare family name "CUDA" ([Device.ml:43]), which [Device.is_cuda_framework]
   ([:140]) also recognises. No backend registers under it — [resolve_framework]
   ([:35-38]) expands a family name to the registered variants, so no [Device.t]
   ever carries "CUDA" in its [framework] field, and the floor is a set of names
   a device can actually have (which is what [applies] and [describes] are
   quantified over). If a backend ever did register under the bare name, the
   staleness check below would name it rather than let it pass.

   Adding them makes round 4's own wiring FAIL: [blocker_needs_soa_abi] fired on
   all three while describing none of them, so the guarantee two comments above —
   "a reason printed on a device its predicate does not describe is now
   unrepresentable" — was false of exactly the frameworks the list had dropped.

   The omissions are the symptom. The defect is a hand-maintained list that only
   a reader's diligence keeps in step with the plugin tree, so the list stops
   being the sole source: the universe is the declared floor below UNION every
   name in [Framework_registry.all_backend_names ()] at check time, and
   {!check_skip_reason_scope} additionally fails if the registry holds a name the
   floor does not declare. A backend added later joins the universe on its own,
   and is reported as a staleness failure here rather than silently widening a
   blocker's claim.

   Why the registry is not the sole source either, though that would be
   drift-proof by construction: a backend enters that table only when something
   forces its registration AND its [is_available ()] returns true on this host.
   Every plugin's [registered_backend] is [lazy], and the forcing comes in two
   shapes: at module init unless the backend is env-disabled
   ([Cuda_ptx_plugin.ml:99], [Hip_plugin.ml:93], [Webgpu_plugin.ml:195]), or only
   from an explicit [register ()] call ([Cuda_c_plugin.ml:98-99]). The table is a
   snapshot of what this host can run, not of what Sarek can name. Three
   consequences, all fatal to deriving from it: in THIS executable
   [Backend_loader.init] forces only [Cuda_plugin.init], which forces only the
   PTX plugin ([Cuda_plugin.ml:11]), so "CUDA/C" never appears in the table
   however complete the tree is; "HIP" and "WebGPU" cannot appear either, because
   module-init forcing needs the module to be LINKED and neither sarek-hip nor
   the webgpu plugin is in this executable's [libraries]; and on a CUDA-less host
   "CUDA/PTX" would drop out of the universe, leaving [describes] naming a
   framework the universe no longer contains. Deriving from the registry would shrink the
   universe to the local hardware — precisely what taking a framework NAME rather
   than a [Device.t] exists to avoid.

   So: the floor buys host-independence, the union buys anti-drift, and the
   staleness check makes a stale floor a named failure instead of a quiet gap.
   The check prints both halves, so a run shows which source contributed what
   rather than leaving the union's second half to be assumed. *)
let declared_frameworks =
  [
    "CUDA/PTX";
    "CUDA/C";
    "HIP";
    "OpenCL";
    "Vulkan";
    "Metal";
    "WebGPU";
    "Native";
    "Interpreter";
  ]

let registered_frameworks () =
  Spoc_framework_registry.Framework_registry.all_backend_names ()

let all_frameworks () =
  List.sort_uniq compare (declared_frameworks @ registered_frameworks ())

(* `v.(i).f <- e` from inside a kernel — a record-field store — is unsupported on
   the two CPU backends and ONLY those two: the Interpreter raises
   [Unsupported_operation "record field assignment"]
   (Sarek_ir_interp_eval.assign_lvalue, [LRecordField] arm), and Native accepts
   it and silently drops the store (measured: y unchanged at 1 where 2 was
   expected). Tracked as backlog-172; when it lands, this blocker is deleted and
   the cases it guards need no other change.

   It has nothing to do with CUDA/PTX. Measured 2026-07-30 on this host: all four
   cases carrying this blocker PASS on OpenCL x2 (radeonsi: RX 7900 XTX +
   7950X iGPU) and Vulkan x2 (RADV: same two), through the packed AoS fallback —
   which is the stronger assertion anyway, since the same [Vector.get] must
   return the same answer under either ABI.

   On the three frameworks in {!declared_frameworks} that no device here provides
   — CUDA/C, WebGPU, HIP — this blocker not applying is a claim about the SCOPE OF
   BACKLOG-172, which is a defect of the two CPU backends' lvalue handling, and
   not a measured pass. Those cases are skipped there by
   {!blocker_needs_soa_abi} in any case, so the distinction costs no coverage
   today; it is stated so the [describes] set is not read as evidence it is not. *)
let blocker_cpu_field_store =
  {
    reason = "SKIP (v.(i).f <- unsupported on this CPU backend: backlog-172)";
    applies = (fun fw -> fw = "Native" || fw = "Interpreter");
    describes = ["Native"; "Interpreter"];
  }

(* The SoA ABI is selected on CUDA/PTX and nowhere else. Unlike the blocker
   above this is by design and permanent, not a defect that will lift — which is
   why the two are separate records rather than one merged "not supported here".

   [Execute.ml:276] gates the SoA path on [framework = "CUDA/PTX"], a string
   equality against that one name, so the skip is semantically right on the three
   frameworks round 4's universe had omitted: CUDA/C is a DIFFERENT registered
   backend name and does not satisfy that equality, and neither do WebGPU or HIP.
   [describes] therefore widens to every declared framework except "CUDA/PTX" —
   the reason was already true of them, only unstated. It is written out in full
   rather than as [List.filter (( <> ) "CUDA/PTX") declared_frameworks] on
   purpose: derived from the universe it would agree with [applies] by
   construction and the comparison could never fail, which is the property the
   [describes] field exists to have. The cost is that a backend registered later
   fails this check until someone restates the claim; that is the alarm, not a
   defect. *)
let blocker_needs_soa_abi =
  {
    reason = "SKIP (needs CUDA/PTX: the SoA ABI dispatches nowhere else)";
    applies = (fun fw -> fw <> "CUDA/PTX");
    describes =
      [
        "CUDA/C";
        "HIP";
        "OpenCL";
        "Vulkan";
        "Metal";
        "WebGPU";
        "Native";
        "Interpreter";
      ];
  }

let all_blockers = [blocker_cpu_field_store; blocker_needs_soa_abi]

(* THE check that would have caught round 3, and it needs no device at all — so
   it runs on a CUDA-less host, and before the no-device early exit.

   For each blocker, evaluate [applies] over {!all_frameworks} and compare the
   set that fires with the set the reason claims to describe. With round 3's
   wiring the backlog-172 reason fired on OpenCL, Vulkan and Metal, none of which
   it describes, and this would have printed all three and failed.

   [describes] is also required to be a subset of the universe, so a typo
   ("Cuda/PTX") shows up as a named failure instead of quietly shrinking the
   expected set to nothing.

   Three things are printed before the verdict, because a universe that shrank —
   to the empty list, to the floor alone, or to a floor that no longer covers the
   plugin tree — would otherwise satisfy every comparison below vacuously: the
   registry names actually found at this point, the universe the comparisons run
   over, and a named failure for any registered name the floor does not declare.
   [Benchmarks.init] (the first statement of [main]) runs [Backend_loader.init],
   so the registry is populated by the time this runs; the printed list is the
   evidence, not the assumption. *)
let check_skip_reason_scope () =
  let ok = ref true in
  let registered = registered_frameworks () in
  let all_frameworks = all_frameworks () in
  Printf.printf
    "  skip-reason universe: %d declared, registry contributed [%s], \
     quantifying over [%s]\n\
     %!"
    (List.length declared_frameworks)
    (String.concat "; " (List.sort compare registered))
    (String.concat "; " all_frameworks) ;
  (* A registered backend the floor does not declare means the floor is stale:
     the union above already put it in the universe, so the blocker comparisons
     did cover it, but a name nobody wrote down is a name nobody checked the
     claims against. Fail here so it is restated deliberately. *)
  let undeclared =
    List.filter (fun fw -> not (List.mem fw declared_frameworks)) registered
  in
  if undeclared <> [] then begin
    Printf.printf
      "  registered backend(s) missing from the declared framework floor: [%s]\n\
       %!"
      (String.concat "; " (List.sort compare undeclared)) ;
    ok := false
  end ;
  if registered = [] then begin
    Printf.printf
      "  the backend registry is empty here — the union half of the universe \
       contributed nothing, so this check ran against the declared floor alone\n\
       %!" ;
    ok := false
  end ;
  List.iter
    (fun b ->
      let unknown =
        List.filter (fun fw -> not (List.mem fw all_frameworks)) b.describes
      in
      if unknown <> [] then begin
        Printf.printf
          "  skip reason %S names framework(s) Sarek cannot enumerate: [%s]\n%!"
          b.reason
          (String.concat "; " unknown) ;
        ok := false
      end ;
      let fires = List.filter b.applies all_frameworks in
      let sorted = List.sort_uniq compare in
      if sorted fires <> sorted b.describes then begin
        Printf.printf
          "  skip reason %S fires on [%s] but describes [%s]\n%!"
          b.reason
          (String.concat "; " fires)
          (String.concat "; " b.describes) ;
        ok := false
      end)
    all_blockers ;
  Printf.printf
    "  %-56s %s\n%!"
    "each skip reason is true of exactly the devices it prints on"
    (if !ok then "OK" else "FAILED") ;
  !ok

(* Run [body], or print the reason of the FIRST blocker that applies to this
   device. The printed reason is selected by the predicate, which is what keeps
   it honest — there is no else-branch holding a second opinion. *)
let guarded (dev : Device.t) ~(blockers : blocker list) ~(label : string)
    (body : unit -> bool) =
  match List.find_opt (fun b -> b.applies dev.Device.framework) blockers with
  | Some b ->
      Printf.printf "  %-56s %s\n%!" label b.reason ;
      true
  | None -> body ()

(* Launch the SoA compilation of [ir] whose first param (a flat 2/3-field record
   vector) is lowered SoA. [leaves] are the per-leaf scalar vectors (declaration
   order); [out] the scalar output. Arg order mirrors the emitted param block:
   leaf pointers, the shared length, then (out ptr, out length), then n — all
   with inject_lengths:false so we control every slot. *)
let run_soa dev ir ~leaves ~out ~n ~block ~grid =
  let ptx = Sarek_ir_ptx.generate ~soa_params:[first_param_name ir] ir in
  let leaf_args = List.map (fun v -> Sarek.Execute.Vec v) leaves in
  let len = Sarek.Execute.Int32 (Int32.of_int n) in
  let args =
    leaf_args @ [len; Sarek.Execute.Vec out; len; Sarek.Execute.Int n]
  in
  Sarek.Execute.run_source
    ~device:dev
    ~source:ptx
    ~lang:Sarek.Execute.PTX
    ~kernel_name:ir.Sarek_ir_types.kern_name
    ~block
    ~grid
    ~inject_lengths:false
    args ;
  Transfer.flush dev

(* Tier 1c: the SAME kernel driven through the real user-facing API —
   Soa_vector storage + Soa_launch.run_soa. Unlike run_soa above (which pokes
   the emitter directly), this exercises the whole host path: SoA storage
   allocation, host AoS->leaf scatter, per-leaf H2D transfer, N-base-pointer
   launch expansion, and the CUDA/PTX gate. [sv] is the SoA input vector (kernel
   param 0), [out] the scalar output, [n] the length. *)
let run_soa_via_api dev ir ~sv ~out ~n ~block ~grid =
  Soa_launch.run_soa
    ~device:dev
    ~ir
    ~args:
      [
        Soa_launch.SA_Soa sv;
        Soa_launch.SA_Reg (Sarek.Execute.Vec out);
        Soa_launch.SA_Reg (Sarek.Execute.Int n);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev

(* ---- point3d (f32) ---- *)

let run_p3 dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let src = Vector.create_custom point3d_custom n in
  for i = 0 to n - 1 do
    Vector.set
      src
      i
      {
        x = float_of_int i;
        y = (float_of_int i *. 0.5) +. 1.0;
        z = float_of_int (n - i);
      }
  done ;
  let ir = ir_of p3_kernel in
  (* AoS *)
  let out_aos = Vector.create Vector.float32 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec src; Vec out_aos; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (* SoA (PTX only) *)
  let out_soa =
    if not (is_ptx dev) then None
    else begin
      let plan =
        Soa.plan
          ~name:"point3d"
          Sarek_ir_types.[("x", TFloat32); ("y", TFloat32); ("z", TFloat32)]
      in
      let xs = Vector.create Vector.float32 n in
      let ys = Vector.create Vector.float32 n in
      let zs = Vector.create Vector.float32 n in
      Soa.scatter
        plan
        ~aos:(Vector.to_ctypes_ptr src)
        ~length:n
        ~leaves:
          [|
            Vector.to_ctypes_ptr xs;
            Vector.to_ctypes_ptr ys;
            Vector.to_ctypes_ptr zs;
          |] ;
      let out = Vector.create Vector.float32 n in
      run_soa dev ir ~leaves:[xs; ys; zs] ~out ~n ~block ~grid ;
      Some out
    end
  in
  (* SoA via the real user-facing API (Soa_vector + Soa_launch.run_soa). *)
  let out_api =
    if not (is_ptx dev) then None
    else begin
      let sv = Soa_vector.create point3d_custom n in
      for i = 0 to n - 1 do
        Soa_vector.set
          sv
          i
          {
            x = float_of_int i;
            y = (float_of_int i *. 0.5) +. 1.0;
            z = float_of_int (n - i);
          }
      done ;
      let out = Vector.create Vector.float32 n in
      run_soa_via_api dev ir ~sv ~out ~n ~block ~grid ;
      Some out
    end
  in
  let reference i =
    let p = Vector.get src i in
    p.x +. p.y +. p.z
  in
  (out_aos, out_soa, out_api, reference)

(* ---- dpair (f64) ---- *)

let run_dpair dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let src = Vector.create_custom dpair_custom n in
  for i = 0 to n - 1 do
    Vector.set
      src
      i
      {u = float_of_int i *. 1.5; v = float_of_int (n - i) -. 0.25}
  done ;
  let ir = ir_of dpair_kernel in
  let out_aos = Vector.create Vector.float64 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec src; Vec out_aos; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let out_soa =
    if not (is_ptx dev) then None
    else begin
      let plan =
        Soa.plan ~name:"dpair" Sarek_ir_types.[("u", TFloat64); ("v", TFloat64)]
      in
      let us = Vector.create Vector.float64 n in
      let vs = Vector.create Vector.float64 n in
      Soa.scatter
        plan
        ~aos:(Vector.to_ctypes_ptr src)
        ~length:n
        ~leaves:[|Vector.to_ctypes_ptr us; Vector.to_ctypes_ptr vs|] ;
      let out = Vector.create Vector.float64 n in
      run_soa dev ir ~leaves:[us; vs] ~out ~n ~block ~grid ;
      Some out
    end
  in
  let out_api =
    if not (is_ptx dev) then None
    else begin
      let sv = Soa_vector.create dpair_custom n in
      for i = 0 to n - 1 do
        Soa_vector.set
          sv
          i
          {u = float_of_int i *. 1.5; v = float_of_int (n - i) -. 0.25}
      done ;
      let out = Vector.create Vector.float64 n in
      run_soa_via_api dev ir ~sv ~out ~n ~block ~grid ;
      Some out
    end
  in
  let reference i =
    let p = Vector.get src i in
    p.u +. p.v
  in
  (out_aos, out_soa, out_api, reference)

let check name dev n runner =
  Printf.printf
    "SoA-emitter %s [%s] %s: %!"
    name
    dev.Device.framework
    dev.Device.name ;
  try
    let out_aos, out_soa, out_api, reference = runner dev n in
    let ok = ref true in
    let check_leg label o a r i =
      match o with
      | None -> ()
      | Some o ->
          let s = Vector.get o i in
          if abs_float (s -. r) > 1e-3 || abs_float (s -. a) > 1e-4 then begin
            ok := false ;
            if i < 5 then
              Printf.printf
                "\n  %s mismatch @%d: %s=%f aos=%f ref=%f%!"
                label
                i
                label
                s
                a
                r
          end
    in
    for i = 0 to n - 1 do
      let r = reference i in
      let a = Vector.get out_aos i in
      if abs_float (a -. r) > 1e-3 then begin
        ok := false ;
        if i < 5 then
          Printf.printf "\n  AoS mismatch @%d: aos=%f ref=%f%!" i a r
      end ;
      (* Direct-emitter SoA leg. *)
      check_leg "SoA" out_soa a r i ;
      (* Real user-facing API leg (Soa_vector + Soa_launch.run_soa). *)
      check_leg "SoA-API" out_api a r i
    done ;
    let soa_note =
      match out_soa with None -> " (SoA skipped: non-PTX)" | Some _ -> ""
    in
    if !ok then (
      Printf.printf "PASSED%s\n%!" soa_note ;
      true)
    else (
      Printf.printf "FAILED\n%!" ;
      false)
  with e ->
    Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e) ;
    false

(* Item 3 gate: run_soa on a non-PTX device MUST raise a located error rather
   than binding the SoA N-pointer ABI to an AoS kernel signature (which would
   read wrong data). This is the "never wrong data" guarantee, checked
   concretely on whatever non-PTX backends are present. *)
let check_gate dev =
  if is_ptx dev then true
  else begin
    Printf.printf "SoA-gate [%s] %s: %!" dev.Device.framework dev.Device.name ;
    let ir = ir_of p3_kernel in
    let sv = Soa_vector.create point3d_custom 16 in
    let out = Vector.create Vector.float32 16 in
    match
      run_soa_via_api dev ir ~sv ~out ~n:16 ~block:(dims 16) ~grid:(dims 1)
    with
    | () | (exception Not_found) ->
        Printf.printf "FAILED (run_soa did not reject a non-PTX device)\n%!" ;
        false
    | exception Sarek.Execute_error.Execution_error _ ->
        Printf.printf "rejected (located error) OK\n%!" ;
        true
  end

(* Round-trip: a kernel WRITES the y leaf on the device; then we transfer each
   leaf back (D2H) and gather into the AoS vector, and check the AoS y values.
   Exercises the leaf-writeback + Soa_vector.gather path (no other shipped test
   writes an SoA leaf). CUDA/PTX only. *)
let check_roundtrip dev n =
  if not (is_ptx dev) then true
  else begin
    Printf.printf
      "SoA-roundtrip [%s] %s: %!"
      dev.Device.framework
      dev.Device.name ;
    try
      let threads = min 128 n in
      let block = dims threads and grid = dims ((n + threads - 1) / threads) in
      let sv = Soa_vector.create point3d_custom n in
      let orig i =
        {
          x = float_of_int i;
          y = (float_of_int i *. 0.5) +. 1.0;
          z = float_of_int (n - i);
        }
      in
      for i = 0 to n - 1 do
        Soa_vector.set sv i (orig i)
      done ;
      let ir = ir_of p3_scale_y_kernel in
      Soa_launch.run_soa
        ~device:dev
        ~ir
        ~args:[Soa_launch.SA_Soa sv; Soa_launch.SA_Reg (Sarek.Execute.Int n)]
        ~block
        ~grid
        () ;
      (* Device wrote the y leaf; round-trip explicitly per the run_soa
         contract: D2H every leaf, then gather back into the AoS vector. *)
      Array.iter
        (fun (Soa_vector.Leaf v) -> Transfer.to_cpu ~force:true v)
        (Soa_vector.leaves sv) ;
      Soa_vector.gather sv ;
      let ok = ref true in
      for i = 0 to n - 1 do
        let got = Soa_vector.get sv i in
        let o = orig i in
        (* y doubled; x and z untouched. *)
        if
          abs_float (got.y -. (o.y *. 2.0)) > 1e-3
          || abs_float (got.x -. o.x) > 1e-3
          || abs_float (got.z -. o.z) > 1e-3
        then begin
          ok := false ;
          if i < 5 then
            Printf.printf
              "\n\
              \  roundtrip mismatch @%d: got {x=%f;y=%f;z=%f} expected \
               {x=%f;y=%f;z=%f}%!"
              i
              got.x
              got.y
              got.z
              o.x
              (o.y *. 2.0)
              o.z
        end
      done ;
      if !ok then (
        Printf.printf "PASSED\n%!" ;
        true)
      else (
        Printf.printf "FAILED\n%!" ;
        false)
    with e ->
      Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e) ;
      false
  end

(* ── the launch still checks the layout, on a DIFFERENT axis than create ─────
   History, because it changes what these cases are for. [Soa_vector.create] used
   to take the field layout as a [~fields] argument, and a wrong list transposed
   against the wrong byte offsets — silently corrupted data, not an error. The
   launch check below existed to catch that at the last moment before any data
   moved. [create] now DERIVES the layout from [custom_type.ir_fields], so a
   caller can no longer describe it wrongly and that particular hazard is gone at
   the source rather than intercepted here.

   These cases are still real, and still guard something [create] cannot see:
   [create] knows only the VECTOR's element type, while the launch also holds the
   KERNEL's [DParam] [TRecord]. Those are two independent declarations, and
   binding a vector of one record type to a kernel parameter of another is still
   expressible — a mismatch that no amount of deriving inside [create] can
   detect. That is the axis these cases pin, which is why they build their
   declared plans by hand instead of going through [create]: the point is
   precisely to present the check with a plan that disagrees with the kernel.

   Device-independent by construction: the check is a pure function of (param
   name, kernel element type, declared plan), so it runs on every device this
   host enumerates, CUDA/PTX or not. It is also ordered BEFORE run_soa's PTX gate
   precisely so the refusal is reachable on a NON-PTX device — behind the gate it
   could only ever fire where the gate passes, and it would then be asserting
   nothing on the 7 non-PTX devices here. *)

let xyz_ty =
  Sarek_ir_types.TRecord
    ("point3d", [("x", TFloat32); ("y", TFloat32); ("z", TFloat32)])

(* A hand-built record type that exists ONLY as a layout-mismatch fixture. Named
   [mismatch_ty] over the record "ab_pair", deliberately not "mixed": the real
   [mixed] record declared above this file's kernels is [{i : int32; d : float64}]
   and a fixture sharing its name reads as a description of it. Nothing compares
   the NAME — [Soa_launch.check_soa_layout] compares leaves and stride only
   (Soa_launch.ml:117-121) — so the rename cannot change what these cases
   assert. *)
let mismatch_ty =
  Sarek_ir_types.TRecord ("ab_pair", [("a", TInt32); ("b", TFloat64)])

(* [check_soa_layout] raises via Execute_error.raise_error; a refusal is any
   Execution_error whose rendering mentions the parameter. Asserting on the
   MESSAGE as well as the exception, because "raised something" would also be
   satisfied by an unrelated failure inside the plan builders. *)
let contains hay needle =
  let nh = String.length hay and nn = String.length needle in
  let rec go i =
    if i + nn > nh then false
    else if String.sub hay i nn = needle then true
    else go (i + 1)
  in
  nn = 0 || go 0

let refuses ~label ~param ~kernel_ty ~declared ~expect_substr =
  match Sarek.Soa_launch.check_soa_layout ~param ~kernel_ty ~declared with
  | () ->
      Printf.printf "  %-56s FAIL (accepted a mismatch)\n%!" label ;
      false
  | exception Sarek.Execute_error.Execution_error e ->
      let msg = Sarek.Execute_error.error_to_string e in
      let has_param = contains msg param in
      let has_expect = contains msg expect_substr in
      if has_param && has_expect then (
        Printf.printf "  %-56s OK (refused)\n%!" label ;
        true)
      else (
        Printf.printf
          "  %-56s FAIL (refused, but message names neither %S nor %S: %s)\n%!"
          label
          param
          expect_substr
          msg ;
        false)

(* The DERIVATION, which replaced the [~fields] argument (backlog-54 slice 1).
   [Soa_vector.create] now builds its plan from [custom_type.ir_fields], so this
   asserts the derived plan is the RIGHT one — the leaf list and stride these
   records used to be given by hand. Without it, deriving from a wrong source
   (say a reversed or truncated [ir_fields]) would still typecheck, still refuse
   nothing, and silently transpose at the wrong offsets: exactly the failure the
   argument's removal was meant to make unreachable.

   Device-independent — [create] allocates host buffers and touches no device, so
   this runs on a machine with no GPU at all. Stride is asserted alongside the
   leaves because the two are what scatter/gather index with, and a plan can have
   correct leaves with a wrong stride (padding), which would corrupt every
   element after the first. *)
let check_field_derivation () =
  let ok = ref true in
  let check_plan label (plan : Soa.plan) ~expect_leaves ~expect_stride =
    let got =
      List.map
        (fun (l : Soa.leaf) -> (l.Soa.path, l.Soa.aos_offset, l.Soa.size))
        plan.Soa.leaves
    in
    if got <> expect_leaves then (
      let show l =
        String.concat
          ", "
          (List.map (fun (p, o, s) -> Printf.sprintf "%s@%d:%d" p o s) l)
      in
      Printf.printf
        "  %-56s FAIL (leaves [%s], expected [%s])\n%!"
        label
        (show got)
        (show expect_leaves) ;
      ok := false)
    else if plan.Soa.aos_stride <> expect_stride then (
      Printf.printf
        "  %-56s FAIL (stride %d, expected %d)\n%!"
        label
        plan.Soa.aos_stride
        expect_stride ;
      ok := false)
    else Printf.printf "  %-56s OK\n%!" label
  in
  (* point3d: three 4-byte f32 leaves, packed, stride 12. *)
  check_plan
    "derived plan for point3d (3 x f32)"
    (Soa_vector.plan (Soa_vector.create point3d_custom 4))
    ~expect_leaves:[("x", 0, 4); ("y", 4, 4); ("z", 8, 4)]
    ~expect_stride:12 ;
  (* dpair: two 8-byte f64 leaves, stride 16. A different width AND a different
     leaf count, so a derivation hard-wired to point3d cannot pass both. *)
  check_plan
    "derived plan for dpair (2 x f64)"
    (Soa_vector.plan (Soa_vector.create dpair_custom 4))
    ~expect_leaves:[("u", 0, 8); ("v", 8, 8)]
    ~expect_stride:16 ;
  (* The two MIXED-width records, which point3d and dpair cannot cover: both are
     uniform-width, so neither can distinguish a correct offset from one that
     ignores alignment padding. These two put the padding in different places —
     [mixed] is 4B then 8B, so the f64 leaf is at 8 and not at 4; [longpair] is 8B
     then 4B, so the stride is 16 and not 12 (trailing pad).

     Here rather than only in [check_mixed_widths] because that case needs a
     device — a CUDA/PTX one for the SoA leg, and fp64/int64 capability for the
     leaves — while an offset or stride regression is a property of the derived
     plan alone. On a host with no GPU, and in this repository's CI, the device
     case cannot run at all, so without these two rows the padding-sensitive
     layouts have no check that executes there. *)
  check_plan
    "derived plan for mixed (i32 then f64: pad after the i32)"
    (Soa_vector.plan (Soa_vector.create mixed_custom 4))
    ~expect_leaves:[("i", 0, 4); ("d", 8, 8)]
    ~expect_stride:16 ;
  check_plan
    "derived plan for longpair (i64 then i32: trailing pad)"
    (Soa_vector.plan (Soa_vector.create longpair_custom 4))
    ~expect_leaves:[("p", 0, 8); ("q", 8, 4)]
    ~expect_stride:16 ;
  !ok

let check_layout_validation () =
  let authoritative = Soa.plan_of_elttype xyz_ty in
  let ok = ref true in
  print_endline "  --- precondition enforced at launch ---" ;
  (* POSITIVE CONTROL first. Without it, "refuses a wrong list" and "refuses
     every list" are the same observation, and the second would make SoA
     unusable rather than safe. *)
  if
    match
      Sarek.Soa_launch.check_soa_layout
        ~param:"pts"
        ~kernel_ty:(Some xyz_ty)
        ~declared:authoritative
    with
    | () -> true
    | exception e ->
        Printf.printf
          "  %-56s FAIL (rejected the CORRECT layout: %s)\n%!"
          "matching is accepted"
          (Printexc.to_string e) ;
        false
  then Printf.printf "  %-56s OK\n%!" "matching is accepted"
  else ok := false ;
  (* Wrong ORDER: same fields, same widths, permuted. The offsets move, so the
     transpose would read every field from the wrong column. *)
  if
    not
      (refuses
         ~label:"permuted is refused"
         ~param:"pts"
         ~kernel_ty:(Some xyz_ty)
         ~declared:
           (Soa.plan
              ~name:"point3d"
              [("y", Sarek_ir_types.TFloat32); ("x", TFloat32); ("z", TFloat32)])
         ~expect_substr:"wrong byte offsets")
  then ok := false ;
  (* Wrong WIDTH at the same position: f32 declared where the record has f64.
     This is the case a name-and-order-only comparison would accept. *)
  if
    not
      (refuses
         ~label:"wrong leaf WIDTH is refused"
         ~param:"m"
         ~kernel_ty:(Some mismatch_ty)
         ~declared:
           (Soa.plan
              ~name:"ab_pair"
              [("a", Sarek_ir_types.TInt32); ("b", TFloat32)])
         ~expect_substr:"wrong byte offsets")
  then ok := false ;
  (* MISSING field: fewer leaves than the record has. *)
  if
    not
      (refuses
         ~label:"missing field is refused"
         ~param:"pts"
         ~kernel_ty:(Some xyz_ty)
         ~declared:
           (Soa.plan
              ~name:"point3d"
              [("x", Sarek_ir_types.TFloat32); ("y", TFloat32)])
         ~expect_substr:"wrong byte offsets")
  then ok := false ;
  (* A SoA argument bound to a SCALAR parameter: there is no record to compare
     against, and N leaf pointers cannot bind to it. *)
  if
    not
      (refuses
         ~label:"SoA bound to a non-array param is refused"
         ~param:"scalar"
         ~kernel_ty:None
         ~declared:authoritative
         ~expect_substr:"non-array")
  then ok := false ;
  !ok

(* The WIRING, which the direct calls above cannot establish: run_soa must
   actually consult the check. Driven on ANY device, including non-PTX, and that
   is the point — the mismatch must surface the LAYOUT error rather than the
   CUDA/PTX device gate. If the check sat behind the gate (where it originally
   was) this case would report the gate message instead, so it pins the ordering
   as well as the call.

   The mismatch is now built by binding a SoA vector of the WRONG RECORD TYPE to
   the parameter, not by handing [create] a wrong field list. That is a
   deliberate change of mechanism, forced by [create] deriving its layout from
   [ir_fields]: a permuted list is no longer expressible, so the case that used
   one could no longer fail for its stated reason. What IS still expressible is
   this: [SA_Soa] is existential ([SA_Soa : 'a Soa_vector.t -> soa_arg]), so the
   type system does not relate the vector's element type to the parameter's, and
   a [dpair] vector (2 x f64, stride 16) binds to a [point3d] parameter (3 x f32,
   stride 12) without complaint. Both the leaf list and the stride disagree, so
   the launch check is what stands between that and a kernel reading garbage. *)
(* A SHORT argument list must be refused ON ARITY (backlog-182, H4).

   run_soa called B.run_source directly and never reached
   Execute.check_launch_args. That check's own message says why it matters: the
   launch sizes its device argument array from the SUPPLIED count (Cuda_api.launch:
   CArray.make (ptr void) (List.length args); Hip_api.launch likewise) while the
   driver reads the COMPILED parameter count. A short list therefore leaves the
   driver reading slots past the end of that array and dereferencing whatever it
   finds as a device address. The array is correctly sized FOR THE LIST and wrong
   FOR THE KERNEL.

   The positional loop already refused too MANY arguments at an SA_Soa position
   (List.nth_opt returning None); it said nothing about too FEW.

   ASSERTS THE MESSAGE, not merely that something was raised, and that is what
   makes this case able to fail. Pre-fix, a short list is still refused on a
   non-PTX device -- but by the CUDA/PTX gate, for an unrelated reason. Only the
   arity wording separates "refused because the call is wrong" from "refused
   because the device is wrong".

   Deliberately NOT asserted by observing a crash: an out-of-bounds read is not a
   reliable observation, and a test whose red state is a segfault is not
   evidence. *)
let check_short_arg_list dev =
  Printf.printf "SoA-arity  [%s] %s: %!" dev.Device.framework dev.Device.name ;
  let sv = Soa_vector.create point3d_custom 16 in
  let ir = ir_of p3_kernel in
  let contains hay needle =
    let n = String.length needle and m = String.length hay in
    let rec go i = i + n <= m && (String.sub hay i n = needle || go (i + 1)) in
    go 0
  in
  (* p3_kernel takes (pts, out, n) -- three parameters. Supply two. *)
  match
    Soa_launch.run_soa
      ~device:dev
      ~ir
      ~args:[Soa_launch.SA_Soa sv; Soa_launch.SA_Reg (Sarek.Execute.Int 16)]
      ~block:(dims 16)
      ~grid:(dims 1)
      ()
  with
  | () ->
      Printf.printf "FAILED (a 2-arg call to a 3-param kernel was accepted)\n%!" ;
      false
  | exception Sarek.Execute_error.Execution_error err ->
      let msg = Sarek.Execute_error.error_to_string err in
      (* Matching the count language rather than an exact string, so a reworded
         diagnostic still passes while a refusal for a DIFFERENT reason fails. *)
      if contains msg "argument" && contains msg "3" then begin
        Printf.printf "refused on arity OK\n%!" ;
        true
      end
      else begin
        Printf.printf "FAILED (refused, but not on arity: %s)\n%!" msg ;
        false
      end

(* The TRANSPARENT path (backlog-54): Soa_vector.create_transparent + the
   GENERIC Execute.run_vectors, with no SoA-specific launch entry point. This is
   what the item is actually about — the caller opts a vector into SoA storage
   and then launches normally.

   Two properties, and they need different devices to be interesting:

   - on CUDA/PTX the result must equal the AoS result (same kernel, same IR,
     N-leaf ABI instead of one packed pointer);
   - on every OTHER backend it must ALSO equal the AoS result, by silently taking
     the packed path — that is the documented "never wrong data" fallback, and
     asserting it here is the only thing that distinguishes "the fallback works"
     from "the fallback was never exercised". It would catch a soa_dispatch
     predicate that fired on the wrong backend.

     CORRECTED 2026-07-30 — this comment used to add "this machine has no NVIDIA
     device, so locally it is the fallback half that runs", which contradicted the
     header of the integer-combo cases in this same file ("it needs a CUDA/PTX
     device. ZLUDA provides one") and pointed a reader away from the arm that
     carries every fix in this file. Measured: 9 devices enumerate here and TWO of
     them are CUDA/PTX — ZLUDA on an AMD RX 7900 XTX, reached with
     LD_LIBRARY_PATH=$HOME/opt/zluda. There is no NVIDIA hardware in this host,
     which is the grain of truth the wrong half grew from, but "no NVIDIA device"
     and "no CUDA/PTX device" are not the same statement and only the second one
     is what [is_ptx] tests. BOTH halves run locally.

   Correctness is checked against the pure-OCaml reference rather than against a
   second GPU run, so a bug common to both device paths cannot hide. *)
let check_transparent dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  for i = 0 to n - 1 do
    Vector.set
      sv
      i
      {
        x = float_of_int i;
        y = (float_of_int i *. 0.5) +. 1.0;
        z = float_of_int (n - i);
      }
  done ;
  let out = Vector.create Vector.float32 n in
  let ir = ir_of p3_kernel in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec sv; Vec out; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let want =
      float_of_int i +. ((float_of_int i *. 0.5) +. 1.0) +. float_of_int (n - i)
    in
    let got = Vector.get out i in
    if Float.abs (got -. want) > 1e-3 then begin
      if !ok then
        Printf.printf
          "  transparent SoA mismatch @%d: got=%g want=%g\n%!"
          i
          got
          want ;
      ok := false
    end
  done ;
  Printf.printf
    "  %-56s %s\n%!"
    (Printf.sprintf
       "transparent SoA == reference (%s)"
       (if is_ptx dev then "PTX: N-leaf ABI" else "non-PTX: AoS fallback"))
    (if !ok then "OK" else "FAILED") ;
  !ok

(* Transparent OUTPUT round-trip: a kernel writes the y leaf and the host reads
   the result back with a plain [Vector.get] — no leaf iteration, no explicit
   gather. That is the whole claim of the transparent path, and it is a different
   claim from {!check_roundtrip}, which does the D2H and the gather BY HAND
   because [Soa_launch.run_soa] documents that as the caller's job.

   Nothing covered it before, and the gap was not cosmetic: [check_transparent]
   above writes its results into a separate plain [out] vector, so it holds the
   SoA vector INPUT-only. With the SoA ABI selected, a launch writes the N leaf
   buffers and leaves the packed AoS buffer untouched — which is precisely what
   an ordinary [Transfer.to_cpu] downloads. Every assertion above would still
   have passed while a kernel's output was silently discarded.

   Runs on every device EXCEPT the two CPU backends — see
   {!blocker_cpu_field_store}. The stronger test was always the wider one: the
   same [Vector.get] should return the same answer through the packed AoS
   fallback, and as of 2026-07-30 it is asserted there, on OpenCL x2 and Vulkan
   x2 as well as on CUDA/PTX.

   Until round 4 this was gated on [is_ptx], with the CPU-backend defect
   (backlog-172) given as the reason. The defect is real — the Interpreter
   REFUSES [pts.(tid).y <- …] (Sarek_ir_interp_eval.assign_lvalue's
   [LRecordField] arm raises [Unsupported_operation "record field assignment"])
   and Native accepts it and silently drops the store (measured: y unchanged at 1
   where 2 was expected) — but it is a defect of those two backends only, and
   using it to withhold the case from OpenCL and Vulkan withheld four real
   assertions under a reason false of all four devices. When 172 lands, delete
   the blocker; the assertion below needs no other change. *)
(* TWO launches on ONE vector, with different host data in between (H5,
   backlog-181). This is the case no existing test constructs — every other one
   allocates a fresh vector — and that is exactly why the defect was invisible to
   a green suite rather than merely untested.

   [Soa_vector.scatter] writes each leaf's host buffer through a raw ctypes
   pointer, which does no location bookkeeping. After the first launch a leaf
   sits at [Both dev], and [Transfer.to_device] short-circuits that state
   ("skip (Both)") — so the second launch ran against the FIRST launch's device
   data. Silent, and with no user workaround.

   The assertion that matters is the SECOND launch's: the first would pass either
   way. And the two rounds' inputs must not be scalar multiples of each other,
   or a stale-input result could coincide with a correct one. Round 1 doubles
   y = i+1; round 2 starts from y = 1000-i, so a stale second round yields
   4*(i+1) and a correct one 2*(1000-i) — never equal for i in range. *)
(* ONE definition of this case's label, because it is printed from two places
   that must not drift: the pass/FAILED line at the end of [check_relaunch] and
   the [guarded] SKIP line at the call site, which is printed INSTEAD of running
   the function. Two [Printf.sprintf]s meant a reword in one desynchronised the
   skip row from the pass row for the same case. *)
let relaunch_label writer_name =
  Printf.sprintf "second launch sees the second host write (%s)" writer_name

let check_relaunch dev n ~writer_name ~write ~y_expected ~stale_note =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  let ir = ir_of p3_scale_y_kernel in
  let launch () =
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir
      ~args:[Vec sv; Int n]
      ~block
      ~grid
      () ;
    Transfer.flush dev
  in
  (* The WRITER is a parameter, because the Stale_CPU write-loss is a property of
     each host writer separately and not of the SoA path (backlog-190). Vector.set
     was fixed first; unsafe_set carried the identical arm and fill's was a `| _
     -> ()` catch-all, wider still. One case per writer, so a fix to one cannot
     make the others look covered -- which is exactly how the class survived the
     first fix. *)
  let fill y_of = write sv n y_of in
  (* Round 1 — passes with or without the fix; it is here to establish the
     post-launch leaf state that the bug depends on. *)
  fill (fun i -> float_of_int (i + 1)) ;
  launch () ;
  (* Round 2 — the real assertion. [Vector.set] below reads the results back
     first (auto-sync -> leaf D2H -> gather), so this also exercises the write
     path landing on top of a device-authoritative vector. *)
  fill (fun i -> float_of_int (1000 - i)) ;
  launch () ;
  let ok = ref true in
  for i = 0 to n - 1 do
    let want = y_expected i in
    let got = (Vector.get sv i).y in
    if Float.abs (got -. want) > 1e-3 then begin
      if !ok then
        Printf.printf
          "  relaunch mismatch @%d: got=%g want=%g (stale would be %g)\n%!"
          i
          got
          want
          (stale_note i) ;
      ok := false
    end
  done ;
  Printf.printf
    "  %-56s %s\n%!"
    (relaunch_label writer_name)
    (if !ok then "OK" else "FAILED") ;
  !ok

let check_transparent_roundtrip dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  let orig i =
    {
      x = float_of_int i;
      y = (float_of_int i *. 0.5) +. 1.0;
      z = float_of_int (n - i);
    }
  in
  for i = 0 to n - 1 do
    Vector.set sv i (orig i)
  done ;
  let ir = ir_of p3_scale_y_kernel in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec sv; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let ok = ref true in
  for i = 0 to n - 1 do
    (* Plain host read. No Soa_vector.leaves, no Soa_vector.gather. *)
    let got = Vector.get sv i in
    let o = orig i in
    if
      Float.abs (got.y -. (o.y *. 2.0)) > 1e-3
      || Float.abs (got.x -. o.x) > 1e-3
      || Float.abs (got.z -. o.z) > 1e-3
    then begin
      if !ok then
        Printf.printf
          "  transparent round-trip mismatch @%d: got {x=%g;y=%g;z=%g} want \
           {x=%g;y=%g;z=%g}\n\
           %!"
          i
          got.x
          got.y
          got.z
          o.x
          (o.y *. 2.0)
          o.z ;
      ok := false
    end
  done ;
  Printf.printf
    "  %-56s %s\n%!"
    (Printf.sprintf
       "transparent SoA output read back (%s)"
       (* The label named the PTX mechanism unconditionally while the case now
          runs on OpenCL and Vulkan too, where the round trip goes through the
          packed AoS buffer. Same claim, two mechanisms — and naming the wrong
          one is the same species of false line as a skip reason that does not
          describe its device. *)
       (if is_ptx dev then "PTX: leaf D2H + gather"
        else "non-PTX: packed AoS fallback"))
    (if !ok then "OK" else "FAILED") ;
  !ok

(* SoA ABI, then PACKED AoS ABI, on ONE vector, with NO host read-back in
   between.

   [soa_leaves_live] was only ever set — nothing cleared it. So once a transparent
   CUDA/PTX launch had run, EVERY later read-back on that vector followed the
   leaves for the rest of its life, no matter which ABI the most recent launch
   actually used. [Execute.run_source] deliberately keeps the packed ABI (it hands
   the backend a source string Sarek did not emit), which makes this sequence
   reachable through the public API with no unusual step.

   The second launch is the assertion; the first would pass either way. And the
   missing read-back in between is the whole point — inserting one would clear the
   staleness by accident and the case would pass unfixed.

   The packed source is the SAME IR compiled WITHOUT ~soa_params rather than
   hand-written PTX, so this case cannot pass or fail for a PTX-authoring reason.

   Both expectations are distinguishable: y0 = i+1, each launch doubles y, so the
   correct answer is 4*(i+1) and following the stale leaves gives 2*(i+1) —
   never equal. Pre-fix the failure is in fact LOUDER than that (the packed
   launch has no buffer to bind, because the transparent path never allocates
   one), which is why the exception is caught and reported rather than left to
   abort the binary: a red state that is a crash is not an observation. *)
(* A HOST WRITE between a transparent launch and a packed one must survive.

   Same two launches as {!check_soa_then_packed}, with one statement added
   between them, and that statement is the whole case: after the transparent
   launch the vector is [Stale_CPU dev] with the leaves authoritative, and
   [Vector.set] gathers them, writes the element, and records [Stale_GPU dev] —
   the host copy is now the NEWER one while [soa_leaves_live] still says
   "leaves".

   The packed launch must then normalise that flag WITHOUT re-gathering. A gather
   there replays the leaves over the host write and the write vanishes with no
   diagnostic. Found by review of this round's first attempt, which made the
   gather unconditional for the opposite failure (a gather skipped on a
   device-authoritative location strands the launch output in the leaves) — the
   two are one condition, on which copy is authoritative, and the case exists
   because pinning only one of them is what let the other in.

   Distinguishable at EVERY index, which took a second attempt. y0 = i+1; the
   transparent launch doubles y to 2*y0; the host write sets y := i + 0.5; the
   packed launch doubles that to 2i+1. Discarding the host write gives 4*y0 =
   4i+4 instead. 2i+1 = 4i+4 has the single solution i = -1.5, so the two never
   coincide on an index — whereas the first version of this case wrote y := 100+i
   and 2*(100+i) = 4*(i+1) holds at i = 98, inside the n = 1024 range. It would
   still have failed (at i = 0, and the loop checks every index), but one row of
   the check was asserting nothing.

   The half-integer is also what makes the write independent of the leaf value in
   the ONE way that matters here: 2i+1 is odd and 4i+4 is a multiple of 4. *)
let check_host_write_survives_packed dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  let y0 i = float_of_int (i + 1) in
  for i = 0 to n - 1 do
    Vector.set sv i {x = float_of_int i; y = y0 i; z = float_of_int (n - i)}
  done ;
  let ir = ir_of p3_scale_y_kernel in
  let written i = float_of_int i +. 0.5 in
  let ok = ref true in
  (match
     Sarek.Execute.run_vectors
       ~device:dev
       ~ir
       ~args:[Vec sv; Int n]
       ~block
       ~grid
       () ;
     Transfer.flush dev ;
     (* The host write. With auto-sync on (the default, and this suite does not
        disable it) [Vector.set] gathers the leaves first, so what it lands on top
        of is the transparent launch's result, and it leaves the vector
        [Stale_GPU dev]. That precondition is named because it is load-bearing:
        with auto-sync OFF the gather does not happen and the vector stays
        [Stale_CPU dev] with a pending host write, which the packed launch then
        replays the leaves over. That is a pre-existing consequence of disabling
        auto-sync — undocumented in [Vector.ml], so stated here rather than cited
        — and NOT what this case covers. *)
     for i = 0 to n - 1 do
       Vector.set
         sv
         i
         {x = float_of_int i; y = written i; z = float_of_int (n - i)}
     done ;
     (* PACKED launch: run_source defaults to ~soa_abi:false. *)
     Sarek.Execute.run_source
       ~device:dev
       ~source:(Sarek_ir_ptx.generate ir)
       ~lang:Sarek.Execute.PTX
       ~kernel_name:ir.Sarek_ir_types.kern_name
       ~block
       ~grid
       [Sarek.Execute.Vec sv; Sarek.Execute.Int n] ;
     Transfer.flush dev
   with
  | exception e ->
      Printf.printf
        "  host write then packed launch raised: %s\n%!"
        (Printexc.to_string e) ;
      ok := false
  | () ->
      let first = ref true in
      for i = 0 to n - 1 do
        let got = (Vector.get sv i).y and want = written i *. 2.0 in
        if Float.abs (got -. want) > 1e-3 && !first then begin
          first := false ;
          Printf.printf
            "  host write lost @%d: got=%g want=%g (re-gathered leaves would \
             give %g)\n\
             %!"
            i
            got
            want
            (y0 i *. 4.0) ;
          ok := false
        end
      done) ;
  Printf.printf
    "  %-56s %s\n%!"
    "a host write between an SoA and a packed launch survives"
    (if !ok then "OK" else "FAILED") ;
  !ok

let check_soa_then_packed dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  let y0 i = float_of_int (i + 1) in
  for i = 0 to n - 1 do
    Vector.set sv i {x = float_of_int i; y = y0 i; z = float_of_int (n - i)}
  done ;
  let ir = ir_of p3_scale_y_kernel in
  let outcome =
    match
      (* Launch 1 — TRANSPARENT. run_vectors passes ~soa_abi:true, so on CUDA/PTX
         this binds the N-leaf ABI, writes the y LEAF, and leaves the vector
         SoA-owned. *)
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:[Vec sv; Int n]
        ~block
        ~grid
        () ;
      Transfer.flush dev ;
      (* Launch 2 — PACKED. run_source defaults to ~soa_abi:false. *)
      Sarek.Execute.run_source
        ~device:dev
        ~source:(Sarek_ir_ptx.generate ir)
        ~lang:Sarek.Execute.PTX
        ~kernel_name:ir.Sarek_ir_types.kern_name
        ~block
        ~grid
        [Sarek.Execute.Vec sv; Sarek.Execute.Int n] ;
      Transfer.flush dev
    with
    | () -> None
    | exception e -> Some (Printexc.to_string e)
  in
  let ok = ref true in
  (match outcome with
  | Some msg ->
      Printf.printf "  packed launch after a transparent one raised: %s\n%!" msg ;
      ok := false
  | None ->
      for i = 0 to n - 1 do
        let got = (Vector.get sv i).y in
        let want = y0 i *. 4.0 in
        if Float.abs (got -. want) > 1e-3 then begin
          if !ok then
            Printf.printf
              "  SoA->packed mismatch @%d: got=%g want=%g (stale leaves would \
               be %g)\n\
               %!"
              i
              got
              want
              (y0 i *. 2.0) ;
          ok := false
        end
      done) ;
  Printf.printf
    "  %-56s %s\n%!"
    "packed AoS launch after a transparent SoA one"
    (if !ok then "OK" else "FAILED") ;
  !ok

(* Freeing the device buffers must not throw the results away.

   [Transfer.free_buffer] and [free_all_buffers] read a vector back before
   releasing its memory — but they called [copy_device_to_host] DIRECTLY, so
   they had no SoA arm. After a transparent launch there is no packed buffer at
   all, so [free_all_buffers] raised "no device buffer to transfer from" from
   inside cleanup, and [free_buffer]'s [get_buffer = None -> ()] early return
   discarded the output silently.

   No explicit [to_cpu] before the free, deliberately: that is the sequence with
   the defect, and adding one would make this case pass unfixed. Both a raised
   exception and wrong data are reported the same way, so either red state is an
   observation rather than an abort. *)
let check_free_preserves_soa dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  let y0 i = float_of_int (i + 1) in
  for i = 0 to n - 1 do
    Vector.set sv i {x = float_of_int i; y = y0 i; z = float_of_int (n - i)}
  done ;
  let ir = ir_of p3_scale_y_kernel in
  let outcome =
    match
      Sarek.Execute.run_vectors
        ~device:dev
        ~ir
        ~args:[Vec sv; Int n]
        ~block
        ~grid
        () ;
      Transfer.flush dev ;
      Transfer.free_all_buffers sv
    with
    | () -> None
    | exception e -> Some (Printexc.to_string e)
  in
  let ok = ref true in
  (match outcome with
  | Some msg ->
      Printf.printf "  free_all_buffers raised: %s\n%!" msg ;
      ok := false
  | None ->
      for i = 0 to n - 1 do
        let got = (Vector.get sv i).y in
        let want = y0 i *. 2.0 in
        if Float.abs (got -. want) > 1e-3 then begin
          if !ok then
            Printf.printf
              "  free-then-read mismatch @%d: got=%g want=%g (pre-launch host \
               value is %g)\n\
               %!"
              i
              got
              want
              (y0 i) ;
          ok := false
        end
      done) ;
  Printf.printf
    "  %-56s %s\n%!"
    "free_all_buffers preserves a transparent SoA result"
    (if !ok then "OK" else "FAILED") ;
  !ok

(* Freeing must also RELEASE something. Correctness and reclamation are two
   different claims, and {!check_free_preserves_soa} above only makes the first:
   it proves the data survives the free. It passed while the free released ZERO
   bytes.

   Under this ABI the AoS vector never gets a packed buffer, so
   [Transfer.free_all_buffers] iterated an EMPTY [device_buffers] table and
   returned successfully having freed nothing. Measured with
   [Gpu_memory.usage()] at n=32: 3840 B before the launch, 4224 B after the
   free — a delta of +384 = 32 x 3 leaves x 4 B, i.e. exactly the memory the call
   was asked to release still held.

   Not a correctness bug, which is why nothing caught it: every leaf carries a
   [Gpu_memory.register_finalizer], so the bytes come back once the structure
   becomes unreachable. What the caller lost is the only thing an explicit free
   offers over the GC — releasing at a moment it chooses, while the vector is
   still live. So this case keeps [sv] reachable across the measurement
   ([Sys.opaque_identity] below): let it be collected and the finalizer frees the
   leaves for us and the assertion passes on the unfixed code.

   [allocated > 0] is asserted, not assumed. Without it a launch that allocated
   nothing would give released = allocated = 0 and the case would pass while
   checking nothing at all. *)
let check_free_releases_leaves dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  for i = 0 to n - 1 do
    Vector.set
      sv
      i
      {x = float_of_int i; y = float_of_int (i + 1); z = float_of_int (n - i)}
  done ;
  (* Flush finalizers pending from earlier cases BEFORE the baseline, so their
     bytes are not counted against this one. *)
  Gc.full_major () ;
  let before = Gpu_memory.usage () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir:(ir_of p3_scale_y_kernel)
    ~args:[Vec sv; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let after_launch = Gpu_memory.usage () in
  Transfer.free_all_buffers sv ;
  let after_free = Gpu_memory.usage () in
  (* Load-bearing: see the header. A collected [sv] frees its own leaves. *)
  ignore (Sys.opaque_identity sv) ;
  let allocated = after_launch - before in
  let released = after_launch - after_free in
  let ok = allocated > 0 && released >= allocated in
  if not ok then
    Printf.printf
      "  free_all_buffers released %d of the %d bytes the launch allocated \
       (usage: %d before -> %d after launch -> %d after free)\n\
       %!"
      released
      allocated
      before
      after_launch
      after_free ;
  Printf.printf
    "  %-56s %s\n%!"
    "free_all_buffers RELEASES a transparent SoA vector's leaves"
    (if ok then "OK" else "FAILED") ;
  ok

(* [free_buffer] — the SINGLE-device free — must leave the vector in a coherent
   state, and nothing covered it: every case above frees through
   [free_all_buffers]. That gap is why the incoherence shipped, so this case
   exists as much for the coverage as for the assertion.

   [free_all_buffers] escapes the bug by assigning [CPU] unconditionally at the
   end. [free_buffer] kept its location reset INSIDE the [get_buffer] [Some buf]
   arm, and under this ABI there is never a packed buffer — so the [None -> ()]
   arm was taken ALWAYS, after the leaves had already been released. The vector
   came back with its device memory gone and [location] still [Both dev].

   Three assertions, because "coherent" is three separate observable claims and
   the location value itself is the least of them:

   1. [location = CPU]. The direct statement. Checked first so a failure names
      the cause rather than a downstream symptom.
   2. [to_cpu ~force:true] must not raise. [Both dev] + [force] means "read the
      device back"; with the leaves gone there is nothing to read and
      [copy_device_to_host] raises [Failure "to_cpu: no device buffer to transfer
      from"] — on a vector whose data is intact in host storage. The data is
      re-checked after the call, so a [to_cpu] that returns quietly having
      overwritten the host copy is not mistaken for a pass.
   3. [to_device] on the same device must ALLOCATE. This is the one that matters
      most: with [location] left at [Both dev], [to_device] logs "skip (Both)"
      and returns having allocated nothing — reinstating the very [skip (Both)]
      short-circuit the [Stale_CPU]/scatter work earlier in this branch was
      written to eliminate. Asserted in BYTES via [Gpu_memory.usage()], because
      "did not raise" is exactly what the broken version also does.

   [allocated_before_free > 0] is asserted too: without it a launch that
   allocated nothing would make claim 3 vacuous. *)
let check_free_buffer_coherent dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let sv = Soa_vector.create_transparent point3d_custom n in
  let y0 i = float_of_int (i + 1) in
  for i = 0 to n - 1 do
    Vector.set sv i {x = float_of_int i; y = y0 i; z = float_of_int (n - i)}
  done ;
  Gc.full_major () ;
  let before = Gpu_memory.usage () in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir:(ir_of p3_scale_y_kernel)
    ~args:[Vec sv; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let after_launch = Gpu_memory.usage () in
  Transfer.free_buffer sv dev ;
  let ok = ref true in
  let fail fmt =
    Printf.ksprintf
      (fun s ->
        Printf.printf "  %s\n%!" s ;
        ok := false)
      fmt
  in
  if after_launch - before <= 0 then
    fail
      "the launch allocated %d bytes, so the re-upload claim below would be \
       vacuous"
      (after_launch - before) ;
  (* Claim 1. *)
  (match sv.Vector.location with
  | Vector.CPU -> ()
  | Vector.GPU d ->
      fail "location after free_buffer: GPU %d, want CPU" d.Device.id
  | Vector.Both d ->
      fail
        "location after free_buffer: Both %d, want CPU — the leaves were freed \
         and the location still names the device"
        d.Device.id
  | Vector.Stale_CPU d ->
      fail "location after free_buffer: Stale_CPU %d, want CPU" d.Device.id
  | Vector.Stale_GPU d ->
      fail "location after free_buffer: Stale_GPU %d, want CPU" d.Device.id) ;
  (* Claim 2. [first_mismatch] is per-CLAIM, not the function-wide [ok]: gating
     the print on [!ok] made claim 1 (or claim 3) failing suppress every index
     line here and in claim 5, so the case reported FAILED naming a claim while
     printing no evidence for the claim that actually broke. One ref per loop
     keeps "print only the first index" without letting one claim silence
     another. *)
  let first_mismatch = ref true in
  (match Transfer.to_cpu ~force:true sv with
  | () ->
      for i = 0 to n - 1 do
        let got = (Vector.get sv i).y and want = y0 i *. 2.0 in
        if Float.abs (got -. want) > 1e-3 && !first_mismatch then (
          first_mismatch := false ;
          fail
            "to_cpu after free_buffer corrupted @%d: got=%g want=%g \
             (pre-launch host value is %g)"
            i
            got
            want
            (y0 i))
      done
  | exception e ->
      fail
        "to_cpu ~force:true after free_buffer raised: %s"
        (Printexc.to_string e)) ;
  (* Claim 3. *)
  let before_reupload = Gpu_memory.usage () in
  (match Transfer.to_device sv dev with
  | () ->
      let reallocated = Gpu_memory.usage () - before_reupload in
      if reallocated <= 0 then
        fail
          "to_device after free_buffer allocated %d bytes — it took the \
           skip-when-already-resident short-circuit on a vector with no device \
           memory left"
          reallocated
  | exception e ->
      fail "to_device after free_buffer raised: %s" (Printexc.to_string e)) ;
  (* Claims 4 and 5. Round 3 verified both BY HAND against its own version of
     [free_buffer]; this round restructured that function's control flow — all
     four steps moved out of the [get_buffer] [Some] arm — so a hand check of the
     old shape says nothing about the new one. Asserted here instead.

     4. Double free. The second call must be a no-op, not a double release: the
        leaves are gone, the packed buffer is gone, and [location] is already
        [CPU], so every step must decline. Step 4 in particular must not fire on
        a [CPU] location.
     5. Free, then relaunch. [location = CPU] is what makes this reachable at
        all — the scatter + upload has to run again from host storage — and the
        result must be the SECOND launch's, not the first's. y is doubled twice
        from the same host values, so a relaunch that reused stale device data
        would give 2*y0 where 4*y0 is expected. *)
  (match Transfer.free_buffer sv dev with
  | () -> ()
  | exception e -> fail "second free_buffer raised: %s" (Printexc.to_string e)) ;
  (match
     Sarek.Execute.run_vectors
       ~device:dev
       ~ir:(ir_of p3_scale_y_kernel)
       ~args:[Vec sv; Int n]
       ~block
       ~grid
       () ;
     Transfer.flush dev
   with
  | () ->
      (* Claim 5's own first-mismatch ref, for the reason given at claim 2. *)
      let first_mismatch = ref true in
      for i = 0 to n - 1 do
        (* Round 1 doubled y0 to 2*y0 and [to_cpu] above gathered it into host
           storage, so this second launch doubles that to 4*y0. Stale device data
           would leave 2*y0 — never equal for y0 > 0. *)
        let got = (Vector.get sv i).y and want = y0 i *. 4.0 in
        if Float.abs (got -. want) > 1e-3 && !first_mismatch then (
          first_mismatch := false ;
          fail
            "relaunch after free_buffer is wrong @%d: got=%g want=%g (the \
             first launch's result is %g)"
            i
            got
            want
            (y0 i *. 2.0))
      done
  | exception e ->
      fail "relaunch after free_buffer raised: %s" (Printexc.to_string e)) ;
  Transfer.free_all_buffers sv ;
  (* Load-bearing, as in {!check_free_releases_leaves}: a collected [sv] frees
     its own leaves through the finalizer and the byte deltas above stop meaning
     what they say. *)
  ignore (Sys.opaque_identity sv) ;
  Printf.printf
    "  %-56s %s\n%!"
    "free_buffer leaves the vector coherent (location/re-read/re-upload)"
    (if !ok then "OK" else "FAILED") ;
  !ok

(* The two integer-leaf device rows the handoff left open. Driven through the
   TRANSPARENT path, so one case covers both the mixed-width leaf addressing and
   the generic dispatch. Each leaf is compared to the reference independently. *)
let check_mixed_widths dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let ok = ref true in
  let report label good =
    Printf.printf
      "  %-56s %s\n%!"
      (Printf.sprintf
         "%s (%s)"
         label
         (if is_ptx dev then "PTX: N-leaf ABI" else "non-PTX: AoS fallback"))
      (if good then "OK" else "FAILED") ;
    if not good then ok := false
  in
  (* i32 + f64. Gated on the DEVICE's fp64 capability rather than on
     "CUDA/PTX only": the f64 leaf makes this an fp64 kernel, and the launch
     gate refuses it wherever the driver does not ADVERTISE double precision.

     Note what the skip does and does not mean. On this host the two OpenCL
     devices are rusticl, which does not expose cl_khr_fp64 unless
     RUSTICL_FEATURES=fp64 is set — measured both ways with clinfo and with this
     test, which goes from 5 OK + 2 SKIP to 7 OK with the variable. The RX 7900
     XTX has fp64 in hardware. So a skip here reports a DRIVER CONFIGURATION,
     never a hardware limitation, and the gate is still correct: emitting an fp64
     kernel against a driver that has not enabled the extension is exactly what
     it must refuse. A CPU backend evaluates in OCaml doubles and is always
     able. *)
  let f = dev.Device.framework in
  let can_f64 = f = "Native" || f = "Interpreter" || Device.allows_fp64 dev in
  if not can_f64 then
    Printf.printf
      "  %-56s SKIP (device reports no fp64)\n%!"
      "SoA i32+f64 leaves == reference"
  else begin
    let mv = Soa_vector.create_transparent mixed_custom n in
    for k = 0 to n - 1 do
      Vector.set mv k {i = Int32.of_int (k * 3); d = float_of_int k *. 0.25}
    done ;
    let oi = Vector.create Vector.int32 n in
    let od = Vector.create Vector.float64 n in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of mixed_kernel)
      ~args:[Vec mv; Vec oi; Vec od; Int n]
      ~block
      ~grid
      () ;
    Transfer.flush dev ;
    let good = ref true in
    for k = 0 to n - 1 do
      if Int32.to_int (Vector.get oi k) <> k * 3 then good := false ;
      if Float.abs (Vector.get od k -. (float_of_int k *. 0.25)) > 1e-12 then
        good := false
    done ;
    report "SoA i32+f64 leaves == reference" !good
  end ;
  (* i64 + i32. No f64 leaf, but a 64-bit INTEGER one, which is its own
     device-optional capability (shaderInt64 on Vulkan) — same reasoning, same
     shape of gate. *)
  let can_i64 = f = "Native" || f = "Interpreter" || Device.allows_int64 dev in
  if not can_i64 then
    Printf.printf
      "  %-56s SKIP (device reports no int64)\n%!"
      "SoA i64+i32 leaves == reference"
  else begin
    let lv = Soa_vector.create_transparent longpair_custom n in
    for k = 0 to n - 1 do
      Vector.set
        lv
        k
        {p = Int64.of_int ((k * 1000003) + 7); q = Int32.of_int (k - 5)}
    done ;
    let op = Vector.create Vector.int64 n in
    let oq = Vector.create Vector.int32 n in
    Sarek.Execute.run_vectors
      ~device:dev
      ~ir:(ir_of longpair_kernel)
      ~args:[Vec lv; Vec op; Vec oq; Int n]
      ~block
      ~grid
      () ;
    Transfer.flush dev ;
    let good = ref true in
    for k = 0 to n - 1 do
      if Int64.to_int (Vector.get op k) <> (k * 1000003) + 7 then good := false ;
      if Int32.to_int (Vector.get oq k) <> k - 5 then good := false
    done ;
    report "SoA i64+i32 leaves == reference" !good
  end ;
  !ok

let check_layout_wired dev =
  let ir = ir_of p3_kernel in
  let sv = Soa_vector.create dpair_custom 8 in
  let out = Vector.create Vector.float32 8 in
  match
    Soa_launch.run_soa
      ~device:dev
      ~ir
      ~args:
        [
          Soa_launch.SA_Soa sv;
          Soa_launch.SA_Reg (Sarek.Execute.Vec out);
          Soa_launch.SA_Reg (Sarek.Execute.Int 8);
        ]
      ~block:(dims 8)
      ~grid:(dims 1)
      ()
  with
  | () ->
      Printf.printf
        "  %-56s FAIL (ran with a mismatched record type)\n%!"
        "run_soa consults the layout check"
      |> fun () -> false
  | exception Sarek.Execute_error.Execution_error e ->
      let msg = Sarek.Execute_error.error_to_string e in
      if contains msg "wrong byte offsets" then (
        Printf.printf "  %-56s OK\n%!" "run_soa consults the layout check" ;
        true)
      else (
        Printf.printf
          "  %-56s FAIL (raised, but not the layout error: %s)\n%!"
          "run_soa consults the layout check"
          msg ;
        false)

let () =
  Benchmarks.init () ;
  let n = 1024 in
  (* Device-independent, so it runs BEFORE the no-device early exit — otherwise
     a machine with no device would report SKIPPED while silently not checking a
     property that needs no device at all. *)
  let derive_ok = check_field_derivation () in
  (* Also device-independent, and deliberately so — it quantifies over the
     declared framework floor unioned with the registered backends rather than
     over the devices present here, so a skip reason that is false of a device
     class stays caught on a host that has no such device. Before the no-device
     exit for the same reason as the two above. It must stay AFTER
     [Benchmarks.init] above, which is what populates the registry half. *)
  let scope_ok = check_skip_reason_scope () in
  let layout_ok = check_layout_validation () && derive_ok && scope_ok in
  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline
      "test_soa_emitter_equiv: no device - SKIPPED (layout validation still \
       checked above)" ;
    exit (if layout_ok then 0 else 1)) ;
  let any_ptx = Array.exists is_ptx devs in
  if not any_ptx then
    print_endline
      "test_soa_emitter_equiv: no CUDA/PTX device - SoA leg skipped (AoS + \
       reference still checked)" ;
  let ok = ref true in
  Array.iter
    (fun dev ->
      (* point3d (f32) runs everywhere: cross-backend AoS + reference, plus the
         PTX SoA leg. *)
      if not (check "point3d(f32)" dev n run_p3) then ok := false ;
      (* dpair (f64) proves the 8-byte SoA leaf. Gated on the DEVICE's fp64
         capability, not on CUDA/PTX.

         CORRECTED 2026-07-30 — the previous restriction to PTX was justified as
         "some non-PTX backends, e.g. OpenCL/radeonsi, have an unrelated f64
         custom-vector gap". That attribution is wrong. Measured: with
         RUSTICL_FEATURES=fp64 set, this row PASSES on OpenCL/radeonsi (both
         devices), Vulkan, Native and the Interpreter. There is no f64
         custom-vector gap; rusticl simply does not advertise cl_khr_fp64 unless
         that variable is set, and Sarek's fp64 launch gate then correctly
         refuses. clinfo agrees: "Double-precision Floating-point support (n/a)"
         becomes "(cl_khr_fp64)" with the variable set. The RX 7900 XTX has fp64
         in hardware, so nothing about the DEVICE was ever the reason.

         Keying on the capability therefore turns a 1-device row into a 7-device
         one (or 5 + 2 honest skips without the variable), and stops the test
         asserting a device limitation that does not exist. *)
      let dev_can_f64 =
        dev.Device.framework = "Native"
        || dev.Device.framework = "Interpreter"
        || Device.allows_fp64 dev
      in
      (* The skip is NAMED. Gating the call on [dev_can_f64] alone printed
         nothing at all on a device without fp64, and a row that prints nothing is
         indistinguishable from one that passed — the exact skip-as-pass shape the
         guarded cases below were restructured to make unrepresentable. Same
         wording as the fp64 arm of [check_mixed_widths] (its other arm reports
         int64, a different capability). Not the last bare gate in this loop:
         [check_roundtrip] at the end is still [if is_ptx dev && …] and prints
         nothing on a non-PTX device. *)
      if not dev_can_f64 then
        Printf.printf
          "SoA-emitter dpair(f64) [%s] %s: SKIP (device reports no fp64)\n%!"
          dev.Device.framework
          dev.Device.name
      else if not (check "dpair(f64)" dev n run_dpair) then ok := false ;
      if not (check_transparent dev n) then ok := false ;
      (* Transparent OUTPUT read-back. Its only blocker is the CPU field-store
         gap, NOT the SoA ABI: on OpenCL/Vulkan the launch takes the packed AoS
         fallback and the same [Vector.get] must return the same answer, which is
         the stronger claim. Round 3 gated this on [is_ptx] and printed the
         backlog-172 reason on four devices that reason is false of. *)
      if
        not
          (guarded
             dev
             ~blockers:[blocker_cpu_field_store]
             ~label:"transparent SoA output read back"
             (fun () -> check_transparent_roundtrip dev n))
      then ok := false ;
      (* H5 (backlog-181): relaunch on the SAME vector. Same single blocker —
         `v.(i).f <- e` is what the kernel does, so the two CPU backends cannot
         run it, and every other backend can.

         One case per host writer (backlog-190). Vector.set was fixed with H5;
         unsafe_set carried the identical Stale_CPU arm and fill's was a wider
         catch-all. Separate cases so a fix to one cannot make the others read as
         covered. That write-loss is HOST-side and not PTX-specific, so running
         these three on OpenCL and Vulkan is not padding: before round 4 the fix
         had no non-PTX coverage anywhere in the suite. kernel_set is deliberately
         absent: it is documented to skip location handling entirely, because a
         per-element update would race across the threads it exists to serve. *)
      List.iter
        (fun (writer_name, write, y_expected, stale_note) ->
          if
            not
              (guarded
                 dev
                 ~blockers:[blocker_cpu_field_store]
                 ~label:(relaunch_label writer_name)
                 (fun () ->
                   check_relaunch
                     dev
                     n
                     ~writer_name
                     ~write
                     ~y_expected
                     ~stale_note))
          then ok := false)
        [
          ( "Vector.set",
            (fun sv n y_of ->
              for i = 0 to n - 1 do
                Vector.set
                  sv
                  i
                  {x = float_of_int i; y = y_of i; z = float_of_int (n - i)}
              done),
            (fun i -> float_of_int (1000 - i) *. 2.0),
            fun i -> float_of_int (i + 1) *. 4.0 );
          ( "Vector.unsafe_set",
            (fun sv n y_of ->
              for i = 0 to n - 1 do
                Vector.unsafe_set
                  sv
                  i
                  {x = float_of_int i; y = y_of i; z = float_of_int (n - i)}
              done),
            (fun i -> float_of_int (1000 - i) *. 2.0),
            fun i -> float_of_int (i + 1) *. 4.0 );
          (* fill writes ONE value to every element, so both the expectation
               and the stale value are uniform. It takes y_of 0: round 1 fills
               y=1 which the kernel doubles to 2; round 2 fills y=1000, doubled
               to 2000. A stale round 2 would re-double round 1's result to 4 --
               far from 2000, so the two cannot be confused. *)
          ( "Vector.fill",
            (fun sv n y_of ->
              Vector.fill sv {x = 0.0; y = y_of 0; z = float_of_int n}),
            (fun _ -> 2000.0),
            fun _ -> 4.0 );
        ] ;
      (* The cases below genuinely need the SoA ABI itself — they are about which
         ABI a read-back or a free follows, so a device that never selects the
         SoA ABI has no such question to get wrong. That reason is by design and
         permanent, which is why it is a different blocker from the one above and
         is not collapsed into a single "not supported here".

         Whose ABI does read-back follow? The first two are the two ways that
         question got a stale answer, and they are separate cases because the two
         paths that asked it are separate: the LAUNCH path
         (Execute.transfer_vectors_to_device, which never cleared the flag) and
         the CLEANUP path (Transfer.free_*, which had no SoA arm at all). *)
      List.iter
        (fun (label, body) ->
          if
            not
              (guarded dev ~blockers:[blocker_needs_soa_abi] ~label (fun () ->
                   body ()))
          then ok := false)
        [
          ( "packed AoS launch after a transparent SoA one",
            fun () -> check_soa_then_packed dev n );
          (* Same pair of launches with a HOST WRITE in between: the flag must be
             normalised without the gather replaying the leaves over it. The two
             cases pin the two directions of one condition. *)
          ( "a host write between an SoA and a packed launch survives",
            fun () -> check_host_write_survives_packed dev n );
          ( "free_all_buffers preserves a transparent SoA result",
            fun () -> check_free_preserves_soa dev n );
          (* And the free must RELEASE, not merely preserve — a separate claim
             from the case above, which passed while zero bytes came back. *)
          ( "free_all_buffers RELEASES a transparent SoA vector's leaves",
            fun () -> check_free_releases_leaves dev n );
          (* free_buffer, the SINGLE-device free, was covered by nothing at all —
             every case above frees through free_all_buffers, which is why its
             incoherent location bookkeeping shipped. *)
          ( "free_buffer leaves the vector coherent (location/re-read/re-upload)",
            fun () -> check_free_buffer_coherent dev n );
        ] ;
      if not (check_mixed_widths dev n) then ok := false ;
      (* Item 3: SoA launch must be rejected (never wrong data) on non-PTX. *)
      if not (check_gate dev) then ok := false ;
      (* Wiring + ordering: a mismatched must surface the LAYOUT error,
         not the device gate, on this very non-PTX device. *)
      if not (check_layout_wired dev) then ok := false ;
      (* H4: a short arg list must be refused on ARITY, on every device — the
         check is device-independent, which is what makes it testable without a
         CUDA host. *)
      if not (check_short_arg_list dev) then ok := false ;
      (* Leaf-write round-trip (D2H + gather) on CUDA/PTX. *)
      if is_ptx dev && not (check_roundtrip dev n) then ok := false)
    devs ;
  if not (!ok && layout_ok) then exit 1
