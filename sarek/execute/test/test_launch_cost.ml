(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * End-to-end launch cost gate (specs/cheap-repeated-launches.md).
 *
 * #522 added a whole-IR fold to every launch ([Execute.check_launch_warp]) and
 * took 42 to 53 percent off a downstream LLM decode (359 to 168 tok/s, 281 to 497
 * launches per token). The roster gate, review and CI all passed; only the
 * consumer noticed. test_launch_check_cost.ml pins the two checks that regressed
 * then; this test pins the launch path of Execute.run_vectors on the two probe
 * backends below, so the next such regression is caught whatever check it
 * hides in, as long as it is on the paths those probes reach (the list of what
 * is not covered is in docs/launch-cost-gate.md).
 *
 * What is measured: [Execute.run_vectors] through a JIT probe backend that does
 * nothing at the device (no GPU, no clock-dependent driver), N launches of one
 * kernel after a warming launch, repeated; the minimum over the rounds is kept
 * (filters slow outliers; it may also pick a fast round, which both assertions
 * absorb by comparing figures taken the same way).
 *
 * Two assertions, both machine independent by construction:
 *
 * 1. IR-size independence. A kernel of 20000 statements must launch within a
 *    small factor of a kernel of one statement. Anything that folds the IR at
 *    each launch costs thousands of times more on the big one. The factor is
 *    generous because the only legitimate dependence on the IR is a cache
 *    lookup.
 * 2. Fixed cost, relative to a reference loop measured in the same process
 *    (hash-table lookups and a formatted string, the kind of work a launch
 *    legitimately does). A launch may cost at most [allowed_reference_units]
 *    of them. This catches a constant added to every launch, which (1) cannot.
 *
 * Both assertions run on two probe backends: one on the immediate launch path
 * and one named like a GPU backend, which takes the deferred (launch window)
 * path and is tried by the matrix-unit rewrite, the two places a fold hid in
 * #522 and could hide again.
 *
 * The measured figures are printed (the runner is verbose: alcotest otherwise
 * keeps the output of a passing test in a file) so a CI log shows the trend,
 * not only the verdict. Calibration of the thresholds: docs/launch-cost-gate.md.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework.Framework_sig
open Spoc_framework_registry
module Execute = Sarek_execute.Execute
module Device = Spoc_core.Device

(* Two probe backends, same code. "LaunchCostProbe" takes the immediate launch
   path. "HIP" is a name the real backend never shares with this executable
   (sarek-hip is not linked) and is what makes the launch take the path of a
   GPU: deferred by the launch window ({!Window_gate.default_backends}) and
   tried by the matrix-unit rewrite ({!Matrix_specialise.device_possible}). A
   gate that only measures the first misses a fold added to either of those. *)
let immediate_name = "LaunchCostProbe"

let windowed_name = "HIP"

let codegen_calls = Atomic.make 0

(* Launches that reached the backend. The only state [run_source] touches: it
   allocates nothing that grows with the number of launches. *)
let reached_backend = Atomic.make 0

module Make_probe (Name : sig
  val name : string
end) : BACKEND = struct
  let name = Name.name

  let version = (1, 0, 0)

  let is_available () = true

  let execution_model = JIT

  let supported_source_langs = [CUDA_Source]

  let generate_source ?block ?soa_params:_ (ir : Sarek_ir_types.kernel) =
    Atomic.incr codegen_calls ;
    let bx = match block with Some b -> b.x | None -> 0 in
    Some (Printf.sprintf "// %s block=%d\n" ir.Sarek_ir_types.kern_name bx)

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args = ()

  let wrap_kargs () = failwith "probe: wrap_kargs"

  let unwrap_kargs _ = None

  let run_source ~source:_ ~lang:_ ~kernel_name:_ ~block:_ ~grid:_ ~shared_mem:_
      _ =
    Atomic.incr reached_backend

  module Intrinsics = Intrinsic_registry.Make ()

  module Device = struct
    type t = unit

    type id = int

    let init () = ()

    let count () = 1

    let get _id = ()

    let id () = 0

    let set_current () = ()

    let synchronize () = ()

    let name () = "Launch cost probe"

    (* Not read by the launch path: launches use the [Device.t] record built
       below ([caps]), which is what the launch rules and the window consult. *)
    let capabilities () =
      {
        max_threads_per_block = 256;
        max_block_dims = (256, 256, 64);
        max_grid_dims = (65535, 65535, 65535);
        shared_mem_per_block = 16384;
        total_global_mem = 1073741824L;
        compute_capability = (0, 0);
        device_features = [];
        coopmat = None;
        supports_atomics = false;
        warp_size = 1;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu = true;
      }
  end

  module Stream = struct
    type t = unit

    let create () = ()

    let create_non_blocking () = ()

    let default () = ()

    let synchronize () = ()

    let destroy () = ()
  end

  module Memory = struct
    type 'a buffer = unit

    let alloc _dev _size _kind = ()

    let alloc_custom _dev ~size:_ ~elem_size:_ = ()

    let alloc_zero_copy _dev _ba _kind = None

    let alloc_host_ptr _dev _ba _kind = Error Framework_sig.Host_ptr_unsupported

    let device_ptr _buf = Nativeint.zero

    let is_zero_copy _buf = false

    let host_to_device ~src:_ ~dst:_ = ()

    let device_to_host ~src:_ ~dst:_ = ()

    let device_to_device ~src:_ ~dst:_ = ()

    let host_ptr_to_device ~src_ptr:_ ~byte_size:_ ~dst:_ = ()

    let device_to_host_ptr ~src:_ ~dst_ptr:_ ~byte_size:_ = ()

    let size _buf = 0

    let free _buf = ()
  end

  module Kernel = struct
    type t = unit

    type args = unit

    let compile _dev ~name:_ ~source:_ = ()

    let compile_cached _dev ~name:_ ~source:_ = ()

    let clear_cache () = ()

    let load_from_ptx ~name:_ ~ptx:_ = ()

    let create_args () = ()

    let set_arg_buffer _args _idx _buf = ()

    let set_arg_int32 _args _idx _v = ()

    let set_arg_int64 _args _idx _v = ()

    let set_arg_float32 _args _idx _v = ()

    let set_arg_float64 _args _idx _v = ()

    let set_arg_ptr _args _idx _ptr = ()

    let launch _kernel ~args:_ ~grid:_ ~block:_ ~shared_mem:_ ~stream:_ = ()
  end

  module Event = struct
    type t = unit

    let create () = ()

    let record _event _stream = ()

    let synchronize _event = ()

    let elapsed ~start:_ ~stop:_ = 0.0

    let destroy _event = ()
  end

  module Async = Spoc_framework.Framework_sig.Async_refused (struct
    let backend = name

    type device = Device.t

    type stream = Stream.t

    type event = Event.t

    type 'a buffer = 'a Memory.buffer
  end)

  let run_source_on_stream ~stream:_ ~source:_ ~lang:_ ~kernel_name:_ ~block:_
      ~grid:_ ~shared_mem:_ _args =
    Spoc_framework.Framework_sig.refuse_stream_launch ~backend:name

  include Spoc_framework.Framework_sig.Make_refs (struct
    type stream = Stream.t

    type event = Event.t
  end)

  let enable_profiling () = ()

  let disable_profiling () = ()
end

let () =
  Framework_registry.register_backend
    ~priority:1
    (module Make_probe (struct
      let name = immediate_name
    end)) ;
  Framework_registry.register_backend
    ~priority:1
    (module Make_probe (struct
      let name = windowed_name
    end))

let caps : capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = [Sarek_ir_analysis.Warp_shuffle];
    coopmat = None;
    supports_atomics = false;
    warp_size = 32;
    max_registers_per_block = 0;
    clock_rate_khz = 0;
    multiprocessor_count = 1;
    is_cpu = false;
  }

let device_of framework : Device.t =
  {id = 0; backend_id = 0; name = "probe"; framework; capabilities = caps}

let immediate_device = device_of immediate_name

let windowed_device = device_of windowed_name

open Sarek_ir_types

let one = EConst (CInt32 1l)

let filler = SExpr (EBinop (Add, one, EBinop (Mul, one, one)))

let shuffle = SExpr (EIntrinsic ([], "warp_shuffle_xor_i32", [one; one]))

(* kernel (n : int32) -> (); [statements] filler statements, one warp shuffle
   when [with_shuffle] (which turns the launch rule on). A fresh record per
   call: structurally equal calls give physically distinct IRs. *)
let kernel ~statements ~with_shuffle : kernel =
  let n =
    {var_name = "n"; var_id = 0; var_type = TInt32; var_mutable = false}
  in
  let body = List.init statements (fun _ -> filler) in
  let body = if with_shuffle then body @ [shuffle] else body in
  {
    default_kernel with
    kern_name = "launch_cost_k";
    kern_params = [DParam (n, None)];
    kern_body = SSeq body;
  }

(* A warp-multiple block: legal for the shuffle kernels. *)
let launch ?(device = immediate_device) ir =
  Execute.run_vectors
    ~device
    ~ir
    ~args:[Execute.Int32 1l]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    ()

let now = Unix.gettimeofday

let launches_per_round = 2000

let rounds = 7

(* One round of [launches_per_round] launches over the IRs (warmed first: the
   first launch of each pays codegen). The flush is in the timed region: a
   windowed device issues its launches there. *)
let launch_round device irs =
  let irs = Array.of_list irs in
  Array.iter (launch ~device) irs ;
  Execute.Window.flush () ;
  fun () ->
    for i = 1 to launches_per_round do
      launch ~device irs.(i mod Array.length irs)
    done ;
    Execute.Window.flush ()

(* Reference unit: what an ordinary launch is made of, a few hash-table lookups
   and a formatted string. *)
let reference_round () =
  let table = Hashtbl.create 64 in
  for i = 0 to 63 do
    Hashtbl.replace table (string_of_int i) i
  done ;
  let sink = ref 0 in
  let one_unit () =
    for i = 0 to 7 do
      sink := !sink + Hashtbl.find table (string_of_int (i * 8))
    done ;
    sink := !sink + String.length (Printf.sprintf "// k block=%d\n" !sink)
  in
  one_unit () ;
  fun () ->
    for _ = 1 to launches_per_round do
      one_unit ()
    done ;
    ignore (Sys.opaque_identity !sink)

(* Seconds per iteration of each round function: the minimum over [rounds]
   rounds. The functions are timed in turn inside every round, not one after
   the other, so a stretch of load on a shared runner reaches all of them and
   the ratios between them stay meaningful; the minimum then drops the
   stretches that reached only some. *)
let interleaved_minimum round_functions =
  let best = Array.make (List.length round_functions) infinity in
  for _ = 1 to rounds do
    List.iteri
      (fun slot round_function ->
        let t0 = now () in
        round_function () ;
        best.(slot) <-
          Float.min
            best.(slot)
            ((now () -. t0) /. float_of_int launches_per_round))
      round_functions
  done ;
  best

(* A launch of the 20000-statement kernel may cost this many launches of the
   one-statement kernel. A cache lookup costs 1; a whole-IR fold costs
   thousands. *)
let allowed_size_factor = 25.

(* A launch may cost this many reference units. Calibrated in
   docs/launch-cost-gate.md: 0.7 to 1.3 idle and at most 1.4 with 24 busy loops
   on 10 cores. The downstream decode loss of #522 (359 to 168 tok/s at 497 launches per
   token, about 6.4 us per launch) is about 12 units, over this limit (a
   margin of 1.5); a limit of 30 let that fixed cost through. *)
let allowed_reference_units = 8.

let big_statements = 20000

(* Launches rotate over this many distinct IRs of the same size: a loop on one
   IR only ever hits the per-domain last-lookup cache of [Ir_memo], never the
   table lookup a decode (many kernels per token) pays. *)
let distinct_kernels = 4

let report label ~small ~big ~unit_ =
  Printf.printf
    "launch-cost %-9s small = %.2f us, big = %.2f us (x%.2f), reference unit = \
     %.2f us (small = %.1f units)\n\
     %!"
    label
    (small *. 1e6)
    (big *. 1e6)
    (big /. small)
    (unit_ *. 1e6)
    (small /. unit_)

let check_gate label device ~with_shuffle () =
  Execute.clear_launch_cache () ;
  let family statements =
    List.init distinct_kernels (fun _ -> kernel ~statements ~with_shuffle)
  in
  let timings =
    interleaved_minimum
      [
        launch_round device (family 1);
        launch_round device (family big_statements);
        reference_round ();
      ]
  in
  let small = timings.(0) and big = timings.(1) and unit_ = timings.(2) in
  report label ~small ~big ~unit_ ;
  if big /. small > allowed_size_factor then
    Alcotest.failf
      "%s: a launch of a %d-statement kernel costs x%.1f a launch of a \
       1-statement kernel (%.2f us vs %.2f us); the limit is x%.1f. Something \
       walks the IR at every launch."
      label
      big_statements
      (big /. small)
      (big *. 1e6)
      (small *. 1e6)
      allowed_size_factor ;
  if small /. unit_ > allowed_reference_units then
    Alcotest.failf
      "%s: a launch costs %.1f reference units (%.2f us, unit = %.2f us); the \
       limit is %.0f. A fixed cost was added to every launch."
      label
      (small /. unit_)
      (small *. 1e6)
      (unit_ *. 1e6)
      allowed_reference_units

(* The gate itself must see a regression: a launch that folds the IR each time
   (what #522 did) has to fail assertion 1. Without this, a threshold loosened
   until nothing can fail would pass silently. *)
let test_gate_sees_a_whole_ir_fold () =
  let folding_launch ir =
    ignore (Sarek_ir_analysis.kernel_uses_shuffle ir) ;
    launch ir
  in
  let measure ir =
    folding_launch ir ;
    let best = ref infinity in
    for _ = 1 to 3 do
      let t0 = now () in
      for _ = 1 to 200 do
        folding_launch ir
      done ;
      best := Float.min !best ((now () -. t0) /. 200.)
    done ;
    !best
  in
  let small = measure (kernel ~statements:1 ~with_shuffle:true) in
  let big = measure (kernel ~statements:big_statements ~with_shuffle:true) in
  if big /. small <= allowed_size_factor then
    Alcotest.failf
      "a per-launch whole-IR fold is not detected: x%.2f (limit x%.1f)"
      (big /. small)
      allowed_size_factor

let test_launches_reach_the_backend device () =
  Execute.clear_launch_cache () ;
  Atomic.set reached_backend 0 ;
  Atomic.set codegen_calls 0 ;
  let ir = kernel ~statements:1 ~with_shuffle:true in
  for _ = 1 to 10 do
    launch ~device ir
  done ;
  Execute.Window.flush () ;
  Alcotest.(check int)
    "launches reached run_source"
    10
    (Atomic.get reached_backend) ;
  Alcotest.(check int) "one codegen" 1 (Atomic.get codegen_calls)

(* The windowed probe must really take the deferred path (that is its point):
   if the launch window stopped deferring "HIP", the second column of the gate
   would silently measure the immediate path again. *)
let test_windowed_probe_defers () =
  Alcotest.(check bool)
    "the immediate probe is not deferred"
    false
    (Execute.Window.defers_device immediate_device) ;
  Alcotest.(check bool)
    "the windowed probe is deferred"
    true
    (Execute.Window.defers_device windowed_device)

let () =
  Alcotest.run
    ~verbose:true
    "launch cost"
    [
      ( "sanity",
        [
          Alcotest.test_case
            "launches reach the backend"
            `Quick
            (test_launches_reach_the_backend immediate_device);
          Alcotest.test_case
            "windowed launches reach the backend"
            `Quick
            (test_launches_reach_the_backend windowed_device);
          Alcotest.test_case
            "the windowed probe defers"
            `Quick
            test_windowed_probe_defers;
          Alcotest.test_case
            "a per-launch IR fold is visible"
            `Quick
            test_gate_sees_a_whole_ir_fold;
        ] );
      ( "cost",
        [
          Alcotest.test_case
            "plain kernel"
            `Quick
            (check_gate "plain" immediate_device ~with_shuffle:false);
          Alcotest.test_case
            "warp-shuffle kernel"
            `Quick
            (check_gate "shuffle" immediate_device ~with_shuffle:true);
          Alcotest.test_case
            "plain kernel, windowed"
            `Quick
            (check_gate "plain-win" windowed_device ~with_shuffle:false);
          Alcotest.test_case
            "warp-shuffle kernel, windowed"
            `Quick
            (check_gate "shuf-win" windowed_device ~with_shuffle:true);
        ] );
    ]
