(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Runtime - Transfer Control & Streams
 *
 * Provides explicit and async data transfer API with stream management.
 * Phase 2 of runtime V2 feature parity roadmap.
 *
 * Uses the framework plugin system directly.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework_registry

(** {1 Auto-Transfer Mode} *)

let auto_mode = ref true

let enable_auto () = auto_mode := true

let disable_auto () = auto_mode := false

let is_auto () = !auto_mode

let set_auto enabled = auto_mode := enabled

(** {1 Device Buffer Allocation} *)

(** Allocate a device buffer for a scalar vector, returning a DEVICE_BUFFER
    module. The buffer is packaged with its backend for type-safe operations.

    For CPU devices (OpenCL CPU, Native), uses zero-copy allocation when
    possible to avoid memory transfers entirely. *)
let alloc_scalar_buffer (type a b) (dev : Device.t) (length : int)
    (sk : (a, b) Vector.scalar_kind) : Vector.device_buffer =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let ba_kind = Vector.to_bigarray_kind sk in
      let buf = B.Memory.alloc backend_dev length ba_kind in
      let elem_sz = Vector.scalar_elem_size sk in
      let device_ptr = B.Memory.device_ptr buf in
      (module struct
        let device = dev

        let size = length

        let elem_size = elem_sz

        let device_ptr = device_ptr

        let bind_to_kargs kargs idx =
          (* Unwrap kargs to the backend's kernel args type and bind buffer *)
          Log.debugf
            Log.Transfer
            "bind_to_kargs: idx=%d ptr=%Ld"
            idx
            (Int64.of_nativeint device_ptr) ;
          match B.unwrap_kargs kargs with
          | Some args -> B.Kernel.set_arg_buffer args idx buf
          | None -> failwith "bind_to_kargs: backend mismatch"

        let host_ptr_to_device src_ptr ~byte_size =
          Log.debugf
            Log.Transfer
            "host_ptr_to_device: ptr=%Ld size=%d"
            (Int64.of_nativeint device_ptr)
            byte_size ;
          if B.Memory.is_zero_copy buf then () (* Skip for zero-copy *)
          else
            B.Memory.host_ptr_to_device
              ~src_ptr:(Ctypes.raw_address_of_ptr src_ptr)
              ~byte_size
              ~dst:buf

        let device_to_host_ptr dst_ptr ~byte_size =
          if B.Memory.is_zero_copy buf then () (* Skip for zero-copy *)
          else
            B.Memory.device_to_host_ptr
              ~src:buf
              ~dst_ptr:(Ctypes.raw_address_of_ptr dst_ptr)
              ~byte_size

        let free () = B.Memory.free buf
      end : Vector.DEVICE_BUFFER)

(** Allocate a device buffer using zero-copy (host memory sharing) if supported.
    Returns None if the backend doesn't support zero-copy for this device. *)
let alloc_scalar_buffer_zero_copy (type a b) (dev : Device.t)
    (ba : (a, b, Bigarray.c_layout) Bigarray.Array1.t)
    (sk : (a, b) Vector.scalar_kind) : Vector.device_buffer option =
  match Framework_registry.find_backend dev.framework with
  | None -> None
  | Some (module B : Framework_sig.BACKEND) -> (
      let backend_dev = B.Device.get dev.backend_id in
      let ba_kind = Vector.to_bigarray_kind sk in
      match B.Memory.alloc_zero_copy backend_dev ba ba_kind with
      | None -> None
      | Some buf ->
          let elem_sz = Vector.scalar_elem_size sk in
          let length = Bigarray.Array1.dim ba in
          let device_ptr = B.Memory.device_ptr buf in
          Some
            (module struct
              let device = dev

              let size = length

              let elem_size = elem_sz

              let device_ptr = device_ptr

              let bind_to_kargs kargs idx =
                Log.debugf
                  Log.Transfer
                  "bind_to_kargs (zero-copy): idx=%d ptr=%Ld"
                  idx
                  (Int64.of_nativeint device_ptr) ;
                match B.unwrap_kargs kargs with
                | Some args -> B.Kernel.set_arg_buffer args idx buf
                | None -> failwith "bind_to_kargs: backend mismatch"

              let host_ptr_to_device _src_ptr ~byte_size:_ =
                () (* No-op for zero-copy *)

              let device_to_host_ptr _dst_ptr ~byte_size:_ =
                () (* No-op for zero-copy *)

              let free () = B.Memory.free buf
            end : Vector.DEVICE_BUFFER))

(** Allocate a device buffer for a custom vector *)
let alloc_custom_buffer (dev : Device.t) (length : int) (elem_sz : int) :
    Vector.device_buffer =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let buf =
        B.Memory.alloc_custom backend_dev ~size:length ~elem_size:elem_sz
      in
      let device_ptr = B.Memory.device_ptr buf in
      (module struct
        let device = dev

        let size = length

        let elem_size = elem_sz

        let device_ptr = device_ptr

        let bind_to_kargs kargs idx =
          (* Unwrap kargs to the backend's kernel args type and bind buffer *)
          Log.debugf
            Log.Transfer
            "bind_to_kargs: idx=%d ptr=%Ld"
            idx
            (Int64.of_nativeint device_ptr) ;
          match B.unwrap_kargs kargs with
          | Some args -> B.Kernel.set_arg_buffer args idx buf
          | None -> failwith "bind_to_kargs: backend mismatch"

        let host_ptr_to_device src_ptr ~byte_size =
          B.Memory.host_ptr_to_device
            ~src_ptr:(Ctypes.raw_address_of_ptr src_ptr)
            ~byte_size
            ~dst:buf

        let device_to_host_ptr dst_ptr ~byte_size =
          B.Memory.device_to_host_ptr
            ~src:buf
            ~dst_ptr:(Ctypes.raw_address_of_ptr dst_ptr)
            ~byte_size

        let free () = B.Memory.free buf
      end : Vector.DEVICE_BUFFER)

(** {1 Buffer Management for Vectors} *)

(** Ensure vector has a device buffer, allocating if needed. For backends that
    support zero-copy (typically CPU backends), automatically uses zero-copy to
    avoid memory transfer overhead. The backend decides via alloc_zero_copy. *)
let ensure_buffer (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
    Vector.device_buffer =
  match Vector.get_buffer vec dev with
  | Some buf -> buf
  | None ->
      Log.debugf
        Log.Transfer
        "ensure_buffer: allocating for dev=%d len=%d"
        dev.id
        vec.length ;
      let buf =
        Gpu_memory.with_retry (fun () ->
            match (vec.kind, vec.host) with
            | Vector.Scalar sk, Vector.Bigarray_storage ba -> (
                (* Try zero-copy first - backend decides if supported *)
                Log.debug Log.Transfer "  -> trying zero-copy path" ;
                match alloc_scalar_buffer_zero_copy dev ba sk with
                | Some zc_buf ->
                    Log.debugf
                      Log.Transfer
                      "  -> using zero-copy for device %d"
                      dev.id ;
                    zc_buf
                | None ->
                    Log.debug
                      Log.Transfer
                      "  -> zero-copy not supported, using regular alloc" ;
                    let buf = alloc_scalar_buffer dev vec.length sk in
                    buf)
            | Vector.Scalar _, Vector.Custom_storage _ -> .
            | Vector.Custom c, _ ->
                Log.debug Log.Transfer "  -> custom alloc" ;
                alloc_custom_buffer dev vec.length c.elem_size)
      in
      Log.debugf
        Log.Transfer
        "ensure_buffer: storing buffer for dev=%d (hashtbl key=%d)"
        dev.id
        dev.id ;
      (* Register GC finalizer on first device buffer allocation *)
      if Hashtbl.length vec.device_buffers = 0 then
        Gpu_memory.register_finalizer vec ;
      Hashtbl.replace vec.device_buffers dev.id buf ;
      let (module B : Vector.DEVICE_BUFFER) = buf in
      (* Track GPU memory usage *)
      Gpu_memory.track_alloc (B.size * B.elem_size) ;
      Log.debugf
        Log.Transfer
        "ensure_buffer: stored buffer ptr=%Ld size=%d"
        (Int64.of_nativeint B.device_ptr)
        B.size ;
      buf

(** {1 Transfer Operations} *)

(** Everything that reads device memory back into host storage, and NOTHING
    else. The module exists for its signature: [copy_device_to_host] is defined
    inside and deliberately absent from it, so the only code that can call the
    packed-buffer read directly is {!Read_back.read_back_to_host} below.

    That is the enforcement the previous version of this claim lacked. It said
    the call sites were "exhaustive BY CONSTRUCTION" while nothing constructed
    anything: there is no [Transfer.mli], so every top-level binding in this
    file is exported and a fifth direct caller compiled fine. The claim happened
    to be true when it was written — it was re-audited, and it held — but an
    audited fact and an enforced one differ exactly where it matters, which is
    the next person to add a read-back path. Three of the four existing call
    sites were added or corrected in this branch precisely because they had
    bypassed the decision, so "someone adds a fifth and bypasses it too" is the
    observed failure mode, not a hypothetical one.

    Now a bypass does not compile. *)
module Read_back : sig
  val has_device_data : ('a, 'b) Vector.t -> Device.t -> bool

  val read_back_to_host : ('a, 'b) Vector.t -> Device.t -> unit
end = struct
  (** Copy vector data from a device buffer to CPU storage. *)
  let copy_device_to_host (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
      unit =
    match Vector.get_buffer vec dev with
    | None -> failwith "to_cpu: no device buffer to transfer from"
    | Some buf -> (
        let (module B : Vector.DEVICE_BUFFER) = buf in
        Log.debugf
          Log.Transfer
          "to_cpu: got buffer ptr=%Ld size=%d"
          (Int64.of_nativeint B.device_ptr)
          B.size ;
        match vec.host with
        | Vector.Bigarray_storage ba ->
            let ptr, byte_size =
              Vector_transfer.bigarray_to_ptr ba B.elem_size
            in
            B.device_to_host_ptr ptr ~byte_size ;
            (* [device_to_host_ptr] takes a bare [nativeint] address (see
             Framework_sig.ml:218-223), which is not a GC root, and this is a
             device->HOST WRITE: if [ba] were collected mid-transfer the backend
             would write into freed memory. Keep it reachable across the call —
             without this the transfer was the function's last expression and
             nothing rooted [ba]. *)
            ignore (Sys.opaque_identity ba)
        | Vector.Custom_storage {ptr; custom; length} ->
            B.device_to_host_ptr ptr ~byte_size:(length * custom.elem_size) ;
            ignore (Sys.opaque_identity ptr))

  (** Is there device-resident data for [vec] that a read-back could actually
      fetch? Two disjoint sources, and under the SoA ABI it is the second one:
      the packed buffer is not merely stale there, it was never allocated.

      [dev] selects the packed buffer, and the SoA arm ignores it: the binding
      holds ONE [soa_leaves_live] flag for the whole vector, not one per device,
      so it cannot answer "on [dev]" and does not pretend to. Unreachable as a
      difference today — both callers pass the device the vector's own
      [location] names, so the two questions coincide — and making the flag
      per-device is the change that would let this arm honour [dev]. Until then
      the honest reading of the result is "somewhere on a device", not "on
      [dev]".

      That the flag is whole-vector while {!free_buffer} releases per-device is
      not a cosmetic asymmetry: [soa_free_leaves] used to CLEAR the flag on a
      per-device free, which made this function answer [false] for a device
      whose leaves were still live, and the drain-before-free in
      {!free_all_buffers} consults exactly this answer. It now NARROWS the flag
      — still set iff it already was and some leaf survives — so a per-device
      release cannot disown another device's, and a release cannot resurrect
      ownership a packed launch gave up either. *)
  let has_device_data (type a b) (vec : (a, b) Vector.t) (dev : Device.t) : bool
      =
    Option.is_some (Vector.get_buffer vec dev)
    ||
    match vec.Vector.soa with
    | Some b -> !(b.Vector.soa_leaves_live)
    | None -> false

  (** The ONE place that decides WHERE a read-back reads from, shared by every
      path in this module that pulls device data into host storage. As of
      2026-07-30 that is four call sites, and the list is exhaustive because
      [copy_device_to_host] is not in the enclosing module's signature — a fifth
      direct caller does not compile:

      - {!to_cpu} (and {!sync} through it) — the explicit read-back;
      - {!to_device}'s cross-device MIGRATION arm — a vector resident on one
        device is drained to the host before being uploaded to another;
      - {!free_buffer} and {!free_all_buffers} — drain before release.

      It exists as a function because those paths did not have this decision:
      they called {!copy_device_to_host} directly, so after a transparent SoA
      launch they downloaded a packed buffer the kernel never wrote (silently
      discarding the output) or, with no packed buffer at all, failed with "no
      device buffer to transfer from" from inside a free or a migration.

      An earlier version of this comment claimed the migration arm already
      shared this decision. It did not — it still called {!copy_device_to_host}
      — and both halves of the failure pair above were reachable through it
      (measured 2026-07-30 on two CUDA/PTX devices: the raise with no packed
      buffer, and silent wrong data with one present). [check_cross_device_soa]
      in [test_soa_emitter_equiv.ml] now pins both.

      The condition reads the flag the UPLOAD set; it does not re-decide "SoA or
      AoS?" from the device or the vector's shape. A second independent answer
      to that question is how a round trip ends up uploading leaves and
      downloading a packed buffer. *)
  let read_back_to_host (type a b) (vec : (a, b) Vector.t) (dev : Device.t) :
      unit =
    match vec.Vector.soa with
    | Some b when !(b.Vector.soa_leaves_live) ->
        (* The launch took the SoA ABI, so the results are in the N leaf buffers
           and the packed device buffer this function would otherwise download
           was never written. Reading it back would hand the caller the
           pre-launch host contents with no error anywhere. *)
        b.Vector.soa_from_device dev
    | Some _ | None -> copy_device_to_host vec dev
end

let has_device_data = Read_back.has_device_data

let read_back_to_host = Read_back.read_back_to_host

(** Transfer vector data to a device *)
let to_device (type a b) (vec : (a, b) Vector.t) (dev : Device.t) : unit =
  let loc_str =
    match vec.location with
    | Vector.CPU -> "CPU"
    | Vector.GPU _ -> "GPU"
    | Vector.Both _ -> "Both"
    | Vector.Stale_CPU _ -> "Stale_CPU"
    | Vector.Stale_GPU _ -> "Stale_GPU"
  in
  Log.debugf Log.Transfer "to_device: location=%s dev=%d" loc_str dev.id ;
  (* MIGRATION: the vector is resident on another device, so drain it to the
     host before uploading to [dev]. Through {!read_back_to_host}, not
     {!copy_device_to_host} — this arm is one of the four paths that pull device
     data into host storage, and it used to be the one that did NOT share the
     decision. After a transparent SoA launch the results are in the leaves, so
     the direct call either raised "no device buffer to transfer from" (no packed
     buffer exists on that path) or, when an earlier packed upload had left one,
     downloaded its pre-launch bytes over the host storage and discarded the
     launch's output silently. *)
  (match vec.location with
  | (Vector.GPU d | Vector.Stale_CPU d) when d.id <> dev.id ->
      read_back_to_host vec d ;
      vec.location <- Vector.Both d
  | _ -> ()) ;
  (* Check if already up-to-date on this device *)
  match vec.location with
  | Vector.GPU d when d.id = dev.id -> Log.debug Log.Transfer "-> skip (GPU)"
  | Vector.Both d when d.id = dev.id -> Log.debug Log.Transfer "-> skip (Both)"
  | Vector.Stale_CPU d when d.id = dev.id ->
      Log.debug Log.Transfer "-> skip (Stale_CPU)"
  | _ -> (
      (* Ensure buffer exists and transfer *)
      let buf = ensure_buffer vec dev in
      let (module B : Vector.DEVICE_BUFFER) = buf in
      Log.debugf
        Log.Transfer
        "to_device: transferring %d bytes to dev=%d"
        (vec.length * B.elem_size)
        dev.id ;
      Log.debugf
        Log.Transfer
        "-> transferring %d bytes"
        (vec.length * B.elem_size) ;
      (match vec.host with
      | Vector.Bigarray_storage ba ->
          let ptr, byte_size = Vector_transfer.bigarray_to_ptr ba B.elem_size in
          Log.debugf
            Log.Transfer
            "to_device: calling host_ptr_to_device byte_size=%d"
            byte_size ;
          B.host_ptr_to_device ptr ~byte_size ;
          (* Same keep-alive obligation as [copy_device_to_host] above. *)
          ignore (Sys.opaque_identity ba)
      | Vector.Custom_storage {ptr; custom; length} ->
          B.host_ptr_to_device ptr ~byte_size:(length * custom.elem_size) ;
          ignore (Sys.opaque_identity ptr)) ;
      vec.location <- Vector.Both dev ;
      (* The packed buffer on [dev] now holds host storage, so a read-back must
         read IT and not the leaves — and only this arm may say so, because only
         this arm uploaded anything. The three already-resident arms above
         ([GPU], [Both], [Stale_CPU], each on [dev] itself) upload nothing — and
         [Stale_CPU dev] on a transparent vector is exactly the case where the
         leaves ARE the device copy — so clearing there would disown live
         leaves.

         Reaching here with the flag set means the host copy was authoritative or
         in sync first — that is what makes the upload above meaningful, and it is
         what every path into this arm establishes: the migration arm at the top of
         this function has just drained the other device through
         [read_back_to_host], and the remaining locations ([CPU], [Both d],
         [Stale_GPU d]) each assert the host copy is not behind. So this does not
         discard a pending leaf result; it
         stops [read_back_to_host] from answering a question about [dev] by
         reading leaves on another device, which is the one thing the whole-vector
         flag cannot express (see {!Read_back.has_device_data}).

         [Execute.transfer_vectors_to_device] normalises the same flag on the
         packed-launch path. That site is not this one: it runs before a LAUNCH
         and gathers first, this one after an UPLOAD that has already made the
         host copy authoritative. Both exist because the flag names the ABI of the
         most recent operation that made a device copy authoritative, and an
         upload is such an operation. *)
      match vec.Vector.soa with
      | Some b -> b.Vector.soa_leaves_live := false
      | None -> ())

(** Transfer vector data from device to CPU.
    @param force
      If true, always transfer even if location is Both (useful after kernel
      writes) *)
let to_cpu ?(force = false) (type a b) (vec : (a, b) Vector.t) : unit =
  Log.debugf
    Log.Transfer
    "to_cpu: CALLED: force=%b location=%s"
    force
    (match vec.location with
    | Vector.CPU -> "CPU"
    | Vector.GPU _ -> "GPU"
    | Vector.Both _ -> "Both"
    | Vector.Stale_GPU _ -> "Stale_GPU"
    | Vector.Stale_CPU _ -> "Stale_CPU") ;
  let needs_transfer =
    match vec.location with
    | Vector.CPU -> false (* No device buffer *)
    | Vector.Both _ -> force (* Transfer if forced *)
    | Vector.Stale_GPU _ -> false (* CPU already authoritative *)
    | Vector.GPU _ | Vector.Stale_CPU _ -> true
  in
  Log.debugf Log.Transfer "to_cpu: needs_transfer=%b" needs_transfer ;
  if needs_transfer then begin
    let dev =
      match vec.location with
      | Vector.GPU d | Vector.Stale_CPU d | Vector.Both d -> d
      | _ -> failwith "to_cpu: no device"
    in
    Log.debugf
      Log.Transfer
      "to_cpu: transferring from dev=%d (force=%b)"
      dev.id
      force ;
    (* [read_back_to_host] carries the SoA-vs-packed decision; the packed arm's
       own "no device buffer to transfer from" failure comes from
       [copy_device_to_host], so the precondition stays where the read happens
       rather than being re-derived here. *)
    read_back_to_host vec dev ;
    vec.location <- Vector.Both dev
  end
  else
    Log.debugf
      Log.Transfer
      "to_cpu: skip (location=%s, force=%b)"
      (match vec.location with
      | Vector.CPU -> "CPU"
      | Vector.GPU _ -> "GPU"
      | Vector.Both _ -> "Both"
      | Vector.Stale_CPU _ -> "Stale_CPU"
      | Vector.Stale_GPU _ -> "Stale_GPU")
      force

(** Ensure vector is fully synchronized *)
let sync (type a b) (vec : (a, b) Vector.t) : unit =
  match vec.location with
  | Vector.CPU -> ()
  | Vector.Both _ -> ()
  | Vector.GPU dev ->
      to_cpu vec ;
      vec.location <- Vector.Both dev
  | Vector.Stale_CPU dev ->
      to_cpu vec ;
      vec.location <- Vector.Both dev
  | Vector.Stale_GPU dev ->
      to_device vec dev ;
      vec.location <- Vector.Both dev

(** {1 Buffer Cleanup} *)

(** Release the device memory [vec] holds on [dev], draining it to host storage
    first.

    FOUR steps, and not one of them may sit inside [get_buffer]'s [Some] arm.
    Under the SoA ABI the leaves are the only device memory the vector has and
    the packed buffer is never allocated, so on a transparent vector
    [get_buffer] returns [None] ALWAYS — an early return there does not skip a
    corner case, it skips the whole function.

    The previous version of this comment already said "under the SoA ABI they
    are the only device memory this vector has" while the bookkeeping below it
    still lived in the [Some buf] arm and assumed a packed buffer existed. That
    mismatch was the bug, not a wording slip: the location reset is the only
    code here that assigns [CPU], so a transparent SoA vector came out of this
    function with its leaves freed and [location] still naming [dev]. Two
    measured consequences, both on a vector whose data was intact in host
    storage:

    - [to_cpu ~force:true] raised
      [Failure "to_cpu: no device buffer to transfer from"] — [Both dev] plus
      [force] means "read the device back", and there is nothing there to read;
    - [to_device] on the same device took the "skip (Both)" short-circuit and
      allocated nothing, reinstating the exact short-circuit the [Stale_CPU]
      work earlier in this branch exists to eliminate.

    {!free_all_buffers} was never exposed to this because it assigns [CPU]
    unconditionally at the end. This function now does the same, scoped to
    [dev]. *)
let free_buffer (vec : (_, _) Vector.t) (dev : Device.t) : unit =
  (* 1. Read back BEFORE freeing, and through {!read_back_to_host}: an
     SoA-dispatched vector has no packed buffer of its own, so the old
     [None -> ()] early return threw the output of a transparent launch away
     outright. [has_device_data] keeps the no-op for the case that return was
     actually for — an AoS vector with nothing on this device. *)
  (match vec.location with
  | (Vector.GPU d | Vector.Stale_CPU d)
    when d.id = dev.id && has_device_data vec dev ->
      read_back_to_host vec dev ;
      vec.location <- Vector.Both dev
  | _ -> ()) ;
  (* 2. Release the LEAVES. Under the SoA ABI they are the only device memory
     this vector has, so without this the call returned having freed nothing at
     all. Ordered after the read-back above and before the flag it updates is
     needed again. *)
  (match vec.Vector.soa with
  | Some b -> b.Vector.soa_free_leaves (Some dev)
  | None -> ()) ;
  (* 3. Release the packed buffer, if this vector ever had one on [dev]. *)
  (match Vector.get_buffer vec dev with
  | None -> ()
  | Some buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      B.free () ;
      Gpu_memory.track_free (B.size * B.elem_size) ;
      Hashtbl.remove vec.device_buffers dev.id) ;
  (* 4. Location. Outside step 3's [Some] arm, which is the whole fix: steps 2
     and 3 together released everything this vector held on [dev], whether that
     was leaves, a packed buffer, or both, so the state that follows is the same
     in all three cases and [CPU] is it.

     Guarded on the device alone rather than per-constructor: every
     device-naming constructor asserts something about [dev]'s contents, and
     after steps 2-3 [dev] holds nothing, so all four collapse to the same
     answer. A location naming a DIFFERENT device is untouched — that device's
     memory was not freed. *)
  match vec.location with
  | (Vector.GPU d | Vector.Both d | Vector.Stale_CPU d | Vector.Stale_GPU d)
    when d.id = dev.id ->
      vec.location <- Vector.CPU
  | _ -> ()

(** Free all device buffers for a vector *)
let free_all_buffers (vec : (_, _) Vector.t) : unit =
  (* Same SoA-aware read-back as {!free_buffer}. Pre-fix this arm called
     [copy_device_to_host] on a vector whose packed buffer does not exist after a
     transparent launch, so freeing without an explicit [to_cpu] raised
     "no device buffer to transfer from" from inside cleanup — and, when a packed
     buffer DID exist from an earlier AoS launch, silently overwrote the host
     storage with its pre-SoA-launch contents. *)
  (match vec.location with
  | (Vector.GPU d | Vector.Stale_CPU d) when has_device_data vec d ->
      read_back_to_host vec d ;
      vec.location <- Vector.CPU
  | _ -> ()) ;
  (* Same leaf release as {!free_buffer}, on every device. Pre-fix this function
     was a no-op on a transparent SoA vector's memory: [device_buffers] is EMPTY
     under that ABI, so the iteration below had nothing to free and the call
     reported success having released zero bytes (measured with
     [Gpu_memory.usage()]: 3840 B -> 4224 B across a launch + free). *)
  (match vec.Vector.soa with
  | Some b -> b.Vector.soa_free_leaves None
  | None -> ()) ;
  Hashtbl.iter
    (fun _ buf ->
      let (module B : Vector.DEVICE_BUFFER) = buf in
      B.free () ;
      Gpu_memory.track_free (B.size * B.elem_size))
    vec.device_buffers ;
  Hashtbl.clear vec.device_buffers ;
  vec.location <- Vector.CPU

(** {1 Device Synchronization} *)

(** Synchronize all pending operations on a device *)
let flush (dev : Device.t) : unit =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      B.Device.synchronize backend_dev

(** {1 Stream Operations} *)

(** Stream handle - packages backend stream with its operations *)
module type STREAM = sig
  val device : Device.t

  val synchronize : unit -> unit

  val destroy : unit -> unit
end

type stream = (module STREAM)

(** Create a new stream on a device *)
let create_stream (dev : Device.t) : stream =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let s = B.Stream.create backend_dev in
      (module struct
        let device = dev

        let synchronize () = B.Stream.synchronize s

        let destroy () = B.Stream.destroy s
      end : STREAM)

(** Get default stream for a device *)
let default_stream (dev : Device.t) : stream =
  match Framework_registry.find_backend dev.framework with
  | None -> failwith ("Unknown framework: " ^ dev.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let backend_dev = B.Device.get dev.backend_id in
      let s = B.Stream.default backend_dev in
      (module struct
        let device = dev

        let synchronize () = B.Stream.synchronize s

        let destroy () = () (* Don't destroy default stream *)
      end : STREAM)

let synchronize_stream (s : stream) =
  let (module S : STREAM) = s in
  S.synchronize ()

let destroy_stream (s : stream) =
  let (module S : STREAM) = s in
  S.destroy ()

(** {1 Batch Operations} *)

let to_device_all (vecs : (_, _) Vector.t list) (dev : Device.t) : unit =
  List.iter (fun v -> to_device v dev) vecs

let to_cpu_all (vecs : (_, _) Vector.t list) : unit = List.iter to_cpu vecs

let sync_all (vecs : (_, _) Vector.t list) : unit = List.iter sync vecs

(** {1 Auto-sync Callback Registration} *)

(** Register auto-sync callback with Vector module. The callback respects the
    global auto_mode setting. *)
let () =
  Vector.register_sync_callback
    {
      Vector.sync =
        (fun (type a b) (vec : (a, b) Vector.t) ->
          if not !auto_mode then false
          else begin
            to_cpu vec ;
            true
          end);
    }
