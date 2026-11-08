(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Keep the host-device link out of power-saving states during long
    submissions.

    On some platforms the link between the host and a GPU (PCIe active-state
    power management, runtime power management) enters a low-power state while
    the device runs one long kernel, and the next kernel then starts late by the
    link's wake-up latency (hundreds of microseconds were measured on a discrete
    AMD GPU). Launching a trivial kernel on a separate stream every few hundred
    microseconds keeps the link awake. This module does that, on request, for
    the duration of a long submission.

    {b Opt-in.} Nothing else in the runtime calls this module, and the launch
    path of code that does not call it is unchanged. Whether it helps depends on
    the platform; measure before enabling it.

    {b How.} {!start} creates, on the device, a non-blocking stream and an empty
    kernel, and hands them to a native thread that calls the driver's launch
    function directly every [period_us] microseconds, on absolute deadlines (no
    drift). The thread never enters the OCaml runtime: no callback, no
    allocation in the OCaml heap, no runtime lock, so it takes part in no
    stop-the-world collection and a main domain running OCaml code for tens of
    milliseconds cannot delay it. The main domain waits only in {!start}
    (set-up) and {!stop} (join). Set-up and release run in a short-lived domain,
    so the calling domain's current device is not changed.

    {b Backends.} HIP and CUDA (both the PTX and the C variants), which have
    non-blocking side streams: a backend has the keep-alive when it registered a
    provider with {!register_native} (the HIP and CUDA plugins do so when they
    are registered). On every other backend (OpenCL, Vulkan, Metal, Native,
    Interpreter) {!start} is a documented no-op that returns a handle whose
    {!status} is [Unsupported]: no domain, no stream, no launch.

    {b Failure.} A keep-alive is a performance hint, so it never raises: if
    set-up or a launch fails (no NVRTC/hiprtc, out of resources, no domain
    available, a launch error) the keep-alive stops and {!status} reports
    [Failed reason]. On CUDA the empty kernel is compiled with NVRTC, so a
    CUDA/PTX install without NVRTC reports [Failed].

    The keep-alive's launches bypass the backends' per-launch argument
    bookkeeping, so a [Device.synchronize] on the device does not touch them. *)

(** State of a keep-alive. *)
type status =
  | Running  (** The thread is launching the empty kernel periodically. *)
  | Suspended  (** Parked by {!suspend}: no launch, resources kept. *)
  | Stopped  (** Started and then stopped; the stream is destroyed. *)
  | Unsupported  (** This device's backend has no keep-alive: a no-op. *)
  | Failed of string  (** Set-up or a launch failed; the worker has exited. *)

type t

(** Smallest and largest accepted period, in microseconds. {!start} clamps its
    argument to this range, so it is total. The upper bound also bounds the
    interval at which a running worker notices {!stop}. *)
val min_period_us : int

val max_period_us : int

(** Period used when {!start} gets none: 500 microseconds. *)
val default_period_us : int

(** Whether the device's backend has registered a keep-alive provider (HIP and
    CUDA, once their plugin is registered). *)
val supported : Device.t -> bool

(** [start ?period_us dev] starts a keep-alive on [dev] and returns once the
    worker has finished set-up.

    Idempotent per device: while a keep-alive is [Running] on [dev], [start]
    returns it, and a [Suspended] one is resumed (the new [period_us] is ignored
    in both cases); the handle is shared, so a {!stop} from any holder stops it
    for all. After {!stop} (or a failure) a new call starts a fresh one;
    repeated start/stop cycles are supported and leak neither domains, streams
    nor kernel modules (the module is compiled once per device and cached). On
    an unsupported backend it returns an [Unsupported] handle without side
    effects.

    Thread-safe. A keep-alive still running at exit is stopped by an [at_exit]
    handler. *)
val start : ?period_us:int -> Device.t -> t

(** Stop launching and return once the thread is parked: no launch is made after
    [suspend] returns, until {!start} resumes it. The thread, stream and kernel
    are kept, so the next {!start} costs microseconds instead of the
    milliseconds of creating them (about 1-2 ms of set-up and 1 ms of tear-down
    were measured on HIP, which is what a short prompt would pay on every call).
    Idempotent; a no-op on a handle that is not [Running]. *)
val suspend : t -> unit

(** Stop the keep-alive, wait for its thread to exit, drain and destroy its
    stream. Idempotent, and safe on any status. When it returns the thread no
    longer touches the device. The thread notices the request within one period;
    the final drain lasts as long as the device takes to run the few empty
    kernels still queued, which is longer only if the device cannot schedule
    them (the side stream is stuck behind other work).

    Stop every keep-alive before [Kernel.clear_cache] or destroying the device:
    the thread uses a cached compiled module. *)
val stop : t -> unit

(** Current state. *)
val status : t -> status

val device : t -> Device.t

(** The period in effect, after clamping. *)
val period_us : t -> int

(** Number of empty kernels enqueued so far by this handle (0 when
    [Unsupported]). For tests and diagnostics. *)
val launches : t -> int

(** {2 For backends} *)

(** What a backend gives the native thread: the addresses of its driver's launch
    function and of a call that makes the right device current in a new thread,
    the argument of that call, and the native handles of the empty kernel and of
    a fresh non-blocking stream on the device.

    [launch_fn] must have the signature of [hipModuleLaunchKernel] and
    [cuLaunchKernel] (function, grid x y z, block x y z, shared bytes, stream,
    params, extra) and [prep_fn] that of an [int (intptr_t)] call returning 0 on
    success ([hipSetDevice] with the device index, [cuCtxSetCurrent] with the
    context). [release] drains and destroys the stream; it runs after the thread
    has stopped. *)
type native = {
  launch_fn : nativeint;
  prep_fn : nativeint;
  prep_arg : nativeint;
  kernel : nativeint;
  stream : nativeint;
  release : unit -> unit;
}

(** [register_native ~family provider]: devices whose framework is [family] or
    ["family/..."] get a keep-alive built by [provider]. [provider] runs in a
    short-lived domain, may select the device and may raise (the keep-alive then
    reports [Failed]). Registering again replaces the provider. *)
val register_native : family:string -> (Device.t -> native) -> unit

(** Name and CUDA/HIP C source of the empty kernel a provider compiles. *)
val kernel_name : string

val kernel_source : string
