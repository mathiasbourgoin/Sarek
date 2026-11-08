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

    {b How.} {!start} spawns a domain that owns a non-blocking stream and an
    empty kernel on the device, and enqueues the kernel every [period_us]
    microseconds. A separate domain, not a systhread, so that a main domain
    running OCaml code for tens of milliseconds cannot delay the launches. The
    main domain is not otherwise involved: it never waits on the keep-alive
    except in {!start} (set-up) and {!stop} (join). The calling domain's current
    device is not changed. The domain takes a short stop-the-world pause about
    once a second at the default period (the backend's launch path allocates).

    {b Backends.} HIP and CUDA (both the PTX and the C variants), which have
    non-blocking side streams. On every other backend (OpenCL, Vulkan, Metal,
    Native, Interpreter) {!start} is a documented no-op that returns a handle
    whose {!status} is [Unsupported]: no domain, no stream, no launch.

    {b Failure.} A keep-alive is a performance hint, so it never raises: if
    set-up or a launch fails (no NVRTC/hiprtc, out of resources, no domain
    available) the worker stops and {!status} reports [Failed reason]. *)

(** State of a keep-alive. *)
type status =
  | Running  (** The domain is launching the empty kernel periodically. *)
  | Stopped  (** Started and then stopped; the stream is destroyed. *)
  | Unsupported  (** This device's backend has no keep-alive: a no-op. *)
  | Failed of string  (** Set-up or a launch failed; the worker has exited. *)

type t

(** Smallest and largest accepted period, in microseconds. {!start} clamps its
    argument to this range, so it is total. The upper bound also bounds the
    latency of {!stop}. *)
val min_period_us : int

val max_period_us : int

(** Period used when {!start} gets none: 500 microseconds. *)
val default_period_us : int

(** Whether the device's backend implements the keep-alive (HIP and CUDA). *)
val supported : Device.t -> bool

(** [start ?period_us dev] starts a keep-alive on [dev] and returns once the
    worker has finished set-up.

    Idempotent per device: while a keep-alive is [Running] on [dev], [start]
    returns it (the new [period_us] is ignored). After {!stop} (or a failure) a
    new call starts a fresh one; repeated start/stop cycles are supported and
    leak neither domains, streams nor kernel modules (the module is compiled
    once per device and cached). On an unsupported backend it returns an
    [Unsupported] handle without side effects.

    Thread-safe. A keep-alive still running at exit is stopped by an [at_exit]
    handler. *)
val start : ?period_us:int -> Device.t -> t

(** Stop the keep-alive, wait for its domain to exit, drain and destroy its
    stream. Idempotent, and safe on any status. When it returns the worker no
    longer touches the device. *)
val stop : t -> unit

(** Current state. *)
val status : t -> status

val device : t -> Device.t

(** The period in effect, after clamping. *)
val period_us : t -> int

(** Number of empty kernels enqueued so far by this handle (0 when
    [Unsupported]). For tests and diagnostics. *)
val launches : t -> int
