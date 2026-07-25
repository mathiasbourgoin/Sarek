(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Cross-layer cache-teardown notifications.

    Compiled kernels are memoized at {e two} levels: each backend keeps its own
    {!Guarded_cache} of resolved backend handles, and [Sarek.Runtime] keeps an
    outer memo of [Kernel.t] closures {e over} those handles. Only the backend
    layer owns the handles, so only it releases them — but a release performed
    there invalidates every outer layer that closed over the handle, and the
    outer layers sit in libraries the backend cannot reference (they are loaded
    the other way round). This module is the missing back-channel: a layer above
    the backends registers an invalidation callback here, and the backends fire
    it when they are about to release handles.

    Without it, an outer memo keeps serving [Kernel.t] values whose backend
    handle has already been [clReleaseKernel]d / [cuModuleUnload]ed, and the
    next launch dereferences freed driver memory.

    Callbacks registered here must only {e drop memoization} — they must not
    release backend resources, since they do not own any. Registration and
    notification are safe from any domain: the registry is mutex-guarded and
    callbacks run outside the lock. If a callback raises, the remaining
    callbacks still run and the first exception is re-raised afterwards. *)

(** [on_device_destroy f] registers [f] to be called with a {e backend-local}
    device id whenever that device is being torn down. Note the id is the
    backend's own device index, not a global [Device.id]: a listener that keys
    by a different id space must widen (never narrow) its invalidation
    accordingly. *)
val on_device_destroy : (int -> unit) -> unit

(** [notify_device_destroy backend_device_id] runs every callback registered
    with {!on_device_destroy}. Called by a backend from its device-destroy path,
    before it releases anything. *)
val notify_device_destroy : int -> unit

(** [on_clear_all f] registers [f] to be called whenever a backend's whole
    kernel cache is cleared. *)
val on_clear_all : (unit -> unit) -> unit

(** [notify_clear_all ()] runs every callback registered with {!on_clear_all}.
    Called before a backend clears its own cache. *)
val notify_clear_all : unit -> unit
