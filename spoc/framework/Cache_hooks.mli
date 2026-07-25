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
    callbacks still run and the first exception is re-raised afterwards.

    {2 What actually notifies today — this back-channel is not yet universal}

    Registering a listener does {b not} mean every handle release will reach it.
    The current coverage is exactly:

    - {!notify_clear_all} is fired by [Sarek.Kernel.clear_cache] (on both sides
      of the backend clear) and by [Sarek.Device.reset]. Those are the only
      notifying entry points. A caller that resolves a backend through
      [Framework_registry.find_backend] and calls [B.Kernel.clear_cache ()]
      {e directly} releases the handles without notifying anything, and an outer
      memo will then serve a released handle — the same failure this module
      exists to prevent, through a sibling entry point. Go through
      [Sarek.Kernel.clear_cache].
    - {!notify_device_destroy} is fired by the CUDA and HIP backends only.
      Vulkan has a device-destroy path ([Vulkan_api_device.destroy]) and does
      {e not} fire it; its kernel cache also passes no [~device_id] to
      {!Guarded_cache.find_or_build}, so per-device eviction is not expressible
      there at either layer. OpenCL and Metal expose no device-destroy entry
      point at all, so there is nothing for them to notify from.

    Treat this list as the contract, not as an implementation detail: a listener
    that assumes total coverage is wrong today.

    {2 Notifying from a teardown path}

    Because the first callback failure is re-raised, [notify_*] can throw out of
    the middle of a device-destroy sequence, where escaping early would skip the
    release of the very resources the teardown exists to free. A backend calling
    these from [Device.destroy] must therefore
    {b complete its teardown first and re-raise afterwards} — capture the
    exception, unload modules, retire streams, destroy the context, then let it
    out. The CUDA and HIP backends do exactly this; copy that shape in any new
    backend. *)

(** [on_device_destroy f] registers [f], called as [f ~backend index] whenever a
    device is being torn down.

    [backend] is the backend {e family} name as used by [Device.init] /
    [Device.resolve_framework] (["CUDA"], ["HIP"], ...); a device's
    [Device.t.framework] either equals it or is a ["<family>/<variant>"]
    refinement of it (["CUDA/PTX"]), and a listener must treat both as matching.
    [index] is the backend's own device index, {e not} a global [Device.id]: the
    pair ([backend], [index]) is what identifies the device across id spaces.
    Matching on [index] alone aliases unrelated backends' devices. *)
val on_device_destroy : (backend:string -> int -> unit) -> unit

(** [notify_device_destroy ~backend index] runs every callback registered with
    {!on_device_destroy}. Called by a backend from its device-destroy path,
    before it releases anything — see "Notifying from a teardown path" above for
    the obligation that comes with the re-raise. *)
val notify_device_destroy : backend:string -> int -> unit

(** [on_clear_all f] registers [f] to be called whenever a backend's whole
    kernel cache is cleared. *)
val on_clear_all : (unit -> unit) -> unit

(** [notify_clear_all ()] runs every callback registered with {!on_clear_all}.
    Called before a backend clears its own cache. *)
val notify_clear_all : unit -> unit
