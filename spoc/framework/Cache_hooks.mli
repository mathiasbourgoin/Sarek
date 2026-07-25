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

    {2 What actually notifies today}

    - {!notify_clear_all} is fired by {!around_clear}, which wraps the body of
      {e every} backend's [Kernel.clear_cache] — so it fires whether the caller
      goes through [Sarek.Kernel.clear_cache] or resolves the backend itself
      through [Framework_registry.find_backend] and calls
      [B.Kernel.clear_cache ()] directly. It is also fired by
      [Sarek.Device.reset], which retires the global device id space without
      clearing anything.
    - {!notify_device_destroy} is fired by every backend that has a
      device-destroy entry point: CUDA ([Cuda_api.Device.destroy]), HIP
      ([Hip_api.Device.destroy]) and Vulkan ([Vulkan_api_device.destroy]).
      OpenCL and Metal expose no device-destroy entry point at all, so there is
      nothing for them to notify {e from} — but their kernel caches register the
      same listener as everyone else, so a notification aimed at them is
      honoured if one is ever fired.

    Every backend kernel cache is per-device on both sides: its key carries the
    backend-local device index (via {!Compile_cache.make_key}) and it passes
    that same index as [~device_id] to {!Guarded_cache.find_or_build}, so
    {!Guarded_cache.evict_device} can reach its entries. Listeners may rely on
    that uniformity.

    {2 Backend obligations}

    A new backend must, for its kernel cache:
    - wrap its [Kernel.clear_cache] body in {!around_clear};
    - pass [~device_id] (the backend-local index, the same one that goes into
      the cache key) to {!Guarded_cache.find_or_build};
    - register an {!on_device_destroy} listener that evicts that index when
      [~backend] equals its own family name;
    - fire {!notify_device_destroy} from its device-destroy path, if it has one,
      {e before} it releases anything — see below.

    {2 Notifying from a teardown path}

    Because the first callback failure is re-raised, [notify_*] can throw out of
    the middle of a device-destroy sequence, where escaping early would skip the
    release of the very resources the teardown exists to free. A backend calling
    these from [Device.destroy] must therefore
    {b complete its teardown first and re-raise afterwards} — capture the
    exception, unload modules, retire streams, destroy the context, then let it
    out. The CUDA, HIP and Vulkan backends do exactly this; copy that shape in
    any new backend. *)

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
    Prefer {!around_clear}, which fires it on both sides of the clear. *)
val notify_clear_all : unit -> unit

(** [around_clear clear] runs [clear] with a {!notify_clear_all} on {e each}
    side of it, and is how a backend's [Kernel.clear_cache] must be written.
    Putting the notification here rather than in the caller is what makes a
    direct [B.Kernel.clear_cache ()] safe: the handles are never released
    without the layers above being told.

    Both notifications are load-bearing. The first drops what is already
    memoized, but cannot cover an outer build that {e starts} after it: such a
    build snapshots the new generation, closes over a handle [clear] then
    releases, and on re-acquire would find the generation unchanged and install
    a value over a dead handle. The second bumps the generation again, so any
    build spanning the release is rejected (see {!Guarded_cache.find_or_build}'s
    [~invalidated_by_clear]).

    A listener failure is isolated from [clear] and re-raised only after it has
    completed: a listener owns no handles, so letting it escape early would skip
    the release [clear] exists to perform and leak every backend handle. If
    [clear] itself raises, that exception wins.

    Nested calls on the same domain notify once, not twice per level:
    [Sarek.Kernel.clear_cache] wraps the backend's own [clear_cache], and the
    contract is two notifications per teardown, not four. Nesting is tracked per
    domain, so a concurrent clear on another domain still notifies. *)
val around_clear : (unit -> unit) -> unit
