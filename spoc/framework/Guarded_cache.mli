(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Concurrency-safe kernel-compilation cache, shared by every GPU backend.

    Each backend memoizes compiled kernels in a [(string, t) Hashtbl.t] keyed by
    a {!Compile_cache.make_key} string. Under OCaml 5 multi-domain execution
    (now a supported use case) an unsynchronized [Hashtbl] is a data race: two
    domains inserting concurrently can corrupt the table's internal buckets
    during a resize, or a lookup can observe a half-linked bucket and raise.
    This module wraps the table behind a single {!Mutex.t} so cache lookup,
    insertion, per-device key tracking, eviction, and clearing are all atomic
    critical sections, while the expensive JIT compile
    ([nvrtc]/[hiprtc]/[clBuildProgram]/ shader compilation) runs {e outside} the
    lock.

    The value type ['v] (resolved module/function handles, compiled pipelines,
    SPIR-V-backed objects, ...) and the key type ['k] differ per backend, so the
    cache is parametric in both. A per-cache [destroy] callback releases the
    backend resource held by an evicted or cleared value.

    {2 Value lifetime is the caller's obligation}

    What this module makes concurrency-safe is the {b table}, not the
    {b values}. A value returned by {!find_or_build} escapes the lock, and
    {!clear} / {!evict_device} run [destroy] outside the lock by design (so
    [destroy] may re-enter the cache). Nothing here refcounts a value or defers
    its release, so this interleaving is {e not} prevented:

    - domain A resolves a kernel from the cache and enters [Kernel.launch];
    - domain B calls [clear_cache], which unlinks that same value and runs
      [destroy] on it ([clReleaseKernel] / [cuModuleUnload] /
      [vkDestroyPipeline]);
    - domain A's launch dereferences a released driver object.

    The contract is therefore:
    {b a cached value must not be released while any domain may still be using
       it.} Callers must not call {!clear} or {!evict_device} concurrently with
    work that is using values obtained from the same cache — quiesce (or
    externally serialize) the users of the cache first. Backend teardown paths
    satisfy this by running device destroy / cache clear as a program-wide
    quiescent operation.

    Adding refcounting or deferred destroy here would remove the obligation;
    that is a deliberate, tracked follow-up rather than part of the current
    contract. Note that the concurrency tests only assert that nothing raises,
    which by construction cannot detect a violation of this obligation. *)

(** A guarded cache from keys ['k] to compiled values ['v]. *)
type ('k, 'v) t

(** Raised by {!find_or_build} when the [device_id] it was building for was
    retired by {!evict_device} while the (unlocked) build was in flight. The
    freshly built value is [destroy]ed rather than installed: the device it was
    compiled against is gone, so no valid value can be returned. Callers that
    tear devices down concurrently with compilation must handle this (retry
    against a live device, or propagate). *)
exception Device_destroyed_during_build of int

(** Raised by {!find_or_build} on a cache created with
    [~invalidated_by_clear:true] when {!clear} ran while the (unlocked) build
    was in flight. The value borrows handles that the clear released, so it is
    [destroy]ed rather than installed. Callers may retry: a fresh build after
    the clear has completed will borrow live handles again. *)
exception Cache_cleared_during_build

(** [create ~destroy ()] builds an empty cache. [destroy] is applied to every
    value removed by {!clear} or {!evict_device} (and to the loser of a rare
    concurrent double-compile in {!find_or_build}); it should release the
    backend resource the value owns (e.g. [cuModuleUnload]). [destroy] is always
    invoked {e outside} the cache lock, so it may re-enter this cache.

    [destroy] is allowed to raise (some backends' release calls go through an
    error check that does). One failing [destroy] never prevents the remaining
    victims of a {!clear} / {!evict_device} from being released: every victim is
    attempted and the first exception is re-raised afterwards.

    On every path in {!find_or_build} that discards a value it has just built —
    losing the race, {!Device_destroyed_during_build},
    {!Cache_cleared_during_build} — a failing [destroy] of that discarded value
    is dropped rather than raised. On the lost-race path this keeps a release
    error from turning a successfully resolved [winner] into an intermittent,
    multi-domain-only compile failure; on the other two it keeps the exception
    the caller sees the one that describes {e why} no value could be returned,
    rather than a secondary release error from the cleanup. In all three cases
    the release error is not reported at all, which is a deliberate trade
    against masking the primary outcome.

    [size] is the initial table size (default 16).

    [invalidated_by_clear] (default [false]) declares whether the cached values
    {b own} the resources they refer to or merely {b borrow} them from a cache
    underneath, and it changes what {!clear} means for a build that is already
    in flight:

    - [false] — {b owning} cache (every backend's kernel cache). [clear]
      releases the handles this cache owns; it says nothing about the device, so
      a value built while the clear was running is still valid and is installed
      normally.
    - [true] — {b borrowing} cache (a memo layered on top of another cache, e.g.
      [Sarek.Runtime]'s outer memo of [Kernel.t] closures over backend handles).
      For such a cache, [clear] is the signal that the borrowed handles are
      about to be released. A build that started before the clear and finishes
      after it closes over a handle that is dead by the time it would be
      installed — and, because the clear has already walked the table without
      seeing the not-yet-inserted entry, that dead value would then be served
      for the rest of the process, outliving the teardown that was in progress.
      With the flag set, such a build is [destroy]ed and
      {!Cache_cleared_during_build} is raised instead.

    Set it iff the values borrow. Setting it on an owning cache turns benign
    concurrent clears into spurious build failures; leaving it off on a
    borrowing cache is a use-after-free. *)
val create :
  ?size:int ->
  ?invalidated_by_clear:bool ->
  destroy:('v -> unit) ->
  unit ->
  ('k, 'v) t

(** [find_or_build t ~key ?device_id build] returns the cached value for [key],
    compiling it with [build] on a miss.

    Lock discipline: the cache is checked under the lock; on a hit the value is
    returned immediately. On a miss the lock is {e released} before [build] runs
    (JIT compilation must never hold the cache lock), then re-acquired to
    insert. A double-check on re-acquire handles the case where another domain
    compiled the same key while the lock was released: the value already present
    wins and is returned, and the just-built loser is passed to the cache's
    [destroy] (outside the lock) so no backend resource leaks. This means, under
    a race, the same key may be compiled more than once, but at most one value
    is ever retained and every caller observes that single value.

    If [device_id] is supplied, [key] is recorded against that device so
    {!evict_device} can later retire exactly the entries compiled for it (used
    by backends whose device-destroy hook must unload modules before the
    underlying context is torn down). Omit it for caches with no per-device
    eviction.

    Supplying [device_id] also makes the build window {e eviction-aware}: the
    device's eviction epoch is read under the lock at miss time and re-checked
    on re-acquire. If {!evict_device} ran for that device in between, the built
    value is [destroy]ed instead of installed and
    {!Device_destroyed_during_build} is raised — an entry that is not yet in the
    table cannot be seen by {!evict_device}, so without this check the value
    would be installed against a dead device id (never unloaded, and served to
    later lookups, including after the id is recreated).

    On a cache created with [~invalidated_by_clear:true] the cache-wide
    generation is snapshotted and re-checked the same way, raising
    {!Cache_cleared_during_build}. This check does not need [device_id]: a clear
    invalidates every in-flight build on such a cache, not just one device's. *)
val find_or_build : ('k, 'v) t -> key:'k -> ?device_id:int -> (unit -> 'v) -> 'v

(** [evict_device t device_id] removes and {!destroy}s every value recorded for
    [device_id] via {!find_or_build}'s [~device_id]. Safe to call from a
    device-destroy hook: it acquires the lock only to snapshot and unlink the
    affected entries, then releases it before running [destroy], so it never
    re-enters the cache while holding the lock. Also bumps [device_id]'s
    eviction epoch, so a {!find_or_build} currently building for it discards its
    result (see {!find_or_build}); this is why the call is {e not} a no-op for a
    [device_id] with no cached entries. *)
val evict_device : ('k, 'v) t -> int -> unit

(** [clear t] removes and {!destroy}s every cached value and forgets all
    per-device key tracking. Values are snapshotted under the lock and destroyed
    after it is released. Also bumps the cache-wide generation, which aborts
    in-flight builds iff the cache was created with [~invalidated_by_clear:true]
    (see {!create}). *)
val clear : ('k, 'v) t -> unit

(** [length t] is the number of entries currently cached (under the lock).
    Intended for tests and diagnostics. *)
val length : ('k, 'v) t -> int
