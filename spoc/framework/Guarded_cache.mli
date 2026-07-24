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
    backend resource held by an evicted or cleared value. *)

(** A guarded cache from keys ['k] to compiled values ['v]. *)
type ('k, 'v) t

(** [create ~destroy ()] builds an empty cache. [destroy] is applied to every
    value removed by {!clear} or {!evict_device} (and to the loser of a rare
    concurrent double-compile in {!find_or_build}); it should release the
    backend resource the value owns (e.g. [cuModuleUnload]). [destroy] is always
    invoked {e outside} the cache lock, so it must not itself call back into
    this cache. [size] is the initial table size (default 16). *)
val create : ?size:int -> destroy:('v -> unit) -> unit -> ('k, 'v) t

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
    eviction. *)
val find_or_build : ('k, 'v) t -> key:'k -> ?device_id:int -> (unit -> 'v) -> 'v

(** [evict_device t device_id] removes and {!destroy}s every value recorded for
    [device_id] via {!find_or_build}'s [~device_id]. Safe to call from a
    device-destroy hook: it acquires the lock only to snapshot and unlink the
    affected entries, then releases it before running [destroy], so it never
    re-enters the cache while holding the lock. A no-op for an unknown
    [device_id]. *)
val evict_device : ('k, 'v) t -> int -> unit

(** [clear t] removes and {!destroy}s every cached value and forgets all
    per-device key tracking. Values are snapshotted under the lock and destroyed
    after it is released. *)
val clear : ('k, 'v) t -> unit

(** [length t] is the number of entries currently cached (under the lock).
    Intended for tests and diagnostics. *)
val length : ('k, 'v) t -> int
