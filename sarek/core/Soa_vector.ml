(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* See Soa_vector.mli for the design rationale. *)

type packed_leaf = Leaf : ('e, 'f) Vector.t -> packed_leaf

type 'a t = {
  aos : ('a, unit) Vector.t;
  plan : Soa.plan;
  leaves : packed_leaf array;
  length : int;
}

(* A leaf host buffer is a bit-preserving byte transport sized to the leaf's
   scalar width; Soa.scatter/gather copy raw 4/8-byte words, so an int32/int64
   scalar vector of the right width is a correct container for any leaf type
   (f32/f64/i32/i64). *)
let leaf_vector_of_size length size : packed_leaf =
  match size with
  | 4 -> Leaf (Vector.create Vector.int32 length)
  | 8 -> Leaf (Vector.create Vector.int64 length)
  | _ ->
      invalid_arg
        (Printf.sprintf
           "Soa_vector: unsupported leaf byte size %d (expected 4 or 8)"
           size)

(* The leaf layout is DERIVED from the element type, never supplied. [create]
   used to take a [~fields] list, on the stated premise that "the PPX
   [custom_type] carries no layout" — which stopped being true once
   [custom_type.ir_fields] landed. The PPX populates it for every
   [[@@sarek.type]] record (Sarek_ppx.ml), from the same
   [aligned_record_offsets] call that produces [elem_size]/[get]/[set], and
   [test_ir_fields.ml] pins that agreement against the bytes [set] actually
   writes.

   Deriving it removes a hazard rather than documenting one: a caller-supplied
   list that disagreed with the real record (wrong order, wrong widths,
   missing/extra field) made scatter/gather transpose against the wrong byte
   offsets, which is silently corrupted data and not an error. That failure mode
   is now unreachable — there is no longer a second description of the layout to
   disagree with the first.

   [ir_fields = None] means "no derivable flat-scalar layout", which the
   [custom_type] doc requires consumers to read as "no SoA". The only producer
   that sets it is the variant deriver, and [Soa.plan] rejects a non-flat-record
   anyway, so this refuses exactly the element types SoA could never represent. *)
let create (custom : 'a Vector.custom_type) (length : int) : 'a t =
  let fields =
    match custom.Vector.ir_fields with
    | Some fields -> fields
    | None ->
        raise
          (Soa.Unsupported
             (Printf.sprintf
                "element type %S has no derivable flat-record layout \
                 (custom_type.ir_fields is None), so it cannot be stored as \
                 Structure-of-Arrays"
                custom.Vector.name))
  in
  let plan = Soa.plan ~name:custom.Vector.name fields in
  let aos = Vector.create_custom custom length in
  let leaves =
    Array.of_list
      (List.map
         (fun (l : Soa.leaf) -> leaf_vector_of_size length l.Soa.size)
         plan.Soa.leaves)
  in
  {aos; plan; leaves; length}

let aos_vector t = t.aos

let plan t = t.plan

let leaves t = t.leaves

let num_leaves t = Array.length t.leaves

let length t = t.length

let set t i v = Vector.set t.aos i v

let get t i = Vector.get t.aos i

let leaf_ptrs t = Array.map (fun (Leaf v) -> Vector.to_ctypes_ptr v) t.leaves

(* backlog-220. [Vector.ensure_cpu_sync] below is a no-op whenever auto-sync is
   off, REGARDLESS of whether the host copy is actually behind a device's data
   ([Vector_transfer.ensure_cpu_sync] tests [auto_sync] before it even looks at
   [location]). So on [Stale_CPU] — the state a device write leaves the AoS
   vector in — an auto-sync-off [scatter] used to transpose the STALE host
   bytes into the leaves and return [unit], indistinguishable from a correct
   call. That is silent data corruption, not a no-op: every subsequent read of
   this SoA vector's leaves (and anything transferred from them to a device)
   is now wrong, with nothing in the return type or the logs to say so.

   [CPU]/[Both]/[Stale_GPU] are unaffected because the host buffer there
   already IS the fresh data. Pure [GPU] (no host buffer written yet) is a
   pre-existing gap [ensure_cpu_sync] never covered even with auto-sync on —
   nothing constructs that state for an AoS vector in this codebase today, and
   this refusal does not attempt to cover it either, so it is excluded from
   the match rather than silently folded into the same message. Only
   [Stale_CPU] means "a device holds newer data than what scatter is about to
   read", so only that state is refused. *)
let scatter t =
  (match (Vector.location t.aos, Vector.auto_sync t.aos) with
  | Vector.Stale_CPU _, false ->
      raise
        (Soa.Unsupported
           "Soa_vector.scatter: this vector's host data is out of date and \
            auto-sync is off, so scattering now would copy stale values into \
            this vector's per-leaf host buffers with no error. Before \
            scattering, either call Transfer.to_cpu on its AoS vector \
            (Soa_vector.aos_vector) to refresh the host copy, or call \
            Vector.set_auto_sync on it to turn auto-sync back on.")
  | (Vector.CPU | Vector.GPU _ | Vector.Both _ | Vector.Stale_GPU _), _
  | Vector.Stale_CPU _, true ->
      ()) ;
  (* Make the AoS host copy authoritative before transposing out of it. *)
  Vector.ensure_cpu_sync t.aos ;
  Soa.scatter
    t.plan
    ~aos:(Vector.to_ctypes_ptr t.aos)
    ~length:t.length
    ~leaves:(leaf_ptrs t) ;
  (* The transpose above writes each leaf's HOST buffer through a raw ctypes
     pointer, which performs no location bookkeeping. So without this a leaf
     keeps whatever location it already had — and after a previous launch that
     is [Both dev], on which [Transfer.to_device] logs "skip (Both)" and returns
     without copying. A SECOND launch on the same vector then ran against the
     FIRST launch's device data, silently, with no user workaround.

     Marking [Stale_GPU dev] is the accurate statement and not a nudge: the host
     copy is the one just written and the device copy is now out of date. It is
     also the one location [to_device] does not short-circuit.

     Only leaves that actually HAVE a buffer on a device are touched; a leaf
     never transferred stays [CPU], which is already correct.

     Done here rather than in the [soa_to_device] closure so the explicit
     [Soa_launch.run_soa] path gets it too — both callers scatter through this
     function, and this invariant must not have two answers. *)
  Array.iter
    (fun (Leaf lv) ->
      match lv.Vector.location with
      | Vector.Both d | Vector.GPU d | Vector.Stale_CPU d ->
          lv.Vector.location <- Vector.Stale_GPU d
      | Vector.CPU | Vector.Stale_GPU _ -> ())
    t.leaves

let gather t =
  Soa.gather
    t.plan
    ~leaves:(leaf_ptrs t)
    ~length:t.length
    ~aos:(Vector.to_ctypes_ptr t.aos)

(* ── Transparent SoA: a plain Vector the generic launch path can dispatch on ──
   backlog-54. Returns the AoS vector with its [soa] binding populated, so
   [Execute.run] can bind the N-leaf ABI without ever naming this module.

   Why that indirection exists: sarek/execute/jsoo/dune copies Execute.ml but
   NOT Soa_launch.ml, so Execute.ml is compiled in a build where Soa_vector does
   not exist. It therefore cannot call into here — the binding's closures are the
   only channel, which is why they are closures and not a plan.

   Why the constructor is HERE and not [Vector.create ~layout:SoA] as the Tier 1b
   handoff proposed: Soa_vector depends on Vector, so putting it there would be a
   layer inversion. The transparency that the item is actually about is at the
   LAUNCH site — the user calls the generic Execute.run and the N-pointer ABI is
   selected for them — and that is unaffected by which module names the
   constructor. *)
let create_transparent (custom : 'a Vector.custom_type) (length : int) :
    ('a, unit) Vector.t =
  let t = create custom length in
  let leaf_buf (dev : Device.t) (Leaf lv) =
    match Vector.get_buffer lv dev with
    | Some b -> b
    | None ->
        (* Unreachable via the launch path: soa_to_device runs first and
           allocates every leaf. Loud rather than silent, because a missing leaf
           buffer would otherwise bind a short argument list to an N-pointer
           signature — the kernel would then read whatever followed. *)
        invalid_arg
          (Printf.sprintf
             "Soa_vector: leaf buffer not allocated on device %d; \
              soa_to_device must run before soa_leaf_bufs"
             dev.Device.id)
  in
  let v = t.aos in
  (* SET only by [soa_to_device] below. Three other writers can clear or narrow
     it: [soa_free_leaves], also in this record, narrows it; and
     [Execute.transfer_vectors_to_device] and [Transfer.to_device] clear it. That
     list is exhaustive — see [soa_leaves_live]'s doc in Spoc_core_base.ml. No
     reader re-derives "SoA or AoS?" from the vector's shape; that is the point of
     the flag, so that read-back follows the launch's ABI decision. *)
  let leaves_live = ref false in
  v.Vector.soa <-
    Some
      {
        Vector.soa_num_leaves = num_leaves t;
        soa_aos_stride = t.plan.Soa.aos_stride;
        soa_scatter = (fun () -> scatter t);
        soa_gather = (fun () -> gather t);
        soa_leaves_live = leaves_live;
        soa_to_device =
          (fun dev ->
            scatter t ;
            Array.iter (fun (Leaf lv) -> Transfer.to_device lv dev) t.leaves ;
            leaves_live := true ;
            (* The AoS vector itself gets no device buffer under this ABI — the
               leaves hold the data — so without this its location stays [CPU]
               and EVERY read-back path short-circuits: [Transfer.to_cpu],
               [Transfer.sync] and the auto-sync callback all treat [CPU] as
               "nothing on a device to fetch" and return before looking at the
               SoA binding at all. Recording the device as authoritative and the
               host copy as stale is not bookkeeping; it is what makes the
               read-back reachable. *)
            v.Vector.location <- Vector.Stale_CPU dev);
        soa_from_device =
          (fun _dev ->
            (* [~force:true]: after a launch the leaves are [Vector.Both dev] —
               the host copy is the pre-launch scatter, the device copy is the
               result — and an unforced [to_cpu] treats [Both] as already in
               sync and returns without reading anything back. That is exactly
               the silent-stale-output failure this closure exists to prevent,
               so the force is load-bearing and not defensive. *)
            Array.iter
              (fun (Leaf lv) -> Transfer.to_cpu ~force:true lv)
              t.leaves ;
            gather t);
        soa_free_leaves =
          (fun dev ->
            (* No read-back here. The CALLER (Transfer.free_buffer /
               free_all_buffers) drains through [read_back_to_host] first, and
               doing it again here would either be redundant or — once the flag
               below is cleared — read the packed buffer instead. One drain, at
               the site that owns the ordering.

               [Transfer.free_buffer]/[free_all_buffers] on a LEAF, not
               [B.free ()] directly: the leaves are ordinary scalar vectors, so
               they get the location bookkeeping and the [Gpu_memory.track_free]
               accounting for free, and a leaf whose buffer is already gone is a
               no-op rather than a double free. *)
            (match dev with
            | Some d ->
                Array.iter (fun (Leaf lv) -> Transfer.free_buffer lv d) t.leaves
            | None ->
                Array.iter
                  (fun (Leaf lv) -> Transfer.free_all_buffers lv)
                  t.leaves) ;
            (* The freed leaves hold nothing a read-back could fetch, so the flag
               that says "the leaves are authoritative" must stop saying it —
               otherwise the next read-back follows freed buffers.

               DERIVED from what is still allocated, not assigned [false]. The
               previous version cleared it unconditionally and justified that
               with "this binding has one flag for the whole vector, so there is
               no per-device answer to preserve". The premise is right and the
               conclusion is backwards: precisely BECAUSE the flag covers the
               whole vector, a per-device free ([Some d]) must not clear it while
               leaves are still live on another device. Measured after a
               migration — leaves on A, packed buffer on B — [free_buffer sv B]
               released B's half and then reported the whole vector leaf-free
               with A's leaves still allocated. [Transfer.has_device_data sv A]
               then answered [false], so the drain-before-free in
               [Transfer.free_buffer]/[free_all_buffers] skipped A: freeing a
               device the results were NOT on discarded them.

               One rule for both cases, so there is no [None] special case to
               drift: after freeing, the leaves are authoritative iff they
               already WERE and some leaf still has a device buffer somewhere.
               For [None] every leaf was just released on every device, so the
               second conjunct is [false] — and it is so by observing the leaves
               rather than by asserting what the loop above was supposed to have
               done.

               The first conjunct is not redundant, and leaving it out was a
               resurrection bug: allocation is not authority. A packed launch
               CLEARS this flag after gathering the leaves into host storage
               (Execute.transfer_vectors_to_device), and the leaves it gathered
               stay allocated on their device — so deriving from allocation alone
               turned the flag back ON for leaves that now hold PRE-packed-launch
               data. Measured (test_soa_cross_device_migration,
               "a free does not resurrect stale leaf ownership"): SoA launch on
               A, packed launch on B, [free_buffer sv B], packed launch again —
               the last launch gathered A's stale leaves over the host copy of
               B's result and ran on them. Narrowing only: this can clear the
               flag, never set it. *)
            leaves_live :=
              !leaves_live
              && Array.exists
                   (fun (Leaf lv) ->
                     Hashtbl.length lv.Vector.device_buffers > 0)
                   t.leaves);
        soa_leaf_bufs =
          (fun dev -> Array.to_list (Array.map (leaf_buf dev) t.leaves));
      } ;
  v
