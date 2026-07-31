# A capability token for `Soa_vector.scatter` — refused, and what to build instead

**Task:** backlog-221 — "can an auto-sync phantom type state make the `scatter`
hazard unrepresentable, rather than merely refused at runtime?"

**Date:** 2026-07-31.

**Base:** the backlog-220 runtime refusal, on branch `backlog/220-scatter-refuse`.
At the time of writing that branch is local and unpushed, so it is named rather
than cited by sha — the sha would not resolve in a fresh clone, and an unlanded
branch may still be amended.

**Verdict: REFUSE.** No token, no phantom type. The exploration did not produce a
design worth building, and it did produce one thing worth having: **the shipped
refusal has a hole**, measured in §5. Fixing that hole is a one-line predicate
widening and is worth more than every type-level option considered here.

---

## 1. The question was asked about the wrong state variable

The framing that opened the item was:

> "once auto-sync is off the sync token is valid, we must make sure it becomes
> invalid if auto-sync is on again"

That describes a token whose validity is keyed to a **flag flip**. The mechanism
is not keyed to a flag flip. `Soa_vector.scatter` refuses on a **conjunction**
(`sarek/core/Soa_vector.ml:109-110`):

```ocaml
match (Vector.location t.aos, Vector.auto_sync t.aos) with
| Vector.Stale_CPU _, false -> raise (Soa.Unsupported ...)
```

and `Vector_transfer.ensure_cpu_sync` (`sarek/core/Vector_transfer.ml:44-51`) is
a no-op unless *both* halves say so. In `CPU`, `Both` and `Stale_GPU` the host
copy is already the freshest data, so scatter is safe **regardless of the flag**.

So a witness proving "auto-sync is on" is the wrong token in both directions:

- it would **refuse safe operations** — `check_sync_vectors_to_cpu_gathers`
  (`sarek/tests/e2e/test_soa_emitter_equiv.ml:1502`) deliberately calls
  `Vector.set_auto_sync sv false` at line 1511 while the vector is still at
  `CPU`, then launches, and the launch's scatter is correct;
- it would **permit nothing extra** — every state it admits is already admitted.

**The property a caller must establish is "the host copy is current."** It is
reachable via *either* a drain having run *or* the location already being
host-fresh. Any token must witness **freshness**, not a flag. Everything below is
about a freshness witness; the flag-shaped token is not evaluated further because
it is not the property.

The invalidation trigger is correspondingly not a flag flip but a **location
transition** — and the transitions are frequent and remote. See §4.

## 2. A phantom type on the vector is dead on arrival

Not because phantom types are weak, but for two independent reasons, either of
which is sufficient.

**(a) The indexed thing is mutable.** `location` is a mutable field
(`sarek/core_base/Spoc_core_base.mli:91` declares the variant; the field is
written in 6 production files). A phantom index is fixed when the value is
created; a device write changes the fact without changing the type. Measured
census of production writes — `grep -rn "location <- " --include=*.ml sarek/`
minus paths containing `/test`:

| RHS constructor | writes |
|---|---|
| `Stale_GPU` | 14 |
| `Both` | 7 |
| `CPU` | 3 |
| `Stale_CPU` | 2 |
| `GPU` | **0** |

Twenty-six production sites write the location field, none of them returning a
re-typed value.

**(b) Even a type-changing API could not be applied.** The usual escape — have
the freshness-establishing operation *return* a newly-typed handle — requires
that no stale-typed alias survive. Here they always survive, structurally.
`create_transparent` (`sarek/core/Soa_vector.ml:176-222`) builds a **reference
cycle**: `v = t.aos` is handed to the user, and `v.Vector.soa` is populated with
closures that capture `t`, which contains `v`. The `scatter t` inside
`soa_to_device` at line 211 closes over the *original* `t`, and `Execute.ml:310`
invokes that closure through the record. Re-typing `v` would leave it pointing at
the old value.

So the type index on the container is genuinely dead, and the item's own
suspicion was right.

## 3. The surviving shape, sketched concretely

A witness required at the API surface, produced by an operation that establishes
freshness, consumed by `scatter`:

```ocaml
(* in Spoc_core_base — it must live below Vector, see below *)
type host_fresh                                  (* abstract *)
val establish_host_fresh : ('a, 'b) t -> host_fresh option

(* in Soa_vector *)
val scatter : 'a t -> host_fresh -> unit
```

`establish_host_fresh` returns `Some` when the location is already host-fresh, or
after a successful drain; `None` otherwise, and the caller has nowhere to go.

**The layering works.** The witness type must be defined in `Spoc_core_base`,
because the field it has to thread through — `soa_scatter : unit -> unit`
(`sarek/core_base/Spoc_core_base.mli:147`, `.ml:180`) — is declared there, and
because `sarek/execute/jsoo/dune` compiles `Execute.ml` in a build where
`Soa_vector` does not exist. `Spoc_core_base.ml:654` already has its own
`ensure_cpu_sync`, so the producer has a home. Layering is not the obstacle.
(§6 finds that this field is in fact never invoked — which does not make the
sketch cheaper, only more pointless.)

## 4. Soundness: the witness can go stale, and in the launch path it *must*

**This is what refuses the design.** The witness is an ordinary OCaml value.
Upstream OCaml 5.4.0 (the switch this tree builds in — `ocamlfind ocamlopt
-version` → `5.4.0`) has no uniqueness or affinity modes; a `host_fresh` can be
bound, stored in a `ref`, captured in a closure, and used arbitrarily later.
Nothing in the type prevents:

```ocaml
let w = establish_host_fresh v in   (* location: CPU *)
launch v;                           (* location: Stale_CPU dev *)
scatter sv w                        (* w is now a lie *)
```

That is not a hypothetical misuse. It is **the shape the launch path already
has**, and the transition is performed by the SoA code itself:

- `soa_to_device` sets `v.Vector.location <- Vector.Stale_CPU dev`
  (`sarek/core/Soa_vector.ml:222`) at the end of every transparent launch;
- `Execute.ml:1038` sets `Stale_CPU dev` on any `Both` vector after a run;
- so a *second* launch of the same vector re-enters `scatter` from `Stale_CPU` —
  exactly the sequence `check_soa_then_soa_other_device`
  (`sarek/tests/e2e/test_soa_cross_device_migration.ml:180`) exists to pin.

Now apply the token to that path. The relevant closure is `soa_to_device`, which
contains the `scatter t` at `Soa_vector.ml:211`. It is built **once, at vector
creation**, and invoked **later, repeatedly**, by `Execute.ml:310`
(`b.Vector.soa_to_device dev`) through a function pointer. It has exactly two
options:

- **capture a witness produced at creation time.** By the second launch that
  witness is false, and the type signature says it is true. This is strictly
  worse than backlog-220: silent corruption returns, now wearing a proof.
- **produce a witness inside the closure, immediately before use.** Sound — and
  identical to the runtime check, because `establish_host_fresh` returning `None`
  at that point is the refusal, spelled differently.

**Every sound version of this token collapses into the runtime check.** The token
adds compile-time enforcement only for the gap between production and
consumption, and the only way to keep that gap safe in OCaml 5.4 is to make it
empty. A token that must be produced at its consumption site is not a capability;
it is an argument-shaped assertion.

A generation-counter variant (witness carries the location generation; `scatter`
compares) restores soundness — by re-introducing a runtime check that fires
*later and less legibly* than the one already shipped, and paying §6's cost for
it. Rejected on the same grounds.

## 5. What the exploration did find: the shipped refusal has a hole

Reading the predicate for §1 surfaced a live gap in the shipped refusal, and it
was **executed, not argued**.

"Auto-sync is off" has **two spellings** in this tree, and `Execute.ml:356`
already names them as equals: `Vector.set_auto_sync` (per-vector) and
`Transfer.disable_auto` (global, `sarek/core/Transfer.ml:24`, setting the
`auto_mode` ref at line 20). The registered sync callback consults the global one
first — `if not !auto_mode then false` (`Transfer.ml:719`) — and
`ensure_cpu_sync` **ignores that return value** (`ignore (cb.sync vec)`,
`Vector_transfer.ml:49`).

The refusal at `Soa_vector.ml:109` reads `Vector.auto_sync t.aos` only. So:

> `Stale_CPU ∧ per-vector auto_sync = true ∧ Transfer.auto_mode = false` is the
> backlog-220 corruption, **unrefused**.

`test_soa_cross_device_migration.ml:174-176` already states the precondition in
full — "`ensure_cpu_sync` is a no-op when `auto_sync` is false on the vector **or
auto mode is off globally**". The test knew; the refusal does not.

**Measured.** A temporary case added to
`sarek/tests/unit/soa_vector_scatter_refuse/`, run with `dune build
@sarek/tests/unit/soa_vector_scatter_refuse/runtest --force`, then reverted:

| case | output |
|---|---|
| `Transfer.disable_auto ()`, `Stale_CPU`, per-vector `auto_sync = true` | `scatter did NOT refuse` / `still Stale_CPU -> NO drain ran` |
| **control:** same state, `Transfer.enable_auto ()` | `scatter raised Failure("to_cpu: no device buffer to transfer from")` |

The control is the load-bearing half: with the global flag on, the drain path is
genuinely entered and reaches `Transfer.to_cpu`. With it off, `scatter` returns
normally having drained nothing. The two branches differ, so the observation
discriminates. (The control's `Failure` is the test's fake `Device.t` having no
buffer — that is what proves `to_cpu` was reached.)

**This is a one-line fix** to `Soa_vector.ml:109` — read the effective flag, not
the per-vector one — plus a test case. It closes a reachable silent-corruption
path. It should be its own backlog item; it is not filed here because backlog-220
is not to be modified under this task.

## 6. The cost, counted

If the token were built anyway. Commands: `grep -rnE "(Soa_vector\.)?scatter
(sv|t)\b" --include=*.ml sarek/ benchmarks/` filtered to non-test paths, and
`grep -rn "val scatter : 'a t -> unit\|soa_scatter : unit -> unit"`.

| | sites |
|---|---|
| production callers | **4** — `Soa_vector.ml:206`, `Soa_vector.ml:211`, `Soa_launch.ml:342`, `benchmarks/bench_soa_emitter.ml:123` |
| test callers | **5** — all in `test_soa_vector_scatter_refuse.ml` (80, 97, 108, 118, 139) |
| type-level sites | **3** — `Soa_vector.mli:113`, `Spoc_core_base.mli:147`, `Spoc_core_base.ml:180` |
| | **12 total** |

**The 4 is small — and that is the argument against, not for.** Taken one by
one, by whether each can actually reach `Stale_CPU`:

| caller | can it reach the hazard? |
|---|---|
| `Soa_vector.ml:211`, inside `soa_to_device` | **yes, demonstrated.** Line 222 leaves the vector at `Stale_CPU dev`, so a second launch re-enters here — the sequence `check_soa_then_soa_other_device` pins, with auto-sync on |
| `Soa_launch.ml:342`, inside `run_soa` | **not demonstrated either way.** `Soa_launch.ml` writes no location at all (`grep -n location sarek/execute/Soa_launch.ml` is empty), so nothing in that path is shown to produce `Stale_CPU` on the AoS vector |
| `benchmarks/bench_soa_emitter.ml:123` | no — one scatter, from `CPU`, after a host fill |
| `Soa_vector.ml:206`, the `soa_scatter` field | no — **the field is never invoked** |

That last row is worth stating on its own. `grep -rn "\.soa_scatter"
--include=*.ml --include=*.mli .` returns nothing: `soa_scatter` is populated at
`Soa_vector.ml:206`, declared twice in `Spoc_core_base`, and called from
**nowhere**. Its siblings are all live — `soa_to_device` at `Execute.ml:310`,
`soa_from_device` at `Transfer.ml:374` and `Execute.ml:390`, `soa_free_leaves` at
`Transfer.ml:584` and `:629`. `soa_gather` is dead too, by the same grep.

So **one caller is demonstrated to reach the hazard**, and it is inside the
module that would define the token.

**The 3 type-level sites are the expensive ones, and two of them are dead.**
`soa_scatter : unit -> unit` is a public record field in `Spoc_core_base`, the
lowest layer, and that record is the deliberate decoupling channel for the jsoo
build where `Execute.ml` compiles without `Soa_vector` (`Soa_vector.ml:165-168`).
Threading a witness through it is a breaking change to a layer boundary that
exists precisely so the two builds need not agree about SoA — paid on a field no
code calls. Paying that to guard one caller is the wrong trade.

(The dead field is an incidental finding, not this item's business. It is either
an unfinished API or removable; either way it should be looked at separately,
because a populated-but-uncalled closure in a public record is a standing
invitation to assume a code path exists.)

## 7. What it would buy over the shipped refusal — honestly, nothing

Backlog-220 converts silent data corruption into a named exception with two
stated remedies. A sound token converts a runtime error into a compile error
**only for call sites that could produce the witness at the call site** — which,
per §4, is the set of call sites where the runtime check is already correct and
sufficient. For the launch path, the only path where production and consumption
are separated, the token is unsound. The mechanism enforces nothing the runtime check does not,
at any boundary examined.

## 8. `GPU of dev` is unreachable — do not design around it

`Soa_vector.ml:100-104` flags pure `GPU` (no host buffer) as a pre-existing gap.
Established, not assumed: **zero production sites construct it.** The
`location <- ` census in §2 finds no `GPU` writes, and no record literal creates
one (`grep -rn "location = " --include=*.ml sarek/`: six literals, all `CPU`).
The single construction in the tree is a test poking the mutable field —
`sarek/core/test/test_vector_transfer.ml:186`. Production code only ever *reads*
`GPU` in defensive match arms.

So a token with a `GPU` case would carry a case nobody can test. Correctly
excluded, and the exclusion should stay excluded until something constructs it.

## 9. The alternative that is not a token

Worth naming because it is the real competitor: make `scatter` **drain
unconditionally** on `Stale_CPU` — `Transfer.to_cpu ~force:true` instead of
`ensure_cpu_sync` — eliminating the hazard rather than refusing it.
`soa_from_device` already uses exactly that force, for exactly this reason
(`Soa_vector.ml:223-233`).

**Rejected, but on a policy ground rather than a mechanical one.** Both
`set_auto_sync false` and `disable_auto` mean "do not move data behind my back";
a forced drain inside `scatter` overrides a user's explicit instruction. Loud
refusal is the right behaviour, and backlog-220 is it. Recorded so the next
reader does not have to re-derive why the cheaper fix was not taken.

## 10. Recommendation, and what would reverse it

**Refuse. Close backlog-221 without building.** Then:

1. **File the §5 hole** — the refusal must read the effective auto-sync state,
   not the per-vector flag. Reachable through public API, executed above,
   one-line fix. This is the entire actionable output of the item.
2. Leave the backlog-220 refusal alone otherwise.

Three things would reverse the refusal, in decreasing order of likelihood:

- **Affine or unique types in upstream OCaml.** If a `host_fresh` could be
  declared once-usable and non-storable, §4's gap closes by construction and the
  design becomes sound. The oxcaml mode extensions do this today; upstream 5.4
  does not, and this tree builds on upstream. This is the load-bearing blocker —
  everything else in §4 follows from it.
- **More callers reaching the hazard, outside `Soa_vector`.** The cost case in
  §6 rests on the demonstrated count being one, and on that one being internal.
  Several external callers that must establish freshness themselves would change
  the ergonomics arithmetic — though not the soundness one.
- **`location` becoming immutable** — a redesign where a transfer returns a new
  handle rather than mutating in place. That would revive the phantom type of §2
  and make the token unnecessary rather than possible. It is a much larger change
  than anything backlog-221 contemplates.

Note that the first two are independent of each other and the second is not
sufficient alone: more callers without affinity gives a more widely used unsound
token.

---

*The §5 measurement was executed on this worktree at the tip of
`backlog/220-scatter-refuse`, in the OCaml 5.4.0 switch, with a positive
control, and the patch was reverted — only this document is proposed. Every count in §2 and §6 comes from a `grep` named beside
it. Line citations were checked individually against the worktree, not validated
by `scripts/check-cited-paths-exist.sh`, which verifies the path token only.*
