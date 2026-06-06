---
language: ocaml
description: Good patterns and antipatterns for OCaml — type safety, effect discipline, and correctness.
version: 1.0.0
---

# OCaml Patterns

## Good Patterns

**Use `option` and `result` for absence and failure**
Return `'a option` or `('a, 'e) result` instead of raising exceptions for expected failure paths. Reserve exceptions for truly unexpected/unrecoverable errors.

**GADTs for invariants and state machines**
Encode legal state transitions in the type — a `closed` socket cannot be written to if the type forbids it. Use GADTs when phantom types are not expressive enough.

**Phantom types and newtypes via modules for domain values**
Wrap raw `string`/`int` in modules with abstract types: `module UserId : sig type t val of_string : string -> t end`. This prevents mixing up `user_id` and `order_id` at compile time.

**Exhaustive pattern matching**
Enable `-warn-error +8` (non-exhaustive match). Never use a catch-all `_` in a match when adding a new variant should force a review of all call sites.

**Monadic chaining with `let*`**
Use `Result.bind` / `let*` to chain fallible operations instead of nested `match` ladders.

**Pure logic, IO at the boundary**
Keep transformation functions pure (no `ref`, no `Printf`, no `read_line`). Call effectful code only at the top of the call stack or in explicitly named `_io` / `_effect` modules.

**`List.map` / `fold` / `filter` over imperative loops**
Prefer higher-order functions over `while`/`for` with mutable accumulators.

**Polymorphic variants only for open extensibility**
Use regular variants by default. Reach for polymorphic variants only when cross-module extensibility is the explicit goal.

---

## Antipatterns

**Exceptions for control flow**
`raise Not_found` or `failwith "unexpected"` as normal return paths — use `option`/`result` instead.

**`string` or `int` for domain identifiers**
`user_id : string` and `order_id : string` are indistinguishable to the compiler. Wrap them.

**`Obj.magic`**
Bypasses the type system entirely. Almost never justified; document exhaustively if used.

**`assert false` to silence exhaustiveness warnings**
Indicates a match that should have been total. Fix the type or the logic instead.

**Incomplete matches without `-warn-error`**
Silently ignoring new variants as the codebase grows is a latent bug factory.

**`ignore` to discard `result` or `unit` returns**
If a function returns `result`, ignoring it hides errors. Handle or explicitly propagate.

**Mutable `ref` inside pure logic**
A function with `let acc = ref []` buried inside it is not pure. Extract the mutation or use a fold.

**`List.hd` / `List.tl` without guard**
These raise on empty lists. Use pattern matching instead.
