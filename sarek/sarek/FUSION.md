# Kernel Fusion

Sarek provides automatic kernel fusion to eliminate intermediate arrays and reduce memory traffic.

## Overview

When you have a pipeline of kernels where one writes to an array that another reads:

```ocaml
(* Before fusion: 2 kernel launches, temp array allocated *)
let producer = [%kernel fun temp input -> temp.(i) <- input.(i) * 2]
let consumer = [%kernel fun output temp -> output.(i) <- temp.(i) + 1]

(* After fusion: 1 kernel launch, no temp array *)
let fused = [%kernel fun output input -> output.(i) <- (input.(i) * 2) + 1]
```

Fusion inlines the producer's computation into the consumer, eliminating:
- The intermediate array allocation
- Memory bandwidth for writing and reading the intermediate
- Kernel launch overhead

## Access Patterns

The fusion system analyzes array access patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| OneToOne | Same index for read/write | `arr.(i)` |
| Stencil | Neighboring elements | `arr.(i-1) + arr.(i) + arr.(i+1)` |
| Reduction | Fold over array | `for j: acc = acc + arr.(j)` |
| Gather | Indirect indexing | `arr.(indices.(i))` |

## Fusion Types

### Vertical Fusion (OneToOne → OneToOne)

The simplest case: both producer and consumer use the same index.

```ocaml
(* temp[i] = input[i] * 2 *)
(* output[i] = temp[i] + 1 *)
(* → output[i] = (input[i] * 2) + 1 *)
```

### Reduction Fusion (OneToOne → Reduction)

Map computation fused into reduction loop:

```ocaml
(* temp[i] = input[i] * 2 *)
(* result = sum(temp[0..n]) *)
(* → result = sum(input[i] * 2 for i in 0..n) *)
```

### Stencil Fusion

When consumer uses stencil pattern on intermediate:

```ocaml
(* temp[i] = input[i] * 2 *)
(* output[i] = temp[i-1] + temp[i] + temp[i+1] *)
(* → output[i] = input[i-1]*2 + input[i]*2 + input[i+1]*2 *)
```

Combined stencil radius = producer_radius + consumer_radius.

## API

### Runtime API (Sarek.Kirc.Fusion)

```ocaml
(* Check if two kernels can be fused *)
val can_fuse_bodies : k_ext -> k_ext -> intermediate:string -> bool

(* Fuse two kernel bodies *)
val fuse_bodies : k_ext -> k_ext -> intermediate:string -> k_ext

(* Fuse complete kernel records *)
val fuse_kernels : kirc_kernel -> kirc_kernel -> intermediate:string -> kirc_kernel

(* Fuse a pipeline, returns (fused, eliminated_intermediates) *)
val fuse_pipeline_bodies : k_ext list -> k_ext * string list
```

### IR-Level API (Sarek_fusion)

```ocaml
(* Analysis *)
val analyze : kernel -> fusion_info
val should_fuse : kernel -> kernel -> string -> fusion_hint

(* Fusion *)
val fuse : kernel -> kernel -> string -> kernel
val fuse_pipeline_list : kernel list -> kernel list * string list
val auto_fuse_pipeline_list : kernel list -> kernel list * string list * string list

(* Legacy wrappers for pipelines that fully collapse to one kernel.
   They raise Invalid_fusion if unfused stages remain. *)
val fuse_pipeline : kernel list -> kernel * string list
val auto_fuse_pipeline : kernel list -> kernel * string list * string list
```

`fuse_pipeline_list` and `auto_fuse_pipeline_list` preserve every kernel in the
input pipeline unless it is replaced by a fused kernel. The returned kernel list
is the optimized pipeline in execution order.

## Auto-Fusion Heuristics

The `should_fuse` function returns a decision and reason:

| Pattern | Decision | Reason |
|---------|----------|--------|
| OneToOne → OneToOne | **Fuse** | Element-wise producer/consumer |
| OneToOne → Reduction | **Fuse** | Map-reduce pattern |
| OneToOne → Stencil (≤3 pts) | MaybeFuse | Small stencil - profile to decide |
| OneToOne → Stencil (>3 pts) | DontFuse | Large stencil radius - keep separate |
| OneToOne → Gather | DontFuse | Gather pattern - keep separate |
| With barriers | DontFuse | Barrier prevents fusion |

`auto_fuse_pipeline_list` only fuses when decision is `Fuse` (conservative).
Skipped pairs remain in the returned pipeline.

## Constraints

Fusion is **not** performed when:

1. **Barriers present**: Either kernel contains `block_barrier` or `warp_barrier`
2. **Multiple uses**: Intermediate is used by multiple consumers
3. **Incompatible patterns**: Access patterns don't match
4. **Atomics**: Kernel uses atomic operations (currently)

## Example

```ocaml
open Sarek.Kirc.Fusion

let producer = [%kernel fun (temp : int32 vector) (input : int32 vector) ->
  temp.(thread_idx_x) <- input.(thread_idx_x) * 2l]

let consumer = [%kernel fun (output : int32 vector) (temp : int32 vector) ->
  output.(thread_idx_x) <- temp.(thread_idx_x) + 1l]

let () =
  let _, kirc_prod = producer in
  let _, kirc_cons = consumer in
  if can_fuse_bodies kirc_prod.body kirc_cons.body ~intermediate:"temp" then
    let fused = fuse_kernels kirc_prod kirc_cons ~intermediate:"temp" in
    (* Use fused kernel - temp array eliminated *)
    ...
```
