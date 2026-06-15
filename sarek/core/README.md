# sarek/core - Core Runtime Abstractions

The `sarek/core` package provides the foundational runtime abstractions for Sarek GPU computing. It sits between the Sarek PPX (IR generation) and backend plugins (CUDA/OpenCL/Vulkan/Native/Interpreter), providing type-safe device management, vector storage, memory transfers, and kernel execution.

**Key responsibilities:**
- Device enumeration and abstraction across backends
- Type-safe vector storage with location tracking (CPU/GPU)
- Memory buffer management with zero-copy for CPU backends
- Explicit and automatic data transfer control
- Unified logging and error handling

## Table of Contents

- [Quick Start](#quick-start)
- [Core Abstractions](#core-abstractions)
- [Module Overview](#module-overview)
- [Usage Examples](#usage-examples)
- [Design Principles](#design-principles)
- [Testing](#testing)
- [Debugging](#debugging)

## Quick Start

```ocaml
open Sarek_core

(* Initialize devices *)
let () = Device.init () |> ignore

(* Get first GPU device *)
let dev = Device.get_default () in

(* Create vectors *)
let a = Vector.create_float32 1024 in
let b = Vector.create_float32 1024 in
let c = Vector.create_float32 1024 in

(* Initialize data on CPU *)
Vector.init_float32 a (fun i -> float_of_int i);
Vector.init_float32 b (fun i -> float_of_int (i * 2));

(* Explicit transfer to device *)
Transfer.to_device a dev;
Transfer.to_device b dev;
Transfer.to_device c dev;

(* Kernel execution happens here (via sarek/sarek Execute module) *)
(* ... *)

(* Explicit transfer back to CPU *)
Transfer.to_cpu c;

(* Access results *)
Printf.printf "c[10] = %f\n" (Vector.get_float32 c 10)
```

## Core Abstractions

### Device Abstraction

Devices represent compute units (GPUs, CPUs) from any backend:

```ocaml
type Device.t = {
  id : int;                              (* Global device ID *)
  backend_id : int;                      (* Backend-specific ID *)
  name : string;                         (* Human-readable name *)
  framework : string;                    (* "CUDA", "OpenCL", etc. *)
  capabilities : Framework_sig.capabilities;
}
```

**Capabilities** let you query device features:
- `is_cpu` / `is_gpu` - Device type
- `supports_fp64` - Double precision support
- `supports_atomics` - Atomic operations
- `warp_size` - Warp/wavefront size
- `max_work_group_size` - Max threads per block

### Vector Abstraction

Vectors track data location and handle synchronization:

```ocaml
type ('a, 'b) Vector.t = {
  length : int;
  kind : ('a, 'b) kind;              (* Scalar or Custom type *)
  location : location;                (* Where data lives *)
  host : host_storage;                (* CPU-side storage *)
  buffers : device_buffer list;       (* GPU buffers (lazy) *)
}

type location =
  | CPU                                 (* Data only on CPU *)
  | GPU of Device.t                    (* Data only on device *)
  | Both of Device.t                   (* Synchronized on both *)
  | Stale_CPU of Device.t              (* Device has newer data *)
  | Stale_GPU of Device.t              (* CPU has newer data *)
```

**Two storage modes:**
- **Bigarray**: Built-in types (int32, float32, float64, etc.)
- **Custom**: User-defined ctypes structures

### Memory Buffers

Device buffers encapsulate backend-specific memory:

```ocaml
module type Vector.DEVICE_BUFFER = sig
  val device : Device.t
  val size : int                       (* Number of elements *)
  val elem_size : int                  (* Bytes per element *)
  val device_ptr : nativeint          (* Pointer for kernel args *)
  val bind_to_kargs : kargs -> int -> unit
  val host_ptr_to_device : nativeint -> byte_size:int -> unit
  val device_to_host_ptr : nativeint -> byte_size:int -> unit
  val free : unit -> unit
end
```

Zero-copy used for CPU backends when possible.

## Module Overview

### Device Management

#### `Device.ml` - Device Enumeration (236 lines)

Discovers and manages compute devices across all backends.

**Key functions:**
```ocaml
val init : ?frameworks:string list -> unit -> Device.t array
  (* Initialize backends and enumerate devices *)

val all : unit -> Device.t array
  (* Get all initialized devices *)

val get_default : unit -> Device.t
  (* Get default device (first GPU, or first CPU if no GPU) *)

val get : int -> Device.t
  (* Get device by global ID *)

val is_cpu : Device.t -> bool
val is_gpu : Device.t -> bool
  (* Query device type *)

val allows_fp64 : Device.t -> bool
  (* Check double precision support *)

val filter : (Device.t -> bool) -> Device.t array
  (* Filter devices by predicate *)
```

**Device initialization order:**
1. Try each framework in order: CUDA → OpenCL → Vulkan → Metal → Native → Interpreter
2. Skip unavailable backends
3. Assign sequential global IDs
4. Return array of all discovered devices

### Vector Storage & Access

#### `Vector.ml` - High-Level Vector Interface (512 lines)

Unified vector abstraction with location tracking.

**Creation:**
```ocaml
val create_scalar : 
  ('a, 'b) scalar_kind -> int -> ('a, 'b) t

val create_float32 : int -> (float, float32_elt) t
val create_float64 : int -> (float, float64_elt) t
val create_int32 : int -> (int32, int32_elt) t
(* Convenience constructors for common types *)

val create_custom : 
  (module CUSTOM_TYPE with type t = 'a) -> 
  int -> ('a, unit) t
  (* Create vector of custom ctypes structures *)

val of_bigarray : 
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 
  ('a, 'b) scalar_kind -> 
  ('a, 'b) t
  (* Wrap existing bigarray as vector *)
```

**Element access:**
```ocaml
val get : ('a, 'b) t -> int -> 'a
  (* Get element (auto-syncs if needed) *)

val set : ('a, 'b) t -> int -> 'a -> unit
  (* Set element (marks CPU as authoritative) *)

val get_float32 : (float, float32_elt) t -> int -> float
val set_float32 : (float, float32_elt) t -> int -> float -> unit
  (* Type-specific convenience functions *)
```

**Initialization:**
```ocaml
val init_float32 : (float, float32_elt) t -> (int -> float) -> unit
val fill_float32 : (float, float32_elt) t -> float -> unit
  (* Initialize vector with function or constant *)
```

**Subvectors:**
```ocaml
val sub_vector : ('a, 'b) t -> start:int -> len:int -> unit -> ('a, 'b) t
  (* Create view into parent vector (shares storage) *)

val partition : ('a, 'b) t -> int -> ('a, 'b) t array
  (* Split into N equal chunks *)
```

**Metadata:**
```ocaml
val length : ('a, 'b) t -> int
val location : ('a, 'b) t -> location
val kind : ('a, 'b) t -> ('a, 'b) kind
val device : ('a, 'b) t -> Device.t option
```

#### `Vector_types.ml` - Type Definitions (224 lines)

Core type definitions for vectors:
- `scalar_kind` - Bigarray kinds (Int32, Float32, Float64, etc.)
- `custom_type` - User-defined structure metadata
- `kind` - Scalar or Custom GADT
- `location` - Where data resides
- `host_storage` - Bigarray or Custom storage

#### Storage Management — `Spoc_core_base.Make(Ctypes_ops)`

Vector creation, allocation, copying, and sub-vector operations live in the pure
`spoc_core_base` functor (`sarek/core_base/Spoc_core_base.ml`), instantiated
natively as `Vector_types = Spoc_core_base.Make(Ctypes_ops)` and re-exported
through `Vector`. (The numeric path is FFI-free; the browser build instantiates
the same functor with `Js_ops`.)

#### `Vector_transfer.ml` - Sync Callbacks (48 lines)

Synchronization hooks for element access:
- Host pointer extraction
- Auto-sync on get/set
- Custom sync callbacks

### Memory Management

#### `Memory.ml` - Buffer Abstraction (190 lines)

Type-safe buffer operations.

**Buffer module type:**
```ocaml
module type BUFFER = sig
  val device : Device.t
  val size : int
  val elem_size : int
  val device_ptr : nativeint
  val bind_to_kargs : Framework_sig.kargs -> int -> unit
  val host_ptr_to_device : nativeint -> byte_size:int -> unit
  val device_to_host_ptr : nativeint -> byte_size:int -> unit
  val free : unit -> unit
end
```

**Functions:**
```ocaml
val alloc_scalar : 
  Device.t -> int -> ('a, 'b) Vector.scalar_kind -> 
  (module BUFFER)

val alloc_custom :
  Device.t -> int -> elem_size:int ->
  (module BUFFER)

val to_device : (module BUFFER) -> ('a, 'b) Vector.t -> unit
val from_device : (module BUFFER) -> ('a, 'b) Vector.t -> unit
```

Zero-copy used when:
- Device is CPU-type (OpenCL CPU, Native)
- Backend supports `alloc_zero_copy`
- Vector uses Bigarray storage

### Data Transfer

#### `Transfer.ml` - Transfer Control (451 lines)

Explicit and automatic data movement between CPU and device.

**Auto-transfer mode:**
```ocaml
val enable_auto : unit -> unit
val disable_auto : unit -> unit
val is_auto : unit -> bool
  (* Control automatic synchronization *)
```

**Explicit transfers:**
```ocaml
val to_device : ('a, 'b) Vector.t -> Device.t -> unit
  (* CPU → Device *)

val to_cpu : ('a, 'b) Vector.t -> unit
  (* Device → CPU *)

val ensure_on_device : ('a, 'b) Vector.t -> Device.t -> unit
  (* Transfer only if not already there *)

val ensure_on_cpu : ('a, 'b) Vector.t -> unit
  (* Transfer only if stale on CPU *)
```

**Buffer management:**
```ocaml
val alloc_scalar_buffer : 
  Device.t -> int -> ('a, 'b) Vector.scalar_kind -> 
  Vector.device_buffer

val alloc_scalar_buffer_zero_copy :
  Device.t -> int -> ('a, 'b) Vector.scalar_kind ->
  Vector.device_buffer option
  (* Returns None if zero-copy not supported *)

val get_or_alloc_buffer :
  ('a, 'b) Vector.t -> Device.t -> Vector.device_buffer
  (* Get existing buffer or allocate new one *)
```

**Transfer flow:**
1. Check if transfer needed (location tracking)
2. Allocate device buffer if needed
3. Get host pointer from vector storage
4. Call backend's transfer operation
5. Update vector location

**Zero-copy path:**
- Backend allocates pinned/shared memory
- Host pointer = device pointer
- Transfers become no-ops
- Used automatically for CPU backends

### Kernel Execution

#### `Kernel.ml` - Kernel Compilation & Launch (170 lines)

Type-safe kernel operations.

**Kernel module type:**
```ocaml
module type KERNEL = sig
  type t
  val device : t -> Device.t
  val launch : 
    t -> 
    args:(module ARGS) -> 
    grid:Framework_sig.dims -> 
    block:Framework_sig.dims -> 
    shared_mem:int -> 
    unit -> unit
  val profile_launch : ... -> float
  val free : t -> unit
end
```

**Functions:**
```ocaml
val compile_source : 
  Device.t -> 
  source:string -> 
  lang:Framework_sig.source_lang -> 
  kernel_name:string -> 
  (module KERNEL)

val launch :
  (module KERNEL) ->
  args:(module ARGS) ->
  grid:Framework_sig.dims ->
  block:Framework_sig.dims ->
  ?shared_mem:int ->
  unit -> unit
```

#### `Kernel_arg.ml` - Type-Safe Arguments (119 lines)

GADT-based kernel argument handling.

**Argument types:**
```ocaml
type _ t =
  | Vec : ('a, 'b) Vector.t -> ('a, 'b) Vector.t t
  | Scalar_Int : int -> int t
  | Scalar_Int32 : int32 -> int32 t
  | Scalar_Int64 : int64 -> int64 t
  | Scalar_Float32 : float -> float t
  | Scalar_Float64 : float -> float t

type any = Any : 'a t -> any
```

No `Obj.t` - types preserved through compilation.

### Runtime Helpers

#### `Runtime.ml` - High-Level Convenience (127 lines)

Convenience wrappers for common operations.

**Dimension helpers:**
```ocaml
val dims1d : int -> Framework_sig.dims
val dims2d : int -> int -> Framework_sig.dims
val dims3d : int -> int -> int -> Framework_sig.dims

val grid_for_size : problem_size:int -> block_size:int -> int
  (* Calculate grid size: (problem_size + block_size - 1) / block_size *)
```

**Memory allocation shortcuts:**
```ocaml
val alloc_float32 : Device.t -> int -> (module Memory.BUFFER)
val alloc_float64 : Device.t -> int -> (module Memory.BUFFER)
val alloc_int32 : Device.t -> int -> (module Memory.BUFFER)
```

**Transfer shortcuts:**
```ocaml
val to_device : (module Memory.BUFFER) -> 
  (float, float32_elt) Vector.t -> unit
val from_device : (module Memory.BUFFER) -> 
  (float, float32_elt) Vector.t -> unit
```

### Debugging & Monitoring

#### `Log.ml` - Unified Logging (137 lines)

Component-based debug logging controlled by `SAREK_DEBUG` environment variable.

**Log components:**
```ocaml
type component =
  | Device    (* Device enumeration *)
  | Memory    (* Buffer allocation/free *)
  | Transfer  (* Data movement *)
  | Kernel    (* Kernel compilation/launch *)
  | Execute   (* Execution dispatch *)
```

**Usage:**
```bash
# Enable specific components
SAREK_DEBUG=transfer,kernel ./my_program

# Enable all debug output
SAREK_DEBUG=all ./my_program
```

**Functions:**
```ocaml
val debugf : component -> ('a, unit, string, unit) format4 -> 'a
val infof : component -> ('a, unit, string, unit) format4 -> 'a
val warnf : component -> ('a, unit, string, unit) format4 -> 'a
val errorf : component -> ('a, unit, string, unit) format4 -> 'a
```

Example output:
```
[Device] Framework CUDA: 2 device(s)
[Device]   [0] NVIDIA GeForce RTX 3090 (CUDA)
[Transfer] host_ptr_to_device: ptr=139824567890432 size=4096
[Kernel] Compiling kernel 'vector_add' for device 0
```

#### `Error.ml` - Error Handling (222 lines)

Structured error types for runtime failures.

#### `Profiling.ml` - Performance Monitoring (230 lines)

Kernel profiling and timing utilities.

#### `Advanced.ml` - Advanced Features (242 lines)

Advanced memory operations and optimizations.

## Usage Examples

### Basic Vector Operations

```ocaml
open Sarek_core

let () =
  (* Create and initialize vector *)
  let v = Vector.create_float32 1024 in
  Vector.init_float32 v (fun i -> float_of_int i);
  
  (* Element access *)
  let x = Vector.get_float32 v 10 in
  Printf.printf "v[10] = %f\n" x;
  
  (* Modify element *)
  Vector.set_float32 v 10 99.0;
  
  (* Copy vector *)
  let v2 = Vector.copy v in
  
  (* Create subvector (shares storage) *)
  let sub = Vector.sub_vector v ~start:100 ~len:50 () in
  Printf.printf "Subvector length: %d\n" (Vector.length sub)
```

### Device Selection

```ocaml
(* Get all GPUs *)
let gpus = Device.filter Device.is_gpu (Device.all ()) in
Printf.printf "Found %d GPU(s)\n" (Array.length gpus);

(* Get first device with FP64 support *)
let fp64_dev = 
  Array.find Device.allows_fp64 (Device.all ()) in
Printf.printf "FP64 device: %s\n" fp64_dev.name;

(* Get specific backend *)
let cuda_devices = 
  Device.filter (fun d -> d.framework = "CUDA") (Device.all ()) in

(* Check capabilities *)
let dev = Device.get_default () in
let caps = dev.capabilities in
if caps.supports_atomics then
  Printf.printf "Device supports atomic operations\n";
Printf.printf "Warp size: %d\n" caps.warp_size
```

### Explicit Transfer Control

```ocaml
(* Disable auto-transfer for manual control *)
Transfer.disable_auto ();

let dev = Device.get_default () in
let a = Vector.create_float32 1024 in
let b = Vector.create_float32 1024 in
let c = Vector.create_float32 1024 in

(* Initialize on CPU *)
Vector.init_float32 a (fun i -> float_of_int i);
Vector.init_float32 b (fun i -> float_of_int (i * 2));

(* Explicit transfer to device *)
Transfer.to_device a dev;
Transfer.to_device b dev;
Transfer.to_device c dev;

(* Run kernel (via sarek/sarek Execute module) *)
(* ... kernel execution ... *)

(* Explicit transfer back *)
Transfer.to_cpu c;

(* Access results (no auto-sync needed) *)
let result = Vector.get_float32 c 10 in
Printf.printf "Result: %f\n" result
```

### Zero-Copy Optimization

```ocaml
(* Zero-copy for CPU backends *)
let cpu_dev = 
  Array.find Device.is_cpu (Device.all ()) in

let v = Vector.create_float32 1024 in
Vector.init_float32 v (fun i -> float_of_int i);

(* Transfer uses zero-copy (no actual copy) *)
Transfer.to_device v cpu_dev;

(* Modifications visible on both sides *)
Vector.set_float32 v 0 999.0;
(* Device sees updated value immediately *)
```

### Custom Types

```ocaml
open Ctypes

(* Define custom struct *)
type particle = {
  x : float;
  y : float;
  vx : float;
  vy : float;
  mass : float;
}

(* Create ctypes structure *)
let particle_struct : particle structure typ = structure "particle"
let x_field = field particle_struct "x" float
let y_field = field particle_struct "y" float
let vx_field = field particle_struct "vx" float
let vy_field = field particle_struct "vy" float
let mass_field = field particle_struct "mass" float
let () = seal particle_struct

(* Custom type module *)
module Particle_custom = struct
  type t = particle
  let typ = particle_struct
  let size = sizeof particle_struct
  
  let get ptr =
    let s = !@ (from_voidp particle_struct (to_voidp ptr)) in
    {
      x = getf s x_field;
      y = getf s y_field;
      vx = getf s vx_field;
      vy = getf s vy_field;
      mass = getf s mass_field;
    }
  
  let set ptr p =
    let s_ptr = from_voidp particle_struct (to_voidp ptr) in
    setf (!@ s_ptr) x_field p.x;
    setf (!@ s_ptr) y_field p.y;
    setf (!@ s_ptr) vx_field p.vx;
    setf (!@ s_ptr) vy_field p.vy;
    setf (!@ s_ptr) mass_field p.mass
end

(* Create vector of custom type *)
let particles = Vector.create_custom (module Particle_custom) 10000 in

(* Initialize *)
for i = 0 to 9999 do
  Vector.set particles i {
    x = Random.float 100.0;
    y = Random.float 100.0;
    vx = Random.float 10.0;
    vy = Random.float 10.0;
    mass = 1.0;
  }
done
```

### Memory Buffer Management

```ocaml
(* Allocate buffer *)
let dev = Device.get_default () in
let buf = Memory.alloc_scalar dev 1024 Vector.Float32 in

(* Transfer data *)
let v = Vector.create_float32 1024 in
Vector.init_float32 v (fun i -> float_of_int i);
Memory.to_device buf v;

(* Use in kernel *)
let module B = (val buf : Memory.BUFFER) in
Printf.printf "Device pointer: %Ld\n" (Int64.of_nativeint B.device_ptr);
B.bind_to_kargs kargs 0;

(* Transfer back *)
Memory.from_device buf v;

(* Free *)
let module B = (val buf : Memory.BUFFER) in
B.free ()
```

### Subvectors and Partitioning

```ocaml
(* Create large vector *)
let v = Vector.create_float32 10000 in
Vector.init_float32 v (fun i -> float_of_int i);

(* Create subvector (view, shares storage) *)
let sub = Vector.sub_vector v ~start:1000 ~len:100 () in
Printf.printf "Subvector length: %d\n" (Vector.length sub);

(* Modifications to sub affect parent *)
Vector.set_float32 sub 0 999.0;
assert (Vector.get_float32 v 1000 = 999.0);

(* Partition into chunks for multi-device *)
let chunks = Vector.partition v 4 in
Array.iteri (fun i chunk ->
  Printf.printf "Chunk %d: length=%d\n" i (Vector.length chunk)
) chunks;

(* Process chunks on different devices *)
let devices = Device.all () in
Array.iteri (fun i chunk ->
  if i < Array.length devices then begin
    let dev = devices.(i) in
    Transfer.to_device chunk dev;
    (* Run kernel on chunk *)
  end
) chunks
```

## Design Principles

### 1. Type Safety Without Obj.t

The core uses GADTs, first-class modules, and existential types instead of `Obj.t`:

```ocaml
(* Good: GADT preserves element type *)
type _ kind =
  | Scalar : ('a, 'b) scalar_kind -> ('a, 'b) kind
  | Custom : 'a custom_type -> ('a, unit) kind

type (_, _) t = {
  kind : ('a, 'b) kind;
  ...
}

(* Good: First-class module for buffers *)
module type DEVICE_BUFFER = sig ... end
type device_buffer = (module DEVICE_BUFFER)

(* Bad: Loses type information *)
type buffer = Obj.t
```

### 2. Backend-Agnostic Core

Core modules never import backend-specific code:

```ocaml
(* Good: Uses plugin registry *)
match Framework_registry.find_backend "CUDA" with
| Some (module B : Framework_sig.BACKEND) ->
    B.Device.init ()
| None -> (* fallback *)

(* Bad: Direct backend import *)
open Cuda_backend  (* Never do this in core *)
```

All backend interaction goes through:
- `Framework_registry` - Plugin discovery
- `Framework_sig.BACKEND` - Unified interface

### 3. Location Tracking

Vectors track where data lives to optimize transfers:

```ocaml
match vec.location with
| CPU -> (* Need to transfer to device *)
    Transfer.to_device vec dev
| GPU d when d = dev -> (* Already there *)
    ()
| GPU d -> (* On wrong device *)
    Transfer.to_cpu vec;
    Transfer.to_device vec dev
| Both d when d = dev -> (* Synchronized *)
    ()
| Stale_CPU d when d = dev -> (* Device is newer *)
    if auto_sync then Transfer.to_cpu vec
| Stale_GPU d -> (* CPU is newer *)
    Transfer.to_device vec d
```

### 4. Lazy Allocation

Device buffers allocated on first use:

```ocaml
let to_device (vec : ('a, 'b) t) (dev : Device.t) =
  (* Check if buffer exists *)
  let buf = match find_buffer vec dev with
    | Some b -> b  (* Reuse existing *)
    | None -> 
        (* Allocate on first use *)
        let b = alloc_buffer dev vec.length vec.kind in
        vec.buffers <- b :: vec.buffers;
        b
  in
  (* Transfer data *)
  transfer_to_buffer vec buf
```

### 5. Zero-Copy When Possible

CPU backends use shared memory:

```ocaml
let alloc_scalar_buffer dev len kind =
  match Framework_registry.find_backend dev.framework with
  | Some (module B) ->
      (* Try zero-copy first *)
      match B.Memory.alloc_zero_copy dev len kind with
      | Some buf -> buf  (* Host and device share memory *)
      | None -> B.Memory.alloc dev len kind  (* Fallback *)
```

Transfers check zero-copy flag:

```ocaml
let host_ptr_to_device src_ptr ~byte_size =
  if B.Memory.is_zero_copy buf then () (* Skip transfer *)
  else B.Memory.host_ptr_to_device ~src_ptr ~byte_size ~dst:buf
```

## Testing

The core has comprehensive unit tests covering all modules:

```bash
# Run all core tests
dune runtest sarek/core/test

# Run specific test module
dune exec sarek/core/test/test_device.exe
dune exec sarek/core/test/test_vector.exe
dune exec sarek/core/test/test_transfer.exe
```

**Test coverage:**
- `test_device.ml` - Device enumeration, predicates, capabilities
- `test_vector.ml` - Creation, access, subvectors, partitioning
- `test_vector_storage.ml` - Storage allocation, copying
- `test_vector_transfer.ml` - Sync callbacks, host pointers
- `test_memory.ml` - Buffer allocation, transfers
- `test_kernel.ml` - Kernel compilation, argument binding
- `test_kernel_arg.ml` - GADT argument handling

Run with coverage:

```bash
BISECT_FILE=_coverage/core dune runtest sarek/core/test --force
bisect-ppx-report html -o _coverage/core-report
```

## Debugging

### Environment Variables

**`SAREK_DEBUG`** - Enable debug logging:

```bash
# Single component
SAREK_DEBUG=transfer ./my_program

# Multiple components
SAREK_DEBUG=transfer,kernel,device ./my_program

# All components
SAREK_DEBUG=all ./my_program
```

Components: `device`, `memory`, `transfer`, `kernel`, `execute`

### Common Debugging Patterns

**Check device capabilities:**
```ocaml
let dev = Device.get_default () in
let caps = dev.capabilities in
Printf.printf "Device: %s\n" dev.name;
Printf.printf "  Framework: %s\n" dev.framework;
Printf.printf "  Is CPU: %b\n" caps.is_cpu;
Printf.printf "  FP64: %b\n" caps.supports_fp64;
Printf.printf "  Atomics: %b\n" caps.supports_atomics;
Printf.printf "  Warp size: %d\n" caps.warp_size;
Printf.printf "  Max work group: %d\n" caps.max_work_group_size
```

**Track vector location:**
```ocaml
let print_location vec =
  match Vector.location vec with
  | CPU -> Printf.printf "CPU\n"
  | GPU d -> Printf.printf "GPU (device %d)\n" d.id
  | Both d -> Printf.printf "Both (device %d)\n" d.id
  | Stale_CPU d -> Printf.printf "Stale CPU (device %d has newer)\n" d.id
  | Stale_GPU d -> Printf.printf "Stale GPU (CPU has newer, device %d)\n" d.id
```

**Monitor transfers:**
```bash
SAREK_DEBUG=transfer ./my_program
```

Output:
```
[Transfer] host_ptr_to_device: ptr=139824567890432 size=4096
[Transfer] to_device: vector 0 → device 0 (CUDA)
[Transfer] Zero-copy enabled for device 1 (Native)
```

**Check buffer allocation:**
```bash
SAREK_DEBUG=memory ./my_program
```

## See Also

- [../sarek/README.md](../sarek/README.md) - Sarek runtime execution layer
- [../ppx/README.md](../ppx/README.md) - Sarek PPX compiler
- [../../spoc/framework/README.md](../../spoc/framework/README.md) - Plugin interface
- [../../spoc/ir/README.md](../../spoc/ir/README.md) - IR types and transformations
- [../../COVERAGE.md](../../COVERAGE.md) - Coverage infrastructure
- [../../README.md](../../README.md) - Main project documentation
