# sarek/ppx - Sarek PPX Compiler

The Sarek PPX transforms OCaml code annotated with `[%kernel ...]` into GPU-executable
kernels. It performs parsing, type checking, safety analysis, optimization, and IR
generation for multiple backends (CUDA/OpenCL/Vulkan/Native/Interpreter).

## Compilation Pipeline

```
User Code ([%kernel ...])
         ↓
    Sarek_parse      Parse OCaml syntax to Sarek AST
         ↓
    Sarek_typer      Type inference with GPU constraints
         ↓
    Sarek_mono       Monomorphization (polymorphic → concrete types)
         ↓
    Sarek_convergence Thread-varying analysis (barrier safety)
         ↓
    Sarek_tailrec    Tail recursion optimization
         ↓
    Sarek_lower      Lower to typed AST
         ↓
    Sarek_lower_ir   Generate Sarek IR
         ↓
    Sarek_quote      Quote as OCaml expression
         ↓
    Generated Code   Embedded kernel ready for runtime
```

## Language Features

### Complex Types

Sarek supports rich data types in both GPU and CPU code:

```ocaml
(* Custom record types with [@sarek.type] attribute *)
type point = { x : float32; y : float32 } [@@sarek.type]

let%kernel point_offset points result offset =
  let tid = Std.global_idx_x in
  let p = points.(tid) in
  result.(tid) <- { x = p.x +. offset; y = p.y +. offset }

(* Variant types for tagged unions *)
type result = Ok of float32 | Error [@@sarek.type]

let%kernel safe_divide numerator denominator result =
  let tid = Std.global_idx_x in
  let n = numerator.(tid) in
  let d = denominator.(tid) in
  result.(tid) <- if d = 0.0 then Error else Ok (n /. d)
```

### Shared Memory & Supersteps

BSP-style programming with explicit synchronization:

```ocaml
(* Shared memory declaration *)
let%kernel transpose input output width =
  let%shared (tile : float32) = 256l in  (* Explicit size *)
  
  let tid = Std.thread_idx_x in
  let row = Std.block_idx_y * 16l + tid / 16l in
  let col = Std.block_idx_x * 16l + tid % 16l in
  
  (* Superstep 1: Load from global memory *)
  let%superstep load =
    tile.(tid) <- input.(row * width + col)
  in
  
  (* Superstep 2: Store transposed *)
  let%superstep store =
    let new_row = Std.block_idx_x * 16l + tid / 16l in
    let new_col = Std.block_idx_y * 16l + tid % 16l in
    output.(new_row * width + new_col) <- tile.(tid)
  in
  ()
```

Supersteps automatically insert barriers and verify convergence safety.

### Atomics for Reductions

```ocaml
let%kernel histogram input bins n =
  let open Std in
  let open Gpu in
  
  (* Shared memory for thread block reduction *)
  let%shared (local_hist : int32) = 256l in
  
  let tid = thread_idx_x in
  
  (* Initialize local histogram *)
  let%superstep init =
    if tid < 256l then local_hist.(tid) <- 0l
  in
  
  (* Count into shared memory with atomics *)
  let%superstep count =
    let idx = global_thread_id in
    if idx < n then
      let bin = Int32.rem input.(idx) 256l in
      atomic_add_int32 local_hist bin 1l
  in
  
  (* Write to global memory *)
  let%superstep finalize =
    if tid < 256l then
      atomic_add_int32 bins tid local_hist.(tid)
  in
  ()
```

### Internal Helper Functions

Define reusable functions within kernels:

```ocaml
let%kernel vector_ops input output =
  (* Helper function with explicit type *)
  let clamp (x : float32) (min : float32) (max : float32) : float32 =
    if x < min then min
    else if x > max then max
    else x
  in
  
  (* Helper with inferred type *)
  let square x = x *. x in
  
  let tid = Std.global_idx_x in
  let value = input.(tid) in
  output.(tid) <- clamp (square value) 0.0 100.0
```

### Polymorphism

Generic functions are monomorphized at compile time. Here's a polymorphic matrix transpose
that works with any type:

```ocaml
(* Custom type *)
type point3d = { x : float32; y : float32; z : float32 } [@@sarek.type]

(* Polymorphic helper function - works with any type 'a *)
let[@sarek.module] do_transpose 
    (input : 'a vector) (output : 'a vector)
    (width : int) (height : int) (tid : int) : unit =
  let n = width * height in
  if tid < n then begin
    let col = tid mod width in
    let row = tid / width in
    let in_idx = (row * width) + col in
    let out_idx = (col * height) + row in
    output.(out_idx) <- input.(in_idx)
  end

(* Call at int32 - generates int32 version *)
let%kernel transpose_int32 input output width height =
  let open Std in
  do_transpose input output width height global_thread_id

(* Call at float32 - generates float32 version *)
let%kernel transpose_float32 input output width height =
  let open Std in
  do_transpose input output width height global_thread_id

(* Call at point3d - generates point3d version *)
let%kernel transpose_point3d input output width height =
  let open Std in
  do_transpose input output width height global_thread_id
```

Each kernel gets its own monomorphized version of `do_transpose`.

### Recursion

#### Tail Recursion (Automatic Loop Conversion)

```ocaml
let%kernel factorial output n =
  (* Tail-recursive helper - converted to loop *)
  let rec fact_aux (acc : int32) (n : int32) : int32 =
    if n <= 1l then acc
    else fact_aux (acc * n) (n - 1l)
  in
  
  let idx = Std.global_idx_x in
  if idx = 0l then output.(idx) <- fact_aux 1l n
```

#### Bounded Recursion (Pragma-Based Inlining)

```ocaml
let%kernel fibonacci output n =
  (* Non-tail recursion with bounded depth *)
  let rec fib (n : int32) : int32 [@sarek.inline 10] =
    if n <= 1l then n
    else fib (n - 1l) + fib (n - 2l)
  in
  
  let idx = Std.global_idx_x in
  output.(idx) <- fib n  (* Works for small n *)
```

### Control Flow

Standard OCaml control structures with mutable references:

```ocaml
let%kernel control_flow input output =
  let tid = Std.global_idx_x in
  let value = input.(tid) in
  
  (* If-then-else *)
  let result = if value > 0.0 then value else -. value in
  
  (* For loops with mutable accumulator *)
  let sum = mut 0.0 in
  for i = 0 to 10 do
    sum := !sum +. float_of_int i *. value
  done;
  
  (* While loops with mutable variable *)
  let x = mut value in
  while !x > 1.0 do
    x := !x /. 2.0
  done;
  
  (* Pattern matching *)
  let final = match Int32.rem tid 3l with
    | 0l -> result
    | 1l -> !sum
    | _ -> !x
  in
  output.(tid) <- final
```

Note: Use `let x = mut value` for mutable variables (like OCaml refs: `!x` to read, `x := val` to write).

## Safety Guarantees

### 1. Type Safety

**Compile-time type inference** ensures all GPU code is well-typed:

```ocaml
(* ✓ Type inference works *)
let%kernel inferred input output =
  let x = input.(0l) in  (* Infers: x : float32 *)
  output.(0l) <- x *. 2.0

(* ✗ Type errors caught at compile time *)
let%kernel bad input output =
  let x : float32 = input.(0l) in
  output.(0l) <- x + 1l  (* Error: int32 + float32 mismatch *)
```

**Native OCaml generation** provides double validation:

```ocaml
(* The Native backend generates pure OCaml code *)
(* If your kernel type-checks for GPU, it will also compile as native OCaml *)
(* This catches type errors that might slip through other pipelines *)
```

### 2. Convergence Analysis

Barrier safety analysis detects divergent control flow:

```ocaml
(* ✓ OK: Barrier in converged control flow *)
let%kernel safe_barrier input =
  let%superstep phase1 =
    (* All threads execute this *)
    input.(Std.thread_idx_x) <- 0.0
  in
  ()  (* Implicit barrier - all threads converged *)

(* ✗ COMPILE ERROR: Barrier in divergent control flow *)
let%kernel unsafe_barrier input =
  if Std.thread_idx_x > 16l then
    block_barrier ()  (* Error: Some threads skip barrier → deadlock *)
```

The compiler tracks which expressions depend on thread ID:
- **Uniform**: Same value for all threads in a block (e.g., `block_idx_x`)
- **Thread-varying**: Different per thread (e.g., `thread_idx_x`, array accesses)

Barriers are only allowed where all threads in a block are guaranteed to reach them.

### 3. Recursion Checks

**Bounded or tail-recursive only**:

```ocaml
(* ✓ OK: Tail recursion → loop *)
let rec sum_list acc = function
  | [] -> acc
  | x :: xs -> sum_list (acc + x) xs

(* ✓ OK: Bounded recursion with pragma *)
let rec power base exp [@sarek.inline 32] =
  if exp = 0l then 1l else base * power base (exp - 1l)

(* ✗ COMPILE ERROR: Unbounded non-tail recursion *)
let rec bad_fib n =
  if n <= 1l then n
  else bad_fib (n - 1l) + bad_fib (n - 2l)  (* No [@sarek.inline] pragma *)
```

### 4. Memory Space Safety

The type system tracks memory spaces:

```ocaml
type 'a vector = Global of 'a array  (* Global device memory *)
type 'a shared  = Shared of 'a array (* Shared memory (block-local) *)
type 'a local   = Local of 'a        (* Thread-local / register *)

(* Prevents: *)
(* - Passing shared memory between blocks *)
(* - Using thread-local variables across threads *)
(* - Invalid memory access patterns *)
```

## Module Organization

### Frontend (Parsing & Typing)
- **[Sarek_parse](Sarek_parse.ml)** - Parse `[%kernel ...]` to AST
- **[Sarek_typer](Sarek_typer.ml)** - Type inference with GPU constraints
- **[Sarek_types](Sarek_types.ml)** - Type system definitions
- **[Sarek_env](Sarek_env.ml)** - Type environment management
- **[Sarek_scheme](Sarek_scheme.ml)** - Polymorphic type schemes

### AST Representations
- **[Sarek_ast](Sarek_ast.ml)** - Untyped kernel AST
- **[Sarek_typed_ast](Sarek_typed_ast.ml)** - Typed AST after inference
- **[Kirc_Ast](Kirc_Ast.ml)** - Legacy IR (deprecated)

### Safety Analysis & Optimization
- **[Sarek_convergence](Sarek_convergence.ml)** - Thread-varying analysis, barrier safety
- **[Sarek_mono](Sarek_mono.ml)** - Monomorphization of polymorphic code
- **[Sarek_tailrec](Sarek_tailrec.ml)** - Recursion orchestration
- **[Sarek_tailrec_analysis](Sarek_tailrec_analysis.ml)** - Recursion pattern detection
- **[Sarek_tailrec_elim](Sarek_tailrec_elim.ml)** - Tail call → loop transformation
- **[Sarek_tailrec_pragma](Sarek_tailrec_pragma.ml)** - `[@sarek.inline]` pragma handling
- **[Sarek_tailrec_bounded](Sarek_tailrec_bounded.ml)** - Bounded recursion (experimental)

### IR Generation & Code Generation
- **[Sarek_lower](Sarek_lower.ml)** - Lower typed AST to IR-ready form
- **[Sarek_lower_ir](Sarek_lower_ir.ml)** - Generate Sarek IR
- **[Sarek_quote](Sarek_quote.ml)** - Quote kernels as OCaml expressions
- **[Sarek_quote_ir](Sarek_quote_ir.ml)** - Quote IR constructors
- **[Sarek_native_gen](Sarek_native_gen.ml)** - Native OCaml code generation
- **[Sarek_native_helpers](Sarek_native_helpers.ml)** - Native codegen utilities
- **[Sarek_native_intrinsics](Sarek_native_intrinsics.ml)** - Intrinsic → native mapping

### Infrastructure
- **[Sarek_ppx](Sarek_ppx.ml)** - Main PPX entry point
- **[Sarek_ppx_registry](Sarek_ppx_registry.ml)** - Type/intrinsic registration
- **[Sarek_core_primitives](Sarek_core_primitives.ml)** - Primitive operations
- **[Sarek_ir_ppx](Sarek_ir_ppx.ml)** - IR construction helpers
- **[Sarek_error](Sarek_error.ml)** - Error formatting with source locations
- **[Sarek_reserved](Sarek_reserved.ml)** - Reserved keyword checking (C/CUDA/OpenCL)
- **[Sarek_debug](Sarek_debug.ml)** - Debug utilities

## Testing

Test coverage across the compilation pipeline:

```bash
# Unit tests for PPX modules (59 tests)
dune runtest sarek/ppx/test/

# End-to-end tests with multiple backends
dune runtest sarek/tests/e2e/

# Negative tests (must fail compilation)
dune runtest sarek/tests/negative/

# Run specific test with all backends
_build/default/sarek/tests/e2e/test_histogram.exe --benchmark
```

**Unit test coverage (sarek/ppx/test/):**
- `test_sarek_reserved.ml`: 28 tests for C/CUDA/OpenCL reserved keyword validation
- `test_sarek_error.ml`: 23 tests for structured error types and error reporting
- `test_sarek_debug.ml`: 8 tests for debug logging infrastructure

Test organization:
- **Unit tests**: Test individual PPX modules in isolation ([sarek/ppx/test/](test/))
- **E2E tests**: Full compilation and execution ([sarek/tests/e2e/](../tests/e2e/))
- **Negative tests**: Verify safety checks reject invalid code ([sarek/tests/negative/](../tests/negative/))

## Design Principles

### 1. Compile-Time Safety
Catch GPU-incompatible patterns early:
- No closures (cannot be serialized to GPU)
- No exceptions (GPU execution model doesn't support them)
- No unbounded recursion (stack limitations)
- No divergent barriers (deadlock prevention)

### 2. Backend Agnostic
The PPX generates **[Sarek IR](../../spoc/ir/)** which is independent of any specific GPU backend:
- CUDA/OpenCL/Vulkan: JIT compile IR to native GPU code
- Native CPU: Generate pure OCaml for debugging and portability
- Interpreter: Direct IR evaluation for testing

### 3. Type Inference
Type inference reduces the need for explicit annotations:
```ocaml
let%kernel infer a b c =
  let gid = Std.global_idx_x in
  c.(gid) <- a.(gid) + b.(gid)
  (* Types inferred from vector operations *)
```

### 4. OCaml Semantics
Sarek kernels are valid OCaml code:
- Same syntax, same semantics (with GPU constraints)
- Can be tested and debugged as pure OCaml
- Native backend validates correctness by compilation

## Dependencies

- **[ppxlib](https://opam.ocaml.org/packages/ppxlib/)** - PPX infrastructure
- **[spoc/ir](../../spoc/ir/)** - Sarek IR types (backend-agnostic)
- **[sarek_stdlib](../Sarek_stdlib/)** - GPU standard library (for type checking)

No runtime GPU dependencies in the PPX itself.

## Building GPU Libraries

Sarek can be used to build reusable GPU libraries with custom types and intrinsics. Libraries register their types and functions at compile-time, making them available in any kernel.

### Creating a Library

1. **Define custom intrinsics** using `%sarek_intrinsic`:

```ocaml
(* MyGpuLib.ml *)

(* Helper for backend-specific code *)
let dev cuda opencl d = Sarek.Sarek_registry.cuda_or_opencl d cuda opencl

(* Custom type *)
let%sarek_intrinsic complex =
  {device = (fun _ -> "float2"); ctype = Ctypes.(structure "float2")}

(* Custom function *)
let%sarek_intrinsic (complex_mul : complex -> complex -> complex) =
  {
    device = dev "complex_mul" "complex_mul";  (* Call device function *)
    ocaml = fun (r1, i1) (r2, i2) ->
      (r1 *. r2 -. i1 *. i2, r1 *. i2 +. i1 *. r2)
  }

(* Custom constant *)
let%sarek_intrinsic (max_iterations : int32) =
  {device = dev "1000" "1000"}
```

2. **Add dune configuration**:

```dune
(library
 (name my_gpu_lib)
 (libraries sarek sarek_ppx_intrinsic)
 (preprocess (pps sarek_ppx_intrinsic)))
```

3. **Create wrapper module** with initialization:

```ocaml
(* My_gpu_stdlib.ml *)
module MyGpuLib = MyGpuLib

(* Force registration at module load *)
let () = ignore (MyGpuLib.complex_mul, MyGpuLib.max_iterations)
```

4. **Use in kernels**:

```ocaml
(* user_code.ml *)
let%kernel mandelbrot output width =
  let open MyGpuLib in
  let tid = Std.global_idx_x in
  let c = (* compute complex coordinate *) ... in
  let mutable z = (0.0, 0.0) in
  let mutable iter = 0l in
  
  while !iter < max_iterations do
    z := complex_mul !z !z;  (* Use custom function *)
    iter := !iter + 1l
  done;
  output.(tid) <- !iter
```

### Examples in Repository

- **[Sarek_stdlib](../Sarek_stdlib/)** - Core GPU standard library (Float32, Int32, Gpu intrinsics)
- **[Sarek_float64](../Sarek_float64/)** - Optional double-precision support
- **[Sarek_geometry](../Sarek_geometry/)** - Vector/matrix types for graphics

See **[Sarek_stdlib/README.md](../Sarek_stdlib/README.md)** for detailed intrinsic definition guide.

### Library Architecture

```
Your Library Module
       ↓ (at compile-time)
  %sarek_intrinsic PPX
       ↓
  Sarek_ppx_registry    ← Kernel PPX queries for type-checking
       ↓
  Generated Code
       ↓ (at runtime)
  Sarek_registry        ← Backends query for device code
```

Benefits:
- **Type safety**: Libraries provide typed interfaces checked at compile-time
- **Backend agnostic**: Same library works with CUDA/OpenCL/Vulkan/Native
- **Composable**: Multiple libraries can be combined in a single kernel
- **Versioned**: Libraries can evolve independently from kernels

## Related Documentation

- **[spoc/README.md](../../spoc/README.md)** - SDK layer (Framework_sig, IR types)
- **[spoc/ir/README.md](../../spoc/ir/README.md)** - Sarek IR specification
- **[sarek/core/README.md](../core/README.md)** - Runtime (Device, Memory, Vector, Kernel)
- **[sarek/Sarek_stdlib/README.md](../Sarek_stdlib/README.md)** - Building GPU libraries with intrinsics
- **[sarek/plugins/](../plugins/)** - Backend implementations (CUDA, OpenCL, Vulkan, Native)
- **[ARCHITECTURE.md](../../ARCHITECTURE.md)** - Repository structure overview
- **[AGENTS.md](../../AGENTS.md)** - Guidelines for development

## Usage Example

Complete example with error handling:

```ocaml
(* my_kernel.ml *)
open Sarek
module Std = Sarek_stdlib.Std

(* Define kernel with type inference *)
let%kernel saxpy a x y alpha =
  let tid = Std.global_idx_x in
  y.(tid) <- alpha *. x.(tid) +. y.(tid)

(* Execute on GPU *)
let () =
  (* Initialize backends *)
  let devices = Spoc_core.Device.init 
    ~frameworks:["CUDA"; "OpenCL"; "Vulkan"; "Native"] () in
  
  let dev = devices.(0) in
  Printf.printf "Using: %s\n" dev.Spoc_core.Device.name;
  
  (* Create vectors *)
  let n = 1_000_000 in
  let x = Spoc_core.Vector.create_float32 n in
  let y = Spoc_core.Vector.create_float32 n in
  
  (* Execute kernel *)
  let grid = (n / 256, 1, 1) in
  let block = (256, 1, 1) in
  saxpy dev grid block 2.0 x y;
  
  (* Results automatically available on host *)
  Printf.printf "y[0] = %f\n" (Spoc_core.Vector.get y 0)
```

For more examples, see the [test suite](../tests/e2e/).
