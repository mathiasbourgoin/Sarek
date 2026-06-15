# Backend Code Generation System

This document describes the backend code generation system for Sarek benchmarks.

## Overview

The backend code generation system automatically extracts kernel IR from benchmark code and generates equivalent code for all supported backends:

- **CUDA C** - NVIDIA GPUs
- **OpenCL C** - Cross-platform GPUs
- **Vulkan GLSL** - Modern graphics/compute API
- **Metal** - Apple GPUs

Generated code is displayed in the web benchmark viewer as tabbed interfaces, allowing users to see how Sarek kernels compile to different backends.

## Architecture

```
Sarek Kernel (OCaml)
    ↓ [%kernel] PPX
Kernel IR (Internal)
    ↓ generate_backend_code.ml
Generated Code (4 backends)
    ↓ Markdown files
Web Viewer (Tabs UI)
```

## Files

### Generator Tool
- **`benchmarks/generate_backend_code.ml`** - Main generator (300+ lines)
  - Extracts kernel IR from `[%kernel ...]` expressions
  - Calls backend-specific `generate()` functions
  - Outputs markdown with code sections

### Generated Output
- **`benchmarks/descriptions/generated/*.md`** - Generated markdown files
  - One file per benchmark kernel
  - Format: `{benchmark_name}_generated.md`
  - Contains code sections for all 4 backends

### Web Integration
- **`gh-pages/javascripts/benchmark-viewer.js`** - JavaScript viewer
  - `processGeneratedCodeTabs()` - Processes markers in markdown
  - `createGeneratedCodeTabs()` - Creates tabbed HTML interface
  - `setupTabClickHandlers()` - Handles tab switching

### Benchmark Descriptions
- **`benchmarks/descriptions/*.md`** - Benchmark documentation
  - Contains `<!-- GENERATED_CODE_TABS: name -->` markers
  - Markers replaced with tabs at runtime

## Usage

### Regenerate All Backend Code

```bash
# Using Make
make bench-generate-code

# Using dune directly
dune exec benchmarks/generate_backend_code.exe

# With custom output directory
dune exec benchmarks/generate_backend_code.exe -- --output my_output_dir
```

### Run with Benchmarks

```bash
# Run benchmarks and regenerate backend code
./benchmarks/run_all_benchmarks.sh --generate-backend-code
```

### Add to New Benchmark

1. Create kernel in benchmark file:
   ```ocaml
   let my_kernel =
     [%kernel
       fun (input : float32 vector) (output : float32 vector) (n : int32) ->
         (* kernel code *)
     ]
   ```

2. Add kernel to `generate_backend_code.ml`:
   ```ocaml
   let my_kernel =
     [%kernel
       fun (input : float32 vector) (output : float32 vector) (n : int32) ->
         (* same kernel code *)
     ]
   
   (* In main function *)
   generate_backend_code "my_benchmark" my_kernel !output_dir;
   ```

3. Add marker to description file `descriptions/my_benchmark.md`:
   ```markdown
   ## Sarek Kernel
   
   ```ocaml
   [%kernel ...]
   ```
   
   <!-- GENERATED_CODE_TABS: my_benchmark -->
   
   ## Key Features
   ```

4. Regenerate:
   ```bash
   make bench-generate-code
   ```

## CI Integration

The CI system automatically checks that generated code is up to date:

```yaml
# .github/workflows/ci.yml
check-generated-code:
  - Regenerates all backend code
  - Checks for git diff
  - Fails if changes detected
```

This ensures:
- ✅ Generated code stays in sync with kernels
- ✅ Backend generator changes are detected
- ✅ No regressions in code generation

## Generated Code Format

Each generated file contains:

```markdown
# Generated Backend Code: {name}

This file is auto-generated. Do not edit manually.

Generated on: 2026-01-11 00:21:52

## CUDA C

```cuda
__global__ void sarek_kern(...) {
  // CUDA code
}
```

## OpenCL C

```opencl
__kernel void sarek_kern(...) {
  // OpenCL code
}
```

## Vulkan GLSL

```glsl
#version 450
layout(...) in;
void main() {
  // GLSL code
}
```

## Metal

```metal
kernel void sarek_kern(...) {
  // Metal code
}
```
```

## Features Translated

The generator correctly translates:

| Sarek Feature | CUDA | OpenCL | Vulkan | Metal |
|--------------|------|--------|--------|-------|
| `global_thread_id` | `threadIdx.x + blockIdx.x * blockDim.x` | `get_global_id(0)` | `gl_GlobalInvocationID.x` | `thread_position_in_grid.x` |
| `thread_idx_x` | `threadIdx.x` | `get_local_id(0)` | `gl_LocalInvocationID.x` | `thread_position_in_threadgroup.x` |
| `block_idx_x` | `blockIdx.x` | `get_group_id(0)` | `gl_WorkGroupID.x` | `threadgroup_position_in_grid.x` |
| `let%shared` | `__shared__` | `__local__` | `shared` | `threadgroup` |
| `let%superstep` | `__syncthreads()` | `barrier()` | `barrier()` | `threadgroup_barrier()` |
| `while` loops | `while` | `while` | `while` | `while` |
| `for` loops | `for` | `for` | `for` | `for` |
| `mut` variables | Regular vars | Regular vars | Regular vars | Regular vars |

## Benefits

1. **Documentation** - Shows exactly how Sarek compiles to each backend
2. **Learning** - Users can compare idioms across backends
3. **Verification** - Ensures code generation works correctly
4. **Regression Detection** - CI catches unintended changes
5. **Transparency** - No "magic" - users see the generated code

## Maintenance

### When to Regenerate

Regenerate when:
- ✅ A kernel is modified
- ✅ Backend generators are updated
- ✅ New kernels are added
- ✅ PPX preprocessor changes affect IR

### Commit Policy

Always commit generated files:
```bash
git add benchmarks/descriptions/generated/
git add gh-pages/benchmarks/descriptions/generated/
git commit -m "Update generated backend code"
```

CI will fail if generated code is out of sync.
