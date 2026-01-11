# Generated Code Display System

To add generated code tabs to a benchmark description, add this special marker after the Sarek kernel:

```markdown
## Sarek Kernel

\`\`\`ocaml
[%kernel ...]
\`\`\`

<!-- GENERATED_CODE_TABS: benchmark_name -->
```

The system will then fetch and display tabs showing:
- CUDA C
- OpenCL C  
- Vulkan GLSL
- Metal (if available)
- Native OCaml (if available)

## Implementation Plan

1. Store generated code examples in `descriptions/generated/`
2. Markdown parser detects `<!-- GENERATED_CODE_TABS: name -->` marker
3. Creates tabbed interface with code for each backend
4. Uses same styling as code inspector

## Directory Structure

```
descriptions/
  ├── mandelbrot.md
  ├── matrix_mul.md
  ├── images/
  │   └── mandelbrot_example.png
  └── generated/
      ├── mandelbrot_cuda.cu
      ├── mandelbrot_opencl.cl
      ├── mandelbrot_vulkan.comp
      ├── mandelbrot_metal.metal
      └── mandelbrot_native.ml
```
