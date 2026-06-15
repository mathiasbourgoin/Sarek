# Interpreter Plugin - CPU Interpretation Backend

**Package**: `sarek.interpreter`  
**Execution Model**: Custom (IR interpretation)

CPU backend that interprets Sarek IR at runtime for debugging and testing. Provides sequential and parallel execution modes.

## Overview

The Interpreter backend executes kernels by interpreting the Sarek IR (Intermediate Representation) at runtime. Unlike the Native backend which calls pre-compiled functions, the Interpreter walks through the IR tree and evaluates each expression and statement.

### Execution Flow

1. **Kernel Definition**: `[%kernel fun ...]` in OCaml code
2. **PPX Processing**: Sarek PPX generates IR
3. **Interpretation**: `Sarek_ir_interp.run_kernel` interprets IR at runtime
4. **Execution**: Step-by-step evaluation of IR nodes

### Key Features

- Step-by-step IR interpretation for debugging
- Sequential and parallel execution modes
- No compilation overhead
- Full introspection capabilities
- Useful for testing and validation

## Architecture

```
sarek/plugins/interpreter/
├── Interpreter_error.ml         # Structured error handling
├── Interpreter_plugin_base.ml   # Core implementation (677 lines)
├── Interpreter_plugin.ml        # BACKEND interface (168 lines)
└── test/
    └── test_interpreter_error.ml  # Error tests
```

## Usage

```ocaml
(* Select interpreter backend explicitly *)
let device = Sarek.Device.get_device_by_name "CPU Interpreter" in
vector_add ~block:(256, 1, 1) ~grid:(n/256, 1, 1) a b c
```

## Limitations

- Slower than Native backend (interpretation overhead)
- No external GPU source execution
- No raw pointer arguments
- No Ctypes buffer support (Bigarray only)

## Testing

```bash
dune test sarek/plugins/interpreter
```

## Use Cases

- **Debugging**: Step through kernel execution
- **Testing**: Validate kernel behavior without GPU
- **Development**: Quick iteration without compilation
- **Cross-platform**: Works everywhere OCaml runs
