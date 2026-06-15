# spoc_ir - GPU Kernel Intermediate Representation

The `spoc_ir` library defines the intermediate representation (IR) used to
represent GPU kernels. It provides pure type definitions, pretty printing,
and static analysis utilities.

## Module Overview

```
spoc_ir/
├── Sarek_ir_types.ml      # IR type definitions
├── Sarek_ir_pp.ml         # Pretty printing
└── Sarek_ir_analysis.ml   # Static analysis (float64 detection)
[../sarek/core/](../sarek/core/)             # Runtime core that executes generated kernels
[../sarek/ppx/](../sarek/ppx/)               # Sarek PPX compiler (produces this IR)
```

## IR Type Hierarchy

```
                           ┌──────────────┐
                           │    kernel    │
                           └──────┬───────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
     ┌──────▼──────┐       ┌──────▼──────┐      ┌──────▼──────┐
     │   decl      │       │   stmt      │      │ helper_func │
     └──────┬──────┘       └──────┬──────┘      └─────────────┘
            │                     │
  ┌─────────┼─────────┐    ┌──────┼──────┐
  │         │         │    │      │      │
DParam   DLocal   DShared  │   ┌──▼──┐   │
  │                     lvalue  expr  pattern
  │                        │      │
  ▼                        ▼      ▼
var ─────────────────────► elttype ◄── const
                              │
                ┌─────────────┼─────────────┐
                │             │             │
           primitives     compound      container
           TInt32         TRecord       TArray
           TFloat32       TVariant      TVec
           ...
```

## Sarek_ir_types.ml

### Element Types

```ocaml
type elttype =
  | TInt32 | TInt64          (* Integer types *)
  | TFloat32 | TFloat64      (* Floating point *)
  | TBool | TUnit            (* Boolean and unit *)
  | TRecord of string * (string * elttype) list    (* struct *)
  | TVariant of string * (string * elttype list) list  (* enum/union *)
  | TArray of elttype * memspace  (* Fixed-size array *)
  | TVec of elttype              (* GPU vector parameter *)

type memspace = Global | Shared | Local
```

### Variables

```ocaml
type var = {
  var_name : string;
  var_id : int;           (* Unique identifier for alpha-renaming *)
  var_type : elttype;
  var_mutable : bool;
}
```

### Constants

```ocaml
type const =
  | CInt32 of int32
  | CInt64 of int64
  | CFloat32 of float
  | CFloat64 of float
  | CBool of bool
  | CUnit
```

### Expressions (Pure)

```ocaml
type expr =
  | EConst of const                        (* Literal value *)
  | EVar of var                            (* Variable reference *)
  | EBinop of binop * expr * expr          (* Binary operation *)
  | EUnop of unop * expr                   (* Unary operation *)
  | EArrayRead of string * expr            (* arr[idx] *)
  | EArrayReadExpr of expr * expr          (* base_expr[idx] *)
  | ERecordField of expr * string          (* r.field *)
  | EIntrinsic of string list * string * expr list  (* Float32.sin(x) *)
  | ECast of elttype * expr                (* Type cast *)
  | ETuple of expr list                    (* Tuple construction *)
  | EApp of expr * expr list               (* Function application *)
  | ERecord of string * (string * expr) list  (* {x=1; y=2} *)
  | EVariant of string * string * expr list   (* Some(x) *)
  | EArrayLen of string                    (* Array length *)
  | EArrayCreate of elttype * expr * memspace  (* Array allocation *)
  | EIf of expr * expr * expr              (* Ternary if *)
  | EMatch of expr * (pattern * expr) list (* Pattern match *)
```

### Statements (Imperative)

```ocaml
type stmt =
  | SAssign of lvalue * expr              (* Assignment *)
  | SSeq of stmt list                     (* Sequence *)
  | SIf of expr * stmt * stmt option      (* Conditional *)
  | SWhile of expr * stmt                 (* While loop *)
  | SFor of var * expr * expr * for_dir * stmt  (* For loop *)
  | SMatch of expr * (pattern * stmt) list  (* Match statement *)
  | SReturn of expr                       (* Return value *)
  | SBarrier                              (* __syncthreads() *)
  | SWarpBarrier                          (* __syncwarp() *)
  | SMemFence                             (* __threadfence() *)
  | SExpr of expr                         (* Expression statement *)
  | SEmpty                                (* No-op *)
  | SLet of var * expr * stmt             (* Let binding *)
  | SLetMut of var * expr * stmt          (* Mutable let *)
  | SPragma of string list * stmt         (* Pragma hints *)
  | SBlock of stmt                        (* Scoped block *)
  | SNative of { gpu : framework:string -> string; ocaml : Obj.t }
```

### L-Values (Assignable)

```ocaml
type lvalue =
  | LVar of var                   (* Variable *)
  | LArrayElem of string * expr   (* arr[idx] *)
  | LArrayElemExpr of expr * expr (* base_expr[idx] *)
  | LRecordField of lvalue * string  (* r.field *)
```

### Declarations

```ocaml
type decl =
  | DParam of var * array_info option  (* Kernel parameter *)
  | DLocal of var * expr option        (* Local variable *)
  | DShared of string * elttype * expr option  (* Shared memory *)

and array_info = {
  arr_elttype : elttype;
  arr_memspace : memspace;
}
```

### Kernel

```ocaml
type kernel = {
  kern_name : string;
  kern_params : decl list;
  kern_locals : decl list;
  kern_body : stmt;
  kern_types : (string * (string * elttype) list) list;  (* Record types *)
  kern_variants : (string * (string * elttype list) list) list;  (* Variants *)
  kern_funcs : helper_func list;  (* Helper functions *)
  kern_native_fn : native_fn_t option;  (* Pre-compiled CPU function *)
}

type helper_func = {
  hf_name : string;
  hf_params : var list;
  hf_ret_type : elttype;
  hf_body : stmt;
}
```

### Native Arguments

For CPU execution:

```ocaml
type native_arg =
  | NA_Int32 of int32
  | NA_Int64 of int64
  | NA_Float32 of float
  | NA_Float64 of float
  | NA_Vec of {
      length : int;
      elem_size : int;
      type_name : string;
      get_f32 : int -> float;
      set_f32 : int -> float -> unit;
      (* ... *)
    }

(* Helper functions *)
val vec_get_custom : native_arg -> int -> 'a
val vec_set_custom : native_arg -> int -> 'a -> unit
val vec_length : native_arg -> int
```

## Sarek_ir_pp.ml

Pretty printing for debugging and code generation.

### String Conversion

```ocaml
val string_of_elttype : elttype -> string
(* TFloat32 -> "float32", TRecord("point", _) -> "point" *)

val string_of_memspace : memspace -> string
(* Global -> "global", Shared -> "shared" *)

val string_of_binop : binop -> string
(* Add -> "+", Mul -> "*", Eq -> "==" *)

val string_of_unop : unop -> string
(* Neg -> "-", Not -> "!", BitNot -> "~" *)
```

### Pretty Printers

```ocaml
val pp_elttype : Format.formatter -> elttype -> unit
val pp_var : Format.formatter -> var -> unit
val pp_expr : Format.formatter -> expr -> unit
val pp_stmt : Format.formatter -> stmt -> unit
val pp_decl : Format.formatter -> decl -> unit
val pp_pattern : Format.formatter -> pattern -> unit
val pp_kernel : Format.formatter -> kernel -> unit

val print_kernel : kernel -> unit  (* Print to stdout *)
```

### Example Output

```ocaml
let k = { kern_name = "vector_add"; ... } in
Sarek_ir_pp.print_kernel k
```

Output:
```
__kernel void vector_add(global float32* a, global float32* b, global float32* c, int32 n) {
  int32 idx = GPU.global_id_x;
  if ((idx < n)) {
    c[idx] = (a[idx] + b[idx]);
  }
}
```

## Sarek_ir_analysis.ml

Static analysis utilities for IR inspection.

### Float64 Detection

Checks whether a kernel uses double-precision floating point, which
requires FP64 extension on some GPUs:

```ocaml
val elttype_uses_float64 : elttype -> bool
val const_uses_float64 : const -> bool
val expr_uses_float64 : expr -> bool
val stmt_uses_float64 : stmt -> bool
val decl_uses_float64 : decl -> bool
val helper_uses_float64 : helper_func -> bool
val kernel_uses_float64 : kernel -> bool
```

### Usage

```ocaml
let requires_fp64 kernel =
  if Sarek_ir_analysis.kernel_uses_float64 kernel then begin
    (* Check device capability *)
    if not device.capabilities.supports_fp64 then
      failwith "Kernel requires FP64 but device doesn't support it"
  end
```

### What Gets Checked

The analysis traverses:
- All parameter and local variable types
- All constants in expressions
- All casts and intrinsic calls
- Record type definitions (`kern_types`)
- Variant type definitions (`kern_variants`)
- Helper function signatures and bodies

## Integration with Code Generation

The IR is consumed by backend-specific code generators:

```
┌─────────────────┐     ┌─────────────────┐
│  Sarek PPX      │────►│  Sarek_ir.kernel │
│  (parse/type)   │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Sarek_ir_cuda.ml│     │Sarek_ir_opencl.ml│    │Sarek_ir_glsl.ml │
│ generate_cuda() │     │generate_opencl() │    │generate_glsl()  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
     CUDA source           OpenCL source           GLSL source
```

## Testing

```bash
dune build @spoc/ir/test/runtest
```

Tests cover:
- `test_sarek_ir_types.ml` - Type construction, kernel structure
- `test_sarek_ir_pp.ml` - Pretty printing all constructs
- `test_sarek_ir_analysis.ml` - Float64 detection across all IR nodes
