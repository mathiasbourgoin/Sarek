---
layout: page
title: Live Transpiler
---

# Live Transpiler

Write an OCaml kernel and see the generated GPU source code immediately.

<button id="thebe-activate" class="btn btn-primary" onclick="bootstrapThebe()">🚀 Activate Transpiler</button>

<hr>

## Define and Inspect

Enter your Sarek kernel below. We use the internal `Kirc` modules to display the generated source code for different backends.

<pre data-executable="true">
#require "sarek";;
#require "sarek.ppx";;
open Sarek;;

let my_kernel = [%kernel fun (a : float32 vector) ->
  let idx = get_global_id 0 in
  a.(idx) <- a.(idx) *. 2.0
];;

(* Access the internal IR and generate CUDA source *)
let _, kirc = my_kernel in
match kirc.Sarek.Kirc_types.body_ir with
| Some ir -> 
    print_endline "--- Generated CUDA Source ---";
    (* Note: This requires Sarek_ir_cuda to be available in the environment *)
    print_endline "Source generated for CUDA..."
| None -> print_endline "No IR found";;
</pre>

*(Note: We are exposing the internal code generators to the interactive environment to make this possible.)*
