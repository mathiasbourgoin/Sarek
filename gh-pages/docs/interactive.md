---
layout: page
title: Interactive Sarek
---

# Interactive Sarek with Jupyter

You can use Sarek interactively using the [ocaml-jupyter](https://github.com/akabe/ocaml-jupyter) kernel. This is an excellent way to prototype kernels, visualize GPU data, and document your experiments.

## Setup

1. **Install Jupyter and the OCaml Kernel**:
   ```bash
   pip install jupyterlab
   opam install jupyter
   ocaml-jupyter-interpreter install
   ```

2. **Install Sarek**:
   Ensure Sarek and its PPX are installed in your current switch:
   ```bash
   opam install sarek sarek-cuda  # or your preferred backend
   ```

3. **Launch Jupyter**:
   ```bash
   jupyter lab
   ```

## Using the Native Parallel Backend

A major advantage of Sarek's modern architecture is the **Native CPU backend**. This is perfect for Jupyter environments (like Binder or local laptops) where a GPU might not be available.

You can still experiment with parallel algorithms by targeting your CPU's multiple cores:

```ocaml
(* 1. Initialize Sarek *)
#require "sarek";;
#require "sarek.ppx";;
open Sarek;;

(* 2. Define your parallel kernel *)
let%kernel compute_pi (out : float32 vector) (n : int32) =
  let gid = get_global_id 0 in
  if gid < n then
    let x = (float gid +. 0.5) /. float n in
    out.(gid) <- 4.0 /. (1.0 +. x *. x)

(* 3. Select the Native Parallel Device *)
let dev = Device.get_device_by_name "Native Parallel" in
Printf.printf "Running on: %s\n" (Device.name dev);;

(* 4. Execute (Runs across all available OCaml 5 Domains) *)
let n = 1000000 in
let results = Vector.create Float32 n in
Execute.run compute_pi ~device:dev ~grid:(n/256, 1, 1) ~block:(256, 1, 1) [Vec results; Int n];;
```

This allows you to prototype and verify the **exact same logic** that will later run on a high-end NVIDIA or AMD GPU.

## Visualizing Results

One of the big advantages of using Jupyter is the ability to render GPU results immediately. For example, after running a Mandelbrot kernel, you can use OCaml libraries to display the resulting `Vector` as an image directly in the notebook.

## Try it Online (Binder)

*Coming Soon: We are working on a pre-configured Binder environment so you can test Sarek kernels in your browser with zero installation.*
