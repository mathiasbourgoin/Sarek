---
layout: index_sample
title: Live Mandelbrot
---

# Live Mandelbrot

Experiment with Sarek kernels in real-time. Modify the code below and run it to see the results.

<button id="thebe-activate" class="btn btn-primary" onclick="bootstrapThebe()">🚀 Activate Live Demo</button>

<hr>

## 1. Setup and Kernel Definition

First, we load Sarek and define the Mandelbrot kernel. You can modify the escape condition or the coloring logic here.

<pre data-executable="true">
(* Load Sarek *)
#require "sarek";;
#require "sarek.ppx";;
open Sarek;;

let%kernel mandelbrot (output : int32 vector) (width : int32) (height : int32) (max_iter : int32) =
  let px = get_global_id 0 in
  let py = get_global_id 1 in
  
  if px < width && py < height then begin
    let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
    let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
    
    let x = mut 0.0 in
    let y = mut 0.0 in
    let iter = mut 0l in
    
    while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
      let xtemp = (x *. x) -. (y *. y) +. x0 in
      y := (2.0 *. x *. y) +. y0;
      x := xtemp;
      iter := iter + 1l
    done;
    
    output.((py * width) + px) <- iter
  end
</pre>

## 2. Execute and Display

Run the kernel using the **Native Parallel** backend and display the result.

<pre data-executable="true">
let width, height = 400, 300 in
let max_iter = 128l in
let output = Vector.create Int32 (width * height) in
let dev = Device.get_device_by_name "Native Parallel" in

Execute.run mandelbrot
  ~device:dev
  ~block:(16, 16, 1)
  ~grid:((width+15)/16, (height+15)/16, 1)
  [Vec output; Int32 (Int32.of_int width); Int32 (Int32.of_int height); Int32 max_iter];;

(* Display Logic (Pseudo-code, requires jupyter display support) *)
print_endline "Kernel Execution Complete!";;
</pre>

*(Note: Rich image display requires standard Jupyter image libraries to be configured in the Binder environment.)*
