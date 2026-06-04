---
layout: page
title: Mandelbrot Set Example
---

# Mandelbrot Set

The Mandelbrot set is a classic example of a "compute-bound" kernel. It involves heavy arithmetic operations per pixel with no dependencies between pixels.

## Kernel Code

Each thread computes the color of one pixel by iterating the complex function $z_{n+1} = z_n^2 + c$.

```ocaml
open Sarek
module Std = Sarek_stdlib.Std

let mandelbrot_kernel =
  [%kernel
    fun (output : int32 vector)
        (width : int32)
        (height : int32)
        (max_iter : int32) ->
      let open Std in
      let px = global_idx_x in
      let py = global_idx_y in
      
      if px < width && py < height then begin
        (* Map pixel to complex plane coordinates *)
        let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
        let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
        
        (* Use mut for mutable variables *)
        let x = mut 0.0 in
        let y = mut 0.0 in
        let iter = mut 0l in
        
        (* Main iteration loop *)
        while (x *. x) +. (y *. y) <= 4.0 && iter < max_iter do
          let xtemp = (x *. x) -. (y *. y) +. x0 in
          y := (2.0 *. x *. y) +. y0;
          x := xtemp;
          iter := iter + 1l
        done;
        
        (* Store iteration count as pixel value *)
        output.((py * width) + px) <- iter
      end]
```

## Host Code

```ocaml
open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let run_mandelbrot () =
  let width, height = 1024, 1024 in
  let max_iter = 256l in
  
  (* Get IR from kernel *)
  let _, kirc = mandelbrot_kernel in
  let ir = match kirc.Sarek.Kirc_types.body_ir with
    | Some ir -> ir | None -> failwith "No IR" in
  
  (* Create output vector *)
  let output = Vector.create Vector.int32 (width * height) in
  
  (* Calculate grid dimensions - using 16x16 blocks *)
  let block = Execute.dims2d 16 16 in
  let grid = Execute.dims2d ((width + 15) / 16) ((height + 15) / 16) in
  let dev = Device.best () in
  
  (* Run kernel *)
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec output; Int width; Int height; Int (Int32.to_int max_iter)]
    ~block
    ~grid
    ();
  
  (* Output can now be saved as an image *)
  Printf.printf "Generated %dx%d Mandelbrot set\n" width height
```
