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

let%kernel mandelbrot (output : int32 vector) 
                      (width : int32) (height : int32) 
                      (max_iter : int32) =
  let px = get_global_id 0 in (* x-coordinate *)
  let py = get_global_id 1 in (* y-coordinate *)
  
  if px < width && py < height then begin
    (* Map pixel to complex plane coordinates *)
    let x0 = (4.0 *. (float px /. float width)) -. 2.5 in
    let y0 = (3.0 *. (float py /. float height)) -. 1.5 in
    
    let x = ref 0.0 in
    let y = ref 0.0 in
    let iter = ref 0l in
    
    (* Main iteration loop *)
    while (!x *. !x) +. (!y *. !y) <= 4.0 && !iter < max_iter do
      let xtemp = (!x *. !x) -. (!y *. !y) +. x0 in
      y := (2.0 *. !x *. !y) +. y0;
      x := xtemp;
      iter := !iter + 1l
    done;
    
    (* Store iteration count as pixel value *)
    output.((py * width) + px) <- !iter
  end
```

## Host Code

```ocaml
let run_mandelbrot () =
  let width, height = 1024, 1024 in
  let max_iter = 256l in
  let output = Vector.create Int32 (width * height) in
  
  Execute.run mandelbrot
    ~device:(Device.get_default ())
    ~block:(16, 16, 1)
    ~grid:((width+15)/16, (height+15)/16, 1)
    [Vec output; Int32 width; Int32 height; Int32 max_iter]
```
