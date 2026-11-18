(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E: signed int32/int64 [+], [-], [*], unary [-] and [lsl] wrap like OCaml
 * on every device (Sarek_ir_c_arith).
 *
 * Each thread reads two operands, computes the five operations and a chain
 * ([a * 3 - (-b)]), and stores the results; the operands include min_int,
 * max_int and products that overflow. Every device must return OCaml's
 * Int32/Int64 results: Native and the Interpreter compute them with OCaml's
 * arithmetic, the C-family backends through the unsigned type, PTX/GLSL/WGSL
 * natively. Shift counts stay in 0 .. width-1 (outside, OCaml leaves the
 * result unspecified). Native/Interpreter must run; a GPU backend that cannot
 * launch is reported, a wrong result from any device is a failure.
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let ops = 6

let arith32_kernel =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (s : int32 vector)
        (out : int32 vector)
        (n : int32) ->
      let i = thread_idx_x + (block_dim_x * block_idx_x) in
      if i < n then begin
        let x = a.(i) in
        let y = b.(i) in
        out.(i * 6l) <- x + y ;
        out.((i * 6l) + 1l) <- x - y ;
        out.((i * 6l) + 2l) <- x * y ;
        out.((i * 6l) + 3l) <- -x ;
        out.((i * 6l) + 4l) <- x lsl s.(i) ;
        out.((i * 6l) + 5l) <- (x * 3l) - -y
      end]

let arith64_kernel =
  [%kernel
    fun (a : int64 vector)
        (b : int64 vector)
        (s : int64 vector)
        (out : int64 vector)
        (n : int32) ->
      let i = thread_idx_x + (block_dim_x * block_idx_x) in
      if i < n then begin
        let x = a.(i) in
        let y = b.(i) in
        out.(i * 6l) <- x + y ;
        out.((i * 6l) + 1l) <- x - y ;
        out.((i * 6l) + 2l) <- x * y ;
        out.((i * 6l) + 3l) <- -x ;
        out.((i * 6l) + 4l) <- x lsl s.(i) ;
        out.((i * 6l) + 5l) <- (x * 3L) - -y
      end]

let reference32 x y s =
  let open Int32 in
  [|
    add x y;
    sub x y;
    mul x y;
    neg x;
    shift_left x (to_int s);
    sub (mul x 3l) (neg y);
  |]

let reference64 x y s =
  let open Int64 in
  [|
    add x y;
    sub x y;
    mul x y;
    neg x;
    shift_left x (to_int s);
    sub (mul x 3L) (neg y);
  |]

let values32 =
  [|
    0l;
    1l;
    -1l;
    Int32.min_int;
    Int32.max_int;
    0x40000000l;
    65536l;
    -65536l;
    0x7fff0001l;
    0xdeadbeefl;
  |]

let values64 =
  [|
    0L;
    1L;
    -1L;
    Int64.min_int;
    Int64.max_int;
    0x4000000000000000L;
    4294967296L;
    -4294967296L;
    0xdeadbeefcafebabeL;
  |]

(* Every pair of values, with a shift count that cycles through the range. *)
let rows values width =
  let n = Array.length values in
  Array.init (n * n) (fun k -> (values.(k / n), values.(k mod n), k mod width))

let launch dev ir ~n args =
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d ((n + 63) / 64))
    () ;
  Transfer.flush dev

(* Returns (wrong results, total), printing the first few. *)
let compare label total ok show =
  let bad = ref 0 in
  for j = 0 to total - 1 do
    if not (ok j) then begin
      if !bad < 4 then Printf.printf "    %s result %d: %s\n%!" label j (show j) ;
      incr bad
    end
  done ;
  (!bad, total)

let run32 dev ir =
  let rows = rows values32 32 in
  let n = Array.length rows in
  let a = Vector.create Vector.int32 n in
  let b = Vector.create Vector.int32 n in
  let s = Vector.create Vector.int32 n in
  let out = Vector.create Vector.int32 (n * ops) in
  Array.iteri
    (fun i (x, y, k) ->
      Vector.set a i x ;
      Vector.set b i y ;
      Vector.set s i (Int32.of_int k))
    rows ;
  for j = 0 to (n * ops) - 1 do
    Vector.set out j 0x5a5a5a5al
  done ;
  launch
    dev
    ir
    ~n
    [
      Execute.Vec a; Execute.Vec b; Execute.Vec s; Execute.Vec out; Execute.Int n;
    ] ;
  let expected j =
    let x, y, k = rows.(j / ops) in
    (reference32 x y (Int32.of_int k)).(j mod ops)
  in
  compare
    "int32"
    (n * ops)
    (fun j -> Vector.get out j = expected j)
    (fun j ->
      let x, y, k = rows.(j / ops) in
      Printf.sprintf
        "op %d on (%ld, %ld, %d): got %ld, expected %ld"
        (j mod ops)
        x
        y
        k
        (Vector.get out j)
        (expected j))

let run64 dev ir =
  let rows = rows values64 64 in
  let n = Array.length rows in
  let a = Vector.create Vector.int64 n in
  let b = Vector.create Vector.int64 n in
  let s = Vector.create Vector.int64 n in
  let out = Vector.create Vector.int64 (n * ops) in
  Array.iteri
    (fun i (x, y, k) ->
      Vector.set a i x ;
      Vector.set b i y ;
      Vector.set s i (Int64.of_int k))
    rows ;
  for j = 0 to (n * ops) - 1 do
    Vector.set out j 0x5a5a5a5aL
  done ;
  launch
    dev
    ir
    ~n
    [
      Execute.Vec a; Execute.Vec b; Execute.Vec s; Execute.Vec out; Execute.Int n;
    ] ;
  let expected j =
    let x, y, k = rows.(j / ops) in
    (reference64 x y (Int64.of_int k)).(j mod ops)
  in
  compare
    "int64"
    (n * ops)
    (fun j -> Vector.get out j = expected j)
    (fun j ->
      let x, y, k = rows.(j / ops) in
      Printf.sprintf
        "op %d on (%Ld, %Ld, %d): got %Ld, expected %Ld"
        (j mod ops)
        x
        y
        k
        (Vector.get out j)
        (expected j))

let is_host (dev : Device.t) =
  dev.Device.framework = "Native" || dev.Device.framework = "Interpreter"

(* WebGPU has no 64-bit integers and refuses the kernel at code generation. *)
let supports_int64 (dev : Device.t) = dev.Device.framework <> "WebGPU"

let ir_of kern name =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith (name ^ " kernel has no IR")

(* The C-family source of the int32 kernel spells every operation through
   the unsigned type; checked once, without a device. *)
let check_source () =
  let src = Sarek_codegen.Sarek_ir_cuda.generate (ir_of arith32_kernel "a32") in
  let contains sub =
    let n = String.length sub and m = String.length src in
    let rec go i = i + n <= m && (String.sub src i n = sub || go (i + 1)) in
    go 0
  in
  List.for_all
    contains
    [
      "((int)((unsigned int)x + (unsigned int)y))";
      "((int)((unsigned int)x - (unsigned int)y))";
      "((int)((unsigned int)x * (unsigned int)y))";
      "((int)(-(unsigned int)x))";
      "((int)((unsigned int)x << s[i]))";
      "((int)(((unsigned int)x * 3u) - (-(unsigned int)y)))";
    ]

let () =
  let cases =
    [
      ("int32", run32, ir_of arith32_kernel "a32", Fun.const true);
      ("int64", run64, ir_of arith64_kernel "a64", supports_int64);
    ]
  in
  print_endline "=== wrapping signed arithmetic, E2E ===" ;
  let failed = ref false in
  if check_source () then
    print_endline "  CUDA C source spells the wrapping forms PASS"
  else begin
    print_endline "  CUDA C source spells the wrapping forms FAIL" ;
    failed := true
  end ;
  let devs = Device.init () in
  let host_ran = ref false in
  let report (dev : Device.t) label status detail =
    Printf.printf
      "  %-11s %-34s %-6s %s %s\n%!"
      dev.Device.framework
      dev.Device.name
      label
      status
      detail
  in
  Array.iter
    (fun (dev : Device.t) ->
      let host = is_host dev in
      List.iter
        (fun (label, runner, ir, applies) ->
          if not (applies dev) then report dev label "SKIP" "(no int64)"
          else
            match runner dev ir with
            | bad, total ->
                report
                  dev
                  label
                  (if bad = 0 then "PASS" else "FAIL")
                  (Printf.sprintf "(%d/%d ok)" (total - bad) total) ;
                if bad <> 0 then failed := true ;
                if host then host_ran := true
            | exception e ->
                let regression =
                  host || Test_helpers.Backend_failure.verdict e = `Fail
                in
                report
                  dev
                  label
                  (if regression then "FAIL" else "SKIP")
                  (Printexc.to_string e) ;
                if regression then failed := true)
        cases)
    devs ;
  if not !host_ran then begin
    print_endline
      "test_wrapping_arith: FAILED - no native/interpreter device ran" ;
    exit 1
  end ;
  if !failed then begin
    print_endline "test_wrapping_arith: FAILED" ;
    exit 1
  end ;
  print_endline "test_wrapping_arith: PASSED"
