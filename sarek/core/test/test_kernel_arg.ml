(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Kernel_arg module
 *
 * Covers type helpers, accessors, and folds without requiring a backend.
 ******************************************************************************)

open Spoc_core

let sample_vector () = Vector.create_float32 4

(** {1 Accessors} *)

let test_as_vector () =
  let v = sample_vector () in
  let (Kernel_arg.AnyVec (v', kind)) =
    Option.get (Kernel_arg.as_vector (Kernel_arg.Vec v))
  in
  assert (Vector.length v' = 4) ;
  assert (Vector.kind_name kind = "Float32") ;
  print_endline "  as_vector: OK"

let test_vector_length () =
  let v = sample_vector () in
  assert (Kernel_arg.vector_length (Kernel_arg.Vec v) = Some 4) ;
  assert (Kernel_arg.vector_length (Kernel_arg.Int 3) = None) ;
  print_endline "  vector_length: OK"

let test_as_int32 () =
  assert (Kernel_arg.as_int32 (Kernel_arg.Int 7) = Some 7l) ;
  assert (Kernel_arg.as_int32 (Kernel_arg.Int32 9l) = Some 9l) ;
  assert (Kernel_arg.as_int32 (Kernel_arg.Float32 1.0) = None) ;
  print_endline "  as_int32: OK"

let test_as_int64 () =
  assert (Kernel_arg.as_int64 (Kernel_arg.Int 5) = Some 5L) ;
  assert (Kernel_arg.as_int64 (Kernel_arg.Int32 6l) = Some 6L) ;
  assert (Kernel_arg.as_int64 (Kernel_arg.Int64 42L) = Some 42L) ;
  assert (Kernel_arg.as_int64 (Kernel_arg.Float64 3.14) = None) ;
  print_endline "  as_int64: OK"

let test_as_float () =
  assert (Kernel_arg.as_float (Kernel_arg.Float32 1.25) = Some 1.25) ;
  assert (Kernel_arg.as_float (Kernel_arg.Float64 2.5) = Some 2.5) ;
  assert (Kernel_arg.as_float (Kernel_arg.Int 1) = None) ;
  print_endline "  as_float: OK"

(** {1 Fold/Iter/Map} *)

let test_fold () =
  let args =
    [
      Kernel_arg.Vec (sample_vector ());
      Kernel_arg.Int 1;
      Kernel_arg.Int32 2l;
      Kernel_arg.Int64 3L;
      Kernel_arg.Float32 4.0;
      Kernel_arg.Float64 5.0;
    ]
  in
  let folder =
    {
      Kernel_arg.on_vec =
        (fun _ (vc, ic, i32c, i64c, f32c, f64c) ->
          (vc + 1, ic, i32c, i64c, f32c, f64c));
      on_int =
        (fun _ (vc, ic, i32c, i64c, f32c, f64c) ->
          (vc, ic + 1, i32c, i64c, f32c, f64c));
      on_int32 =
        (fun _ (vc, ic, i32c, i64c, f32c, f64c) ->
          (vc, ic, i32c + 1, i64c, f32c, f64c));
      on_int64 =
        (fun _ (vc, ic, i32c, i64c, f32c, f64c) ->
          (vc, ic, i32c, i64c + 1, f32c, f64c));
      on_float32 =
        (fun _ (vc, ic, i32c, i64c, f32c, f64c) ->
          (vc, ic, i32c, i64c, f32c + 1, f64c));
      on_float64 =
        (fun _ (vc, ic, i32c, i64c, f32c, f64c) ->
          (vc, ic, i32c, i64c, f32c, f64c + 1));
    }
  in
  let counts = Kernel_arg.fold folder args (0, 0, 0, 0, 0, 0) in
  assert (counts = (1, 1, 1, 1, 1, 1)) ;
  print_endline "  fold: OK"

let test_iter () =
  let vec_seen = ref false in
  let ints = ref [] in
  let floats = ref [] in
  let args =
    [
      Kernel_arg.Vec (sample_vector ());
      Kernel_arg.Int 10;
      Kernel_arg.Float64 7.5;
    ]
  in
  let iterator =
    {
      Kernel_arg.iter_vec = (fun _ -> vec_seen := true);
      iter_int = (fun n -> ints := n :: !ints);
      iter_int32 = (fun _ -> ());
      iter_int64 = (fun _ -> ());
      iter_float32 = (fun _ -> ());
      iter_float64 = (fun f -> floats := f :: !floats);
    }
  in
  Kernel_arg.iter iterator args ;
  assert !vec_seen ;
  assert (!ints = [10]) ;
  assert (!floats = [7.5]) ;
  print_endline "  iter: OK"

let test_map () =
  let args =
    [
      Kernel_arg.Int 3; Kernel_arg.Float32 2.0; Kernel_arg.Vec (sample_vector ());
    ]
  in
  let mapper =
    {
      Kernel_arg.map_vec = (fun v -> Printf.sprintf "vec:%d" (Vector.length v));
      map_int = string_of_int;
      map_int32 = Int32.to_string;
      map_int64 = Int64.to_string;
      map_float32 = string_of_float;
      map_float64 = string_of_float;
    }
  in
  let mapped = Kernel_arg.map mapper args in
  assert (mapped = ["3"; "2."; "vec:4"]) ;
  print_endline "  map: OK"

(** {1 Main} *)

let () =
  print_endline "Kernel_arg module tests:" ;
  test_as_vector () ;
  test_vector_length () ;
  test_as_int32 () ;
  test_as_int64 () ;
  test_as_float () ;
  test_fold () ;
  test_iter () ;
  test_map () ;
  print_endline "All Kernel_arg module tests passed!"
