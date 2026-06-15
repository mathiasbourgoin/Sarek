(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Tests for GPU memory management (Gpu_memory module) *)

open Spoc_core

let () =
  (* Test 1: auto_gc flag *)
  assert (Gpu_memory.get_auto_gc ()) ;
  Gpu_memory.set_auto_gc false ;
  assert (not (Gpu_memory.get_auto_gc ())) ;
  Gpu_memory.set_auto_gc true ;
  assert (Gpu_memory.get_auto_gc ()) ;
  Printf.printf "PASS: auto_gc flag\n" ;

  (* Test 2: memory tracking *)
  let initial = Gpu_memory.usage () in
  Gpu_memory.track_alloc 1024 ;
  assert (Gpu_memory.usage () = initial + 1024) ;
  Gpu_memory.track_free 512 ;
  assert (Gpu_memory.usage () = initial + 512) ;
  Gpu_memory.track_free 512 ;
  assert (Gpu_memory.usage () = initial) ;
  Printf.printf "PASS: memory tracking\n" ;

  (* Test 3: with_retry succeeds on first try *)
  let count = ref 0 in
  let result =
    Gpu_memory.with_retry (fun () ->
        incr count ;
        42)
  in
  assert (result = 42) ;
  assert (!count = 1) ;
  Printf.printf "PASS: with_retry succeeds immediately\n" ;

  (* Test 4: with_retry retries after GC *)
  let fail_count = ref 0 in
  let result =
    Gpu_memory.with_retry (fun () ->
        incr fail_count ;
        if !fail_count < 2 then failwith "OOM" ;
        99)
  in
  assert (result = 99) ;
  assert (!fail_count = 2) ;
  Printf.printf "PASS: with_retry retries after GC\n" ;

  (* Test 5: with_retry propagates after 3 failures *)
  let fail_count = ref 0 in
  let caught =
    try
      ignore
        (Gpu_memory.with_retry (fun () ->
             incr fail_count ;
             failwith "persistent OOM")
          : int) ;
      false
    with Failure _ -> true
  in
  assert caught ;
  assert (!fail_count = 3) ;
  Printf.printf "PASS: with_retry propagates after 3 failures\n" ;

  (* Test 6: with_retry disabled *)
  Gpu_memory.set_auto_gc false ;
  let fail_count = ref 0 in
  let caught =
    try
      ignore
        (Gpu_memory.with_retry (fun () ->
             incr fail_count ;
             failwith "OOM")
          : int) ;
      false
    with Failure _ -> true
  in
  assert caught ;
  assert (!fail_count = 1) ;
  (* No retry when disabled *)
  Gpu_memory.set_auto_gc true ;
  Printf.printf "PASS: with_retry respects auto_gc=false\n" ;

  (* Test 7: trigger_gc runs without error *)
  Gpu_memory.trigger_gc () ;
  Printf.printf "PASS: trigger_gc\n" ;

  (* Test 8: stats *)
  Gpu_memory.reset_stats () ;
  Gpu_memory.track_alloc 4096 ;
  Gpu_memory.track_alloc 8192 ;
  Gpu_memory.track_free 4096 ;
  let s = Gpu_memory.stats () in
  assert (s.current_bytes = 8192) ;
  assert (s.peak_bytes = 12288) ;
  assert (s.alloc_count = 2) ;
  assert (s.free_count = 1) ;
  Printf.printf "PASS: stats tracking\n" ;

  (* Test 9: reset_stats *)
  Gpu_memory.reset_stats () ;
  let s = Gpu_memory.stats () in
  assert (s.current_bytes = 0) ;
  assert (s.peak_bytes = 0) ;
  assert (s.alloc_count = 0) ;
  Printf.printf "PASS: reset_stats\n" ;

  Printf.printf "\nAll GPU memory tests passed!\n"
