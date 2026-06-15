(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Integration test for Sarek_fusion API

    Tests kernel fusion using the runtime API with actual PPX-defined kernels.
    Uses IR (Sarek_ir.kernel) with Sarek_fusion module. *)

(** Producer kernel: temp[i] = input[i] * 2 *)
let producer =
  [%kernel
    fun (temp : int32 vector) (input : int32 vector) ->
      temp.(thread_idx_x) <- input.(thread_idx_x) * 2l]

(** Consumer kernel: output[i] = temp[i] + 1 *)
let consumer =
  [%kernel
    fun (output : int32 vector) (temp : int32 vector) ->
      output.(thread_idx_x) <- temp.(thread_idx_x) + 1l]

(** Helper to get IR from kernel *)
let get_ir kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "Kernel has no IR"

(** Test that fusion API is accessible and works *)
let test_can_fuse () =
  let _, kirc_prod = producer in
  let _, kirc_cons = consumer in
  let ir_prod = get_ir kirc_prod in
  let ir_cons = get_ir kirc_cons in
  (* Check if fusion is possible *)
  let can_fuse = Sarek.Sarek_fusion.can_fuse ir_prod ir_cons "temp" in
  Printf.printf "can_fuse: %b\n" can_fuse ;
  assert can_fuse ;
  Printf.printf "test_can_fuse: PASSED\n"

(** Test fusing kernels directly *)
let test_fuse () =
  let _, kirc_prod = producer in
  let _, kirc_cons = consumer in
  let ir_prod = get_ir kirc_prod in
  let ir_cons = get_ir kirc_cons in
  let fused = Sarek.Sarek_fusion.fuse ir_prod ir_cons "temp" in
  (* Verify the fused kernel has a name *)
  Printf.printf "fused kernel name: %s\n" fused.Sarek.Sarek_ir.kern_name ;
  Printf.printf "test_fuse: PASSED\n"

(** Three-stage pipeline: a[i] = in[i]*2, b[i] = a[i]+1, out[i] = b[i]*3 *)
let stage1 =
  [%kernel
    fun (a : int32 vector) (input : int32 vector) ->
      a.(thread_idx_x) <- input.(thread_idx_x) * 2l]

let stage2 =
  [%kernel
    fun (b : int32 vector) (a : int32 vector) ->
      b.(thread_idx_x) <- a.(thread_idx_x) + 1l]

let stage3 =
  [%kernel
    fun (output : int32 vector) (b : int32 vector) ->
      output.(thread_idx_x) <- b.(thread_idx_x) * 3l]

(** Test pipeline fusion *)
let test_fuse_pipeline () =
  let _, k1 = stage1 in
  let _, k2 = stage2 in
  let _, k3 = stage3 in
  let kernels = [get_ir k1; get_ir k2; get_ir k3] in
  let fused, eliminated = Sarek.Sarek_fusion.fuse_pipeline kernels in
  Printf.printf "Eliminated intermediates: %s\n" (String.concat ", " eliminated) ;
  (* Should eliminate both a and b *)
  assert (List.mem "a" eliminated) ;
  assert (List.mem "b" eliminated) ;
  (* Verify result has a kernel name *)
  Printf.printf "fused kernel name: %s\n" fused.Sarek.Sarek_ir.kern_name ;
  Printf.printf "test_fuse_pipeline: PASSED\n"

let () =
  Printf.printf "=== Fusion API Integration Tests (runtime) ===\n" ;
  test_can_fuse () ;
  test_fuse () ;
  test_fuse_pipeline () ;
  Printf.printf "=== All integration tests passed! ===\n"
