(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

open Spoc
open Kirc

let f_to_kir f l =
  app
    (GlobalFun
       ( f.body,
         (match snd f.ret_val with
         | Vector.Int32 _ -> "int"
         | Vector.Float32 _ -> "float"
         | Vector.Custom _ -> (
             match fst f.ret_val with
             | CustomVar (s, _, _) -> "struct " ^ s ^ "_sarek"
             | _ ->
                 Kirc_error.raise_error
                   (Unsupported_arg_type
                      {
                        arg_type = "custom return type";
                        reason = "not a CustomVar";
                        context = "f_to_kir";
                      }))
         | _ -> "void"),
         "f" ))
    l

let map_skeleton f =
  spoc_gen_kernel
    (params
       (concat
          (new_int_vec_var 3 "a")
          (concat
             (new_int_vec_var 4 "b")
             (concat (new_int_var 5 "i") (empty_arg ())))))
    (spoc_local_env
       (spoc_declare (new_int_var 7 "tid"))
       (seq
          (spoc_set
             (var 7 "tid")
             (spoc_plus
                (intrinsics "threadIdx.x" "(get_local_id (0))")
                (spoc_mul
                   (intrinsics "blockIdx.x" "(get_group_id (0))")
                   (intrinsics "blockDim.x" "(get_local_size (0))"))))
          (spoc_if
             (lt32 (var 7 "tid") (var 5 "i"))
             (set_vect_var
                (get_vec (var 4 "b") (var 7 "tid"))
                (f_to_kir f [|get_vec (var 3 "a") (var 7 "tid")|])))))

let zip_skeleton f =
  spoc_gen_kernel
    (params
       (concat
          (new_int_vec_var 2 "a")
          (concat
             (new_int_vec_var 3 "b")
             (concat
                (new_int_vec_var 4 "c")
                (concat (new_int_var 5 "i") (empty_arg ()))))))
    (spoc_local_env
       (spoc_declare (new_int_var 7 "tid"))
       (seq
          (spoc_set
             (var 7 "tid")
             (spoc_plus
                (intrinsics "threadIdx.x" "(get_local_id (0))")
                (spoc_mul
                   (intrinsics "blockIdx.x" "(get_group_id (0))")
                   (intrinsics "blockDim.x" "(get_local_size (0))"))))
          (spoc_if
             (lt32 (var 7 "tid") (var 5 "i"))
             (set_vect_var
                (get_vec (var 4 "c") (var 7 "tid"))
                (f_to_kir
                   f
                   [|
                     get_vec (var 2 "a") (var 7 "tid");
                     get_vec (var 3 "b") (var 7 "tid");
                   |])))))
