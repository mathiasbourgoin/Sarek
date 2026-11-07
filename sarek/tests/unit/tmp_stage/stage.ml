(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* INVESTIGATION PROBE (not for merge): a matvec staging weights through shared
   memory in a loop of global loads. Compares what ptxas makes of the PTX SPOC
   emits with what it makes of the PTX NVRTC emits for the CUDA-C SPOC emits for
   the same IR: registers, and the SASS instruction mix. Prints, asserts
   nothing. *)

let staging =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (n : int32) ->
      let%shared (stage : float32) = 256l in
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      let c = mut 0l in
      while c < n do
        stage.(tid) <- w.((block_idx_x * n) + c + tid) ;
        block_barrier () ;
        let j = mut 0l in
        while j < 256l do
          acc := acc +. (stage.(j) *. x.(c + j)) ;
          j := j + 1l
        done ;
        block_barrier () ;
        c := c + 256l
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

(* the batched shape: several global loads per thread per chunk, then the
   shared stores, then one barrier (what a hand-unrolled staging loop does) *)
let staging_for =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (chunks : int32) ->
      let%shared (stage : float32) = 256l in
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      for c = 0 to chunks - 1l do
        stage.(tid) <- w.((((block_idx_x * chunks) + c) * 256l) + tid) ;
        block_barrier () ;
        for j = 0 to 255l do
          acc := acc +. (stage.(j) *. x.((c * 256l) + j))
        done ;
        block_barrier ()
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let read_file path = In_channel.with_open_bin path In_channel.input_all

let run_capture cmd =
  let ic = Unix.open_process_in (cmd ^ " 2>&1") in
  let out = In_channel.input_all ic in
  ignore (Unix.close_process_in ic) ;
  out

let contains hay needle =
  let n = String.length needle and h = String.length hay in
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  go 0

let lines s = String.split_on_char '\n' s

let count_op sass op =
  List.length
    (List.filter
       (fun l ->
         (* "/*0070*/  LDG.E R2, [R2.64] ;" *)
         contains l (" " ^ op ^ ".") || contains l (" " ^ op ^ " "))
       (lines sass))

let analyse label ptx =
  let base = Filename.temp_file "stage_" "" in
  let ptx_f = base ^ ".ptx" and cubin = base ^ ".cubin" in
  Out_channel.with_open_bin ptx_f (fun oc -> output_string oc ptx) ;
  let info =
    run_capture
      (Printf.sprintf
         "ptxas -arch=sm_75 -v -o %s %s"
         (Filename.quote cubin)
         (Filename.quote ptx_f))
  in
  let sass =
    run_capture (Printf.sprintf "nvdisasm -c %s" (Filename.quote cubin))
  in
  Printf.printf "---- %s\n" label ;
  List.iter
    (fun l -> if contains l "Used" || contains l "error" then print_endline l)
    (lines info) ;
  Printf.printf
    "  SASS lines %d | LDG %d  LDS %d  STS %d  BAR %d  BRA %d  FFMA %d  FMUL \
     %d  FADD %d  IMAD %d  MOV %d\n\
     %!"
    (List.length (lines sass))
    (count_op sass "LDG")
    (count_op sass "LDS")
    (count_op sass "STS")
    (count_op sass "BAR")
    (count_op sass "BRA")
    (count_op sass "FFMA")
    (count_op sass "FMUL")
    (count_op sass "FADD")
    (count_op sass "IMAD")
    (count_op sass "MOV") ;
  (sass, ptx)

let on_path tool =
  Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" tool)
  = Unix.WEXITED 0

let () =
  if
    not
      (Sarek_cuda.Cuda_nvrtc.is_available ()
      && on_path "ptxas" && on_path "nvdisasm")
  then print_endline "SKIP: needs libnvrtc, ptxas, nvdisasm"
  else
    List.iter
      (fun (name, k) ->
        let ir = ir_of k in
        let spoc_ptx =
          Sarek_codegen.Sarek_ir_ptx.generate
            ~sm_target:"sm_75"
            ~block:(256, 1, 1)
            ir
        in
        let cuda =
          Sarek_codegen.Sarek_ir_cuda.generate_with_types
            ~block:(256, 1, 1)
            ~types:[]
            ir
        in
        let nvrtc_ptx =
          Sarek_cuda.Cuda_nvrtc.compile_to_ptx
            ~name:"sarek_kern"
            ~arch:"compute_75"
            cuda
        in
        Printf.printf "######## %s\n" name ;
        let sass_spoc, _ = analyse (name ^ ": SPOC PTX") spoc_ptx in
        let sass_nvrtc, _ =
          analyse (name ^ ": NVRTC PTX of the CUDA-C") nvrtc_ptx
        in
        Printf.printf "######## %s: NVRTC PTX (kernel)\n%s\n" name nvrtc_ptx ;
        Printf.printf "######## %s: SASS of SPOC PTX\n%s\n" name sass_spoc ;
        Printf.printf "######## %s: SASS of NVRTC PTX\n%s\n" name sass_nvrtc)
      [("while", staging); ("for", staging_for)]
