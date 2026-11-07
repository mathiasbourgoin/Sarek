(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Loads in flight: the PTX SPOC emits, as ptxas compiles it, against the PTX
 * NVRTC emits for the CUDA-C SPOC emits for the same IR.
 *
 * WHAT THIS PROVES. For a loop with a runtime trip count and several
 * independent global loads per iteration, the PTX generator's output reaches
 * [ptxas] as a rolled loop that [ptxas] unrolls by 4 at most, while NVVM unrolls
 * the CUDA-C by 16. The PTX generator now unrolls such loops itself
 * (Sarek_ir_ptx_unroll), marks the loads of never-written vector parameters
 * [ld.global.nc] (Sarek_ir_ptx_readonly) and forms an address with one
 * [mad.wide.s32]. This test measures it on the machine code, with host tools
 * only (no NVIDIA device): [ptxas -v] for registers, [nvdisasm -c] for the
 * SASS, then
 *   - LDG:      load instructions in the SASS;
 *   - CONSTANT: the ones that go through the read-only path (LDG.E.CONSTANT);
 *   - in flight: the largest number of loads between two control-flow points
 *     (branch, barrier, exit, label), i.e. in one straight-line block: the
 *     loads a thread can have outstanding at once in the unrolled main loop.
 *
 * WHAT IT DOES NOT PROVE. Any timing. Whether more loads in flight is faster on
 * a given card is a hardware question; a contributor with an NVIDIA GPU
 * measures it (see the pull request).
 *
 * Asserts (when ptxas and nvdisasm are on the PATH): ptxas accepts every PTX;
 * for the streaming kernels at least [min_in_flight] loads are in flight, and
 * for the one whose parameters are never written every load is [.CONSTANT].
 * Skips cleanly without the tools. Prints one table.
 ******************************************************************************)

let min_in_flight = 16

let dot =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      let k = mut tid in
      while k < n do
        acc := acc +. (w.((block_idx_x * n) + k) *. x.(k)) ;
        k := k + 256l
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

(* Kernel B of the investigation: staging through shared memory, then a
   constant-trip dot product over the staged tile. *)
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

(* A quantised-weight matvec shape: one packed word of four 8-bit quants, one
   scale and four activations per iteration. *)
let quantised =
  [%kernel
    fun (wq : int32 vector)
        (sc : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (nb : int32) ->
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      let k = mut tid in
      while k < nb do
        let w = wq.((block_idx_x * nb) + k) in
        let s = sc.((block_idx_x * nb) + k) in
        let q0 = float ((w land 255l) - 128l) in
        let q1 = float ((w / 256l land 255l) - 128l) in
        let q2 = float ((w / 65536l land 255l) - 128l) in
        let q3 = float ((w / 16777216l land 255l) - 128l) in
        acc :=
          acc
          +. s
             *. ((q0 *. x.(k * 4l))
                +. (q1 *. x.((k * 4l) + 1l))
                +. (q2 *. x.((k * 4l) + 2l))
                +. (q3 *. x.((k * 4l) + 3l))) ;
        k := k + 256l
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

(* A [for] loop with a runtime bound. *)
let for_dot =
  [%kernel
    fun (w : float32 vector)
        (x : float32 vector)
        (y : float32 vector)
        (n : int32) ->
      let tid = thread_idx_x in
      let acc = mut 0.0 in
      for k = 0 to n - 1l do
        acc := acc +. (w.((block_idx_x * n) + k) *. x.(k))
      done ;
      y.((block_idx_x * 256l) + tid) <- acc]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

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

let on_path tool =
  Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" tool)
  = Unix.WEXITED 0

(* Opcode of a SASS line: "/*0070*/  @P0 LDG.E.CONSTANT R2, [R2.64] ;" -> the
   word after the optional predicate. *)
let opcode line =
  match String.index_opt line '/' with
  | None -> None
  | Some _ -> (
      let body =
        match String.rindex_opt line '/' with
        | Some i -> String.sub line (i + 1) (String.length line - i - 1)
        | None -> line
      in
      let words =
        List.filter (fun w -> w <> "") (String.split_on_char ' ' body)
      in
      let words =
        match words with
        | w :: rest when String.length w > 0 && w.[0] = '@' -> rest
        | l -> l
      in
      match words with w :: _ -> Some w | [] -> None)

let starts_with s prefix =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

type metrics = {
  registers : string;
  ldg : int;
  constant : int;
  in_flight : int;
  imad : int;
  imad_wide : int;
  lea : int;
  sass_lines : int;
}

let analyse_sass sass registers =
  let ldg = ref 0 and constant = ref 0 in
  let imad = ref 0 and imad_wide = ref 0 and lea = ref 0 in
  let run = ref 0 and best = ref 0 in
  let cut () =
    best := max !best !run ;
    run := 0
  in
  List.iter
    (fun line ->
      if starts_with (String.trim line) ".L_" then cut ()
      else
        match opcode line with
        | Some op when starts_with op "IMAD" ->
            incr imad ;
            if starts_with op "IMAD.WIDE" then incr imad_wide
        | Some op when starts_with op "LEA" -> incr lea
        | Some op when starts_with op "LDG" ->
            incr ldg ;
            incr run ;
            if contains op "CONSTANT" then incr constant
        | Some op
          when starts_with op "BRA" || starts_with op "EXIT"
               || starts_with op "BAR" ->
            cut ()
        | _ -> ())
    (lines sass) ;
  cut () ;
  {
    registers;
    ldg = !ldg;
    constant = !constant;
    in_flight = !best;
    imad = !imad;
    imad_wide = !imad_wide;
    lea = !lea;
    sass_lines = List.length (lines sass);
  }

let registers_of info =
  match
    List.find_opt
      (fun l -> contains l "Used" && contains l "registers")
      (lines info)
  with
  | Some l -> String.trim l
  | None -> "?"

let compile label ptx =
  let base = Filename.temp_file "ptx_sass_" "" in
  let ptx_f = base ^ ".ptx" and cubin = base ^ ".cubin" in
  Out_channel.with_open_bin ptx_f (fun oc -> output_string oc ptx) ;
  let info =
    run_capture
      (Printf.sprintf
         "ptxas -arch=sm_75 -v -o %s %s"
         (Filename.quote cubin)
         (Filename.quote ptx_f))
  in
  if contains info "error" || not (Sys.file_exists cubin) then begin
    Printf.printf "ptxas rejected %s:\n%s\n" label info ;
    exit 1
  end ;
  let sass =
    run_capture (Printf.sprintf "nvdisasm -c %s" (Filename.quote cubin))
  in
  List.iter (fun f -> try Sys.remove f with Sys_error _ -> ()) [ptx_f; cubin] ;
  (analyse_sass sass (registers_of info), sass)

(* Opcode histogram of a SASS text: where the instructions of a kernel go. *)
let print_histogram label sass =
  let table = Hashtbl.create 64 in
  List.iter
    (fun line ->
      match opcode line with
      | Some op ->
          let op =
            match String.index_opt op '.' with
            | Some i -> String.sub op 0 i
            | None -> op
          in
          Hashtbl.replace
            table
            op
            (1 + Option.value ~default:0 (Hashtbl.find_opt table op))
      | None -> ())
    (lines sass) ;
  let rows =
    List.sort
      (fun (_, a) (_, b) -> compare b a)
      (List.of_seq (Hashtbl.to_seq table))
  in
  Printf.printf "  histogram %s:" label ;
  List.iter (fun (op, n) -> if n >= 4 then Printf.printf " %s=%d" op n) rows ;
  print_newline ()

(* The full SASS is printed for the one kernel under study. *)
let dump_sass name =
  contains name "fused"

let print_row label m =
  Printf.printf
    "  %-34s LDG %3d  CONSTANT %3d  in flight %3d  IMAD %4d (WIDE %3d)  LEA \
     %4d  SASS lines %5d  %s\n\
     %!"
    label
    m.ldg
    m.constant
    m.in_flight
    m.imad
    m.imad_wide
    m.lea
    m.sass_lines
    m.registers

let failures = ref []

let expect cond what = if not cond then failures := what :: !failures

let () =
  if not (on_path "ptxas" && on_path "nvdisasm") then
    print_endline "SKIP: needs ptxas and nvdisasm"
  else begin
    let nvrtc = Sarek_cuda.Cuda_nvrtc.is_available () in
    if not nvrtc then print_endline "(libnvrtc absent: no NVRTC column)" ;
    List.iter
      (fun (name, kernel, streaming, all_constant) ->
        let ir = kernel in
        let spoc =
          Sarek_codegen.Sarek_ir_ptx.generate
            ~sm_target:"sm_75"
            ~block:(256, 1, 1)
            ir
        in
        Printf.printf "######## %s\n" name ;
        let m_spoc, sass_spoc = compile (name ^ " SPOC PTX") spoc in
        print_row "SPOC PTX" m_spoc ;
        if nvrtc then begin
          let cuda =
            Sarek_codegen.Sarek_ir_cuda.generate_with_types
              ~block:(256, 1, 1)
              ~types:[]
              ir
          in
          let ptx =
            Sarek_cuda.Cuda_nvrtc.compile_to_ptx
              ~name:"sarek_kern"
              ~arch:"compute_75"
              cuda
          in
          let m_nvrtc, sass_nvrtc = compile (name ^ " NVRTC PTX") ptx in
          print_row "NVRTC PTX of the CUDA-C" m_nvrtc ;
          if dump_sass name then begin
            print_histogram "SPOC" sass_spoc ;
            print_histogram "NVRTC" sass_nvrtc ;
            Printf.printf
              "==== SPOC PTX\n%s\n==== SPOC SASS\n%s\n==== NVRTC SASS\n%s\n"
              spoc
              sass_spoc
              sass_nvrtc
          end
        end ;
        if streaming then
          expect
            (m_spoc.in_flight >= min_in_flight)
            (Printf.sprintf
               "%s: %d loads in flight, expected at least %d"
               name
               m_spoc.in_flight
               min_in_flight) ;
        if all_constant then
          expect
            (m_spoc.ldg > 0 && m_spoc.constant = m_spoc.ldg)
            (Printf.sprintf
               "%s: %d of %d loads are LDG.E.CONSTANT"
               name
               m_spoc.constant
               m_spoc.ldg))
      [
        ("dot (runtime-trip while, 2 loads)", ir_of dot, true, true);
        ("quantised matvec (6 loads)", ir_of quantised, true, true);
        ("for loop with a runtime bound", ir_of for_dot, true, true);
        ("staging through shared memory", ir_of staging, false, false);
        ( "staged matvec, horizontally fused",
          Staged_kernels.fused (),
          false,
          false );
      ] ;
    match !failures with
    | [] -> ()
    | l ->
        List.iter (Printf.printf "FAIL: %s\n") (List.rev l) ;
        exit 1
  end
