(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Copies from host memory registered with Transfer.register_host, by how the
 * memory was prepared (specs/transfer-pitfalls.md, F-2):
 *
 *   touched    written before registering
 *   untouched  never touched before registering
 *   reg+write  never touched, registered, then written (a downstream RAM tier)
 *
 * For each: synchronous copies of random 5.5 MiB records into a device-only
 * vector, first pass and an identical repeat pass, in GB/s; plus VmRSS growth
 * caused by registering and VmPin. Then what a copy spanning two
 * registrations does. INFORMATION ONLY (alias e2e-hip-bench).
 *
 * Slow passes were seen at 8 GiB on tartarus and not in a 4 GiB run, so 8 is
 * the default; it needs about 9 GB of free RAM.
 *
 *   SPOC_BENCH_GIB=4 dune exec .../bench_hip_register_host.exe
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_hip.Hip_plugin.register ()

let gib =
  match Option.bind (Sys.getenv_opt "SPOC_BENCH_GIB") int_of_string_opt with
  | Some g -> g
  | None -> 8

let rec_bytes = (5 * 1024 * 1024) + (512 * 1024)

let copies_per_pass = 400

let page = 4096

(* Page-aligned, so the registration starts on a page of its own. *)
let aligned n =
  let raw = Bigarray.Array1.create Bigarray.char Bigarray.c_layout (n + page) in
  let addr =
    Ctypes.(raw_address_of_ptr (to_voidp (bigarray_start array1 raw)))
  in
  let skip =
    (page - Nativeint.to_int (Nativeint.rem addr (Nativeint.of_int page)))
    mod page
  in
  Bigarray.Array1.sub raw skip n

let status_kb field =
  let ic = open_in "/proc/self/status" in
  let r = ref (-1) in
  (try
     while true do
       let l = input_line ic in
       match String.split_on_char ':' l with
       | [k; v] when k = field ->
           r :=
             int_of_string (List.hd (String.split_on_char ' ' (String.trim v)))
       | _ -> ()
     done
   with End_of_file -> ()) ;
  close_in ic ;
  !r

let pass dst pool =
  let slots = Bigarray.Array1.dim pool / rec_bytes in
  Random.init 42 ;
  let t0 = Unix.gettimeofday () in
  for _ = 1 to copies_per_pass do
    Transfer.copy_bytes_to_device
      (Bigarray.Array1.sub pool (Random.int slots * rec_bytes) rec_bytes)
      ~dst
      ~dst_byte_offset:0
  done ;
  float_of_int (copies_per_pass * rec_bytes)
  /. (Unix.gettimeofday () -. t0)
  /. 1e9

let scenario dev dst label =
  let bytes = gib lsl 30 in
  let pool = aligned bytes in
  if label = "touched" then Bigarray.Array1.fill pool '\001' ;
  let rss0 = status_kb "VmRSS" in
  let t0 = Unix.gettimeofday () in
  Transfer.register_host dev pool ;
  let t_reg = Unix.gettimeofday () -. t0 in
  (* "+write": the RAM-tier pattern, filled only after registration. *)
  if label = "reg+write" then Bigarray.Array1.fill pool '\002' ;
  let rss1 = status_kb "VmRSS" in
  let first = pass dst pool in
  let repeat = pass dst pool in
  Printf.printf
    "  %-10s register %5.2f s (VmRSS +%5.2f GB, VmPin %d kB)   first pass \
     %6.2f GB/s   repeat %6.2f GB/s\n\
     %!"
    label
    t_reg
    (float_of_int (rss1 - rss0) /. 1e6)
    (status_kb "VmPin")
    first
    repeat ;
  Transfer.unregister_host pool ;
  ignore (Sys.opaque_identity pool)

let spanning dev dst =
  let pool = aligned (2 * rec_bytes) in
  Bigarray.Array1.fill pool '\001' ;
  let lo = Bigarray.Array1.sub pool 0 rec_bytes
  and hi = Bigarray.Array1.sub pool rec_bytes rec_bytes in
  Transfer.register_host dev lo ;
  Transfer.register_host dev hi ;
  (match
     Transfer.copy_bytes_to_device
       (Bigarray.Array1.sub pool (rec_bytes / 2) rec_bytes)
       ~dst
       ~dst_byte_offset:0
   with
  | () -> print_endline "  copy spanning two registrations: accepted"
  | exception e ->
      Printf.printf
        "  copy spanning two registrations: %s\n"
        (Printexc.to_string e)) ;
  Transfer.unregister_host lo ;
  Transfer.unregister_host hi

let () =
  let devices = Device.init () in
  match
    Array.to_list devices |> List.find_opt (fun d -> d.Device.framework = "HIP")
  with
  | None ->
      print_endline "bench_hip_register_host: [SKIP] no HIP device present"
  | Some dev ->
      Printf.printf
        "%s: %d GiB registrations, %d synchronous %.1f MiB copies per pass\n%!"
        dev.Device.name
        gib
        copies_per_pass
        (float_of_int rec_bytes /. 1048576.) ;
      let dst = Transfer.create_device_only dev Vector.char rec_bytes in
      (* Warm the copy path from ordinary memory first. *)
      let warm = aligned rec_bytes in
      Bigarray.Array1.fill warm 'w' ;
      for _ = 1 to 20 do
        Transfer.copy_bytes_to_device warm ~dst ~dst_byte_offset:0
      done ;
      List.iter
        (fun label ->
          scenario dev dst label ;
          Gc.full_major ())
        ["touched"; "untouched"; "reg+write"] ;
      spanning dev dst ;
      Transfer.free_all_buffers dst
