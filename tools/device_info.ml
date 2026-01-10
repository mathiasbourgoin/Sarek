(* Device Info & Test Utility - Multi-backend device information *)

module Device = Spoc_core.Device

let print_separator () = Printf.printf "%s\n" (String.make 70 '=')

let format_memory bytes =
  let gb = Int64.to_float bytes /. (1024. *. 1024. *. 1024.) in
  if gb >= 1.0 then Printf.sprintf "%.2f GB" gb
  else
    let mb = Int64.to_float bytes /. (1024. *. 1024.) in
    Printf.sprintf "%.0f MB" mb

let print_device_info dev =
  let cap = dev.Device.capabilities in
  Printf.printf "[%d] %s\n" dev.Device.id dev.Device.name ;
  Printf.printf "    Backend: %s\n" dev.Device.framework ;

  (* Memory info *)
  Printf.printf "    Memory: %s\n" (format_memory cap.total_global_mem) ;

  (* Compute capability (mainly for CUDA) *)
  let major, minor = cap.compute_capability in
  if major > 0 then Printf.printf "    Compute Capability: %d.%d\n" major minor ;

  (* Threading info *)
  if not cap.is_cpu then begin
    Printf.printf "    Max Threads/Block: %d\n" cap.max_threads_per_block ;
    Printf.printf "    Compute Units: %d\n" cap.multiprocessor_count ;
    Printf.printf "    Warp/Wavefront Size: %d\n" cap.warp_size
  end ;

  (* Feature support *)
  let features = ref [] in
  if cap.supports_fp64 then features := "FP64" :: !features ;
  if cap.supports_atomics then features := "Atomics" :: !features ;
  if cap.is_cpu then features := "CPU (zero-copy)" :: !features ;
  if !features <> [] then
    Printf.printf "    Features: %s\n" (String.concat ", " !features)

let () =
  (* Initialize all available backends *)
  Backend_init.init () ;

  print_separator () ;
  Printf.printf "SAREK Device Information Utility\n" ;
  Printf.printf "Multi-Backend GPU/CPU Device Diagnostic Tool\n" ;
  print_separator () ;

  try
    let _ = Device.init () in
    let count = Device.count () in

    if count = 0 then begin
      Printf.printf "\n⚠ No devices found\n" ;
      Printf.printf "Ensure you have:\n" ;
      Printf.printf "  - CUDA backend: NVIDIA driver + CUDA toolkit 12.9+\n" ;
      Printf.printf "  - OpenCL backend: OpenCL ICD + drivers\n" ;
      Printf.printf "  - Vulkan backend: Vulkan SDK + drivers\n" ;
      Printf.printf "  - Metal backend: macOS 10.13+\n" ;
      Printf.printf "  - Native/Interpreter: Always available (CPU only)\n" ;
      print_separator () ;
      exit 1
    end ;

    Printf.printf
      "\nFound %d device%s:\n\n"
      count
      (if count > 1 then "s" else "") ;

    (* List all devices *)
    for i = 0 to count - 1 do
      match Device.get i with
      | None -> ()
      | Some dev ->
          print_device_info dev ;
          Printf.printf "\n"
    done ;

    print_separator () ;
    Printf.printf "Usage:\n" ;
    Printf.printf "  • Device.get N returns device N (0-%d)\n" (count - 1) ;
    Printf.printf
      "  • Device.best () selects best device (CUDA > OpenCL > Native)\n" ;
    Printf.printf "  • Device.by_framework \"CUDA\" filters by backend\n" ;
    Printf.printf "  • Run 'make benchmarks' to test all devices\n" ;
    Printf.printf "  • Run 'dune runtest' to verify functionality\n" ;
    print_separator () ;
    exit 0
  with e ->
    Printf.printf "\n❌ Error: %s\n" (Printexc.to_string e) ;
    print_separator () ;
    exit 1
