(* Deduplication tool for benchmark results
   
   Usage: dune exec benchmarks/deduplicate_results.exe -- [--dry-run] [--keep-latest]
   
   This script identifies and removes duplicate benchmark results based on:
   - Same hostname
   - Same benchmark name
   - Same size/parameters
   - Same device name and backend
   
   By default, keeps the oldest result (first submitted).
   With --keep-latest, keeps the newest result (most recent run).
   With --dry-run, only reports duplicates without deleting files.
*)

open Yojson.Basic.Util

type result_key = {
  hostname : string;
  benchmark_name : string;
  size : string;
  device_name : string;
  backend : string;
}

type result_file = {path : string; timestamp : string; key : result_key}

let parse_json_file path =
  try
    let json = Yojson.Basic.from_file path in
    let benchmark = json |> member "benchmark" in
    let system = json |> member "system" in
    let results = json |> member "results" |> to_list in

    (* Get first device from results *)
    let device_name, backend =
      match results with
      | [] -> ("unknown", "unknown")
      | first :: _ ->
          let device_name = first |> member "device_name" |> to_string in
          (* Try 'backend' first, fall back to 'framework' for compatibility *)
          let backend =
            try first |> member "backend" |> to_string
            with _ -> first |> member "framework" |> to_string
          in
          (device_name, backend)
    in

    Some
      {
        path;
        timestamp = benchmark |> member "timestamp" |> to_string;
        key =
          {
            hostname = system |> member "hostname" |> to_string;
            benchmark_name = benchmark |> member "name" |> to_string;
            size =
              benchmark |> member "parameters" |> member "size" |> to_int
              |> string_of_int;
            device_name;
            backend;
          };
      }
  with
  | Yojson.Json_error msg ->
      Printf.eprintf "Warning: Failed to parse %s: %s\n" path msg ;
      None
  | Type_error (msg, _) ->
      Printf.eprintf "Warning: Invalid structure in %s: %s\n" path msg ;
      None
  | Sys_error msg ->
      Printf.eprintf "Warning: Cannot read %s: %s\n" path msg ;
      None

let key_to_string key =
  Printf.sprintf
    "%s/%s/%s/%s/%s"
    key.hostname
    key.benchmark_name
    key.size
    key.device_name
    key.backend

let find_duplicates files =
  (* Group files by their key *)
  let table = Hashtbl.create 100 in
  List.iter
    (fun file ->
      let key_str = key_to_string file.key in
      let existing = try Hashtbl.find table key_str with Not_found -> [] in
      Hashtbl.replace table key_str (file :: existing))
    files ;

  (* Find groups with duplicates *)
  Hashtbl.fold
    (fun _key files acc ->
      if List.length files > 1 then
        (files |> List.sort (fun a b -> String.compare a.timestamp b.timestamp))
        :: acc
      else acc)
    table
    []

let print_help () =
  print_endline "Deduplication tool for benchmark results" ;
  print_endline "" ;
  print_endline
    "Usage: dune exec benchmarks/deduplicate_results.exe -- [OPTIONS]" ;
  print_endline "" ;
  print_endline "Options:" ;
  print_endline "  --dry-run       Report duplicates without deleting files" ;
  print_endline "  --keep-latest   Keep newest result (default: keep oldest)" ;
  print_endline "  -h, --help      Show this help message" ;
  print_endline "" ;
  print_endline
    "This script identifies and removes duplicate benchmark results based on:" ;
  print_endline "  - Same hostname" ;
  print_endline "  - Same benchmark name" ;
  print_endline "  - Same size/parameters" ;
  print_endline "  - Same device name and backend" ;
  print_endline "" ;
  print_endline "By default, keeps the oldest result (first submitted)." ;
  print_endline "With --keep-latest, keeps the newest result (most recent run)." ;
  exit 0

let main () =
  let results_dir = "benchmarks/results" in
  let dry_run = ref false in
  let keep_latest = ref false in

  (* Parse command line arguments *)
  let args = Array.to_list Sys.argv |> List.tl in
  List.iter
    (function
      | "--dry-run" -> dry_run := true
      | "--keep-latest" -> keep_latest := true
      | "--help" | "-h" -> print_help ()
      | arg ->
          Printf.eprintf "Unknown argument: %s\n" arg ;
          Printf.eprintf "Use --help for usage information.\n" ;
          exit 1)
    args ;

  (* Find all JSON files *)
  let json_files =
    Sys.readdir results_dir |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".json")
    |> List.map (fun f -> Filename.concat results_dir f)
  in

  Printf.printf
    "Found %d result files in %s\n"
    (List.length json_files)
    results_dir ;

  (* Parse all files *)
  let parsed_files = List.filter_map parse_json_file json_files in

  Printf.printf "Successfully parsed %d files\n" (List.length parsed_files) ;

  (* Find duplicates *)
  let duplicate_groups = find_duplicates parsed_files in

  if duplicate_groups = [] then begin
    Printf.printf "\n✅ No duplicates found!\n" ;
    exit 0
  end ;

  Printf.printf
    "\n⚠️  Found %d groups with duplicates:\n\n"
    (List.length duplicate_groups) ;

  let total_to_remove = ref 0 in
  let total_to_keep = ref 0 in

  List.iter
    (fun group ->
      let key = (List.hd group).key in
      Printf.printf "Duplicate: %s\n" (key_to_string key) ;

      (* Determine which to keep *)
      let sorted = if !keep_latest then List.rev group else group in
      let to_keep = List.hd sorted in
      let to_remove = List.tl sorted in

      Printf.printf
        "  ✓ KEEP: %s (timestamp: %s)\n"
        (Filename.basename to_keep.path)
        to_keep.timestamp ;

      List.iter
        (fun file ->
          Printf.printf
            "  ✗ REMOVE: %s (timestamp: %s)\n"
            (Filename.basename file.path)
            file.timestamp ;
          incr total_to_remove ;

          if not !dry_run then begin
            try
              Sys.remove file.path ;
              Printf.printf "    → Deleted\n"
            with Sys_error msg ->
              Printf.eprintf "    → Failed to delete: %s\n" msg
          end)
        to_remove ;

      incr total_to_keep ;
      Printf.printf "\n")
    duplicate_groups ;

  Printf.printf "Summary:\n" ;
  Printf.printf "  Files to keep: %d\n" !total_to_keep ;
  Printf.printf "  Files to remove: %d\n" !total_to_remove ;

  if !dry_run then
    Printf.printf "\n💡 Run without --dry-run to actually delete files\n"
  else Printf.printf "\n✅ Deduplication complete!\n"

let () = main ()
