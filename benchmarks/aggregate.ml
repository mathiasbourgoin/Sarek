(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Aggregate benchmark results from multiple machines/runs

    Usage: aggregate.exe <output.json> <input1.json> <input2.json> ...

    Combines multiple benchmark JSON files into a single aggregated file for
    analysis and plotting. *)

(** Read and parse a JSON file *)
let read_json filename =
  try
    let json = Yojson.Basic.from_file filename in
    Some json
  with
  | Sys_error msg ->
      Printf.eprintf "Error reading %s: %s\n" filename msg ;
      None
  | Yojson.Json_error msg ->
      Printf.eprintf "Error parsing %s: %s\n" filename msg ;
      None

(** Aggregate multiple JSON files *)
let aggregate_files output_file input_files =
  Printf.printf "Aggregating %d files...\n" (List.length input_files) ;

  let results = ref [] in

  List.iter
    (fun filename ->
      Printf.printf "  Reading: %s\n" filename ;
      match read_json filename with
      | Some json -> results := json :: !results
      | None -> ())
    input_files ;

  if !results = [] then begin
    Printf.eprintf "No valid results to aggregate\n" ;
    exit 1
  end ;

  (* Get current timestamp *)
  let tm = Unix.gmtime (Unix.time ()) in
  let timestamp =
    Printf.sprintf
      "%04d-%02d-%02dT%02d:%02d:%02dZ"
      (1900 + tm.Unix.tm_year)
      (1 + tm.Unix.tm_mon)
      tm.Unix.tm_mday
      tm.Unix.tm_hour
      tm.Unix.tm_min
      tm.Unix.tm_sec
  in

  (* Create aggregated structure *)
  let aggregated =
    `Assoc
      [
        ("aggregated_at", `String timestamp);
        ("num_results", `Int (List.length !results));
        ("results", `List (List.rev !results));
      ]
  in

  (* Write output *)
  Printf.printf "Writing aggregated results to: %s\n" output_file ;
  Yojson.Basic.to_file output_file aggregated ;
  Printf.printf "Done! Aggregated %d benchmark results\n" (List.length !results)

(** Main entry point *)
let () =
  if Array.length Sys.argv < 3 then begin
    Printf.eprintf
      "Usage: %s <output.json> <input1.json> <input2.json> ...\n"
      Sys.argv.(0) ;
    Printf.eprintf
      "\nAggregates multiple benchmark JSON files into a single file.\n" ;
    exit 1
  end ;

  let output_file = Sys.argv.(1) in
  let input_files =
    Array.to_list (Array.sub Sys.argv 2 (Array.length Sys.argv - 2))
  in

  aggregate_files output_file input_files
