(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Convert benchmark JSON results to CSV format

    Usage: to_csv.exe <input.json> [output.csv]

    Converts benchmark JSON (single or aggregated) to CSV for spreadsheet
    analysis. *)

open Yojson.Basic.Util

(** Extract results from either single benchmark or aggregated format *)
let extract_results json =
  (* Check if this is an aggregated file with "aggregated_at" field *)
  match json |> member "aggregated_at" with
  | `Null ->
      (* Single benchmark file *)
      [json]
  | _ ->
      (* Aggregated file - extract list of benchmark results *)
      json |> member "results" |> to_list

(** Convert a single benchmark result to CSV rows *)
let result_to_csv_rows json =
  let benchmark = json |> member "benchmark" in
  let system = json |> member "system" in
  let results =
    try json |> member "results" |> to_list
    with Type_error _ ->
      Printf.eprintf "Warning: Invalid or missing results field\n" ;
      []
  in

  let benchmark_name = benchmark |> member "name" |> to_string in
  let size = benchmark |> member "parameters" |> member "size" |> to_int in
  let iterations =
    benchmark |> member "parameters" |> member "iterations" |> to_int
  in
  let hostname = system |> member "hostname" |> to_string in
  let timestamp = benchmark |> member "timestamp" |> to_string in

  List.map
    (fun result ->
      let device_name = result |> member "device_name" |> to_string in
      let framework = result |> member "framework" |> to_string in
      let mean_ms = result |> member "mean_ms" |> to_float in
      let stddev_ms = result |> member "stddev_ms" |> to_float in
      let median_ms = result |> member "median_ms" |> to_float in
      let min_ms = result |> member "min_ms" |> to_float in
      let max_ms = result |> member "max_ms" |> to_float in
      let verified = result |> member "verified" |> to_bool in
      let throughput_gflops =
        try Some (result |> member "throughput_gflops" |> to_float)
        with _ -> None
      in

      Printf.sprintf
        "%s,%s,%d,%d,%s,%s,%s,%f,%f,%f,%f,%f,%b,%s"
        hostname
        timestamp
        size
        iterations
        benchmark_name
        device_name
        framework
        mean_ms
        stddev_ms
        median_ms
        min_ms
        max_ms
        verified
        (match throughput_gflops with
        | Some gflops -> Printf.sprintf "%.6f" gflops
        | None -> ""))
    results

(** Convert JSON to CSV *)
let json_to_csv json output_file =
  let out_channel = open_out output_file in

  (* Write header *)
  Printf.fprintf
    out_channel
    "hostname,timestamp,size,iterations,benchmark,device,framework,mean_ms,stddev_ms,median_ms,min_ms,max_ms,verified,throughput_gflops\n" ;

  (* Extract and write rows *)
  let results = extract_results json in
  List.iter
    (fun result ->
      let rows = result_to_csv_rows result in
      List.iter (fun row -> Printf.fprintf out_channel "%s\n" row) rows)
    results ;

  close_out out_channel ;
  Printf.printf "Written CSV to: %s\n" output_file

(** Main entry point *)
let () =
  if Array.length Sys.argv < 2 then begin
    Printf.eprintf "Usage: %s <input.json> [output.csv]\n" Sys.argv.(0) ;
    Printf.eprintf "\nConverts benchmark JSON to CSV format.\n" ;
    exit 1
  end ;

  let input_file = Sys.argv.(1) in
  let output_file =
    if Array.length Sys.argv >= 3 then Sys.argv.(2)
    else if
      (* Replace .json extension with .csv *)
      Filename.check_suffix input_file ".json"
    then Filename.chop_suffix input_file ".json" ^ ".csv"
    else input_file ^ ".csv"
  in

  try
    let json = Yojson.Basic.from_file input_file in
    json_to_csv json output_file
  with
  | Sys_error msg ->
      Printf.eprintf "Error: %s\n" msg ;
      exit 1
  | Yojson.Json_error msg ->
      Printf.eprintf "JSON parse error: %s\n" msg ;
      exit 1
