(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Output formatting for benchmark results *)

type benchmark_params = {
  name : string;
  size : int;
  block_size : int;
  iterations : int;
  warmup : int;
}

type device_result = {
  device_id : int;
  device_name : string;
  framework : string;
  iterations : float array;
  mean_ms : float;
  stddev_ms : float;
  median_ms : float;
  min_ms : float;
  max_ms : float;
  throughput : float option; (* GFLOPS, GB/s, or other metric *)
  verified : bool option;
}

type benchmark_result = {
  params : benchmark_params;
  timestamp : string;
  git_commit : string option;
  system : System_info.system_info;
  results : device_result list;
}

(** Convert device result to JSON *)
let device_result_to_json (r : device_result) =
  let fields =
    [
      ("device_id", `Int r.device_id);
      ("device_name", `String r.device_name);
      ("framework", `String r.framework);
      ( "iterations",
        `List (Array.to_list r.iterations |> List.map (fun x -> `Float x)) );
      ("mean_ms", `Float r.mean_ms);
      ("stddev_ms", `Float r.stddev_ms);
      ("median_ms", `Float r.median_ms);
      ("min_ms", `Float r.min_ms);
      ("max_ms", `Float r.max_ms);
    ]
  in
  let fields =
    match r.throughput with
    | Some t -> ("throughput_gflops", `Float t) :: fields
    | None -> fields
  in
  let fields =
    match r.verified with
    | Some v -> ("verified", `Bool v) :: fields
    | None -> fields
  in
  `Assoc fields

(** Convert full benchmark result to JSON *)
let to_json (result : benchmark_result) =
  `Assoc
    [
      ( "benchmark",
        `Assoc
          [
            ("name", `String result.params.name);
            ("timestamp", `String result.timestamp);
            ( "git_commit",
              match result.git_commit with Some c -> `String c | None -> `Null
            );
            ( "parameters",
              `Assoc
                [
                  ("size", `Int result.params.size);
                  ("block_size", `Int result.params.block_size);
                  ("iterations", `Int result.params.iterations);
                  ("warmup", `Int result.params.warmup);
                ] );
          ] );
      ("system", System_info.to_json result.system);
      ("results", `List (List.map device_result_to_json result.results));
    ]

(** Write JSON to file *)
let write_json filename result =
  let json = to_json result in
  let json_str = Yojson.Basic.pretty_to_string json in
  let oc = open_out filename in
  output_string oc json_str ;
  close_out oc

(** Write CSV header *)
let write_csv_header oc =
  output_string
    oc
    "benchmark,timestamp,hostname,device_id,device_name,framework,size,block_size,iterations_count,mean_ms,stddev_ms,median_ms,min_ms,max_ms,throughput_gflops,verified\n"

(** Write single CSV row *)
let write_csv_row oc result dev_result =
  let verified_str =
    match dev_result.verified with
    | Some true -> "true"
    | Some false -> "false"
    | None -> ""
  in
  let throughput_str =
    match dev_result.throughput with
    | Some t -> Printf.sprintf "%.4f" t
    | None -> ""
  in
  Printf.fprintf
    oc
    "%s,%s,%s,%d,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s\n"
    result.params.name
    result.timestamp
    result.system.hostname
    dev_result.device_id
    dev_result.device_name
    dev_result.framework
    result.params.size
    result.params.block_size
    (Array.length dev_result.iterations)
    dev_result.mean_ms
    dev_result.stddev_ms
    dev_result.median_ms
    dev_result.min_ms
    dev_result.max_ms
    throughput_str
    verified_str

(** Write CSV file *)
let write_csv filename result =
  let oc = open_out filename in
  write_csv_header oc ;
  List.iter (write_csv_row oc result) result.results ;
  close_out oc

(** Append to existing CSV file *)
let append_csv filename result =
  let file_exists = Sys.file_exists filename in
  let oc = open_out_gen [Open_wronly; Open_append; Open_creat] 0o644 filename in
  if not file_exists then write_csv_header oc ;
  List.iter (write_csv_row oc result) result.results ;
  close_out oc

(** Generate output filename *)
let make_filename ~output_dir ~benchmark_name ~size =
  let timestamp = Common.timestamp_filename () in
  let hostname = System_info.get_hostname () in
  Printf.sprintf
    "%s/%s_%s_%d_%s.json"
    output_dir
    hostname
    benchmark_name
    size
    timestamp
