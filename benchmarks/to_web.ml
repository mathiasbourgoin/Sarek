(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Convert benchmark results to web-ready JSON format

    Usage: to_web.exe <output.json> <input1.json> <input2.json> ...

    Converts benchmark JSON files to format expected by gh-pages viewer. *)

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

(** Convert to web format *)
let to_web_format output_file input_files =
  Printf.printf
    "Converting %d files to web format...\n"
    (List.length input_files) ;

  let results = ref [] in

  List.iter
    (fun filename ->
      Printf.printf "  Reading: %s\n" filename ;
      match read_json filename with
      | Some json -> results := json :: !results
      | None -> ())
    input_files ;

  if !results = [] then begin
    Printf.eprintf "No valid results to convert\n" ;
    exit 1
  end ;

  (* Create web format structure *)
  let web_format =
    `Assoc
      [
        ("results", `List (List.rev !results));
        ( "updated_at",
          `String
            (let tm = Unix.gmtime (Unix.time ()) in
             Printf.sprintf
               "%04d-%02d-%02dT%02d:%02d:%02dZ"
               (1900 + tm.Unix.tm_year)
               (1 + tm.Unix.tm_mon)
               tm.Unix.tm_mday
               tm.Unix.tm_hour
               tm.Unix.tm_min
               tm.Unix.tm_sec) );
      ]
  in

  (* Write output *)
  Printf.printf "Writing web format to: %s\n" output_file ;
  Yojson.Basic.to_file output_file web_format ;
  Printf.printf "Done! Converted %d benchmark results\n" (List.length !results)

(** Main entry point *)
let () =
  if Array.length Sys.argv < 3 then begin
    Printf.eprintf
      "Usage: %s <output.json> <input1.json> <input2.json> ...\n"
      Sys.argv.(0) ;
    Printf.eprintf
      "\nConverts benchmark JSON files to web-ready format for gh-pages.\n" ;
    Printf.eprintf "\nExample:\n" ;
    Printf.eprintf
      "  %s gh-pages/benchmarks/data/latest.json results/*.json\n"
      Sys.argv.(0) ;
    exit 1
  end ;

  let output_file = Sys.argv.(1) in
  let input_files =
    Array.to_list (Array.sub Sys.argv 2 (Array.length Sys.argv - 2))
  in

  to_web_format output_file input_files
