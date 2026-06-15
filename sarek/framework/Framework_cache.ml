(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Framework Cache
 *
 * Provides a persistent on-disk cache for compiled kernels (SPIR-V, PTX, etc.)
 * to reduce startup time. Tracks statistics for monitoring cache effectiveness.
 ******************************************************************************)

(** Enable debug logging via environment variable *)
let debug_enabled =
  try
    let dbg = Sys.getenv "SAREK_DEBUG" in
    dbg = "all" || String.split_on_char ',' dbg |> List.mem "device"
  with Not_found -> false

let debugf fmt =
  if debug_enabled then Printf.eprintf ("[Cache] " ^^ fmt ^^ "\n%!")
  else Printf.ifprintf stderr fmt

let warnf fmt = Printf.eprintf ("[Cache] WARNING: " ^^ fmt ^^ "\n%!")

(** Cache statistics *)
type stats = {
  mutable hits : int;  (** Successful cache lookups *)
  mutable misses : int;  (** Failed cache lookups *)
  mutable puts : int;  (** Cache writes *)
  mutable errors : int;  (** I/O errors *)
}

let stats = {hits = 0; misses = 0; puts = 0; errors = 0}

(** Get current cache statistics *)
let get_stats () =
  {
    hits = stats.hits;
    misses = stats.misses;
    puts = stats.puts;
    errors = stats.errors;
  }

(** Reset statistics *)
let reset_stats () =
  stats.hits <- 0 ;
  stats.misses <- 0 ;
  stats.puts <- 0 ;
  stats.errors <- 0

(** Calculate hit rate (0.0 - 1.0) *)
let hit_rate () =
  let total = stats.hits + stats.misses in
  if total = 0 then 0.0 else float_of_int stats.hits /. float_of_int total

(** Print statistics to stdout *)
let print_stats () =
  let total = stats.hits + stats.misses in
  Printf.printf "Cache Statistics:\n" ;
  Printf.printf "  Hits:   %d\n" stats.hits ;
  Printf.printf "  Misses: %d\n" stats.misses ;
  Printf.printf "  Puts:   %d\n" stats.puts ;
  Printf.printf "  Errors: %d\n" stats.errors ;
  if total > 0 then Printf.printf "  Hit rate: %.1f%%\n" (hit_rate () *. 100.0)

let cache_dir_name = "spoc"

(** Get cache directory, creating it if needed *)
let get_cache_dir () =
  try
    let base_dir =
      try Sys.getenv "XDG_CACHE_HOME"
      with Not_found -> (
        try Filename.concat (Sys.getenv "HOME") ".cache"
        with Not_found -> Filename.get_temp_dir_name ())
    in
    let dir = Filename.concat base_dir cache_dir_name in
    if not (Sys.file_exists dir) then begin
      debugf "Creating cache directory: %s" dir ;
      Unix.mkdir dir 0o755
    end ;
    dir
  with e ->
    warnf "Failed to create cache directory: %s" (Printexc.to_string e) ;
    Framework_error.raise_error
      (Cache_error {operation = "get_cache_dir"; reason = Printexc.to_string e})

(** Compute cache key from device, driver, and source *)
let compute_key ~dev_name ~driver_version ~source =
  let ctx = Digest.string (dev_name ^ driver_version ^ source) in
  let key = Digest.to_hex ctx in
  debugf "Cache key: %s (dev=%s, driver=%s)" key dev_name driver_version ;
  key

(** Retrieve cached data by key *)
let get ~key =
  try
    let dir = get_cache_dir () in
    let filename = Filename.concat dir key in
    if Sys.file_exists filename then begin
      try
        let ic = open_in_bin filename in
        let len = in_channel_length ic in
        let data = really_input_string ic len in
        close_in ic ;
        stats.hits <- stats.hits + 1 ;
        debugf "Cache hit: %s (%d bytes)" key len ;
        Some data
      with e ->
        stats.errors <- stats.errors + 1 ;
        warnf "Cache read error for key %s: %s" key (Printexc.to_string e) ;
        None
    end
    else begin
      stats.misses <- stats.misses + 1 ;
      debugf "Cache miss: %s" key ;
      None
    end
  with Framework_error.Framework_error _ as e ->
    stats.errors <- stats.errors + 1 ;
    raise e

(** Store data in cache with given key *)
let put ~key ~data =
  try
    let dir = get_cache_dir () in
    let filename = Filename.concat dir key in
    try
      let oc = open_out_bin filename in
      output_string oc data ;
      close_out oc ;
      stats.puts <- stats.puts + 1 ;
      debugf "Cache put: %s (%d bytes)" key (String.length data)
    with e ->
      stats.errors <- stats.errors + 1 ;
      warnf "Cache write error for key %s: %s" key (Printexc.to_string e)
  with Framework_error.Framework_error _ ->
    stats.errors <- stats.errors + 1 ;
    (* Don't propagate cache errors, just log them *)
    ()
