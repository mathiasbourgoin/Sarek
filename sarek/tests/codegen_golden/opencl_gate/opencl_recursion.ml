(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Recursion detector for generated OpenCL C — the check no compiler makes.

    OpenCL C forbids recursion (OpenCL C 1.2 §6.9.e, 3.0 §6.9.5). Nothing on
    this project's path enforces it:

    - [clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header -fsyntax-only]
      accepts a recursive device function and exits 0. Verified with clang
      22.1.6 for the default host target and for [--target=spirv64],
      [--target=spirv32] and [--target=amdgcn-amd-amdhsa -mcpu=gfx1100].
    - rusticl/radeonsi (Mesa, RX 7900 XTX) does not diagnose it either. It
      recurses through its own compiler on a [clctxworker] thread until the
      stack is gone and takes the host process down with SIGSEGV — backlog #53,
      root-caused in #127.

    So the compile gate is structurally blind here, and this check exists
    precisely to cover that blind spot. It reads the EMITTED source rather than
    the IR on purpose: the IR-level guarantee lives in the backend
    ([Sarek_ir_opencl.resolve_recursive_helpers]), and a gate that re-ran the
    backend's own reasoning would go green on any bug that made the backend emit
    something other than what it reasoned about.

    The parser is deliberately small — it only has to understand the shape this
    project's own emitter produces (top-level function definitions, one brace
    nesting, no strings, no function pointers). It is not a C parser and does
    not try to be. *)

(** Blank out [//] and [/* */] comments, preserving length and newlines so
    nothing downstream has to care. *)
let strip_comments (s : string) : string =
  let n = String.length s in
  let b = Bytes.of_string s in
  let i = ref 0 in
  while !i < n do
    if !i + 1 < n && s.[!i] = '/' && s.[!i + 1] = '/' then begin
      while !i < n && s.[!i] <> '\n' do
        Bytes.set b !i ' ' ;
        incr i
      done
    end
    else if !i + 1 < n && s.[!i] = '/' && s.[!i + 1] = '*' then begin
      let stop = ref false in
      while (not !stop) && !i < n do
        if !i + 1 < n && s.[!i] = '*' && s.[!i + 1] = '/' then begin
          Bytes.set b !i ' ' ;
          Bytes.set b (!i + 1) ' ' ;
          i := !i + 2 ;
          stop := true
        end
        else begin
          if s.[!i] <> '\n' then Bytes.set b !i ' ' ;
          incr i
        end
      done
    end
    else incr i
  done ;
  Bytes.to_string b

let is_ident_char c =
  (c >= 'a' && c <= 'z')
  || (c >= 'A' && c <= 'Z')
  || (c >= '0' && c <= '9')
  || c = '_'

(** The identifier immediately preceding position [i] (skipping whitespace), or
    [None]. *)
let ident_before (s : string) (i : int) : (string * int) option =
  let j = ref (i - 1) in
  while !j >= 0 && (s.[!j] = ' ' || s.[!j] = '\n' || s.[!j] = '\t') do
    decr j
  done ;
  if !j < 0 || not (is_ident_char s.[!j]) then None
  else begin
    let e = !j in
    while !j >= 0 && is_ident_char s.[!j] do
      decr j
    done ;
    Some (String.sub s (!j + 1) (e - !j), !j + 1)
  end

(** Top-level function definitions as [(name, body)]. A definition is a [{] at
    brace depth 0 whose matching [(] is preceded by an identifier — which
    excludes struct/union initialisers and typedef bodies. *)
let definitions (src : string) : (string * string) list =
  let s = strip_comments src in
  let n = String.length s in
  let depth = ref 0 in
  let out = ref [] in
  let i = ref 0 in
  while !i < n do
    (match s.[!i] with
    | '{' ->
        if !depth = 0 then begin
          (* Walk back over the parameter list to the opening paren. *)
          let j = ref (!i - 1) in
          while !j >= 0 && s.[!j] <> ')' && s.[!j] <> ';' && s.[!j] <> '}' do
            decr j
          done ;
          if !j >= 0 && s.[!j] = ')' then begin
            let pd = ref 0 in
            let k = ref !j in
            let stop = ref false in
            while (not !stop) && !k >= 0 do
              if s.[!k] = ')' then incr pd
              else if s.[!k] = '(' then begin
                decr pd ;
                if !pd = 0 then stop := true
              end ;
              if not !stop then decr k
            done ;
            if !stop then
              match ident_before s !k with
              | Some (name, _) ->
                  (* Body span: from this brace to its match. *)
                  let d = ref 0 in
                  let e = ref !i in
                  let fin = ref (-1) in
                  while !fin < 0 && !e < n do
                    if s.[!e] = '{' then incr d
                    else if s.[!e] = '}' then begin
                      decr d ;
                      if !d = 0 then fin := !e
                    end ;
                    incr e
                  done ;
                  let fin = if !fin < 0 then n - 1 else !fin in
                  out := (name, String.sub s !i (fin - !i + 1)) :: !out
              | None -> ()
          end
        end ;
        incr depth
    | '}' -> decr depth
    | _ -> ()) ;
    incr i
  done ;
  List.rev !out

(** Identifiers applied to an argument list inside [body]. *)
let calls_in (body : string) : string list =
  let n = String.length body in
  let out = ref [] in
  for i = 0 to n - 1 do
    if body.[i] = '(' then
      match ident_before body i with
      | Some (name, _) when not (List.mem name !out) -> out := name :: !out
      | _ -> ()
  done ;
  !out

type cycle = string list

(** Every call cycle among the function definitions in [src]. An empty result is
    the OpenCL C requirement satisfied. *)
let cycles (src : string) : cycle list =
  let defs = definitions src in
  let names = List.map fst defs in
  let edges =
    List.map
      (fun (n, body) ->
        (n, List.filter (fun c -> List.mem c names) (calls_in body)))
      defs
  in
  let succ n = try List.assoc n edges with Not_found -> [] in
  let found = ref [] in
  let rec walk path n =
    if List.mem n path then begin
      (* Normalise to the cycle itself so the same loop is not reported once
         per entry point. *)
      let rec suffix = function
        | x :: tl -> if x = n then x :: tl else suffix tl
        | [] -> []
      in
      (* [suffix] returns the loop with its entry node repeated at the end
         ([f; g; f]); drop the repeat so a self-call is [f], not [f; f]. *)
      let cyc =
        match List.rev (suffix (List.rev (n :: path))) with
        | _ :: _ as l -> List.filteri (fun i _ -> i < List.length l - 1) l
        | [] -> []
      in
      let key = List.sort compare cyc in
      if not (List.exists (fun c -> List.sort compare c = key) !found) then
        found := cyc :: !found
    end
    else List.iter (walk (n :: path)) (succ n)
  in
  List.iter (fun n -> walk [] n) names ;
  !found

let describe (c : cycle) : string =
  match c with
  | [x] -> Printf.sprintf "'%s' calls itself" x
  | _ -> String.concat " -> " (List.map (fun n -> "'" ^ n ^ "'") c)
