(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Metal kernel-signature address-space check — layer 1 of the Metal gate
    (#139).

    WHY A TEXT LAYER AT ALL, when {!Metal_compile} exists: the Metal compiler
    ships only with Xcode, so on every Linux machine in this project — which is
    where the code is written — {!Metal_compile} is unavailable and skips. That
    is exactly how two committed Metal goldens ([record_kernel],
    [variant_kernel]) sat in the tree emitting source that has never once
    compiled: the byte-exact goldens pinned the bytes, the sweep for the other
    four backends had no Metal arm, and nothing anywhere read the bytes for
    meaning.

    This layer needs no toolchain and therefore runs on the machine that
    introduces the defect. It reads the kernel signature only, and it checks one
    property that Metal enforces and C does not:

    {b every pointer in a kernel parameter list carries an explicit address
       space, and no parameter is a reference to a pointer.}

    MSL 3.2 §4.2: Metal has no default address space; a pointer or reference
    argument to a kernel must be declared in [device], [constant] or
    [threadgroup]. §4.2.3 further requires that in a reference-to-pointer
    parameter the POINTEE also carry one, so [constant Point2* &pts] is rejected
    outright — the form the [DParam (v, None)] arm used to emit for every
    vec-typed parameter.

    Blind spots, stated so a green run is not read as "our Metal is valid": this
    layer sees the signature and nothing else. Body-level type errors, wrong
    intrinsic names, bad struct layout and everything else are
    {!Metal_compile}'s job, and on Linux nothing covers them. *)

type offence = {param : string; reason : string}

let describe o = Printf.sprintf "  %s\n      -> %s" o.param o.reason

let is_space c = c = ' ' || c = '\t' || c = '\n' || c = '\r'

let trim = String.trim

(* Address spaces Metal accepts on a kernel parameter (MSL 3.2 §4.2). [thread]
   is legal in the language but cannot appear on a kernel entry point's buffer
   argument; it is listed so the diagnostic below stays about the missing
   qualifier rather than about an unfamiliar keyword. *)
let address_spaces = ["device"; "constant"; "threadgroup"; "thread"]

let starts_with_space_kw s =
  List.exists
    (fun kw ->
      let n = String.length kw in
      String.length s > n && String.sub s 0 n = kw && is_space s.[n])
    address_spaces

(** The parameter list of the first [kernel void <name>( ... )] in [src], as
    written, or [None] if there is none. Scans for the matching close paren so a
    nested [[[attribute(0)]]] cannot end the list early. *)
let kernel_signature (src : string) : string option =
  let n = String.length src in
  let marker = "kernel void " in
  let m = String.length marker in
  let rec find_marker i =
    if i + m > n then None
    else if String.sub src i m = marker then Some (i + m)
    else find_marker (i + 1)
  in
  match find_marker 0 with
  | None -> None
  | Some after -> (
      let rec find_open i =
        if i >= n then None
        else if src.[i] = '(' then Some i
        else find_open (i + 1)
      in
      match find_open after with
      | None -> None
      | Some op -> (
          let rec close i depth =
            if i >= n then None
            else
              match src.[i] with
              | '(' -> close (i + 1) (depth + 1)
              | ')' -> if depth = 1 then Some i else close (i + 1) (depth - 1)
              | _ -> close (i + 1) depth
          in
          match close (op + 1) 1 with
          | None -> None
          | Some cl -> Some (String.sub src (op + 1) (cl - op - 1))))

(* Split a parameter list on top-level commas. [[[buffer(0)]]] contains no
   comma, but a future attribute might, so depth is tracked rather than
   assumed. *)
let split_params (s : string) : string list =
  let out = ref [] in
  let buf = Buffer.create 64 in
  let depth = ref 0 in
  String.iter
    (fun c ->
      match c with
      | '(' | '[' ->
          incr depth ;
          Buffer.add_char buf c
      | ')' | ']' ->
          decr depth ;
          Buffer.add_char buf c
      | ',' when !depth = 0 ->
          out := Buffer.contents buf :: !out ;
          Buffer.clear buf
      | c -> Buffer.add_char buf c)
    s ;
  if trim (Buffer.contents buf) <> "" then out := Buffer.contents buf :: !out ;
  (* [!out] is accumulated back-to-front, and [List.rev_map] already reverses
     while mapping — so it alone restores source order. The [List.rev] that used
     to follow put the list back into reverse, which made the reported offences
     read bottom-up. Detection was unaffected (each parameter is inspected
     independently, so no permutation can hide one), but a diagnostic that lists
     parameters in an order the reader cannot find in the source costs exactly
     the time this gate is supposed to save. Order is pinned by a test. *)
  List.rev_map trim !out

let contains_char s c = String.exists (fun x -> x = c) s

(** Every parameter of the kernel in [src] that Metal's address-space rules
    reject. Empty list = this layer is satisfied. *)
let offences (src : string) : offence list =
  match kernel_signature src with
  | None -> []
  | Some params ->
      List.filter_map
        (fun p ->
          let p = trim p in
          if p = "" then None
          else if contains_char p '*' && contains_char p '&' then
            Some
              {
                param = p;
                reason =
                  "reference to a pointer whose POINTEE has no address space. \
                   This is the #139 shape — `constant T* &v` — and Apple clang \
                   17 answers it with: \"invalid address space qualification \
                   for buffer pointee type ... valid address space \
                   qualifications are device and constant\". A Sarek vec is \
                   written through, so the answer is `device T* v`.";
              }
          else if contains_char p '*' && not (starts_with_space_kw p) then
            Some
              {
                param = p;
                reason =
                  "pointer parameter with no address space qualifier. Metal \
                   has no default address space (MSL 3.2 §4.2); say `device`, \
                   `constant` or `threadgroup`.";
              }
          else None)
        (split_params params)
