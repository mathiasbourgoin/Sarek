#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Refuse a diagnostic that names a PPX construct the PPX does not have.
#
# WHY (backlog-193): four PPX errors told the user to "register it with
# [%%ktype]". There has never been a `ktype` extension in this tree. A user
# hitting any of the four was sent to an extension point that does not exist --
# strictly worse than no advice, because it reads as authoritative. Nothing
# checked message text against the PPX's own name table, so the four survived
# every build, every test run and every review.
#
# Two independent checks, because the same lie has two independent mechanisms
# and closing one would not have caught the other:
#
#   1. NAME    -- the construct name itself is not one the PPX handles
#                 (`ktype`). Every name the PPX recognises must appear as a
#                 string literal somewhere under sarek/ppx/, since that is how
#                 ppxlib is told about it (Attribute.declare "sarek.type",
#                 Extension.declare "kernel", has_attr "sarek.module", ...).
#                 So: a construct named in a message but absent from that table
#                 is a name the PPX cannot act on.
#
#   2. RENDER  -- the name is right in the source and wrong on the user's
#                 terminal. `Location.raise_errorf` and `Format.fprintf` are
#                 Format-based: in a format string `@@` prints as `@` and `%%`
#                 prints as `%`. So the literal "[@@sarek.type]" reaches the
#                 user as "[@sarek.type]" -- a DIFFERENT construct (a single-@
#                 attribute cannot sit on a type declaration, so the advice
#                 fails). Sixteen sites were in this state, including the four
#                 above after their `ktype` was corrected. The fix is to double
#                 them ("[@@@@sarek.type]"); this check requires that.
#
# Scope is `git ls-files`, not the filesystem, and excludes sarek/tests/: a
# negative test's whole job is to name a spelling that must be REFUSED (see
# sarek/tests/negative/test_sarek_type_extension.ml, which deliberately writes
# [%%sarek.type]), so the test tree is covered by `make test_negative` asserting
# on the compiler's actual output instead.
#
# Only string literals are read. Comments are stripped first -- a comment is not
# a claim made to a user, and prose about a construct is not advice to use it.
#
# Exit: 0 clean - 1 a message names a construct the user cannot write - 2 the
# check could not run (fails closed).

set -uo pipefail

cd "$(dirname "$0")/.." || { echo "::error::cannot reach repo root" >&2; exit 2; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "::error::not a git work tree -- this check reads 'git ls-files'" >&2
  exit 2
fi

command -v python3 >/dev/null 2>&1 || { echo "::error::python3 required" >&2; exit 2; }

python3 - <<'PY'
import re, subprocess, sys

def tracked(*globs):
    out = subprocess.run(["git", "ls-files", "--", *globs],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print("::error::git ls-files failed", file=sys.stderr)
        sys.exit(2)
    return [p for p in out.stdout.split("\n") if p]

# ---------------------------------------------------------------------------
# Lexing: OCaml comments blanked (offsets preserved), string literals located.
# Hand-rolled because the alternative is a full OCaml parser for a lexical
# question. `(* *)` nests; string escapes are honoured; {|quoted|} strings are
# not used for messages in this tree and are deliberately not special-cased --
# if one appears it is read as ordinary bytes, which can only over-report.
# ---------------------------------------------------------------------------
def lex(src):
    blanked = list(src)
    strings = []          # (start_offset, text)
    i, n, depth = 0, len(src), 0
    while i < n:
        if src.startswith("(*", i):
            depth += 1
            blanked[i] = blanked[i + 1] = " "
            i += 2
            continue
        if depth > 0:
            if src.startswith("*)", i):
                depth -= 1
                blanked[i] = blanked[i + 1] = " "
                i += 2
                continue
            if src[i] != "\n":
                blanked[i] = " "
            i += 1
            continue
        if src[i] == '"':
            start = i
            i += 1
            buf = []
            while i < n and src[i] != '"':
                if src[i] == "\\":
                    # A line continuation ("\<newline><spaces>") is how every
                    # long message in this tree is wrapped; it contributes
                    # nothing to the rendered text.
                    if i + 1 < n and src[i + 1] == "\n":
                        i += 2
                        while i < n and src[i] in " \t":
                            i += 1
                        continue
                    buf.append(src[i]); buf.append(src[i + 1] if i + 1 < n else "")
                    i += 2
                    continue
                buf.append(src[i])
                i += 1
            i += 1
            strings.append((start, "".join(buf)))
            continue
        i += 1
    if depth != 0:
        return None, None
    return "".join(blanked), strings

def line_of(src, off):
    return src.count("\n", 0, off) + 1

# ---------------------------------------------------------------------------
# The PPX's own name table: every construct ppxlib is told about is passed as a
# string literal under sarek/ppx/. Built from the literals themselves rather
# than from a hand-kept list, so a newly declared construct needs no edit here.
# ---------------------------------------------------------------------------
ppx_files = tracked("sarek/ppx/*.ml", "sarek/ppx/*.mli")
if not ppx_files:
    print("::error::no tracked sarek/ppx sources -- cannot build the PPX name "
          "table, and a check that cannot decide is not a pass", file=sys.stderr)
    sys.exit(2)

known = set()
for f in ppx_files:
    src = open(f, errors="replace").read()
    _, strs = lex(src)
    if strs is None:
        print(f"::error::{f}: unterminated comment -- cannot lex", file=sys.stderr)
        sys.exit(2)
    for _, s in strs:
        known.add(s.strip())

# Attributes owned by the OCaml compiler / ppxlib, not by this PPX. A message
# may name these because the user really can write them.
BUILTIN = {
    "warning", "warnerror", "alert", "deriving", "inline", "inlined",
    "specialise", "specialised", "unboxed", "boxed", "immediate", "immediate64",
    "tailcall", "tail_mod_cons", "noalloc", "untagged", "unrolled", "poll",
    "local", "nonlocal", "ppwarning", "pperror",
}
def is_known(name):
    return (name in known
            or name in BUILTIN
            or name.startswith("ocaml.")
            or name.split(".")[0] in BUILTIN)

CONSTRUCT = re.compile(r"\[(@{1,4}|%{1,4})([A-Za-z_][A-Za-z0-9_.']*)\]")

# `[%s]`, `[%d]`, `[%Ld]` are printf conversions inside literal brackets, not
# extension points -- messages are full of them. Every ppxlib construct spelling
# in this tree is a word (>= 2 chars) and no printf conversion is, once the
# optional length modifier is accounted for. Only the SINGLE-% sigil is
# ambiguous: `[%%s]` in a format string renders `[%s]`, which is not an
# extension point either, so the same filter applies to it.
PRINTF_CONV = re.compile(r"^[lLn]?[a-zA-Z]$")
def is_printf_conversion(sigil, name):
    return set(sigil) == {"%"} and PRINTF_CONV.match(name) is not None
# Which calls collapse '@@' and '%%', and which do not. A string literal is
# attributed to the NEAREST PRECEDING call head from either set -- a line window
# is not enough, because a Printf.sprintf sitting two lines below a
# Location.raise_errorf would inherit the wrong rule and the gate would demand a
# doubling that breaks the Printf message. The non-Format set therefore has to be
# listed too: it is what STOPS attribution, so anything missing from it can only
# cause a false positive, which is the failure mode that gets a gate deleted.
FMT = re.compile(r"(?:Ppxlib\.)?(?:Location\.(?:raise_)?errorf"
                 r"|Format\.(?:fprintf|asprintf|sprintf|eprintf|printf"
                 r"|kasprintf|kfprintf|ksprintf|dprintf|ifprintf))"
                 r"|\bppwarning\b")
NONFMT = re.compile(r"\bPrintf\.(?:sprintf|printf|eprintf|fprintf|ksprintf|kprintf)"
                    r"|\bfailwith\b|\binvalid_arg\b|\braise\b|\bprint_endline\b"
                    r"|\bprint_string\b|\bprerr_endline\b|\bprerr_string\b"
                    r"|\bString\.concat\b|\bassert\b|\bstring_of_\w+\b"
                    r"|\bScanf\.\w+|\bexn_of\w*|\bError\b|\bOk\b")
# A format string is an argument of its call, so it is close to it. Beyond this
# many bytes the nearest preceding call head is not evidence of anything.
FMT_REACH = 600

sources = [f for f in tracked("*.ml", "*.mli")
           if not f.startswith("sarek/tests/")]
if not sources:
    print("::error::no tracked OCaml sources matched -- nothing was scanned",
          file=sys.stderr)
    sys.exit(2)

name_viol, render_viol = [], []
scanned_literals = 0

for f in sources:
    src = open(f, errors="replace").read()
    blanked, strs = lex(src)
    if strs is None:
        print(f"::error::{f}: unterminated comment -- cannot lex", file=sys.stderr)
        sys.exit(2)
    # Call heads, by byte offset, tagged with whether they collapse the sigil.
    heads = sorted([(m.start(), True) for m in FMT.finditer(blanked)]
                   + [(m.start(), False) for m in NONFMT.finditer(blanked)])
    def collapses(off):
        """Does the nearest preceding call head collapse '@@' and '%%'?"""
        best = None
        for pos, is_fmt in heads:
            if pos >= off:
                break
            best = (pos, is_fmt)
        return best is not None and best[1] and off - best[0] <= FMT_REACH

    for off, text in strs:
        if "[" not in text:
            continue
        scanned_literals += 1
        ln = line_of(src, off)
        in_fmt = collapses(off)
        for m in CONSTRUCT.finditer(text):
            sigil, name = m.group(1), m.group(2)
            if is_printf_conversion(sigil, name):
                continue
            if not is_known(name):
                name_viol.append((f, ln, m.group(0), name))
            if in_fmt and sigil in ("@@", "%%"):
                render_viol.append((f, ln, m.group(0),
                                    "[" + sigil[0] + name + "]"))

failed = 0
if name_viol:
    print("::error::message text names a PPX construct that does not exist:")
    for f, ln, whole, name in name_viol:
        print(f"  {f}:{ln}: {whole} -- no '{name}' is declared under sarek/ppx/")
    failed = 1
if render_viol:
    print("::error::message text will not render the construct it names "
          "(Format collapses '@@'->'@' and '%%'->'%'):")
    for f, ln, whole, rendered in render_viol:
        print(f"  {f}:{ln}: source {whole} reaches the user as {rendered}")
    print("  Fix: double the sigil in the format string "
          "(\"[@@@@sarek.type]\" prints \"[@@sarek.type]\").")
    failed = 1

if failed:
    print()
    print("An error message is a claim. See backlog-193.")
    sys.exit(1)

print(f"OK -- {len(sources)} tracked source(s), {scanned_literals} bracket-bearing "
      f"literal(s); every named construct is declared under sarek/ppx/ and "
      f"survives Format rendering")
PY
exit $?
