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
#                 fails). Fourteen sites were in this state when this was
#                 written, four of them the `ktype` messages above.
#
#                 The requirement is the DECLARED spelling, not "more sigils".
#                 How many the user writes is a property of the ppxlib
#                 declaration CONTEXT: an expression extension is [%kernel ...]
#                 with ONE '%', so "[%%kernel ...]" in a Format string is already
#                 correct and doubling it is a defect in the other direction --
#                 which is what a flat rule did to [%kernel.real64] on the first
#                 sweep. Both polarities are checked, against the context.
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
# NOT in scope: a construct the PPX matches by hand on the AST rather than
# declaring through ppxlib (`native`, `sarek.module`). Its name is in the table,
# because the table is built from string literals, but no declaration context is
# parseable, so the RENDER half says nothing about it. A stated false negative,
# preferred over guessing a sigil count.
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

# ---------------------------------------------------------------------------
# How many sigils each construct is WRITTEN with, derived from the ppxlib
# declaration site. This is not decoration: the number is a property of the
# CONTEXT, not of the construct kind, and getting it from the kind alone is what
# made the first version of this check wrong in both directions.
#
#   Extension.Context.structure_item  -> [%%sarek_include ...]   two
#   Extension.Context.expression      -> [%kernel.real64 ...]    ONE
#   Attribute.Context.type_declaration-> [@@sarek.type]          two
#   Attribute.Context.expression      -> [@attr]                 one
#
# So "a doubled sigil in a Format string is a bug" is false: for a single-sigil
# construct the doubling is exactly what produces the right output. The first
# sweep applied the flat rule to [%kernel.real64] -- an EXPRESSION extension,
# where "[%%kernel.real64]" was already correct -- and turned a true message into
# "[%%kernel.real64]", which is not a spelling that extension answers to.
#
# A name never declared through Attribute.declare / Extension.declare (sarek's
# `native` and `sarek.module` are matched by hand on the AST) gets NO entry, and
# the render half then says nothing about it. That is a stated false negative,
# preferred over a guess. A DECLARED construct in a context this table cannot
# classify is exit 2: the check cannot decide, so it has not cleared anything.
# ---------------------------------------------------------------------------
ITEM_ATTR = {
    "type_declaration", "type_extension", "type_exception", "value_binding",
    "value_description", "module_binding", "module_declaration",
    "module_type_declaration", "class_declaration", "class_type_declaration",
    "extension_constructor", "open_description", "include_infos",
}
NODE_ATTR = {
    "expression", "pattern", "core_type", "label_declaration",
    "constructor_declaration", "row_field", "object_type_field", "class_field",
    "class_type_field", "module_expr", "module_type",
}
ITEM_EXT = {"structure_item", "signature_item"}
NODE_EXT = {
    "expression", "pattern", "core_type", "class_expr", "class_type",
    "module_expr", "module_type", "class_field", "class_type_field",
}

DECL = re.compile(r"\b(Attribute|Extension)(?:\.V\d+)?\.declare\s+"
                  r'"([^"]+)"\s+'
                  r"(?:Attribute|Extension)\.Context\.([A-Za-z_]+)")

# name -> (sigil char, how many of them the user writes)
spelling = {}
for f in ppx_files:
    for m in DECL.finditer(open(f, errors="replace").read()):
        kind, name, ctx = m.group(1), m.group(2), m.group(3)
        if kind == "Attribute":
            want = ("@", 2) if ctx in ITEM_ATTR else (
                   ("@", 1) if ctx in NODE_ATTR else None)
        else:
            want = ("%", 2) if ctx in ITEM_EXT else (
                   ("%", 1) if ctx in NODE_EXT else None)
        if want is None:
            print(f"::error::{f}: {kind}.declare \"{name}\" uses an unclassified "
                  f"context '{ctx}' -- this check cannot tell how many sigils "
                  f"that construct is written with, so it has not cleared "
                  f"anything. Add '{ctx}' to the table in this script.",
                  file=sys.stderr)
            sys.exit(2)
        prev = spelling.get(name)
        if prev is not None and prev != want:
            print(f"::error::{f}: \"{name}\" is declared in two contexts with "
                  f"different spellings ({prev} vs {want}) -- the render check "
                  f"cannot pick one", file=sys.stderr)
            sys.exit(2)
        spelling[name] = want
if not spelling:
    print("::error::no Attribute.declare / Extension.declare site parsed under "
          "sarek/ppx/ -- the render half would then check nothing at all",
          file=sys.stderr)
    sys.exit(2)

# The name need not be followed immediately by `]`. An earlier version required
# it, and that made the gate blind to every construct named WITH A PAYLOAD --
# which is most of them, and is exactly the shape it walked past on the first
# sweep: `[%%sarek_include \"file.ml\"]` in Sarek_ppx's own payload refusal
# reached the user as `[%sarek_include \"file.ml\"]`, a spelling the extension
# does not answer to ("Uninterpreted extension 'sarek_include'"). A lookahead on
# `]`, whitespace or end-of-string keeps the anchor that stops the name running
# into surrounding prose, without requiring the construct to be argument-free.
CONSTRUCT = re.compile(r"\[(@{1,4}|%{1,4})([A-Za-z_][A-Za-z0-9_.']*)(?=\]|\s|$)")

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
#
# Counted in NON-BLANK bytes. Comments are blanked to spaces with their offsets
# preserved, so a raw byte distance let a long comment between a call and its own
# format string push the literal out of reach -- which silently turns the render
# check OFF for that site, the failure shape where a gate reads green because it
# checked nothing. Whitespace and comments now cost nothing.
FMT_REACH = 200

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
        if best is None or not best[1]:
            return False
        gap = len(blanked[best[0]:off]) - blanked.count(" ", best[0], off) \
            - blanked.count("\n", best[0], off)
        return gap <= FMT_REACH

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
            # The match no longer includes the closing bracket, so reconstruct a
            # readable spelling: `...]` when a payload follows, `]` when not.
            tail = "]" if text[m.end():m.end() + 1] == "]" else " ...]"
            if not is_known(name):
                name_viol.append((f, ln, m.group(0) + tail, name))
            want = spelling.get(name)
            if want is None:
                continue
            # What the user actually sees. In a Format string each PAIR of
            # sigils collapses to one; a leftover single '@' survives ('@@@'
            # prints '@@'). An odd run of '%' cannot occur -- it would start a
            # conversion and the format string would not typecheck.
            shown = (len(sigil) // 2 + len(sigil) % 2) if in_fmt else len(sigil)
            if (sigil[0], shown) != want:
                render_viol.append(
                    (f, ln, m.group(0) + tail,
                     "[" + sigil[0] * shown + name + tail,
                     "[" + want[0] * want[1] + name + tail))

failed = 0
if name_viol:
    print("::error::message text names a PPX construct that does not exist:")
    for f, ln, whole, name in name_viol:
        print(f"  {f}:{ln}: {whole} -- no '{name}' is declared under sarek/ppx/")
    failed = 1
if render_viol:
    print("::error::message text does not reach the user as the spelling the "
          "construct is declared with (in a Format string '@@'->'@', '%%'->'%'):")
    for f, ln, whole, rendered, want in render_viol:
        print(f"  {f}:{ln}: source {whole} reaches the user as {rendered}, "
              f"but the construct is written {want}")
    print("  Fix: adjust the sigil COUNT in the format string so the rendered "
          "spelling matches -- \"[@@@@sarek.type]\" prints \"[@@sarek.type]\" "
          "for a type-declaration attribute, while an expression extension is "
          "already right as \"[%%kernel.real64]\" printing \"[%kernel.real64]\".")
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
