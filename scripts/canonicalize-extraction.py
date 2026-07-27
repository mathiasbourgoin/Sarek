#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
"""Canonicalise Rocq extraction output so it is byte-stable across toolchains.

THE PROBLEM THIS SOLVES

scripts/check-formal-proofs.sh regenerates the extracted layout model and
byte-compares it against the committed copy, so that a change to
theories/PtxLayout.v cannot reach the conformance suite without the model being
regenerated (task #46). The first CI run of that gate failed, and the diff was
this, in full:

    -  { cl_tag = tag; cl_leaves = (flattens (tag :: []) payoff (number_args 0 c));
    -    cl_payload_size =
    +  { cl_tag = tag; cl_leaves =
    +    (flattens (tag :: []) payoff (number_args 0 c)); cl_payload_size =

Identical tokens, different line breaks. Rocq's extractor pretty-prints through
OCaml's Format module, and Format's line-breaking decisions differ between the
OCaml the prover itself was built with: 5.5.0 for a local distro install, 4.14.2
inside rocq/rocq-prover:9.1.1. Same Rocq version, same sources, same theorems —
different wrapping.

A byte-compare of raw extractor output is therefore not a drift check. It is a
check on which machine ran the extractor, and it would have failed on every CI
run forever while being unable to distinguish that from the real drift it exists
to catch. A gate that fires on something harmless is a gate that gets deleted by
the next person in a hurry.

WHY NOT ocamlformat

The obvious fix — normalise both sides through ocamlformat — cannot run where it
is needed. The extraction happens inside rocq/rocq-prover:9.1.1, which ships no
ocamlformat (verified: `command -v ocamlformat` is empty; the image has OCaml
4.14.2 and the prover, nothing else). Moving the comparison to a job that has
ocamlformat would mean moving the extraction there too, and that job has no Rocq.

So the repository owns its own normaliser. It needs only python3, which the Rocq
image does have, and it is small enough to audit at a glance — which matters,
because it stands between a proof and a test.

WHAT IT DOES

Within each paragraph (a maximal run of non-blank lines) it re-flows the text:
split on whitespace, greedily refill to 78 columns, indent continuations by two
spaces. Blank lines and paragraph order are preserved, so the file keeps the
"comment, blank, definition" shape the extractor produces.

WHY RE-FLOWING CANNOT CORRUPT THESE FILES

OCaml is not whitespace-sensitive outside literals, and there are no literals
here to be sensitive about: the extracted model contains no double-quote
character at all (asserted below — the script refuses rather than risks it).
There are no line comments in OCaml, so joining lines cannot comment out code,
and `(* ... *)` delimiters keep their relative order under re-flowing. Type
variables (`'a1`) and any character literals contain no internal whitespace, so
splitting on whitespace never divides one.

That is the argument. The evidence is that the canonicalised model still
compiles and test_layout_conformance.ml still passes on it, which is what is
actually checked in CI.

USAGE

    scripts/canonicalize-extraction.py FILE...            # rewrite in place
    scripts/canonicalize-extraction.py --check FILE...    # exit 1 if not canonical

Idempotent: canonicalising a canonical file is a no-op, which is what lets the
gate canonicalise the regenerated output and compare it against a committed copy
that went through the same function.
"""

import argparse
import re
import sys

WIDTH = 78
CONT_INDENT = "  "


def canonicalize(text, path="<input>"):
    # A double quote would mean a string literal, and re-flowing across one
    # could move a newline into it or split it. There are none in this
    # extractor's output; if that ever changes, this must stop rather than
    # silently produce something that compiles differently.
    if '"' in text:
        raise ValueError(
            "%s contains a double quote, so it may hold a string literal. "
            "Re-flowing whitespace is only safe because these files have none. "
            "Refusing to rewrite it: fix this script to tokenise properly "
            "before allowing string literals through." % path
        )

    out = []
    for para in text.split("\n\n"):
        if not para.strip():
            continue
        words = para.split()
        if not words:
            continue
        lines = []
        current = words[0]
        for w in words[1:]:
            indent = "" if not lines else CONT_INDENT
            if len(current) + 1 + len(w) <= WIDTH:
                current += " " + w
            else:
                lines.append(current)
                current = CONT_INDENT + w
                del indent
        lines.append(current)
        out.append("\n".join(lines))

    return "\n\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any file is not already canonical",
    )
    args = ap.parse_args()

    dirty = []
    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
        try:
            result = canonicalize(original, path)
        except ValueError as exc:
            print("::error::%s" % exc, file=sys.stderr)
            return 1
        if result == original:
            continue
        dirty.append(path)
        if not args.check:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(result)

    if args.check and dirty:
        print(
            "::error::not canonical: %s\n    Run scripts/canonicalize-extraction.py "
            "on them and commit the result." % ", ".join(dirty),
            file=sys.stderr,
        )
        return 1
    if not args.check and dirty:
        print("canonicalised: %s" % ", ".join(dirty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
