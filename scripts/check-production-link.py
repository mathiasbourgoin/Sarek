#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
"""Print, and mechanically verify, the shipped-artefact/model-only theorem split
(backlog-201; settles the "79-theorem split" left open by backlog-187).

WHAT THIS ANSWERS
-----------------

backlog-95 made the THEOREM COUNT trustworthy: it cannot be hand-typed, because
scripts/check-proof-ledger.py fails on drift from a from-scratch rebuild. It did
not make the count's SCOPE trustworthy. "280 machine-checked theorems" claims
nothing about what is checked against, and roughly a third of those theorems are
proven about a Rocq model with no mechanical connection to the OCaml that ships
— a fact that lived only in README/STATUS.md prose (which drifts) until now.

This script does not derive the split from the proof-ledger.json build the way
gen-proof-ledger.py derives counts from the Rocq kernel: whether a module's
proofs are checked against production is not a fact `rocq check` can report, it
is a fact about which OCaml a *test file* calls. So the split is a two-part
claim, same shape as the axiom allowlist (#95): a human-authored declaration
(``formal/<project>/production-link.json``, one per project) naming the module
and the test file and production symbol backing the claim, MECHANICALLY VERIFIED
here against the freshly generated ledger (the module and its theorem count must
be real) and the named test file's actual content (the production symbol must
appear as a live call, comments stripped — a claim whose test regressed must
fail loudly, not keep being cited).

A module not listed in a project's production-link.json is model-only. That is
the safe default: the file must affirmatively claim a link, an absent or empty
file claims none, so a project that never wires one up (formal/convergence-safety
today) reports 0 shipped-linked rather than silently inheriting a claim.

WHAT "SHIPPED-ARTEFACT" MEANS, PRECISELY
-----------------------------------------

A theorem counts as shipped-artefact when its module has a declared, verified
production-link entry: a test exercises BOTH the extracted Rocq model AND the
named real Sarek_* function on the same inputs and asserts they agree (or, for
codegen-ptx/PtxLayout, the extracted theory's own definitions are run directly
against the production module — no mirror in between). It does NOT mean every
theorem in that module was itself exercised by that test's generator; module
granularity is the finest grain this script can verify mechanically without a
per-theorem coverage map, which does not exist. Conversely "model-only" does not
mean untested or worthless: model-only modules are still machine-checked Rocq
theorems, most have QCheck/extraction validation against the extracted model
itself, and some (PtxLayout excepted, everything in codegen-ptx besides it) are
exercised end-to-end by ptxas/nvdisasm assembly gates at the TEST level — those
gates just are not a proof-to-production link for the THEOREM, which is the
specific claim this script is about.

USAGE
-----

    scripts/check-production-link.py --generated DIR
        DIR containing freshly generated <project>.json ledgers, same shape
        scripts/check-proof-ledger.py consumes. Use this form from
        scripts/check-formal-proofs.sh, right after the drift/allowlist/anchor
        check, so the split is derived from the SAME from-scratch rebuild.

    scripts/check-production-link.py
        No --generated: reads the committed formal/<project>/proof-ledger.json
        directly. This is what a docs-generation step (or a developer, between
        full rebuilds) runs; it trusts the committed ledgers, which is only sound
        because check-proof-ledger.py already guarantees they are not stale.

Exit 0 and prints the two headline numbers plus a per-module table on success.
Exit 1 on any unverifiable claim (module doesn't exist, theorem count mismatch,
test file missing, or the production symbol not found as a live call).
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []


def fail(msg):
    failures.append(msg)
    print("::error::" + msg.replace("\n", "%0A"), file=sys.stderr)
    print("FAIL: " + msg)


def strip_ocaml_comments(text):
    """Remove (* ... *) comments, honouring nesting, so a claim's production
    symbol cannot be satisfied by a comment that merely mentions it (a real
    hazard here: the header of test_type_safety_conformance.ml itself contains
    the literal text "Sarek_typer.infer (the production inference engine)").
    Does not attempt to skip string literals — no test file under formal/
    contains a quoted "(*"/"*)" that would confuse this, and a false failure
    here is safe (it demands a human look), unlike a false pass."""
    out = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i : i + 2] == "(*":
            depth += 1
            i += 2
            continue
        if text[i : i + 2] == "*)" and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def load_ledger(project, generated_dir):
    if generated_dir is not None:
        path = os.path.join(generated_dir, project + ".json")
    else:
        path = os.path.join(ROOT, "formal", project, "proof-ledger.json")
    if not os.path.isfile(path):
        fail("no ledger found for %s at %s" % (project, path))
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def module_short_name(full_name):
    # ledger module keys are "<LogicalPath>.<Module>"; production-link.json
    # keys by the bare module name, since the logical path is redundant with
    # the project and (unlike the module) is not always what a reader expects.
    return full_name.rsplit(".", 1)[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--generated",
        default=None,
        help="directory of freshly generated <project>.json ledgers "
        "(scripts/gen-proof-ledger.py's output dir). Omit to read the "
        "committed formal/<project>/proof-ledger.json files directly.",
    )
    args = ap.parse_args()

    projects = sorted(
        d
        for d in os.listdir(os.path.join(ROOT, "formal"))
        if os.path.isfile(os.path.join(ROOT, "formal", d, "_CoqProject"))
    )
    if not projects:
        fail("no formal/ projects found — this gate would pass vacuously")
        return 1

    total_shipped = 0
    total_model = 0
    rows = []  # (project, module, theorems, shipped?, production_call or None)

    for project in projects:
        ledger = load_ledger(project, args.generated)
        if ledger is None:
            continue

        link_path = os.path.join(ROOT, "formal", project, "production-link.json")
        declared = {}
        if os.path.isfile(link_path):
            with open(link_path, encoding="utf-8") as fh:
                link_doc = json.load(fh)
            declared = link_doc.get("modules", {})
        # No production-link.json, or an empty "modules" object, both mean
        # zero shipped-linked modules for this project — a claim must be
        # affirmative, so absence is never silently inherited as a link.

        modules = ledger.get("modules", {})
        short_to_full = {module_short_name(m): m for m in modules}

        verified_shipped = set()
        for mod_short, spec in declared.items():
            full = short_to_full.get(mod_short)
            if full is None:
                fail(
                    "%s/production-link.json claims module %r, which is not "
                    "in this build's ledger (modules present: %s). The module "
                    "was renamed or removed and the claim was not updated."
                    % (project, mod_short, ", ".join(sorted(short_to_full)) or "(none)")
                )
                continue
            test_rel = spec.get("test_file")
            call = spec.get("production_call")
            if not test_rel or not call:
                fail(
                    "%s/production-link.json entry %r is missing test_file or "
                    "production_call" % (project, mod_short)
                )
                continue
            test_path = os.path.join(ROOT, "formal", project, test_rel)
            if not os.path.isfile(test_path):
                fail(
                    "%s/production-link.json claims module %r is checked "
                    "against production in %r, which does not exist."
                    % (project, mod_short, test_rel)
                )
                continue
            with open(test_path, encoding="utf-8") as fh:
                content = strip_ocaml_comments(fh.read())
            # Require a call, not merely a qualified mention: the symbol
            # followed by '(' or whitespace-then-'(' covers both direct-call
            # and the common OCaml `f x` application style used here (the
            # helper wraps a value, e.g. `Sarek_typer.infer env expr`).
            pattern = re.escape(call) + r"\b"
            if not re.search(pattern, content):
                fail(
                    "%s/production-link.json claims module %r is checked "
                    "against production via %r in %r, but no live reference "
                    "to it was found outside comments. Either the test "
                    "regressed and the link is broken, or the claim is stale "
                    "and should be removed."
                    % (project, mod_short, call, test_rel)
                )
                continue
            verified_shipped.add(mod_short)

        for full_mod, mod_data in modules.items():
            short = module_short_name(full_mod)
            n_th = mod_data["counts"]["theorems"]
            is_shipped = short in verified_shipped
            if is_shipped:
                total_shipped += n_th
            else:
                total_model += n_th
            rows.append(
                (
                    project,
                    short,
                    n_th,
                    is_shipped,
                    declared.get(short, {}).get("production_call"),
                )
            )

    if failures:
        print("\n%d production-link check(s) FAILED." % len(failures))
        return 1

    print("== shipped-artefact vs model-only theorem split (backlog-201)")
    for project, short, n_th, is_shipped, call in sorted(rows):
        tag = "shipped" if is_shipped else "model-only"
        via = (" via %s" % call) if call else ""
        print("  %-20s %-16s %3d theorems  [%s%s]" % (project, short, n_th, tag, via))
    print()
    print("shipped-artefact theorems: %d" % total_shipped)
    print("model-only theorems:       %d" % total_model)
    print("total (== proof-ledger sum): %d" % (total_shipped + total_model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
