#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
"""Enforce the generated proof ledgers against the repository (task #95).

Called by ``scripts/check-formal-proofs.sh`` with a directory of ledgers freshly
generated from a from-scratch rebuild + kernel re-check. It compares those
against what is committed, and enforces the two things the committed files are
now allowed to assert.

THE THREE CHECKS
----------------

**drift** — every ``formal/<project>/proof-ledger.json`` must be byte-identical
to what this build produced. This is what makes the ledger a fact rather than a
claim: it cannot be edited, and it cannot fall behind the proofs, because either
one turns the gate red. It is the same shape as the existing
``check-generated-code`` job, applied to the formal side.

**allowlist** — the project-local axioms the kernel actually depends on must
equal ``formal/axiom-allowlist.txt``, in both directions. Note this is NOT
subsumed by the drift check even though the ledger also lists the axioms:
regenerating the ledger after adding an axiom makes the drift check pass. The
allowlist is a separate, human-signed statement, and adding to it is a visible
act in the diff. The reverse direction (a listed axiom no proof reaches) matters
just as much — a list that only grows stops describing the code.

**anchors** — every theorem named in a ``proof-notes.json`` must be a theorem in
the generated ledger. The hand-written ledgers this replaces carried an entry
``check_env_nonvarying_uniform``, marked PROVEN, which is not a theorem: it was
an anchor invented for the PAIR of real lemmas ``..._seq`` and ``..._args``.
Nothing in the repository could notice, and the project's own retrospective had
already flagged that class of entry as "unverifiable anchors".

Python 3, standard library only: this runs inside ``rocq/rocq-prover:9.1.1``,
which has no node and no jq.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWLIST = os.path.join(ROOT, "formal", "axiom-allowlist.txt")

failures = []


def fail(msg):
    failures.append(msg)
    print("::error::" + msg.replace("\n", "%0A"), file=sys.stderr)
    print("FAIL: " + msg)


def read_allowlist(path):
    if not os.path.isfile(path):
        return None
    names = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return sorted(set(names))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generated", required=True,
                    help="directory of freshly generated <project>.json ledgers")
    args = ap.parse_args()

    generated = {}
    for name in sorted(os.listdir(args.generated)):
        if name.endswith(".json"):
            with open(os.path.join(args.generated, name), encoding="utf-8") as fh:
                generated[name[:-5]] = (fh.read(), None)
    # A gate with nothing to check is the failure mode this whole script family
    # exists to remove. An empty --generated directory means the loop that was
    # supposed to fill it never ran, and every check below would pass vacuously.
    if not generated:
        fail("no generated ledgers in %s — the per-project generation step did "
             "not run, so every check below would pass having verified nothing."
             % args.generated)
        return 1

    projects = sorted(
        d for d in os.listdir(os.path.join(ROOT, "formal"))
        if os.path.isfile(os.path.join(ROOT, "formal", d, "_CoqProject"))
    )
    missing = sorted(set(projects) - set(generated))
    if missing:
        fail("no ledger was generated for: %s. Every formal/ project with a "
             "_CoqProject must produce one." % ", ".join(missing))

    all_local_axioms = set()

    for project in projects:
        gen_text = generated.get(project, (None, None))[0]
        if gen_text is None:
            continue
        committed_path = os.path.join(ROOT, "formal", project, "proof-ledger.json")

        if not os.path.isfile(committed_path):
            fail("formal/%s/proof-ledger.json is missing. Regenerate it:\n"
                 "    scripts/gen-proof-ledger.py formal/%s" % (project, project))
            continue

        with open(committed_path, encoding="utf-8") as fh:
            committed_text = fh.read()

        if committed_text != gen_text:
            gen = json.loads(gen_text)
            com = json.loads(committed_text)
            detail = []
            for key in sorted(set(gen.get("counts", {})) | set(com.get("counts", {}))):
                g, c = gen.get("counts", {}).get(key), com.get("counts", {}).get(key)
                if g != c:
                    detail.append("      counts.%s: committed %r, actual %r" % (key, c, g))
            for key in ("axioms_project_local", "axioms_toolchain_base"):
                added = sorted(set(gen.get(key, [])) - set(com.get(key, [])))
                removed = sorted(set(com.get(key, [])) - set(gen.get(key, [])))
                for a in added:
                    detail.append("      %s: + %s" % (key, a))
                for a in removed:
                    detail.append("      %s: - %s" % (key, a))
            fail("formal/%s/proof-ledger.json does not match this build.\n"
                 "%s\n"
                 "    The ledger is generated, not written. Regenerate it:\n"
                 "        scripts/gen-proof-ledger.py formal/%s"
                 % (project, "\n".join(detail) or
                    "      (differs in the per-module theorem lists)", project))
            continue

        gen = json.loads(gen_text)
        all_local_axioms.update(gen["axioms_project_local"])

        theorems = {
            name
            for module in gen["modules"].values()
            for name in module["theorems"]
        }
        notes_path = os.path.join(ROOT, "formal", project, "proof-notes.json")
        if os.path.isfile(notes_path):
            with open(notes_path, encoding="utf-8") as fh:
                notes = json.load(fh)
            for key in ("counts", "summary", "total", "admits"):
                if key in notes:
                    fail("formal/%s/proof-notes.json has a %r key. Counts belong "
                         "to the generated proof-ledger.json and nowhere else — "
                         "a second copy is how the three ledgers this replaced "
                         "came to disagree." % (project, key))
            phantom = sorted(set(notes.get("theorems", {})) - theorems)
            if phantom:
                fail("formal/%s/proof-notes.json annotates %d name(s) that are "
                     "not theorems in this build:\n        %s\n"
                     "    Either the theorem was renamed or removed, or the note "
                     "anchors something Rocq never saw."
                     % (project, len(phantom), "\n        ".join(phantom)))
            print("  %-20s %3d theorems, %3d annotated, %d project-local axiom(s)"
                  % (project, len(theorems), len(notes.get("theorems", {})),
                     len(gen["axioms_project_local"])))
        else:
            print("  %-20s %3d theorems, no notes file, %d project-local axiom(s)"
                  % (project, len(theorems), len(gen["axioms_project_local"])))

    allow = read_allowlist(ALLOWLIST)
    if allow is None:
        fail("formal/axiom-allowlist.txt is missing. Without it the axiom check "
             "cannot run, and its absence must fail the gate rather than skip it.")
    else:
        found = sorted(all_local_axioms)
        unsanctioned = sorted(set(found) - set(allow))
        stale = sorted(set(allow) - set(found))
        if unsanctioned:
            fail("the kernel checker found project-local axiom(s) that "
                 "formal/axiom-allowlist.txt does not sanction:\n        %s\n"
                 "    An axiom is an unproven assumption the proofs rest on. If "
                 "this one is deliberate, add it to the allowlist WITH the "
                 "reason it cannot be proved instead."
                 % "\n        ".join(unsanctioned))
        if stale:
            fail("formal/axiom-allowlist.txt sanctions axiom(s) that no proof "
                 "depends on any more:\n        %s\n"
                 "    Remove them. An allowlist that only grows stops describing "
                 "the code it guards." % "\n        ".join(stale))
        if not unsanctioned and not stale:
            print("  axiom allowlist: %d sanctioned, %d found, exact match"
                  % (len(allow), len(found)))

    if failures:
        print("\n%d ledger/axiom check(s) FAILED." % len(failures))
        return 1
    print("OK: ledgers match this build, axioms match the allowlist, every note "
          "anchors a real theorem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
