#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
"""Generate a formal project's proof ledger from the Rocq toolchain (task #95).

WHAT THIS REPLACES
------------------

The repository carried three hand-maintained ``proof-ledger.json`` files, two of
which described the SAME project, and they disagreed::

    formal/convergence-safety/proof-ledger.json         total 78  (59 proven, 19 defs)
    formal/convergence-safety/proof-ledger/…json        total 86  (68 proven, 18 defs)
    formal/type-safety/proof-ledger/…json               total 90  (90 proven, 36 defs)

and ``formal/codegen-ptx`` had none at all. The divergence was diagnosed inside
the project itself (``formal/convergence-safety/report/WORKFLOW_FEEDBACK.md``,
entry 98: "structurally diverged … a silent false PASS on every T3 tick") and
then not repaired, because repairing today's numbers by hand reintroduces
exactly the defect: two copies of one fact, drifting.

The consequence was concrete. ``scripts/check-formal-proofs.sh`` step 5 could
report the axioms the kernel found but could not ENFORCE an allowlist, because
there was no single trustworthy expected value to compare against. #45 made the
proofs machine-rechecked; the counts stayed prose.

So the ledger is no longer written. It is derived, by this script, from the two
artefacts the toolchain produces and a human cannot fake:

  * ``theories/*.glob`` — Rocq's own cross-reference index, emitted by ``rocq
    compile`` alongside every ``.vo``. Each entry is ``<kind> <bp>:<ep> <sec>
    <name>``. This gives the exact inventory: which names exist and what kind
    each is. It is NOT a grep over the sources: ``prf`` is what the elaborator
    recorded as a proof, so a ``Theorem`` inside a comment or a string does not
    appear, and a ``Lemma`` written on a continuation line does.

  * ``rocq check -o`` — the standalone kernel re-checker's CONTEXT SUMMARY,
    which lists the assumptions the checked proof terms actually depend on,
    transitively. This is the closure, not the declaration: an axiom reachable
    only through an imported library shows up here and nowhere in the sources.

WHY BOTH
--------

They answer different questions and cross-check each other. ``.glob`` says what
this project DECLARES (``ax`` entries = ``Axiom``/``Parameter`` written here);
``rocq check -o`` says what the proofs DEPEND ON. A project-local axiom that
appears in the kernel closure but not in the glob would mean the .vo files do
not correspond to the .v sources beside them. The generator asserts the two
agree and fails if they do not (``--strict``, on by default), so the ledger
cannot be generated from a stale build.

WHAT IS "PROJECT-LOCAL"
-----------------------

An axiom is project-local when its fully-qualified name begins with this
project's logical path (from ``_CoqProject``'s ``-R theories <Logical>`` line).
Everything else is the trusted base: ``Corelib.Floats.PrimFloat.*`` and
``Corelib.Numbers.Cyclic.Int63.*`` are Rocq's primitive float and 63-bit integer
axiomatisations, which any development using ``PrimFloat`` inherits and which
are not this repository's to sanction or remove. They are recorded in full
rather than counted, so that a NEW toolchain axiom entering through a new import
is visible in the diff instead of hidden inside a total.

DETERMINISM
-----------

``rocq check`` prints its axiom list in an unspecified (hash) order, so every
list here is sorted. The output is byte-stable for a given source tree and Rocq
version, which is what lets ``scripts/check-formal-proofs.sh`` regenerate and
diff it as a gate.

USAGE
-----

    scripts/gen-proof-ledger.py formal/<project> [-o OUT]

Requires the project to have been BUILT (``.vo`` and ``.glob`` present); it does
not build, because the one caller that needs it — check-formal-proofs.sh — has
just rebuilt everything from scratch and rebuilding again would only re-check
the checker. Refuses to run against a partially built tree rather than emitting
a ledger that silently omits a module.

Python 3 with the standard library only: the gate runs inside
``rocq/rocq-prover:9.1.1``, which has python3 but neither node nor jq.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# Glob record kinds this script interprets. Rocq emits more (`binder`, `var`,
# `constr`, `scheme`, `proj`, `not`, `R`), all of which are either local to a
# statement or DERIVED from a declaration that is already counted — counting a
# record's projections or an inductive's auto-generated `_rect` scheme would
# inflate the totals with names nobody wrote.
KIND_FIELDS = {
    "prf": "theorems",       # Theorem / Lemma / Corollary / Proposition / Fact / Remark
    "def": "definitions",    # Definition / Fixpoint / Let
    "ind": "inductives",     # Inductive
    "rec": "records",        # Record
    "ax": "axioms_declared", # Axiom / Parameter / Hypothesis
}

GLOB_ENTRY = re.compile(r"^(?P<kind>[a-z]+) \d+:\d+ (?P<sec>\S+) (?P<name>\S+)$")
GLOB_FILE_HEADER = re.compile(r"^F(?P<module>\S+)$")


def die(msg):
    print("::error::gen-proof-ledger: " + msg, file=sys.stderr)
    sys.exit(1)


def logical_path(project_dir):
    """Read the project's logical name from ``-R theories <Logical>``."""
    coqproject = os.path.join(project_dir, "_CoqProject")
    if not os.path.isfile(coqproject):
        die("%s has no _CoqProject" % project_dir)
    with open(coqproject, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^-[RQ]\s+theories\s+(\S+)", line.strip())
            if m:
                return m.group(1)
    die("%s/_CoqProject has no '-R theories <Logical>' line" % project_dir)


def read_globs(project_dir):
    """Inventory every declaration, per module, from the .glob cross-reference.

    Returns ``{module: {field: [names]}}``.
    """
    theories = os.path.join(project_dir, "theories")
    sources = sorted(
        f[:-2] for f in os.listdir(theories) if f.endswith(".v")
    )
    if not sources:
        die("%s/theories has no .v files — a ledger generated from nothing "
            "would be an empty file that passes every check" % project_dir)

    inventory = {}
    for base in sources:
        globfile = os.path.join(theories, base + ".glob")
        vofile = os.path.join(theories, base + ".vo")
        # A missing artefact is fatal, not skippable. Skipping is how a ledger
        # ends up describing four of five modules and still diffing clean.
        if not os.path.isfile(globfile) or not os.path.isfile(vofile):
            die("%s/theories/%s.v has no .glob/.vo — build the project first "
                "(rocq makefile -f _CoqProject -o CoqMakefile && make -f CoqMakefile)"
                % (project_dir, base))

        module = None
        per_kind = {field: [] for field in KIND_FIELDS.values()}
        with open(globfile, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                header = GLOB_FILE_HEADER.match(line)
                if header:
                    module = header.group("module")
                    continue
                m = GLOB_ENTRY.match(line)
                if not m:
                    continue
                field = KIND_FIELDS.get(m.group("kind"))
                if field is None:
                    continue
                sec = m.group("sec")
                qualified = m.group("name") if sec == "<>" else sec + "." + m.group("name")
                per_kind[field].append(qualified)
        if module is None:
            die("%s has no F<module> header line; it is not a Rocq glob file"
                % globfile)
        inventory[module] = {k: sorted(v) for k, v in per_kind.items()}
    return inventory


def kernel_axioms(project_dir, logical):
    """Run ``rocq check -o`` and return its sorted CONTEXT SUMMARY axiom list.

    The checker's exit code is load-bearing, so its output is captured whole
    rather than piped: a pipeline to head/grep would report SIGPIPE 141 as a
    failure of the checker.
    """
    theories = os.path.join(project_dir, "theories")
    vos = sorted(
        os.path.join("theories", f) for f in os.listdir(theories) if f.endswith(".vo")
    )
    checker = "rocq"
    argv = ["rocq", "check", "-silent", "-o", "-R", "theories", logical] + vos
    try:
        proc = subprocess.run(
            argv, cwd=project_dir, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        checker = "coqchk"
        argv = ["coqchk", "-silent", "-o", "-R", "theories", logical] + vos
        try:
            proc = subprocess.run(
                argv, cwd=project_dir, capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            die("no rocq/coqchk on PATH — this ledger is generated BY the "
                "kernel checker; without it there is nothing to record")

    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        die("%s check failed for %s (exit %d):\n%s" % (checker, project_dir, proc.returncode, out))

    # The summary section is `* Axioms:` followed by indented names, terminated
    # by the next `*` heading or a blank-ish line. Absence of the heading is NOT
    # the same as "no axioms": `rocq check -o` prints `* Axioms:` with `<none>`
    # when there are none, so a missing heading means the output was not the
    # CONTEXT SUMMARY at all and the parse must fail loudly.
    if "CONTEXT SUMMARY" not in out:
        die("%s check produced no CONTEXT SUMMARY for %s; refusing to record an "
            "empty axiom list that would look like a clean result:\n%s"
            % (checker, project_dir, out[-2000:]))
    if "* Axioms:" not in out:
        die("%s check printed a CONTEXT SUMMARY with no '* Axioms:' section for "
            "%s — the output format changed and this parser would silently "
            "report zero axioms" % (checker, project_dir))

    axioms = []
    collecting = False
    for line in out.splitlines():
        if line.startswith("* Axioms:"):
            collecting = True
            # `rocq check` puts the whole section on one line when it is empty
            # (`* Axioms: <none>`) and on following indented lines otherwise.
            # The same-line remainder is parsed rather than skipped: skipping it
            # happens to give the right answer for `<none>` and would silently
            # drop a single same-line axiom if the formatting ever changed.
            rest = line[len("* Axioms:"):].strip()
            if rest and rest != "<none>":
                axioms.append(rest)
            if rest == "<none>":
                break
            continue
        if collecting:
            stripped = line.strip()
            if line.startswith("*"):
                break
            if not stripped:
                continue
            if stripped == "<none>":
                break
            axioms.append(stripped)
    return sorted(set(axioms))


def build_ledger(project_dir, rocq_version):
    # abspath, not normpath: the gate invokes this with `.` from inside the
    # project directory, and normpath(".") is "." — which produced a ledger
    # naming the project "." that differed from the committed one in exactly one
    # field, reported as "differs in the per-module theorem lists".
    project = os.path.basename(os.path.abspath(project_dir))
    logical = logical_path(project_dir)
    inventory = read_globs(project_dir)
    closure = kernel_axioms(project_dir, logical)

    local_prefix = logical + "."
    axioms_local = [a for a in closure if a.startswith(local_prefix)]
    axioms_base = [a for a in closure if not a.startswith(local_prefix)]

    # Cross-check the two instruments (see module docstring). The glob records
    # what this project DECLARES; the kernel closure records what the proofs
    # DEPEND ON. A declared axiom that no proof reaches is fine and expected;
    # a project-local axiom in the closure that the sources never declared is
    # not, and means the .vo files do not match the .v beside them.
    # The glob's F-header module name is already fully qualified
    # (`CodegenPtx.AGpuSemantics`), so it is NOT re-prefixed with the logical
    # path — doing so yields `CodegenPtx.CodegenPtx.AGpuSemantics.sin_f32`,
    # which matches nothing and makes every sanctioned Parameter look like an
    # unexplained axiom.
    declared = sorted(
        {module + "." + name
         for module, kinds in inventory.items()
         for name in kinds["axioms_declared"]}
    )
    unexplained = sorted(set(axioms_local) - set(declared))
    if unexplained:
        die("the kernel found project-local axioms that no source file declares:"
            "\n    %s\nThe committed .vo files do not correspond to the .v "
            "sources. Rebuild from scratch." % "\n    ".join(unexplained))

    totals = {field: 0 for field in KIND_FIELDS.values()}
    for kinds in inventory.values():
        for field, names in kinds.items():
            totals[field] += len(names)

    return {
        "schema": 2,
        "generated_by": "scripts/gen-proof-ledger.py",
        "do_not_edit": (
            "Generated from theories/*.glob and `rocq check -o`. Regenerate with "
            "scripts/gen-proof-ledger.py; scripts/check-formal-proofs.sh fails on drift."
        ),
        "project": project,
        "logical_path": logical,
        "rocq_version": rocq_version,
        "counts": {
            "modules": len(inventory),
            "theorems": totals["theorems"],
            "definitions": totals["definitions"],
            "inductives": totals["inductives"],
            "records": totals["records"],
            "axioms_declared": totals["axioms_declared"],
            "axioms_project_local": len(axioms_local),
            "axioms_toolchain_base": len(axioms_base),
        },
        "modules": {
            module: {
                "theorems": kinds["theorems"],
                "counts": {field: len(names) for field, names in sorted(kinds.items())},
            }
            for module, kinds in sorted(inventory.items())
        },
        # The enforced set. scripts/check-formal-proofs.sh compares this against
        # formal/axiom-allowlist.txt and fails on any addition.
        "axioms_project_local": axioms_local,
        # Rocq's own primitive-float / int63 axiomatisation, inherited through
        # imports. Recorded in full, not counted, so that a new one entering
        # through a new import shows up as a diff rather than a changed integer.
        "axioms_toolchain_base": axioms_base,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project_dir", help="path to a formal/<project> directory")
    ap.add_argument("-o", "--output", help="write here instead of <project>/proof-ledger.json")
    args = ap.parse_args()

    if not os.path.isdir(args.project_dir):
        die("%s is not a directory" % args.project_dir)

    try:
        version_out = subprocess.run(
            ["rocq", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            version_out = subprocess.run(
                ["coqc", "--version"], capture_output=True, text=True, check=True
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            die("no rocq/coqc on PATH")
    m = re.search(r"version (\S+)", version_out)
    rocq_version = m.group(1) if m else version_out.strip()

    ledger = build_ledger(args.project_dir, rocq_version)
    out = args.output or os.path.join(args.project_dir, "proof-ledger.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print("wrote %s (%d theorems, %d project-local axioms, %d toolchain-base)"
          % (out, ledger["counts"]["theorems"],
             ledger["counts"]["axioms_project_local"],
             ledger["counts"]["axioms_toolchain_base"]))


if __name__ == "__main__":
    main()
