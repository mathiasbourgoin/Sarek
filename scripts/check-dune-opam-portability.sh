#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Refuse a `dune-project` that declares a `dune` dependency explicitly, and a
# generated .opam that carries more than one `dune` bound (backlog-213).
#
# WHY THIS EXISTS. `scripts/check-opam-clean.sh` regenerates the .opam files
# and diffs them against the tree, but it only ever sees ONE dune -- whichever
# is on PATH -- so it cannot see a divergence BETWEEN dune versions. On
# 2026-07-31 it was reported red on a clean tree under the project switch's
# dune 3.24.1 and green under an ambient dune 3.23.0, and the diagnosis took
# three attempts because a bare exit code names no toolchain.
#
# WHAT WAS ACTUALLY MEASURED, and what was not. The generation divergence
# reproduces exactly, on commit 97a062a2, with the two dune binaries named by
# absolute path and no dependency resolution involved:
#
#     $ /home/mathias/.opam/octez-setup/bin/dune build sarek.opam   # 3.23.0
#     "dune" {>= "3.15"}
#     $ <project>/_opam/bin/dune build sarek.opam                   # 3.24.1
#     "dune" {>= "3.15" & >= "3.15"}
#
# The reported green from check-opam-clean.sh under 3.23.0 was NOT reproduced
# on this machine and should not be quoted as an observation of that gate: that
# script runs `make opam`, i.e. a full `dune build @install`, and no dune 3.23.0
# here can complete it. In the project switch, ctypes 0.24.0 installs a
# `(lang dune 3.24)` dune-package that 3.23 refuses to read; in the octez-setup
# switch, OCaml 5.3.0 and its ppxlib cannot build the PPX. Both leave at the
# build, which before backlog-213 exited through `set -e` with no message --
# indistinguishable from "the tree is dirty". That indistinguishability is the
# better half of the story and is what check-opam-clean.sh's INCONCLUSIVE path
# now exists for.
#
# THE MECHANISM, from dune's own sources rather than inference. Dune injects a
# `"dune" {>= <lang version>}` dependency into every generated .opam
# (src/dune_rules/opam_create.ml, `insert_dune_dep`). When the project ALSO
# declares `dune` in a package's `depends`, the two meet. Dune 3.23
# deduplicated identical version bounds unconditionally -- CHANGES.md, under
# the heading `3.23.0 (2026-05-04)`, "Fixed": "Fix duplicate dune version bounds
# in generated opam files ... the generated opam `depends` no longer contains
# redundant constraints like `{>= \"2.7\" & >= \"2.7\"}`" (#3916, #11106). The
# NEXT PATCH, 3.23.1, GATED that deduplication on the project's lang version --
# CHANGES.md, under the heading `3.23.1 (2026-05-14)`, "Fixed":
# "Gate the `dune` version-bound deduplication in generated opam files
# (introduced in 3.23) on `(lang dune 3.23)`. Projects at earlier lang versions
# get the prior `And [...]` shape -- e.g. `{>= \"3.17\" & >= \"3.20\"}` --
# restoring 3.22 behaviour" (#14436). Neither 3.24.0 nor 3.24.1 touches this at
# all; they inherit 3.23.1's gating.
#
# So for a project at `(lang dune 3.15)`, dune 3.23.0 is the ONLY release that
# collapses the pair: before it there was no deduplication, and from 3.23.1 on
# the deduplication is gated away. The two binaries measured above sit either
# side of that single-patch window, which is exactly why one emits one bound
# and the other two.
#
# THIS ATTRIBUTION WAS WRONG IN THE FIRST VERSION OF THIS FILE, which credited
# the gating to 3.24.1 because the entry sits near the top of a CHANGES.md whose
# 3.24.1 section is also near the top. The headings are at lines 3 (3.24.1), 37
# (3.24.0), 132 (3.23.1) and 164 (3.23.0) of
# _opam/.opam-switch/sources/dune.3.24.1/CHANGES.md; the gating entry is at line
# 143, inside 3.23.1. CodeRabbit's web search contradicted the file and was
# right. The lang threshold below is unaffected -- the gate opens at
# `(lang dune 3.23)` either way -- but a durable claim about which binary does
# what has to name the release that actually did it.
#
# WHY THE FIX IS "DELETE THE DECLARATION" AND NOT SOMETHING ELSE. Two
# alternatives were considered and rejected, and they are recorded here because
# the lang bump is the first thing anyone who trips rule 1 will reach for:
#
#   * Bump `(lang dune 3.15)` to `(lang dune 3.23)`. This re-enables dedup
#     from 3.23.1 on and would let the declarations stay -- but `(lang dune X.Y)`
#     is a semantics switch, not a version number: eight minor versions of
#     changed defaults across 7 packages, a PPX and a jsoo lane. It also raises
#     the EFFECTIVE floor for consumers from 3.15 to 3.23, because dune refuses
#     to read a lang version newer than itself, so it is a user-visible
#     packaging change made to satisfy a gate. And it does not remove the
#     duplicated fact -- it only arranges for today's dune to collapse it.
#
#   * Pin a dune version. There is nowhere to pin it that everyone reads.
#     `.ocamlformat`'s `version=` pin works only because ocamlformat itself
#     refuses to run on a mismatch; dune has no such handshake and lives in
#     every contributor's own switch, so a pin would be a fourth unenforced
#     copy of a number.
#
# Deleting one of two copies of a fact beats adding a mechanism to keep the
# copies agreeing. That is what backlog-213 did, and rule 1 keeps it deleted.
#
# WHAT THIS GATE ASSERTS. Two rules, both textual, neither needing a build or a
# second toolchain -- which is the point: the divergence is decidable from the
# source file, and a check that needed two dune installations to run would be
# the check nobody runs.
#
#   RULE 1  If `(lang dune X.Y)` is BELOW 3.23, no package may declare `dune`
#           in its `depends`, in ANY form -- `(dune (>= 3.15))`, `(dune
#           :build)`, `(dune)` or a bare `dune` atom, on one line or several.
#           The declaration is redundant (dune injects the bound from the lang
#           stanza) and it is what makes the output depend on which dune ran.
#           Above 3.23 the rule is INERT and says so, which is sound rather
#           than a silent deactivation: dune refuses a lang version newer than
#           itself, so at `(lang dune 3.23)` every dune that can read the file
#           is one that deduplicates.
#
#   RULE 2  No generated .opam may carry a `"dune" {...}` constraint with more
#           than one term. That is the committed ARTEFACT of the divergence.
#           Rule 2 holds at every lang version, so rule 1 going inert never
#           leaves this gate with no assertion at all -- though at
#           `(lang dune >= 3.23)` the shapes rule 2 can still catch narrow to a
#           hand-edit and an explicit `(dune :flag)`, which produces a
#           multi-term constraint at EVERY dune version.
#
# SCOPE OF RULE 2, stated rather than assumed. Exactly the `<name>.opam` files
# at the repository root for the packages `dune-project` declares -- seven
# today. Each must exist; a declared package with no root .opam is a refusal,
# not a smaller set to be quietly content with. `.pending-opam/` also holds
# tracked .opam files and they are deliberately OUT of scope: they are not
# generated from this dune-project (different licence, `ocaml >= 4.14.0`, a
# `sarek-framework` package that no longer exists), so applying a rule about
# this dune-project's output to them would be a category error.
#
# WHAT THIS GATE DOES NOT DO. It does not run two dunes and compare. It cannot
# discover a FUTURE dune generation change of some other shape; it enforces the
# one mechanism written down above. The backstop for a different shape is CI's
# own `dune build @install` followed by `git diff --exit-code -- '*.opam'` in
# the "Build SPOC packages" step of the same `build` job this gate runs in. That
# comparison is against the COMMITTED bytes after a single build -- it catches a
# generated file that no longer matches the tree, whatever produced the change,
# which is the property wanted here. It is NOT a two-run convergence check:
# `check-opam-clean.sh` is the only thing that runs `make opam` twice, and no
# workflow invokes that -- locally it is `make check-opam-clean`, or
# `make test-all`, which also builds the GPU suites. So check-opam-clean.sh
# must not be cited as CI's backstop; the step named above is.
#
# Nor does this gate cover every two-copies-of-a-toolchain-version pair in the
# repo. The nearest uncovered sibling is `ocamlformat.0.28.1` in
# .github/workflows/ci.yml against `version=0.28.1` in `.ocamlformat`; that one
# is left alone because ocamlformat refuses to run on a mismatch, making the
# drift loud rather than silent.
#
# BEGIN prove-red-spec
# copy: scripts/check-dune-opam-portability.sh
# copy: dune-project
# copy: spoc.opam
# copy: sarek.opam
# copy: sarek-cuda.opam
# copy: sarek-hip.opam
# copy: sarek-metal.opam
# copy: sarek-opencl.opam
# copy: sarek-vulkan.opam
# invoke: scripts/check-dune-opam-portability.sh
# baseline-exit: 0
# baseline-message: check-dune-opam-portability: OK
#
# mutation: explicit-dune-dep-readded
#   desc: re-add the redundant `(dune (>= 3.15))` to one package's depends -- the exact shape backlog-213 removed, and the one a contributor would restore by copying a neighbouring project's dune-project. This is the recurrence this gate exists for.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "  (ocaml (>= 5.4.0))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))"
#   apply: assert s.count(old) == 1, ("spoc depends anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "  (ocaml (>= 5.4.0))\n  (dune (>= 3.15))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))")
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: declares `dune` in its `depends`
#
# mutation: explicit-dune-dep-wrapped-over-two-lines
#   desc: the same declaration wrapped across two lines. `dune fmt` produces this shape unattended once the entry is long enough, and a line-oriented pattern walks straight past it -- the first version of this gate did. Dune honours it identically, so the divergence is identical.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "  (ocaml (>= 5.4.0))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))"
#   apply: assert s.count(old) == 1, ("spoc depends anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "  (ocaml (>= 5.4.0))\n  (dune\n   (>= 3.15))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))")
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: declares `dune` in its `depends`
#
# mutation: explicit-dune-dep-as-a-bare-atom
#   desc: `dune` as a bare atom with no constraint at all. Legal, redundant, and it carries no parenthesis for a pattern to anchor on. Rule 1's contract is the NAME, not one argument shape, and this pins that.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "  (ocaml (>= 5.4.0))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))"
#   apply: assert s.count(old) == 1, ("spoc depends anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "  (ocaml (>= 5.4.0))\n  dune\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))")
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: declares `dune` in its `depends`
#
# mutation: duplicated-bound-committed
#   desc: commit the duplicated-bound artefact itself into a tracked .opam. Rule 1 alone stays green here -- dune-project is untouched -- so this is the half that catches a regenerated file committed from a tree fixed afterwards, or a hand-edit.
#   apply: python3 - <<'PYEOF'
#   apply: p = "sarek.opam"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = '  "dune" {>= "3.15"}\n'
#   apply: assert s.count(old) == 1, ("dune constraint anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, '  "dune" {>= "3.15" & >= "3.15"}\n')
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: carries a multi-term "dune" constraint
#
# mutation: duplicated-bound-committed-across-lines
#   desc: the same artefact wrapped by an opam formatter. Rule 2 reads a brace-balanced constraint rather than a line, and this is what pins that -- the first version of this gate grepped one physical line and would have passed.
#   apply: python3 - <<'PYEOF'
#   apply: p = "sarek.opam"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = '  "dune" {>= "3.15"}\n'
#   apply: assert s.count(old) == 1, ("dune constraint anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, '  "dune" {>= "3.15" &\n           >= "3.15"}\n')
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: carries a multi-term "dune" constraint
#
# mutation: lang-stanza-deleted
#   desc: delete the lang stanza so the version cannot be read. Rule 1 is conditional on that number, and a gate that cannot read its own condition must REFUSE rather than take the branch that checks less -- the vacuous-green shape this repository keeps finding.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "(lang dune 3.15)\n"
#   apply: assert s.count(old) == 1, ("lang stanza not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "", 1)
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: found 0 of them
#
# mutation: lang-stanza-duplicated
#   desc: the OTHER side of the exactly-one guard, and the reason it is a count rather than a `head -1`. Two lang stanzas cannot both be the language version, and agreeing with whichever comes first is how the wrong one survives. Deleting the stanza alone would not pin this.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "(lang dune 3.15)\n"
#   apply: assert s.count(old) == 1, ("lang stanza not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "(lang dune 3.15)\n(lang dune 3.24)\n", 1)
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: found 2 of them
#
# mutation: one-opam-file-declares-no-dune
#   desc: strip the `"dune"` line from ONE .opam of seven. An at-least-one guard passes this happily and reports success having examined six -- measured at 1-of-7 against an earlier revision of this gate. Dune injects the dependency into EVERY generated package, so the guard has to be all-of-them or its own refusal text is wider than it is.
#   apply: python3 - <<'PYEOF'
#   apply: import re
#   apply: p = "sarek-hip.opam"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: s2, k = re.subn(r'^[ \t]*"dune" \{[^}]*\}\n', "", s, flags=re.M)
#   apply: assert k == 1, ("expected exactly one dune constraint in %s, stripped %d" % (p, k))
#   apply: open(p, "w", encoding="utf-8").write(s2)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: 6 of 7 declare one
#
# mutation: escape-spelled-dune-name
#   desc: declare the dependency as `("\x64une" (>= 3.15))`. Dune decodes the escape and reads it as `dune`; a reader that skipped escapes instead of decoding them read `x64une` and passed. Rule 1's contract is "in ANY form", and a contract that a two-character escape defeats is not that.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "  (ocaml (>= 5.4.0))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))"
#   apply: assert s.count(old) == 1, ("spoc depends anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, '  (ocaml (>= 5.4.0))\n  ("\\x64une" (>= 3.15))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))')
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 1
#   expect-message: declares `dune` in its `depends`
#
# mutation: stray-close-paren
#   desc: a lone ')' at top level. An earlier revision's atom scanner made no progress on it and the gate LOOPED FOREVER -- measured at exit 124 under a 10s timeout. A hang is worse than a wrong answer: in CI it burns the job timeout and reports nothing at all, so this pins the refusal rather than the diagnosis.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: assert not s.endswith(")\n)\n"), "fixture already ends with a stray paren"
#   apply: open(p, "w", encoding="utf-8").write(s + ")\n")
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: unexpected ')'
#
# mutation: unterminated-string
#   desc: a quoted string that never closes. Dune rejects the file; an earlier revision ran off the end, silently produced a garbage atom, and exited 0 -- a verdict about a file that does not parse.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: open(p, "w", encoding="utf-8").write(s + '"unterminated\n')
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: ends inside a quoted string
#
# mutation: malformed-numeric-escape
#   desc: a truncated hex escape, `"\xZZ"`. int() raised an unguarded ValueError, so python left with exit 1 -- this gate's code for "the tree is wrong" -- on input dune cannot read, which the contract puts at exit 2. The adjacent unknown-escape branch already refused correctly, which is what made this one inconsistent rather than merely missing. CodeRabbit, PR #399.
#   apply: python3 - <<'PYEOF'
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "  (ocaml (>= 5.4.0))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))"
#   apply: assert s.count(old) == 1, ("spoc depends anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, '  (ocaml (>= 5.4.0))\n  ("\\xZZ" (>= 3.15))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))')
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: malformed numeric string escape
#
# mutation: depends-field-with-no-bracket
#   desc: an opam `depends:` not followed by '['. text.index() raised an unguarded ValueError and left exit 1, same shape and same contract violation as the escape above, and beside an unclosed-bracket branch that already refused correctly. CodeRabbit, PR #399.
#   apply: python3 - <<'PYEOF'
#   apply: p = "sarek-vulkan.opam"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: i = s.index("\ndepends: [")
#   apply: j = s.index("\n]", i)
#   apply: s = s[:i] + "\ndepends:" + s[j + 2:]
#   apply: assert "\ndepends:\n" in s, "rewrite did not produce a bracketless depends:"
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: not followed by a '['
#
# mutation: rule1-finding-not-lost-behind-a-later-refusal
#   desc: a real rule-1 violation AND a cannot-measure condition at once. `bad()` only queues and the queue was printed at the very end, so the refusal exited 2 having said nothing about the explicit `dune` dependency -- the operator is told about the missing file and not about the bug. Exit 2 is right; losing the finding is not, so the expected message is the rule-1 one.
#   apply: python3 - <<'PYEOF'
#   apply: import os
#   apply: p = "dune-project"
#   apply: s = open(p, encoding="utf-8").read()
#   apply: old = "  (ocaml (>= 5.4.0))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))"
#   apply: assert s.count(old) == 1, ("spoc depends anchor not unique: %d" % s.count(old))
#   apply: s = s.replace(old, "  (ocaml (>= 5.4.0))\n  (dune (>= 3.15))\n  (ctypes (>= 0.24.0))\n  (bisect_ppx :with-test)))")
#   apply: open(p, "w", encoding="utf-8").write(s)
#   apply: assert os.path.exists("sarek-metal.opam"), "fixture missing before mutation"
#   apply: os.remove("sarek-metal.opam")
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: declares `dune` in its `depends`
#
# mutation: declared-package-has-no-opam-file
#   desc: remove a declared package's .opam from the scope rule 2 walks. The gate must refuse rather than silently check a smaller set -- an empty-set green with a smaller denominator is the same defect as an empty-set green.
#   apply: python3 - <<'PYEOF'
#   apply: import os
#   apply: p = "sarek-metal.opam"
#   apply: assert os.path.exists(p), "fixture missing before mutation"
#   apply: os.remove(p)
#   apply: assert not os.path.exists(p), "removal did not land"
#   apply: PYEOF
#   expect-exit: 2
#   expect-message: declares package(s) with no .opam file at the repository root
# END prove-red-spec
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# The whole check is one parse of two file formats, so it lives in python
# rather than in a pile of greps. The first version of this gate WAS a pile of
# greps and three separate reviews each found a different thing it did not
# match -- `(dune :build)`, a `dune fmt`-wrapped `(dune\n (>= 3.15))`, and a
# constraint broken across lines by an opam formatter -- while its comment
# claimed it matched "(dune ...)". A tokenizer states the same contract without
# the gap between the sentence and the pattern.
#
# Held in a variable rather than on python's stdin, so this script never
# consumes the stdin it was handed.
PYPROG=$(cat <<'PYEOF'
import os
import sys

DUNE_PROJECT = "dune-project"
# Dedup became lang-gated in dune 3.23.1; the gate opens at lang 3.23.
DEDUP_LANG = (3, 23)

violations = []


def refuse(msg):
    # Flush anything already found before leaving. `bad()` only queues, and the
    # queue is printed at the very end; without this, a rule-1 violation
    # discovered first is thrown away by any later cannot-measure refusal, and
    # the operator is told about the missing .opam file but not about the
    # explicit `dune` dependency that is the whole point of the gate. Exit 2 is
    # still right -- the tree could not be fully measured -- but a finding
    # already in hand is evidence, not noise.
    for queued in violations:
        sys.stderr.write("check-dune-opam-portability: %s\n" % queued)
    if violations:
        sys.stderr.write("check-dune-opam-portability: (the finding(s) above "
                         "were already established; the refusal below is why "
                         "the rest could not be checked)\n")
    sys.stderr.write("check-dune-opam-portability: %s\n" % msg)
    sys.exit(2)


def bad(msg):
    violations.append(msg)


# --- a dune-lang s-expression reader ---------------------------------------
# Atoms, quoted strings, lists, and `;` line comments. That is the whole of
# dune-project's surface syntax for the purposes of these two questions.
def sexps(text):
    pos, n = 0, len(text)

    def parse_list():
        nonlocal pos
        out = []
        pos += 1  # consume '('
        while True:
            skip_ws()
            if pos >= n:
                refuse("%s ended inside an unclosed '(' -- it is not readable "
                       "as dune s-expressions, so neither rule can be decided."
                       % DUNE_PROJECT)
            if text[pos] == ")":
                pos += 1
                return out
            out.append(parse_one())

    # Escapes are DECODED, not just skipped over. A `\` followed by the escaped
    # character verbatim would read `"\x64une"` as `x64une` and let a quoted,
    # escape-spelled `dune` dependency through -- while rule 1's contract says
    # "in ANY form". Dune uses OCaml's lexical conventions here: the named
    # escapes, `\ddd` decimal, `\xHH` hex, `\o###` octal, and `\<newline>` as a
    # line continuation that also swallows the following indentation.
    SIMPLE = {"n": "\n", "t": "\t", "b": "\b", "r": "\r",
              "\\": "\\", '"': '"', "'": "'", " ": " "}

    def parse_string():
        nonlocal pos
        start_pos = pos
        pos += 1
        buf = []
        while True:
            if pos >= n:
                refuse("%s ends inside a quoted string opened at offset %d. "
                       "Dune rejects the file outright, so this gate has no "
                       "readable input and must not report a verdict about it."
                       % (DUNE_PROJECT, start_pos))
            c = text[pos]
            if c == '"':
                pos += 1
                return "".join(buf)
            if c != "\\":
                buf.append(c)
                pos += 1
                continue
            if pos + 1 >= n:
                refuse("%s ends with a trailing backslash inside a quoted "
                       "string." % DUNE_PROJECT)
            e = text[pos + 1]
            if e in SIMPLE:
                buf.append(SIMPLE[e])
                pos += 2
            elif e == "\n":
                pos += 2
                while pos < n and text[pos] in " \t":
                    pos += 1
            elif e.isdigit() or e in "xo":
                # Truncated or non-digit numeric escapes ("\xZZ", a "\x" at
                # EOF, "\12" with no third digit) make int() raise. An
                # unhandled ValueError leaves python with exit 1 -- this gate's
                # code for "the tree is wrong" -- for input dune cannot read,
                # which the contract says is exit 2. The adjacent unknown-escape
                # branch below already refuses correctly; these were the
                # inconsistent ones. CodeRabbit, PR #399.
                if e == "x":
                    digits, base, width = text[pos + 2:pos + 4], 16, 4
                elif e == "o":
                    digits, base, width = text[pos + 2:pos + 5], 8, 5
                else:
                    digits, base, width = text[pos + 1:pos + 4], 10, 4
                try:
                    code = int(digits, base)
                except ValueError:
                    refuse("%s contains a malformed numeric string escape "
                           "'\\%s%s' at offset %d. Dune rejects it; so does "
                           "this gate, rather than guess at a decoding."
                           % (DUNE_PROJECT, e if e in "xo" else "", digits,
                              pos))
                buf.append(chr(code & 0xFF) if base != 16 else chr(code))
                pos += width
            else:
                # Dune errors on an unknown escape. Refusing matches it, and
                # guessing would be this gate inventing a reading dune does not
                # have.
                refuse("%s contains an unknown string escape '\\%s' at offset "
                       "%d. Dune rejects it; so does this gate, rather than "
                       "guess at a decoding." % (DUNE_PROJECT, e, pos))

    def parse_atom():
        nonlocal pos
        start = pos
        while pos < n and text[pos] not in "()\"; \t\r\n":
            pos += 1
        if pos == start:
            # Only reachable on a character parse_one dispatches here that an
            # atom cannot start with -- in practice a stray ')'. Without this
            # the scanner makes no progress and the gate HANGS, which is worse
            # than a wrong answer because nothing reports it at all.
            refuse("%s has an unexpected '%s' at offset %d -- it is not "
                   "readable as dune s-expressions, so neither rule can be "
                   "decided." % (DUNE_PROJECT, text[pos], pos))
        return text[start:pos]

    def parse_one():
        if text[pos] == "(":
            return parse_list()
        if text[pos] == '"':
            return parse_string()
        return parse_atom()

    def skip_ws():
        nonlocal pos
        while pos < n:
            if text[pos] in " \t\r\n":
                pos += 1
            elif text[pos] == ";":
                while pos < n and text[pos] != "\n":
                    pos += 1
            else:
                return

    forms = []
    while True:
        skip_ws()
        if pos >= n:
            return forms
        forms.append(parse_one())


try:
    with open(DUNE_PROJECT, encoding="utf-8") as fh:
        project_text = fh.read()
except OSError as exc:
    refuse("cannot read %s: %s" % (DUNE_PROJECT, exc))

forms = sexps(project_text)

# --- the lang version, exactly once -----------------------------------------
# Counted, not taken-the-first-of. Two lang stanzas cannot both be the language
# version, and agreeing with whichever comes first is how the wrong one
# survives.
langs = [f for f in forms
         if isinstance(f, list) and len(f) >= 3
         and f[0] == "lang" and f[1] == "dune"]
if len(langs) != 1:
    refuse("expected exactly one '(lang dune X.Y)' stanza in %s, found %d of "
           "them. Rule 1 is conditional on that version and cannot be decided "
           "without exactly one." % (DUNE_PROJECT, len(langs)))

lang_str = langs[0][2]
try:
    lang = tuple(int(p) for p in lang_str.split("."))
except ValueError:
    refuse("could not read a version out of '(lang dune %s)' in %s."
           % (lang_str, DUNE_PROJECT))

# Dune 4 has not shipped, so whether it still honours this gating is unknown. Going
# INERT on an unverified assumption is exactly the silent-deactivation this
# repository hunts; refuse and make someone re-read dune#14436 instead.
if lang[0] > DEDUP_LANG[0]:
    refuse("%s is at (lang dune %s), a major version this gate has never been "
           "validated against. Rule 1's inertness above %d.%d rests on dune "
           "#14436, which is a dune 3.x change; re-read it against dune %d and "
           "update DEDUP_LANG here before trusting either branch."
           % (DUNE_PROJECT, lang_str, DEDUP_LANG[0], DEDUP_LANG[1], lang[0]))

dedup_guaranteed = lang >= DEDUP_LANG

# --- rule 1: no explicit `dune` dependency below the dedup lang gate --------
# The NAME, in any shape: `(dune (>= 3.15))`, `(dune :build)`, `(dune)`, or a
# bare `dune` atom, on one line or wrapped over several. Rule 1's contract is
# that the declaration must not exist, not that one spelling of it must not.
def dep_name(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list) and entry and isinstance(entry[0], str):
        return entry[0]
    return None


packages = []
explicit = []
for form in forms:
    if not (isinstance(form, list) and form and form[0] == "package"):
        continue
    name = None
    depends = []
    for field in form[1:]:
        if isinstance(field, list) and field and field[0] == "name":
            name = field[1] if len(field) > 1 else None
        elif isinstance(field, list) and field and field[0] == "depends":
            depends = field[1:]
    if name is None:
        refuse("a (package ...) stanza in %s has no (name ...) field, so the "
               "set of packages rule 2 must walk cannot be established."
               % DUNE_PROJECT)
    packages.append(name)
    for entry in depends:
        if dep_name(entry) == "dune":
            explicit.append(name)

if not packages:
    refuse("%s declares no (package ...) stanzas. Both rules are about this "
           "project's packages, and there are none to be about."
           % DUNE_PROJECT)

# `--list-packages` publishes the declared package names and stops. It exists so
# scripts/check-opam-clean.sh can PIN the set of generated .opam files it speaks
# for against dune-project, instead of trusting a glob or `git ls-files` -- both
# of which shrink silently when a file is deleted or untracked, and a verdict
# over a set that quietly got smaller is the shape this repository keeps
# finding. One parser, one answer, no second copy of the package list.
if "--list-packages" in sys.argv[1:]:
    for name in packages:
        print(name)
    sys.exit(0)

if not dedup_guaranteed and explicit:
    bad("%s is at (lang dune %s), below %d.%d, and declares `dune` in its "
        "`depends`: %s."
        % (DUNE_PROJECT, lang_str, DEDUP_LANG[0], DEDUP_LANG[1],
           ", ".join(sorted(set(explicit)))))
    bad("  Dune injects \"dune\" {>= \"%s\"} into every generated .opam from "
        "the (lang dune %s) stanza already, so the declaration is redundant "
        "AND it makes the generated output depend on which dune ran: 3.23"
        % (lang_str, lang_str))
    bad("  3.23.0 deduplicates the identical bounds; 3.23.1 and later do not "
        "below (lang dune %d.%d) and emit {>= \"%s\" & >= \"%s\"}. Delete the "
        "entry -- do not bump the lang version to make it collapse; see the "
        "header of this script for why. backlog-213."
        % (DEDUP_LANG[0], DEDUP_LANG[1], lang_str, lang_str))

# --- rule 2: no generated .opam carries a multi-term `dune` constraint ------
# Scope is exactly the packages dune-project declares, resolved to <name>.opam
# at the repository root. Stated, not globbed: `*.opam` would also be satisfied
# by a shrinking set, and `.pending-opam/` holds tracked .opam files that are
# not this dune-project's output at all.
missing = [p for p in packages if not os.path.isfile(p + ".opam")]
if missing:
    refuse("%s declares package(s) with no .opam file at the repository root: "
           "%s. Rule 2's scope is the declared packages; checking the ones "
           "that happen to be present would be reporting success about a set "
           "this gate had let shrink." % (DUNE_PROJECT, ", ".join(missing)))


def depends_section(text, path):
    """The body of the opam `depends: [ ... ]` field, bracket-balanced and
    string-aware, or None when the file has no such field.

    Scoped rather than searched. A whole-file scan for `"dune"` also reaches
    the `build:` field, where dune emits `["dune" "subst"] {dev}` and
    `["dune" "build" "-p" name ...]`; the `{dev}` there is a command filter and
    has nothing to do with the dependency bound. Today those happen to sit
    AFTER `depends:` and happen not to be followed directly by `{`, so a
    first-match scan lands on the right token by luck. Luck is not a scope."""
    key = "\ndepends:"
    at = text.find(key) if not text.startswith("depends:") else 0
    if at < 0:
        return None
    # The '[' must be the next non-whitespace thing after `depends:`, not merely
    # present SOMEWHERE later in the file. `text.index("[", at)` raised an
    # unguarded ValueError (exit 1, this gate's "the tree is wrong", on input it
    # simply cannot read) and the obvious repair -- `find` plus a `< 0` check --
    # is worse than it looks: every opam file has a `build: [` further down, so
    # the search always succeeds and silently parses the BUILD list as the
    # depends list. prove-red caught that: the mutation exited 2 with an
    # unrelated message, which is the harness refusing to credit a guard that
    # cannot be reached. Bounding the search is what makes it reachable.
    # CodeRabbit flagged the exception; the reachability was found proving it.
    cur = at + len(key) if at else len("depends:")
    while cur < len(text) and text[cur] in " \t\r\n":
        cur += 1
    if cur >= len(text) or text[cur] != "[":
        refuse("%s has a `depends:` field that is not followed by a '[', so it "
               "is not readable as an opam dependency list and rule 2 cannot "
               "be decided for it." % path)
    depth, start = 0, cur
    in_string = False
    while cur < len(text):
        c = text[cur]
        if in_string:
            if c == "\\":
                cur += 2
                continue
            if c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start:cur + 1]
        cur += 1
    refuse("%s has a `depends:` field whose '[' is never closed." % path)


def dune_constraints(section):
    """Every brace-delimited constraint attached to a `"dune"` entry in the
    given depends section. A list, not a first match: two `dune` entries would
    otherwise let the second one through unread."""
    out = []
    key = '"dune"'
    at = 0
    while True:
        at = section.find(key, at)
        if at < 0:
            return out
        cur = at + len(key)
        while cur < len(section) and section[cur] in " \t\r\n":
            cur += 1
        if cur < len(section) and section[cur] == "{":
            depth, start = 0, cur
            while cur < len(section):
                if section[cur] == "{":
                    depth += 1
                elif section[cur] == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(section[start:cur + 1])
                        break
                cur += 1
            else:
                refuse("a \"dune\" constraint in a depends: section is never "
                       "closed.")
        else:
            # `"dune"` with no constraint at all: still a declared dependency,
            # recorded as such with an empty constraint so the all-of-them
            # guard below counts it.
            out.append("")
        at = cur


declared = []
for pkg in packages:
    path = pkg + ".opam"
    try:
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
    except OSError as exc:
        refuse("cannot read %s: %s" % (path, exc))
    section = depends_section(body, path)
    if section is None:
        refuse("%s has no `depends:` field. Every dune-generated package has "
               "one -- dune injects a `dune` dependency into it -- so this "
               "file is either not dune-generated or not readable as opam, and "
               "rule 2 cannot be decided for it." % path)
    constraints = dune_constraints(section)
    if not constraints:
        continue
    declared.append(pkg)
    for constraint in constraints:
        # More than one term: `&` (and) or `|` (or) joining them. One bound is
        # what dune injects; anything more came from somewhere else. The
        # comparison operators dune emits (`>=`, `=`, `<`, `!=`) contain
        # neither character, so this does not fire on a single bound.
        if "&" in constraint or "|" in constraint:
            flat = " ".join(constraint.split())
            bad("%s carries a multi-term \"dune\" constraint: %s"
                % (path, flat))
            bad("  Exactly one bound is what dune injects from the lang "
                "stanza; a second term means a `dune` entry was declared in "
                "`depends`. Under (lang dune <%d.%d) every dune except 3.23.0 "
                "emits the duplicated-bound shape for an explicit "
                "(dune (>= X)) -- 3.23.0 alone deduplicates it unconditionally "
                "-- while an explicit (dune :flag) produces a multi-term "
                "constraint at every dune version."
                % (DEDUP_LANG[0], DEDUP_LANG[1]))
            bad("  Fix %s (rule 1) and regenerate; do not hand-edit this "
                "generated file." % DUNE_PROJECT)

# Dune injects the dependency into EVERY generated package, so the guard is
# all-of-them. An at-least-one guard would report success having examined one
# file in seven -- measured at exactly that against an earlier revision.
if len(declared) != len(packages):
    refuse("read %d declared package .opam file(s) and only %d of %d declare "
           "one: %s. Dune injects a \"dune\" dependency into every generated "
           "package, so either these files are not all dune-generated or the "
           "shape this gate looks for has gone stale. Rule 2 examined %d of "
           "%d."
           % (len(packages), len(declared), len(packages),
              ", ".join(sorted(set(packages) - set(declared))),
              len(declared), len(packages)))

if violations:
    for v in violations:
        sys.stderr.write("check-dune-opam-portability: %s\n" % v)
    sys.exit(1)

if dedup_guaranteed:
    rule1 = ("rule 1 INERT ((lang dune %s) is at or above %d.%d, where every "
             "dune able to read this project deduplicates the bound)"
             % (lang_str, DEDUP_LANG[0], DEDUP_LANG[1]))
else:
    rule1 = ("rule 1 ACTIVE ((lang dune %s) is below %d.%d): none of the %d "
             "declared package(s) declares `dune` in `depends`"
             % (lang_str, DEDUP_LANG[0], DEDUP_LANG[1], len(packages)))

print("check-dune-opam-portability: OK — %s; rule 2: all %d declared "
      "package .opam file(s) at the repository root carry a single-term "
      "\"dune\" constraint." % (rule1, len(packages)))
PYEOF
)

command -v python3 > /dev/null 2>&1 || {
  echo "check-dune-opam-portability: no python3 on PATH — this gate parses dune-project and cannot run without it." >&2
  exit 2
}

# `--` so an argument can never be read as a python option. Only
# `--list-packages` is understood; anything else falls through to the checker,
# which ignores argv, so an unknown flag must be refused here rather than
# silently running the wrong mode.
case "${1-}" in
  "" | --list-packages) ;;
  *)
    echo "check-dune-opam-portability: unknown argument '$1' (only --list-packages is accepted)." >&2
    exit 2 ;;
esac

python3 -c "$PYPROG" -- ${1+"$1"}
