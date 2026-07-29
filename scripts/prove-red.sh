#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Mechanise "prove red before trusting green" (backlog-151, health-2026-07-27 P3).
#
# WHY THIS EXISTS
#
# kb/properties.md states the rule this repository learned the expensive way:
#
#   > A checker is not evidence until it has been mutated and observed to fail
#   > WITH THE MESSAGE IT PROMISES. A positive control is not optional --
#   > without it, "went red" and "is always red" are the same observation.
#
# That rule is currently held by a habit. The 2026-07-27 skill-health report
# counted 12 new `gate-vacuous` instances in one session and 9 of them were
# caught only because somebody happened to remember. `check-kb-properties.sh`
# enforces that every gate DECLARES whether it has a red-path test, which is a
# real and load-bearing check -- but a declared red-path test that asserts
# nothing satisfies it perfectly. This script closes that last step: it applies
# a declared mutation and watches the gate fail.
#
# WHAT A SUBJECT DECLARES, AND WHERE
#
# Beside itself, in a comment block delimited by two marker lines whose stripped
# text is exactly `BEGIN prove-red-spec` and `END prove-red-spec`. Leading `#` or
# `//` and indentation are stripped, so the block is a comment in shell, JS and
# Python alike. Keys, one per line, `key: value`, values single-line. There is no
# continuation syntax on purpose: a continuation line and a mistyped key are the
# same two tokens, and treating an unrecognised key as prose is how a
# declaration ends up not executed while looking like it is. Repeat the key
# instead (`apply:` and `copy:` are repeatable) or write one long line.
#
#   HEADER (before the first `mutation:`)
#     copy:              path relative to the repo root, file or directory,
#                        copied into the scratch tree at the same relative
#                        path. Repeatable; at least one required. This is the
#                        subject's declared world -- if the checker reads
#                        something not listed here it will not find it.
#     invoke:            argv, whitespace-split, run with cwd = scratch root.
#                        Exactly one.
#     baseline-argv:     appended to `invoke` for the positive control.
#     baseline-stdin:    `empty` or `file:PATH` (PATH inside the scratch).
#                        Default: an empty pipe (never a terminal, never
#                        inherited).
#     baseline-exit:     exact exit code of the UNMUTATED run. Exactly one.
#     baseline-message:  substring the unmutated run must print. Exactly one.
#
#   PER MUTATION (a `mutation: <id>` line opens one)
#     desc:              one line saying which real shape this mutation is.
#     apply:             shell, run with `bash -euo pipefail` and cwd = scratch
#                        root. Repeatable; the lines are joined into one script,
#                        so variables set on one line are visible on the next.
#     stdin:             overrides baseline-stdin for this run.
#     argv:              REPLACES baseline-argv for this run.
#     expect-exit:       exact exit code. Not "non-zero" -- a contract that
#                        names 2 for a broken declaration and 1 for a real
#                        violation is not satisfied by "something failed", and
#                        an assertion that accepts either is a weakened
#                        assertion (health-2026-07-27, P3's own sub-case).
#     expect-message:    substring the mutated run must print.
#
# A mutation is therefore NOT limited to editing a source file, and that is the
# point: the defects that motivated this were a deleted root directory
# (`add-license-headers.sh`), an absent tool, and an empty stdin
# (`test-suite-counts.sh`). `apply:` is arbitrary shell over the whole scratch
# tree, and `stdin:`/`argv:` mutate the invocation, so the environment is as
# mutable as the source. There is no example block in this header on purpose:
# two blocks in one file is exit 2 (see below), and a format documented by a
# live instance cannot rot. Read `scripts/check-test-alias-coverage.sh`.
#
# WHAT STOPS THIS FROM BEING A SECOND GREEN
#
# A tool that runs mutations and prints "all checkers went red" is itself a
# checker, and the obvious implementation of it is exactly the shape it exists
# to police. Four things, none optional:
#
#   1. THE BASELINE IS MANDATORY AND MUST CARRY ITS MESSAGE. Before any
#      mutation, the subject runs unmutated and must exit `baseline-exit` AND
#      print `baseline-message`. A subject that is red on arrival, or that
#      passes silently, is exit 2 here -- because for such a subject "went red
#      under mutation" and "is always red" are indistinguishable, which is the
#      sentence at the top of this file.
#   2. A MUTATION MUST ACTUALLY MUTATE. After `apply:` runs, the scratch tree
#      is re-fingerprinted (path, mode, sha256, symlink-ness). If nothing
#      changed and the mutation overrides neither stdin nor argv, that is exit 2
#      -- a no-op mutation would let a subject collect a red it did not earn.
#      An `apply:` that itself fails is exit 2 for the same reason.
#   3. DECLARED == EXECUTED, AND FOUND == EXPECTED. Refusing to report success
#      over a spec this script did not fully run is copied from
#      check-kb-properties.sh; the subject-count pin is the `EXPECTED_PROJECTS`
#      / `EXPECTED_CHECKS` idiom already used by check-formal-proofs.sh and
#      ci/assert-toolchain.sh. A subject silently dropping out of the scan is
#      how a strict check becomes a permissive one.
#   4. IT IS ITS OWN SUBJECT. This script carries a prove-red-spec block whose
#      mutations break a committed fixture gate in the four ways this tool can
#      lie -- an immune checker, a checker red on arrival, a subject that
#      vanishes, and a declared message the gate never prints -- and require
#      this script to report each one. `scripts/prove-red.test.sh` is the
#      covering test that asserts those same four shapes with exact exit codes.
#
# The recursion stops there, and it is worth saying where: prove-red.test.sh has
# no red-path of its own. It is a test with exact assertions rather than a gate
# with a coverage set, its failures are its output, and one more turtle would be
# a test asserting that a test can fail. That is the same line kb/properties.md
# draws with KB-GATE-SELF.
#
# WHAT IT COVERS, AND WHAT IT DELIBERATELY DOES NOT
#
# Subjects are discovered by the presence of the block, so coverage is opt-in
# and grows one deliberate edit at a time; `EXPECTED_SUBJECTS` below is what
# stops it shrinking. This is declaration-completeness over
# coverage-completeness, the same trade kb/properties.md argues for
# `gate-red-path`: the strong version ("every checker with no declared mutation
# is a failure") would be red on arrival across a dozen gates today, and a gate
# that is red on arrival gets disabled rather than fixed. The uncovered ones and
# the reason for each are recorded in kb/properties.md, not in folklore.
#
# `*.test.sh` and `*.test.js` are never subjects: a covering test IS a red path,
# not a gate with one, and the fixture blocks inside prove-red.test.sh would
# otherwise be discovered as real declarations.
#
# Exit codes:
#   0  every declared mutation produced its declared failure
#   1  at least one mutation did NOT -- a checker that cannot fail, or that
#      fails with a different code or a different message than it promises
#   2  the mechanism could not produce evidence: a malformed or missing spec, a
#      subject red on arrival, a mutation that changed nothing or failed to
#      apply, a subject count that does not match the pin, a `copy:` target git
#      does not track (present here, absent on a fresh clone), a timed-out run.
#      Never a skip -- an unusable declaration reporting 0 would be this file's
#      own failure mode.
#
# Usage:
#   scripts/prove-red.sh [--root DIR] [--expect-subjects N] [-v]
#
# MEASURED COST: see the `prove-red` step in .github/workflows/ci.yml.
# ---------------------------------------------------------------------------
set -euo pipefail

# How many subjects the scan must find. Pinned, not counted: a scan that
# reports what it happened to find is complete about a set it chose. Raise it
# in the same commit that adds a block.
EXPECTED_SUBJECTS=5

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECT=""
VERBOSE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --root)
      [ $# -ge 2 ] || { echo "ERROR: --root needs a value" >&2; exit 2; }
      ROOT="$2"; shift 2 ;;
    --root=*) ROOT="${1#--root=}"; shift ;;
    --expect-subjects)
      [ $# -ge 2 ] || { echo "ERROR: --expect-subjects needs a value" >&2; exit 2; }
      EXPECT="$2"; shift 2 ;;
    --expect-subjects=*) EXPECT="${1#--expect-subjects=}"; shift ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help)
      cat <<'USAGE'
usage: scripts/prove-red.sh [--root DIR] [--expect-subjects N] [-v]

  --root DIR            tree to scan (default: this repository)
  --expect-subjects N   override the pinned subject count; used when running
                        against a fixture root

exit 0  every declared mutation produced its declared failure
exit 1  a checker did not fail as declared
exit 2  the mechanism could not produce evidence (malformed spec, subject red
        on arrival, no-op mutation, subject count mismatch)
USAGE
      exit 0 ;;
    -*) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    *)  echo "ERROR: unexpected argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$EXPECT" ] || EXPECT="$EXPECTED_SUBJECTS"
case "$EXPECT" in
  ''|*[!0-9]*) echo "ERROR: --expect-subjects must be a non-negative integer, got: $EXPECT" >&2; exit 2 ;;
esac

[ -d "$ROOT" ] || { echo "::error::--root $ROOT is not a directory."; exit 2; }
ROOT="$(cd "$ROOT" && pwd)"

# The parser lives in a variable rather than on python's stdin. Same reason
# test-suite-counts.sh does it: `python3 - <<EOF` hands python its program on
# stdin, and this script hands stdin to the subjects it runs.
PYPROG=$(cat <<'PYEOF'
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = sys.argv[1]
EXPECT_SUBJECTS = int(sys.argv[2])
VERBOSE = sys.argv[3] == "1"

SCAN_DIRS = ["scripts", "ci"]
BEGIN = "BEGIN prove-red-spec"
END = "END prove-red-spec"
TIMEOUT = 300

HEADER_KEYS = {"copy", "invoke", "baseline-argv", "baseline-stdin",
               "baseline-exit", "baseline-message"}
MUT_KEYS = {"desc", "apply", "stdin", "argv", "expect-exit", "expect-message"}


def die(msg):
    """Exit 2. The mechanism could not produce evidence; this is never a skip."""
    print("::error::%s" % msg)
    sys.exit(2)


def strip_comment(line):
    s = line.strip()
    for marker in ("#", "//"):
        if s.startswith(marker):
            s = s[len(marker):]
            if s.startswith(" "):
                s = s[1:]
            return s
    return s


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover():
    found, roots_present = [], 0
    for d in SCAN_DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        roots_present += 1
        for name in sorted(os.listdir(full)):
            path = os.path.join(full, name)
            if not os.path.isfile(path):
                continue
            # A covering test is a red path, not a gate that has one.
            if name.endswith(".test.sh") or name.endswith(".test.js"):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if any(strip_comment(l) == BEGIN for l in text.split("\n")):
                found.append((os.path.join(d, name), text))
    if roots_present == 0:
        die("none of %s exists under %s, so the scan read nothing. An empty "
            "scan set turns this into a check that always passes."
            % (", ".join(SCAN_DIRS), ROOT))
    return found


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------
def parse(rel, text):
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if strip_comment(l) == BEGIN]
    ends = [i for i, l in enumerate(lines) if strip_comment(l) == END]
    if len(starts) != 1 or len(ends) != 1:
        die("%s carries %d `%s` and %d `%s` marker(s); exactly one of each is "
            "allowed. With two blocks, which one is authoritative is undefined "
            "and a declaration can hide in the one nothing reads."
            % (rel, len(starts), BEGIN, len(ends), END))
    if ends[0] < starts[0]:
        die("%s: `%s` appears before `%s`." % (rel, END, BEGIN))

    header, muts = {}, []
    cur = None
    for n in range(starts[0] + 1, ends[0]):
        raw = strip_comment(lines[n])
        if not raw.strip():
            continue
        if ":" not in raw:
            die("%s line %d: `%s` is not `key: value`." % (rel, n + 1, raw))
        key, _, val = raw.partition(":")
        key, val = key.strip(), val.strip()
        if key == "mutation":
            if not val:
                die("%s line %d: `mutation:` needs an id." % (rel, n + 1))
            if any(m["id"] == val for m in muts):
                die("%s line %d: duplicate mutation id `%s`. The id names the "
                    "mutation in the report, so two sharing one make it "
                    "ambiguous." % (rel, n + 1, val))
            cur = {"id": val, "apply": [], "line": n + 1}
            muts.append(cur)
            continue
        target, allowed = (header, HEADER_KEYS) if cur is None else (cur, MUT_KEYS)
        if key not in allowed:
            die("%s line %d: unknown key `%s` in the %s section. Known: %s. A "
                "typo'd key that was ignored would be a declaration nothing "
                "executes wearing the badge of one that passed."
                % (rel, n + 1, key, "header" if cur is None else "mutation",
                   ", ".join(sorted(allowed))))
        if key in ("copy", "apply"):
            target.setdefault(key, []).append(val)
        elif key in target and key != "id":
            die("%s line %d: `%s` given twice." % (rel, n + 1, key))
        else:
            target[key] = val

    for k in ("invoke", "baseline-exit", "baseline-message"):
        if not header.get(k):
            die("%s: spec block is missing required header key `%s`." % (rel, k))
    if not header.get("copy"):
        die("%s: spec block declares no `copy:`. A subject with no declared "
            "world would be run against an empty scratch tree and fail for the "
            "wrong reason." % rel)
    if not muts:
        die("%s: spec block declares no mutation. A checker with no declared "
            "mutation is not evidence -- it is an assertion about itself." % rel)

    for k in ("baseline-exit",):
        try:
            header[k] = int(header[k])
        except ValueError:
            die("%s: `%s` must be an integer, got %r." % (rel, k, header[k]))

    for m in muts:
        for k in ("desc", "expect-exit", "expect-message"):
            if not m.get(k):
                die("%s mutation `%s` (line %d): missing required key `%s`."
                    % (rel, m["id"], m["line"], k))
        try:
            m["expect-exit"] = int(m["expect-exit"])
        except ValueError:
            die("%s mutation `%s`: `expect-exit` must be an integer, got %r. "
                "\"non-zero\" is not a contract." % (rel, m["id"], m["expect-exit"]))
        if not m["apply"] and "stdin" not in m and "argv" not in m:
            die("%s mutation `%s`: declares no `apply:`, no `stdin:` and no "
                "`argv:`, so it changes nothing. A mutation that mutates "
                "nothing collects a red the subject did not earn."
                % (rel, m["id"]))
        if m["expect-exit"] == header["baseline-exit"] and m["expect-message"] \
                == header["baseline-message"]:
            die("%s mutation `%s`: expects the baseline's exit code AND the "
                "baseline's message, so it asserts the subject did not "
                "notice." % (rel, m["id"]))

    for k in ("baseline-stdin",):
        v = header.get(k)
        if v is not None and v != "empty" and not v.startswith("file:"):
            die("%s: `%s` must be `empty` or `file:PATH`, got %r." % (rel, k, v))
    for m in muts:
        v = m.get("stdin")
        if v is not None and v != "empty" and not v.startswith("file:"):
            die("%s mutation `%s`: `stdin` must be `empty` or `file:PATH`, got "
                "%r." % (rel, m["id"], v))

    return header, muts


# ---------------------------------------------------------------------------
# Scratch trees
# ---------------------------------------------------------------------------
def git_tracked():
    """The set of paths git tracks under ROOT, or None if ROOT is not a work
    tree (a synthetic root under test, for instance).

    An untracked `copy:` target verifies perfectly on the workstation that has
    it and is missing on a fresh clone -- the defect KB-GATE-BUNDLE-TRACKED
    exists for, and one this file walked straight into: the first
    test-suite-counts fixture was named `*.log` and .gitignore swallowed it."""
    try:
        inside = subprocess.run(["git", "-C", ROOT, "rev-parse",
                                 "--is-inside-work-tree"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if inside.returncode != 0 or inside.stdout.strip() != b"true":
        return None
    ls = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        timeout=120)
    if ls.returncode != 0:
        return None
    return {p.decode("utf-8", "replace")
            for p in ls.stdout.split(b"\0") if p}


TRACKED = git_tracked()


def build_scratch(rel, header, tmp):
    scratch = tempfile.mkdtemp(dir=tmp)
    for p in header["copy"]:
        src = os.path.join(ROOT, p)
        dst = os.path.join(scratch, p)
        if not os.path.exists(src):
            die("%s: declared `copy: %s` does not exist under %s. The subject "
                "would run against a world it did not ask for."
                % (rel, p, ROOT))
        if TRACKED is not None:
            norm = p.rstrip("/")
            if norm not in TRACKED and not any(
                    t.startswith(norm + "/") for t in TRACKED):
                die("%s: declared `copy: %s` is not tracked by git. It exists "
                    "here and would be absent on a fresh clone, so this "
                    "subject would be exit 2 in CI while passing on the "
                    "workstation that has the file. Commit it -- and check "
                    ".gitignore, which is how this usually happens."
                    % (rel, p))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst, follow_symlinks=False)
    return scratch


def fingerprint(d):
    """Path, mode, symlink-ness and content of every file. A mutation that
    leaves this unchanged mutated nothing."""
    out = []
    for dirpath, dirnames, filenames in os.walk(d):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, d)
            if os.path.islink(p):
                out.append((rel, "link", os.readlink(p)))
                continue
            try:
                with open(p, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
                mode = oct(os.stat(p).st_mode & 0o777)
            except OSError as exc:
                digest, mode = "unreadable:%s" % exc, "?"
            out.append((rel, mode, digest))
        for name in sorted(dirnames):
            out.append((os.path.relpath(os.path.join(dirpath, name), d), "dir", ""))
    return out


def stdin_bytes(spec, scratch, rel, what):
    if spec is None or spec == "empty":
        return b""
    path = os.path.join(scratch, spec[len("file:"):])
    if not os.path.isfile(path):
        die("%s: %s names `%s`, which is not in the scratch tree. Add it to "
            "`copy:` or create it in `apply:`." % (rel, what, spec))
    with open(path, "rb") as fh:
        return fh.read()


def run(argv, scratch, data, rel, what):
    try:
        p = subprocess.run(argv, cwd=scratch, input=data,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=TIMEOUT)
    except OSError as exc:
        # PermissionError as well as FileNotFoundError: a subject copied
        # without its executable bit would otherwise leave a traceback and
        # python's exit 1, which is this script's code for "a checker did not
        # fail as declared" -- a mechanism failure wearing a finding's badge.
        die("%s: cannot execute `%s` in the scratch tree (%s): %s. Is it listed "
            "in `copy:`, and is the committed file +x?"
            % (rel, argv[0], " ".join(argv), exc))
    except subprocess.TimeoutExpired:
        die("%s: %s did not finish within %ds. A checker that hangs is not "
            "evidence either." % (rel, what, TIMEOUT))
    return p.returncode, p.stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
subjects = discover()

if len(subjects) != EXPECT_SUBJECTS:
    names = "\n        ".join(s[0] for s in subjects) or "(none)"
    die("expected %d subject(s) carrying a prove-red-spec block under %s, "
        "found %d:\n        %s\n        A subject dropping out of the scan is "
        "how a strict check becomes a permissive one. If this is deliberate, "
        "change EXPECTED_SUBJECTS in scripts/prove-red.sh in the same commit."
        % (EXPECT_SUBJECTS, ", ".join(SCAN_DIRS), len(subjects), names))

violations = []
executed = 0
declared = 0
tmp = tempfile.mkdtemp(prefix="prove-red.")
try:
    for rel, text in subjects:
        header, muts = parse(rel, text)
        declared += len(muts)
        invoke = shlex.split(header["invoke"])
        base_argv = shlex.split(header.get("baseline-argv", ""))

        print("== %s" % rel)

        # -- positive control ------------------------------------------------
        scratch = build_scratch(rel, header, tmp)
        code, out = run(invoke + base_argv, scratch,
                        stdin_bytes(header.get("baseline-stdin"), scratch, rel,
                                    "baseline-stdin"),
                        rel, "the baseline run")
        if code != header["baseline-exit"]:
            print(out)
            die("%s: BASELINE is not green. The unmutated subject exited %d, "
                "declared %d. Every red below it would prove nothing -- \"went "
                "red\" and \"is always red\" are the same observation. Fix the "
                "subject, or its `copy:` set if the scratch world is wrong."
                % (rel, code, header["baseline-exit"]))
        if header["baseline-message"] not in out:
            print(out)
            die("%s: BASELINE exited %d as declared but never printed %r. A "
                "positive control that only checks an exit code passes for a "
                "subject that did nothing and said nothing."
                % (rel, code, header["baseline-message"]))
        print("   baseline: exit %d, says %r" % (code, header["baseline-message"]))
        shutil.rmtree(scratch)

        # -- mutations -------------------------------------------------------
        for m in muts:
            scratch = build_scratch(rel, header, tmp)
            before = fingerprint(scratch)
            if m["apply"]:
                script = "\n".join(m["apply"])
                try:
                    p = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                                       cwd=scratch, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    die("%s mutation `%s`: the `apply:` script did not finish "
                        "within %ds." % (rel, m["id"], TIMEOUT))
                if p.returncode != 0:
                    print(p.stdout.decode("utf-8", "replace"))
                    die("%s mutation `%s`: the `apply:` script exited %d. A "
                        "mutation that failed to apply proves nothing about "
                        "the subject." % (rel, m["id"], p.returncode))
                if fingerprint(scratch) == before and "stdin" not in m \
                        and "argv" not in m:
                    die("%s mutation `%s`: `apply:` ran successfully and "
                        "changed no file in the scratch tree. The subject "
                        "would be about to collect a red it did not earn."
                        % (rel, m["id"]))
            argv = invoke + (shlex.split(m["argv"]) if "argv" in m else base_argv)
            data = stdin_bytes(m.get("stdin", header.get("baseline-stdin")),
                               scratch, rel, "mutation `%s` stdin" % m["id"])
            code, out = run(argv, scratch, data, rel,
                            "mutation `%s`" % m["id"])
            executed += 1
            if code != m["expect-exit"]:
                if code == header["baseline-exit"]:
                    why = ("the subject DID NOT FAIL: it exited %d, exactly as "
                           "it does unmutated" % code)
                else:
                    why = ("the subject exited %d, declared %d. \"It failed\" "
                           "is not the contract; the code is"
                           % (code, m["expect-exit"]))
                violations.append((rel, m["id"], m["desc"], why, out))
                print("   [%s] FAIL -- %s" % (m["id"], why))
                shutil.rmtree(scratch)
                continue
            if m["expect-message"] not in out:
                why = ("the subject exited %d as declared, but its output never "
                       "mentioned %r -- so the failure it reports is not the "
                       "one declared here"
                       % (code, m["expect-message"]))
                violations.append((rel, m["id"], m["desc"], why, out))
                print("   [%s] FAIL -- %s" % (m["id"], why))
                shutil.rmtree(scratch)
                continue
            print("   [%s] red: exit %d, says %r"
                  % (m["id"], code, m["expect-message"]))
            if VERBOSE:
                print("        %s" % m["desc"])
                for line in out.rstrip("\n").split("\n"):
                    print("        | %s" % line)
            shutil.rmtree(scratch)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("")
if executed != declared:
    die("%d mutation(s) declared but %d executed. Refusing to report on a set "
        "this script did not fully run." % (declared, executed))
if executed == 0:
    die("0 mutations executed. A prover that proved nothing must not exit 0.")

if violations:
    print("::error::%d declared mutation(s) did not produce their declared "
          "failure." % len(violations))
    for rel, mid, desc, why, out in violations:
        print("    - %s [%s]: %s" % (rel, mid, why))
        print("      the mutation: %s" % desc)
        for line in out.rstrip("\n").split("\n")[-12:]:
            print("      | %s" % line)
    print("")
    print("A checker that survives its own declared mutation is not evidence. "
          "Either the checker cannot fail, or the declaration beside it "
          "describes a checker that no longer exists.")
    sys.exit(1)

print("OK: %d subject(s), %d mutation(s); every declared mutation produced its "
      "declared exit code and message, and every baseline was green."
      % (len(subjects), executed))
PYEOF
)

python3 -c "$PYPROG" "$ROOT" "$EXPECT" "$VERBOSE"

# ---------------------------------------------------------------------------
# BEGIN prove-red-spec
# copy: scripts/prove-red.sh
# copy: scripts/prove-red-fixtures
# invoke: scripts/prove-red.sh
# baseline-argv: --root scripts/prove-red-fixtures/root --expect-subjects 1
# baseline-exit: 0
# baseline-message: every declared mutation produced its declared exit code
#
# mutation: immune-checker
#   desc: the fixture gate is replaced by one that always exits 0 while still printing its success message -- the "gate that cannot fail" shape itself. This must be a finding (1), not an error (2).
#   apply: G=scripts/prove-red-fixtures/root/scripts/fixture-gate.sh
#   apply: sed -n "/BEGIN prove-red-spec/,/END prove-red-spec/p" "$G" > .blk
#   apply: printf '#!/usr/bin/env bash\necho "OK: input.txt is well-formed"\nexit 0\n' > "$G"
#   apply: cat .blk >> "$G"
#   apply: rm -f .blk
#   apply: chmod +x "$G"
#   argv: --root scripts/prove-red-fixtures/root --expect-subjects 1
#   expect-exit: 1
#   expect-message: DID NOT FAIL
#
# mutation: subject-red-on-arrival
#   desc: the fixture gate always fails, so every mutation would "go red" and none of those reds would mean anything. The positive control must stop the run before any of them is credited -- and as an error, not a finding.
#   apply: G=scripts/prove-red-fixtures/root/scripts/fixture-gate.sh
#   apply: sed -n "/BEGIN prove-red-spec/,/END prove-red-spec/p" "$G" > .blk
#   apply: printf '#!/usr/bin/env bash\necho "::error::input.txt is malformed"\nexit 1\n' > "$G"
#   apply: cat .blk >> "$G"
#   apply: rm -f .blk
#   apply: chmod +x "$G"
#   argv: --root scripts/prove-red-fixtures/root --expect-subjects 1
#   expect-exit: 2
#   expect-message: BASELINE is not green
#
# mutation: subject-vanishes
#   desc: the only subject in the fixture root is deleted. A scan that reports what it happens to find would say "0 subjects, 0 failures" and exit 0.
#   apply: rm -f scripts/prove-red-fixtures/root/scripts/fixture-gate.sh
#   argv: --root scripts/prove-red-fixtures/root --expect-subjects 1
#   expect-exit: 2
#   expect-message: found 0
#
# mutation: message-half-is-live
#   desc: the fixture's declared expect-message is changed to something its gate never prints, with the exit code left alone. Without this, expect-message would be decorative and a gate could satisfy its declaration by failing for an unrelated reason.
#   apply: G=scripts/prove-red-fixtures/root/scripts/fixture-gate.sh
#   apply: sed -i "s/^#   expect-message: does not carry MARKER-OK$/#   expect-message: a string this gate never prints/" "$G"
#   apply: grep -q "a string this gate never prints" "$G"
#   argv: --root scripts/prove-red-fixtures/root --expect-subjects 1
#   expect-exit: 1
#   expect-message: never mentioned
# END prove-red-spec
# ---------------------------------------------------------------------------
