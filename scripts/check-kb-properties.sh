#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Execute the machine-checkable declarations in kb/properties.md (backlog-100).
#
# WHY THIS EXISTS
#
# kb/properties.md carries a fenced `code-intel` block whose envelope (the fence
# tag and the one-JSON-object-per-line shape) is owned by the roster KB schema.
# The roster's own runner for it, `scripts/code-intel-resolve.js`, is NOT
# installed in this repository, and no `.agents/skills/*/SKILL.md` here declares
# `capability: code-intel`. roster-qa's documented behaviour in that situation is
# `RESULT: skip`.
#
# So a code-intel block added on its own would be a set of invariants that read
# as enforced and are executed by nothing -- in a repository whose single most
# expensive recurring defect is the gate that cannot fail. This script is the
# in-tree executor that closes that hole: it is tracked, it is wired into CI, and
# a fresh clone runs it without any roster install.
#
# WHAT IT DOES AND DOES NOT GUARANTEE
#
# `gate-red-path` enforces DECLARATION completeness, not COVERAGE completeness.
# A gate with no red-path test passes -- but only by naming, in kb/properties.md,
# that it has none and why. That is a deliberately weaker contract than "every
# gate is proven able to fail", and it is stated here rather than left for a
# reader to discover: the strong version would fail today on five pre-existing
# gates, and a gate that is red on arrival gets disabled rather than fixed.
# The part that is strong is `gate-inventory-complete`: a gate cannot be added
# to CI without a row here saying which of the two it is.
#
# Exit codes:
#   0  every declaration holds
#   1  at least one declaration is violated
#   2  the block or a declaration is MALFORMED, or kb/properties.md is missing
#      or carries no block. This is never a skip. An unparseable declaration
#      that exited 0 would be the exact shape this file exists to prevent.
# ---------------------------------------------------------------------------
#
# Usage:
#   scripts/check-kb-properties.sh [properties-file] [root]
#
# `root` defaults to the repository, and every path inside a declaration is
# resolved against it. The parameter exists so the covering test can point the
# real checker at a synthetic tree and mutate the *targets* -- proving each check
# goes red on a source change rather than only on a doctored declaration, which
# would leave the interesting half untested.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
PROPS="${1:-kb/properties.md}"
cd "$ROOT"

if [ ! -f "$PROPS" ]; then
  echo "::error::$PROPS is missing. The code-intel declarations it carries are" \
       "this checker's entire input, so there is nothing to verify and a pass" \
       "would mean nothing."
  exit 2
fi

python3 - "$PROPS" <<'PY'
import json
import os
import re
import sys

props_path = sys.argv[1]
text = open(props_path, encoding="utf-8").read()

# ---------------------------------------------------------------------------
# Envelope extraction. Exactly one ```code-intel fence is permitted: two blocks
# would let a later one silently shadow, or simply hide, the first.
# ---------------------------------------------------------------------------
blocks = re.findall(r"^```code-intel[ \t]*\n(.*?)^```[ \t]*$",
                    text, re.M | re.S)

if not blocks:
    print("::error::%s carries no ```code-intel block. The prose in this file "
          "is not executable; without the block this checker verifies nothing "
          "and must not report success." % props_path)
    sys.exit(2)

if len(blocks) > 1:
    print("::error::%s carries %d ```code-intel blocks. Exactly one is allowed "
          "-- with more, which one is authoritative is undefined and a "
          "declaration can hide in the block nothing reads."
          % (props_path, len(blocks)))
    sys.exit(2)

lines = [(n, ln) for n, ln in enumerate(blocks[0].split("\n"), 1) if ln.strip()]

if not lines:
    print("::error::the ```code-intel block in %s is empty. An empty block is "
          "a gate that cannot fail; delete the block or declare something."
          % props_path)
    sys.exit(2)

decls = []
seen_ids = set()
for n, ln in lines:
    try:
        obj = json.loads(ln)
    except json.JSONDecodeError as exc:
        print("::error::%s code-intel line %d is not valid JSON: %s"
              % (props_path, n, exc))
        sys.exit(2)
    if not isinstance(obj, dict):
        print("::error::%s code-intel line %d is not a JSON object."
              % (props_path, n))
        sys.exit(2)
    for field in ("id", "type", "description"):
        if not isinstance(obj.get(field), str) or not obj[field].strip():
            print("::error::%s code-intel line %d: missing or empty string "
                  "field `%s` (envelope requires id/type/description/check)."
                  % (props_path, n, field))
            sys.exit(2)
    if not isinstance(obj.get("check"), dict):
        print("::error::%s code-intel line %d (%s): `check` must be an object."
              % (props_path, n, obj["id"]))
        sys.exit(2)
    if obj["id"] in seen_ids:
        print("::error::%s code-intel line %d: duplicate id `%s`. Ids name the "
              "violation in the failure message, so two rows sharing one make "
              "the report ambiguous." % (props_path, n, obj["id"]))
        sys.exit(2)
    seen_ids.add(obj["id"])
    decls.append((n, obj))

violations = []
checked = 0


def malformed(line_no, decl_id, msg):
    print("::error::%s code-intel line %d (%s): %s"
          % (props_path, line_no, decl_id, msg))
    sys.exit(2)


def read(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Carrier scan. A "carrier" is a file a fresh clone executes by itself. Only an
# executable reference counts as an edge -- a mention inside a `#` comment does
# not run anything, and counting it is how a gate ends up documented-but-inert.
# ---------------------------------------------------------------------------
_carrier_cache = {}


def carrier_refs(carriers):
    key = tuple(carriers)
    if key in _carrier_cache:
        return _carrier_cache[key]
    refs = {}
    for c in carriers:
        src = read(c)
        if src is None:
            print("::error::declared carrier `%s` does not exist. The "
                  "reachability answer computed without it would be wrong in "
                  "the permissive direction." % c)
            sys.exit(2)
        body = "\n".join(l for l in src.split("\n")
                         if not l.lstrip().startswith("#"))
        # `ci/` as well as `scripts/`: gates do land outside scripts/ -- the pocl
        # runner probe and its covering test are both under ci/ -- and an
        # inventory that only knows one directory is complete about a set it
        # chose rather than about what CI runs.
        #
        # The lookbehind rejects `gh-pages/javascripts/x.js` (a letter precedes
        # "scripts/") while accepting both `scripts/x.sh` and `./scripts/x.sh`.
        # Excluding `/` and `.` as well -- the first shape this was written with
        # -- silently dropped every `./scripts/...` invocation in ci.yml, i.e.
        # most of the gates, and the inventory check passed having seen seven
        # entries instead of twenty-one.
        for m in re.findall(
                r"(?<![A-Za-z0-9_-])(?:scripts|ci)/[A-Za-z0-9._-]+", body):
            refs.setdefault(m, set()).add(c)
    _carrier_cache[key] = refs
    return refs


DEFAULT_CARRIERS = [".github/workflows/ci.yml", "Makefile"]


for line_no, d in decls:
    kind, check, did = d["type"], d["check"], d["id"]

    # -- a gate must be wired, and must say whether its red path is covered ---
    if kind == "gate-red-path":
        tool = check.get("tool")
        if not isinstance(tool, str) or not tool:
            malformed(line_no, did, "`check.tool` must be a non-empty string.")
        carriers = check.get("carriers", DEFAULT_CARRIERS)
        refs = carrier_refs(carriers)
        invocation = check.get("invocation", "carrier")
        if invocation not in ("carrier", "manual"):
            malformed(line_no, did,
                      "`check.invocation` must be \"carrier\" or \"manual\", "
                      "got %r." % invocation)
        if not os.path.exists(tool):
            violations.append("%s: declared gate `%s` does not exist."
                              % (did, tool))
        elif invocation == "carrier" and tool not in refs:
            violations.append(
                "%s: gate `%s` is not invoked from any carrier (%s). A gate "
                "nothing runs is indistinguishable from a gate that passes."
                % (did, tool, ", ".join(carriers)))
        elif invocation == "manual":
            # "manual" says CI does not run the tool itself -- only the rule it
            # embodies, via its red-path test. That is a real weakening, so it
            # has to be argued in the KB rather than asserted by a flag.
            if not isinstance(check.get("reason"), str) \
                    or not check["reason"].strip():
                violations.append(
                    "%s: gate `%s` is declared `invocation: manual` with no "
                    "`reason`. Exempting a tool from carrier reachability is "
                    "the move that produces an inert gate; it must be argued."
                    % (did, tool))
            if tool in refs:
                violations.append(
                    "%s: gate `%s` is declared `invocation: manual` but IS "
                    "invoked from a carrier. Drop the declaration -- it "
                    "excuses a weakness that no longer exists." % (did, tool))
        red = check.get("red_path", None)
        if red is None:
            reason = check.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                violations.append(
                    "%s: gate `%s` declares no red-path test and gives no "
                    "`reason`. Uncovered is allowed; silently uncovered is not."
                    % (did, tool))
        elif not isinstance(red, str) or not red:
            malformed(line_no, did,
                      "`check.red_path` must be a non-empty string or null.")
        elif not os.path.exists(red):
            violations.append("%s: red-path test `%s` for gate `%s` does not "
                              "exist." % (did, red, tool))
        elif red not in refs:
            violations.append(
                "%s: red-path test `%s` exists but is not invoked from any "
                "carrier (%s) -- so nothing ever proves `%s` can fail."
                % (did, red, ", ".join(carriers), tool))
        checked += 1

    # -- the inventory itself must be complete -------------------------------
    elif kind == "gate-inventory-complete":
        carriers = check.get("carriers", DEFAULT_CARRIERS)
        exempt = check.get("exempt", [])
        if not isinstance(exempt, list) or any(not isinstance(e, str)
                                               for e in exempt):
            malformed(line_no, did, "`check.exempt` must be a list of strings.")
        # Review-bundle members are exempt by delegation, not by opinion:
        # scripts/check-review-bundle-tracked.sh already computes and publishes
        # their reachability, and REVIEW-BUNDLE.md records which of them CI does
        # not reach and why. Re-listing them here would fork that authority and
        # rot on the next bundle upgrade; naming the manifest tracks it instead.
        manifest_path = check.get("exempt_manifest")
        manifest_exempt = set()
        if manifest_path is not None:
            if not isinstance(manifest_path, str) or not manifest_path:
                malformed(line_no, did,
                          "`check.exempt_manifest` must be a non-empty string.")
            raw = read(manifest_path)
            if raw is None:
                malformed(line_no, did,
                          "`check.exempt_manifest` names %s, which does not "
                          "exist." % manifest_path)
            try:
                entries = json.loads(raw)["files"]
                manifest_exempt = {e["path"] for e in entries}
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                malformed(line_no, did,
                          "could not read files[].path out of %s: %s"
                          % (manifest_path, exc))
            if not manifest_exempt:
                malformed(line_no, did,
                          "%s listed no files. An empty delegated exemption "
                          "set would silently become a strict check on paths "
                          "this repository does not own." % manifest_path)
        refs = carrier_refs(carriers)
        declared = set()
        for _, other in decls:
            if other["type"] == "gate-red-path":
                t = other["check"].get("tool")
                r = other["check"].get("red_path")
                if isinstance(t, str):
                    declared.add(t)
                if isinstance(r, str):
                    declared.add(r)
        undeclared = sorted(set(refs) - declared - set(exempt)
                            - manifest_exempt)
        if undeclared:
            violations.append(
                "%s: script(s) invoked from a carrier with no `gate-red-path` "
                "row in %s and no exemption:\n        %s\n        Add a row "
                "(with `red_path`, or null plus a `reason`), or list it under "
                "`exempt` if it is a tool rather than a gate."
                % (did, props_path, "\n        ".join(undeclared)))
        # An exemption that no longer names anything a carrier runs is a stale
        # permission. Left in place it silently widens on the next rename.
        stale = sorted(e for e in exempt if e not in refs)
        if stale:
            violations.append(
                "%s: stale entr%s in `exempt` -- no carrier references %s. "
                "Remove them; an exemption for a path nothing runs can only "
                "excuse something later."
                % (did, "y" if len(stale) == 1 else "ies", ", ".join(stale)))
        checked += 1

    # -- a literal that must be present in a named file -----------------------
    elif kind == "grep-present":
        path = check.get("file")
        literal = check.get("literal")
        if not isinstance(path, str) or not isinstance(literal, str) \
                or not path or not literal:
            malformed(line_no, did,
                      "`check.file` and `check.literal` must be non-empty "
                      "strings.")
        src = read(path)
        if src is None:
            violations.append(
                "%s: %s does not exist, so the literal it is required to "
                "carry cannot be there." % (did, path))
        else:
            n_found = src.count(literal)
            minimum = check.get("min", 1)
            if not isinstance(minimum, int) or minimum < 1:
                malformed(line_no, did, "`check.min` must be an integer >= 1.")
            if n_found < minimum:
                violations.append(
                    "%s: %s contains %d occurrence(s) of %r, expected at least "
                    "%d.\n        %s"
                    % (did, path, n_found, literal, minimum, d["description"]))
        checked += 1

    # -- a literal that must appear nowhere under the named paths -------------
    elif kind == "grep-absent":
        paths = check.get("paths")
        literal = check.get("literal")
        if not isinstance(paths, list) or not paths \
                or any(not isinstance(p, str) for p in paths) \
                or not isinstance(literal, str) or not literal:
            malformed(line_no, did,
                      "`check.paths` must be a non-empty list of strings and "
                      "`check.literal` a non-empty string.")
        suffixes = check.get("suffixes", [])
        if not isinstance(suffixes, list):
            malformed(line_no, did, "`check.suffixes` must be a list.")
        hits, scanned = [], 0
        for p in paths:
            if not os.path.exists(p):
                violations.append(
                    "%s: declared scan root `%s` does not exist -- this check "
                    "would otherwise have passed having read nothing."
                    % (did, p))
                continue
            files = []
            if os.path.isfile(p):
                files = [p]
            else:
                for root, _dirs, names in os.walk(p):
                    for nm in names:
                        files.append(os.path.join(root, nm))
            for f in files:
                if suffixes and not any(f.endswith(s) for s in suffixes):
                    continue
                src = read(f)
                scanned += 1
                if src and literal in src:
                    hits.append(f)
        if scanned == 0:
            violations.append(
                "%s: scanned 0 files under %s. An absence check that read "
                "nothing always passes." % (did, ", ".join(paths)))
        if hits:
            violations.append(
                "%s: %r found in %d file(s):\n        %s\n        %s"
                % (did, literal, len(hits), "\n        ".join(sorted(hits)),
                   d["description"]))
        checked += 1

    else:
        # Unknown types are loud, not skipped. A typo'd `type` that exited 0
        # would be a declaration nothing executes wearing the badge of one that
        # passed -- which is the failure mode this whole file is about.
        malformed(line_no, did,
                  "unknown check type %r. Known types: gate-red-path, "
                  "gate-inventory-complete, grep-present, grep-absent."
                  % kind)

if checked != len(decls):
    print("::error::%d declarations parsed but only %d executed. Refusing to "
          "report success on a block this checker did not fully run."
          % (len(decls), checked))
    sys.exit(2)

if violations:
    print("::error::kb/properties.md: %d declaration(s) violated."
          % len(violations))
    for v in violations:
        print("    - %s" % v)
    sys.exit(1)

print("OK: %d code-intel declaration(s) in %s hold." % (checked, props_path))
PY
