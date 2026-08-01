#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# ---------------------------------------------------------------------------
# Covering test for scripts/check-production-link.py (task #201, kb row
# KB-GATE-PRODUCTION-LINK).
#
# WHY THIS EXISTS AS A TEST AND NOT AS A README LINE
#
# check-production-link.py is a gate: kb/properties.md's KB-GATE-SELF /
# KB-GATE-PROVE-RED doctrine is that a gate not proven able to fail is not
# trusted to have caught anything. Round-1 review of this gate found it FALSE-
# PASSING (a production_call surviving only inside an OCaml string literal —
# `strip_ocaml_comments` stripped comments but not strings) with no red-path
# companion to have caught it. Every scenario below is a mutation that MUST
# fail, asserted on the failure message, not just the exit code.
#
# Runs against a synthetic single-project formal/ tree in a temp directory —
# not a copy of the real formal/ — so this needs no Rocq and runs in a second.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0

expect() {
  local name="$1" want_rc="$2" want_msg="$3"
  local out="$WORK/out.txt" rc=0
  set +e
  python3 "$WORK/repo/scripts/check-production-link.py" --generated "$WORK/gen" \
    > "$out" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne "$want_rc" ]; then
    echo "FAIL [$name]: exit $rc, expected $want_rc"
    sed 's/^/       /' "$out"
    fail=$((fail + 1))
    return
  fi
  if [ -n "$want_msg" ] && ! grep -qF -- "$want_msg" "$out"; then
    echo "FAIL [$name]: exit $rc as expected, but the message is wrong."
    echo "       expected to contain: $want_msg"
    sed 's/^/       /' "$out"
    fail=$((fail + 1))
    return
  fi
  echo "ok   [$name]"
  pass=$((pass + 1))
}

edit_json() {  # edit_json <file> <python statement over `d`>
  python3 - "$1" "$2" <<'PY'
import json, sys
path, stmt = sys.argv[1], sys.argv[2]
d = json.load(open(path))
exec(stmt)
with open(path, "w") as fh:
    json.dump(d, fh, indent=2); fh.write("\n")
PY
}

write_test_file() {  # write_test_file <content>
  printf '%s' "$1" > "$WORK/repo/formal/proj1/test/test_foo.ml"
}

# A synthetic one-project formal/ tree: one module (Foo) with a declared
# production-link entry backed by a live call, one model-only module (Bar)
# with no entry. Deliberately not a copy of the real formal/ — this tests the
# ENFORCEMENT, independent of what any real project currently claims.
reset_sandbox() {
  rm -rf "$WORK/repo" "$WORK/gen"
  mkdir -p "$WORK/repo/scripts" "$WORK/repo/formal/proj1/test" "$WORK/gen"
  cp "$REPO/scripts/check-production-link.py" "$WORK/repo/scripts/"
  : > "$WORK/repo/formal/proj1/_CoqProject"
  cat > "$WORK/repo/formal/proj1/production-link.json" <<'JSON'
{
  "schema": 1,
  "purpose": "synthetic fixture for check-production-link.test.sh",
  "modules": {
    "Foo": {
      "test_file": "test/test_foo.ml",
      "production_call": "Prod_mod.real_call",
      "evidence": "differential_sampled",
      "note": "fixture"
    }
  }
}
JSON
  write_test_file 'let () = ignore (Prod_mod.real_call 1 2)
'
  cat > "$WORK/gen/proj1.json" <<'JSON'
{
  "modules": {
    "Proj1.Foo": {"counts": {"theorems": 3}},
    "Proj1.Bar": {"counts": {"theorems": 2}}
  }
}
JSON
}

echo "== baseline: a live call in a real test file must pass"
reset_sandbox
expect "unmutated sandbox is green" 0 "checked against production, TOTAL: 3"

echo
echo "== H2 regression pin: comment-only mention must still be caught"
reset_sandbox
write_test_file '(* Prod_mod.real_call is the production inference engine *)
let () = ()
'
expect "comment-only mention is caught" 1 \
  "no live reference to it was found outside comments or string literals"

echo
echo "== H2 fix pin: STRING-LITERAL-only mention must be caught"
echo "   (this is the exact false-pass the round-1 review mutation-proved:"
echo "   replacing the live call with the symbol quoted in a string used to"
echo "   leave this gate green. If this regresses, THIS case goes green.)"
reset_sandbox
write_test_file 'let () = print_string "calling Prod_mod.real_call now"
'
expect "string-literal-only mention is caught" 1 \
  "no live reference to it was found outside comments or string literals"

echo
echo "== a live call still counts when a comment ALSO mentions the symbol"
reset_sandbox
write_test_file '(* see Prod_mod.real_call docs *)
let () = ignore (Prod_mod.real_call 1 2)
'
expect "comment plus live call is still green" 0 \
  "checked against production, TOTAL: 3"

echo
echo "== a live call still counts when a string ALSO mentions the symbol"
reset_sandbox
write_test_file 'let () =
  print_string "Prod_mod.real_call";
  ignore (Prod_mod.real_call 1 2)
'
expect "string plus live call is still green" 0 \
  "checked against production, TOTAL: 3"

echo
echo "== an escaped quote inside a string must not end the string early"
reset_sandbox
write_test_file 'let () = print_string "a \" Prod_mod.real_call \" b"
'
expect "text after an escaped quote stays inside the string" 1 \
  "no live reference to it was found outside comments or string literals"

echo
echo "== claim integrity"
reset_sandbox
rm "$WORK/repo/formal/proj1/test/test_foo.ml"
expect "a missing test file is caught" 1 "which does not exist"

reset_sandbox
edit_json "$WORK/repo/formal/proj1/production-link.json" \
  'd["modules"]["Foo"]["evidence"] = "made_up_kind"'
expect "an unrecognised evidence kind is caught" 1 \
  "is not one of"

reset_sandbox
edit_json "$WORK/repo/formal/proj1/production-link.json" \
  'd["modules"]["Baz"] = d["modules"].pop("Foo")'
expect "a claimed module absent from the ledger is caught" 1 \
  "is not in this build's ledger"

reset_sandbox
edit_json "$WORK/repo/formal/proj1/production-link.json" \
  'del d["modules"]["Foo"]["production_call"]'
expect "a claim missing production_call is caught" 1 \
  "missing test_file or production_call"

echo
echo "== theorem-count clause is NOT enforced (docstring narrowed to match,"
echo "   backlog-201 round-1 MEDIUM) -- pinned here so nobody re-adds the"
echo "   overclaim without also adding the missing check"
reset_sandbox
edit_json "$WORK/gen/proj1.json" 'd["modules"]["Proj1.Foo"]["counts"]["theorems"] = 5000'
expect "an inflated ledger count is NOT caught (documents current scope)" 0 \
  "checked against production, TOTAL: 5000"

echo
echo "== F-1 (CodeRabbit round 2): OCaml quoted-string literals must be"
echo "   stripped like \"...\" strings, and the identifier boundary must not"
echo "   accept a near-miss identifier that merely starts with the claimed name"
reset_sandbox
write_test_file 'let () = print_string {|Prod_mod.real_call|}
'
expect "a plain {|...|} quoted-string-only mention is caught" 1 \
  "no live reference to it was found outside comments or string literals"

reset_sandbox
write_test_file 'let () = print_string {tag|Prod_mod.real_call|tag}
'
expect "a tagged {tag|...|tag} quoted-string-only mention is caught" 1 \
  "no live reference to it was found outside comments or string literals"

reset_sandbox
write_test_file "let () = ignore (Prod_mod.real_call' 1 2)
"
expect "a near-miss identifier ending in an apostrophe is NOT a live reference" 1 \
  "no live reference to it was found outside comments or string literals"

reset_sandbox
write_test_file 'let () = ignore (Prod_mod.real_call_extra 1 2)
'
expect "a near-miss identifier with a trailing underscore-suffix is NOT a live reference" 1 \
  "no live reference to it was found outside comments or string literals"

echo
echo "== F-1 positive control: a genuine reference must still count in every"
echo "   syntactic position it legitimately appears -- over-strict boundary"
echo "   handling would silently reclassify a shipped module as model-only"
reset_sandbox
write_test_file 'let () =
  ignore (Prod_mod.real_call 1 2);
  let x = Prod_mod.real_call in
  ignore x;
  Prod_mod.real_call
'
expect "a live call before '(', before a space, before ';', and at end-of-line all still count" 0 \
  "checked against production, TOTAL: 3"

reset_sandbox
write_test_file 'let () = print_string {|Prod_mod.real_call|};
  ignore (Prod_mod.real_call 1 2)
'
expect "a quoted-string decoy plus a live call is still green" 0 \
  "checked against production, TOTAL: 3"

reset_sandbox
write_test_file "let () = ignore (Prod_mod.real_call' 2 3);
  ignore (Prod_mod.real_call 1 2)
"
expect "a near-miss decoy plus a live call is still green" 0 \
  "checked against production, TOTAL: 3"

echo
echo "== F-2 (CodeRabbit): manifest schema must be validated before classifying"
reset_sandbox
edit_json "$WORK/repo/formal/proj1/production-link.json" 'd["schema"] = 2'
expect "an unsupported schema is refused, not treated as an empty declaration" 2 \
  "does not know how to read"

reset_sandbox
edit_json "$WORK/repo/formal/proj1/production-link.json" 'del d["schema"]'
expect "a missing schema is refused the same way as an unsupported one" 2 \
  "does not know how to read"

reset_sandbox
edit_json "$WORK/repo/formal/proj1/production-link.json" 'd["modules"] = ["not", "an", "object"]'
expect "a non-object \"modules\" is refused, not an unhandled AttributeError" 2 \
  "is a list, not a JSON object"

echo
echo "== F-3 (CodeRabbit): a ledger with two modules ending in the same bare"
echo "   name must be refused, not silently collapsed onto one of them"
reset_sandbox
edit_json "$WORK/gen/proj1.json" 'd["modules"]["Other.Foo"] = {"counts": {"theorems": 7}}'
expect "duplicate module_short_name across the ledger is refused as ambiguous" 2 \
  "cannot unambiguously refer to one by its short name"

echo
echo "== F-4 (CodeRabbit): test_file must resolve inside the declaring"
echo "   project's own formal/<project>/ directory"
reset_sandbox
mkdir -p "$WORK/repo/formal/proj1/test"
printf '%s' 'let () = ignore (Prod_mod.real_call 1 2)
' > "$WORK/repo/formal/secret.ml"
edit_json "$WORK/repo/formal/proj1/production-link.json" \
  'd["modules"]["Foo"]["test_file"] = "../secret.ml"'
expect "a ../ path escaping the project directory is rejected" 1 \
  "resolves outside"

reset_sandbox
printf '%s' 'let () = ignore (Prod_mod.real_call 1 2)
' > "$WORK/outside.ml"
edit_json "$WORK/repo/formal/proj1/production-link.json" \
  "d[\"modules\"][\"Foo\"][\"test_file\"] = \"$WORK/outside.ml\""
expect "an absolute path escaping the project directory is rejected" 1 \
  "resolves outside"

echo
echo "== vacuity"
reset_sandbox
edit_json "$WORK/gen/proj1.json" \
  'd["modules"]["Proj1.Foo"]["counts"]["theorems"] = 0; d["modules"]["Proj1.Bar"]["counts"]["theorems"] = 0'
expect "an all-zero ledger is caught rather than reporting TOTAL: 0" 1 \
  "would pass vacuously"

reset_sandbox
rmdir "$WORK/repo/formal/proj1" 2>/dev/null || rm -rf "$WORK/repo/formal/proj1"
mkdir -p "$WORK/repo/formal"
expect "no formal/ projects at all is caught" 1 \
  "no formal/ projects found"

echo
echo "-- $pass passed, $fail failed"
[ "$fail" -eq 0 ]
