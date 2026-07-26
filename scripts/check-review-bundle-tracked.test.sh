#!/usr/bin/env bash
# Tests for scripts/check-review-bundle-tracked.sh.
#
# The guard is load-bearing: it is the only thing standing between the project
# and a repeat of #108. It is tested against a synthetic bundle in a throwaway
# git repo, so the assertions are hermetic and do not depend on the state of
# the real tree.
#
# The case that matters most is F2: a guard that derives its worklist from the
# manifest cannot police the manifest. Dropping one files[] entry used to be
# fully green.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="$HERE/check-review-bundle-tracked.sh"
VERIFY="$HERE/review-bundle-verify.js"

PASS=0; FAIL=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
ok()  { PASS=$((PASS+1)); echo "  ok — $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL — $1"; }

expect() { # <label> <rc> <substr> -- cmd...
  local label="$1" want_rc="$2" want="$3"; shift 4
  local out rc; out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -ne "$want_rc" ]; then bad "$label: exit $rc wanted $want_rc"; echo "$out"|sed 's/^/      /'; return; fi
  if ! printf '%s' "$out" | grep -qF -- "$want"; then bad "$label: missing '$want'"; echo "$out"|sed 's/^/      /'; return; fi
  ok "$label"
}

git_q() { git -c init.defaultBranch=main -c user.email=t@t -c user.name=t "$@"; }

# Mirrors the real bundle's shape: the anchor paths named inside the guard,
# plus the wholly bundle-owned directories it discovers independently.
ANCHORS="schema/review-finding.schema.json schema/review-trace.schema.json
scripts/REVIEW-BUNDLE.md scripts/check-review-convergence.js scripts/check-scope-diff.sh
scripts/review-bundle-verify.js scripts/review-normalize.js scripts/xruntime-exec.sh
scripts/xruntime-review.js"
DIRFILES="scripts/lib/review/review-trace.js scripts/lib/review/normalize-rules.js
scripts/lib/xruntime/xruntime-classify.js
tools/data-schema/fixtures/review-finding/valid/basic.jsonl"

# build_fixture <name> -> path to a fresh git repo with a complete, valid bundle
build_fixture() {
  local root="$TMP/$1"; mkdir -p "$root"; git_q init -q "$root"
  local f
  for f in $ANCHORS $DIRFILES; do
    mkdir -p "$root/$(dirname "$f")"
    printf 'content of %s\n' "$f" > "$root/$f"
  done
  # The real verifier is the one under test's collaborator — use it, not a stub.
  cp "$VERIFY" "$root/scripts/review-bundle-verify.js"
  mkdir -p "$root/scripts"
  cp "$GUARD" "$root/scripts/check-review-bundle-tracked.sh"
  chmod +x "$root/scripts/check-review-bundle-tracked.sh"

  node -e '
const fs=require("fs"),crypto=require("crypto"),path=require("path");
const root=process.argv[1], list=process.argv.slice(2);
const sha=f=>crypto.createHash("sha256").update(fs.readFileSync(path.join(root,f))).digest("hex");
fs.writeFileSync(path.join(root,"scripts/review-bundle.manifest.json"),
  JSON.stringify({schema_version:"1.0",bundle_version:"test",channel:"test",
    files:list.map(p=>({path:p,sha256:sha(p)}))},null,2)+"\n");
' "$root" $ANCHORS $DIRFILES

  git_q -C "$root" add -A >/dev/null
  git_q -C "$root" commit -qm "bundle"
  echo "$root"
}

drop_entry() { # <root> <path>
  node -e '
const fs=require("fs");const p=process.argv[1]+"/scripts/review-bundle.manifest.json";
const m=JSON.parse(fs.readFileSync(p,"utf8"));
m.files=m.files.filter(f=>f.path!==process.argv[2]);
fs.writeFileSync(p,JSON.stringify(m,null,2)+"\n");' "$1" "$2"
  git_q -C "$1" add -A >/dev/null; git_q -C "$1" commit -qm "drop $2" >/dev/null
}

echo "== positive control: a complete, tracked bundle passes"
R="$(build_fixture ok)"
expect "complete bundle is green" 0 "bundle tracked, un-ignored, and sha-matched" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

echo "== F2: dropping a manifest entry must not silently shrink the checked set"
R="$(build_fixture dropanchor)"; drop_entry "$R" "scripts/check-review-convergence.js"
expect "dropped ANCHOR entry is caught" 1 "MANIFEST OMISSION scripts/check-review-convergence.js" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(build_fixture droplib)"; drop_entry "$R" "scripts/lib/review/review-trace.js"
expect "dropped bundle-owned-directory entry is caught" 1 "MANIFEST OMISSION scripts/lib/review/review-trace.js" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(build_fixture dropfixture)"; drop_entry "$R" "tools/data-schema/fixtures/review-finding/valid/basic.jsonl"
expect "dropped fixture entry is caught" 1 "MANIFEST OMISSION" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

# The point of F2, stated as an assertion: the collaborating verifier is happy
# with the shrunken manifest. If review-bundle-verify alone were the gate, this
# tampering would be invisible.
R="$(build_fixture verifyblind)"; drop_entry "$R" "scripts/check-review-convergence.js"
out="$(cd "$R" && node scripts/review-bundle-verify.js 2>&1)"; rc=$?
if [ "$rc" -eq 0 ]; then
  ok "confirms review-bundle-verify alone reports OK on the shrunken manifest (why the guard must not ask only the manifest)"
else
  bad "expected review-bundle-verify to be blind here, got exit $rc"; echo "$out"|sed 's/^/      /'
fi

echo "== boundary with other PRs' tooling (F10 / #307 coupling)"
# scripts/lib/ is NOT wholly bundle-owned — only scripts/lib/review/ and
# scripts/lib/xruntime/ are. A sibling PR adding scripts/lib/<its-own>.js must
# not be dragged into this bundle's manifest, or the two PRs fight over a file
# neither of them owns.
R="$(build_fixture sibling)"
mkdir -p "$R/scripts/lib"
echo "// another PR's tool" > "$R/scripts/lib/mandated-steps-rules.js"
echo "// another PR's tool" > "$R/scripts/check-mandated-steps.js"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm sibling >/dev/null
out="$(bash "$R/scripts/check-review-bundle-tracked.sh" "$R" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && ! printf '%s' "$out" | grep -q "MANIFEST OMISSION"; then
  ok "a sibling PR's scripts/lib/*.js is not claimed as a missing bundle member"
else
  bad "guard claimed a sibling PR's file (exit $rc)"; echo "$out"|sed 's/^/      /'
fi

# An orphan is reported but must not fail the gate: CI never sees orphans, and
# failing would let one agent's uncommitted work break another agent's build.
R="$(build_fixture orphan)"
echo "// not tracked, not ignored" > "$R/scripts/some-orphan-tool.js"
out="$(bash "$R/scripts/check-review-bundle-tracked.sh" "$R" 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -qF "scripts/some-orphan-tool.js"; then
  ok "an orphaned tool is named in the report but does not fail the gate"
else
  bad "orphan handling wrong (exit $rc)"; echo "$out"|sed 's/^/      /'
fi

echo "== reachability reporting (F3) must respond to reality, not always print the same list"
R="$(build_fixture reach)"
mkdir -p "$R/.github/workflows"
# A doc mention must NOT count as a caller — that was the first version of this
# check, and it reported nearly everything reached while proving nothing.
printf 'Run it with `node scripts/review-normalize.js --task x`.\n' > "$R/docs-note.md"
printf 'name: CI\njobs:\n  b:\n    steps:\n      - run: echo hi\n' > "$R/.github/workflows/ci.yml"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm docs >/dev/null
out="$(bash "$R/scripts/check-review-bundle-tracked.sh" "$R" 2>&1)"
if printf '%s' "$out" | grep -qF "scripts/review-normalize.js"; then
  ok "a documentation mention does not count as a caller"
else
  bad "doc mention was counted as a caller"; echo "$out"|sed 's/^/      /'
fi

# Now genuinely invoke it from CI: the tool must drop off the unreached list.
printf 'name: CI\njobs:\n  b:\n    steps:\n      - run: node scripts/review-normalize.js --task x\n' \
  > "$R/.github/workflows/ci.yml"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm wire >/dev/null
out="$(bash "$R/scripts/check-review-bundle-tracked.sh" "$R" 2>&1)"
if ! printf '%s' "$out" | grep -qE "^    scripts/review-normalize\.js$"; then
  ok "wiring a tool into CI removes it from the unreached list (the check can change its mind)"
else
  bad "tool stayed unreached after being invoked from CI"; echo "$out"|sed 's/^/      /'
fi

# Transitivity: CI -> wrapper -> tool must count as reached.
printf 'name: CI\njobs:\n  b:\n    steps:\n      - run: bash scripts/wrapper.sh\n' > "$R/.github/workflows/ci.yml"
printf '#!/bin/sh\nnode scripts/review-normalize.js --task x\n' > "$R/scripts/wrapper.sh"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm transitive >/dev/null
out="$(bash "$R/scripts/check-review-bundle-tracked.sh" "$R" 2>&1)"
if ! printf '%s' "$out" | grep -qE "^    scripts/review-normalize\.js$"; then
  ok "reachability is transitive through a wrapper script"
else
  bad "transitive reachability not detected"; echo "$out"|sed 's/^/      /'
fi

echo "== local_patches[] must be an enforced claim, not a comment"
# Simulates what an upstream upgrade does: overwrite a patched file with the
# pristine version and regenerate its sha256. The hashes are then perfectly
# consistent and the fix is gone. Only the marker notices.
patch_fixture() { # <name>
  local root; root="$(build_fixture "$1")"
  printf 'exports.f = 1; // SPOC-LOCAL-MARKER-XYZ\n' > "$root/scripts/lib/review/normalize-rules.js"
  node -e '
const fs=require("fs"),crypto=require("crypto"),path=require("path");
const root=process.argv[1], p=root+"/scripts/review-bundle.manifest.json";
const m=JSON.parse(fs.readFileSync(p,"utf8"));
m.local_patches=[{ref:"demo-patch",reapply_on_upgrade:true,
  paths:["scripts/lib/review/normalize-rules.js"],
  markers:{"scripts/lib/review/normalize-rules.js":["SPOC-LOCAL-MARKER-XYZ"]}}];
for (const e of m.files) e.sha256 = crypto.createHash("sha256")
  .update(fs.readFileSync(path.join(root,e.path))).digest("hex");
fs.writeFileSync(p, JSON.stringify(m,null,2)+"\n");' "$root"
  git_q -C "$root" add -A >/dev/null; git_q -C "$root" commit -qm patched >/dev/null
  echo "$root"
}

R="$(patch_fixture patched)"
expect "an applied patch passes" 0 "bundle tracked, un-ignored, and sha-matched" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(patch_fixture reverted)"
printf 'exports.f = 1; // pristine upstream, patch gone\n' > "$R/scripts/lib/review/normalize-rules.js"
node -e '
const fs=require("fs"),crypto=require("crypto"),path=require("path");
const root=process.argv[1], p=root+"/scripts/review-bundle.manifest.json";
const m=JSON.parse(fs.readFileSync(p,"utf8"));
for (const e of m.files) e.sha256 = crypto.createHash("sha256")
  .update(fs.readFileSync(path.join(root,e.path))).digest("hex");
fs.writeFileSync(p, JSON.stringify(m,null,2)+"\n");' "$R"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm "simulated upgrade" >/dev/null
# The hashes agree with the new content, so the verifier is satisfied.
if (cd "$R" && node scripts/review-bundle-verify.js >/dev/null 2>&1); then
  ok "confirms review-bundle-verify is green after an upgrade silently reverted the patch"
else
  bad "expected the verifier to be green on the reverted tree"
fi
expect "a reverted local patch is caught by its marker" 1 "marker absent" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

# The array form of `markers`: the natural way to write a single-file patch,
# and the shape the first outside contributor actually used.
patch_fixture_array() { # <name>
  local root; root="$(build_fixture "$1")"
  printf 'exports.f = 1; // SPOC-LOCAL-MARKER-XYZ and legacy_skip_authorized\n' \
    > "$root/scripts/lib/review/normalize-rules.js"
  node -e '
const fs=require("fs"),crypto=require("crypto"),path=require("path");
const root=process.argv[1], p=root+"/scripts/review-bundle.manifest.json";
const m=JSON.parse(fs.readFileSync(p,"utf8"));
m.local_patches=[{ref:"array-form",reapply_on_upgrade:true,
  paths:["scripts/lib/review/normalize-rules.js"],
  markers:["SPOC-LOCAL-MARKER-XYZ","legacy_skip_authorized"]}];
for (const e of m.files) e.sha256 = crypto.createHash("sha256")
  .update(fs.readFileSync(path.join(root,e.path))).digest("hex");
fs.writeFileSync(p, JSON.stringify(m,null,2)+"\n");' "$root"
  git_q -C "$root" add -A >/dev/null; git_q -C "$root" commit -qm arrayform >/dev/null
  echo "$root"
}

R="$(patch_fixture_array arrayok)"
expect "markers given as an array is accepted" 0 "bundle tracked, un-ignored, and sha-matched" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(patch_fixture_array arrayrevert)"
printf 'exports.f = 1; // pristine upstream\n' > "$R/scripts/lib/review/normalize-rules.js"
node -e '
const fs=require("fs"),crypto=require("crypto"),path=require("path");
const root=process.argv[1], p=root+"/scripts/review-bundle.manifest.json";
const m=JSON.parse(fs.readFileSync(p,"utf8"));
for (const e of m.files) e.sha256 = crypto.createHash("sha256")
  .update(fs.readFileSync(path.join(root,e.path))).digest("hex");
fs.writeFileSync(p, JSON.stringify(m,null,2)+"\n");' "$R"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm reverted >/dev/null
expect "a reverted patch declared in array form is still caught" 1 "marker absent" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

# A malformed markers block must say so, not invent missing files. The array
# form applied to an empty `paths` used to report "patched file is missing: 0".
malformed_fixture() { # <name> <json-fragment-for-local_patches>
  local root; root="$(build_fixture "$1")"
  node -e '
const fs=require("fs");const p=process.argv[1]+"/scripts/review-bundle.manifest.json";
const m=JSON.parse(fs.readFileSync(p,"utf8"));
m.local_patches=[JSON.parse(process.argv[2])];
fs.writeFileSync(p,JSON.stringify(m,null,2)+"\n");' "$root" "$2"
  git_q -C "$root" add -A >/dev/null; git_q -C "$root" commit -qm malformed >/dev/null
  echo "$root"
}

R="$(malformed_fixture nopaths '{"ref":"x","reapply_on_upgrade":true,"markers":["a"],"paths":[]}')"
expect "array markers with no paths reports the real problem" 1 "there is no file to look for them in" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(malformed_fixture badtype '{"ref":"x","reapply_on_upgrade":true,"markers":"a-string","paths":["scripts/review-normalize.js"]}')"
expect "a markers value of the wrong type is named as such" 1 "markers must be an object" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(malformed_fixture badneedle '{"ref":"x","reapply_on_upgrade":true,"markers":[""],"paths":["scripts/review-normalize.js"]}')"
expect "an empty-string needle is rejected rather than matching everything" 1 "non-empty strings" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(build_fixture nomarkers)"
node -e '
const fs=require("fs");const p=process.argv[1]+"/scripts/review-bundle.manifest.json";
const m=JSON.parse(fs.readFileSync(p,"utf8"));
m.local_patches=[{ref:"lazy",reapply_on_upgrade:true,paths:["scripts/review-normalize.js"]}];
fs.writeFileSync(p,JSON.stringify(m,null,2)+"\n");' "$R"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm lazy >/dev/null
expect "a patch declaring reapply_on_upgrade with no markers is itself a failure" 1 "names no markers" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

echo "== untracked and ignored bundle members"
R="$(build_fixture untracked)"
git_q -C "$R" rm -q --cached scripts/xruntime-review.js >/dev/null
expect "an untracked bundle member is caught" 1 "UNTRACKED scripts/xruntime-review.js" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

R="$(build_fixture ignored)"
echo "scripts/xruntime-exec.sh" > "$R/.gitignore"
git_q -C "$R" add -A >/dev/null; git_q -C "$R" commit -qm ignore >/dev/null
expect "an ignored-but-tracked bundle member is caught" 1 "IGNORED scripts/xruntime-exec.sh" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

echo "== content drift still fails"
R="$(build_fixture drift)"
echo "tampered" >> "$R/scripts/review-normalize.js"
expect "modified content is caught by the verifier" 1 "SHA MISMATCH" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

echo "== a missing manifest is a hard failure, not an empty pass"
R="$(build_fixture nomanifest)"; rm "$R/scripts/review-bundle.manifest.json"
expect "missing manifest fails loudly" 1 "is missing" -- \
  bash "$R/scripts/check-review-bundle-tracked.sh" "$R"

echo ""
echo "check-review-bundle-tracked.test: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
