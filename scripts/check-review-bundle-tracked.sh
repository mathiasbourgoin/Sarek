#!/usr/bin/env bash
# check-review-bundle-tracked.sh — the review-tool bundle must survive a fresh clone.
#
# Failure mode this guard exists for (backlog #108): every deterministic quality
# gate this project relies on (review normalization, convergence, scope-diff,
# cross-runtime probing, the finding/trace schemas and their fixtures) lived
# ONLY as untracked files on one workstation, hidden behind `.git/info/exclude`
# — a machine-local, unshared, unreviewable ignore file. `git status` was clean,
# `.gitignore` said nothing, and a fresh clone got no gates at all.
#
# `scripts/review-bundle-verify.js` checks that the installed bundle is complete
# and unmodified. It cannot see the tracking problem: an untracked bundle
# verifies perfectly. This guard checks the orthogonal property — that git will
# actually hand the bundle to the next clone.
#
# Checks, in order:
#   1. the manifest exists and is itself tracked
#   2. every manifest-listed path is tracked
#   3. no manifest-listed path is matched by ANY ignore source
#      (.gitignore, .git/info/exclude, core.excludesFile)
#   4. content integrity, delegated to review-bundle-verify.js
#
# Exit: 0 all good; 1 a bundle file is untracked or ignored, or content drifted;
#       2 usage/environment error.
set -uo pipefail

ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$ROOT" ] || { echo "check-review-bundle-tracked: not inside a git repository" >&2; exit 2; }
cd "$ROOT" || exit 2

MANIFEST="scripts/review-bundle.manifest.json"
[ -f "$MANIFEST" ] || {
  echo "check-review-bundle-tracked: FAIL — $MANIFEST is missing." >&2
  echo "  The review-tool bundle is not installed. Install it before relying on any review gate." >&2
  exit 1
}

command -v node >/dev/null 2>&1 || { echo "check-review-bundle-tracked: node is required" >&2; exit 2; }

# The manifest is the sentinel and is not listed inside its own `files` array,
# so it is checked explicitly alongside the paths it names.
PATHS=$(node -e '
const m = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
if (!Array.isArray(m.files)) { process.exit(3); }
for (const f of m.files) { if (f && typeof f.path === "string") console.log(f.path); }
' "$MANIFEST") || { echo "check-review-bundle-tracked: $MANIFEST is unreadable or has no files[]" >&2; exit 2; }

FAILED=0
report() { echo "check-review-bundle-tracked: $1" >&2; FAILED=1; }

# ── The worklist must not come only from the file being policed ─────────────
#
# Deriving the list of files to check FROM the manifest means the manifest
# defines its own scope of enforcement. Delete one `files[]` entry and that
# file simply stops being checked: it is no longer verified, no longer
# required to be tracked, and both this guard and review-bundle-verify report
# a clean, fully green result on a bundle that is now missing a gate. The
# checker agrees with the manifest because it only ever asked the manifest.
#
# Two independent sources of expectation close that:
#
#   1. DIRECTORY DISCOVERY. scripts/lib/review/, scripts/lib/xruntime/ and
#      tools/data-schema/fixtures/review-finding/ are wholly bundle-owned.
#      Every file git tracks under them must appear in files[]. This comes
#      from the filesystem and the git index, not from the manifest, so a
#      dropped entry shows up as a tracked file the manifest does not claim.
#
#   2. AN ANCHOR LIST IN THIS FILE. The scattered top-level bundle members are
#      named here, in a different tracked file from the manifest. Dropping a
#      manifest entry is no longer sufficient — it now requires editing two
#      tracked files in the same change, which is visible in review.
#
# What this still cannot do: stop someone who edits both files deliberately.
# Nothing self-contained can. The goal is that silent single-file drift fails
# loudly, not that tampering is impossible.
BUNDLE_DIRS="scripts/lib/review scripts/lib/xruntime tools/data-schema/fixtures/review-finding"
ANCHOR_PATHS="
schema/review-finding.schema.json
schema/review-trace.schema.json
scripts/REVIEW-BUNDLE.md
scripts/check-review-convergence.js
scripts/check-scope-diff.sh
scripts/review-bundle-verify.js
scripts/review-normalize.js
scripts/xruntime-exec.sh
scripts/xruntime-review.js
"

manifest_claims() { printf '%s\n' "$PATHS" | grep -qxF -- "$1"; }

for anchor in $ANCHOR_PATHS; do
  [ -n "$anchor" ] || continue
  if ! manifest_claims "$anchor"; then
    report "MANIFEST OMISSION $anchor — a known bundle member is absent from files[], so nothing verifies it. Restore the entry (or update ANCHOR_PATHS in this script if the bundle genuinely dropped it)."
  fi
done

for dir in $BUNDLE_DIRS; do
  [ -d "$dir" ] || continue
  while IFS= read -r tracked; do
    [ -n "$tracked" ] || continue
    if ! manifest_claims "$tracked"; then
      report "MANIFEST OMISSION $tracked — tracked under bundle-owned $dir/ but absent from files[], so it is neither hash-verified nor required to stay tracked."
    fi
  done <<EOF
$(git ls-files -- "$dir")
EOF
done

while IFS= read -r rel; do
  [ -n "$rel" ] || continue

  if ! git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
    report "UNTRACKED $rel — present on this workstation only; a fresh clone will not have it."
  fi

  # `git check-ignore` reports the matching rule and its source file, which is
  # what makes a re-introduced .git/info/exclude entry diagnosable rather than
  # just "mysteriously missing".
  if RULE=$(git check-ignore -v --no-index -- "$rel" 2>/dev/null); then
    report "IGNORED $rel — matched by: $RULE"
  fi
done <<EOF
$MANIFEST
$PATHS
EOF

if ! node scripts/review-bundle-verify.js; then
  report "content verification failed (see review-bundle-verify output above)."
fi

# ── local_patches[] must be a claim, not a comment ──────────────────────────
#
# The manifest declares SPOC-local edits to upstream-owned bundle files, each
# with reapply_on_upgrade. As written that was documentation and nothing more:
# an upgrade overwrites the patched file, the installer regenerates the
# sha256, and review-bundle-verify goes green on a bundle where the fix is
# gone. The declaration outlived the thing it described.
#
# Each patch therefore names MARKERS — strings that exist only because the
# patch is applied. Marker absent => the patch was reverted, whatever the
# hashes say. The covering test must also exist and be reachable from CI, so
# a patch cannot be "verified" by a test nothing runs.
PATCH_PROBLEMS=$(node -e '
const fs = require("fs");
const m = JSON.parse(fs.readFileSync("scripts/review-bundle.manifest.json", "utf8"));
const out = [];

// ── The mechanism must survive the operation it exists to survive ──────────
//
// local_patches[] lives in the manifest, and an upgrade REGENERATES the
// manifest. So the one event this mechanism is for — an upgrade overwriting a
// patched file — is also the event that deletes the mechanism describing it.
// Executed: revert the patch, drop local_patches, regenerate files[] hashes,
// and the guard reported "OK — bundle tracked, un-ignored, and sha-matched".
// Same for renaming the key, or setting it to an object, or to entries that
// are null or strings.
//
// This is the F2 class, fixed there for files[] with a second independent
// source and not carried over to the newer block. The second source here is
// the same one: a list that lives in THIS file, which an upgrade does not
// touch. Every pair below must hold regardless of what the manifest says.
const REQUIRED = [
  ["scripts/lib/xruntime/xruntime-classify.js", "FAULT_BY_OUTCOME", "#102 fault attribution"],
  ["scripts/lib/xruntime/xruntime-classify.js", "runtime-error", "F1 nonzero-exit is a runtime fault"],
  ["scripts/lib/xruntime/xruntime-journal.js", "isCallerFault", "#102 breaker exemption"],
  ["scripts/lib/xruntime/xruntime-contract.js", "OUTPUT_CONTRACT", "#102 --emit-contract"],
  ["scripts/lib/review/review-lifecycle.js", "expected \"GO\" or \"NO-GO\"", "verdict-status gate"],
];
// Identifier-like markers match on word boundaries. Plain String.includes let
// FAULT_BY_OUTCOMEX satisfy FAULT_BY_OUTCOME — a renamed symbol reads as an
// applied patch.
function present(text, needle) {
  if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(needle)) {
    return new RegExp(`\\b${needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`).test(text);
  }
  return text.includes(needle);
}
// A marker satisfied only by a comment is not an applied patch — the guard
// would be strictly weaker than the test it claims to be backed by.
function stripJsComments(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/(^|[^:])\/\/[^\n]*/g, "$1 ");
}
function codeOf(file, text) {
  return file.endsWith(".js") ? stripJsComments(text) : text;
}
for (const [file, needle, why] of REQUIRED) {
  let text;
  try { text = fs.readFileSync(file, "utf8"); }
  catch { out.push(`REQUIRED patch file missing: ${file} (${why})`); continue; }
  if (!present(codeOf(file, text), needle)) {
    out.push(`REQUIRED marker absent from ${file}: ${JSON.stringify(needle)} (${why}) — this is checked independently of local_patches[], which an upgrade regenerates`);
  }
}

if (m.local_patches !== undefined && !Array.isArray(m.local_patches)) {
  out.push(`local_patches must be an array, got ${Array.isArray(m.local_patches) ? "array" : typeof m.local_patches}`);
}
const patches = Array.isArray(m.local_patches) ? m.local_patches : [];
for (const [i, p] of patches.entries()) {
  if (!p || typeof p !== "object" || Array.isArray(p)) {
    out.push(`local_patches[${i}] is not an object (${p === null ? "null" : typeof p})`);
  }
}
for (const p of patches) {
  const ref = p && p.ref ? p.ref : "(unnamed patch)";
  if (!p || p.reapply_on_upgrade !== true) continue;
  // `markers` accepts two shapes:
  //   {file: [needle, ...]}  — per-file, when a patch spans files that each
  //                            need a different string
  //   [needle, ...]          — the same needles must appear in every entry of
  //                            `paths`; the natural form for a single-file patch
  // The array form is not a convenience afterthought: passing one to an
  // Object.entries loop yields index keys, and the guard reported
  // "patched file is missing: 0" — a real defect surfaced the first time
  // someone outside this file wrote an entry. A malformed markers block must
  // fail as a malformed markers block, not as three imaginary missing files.
  const rawMarkers = p && p.markers;
  let markers;
  if (Array.isArray(rawMarkers)) {
    const paths = Array.isArray(p.paths) ? p.paths : [];
    if (paths.length === 0) {
      out.push(`${ref}: markers is an array but paths is empty — there is no file to look for them in`);
      continue;
    }
    if (rawMarkers.length === 0) {
      // Array.isArray passes, [].every() is vacuously true, and
      // Object.fromEntries then yields N keys with zero needles, so the
      // "names no markers" check below never fires. Executed: markers: []
      // with the #102 patch genuinely reverted and hashes regenerated
      // returned exit 0, 23/23 green. markers: {} failed correctly; the
      // array shape had to be taught the same thing.
      out.push(`${ref}: markers is an empty array — that verifies nothing; give it the strings that exist only while the patch is applied, or drop reapply_on_upgrade`);
      continue;
    }
    if (!rawMarkers.every((n) => typeof n === "string" && n.length > 0)) {
      out.push(`${ref}: markers array must contain only non-empty strings`);
      continue;
    }
    markers = Object.fromEntries(paths.map((f) => [f, rawMarkers]));
  } else if (rawMarkers && typeof rawMarkers === "object") {
    markers = rawMarkers;
    const badValue = Object.entries(markers).find(
      ([, v]) => !Array.isArray(v) || !v.every((n) => typeof n === "string" && n.length > 0)
    );
    if (badValue) {
      out.push(`${ref}: markers[${JSON.stringify(badValue[0])}] must be an array of non-empty strings`);
      continue;
    }
  } else if (rawMarkers === undefined) {
    markers = {};
  } else {
    out.push(`${ref}: markers must be an object {file: [needle]} or an array [needle], got ${typeof rawMarkers}`);
    continue;
  }

  const totalNeedles = Object.values(markers).reduce((n, v) => n + v.length, 0);
  if (Object.keys(markers).length === 0 || totalNeedles === 0) {
    out.push(`${ref}: declares reapply_on_upgrade but names no markers — nothing can tell whether it is still applied`);
    continue;
  }
  // A marker keyed on a file the patch does not touch "verifies" the patch
  // against unrelated content: {"README.md": ["SPOC"]} passed for #102.
  const declaredPaths = Array.isArray(p.paths) ? p.paths : [];
  for (const file of Object.keys(markers)) {
    if (declaredPaths.length && !declaredPaths.includes(file)) {
      out.push(`${ref}: markers names ${JSON.stringify(file)}, which is not in paths — a marker in a file the patch does not touch attests nothing`);
    }
  }
  for (const [file, needles] of Object.entries(markers)) {
    let text;
    try { text = fs.readFileSync(file, "utf8"); }
    catch { out.push(`${ref}: patched file is missing: ${file}`); continue; }
    for (const needle of needles) {
      if (!present(codeOf(file, text), needle)) {
        out.push(`${ref}: marker absent from ${file}: ${JSON.stringify(needle)} — the patch has been reverted (an upgrade probably overwrote it); re-apply it and re-run ${p.tests || "its tests"}`);
      }
    }
  }
  if (p.tests) {
    if (!fs.existsSync(p.tests)) {
      out.push(`${ref}: declares tests ${p.tests}, which does not exist`);
    } else {
      // The comment above this block promised the covering test is reachable
      // from CI; only existsSync was implemented. A covering test nothing runs
      // is the same shape of claim as an unverified marker.
      const base = p.tests.split("/").pop();
      let wired = false;
      const wfDir = ".github/workflows";
      try {
        for (const f of fs.readdirSync(wfDir)) {
          if (fs.readFileSync(`${wfDir}/${f}`, "utf8").includes(base)) { wired = true; break; }
        }
      } catch { /* no workflows dir: reported as not wired */ }
      if (!wired) {
        out.push(`${ref}: covering test ${p.tests} is not invoked from any CI workflow — a test nothing runs cannot attest the patch`);
      }
    }
  }
}
for (const line of out) console.log(line);
')
if [ -n "$PATCH_PROBLEMS" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && report "LOCAL PATCH $line"
  done <<EOF
$PATCH_PROBLEMS
EOF
fi

# ── Orphaned tooling: neither tracked nor ignored ───────────────────────────
#
# The #108 failure mode generalizes. A tool under scripts/ or schema/ that is
# neither tracked nor ignored is in exactly the state the review bundle was in:
# working on this machine, absent from every clone, and invisible because
# nothing ever asks. It accumulates silently — a second workstation, or the
# next agent, simply does not have it.
#
# This is a WARNING, not a failure, and the reason matters: CI runs on a clean
# checkout where orphans do not exist by construction, so failing here would
# be a check that can only ever fire locally and never in the gate. Escalating
# it would also make one agent's in-progress, not-yet-committed work break
# another agent's build in a shared checkout. It is reported so the state is
# named rather than assumed.
# ── Reachability: a tool nobody tracked can invoke is only half-delivered ───
#
# #108 was "the tools do not survive a fresh clone". Tracking them fixes that
# for the FILES. It does not fix it for the CALLERS: most of these tools are
# invoked from roster skill prose under .harness/skills/, .claude/commands/
# and .agents/skills/, all of which are deliberately machine-local. A fresh
# clone therefore gets every gate and, for some of them, nothing that runs it.
#
# The decision (see scripts/REVIEW-BUNDLE.md) is to keep the callers local and
# make the consequence measurable rather than to fork the roster install into
# this repo. This block is that measurement: it names, per tool, whether any
# TRACKED file actually invokes it. Reported, never failed — a tool reachable
# only from the roster install is the accepted design, not a defect.
# Reachability is TRANSITIVE and starts from the roots a fresh clone actually
# executes: the CI workflows and the Makefile. A mention inside a .md or a
# .json is documentation, not a caller — counting those was the first version
# of this check, and it reported almost everything "reached" while proving
# nothing. Only executable carriers (.sh, .js, .yml, Makefile) form edges.
INVOCABLE="scripts/review-normalize.js scripts/check-review-convergence.js
scripts/check-scope-diff.sh scripts/xruntime-review.js scripts/xruntime-exec.sh
scripts/review-verdict-assemble.js scripts/review-bundle-verify.js"

UNREACHED=$(git ls-files -- '*.sh' '*.js' '*.yml' '*.yaml' Makefile 2>/dev/null | node -e '
const fs = require("fs"), path = require("path");
const targets = process.argv.slice(1);
const carriers = fs.readFileSync(0, "utf8").split("\n").filter(Boolean);

// One read per carrier, not one per (carrier, tool) pair.
const byBase = new Map();
for (const c of carriers) byBase.set(path.basename(c), c);
const edges = new Map();
for (const c of carriers) {
  let text = "";
  try { text = fs.readFileSync(c, "utf8"); } catch { continue; }
  const out = new Set();
  for (const [base, target] of byBase) {
    if (target === c) continue;
    const re = new RegExp("(node|bash|sh|\\./)[^\"\x27\\n]*" + base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (re.test(text)) out.add(target);
  }
  edges.set(c, out);
}

// BFS from the roots a fresh clone executes on its own.
const roots = carriers.filter((c) => c.startsWith(".github/workflows/") || c === "Makefile");
const seen = new Set(roots);
const queue = [...roots];
while (queue.length) {
  for (const nxt of edges.get(queue.shift()) || []) {
    if (!seen.has(nxt)) { seen.add(nxt); queue.push(nxt); }
  }
}
for (const t of targets) if (!seen.has(t)) console.log("  " + t);
' $INVOCABLE)
if [ -n "$UNREACHED" ]; then
  echo "" >&2
  echo "check-review-bundle-tracked: NOTE — tracked, verified, and invoked by no tracked caller:" >&2
  for u in $UNREACHED; do echo "    $u" >&2; done
  echo "  These run only from the machine-local roster install (.harness/skills/," >&2
  echo "  .claude/commands/, .agents/skills/). A fresh clone has the tool but" >&2
  echo "  nothing that runs it until /recruit installs the roster. Deliberate —" >&2
  echo "  see the Reachability section of scripts/REVIEW-BUNDLE.md." >&2
fi

ORPHANS=$(git ls-files --others --exclude-standard -- scripts schema 2>/dev/null)
if [ -n "$ORPHANS" ]; then
  echo "" >&2
  echo "check-review-bundle-tracked: NOTE — tooling present but neither tracked nor ignored:" >&2
  printf '%s\n' "$ORPHANS" | sed 's/^/    /' >&2
  echo "  Each of these exists only in this working tree. Track it, ignore it," >&2
  echo "  or land the PR that owns it — leaving it in this state is how #108 happened." >&2
fi

if [ "$FAILED" -ne 0 ]; then
  echo "" >&2
  echo "check-review-bundle-tracked: FAIL." >&2
  echo "  Track the listed files (git add), and remove any bundle entry from" >&2
  echo "  .gitignore / .git/info/exclude. Local-only roster infrastructure" >&2
  echo "  (briefs/, skills-meta/, kb/, .harness/, .agents/) stays untracked —" >&2
  echo "  the bundle does not." >&2
  exit 1
fi

echo "check-review-bundle-tracked: OK — bundle tracked, un-ignored, and sha-matched."
