// scripts/lib/redgreen-scratch.js — CommonJS
// Low-level, git/filesystem-facing primitives for red-before-green check
// verification (spec: specs/pipeline-loop-convergence.md FR-035..FR-039,
// A-6). Split out of scripts/check-review-convergence.js (FIX-1 follow-on:
// the round/strike/audit/breaker extraction alone left the main script over
// the repo's 500-line limit; this module carries the same "split by
// responsibility" principle one level further) so each file stays under
// that limit.
//
// Responsibility boundary: this module owns "safely run a check command
// against an isolated pre-fix copy of the repo, never mutating the real
// tree or .git" — extracting the pre-fix tree via `git archive | tar -x`
// (never `git worktree add`, FR-022/A-2), running the red/green halves with
// a cwd jail and timeout, and the exit-code interpretation (A-6:
// 0=pass, 1=assertion fired, >=2=error). The main script keeps the
// per-finding orchestration (which findings need checking, how a raw
// outcome becomes a violation) and everything else (CLI parsing, structural
// finding checks, round/strike/audit — the last three via
// scripts/lib/review-convergence-rules.js).
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const crypto = require("crypto");
const { execFileSync } = require("child_process");

function isFullSha(s) {
  return typeof s === "string" && /^[0-9a-f]{40}$/.test(s);
}

// Verify a single ratcheted check: red against pre_fix_sha (scratch, overlay-only),
// green against the current tree. Never mutates the real repo or .git.
// Guard clauses only — the two verification branches are extracted below.
function verifyCheck({ repoRoot, checkRelPath, preFixSha, recordedBlob, redVerified, timeoutMs }) {
  // FIX-A (RGC-1): path containment — single choke point above every resolve,
  // hash-object, copyFileSync and node exec (and above the convergence-gate
  // red-proof fast path, RGC-5). Reject before any I/O touches checkRelPath.
  if (typeof checkRelPath !== "string" || checkRelPath === "" || path.isAbsolute(checkRelPath)) {
    return { inconclusive: true, reason: `check path escapes repo root: ${checkRelPath}` };
  }
  const relToRoot = path.relative(repoRoot, path.resolve(repoRoot, checkRelPath));
  if (relToRoot === "" || relToRoot.startsWith("..") || path.isAbsolute(relToRoot)) {
    return { inconclusive: true, reason: `check path escapes repo root: ${checkRelPath}` };
  }
  if (!isFullSha(preFixSha)) {
    return { inconclusive: true, reason: `pre_fix_sha is not a full 40-hex sha: ${preFixSha}` };
  }
  try {
    execFileSync("git", ["rev-parse", "-q", "--verify", `${preFixSha}^{commit}`], {
      cwd: repoRoot,
      stdio: "pipe",
    });
  } catch (e) {
    return { inconclusive: true, reason: `pre_fix_sha unreachable: ${preFixSha}` };
  }

  const checkAbsPath = path.resolve(repoRoot, checkRelPath);
  if (!fs.existsSync(checkAbsPath)) {
    return { inconclusive: true, reason: `check file not found in current tree: ${checkRelPath}` };
  }

  let currentBlob;
  try {
    currentBlob = execFileSync("git", ["hash-object", checkRelPath], { cwd: repoRoot, stdio: "pipe" })
      .toString()
      .trim();
  } catch (e) {
    return { inconclusive: true, reason: `git hash-object failed for ${checkRelPath}` };
  }

  // FIX-B (CGF-3/4/5, INV-B): a matching check_blob is only proof of a prior
  // red run when the persisted finding says so (red_verified === true). A
  // blob match with no such proof falls through to the scratch red run
  // instead of short-circuiting green-only.
  const blobMatches = recordedBlob !== null && recordedBlob === currentBlob;
  if (blobMatches && redVerified === true) {
    return reverifyGreenOnly(checkAbsPath, repoRoot, timeoutMs, currentBlob);
  }
  return verifyViaScratch({ repoRoot, checkRelPath, checkAbsPath, preFixSha, recordedBlob, currentBlob, timeoutMs });
}

// Already red-verified at this exact blob — only the green half re-runs.
function reverifyGreenOnly(checkAbsPath, repoRoot, timeoutMs, currentBlob) {
  const green = runGreenPhase(checkAbsPath, repoRoot, timeoutMs);
  if (green.inconclusive) return { inconclusive: true, reason: green.reason, check_blob: currentBlob };
  return { red_verified: true, check_blob: currentBlob, greenFailed: green.code !== 0 };
}

// First-time (or blob-mismatch re-)verification: extract the pre-fix tree,
// run red there, then run green against the current tree.
function verifyViaScratch({ repoRoot, checkRelPath, checkAbsPath, preFixSha, recordedBlob, currentBlob, timeoutMs }) {
  let scratchDir;
  try {
    scratchDir = extractPreFixTree(repoRoot, preFixSha, checkRelPath, checkAbsPath);
  } catch (e) {
    return { inconclusive: true, reason: e.message, check_blob: currentBlob };
  }

  try {
    const scratchCheckPath = path.resolve(scratchDir, checkRelPath);
    const red = runRedPhase(scratchCheckPath, scratchDir, repoRoot, timeoutMs);
    if (red.inconclusive) return { inconclusive: true, reason: red.reason, check_blob: currentBlob };
    if (red.code === 0) return { vacuous: true, check_blob: currentBlob };
    if (red.code >= 2) {
      return {
        inconclusive: true,
        reason: `red command exited ${red.code} (>=2, error/setup)`,
        check_blob: currentBlob,
      };
    }

    // red.code === 1: assertion fired as expected.
    // FIX-B label refinement (spec Defect #1): classify `weakened` vs
    // `greenFailed` on an ACTUAL blob mismatch, not merely "had a recorded
    // blob" — otherwise the blob-match-but-unproven fall-through (FIX-B) and
    // a first-time (null) green-failure both get mislabeled. The
    // `recordedBlob !== null` conjunct is REQUIRED: without it, a bare
    // `!== currentBlob` would misclassify the first-time green-failure as
    // weakened.
    const wasPreviouslyVerified = recordedBlob !== null && recordedBlob !== currentBlob;
    const green = runGreenPhase(checkAbsPath, repoRoot, timeoutMs);
    if (green.inconclusive) return { inconclusive: true, reason: green.reason, check_blob: currentBlob };

    return {
      red_verified: true,
      check_blob: currentBlob,
      weakened: wasPreviouslyVerified && green.code !== 0,
      greenFailed: !wasPreviouslyVerified && green.code !== 0,
    };
  } finally {
    try {
      fs.rmSync(scratchDir, { recursive: true, force: true });
    } catch (e) {
      // best-effort cleanup; scratch dir is outside the repo, never blocks the verdict
    }
  }
}

// Extracts the pre-fix tree via `git archive | tar -x` into a fresh scratch
// directory — never `git worktree add` (FR-022/A-2) — then overlays ONLY the
// new check file (copied in from the CURRENT tree). Throws on any setup
// failure; the caller treats that as inconclusive.
function extractPreFixTree(repoRoot, preFixSha, checkRelPath, checkAbsPath) {
  const scratchDir = fs.mkdtempSync(path.join(os.tmpdir(), "check-review-convergence-"));
  try {
    execFileSync("sh", ["-c", `git archive ${preFixSha} | tar -x -C "${scratchDir}"`], {
      cwd: repoRoot,
      stdio: "pipe",
    });
  } catch (e) {
    fs.rmSync(scratchDir, { recursive: true, force: true });
    throw new Error(`git archive extraction failed for ${preFixSha}`);
  }

  const scratchCheckPath = path.resolve(scratchDir, checkRelPath);
  try {
    fs.mkdirSync(path.dirname(scratchCheckPath), { recursive: true });
    fs.copyFileSync(checkAbsPath, scratchCheckPath);
  } catch (e) {
    fs.rmSync(scratchDir, { recursive: true, force: true });
    throw new Error(`overlay of check file failed: ${e.message}`);
  }

  return scratchDir;
}

// Runs the red half: the check executed inside the scratch (pre-fix) tree.
function runRedPhase(scratchCheckPath, scratchDir, repoRoot, timeoutMs) {
  return runNode(scratchCheckPath, scratchDir, repoRoot, timeoutMs);
}

// FIX-B (RGC-3): whole-tree snapshot, the xruntime-exec.sh idiom
// (`git status --porcelain -uall | sha256sum`). Read-only — never
// add/commit/stash/checkout/worktree.
function snapshotTree(repoRoot) {
  const status = execFileSync("git", ["status", "--porcelain", "-uall"], { cwd: repoRoot, stdio: "pipe" });
  return crypto.createHash("sha256").update(status).digest("hex");
}

// Runs the green half: the check executed against the current (real) tree.
// FIX-B (RGC-2): before/after whole-tree snapshot around the green run.
// D-5 (binding): if the snapshot cannot be taken (non-repo, git binary/
// permission failure) FAIL CLOSED (inconclusive) — never skip, never crash.
function runGreenPhase(checkAbsPath, repoRoot, timeoutMs) {
  let before;
  try {
    before = snapshotTree(repoRoot);
  } catch (e) {
    return { inconclusive: true, reason: "green-phase tree snapshot unavailable (cannot verify read-only)" };
  }
  const result = runNode(checkAbsPath, repoRoot, repoRoot, timeoutMs);
  let after;
  try {
    after = snapshotTree(repoRoot);
  } catch (e) {
    return { inconclusive: true, reason: "green-phase tree snapshot failed after green run" };
  }
  if (before !== after) {
    return { inconclusive: true, reason: "green run mutated the working tree" };
  }
  return result;
}

// Runs `node <absPath>` honoring the red-command exit convention (A-6):
// 0 = pass, 1 = assertion fired, >=2 = error. NODE_PATH points at the live
// repo's node_modules so scratch-tree runs have dependency availability
// without repo mutation.
function runNode(absPath, cwd, repoRoot, timeoutMs) {
  try {
    execFileSync("node", [absPath], {
      cwd,
      timeout: timeoutMs,
      env: Object.assign({}, process.env, { NODE_PATH: path.resolve(repoRoot, "node_modules") }),
      stdio: "pipe",
    });
    return { code: 0 };
  } catch (e) {
    if (e.signal) {
      return { inconclusive: true, reason: `red/green command timed out or was killed (signal ${e.signal})` };
    }
    if (typeof e.status === "number") {
      return { code: e.status };
    }
    return { inconclusive: true, reason: `red/green command failed to run: ${e.message}` };
  }
}

// ── per-finding orchestration (D1, r5-trace-enforcement) ────────────────
// Moved from scripts/check-review-convergence.js (its documented home,
// per this module's docblock) to keep the main gate file under the repo's
// 500-line limit as the trace dispatch is added — zero behavior change.

// Verifies a single finding's ratcheted check and translates the raw
// verifyCheck() result into report entries (violations + a checks[] entry).
function verifyFinding(f, repoRoot, timeoutMs) {
  if (f.pre_fix_sha === null || f.pre_fix_sha === undefined) {
    // FR-034: uncommitted-tree task — accepted, flagged, not a violation.
    return {
      checkEntry: {
        check: f.check,
        fingerprint: f.fingerprint,
        red_verified: null,
        flagged: "no pre_fix_sha recorded (uncommitted-tree task, FR-034)",
      },
      violations: [],
      inconclusive: false,
    };
  }

  const result = verifyCheck({
    repoRoot,
    checkRelPath: f.check,
    preFixSha: f.pre_fix_sha,
    recordedBlob: f.check_blob || null,
    redVerified: f.red_verified,
    timeoutMs,
  });

  return {
    // E-3: checks[] is keyed by (check, fid) with a fingerprint fallback for
    // legacy findings that predate fid — the normalizer's gate-report lookup
    // (buildGateCheckIndex) uses this exact same key shape.
    checkEntry: Object.assign({ check: f.check, fingerprint: f.fingerprint, fid: f.fid || null }, result),
    violations: buildRedGreenViolations(result, f),
    inconclusive: !!result.inconclusive,
  };
}

// Translates a verifyCheck() outcome into 0-2 violation report entries.
function buildRedGreenViolations(result, f) {
  const violations = [];

  if (result.vacuous) {
    violations.push({
      type: "vacuous-check",
      cause: "unencodable-finding",
      fingerprint: f.fingerprint,
      check: f.check,
      detail: "red command exited 0 against the pre-fix tree — check never fails (FR-036)",
    });
  } else if (result.weakened) {
    violations.push({
      type: "weakened-check",
      cause: "unencodable-finding",
      fingerprint: f.fingerprint,
      check: f.check,
      detail: "check_blob mismatch and re-verification could not reproduce red (FR-038)",
    });
  }

  if (result.greenFailed) {
    violations.push({
      type: "green-failure",
      cause: "unencodable-finding",
      fingerprint: f.fingerprint,
      check: f.check,
      detail: "check does not pass against the current tree (FR-039)",
    });
  }

  return violations;
}

module.exports = { isFullSha, verifyCheck, verifyFinding, buildRedGreenViolations };
