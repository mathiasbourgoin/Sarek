#!/usr/bin/env node
// check-review-convergence.js — CommonJS
// Read-only mechanical gate deciding whether a NO-GO route-back is permitted
// (spec: specs/pipeline-loop-convergence.md, FR-021..FR-039; extended by
// specs/review-fanout-convergence.md FR-050..FR-085 + Amendments B-1..B-9;
// specs/review-v2-corrections.md Amendments E-1/E-3/E-4). Patterned on
// scripts/check-scope-diff.sh's exit contract.
//
// Override-aware (E-1): a `streak_override` on review.json with
// `round == review.round` and `by == "human"` suppresses the current round's
// novel-finding-streak strike (round-cap is never suppressible) — see
// scripts/lib/review/review-convergence-rules.js's computeStrikeMap. Reopened-strike
// rule (E-4): a round also strikes when it reopens >=1 HIGH+ finding
// (`reopened_at_round == round`), not only on a novel one. checks[] entries
// are keyed by (check, fid) with a fingerprint fallback (E-3).
//
// Usage: node scripts/check-review-convergence.js <review.json path>
//   [--static] [--max-rounds N] [--timeout S] [--strikes N]
//
// Exit contract:
//   0 = pass (no violations, no degraded input)
//   1 = design violation(s) — top-level `cause` is one of round-cap |
//       unencodable-finding | novel-finding-streak (precedence:
//       unencodable-finding > novel-finding-streak > round-cap; B-5/FR-059)
//   2 = degraded input (absent/malformed review.json, an unknown flag, or an
//       inconclusive red/green verification — fail-closed; FR-023, FR-036)
//   3 = process-incomplete-only (missing/incomplete rounds_audit entry for
//       the current round, B-5/FR-079) — repair the draft and re-gate, do
//       NOT route; top-level `cause` is never "process-incomplete"
//
// Read-only guarantee (FR-022, amended A-2): this script MUST NOT modify any
// repo file or `.git` state. Pre-fix trees are extracted via
// `git archive <sha> | tar -x` into a scratch directory — `git worktree add`
// is NEVER used. Red/green commands run inside scratch copies with a cwd
// jail and a per-command timeout (default 120s, --timeout).
//
// Check-value contract: a finding's `check` field MUST be a node-runnable
// file path (invoked as `node <path>`, A-6). A spec-level `CHECK-N` id with
// no corresponding file is recorded by roster-review but is NOT red/green
// executed here (see roster-review.md §5.5) — out of mechanical-verification
// scope, not a violation.
//
// Legacy handling (B-8): when review.json lacks the `round` key (the
// physical per-cycle counter), strike classification and the rounds_audit
// completeness check are SKIPPED with a warning — the 17 pre-existing
// fixtures (keyed on `no_go_round` only) pass unmodified.
//
// Config-echo contract (B-4, FIX-6): the JSON report ALWAYS includes
// `config: {max_rounds, strikes, static}` on every exit code — this is how a
// consumer (roster-review) detects a stale copy of this script predating
// review-fanout-convergence (its report would lack `config` entirely).
//
// Module boundary (FIX-1, extended D1/C-5/r5-trace-enforcement): round/
// strike/audit/breaker rule functions (isNovelStrikeFinding, computeStrikeMap,
// computeStreakViolation, computeMissingAuditViolation,
// computeCrossRuntimeWarnings, selectCause, HIGH_PLUS) live in
// scripts/lib/review/review-convergence-rules.js; the scratch-tree git
// mechanics AND per-finding red/green orchestration (isFullSha, verifyCheck,
// verifyFinding, buildRedGreenViolations) live in
// scripts/lib/review/redgreen-scratch.js; trace obligation/coverage/
// correspondence PURE rules live in scripts/lib/review/review-trace-rules.js
// (FR-178) and the trace mechanism's I/O-facing orchestration (task-slug
// validation, sibling-path derivation, read-only trace/journal file reads)
// lives in scripts/lib/review/review-trace-dispatch.js (C-5 line-budget
// contingency — never trim the trace check itself). This file keeps CLI
// parsing, input validation, structural finding checks, and thin
// orchestration/dispatch only.
"use strict";

const fs = require("fs");
const path = require("path");
const {
  HIGH_PLUS,
  computeStrikeMap,
  computeStreakViolation,
  computeMissingStrikeWarnings,
  computeMissingAuditViolation,
  computeCrossRuntimeWarnings,
  selectCause,
} = require("./lib/review/review-convergence-rules");
const { isFullSha, verifyFinding } = require("./lib/review/redgreen-scratch");
const { evaluateTrace } = require("./lib/review/review-trace-dispatch");

const KNOWN_FLAGS = new Set(["--static", "--max-rounds", "--timeout", "--strikes"]);
const DEFAULT_STRIKES = 2;

// B-4: unknown flags are rejected (exit 2) so a stale/mismatched invocation
// fails loudly instead of silently ignoring a new flag.
function parseArgs(argv) {
  const out = { reviewPath: null, static: false, maxRounds: 5, timeout: 120, strikes: DEFAULT_STRIKES, unknownFlag: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--static") {
      out.static = true;
    } else if (a === "--max-rounds") {
      out.maxRounds = parseInt(argv[++i], 10);
    } else if (a === "--timeout") {
      out.timeout = parseInt(argv[++i], 10);
    } else if (a === "--strikes") {
      out.strikes = parseInt(argv[++i], 10);
    } else if (a.startsWith("--") && !KNOWN_FLAGS.has(a)) {
      out.unknownFlag = a;
    } else if (!out.reviewPath) {
      out.reviewPath = a;
    } else {
      out.unknownFlag = a;
    }
  }
  return out;
}

function fail(code, message) {
  if (message) process.stderr.write(`check-review-convergence: ${message}\n`);
  process.exit(code);
}

function findingFingerprint(f) {
  return typeof f.fingerprint === "string" ? f.fingerprint : `${f.path || "?"}:${f.line ?? 0}:${f.category || "?"}`;
}

// ── argument + input validation ─────────────────────────────────────────
// Parses argv, reads review.json, and validates its shape. Exits (via fail)
// on any usage or degraded-input error — never returns in that case.
function validateArgsAndReview(argv) {
  const args = parseArgs(argv);
  if (args.unknownFlag) fail(2, `unknown flag or extra argument: ${args.unknownFlag}`);
  if (!args.reviewPath) fail(2, "usage: <review.json path> [--static] [--max-rounds N] [--timeout S] [--strikes N]");
  if (Number.isNaN(args.maxRounds) || args.maxRounds < 1) fail(2, "--max-rounds must be a positive integer");
  if (Number.isNaN(args.timeout) || args.timeout < 1) fail(2, "--timeout must be a positive integer (seconds)");
  if (Number.isNaN(args.strikes) || args.strikes < 1) fail(2, "--strikes must be a positive integer");

  const reviewAbsPath = path.resolve(process.cwd(), args.reviewPath);
  const review = readReviewJson(reviewAbsPath, args.reviewPath);
  const roundState = deriveGateRoundInputs(review);

  return Object.assign({ args, review, reviewAbsPath }, roundState);
}

// Reads and parses review.json. Exits (via fail) on any absent/unreadable/
// malformed/non-object input — never returns in that case.
function readReviewJson(reviewPath, displayPath) {
  if (!fs.existsSync(reviewPath)) fail(2, `review.json not found: ${displayPath}`);

  let raw;
  try {
    raw = fs.readFileSync(reviewPath, "utf8");
  } catch (e) {
    fail(2, `cannot read review.json: ${e.message}`);
    return;
  }

  let review;
  try {
    review = JSON.parse(raw);
  } catch (e) {
    fail(2, `review.json is not valid JSON: ${e.message}`);
    return;
  }
  if (!review || typeof review !== "object" || Array.isArray(review)) {
    fail(2, "review.json must be a JSON object");
    return;
  }
  return review;
}

// FIX-A (CGF-1/CGF-2, INV-A): an absent `findings` key is legacy-safe ([]);
// a PRESENT but non-array `findings` value is degraded input and must fail
// closed, never be silently coerced to an empty finding set.
function coerceFindings(review) {
  if (!Object.prototype.hasOwnProperty.call(review, "findings")) return [];
  if (!Array.isArray(review.findings)) {
    fail(2, "review.json field findings is present but not an array (degraded input)");
    return; // unreachable: fail() exits
  }
  return review.findings;
}

// Derives no_go_round (FR-030 legacy handling), the physical `round` counter
// (B-8 legacy handling), the findings array, and the B-7 re-keyed
// carry-forward-inconsistency warning from a validated review object.
//
// NOT the lifecycle witness: this is the GATE's own single-file input
// derivation (legacy-round detection, no_go_round validation) — narrower and
// differently-shaped than scripts/lib/review/review-lifecycle.js's deriveRoundState,
// which derives the NEXT draft's round/cycle from the PRIOR verdict. Named
// distinctly (deriveGateRoundInputs) to reserve `deriveRoundState` for that
// lifecycle witness and avoid the two being confused (review finding LOW-1).
// r5-trace-enforcement: `cycle` alongside `round` (same hasOwnProperty
// presence check) purely for the trace mechanism's (cycle, round) scoping —
// absence or a non-numeric value is legacy-safe (null), never fatal.
function deriveCycle(review) {
  if (!Object.prototype.hasOwnProperty.call(review, "cycle")) return null;
  return typeof review.cycle === "number" && Number.isFinite(review.cycle) ? review.cycle : null;
}

function deriveGateRoundInputs(review) {
  const warnings = [];

  let noGoRound = 0;
  let legacyNoGoRound = false;
  if (Object.prototype.hasOwnProperty.call(review, "no_go_round")) {
    if (typeof review.no_go_round !== "number" || !Number.isFinite(review.no_go_round) || review.no_go_round < 0) {
      fail(2, "review.json field no_go_round must be a non-negative number");
      return;
    }
    noGoRound = review.no_go_round;
  } else {
    legacyNoGoRound = true;
    warnings.push("legacy review.json: no_go_round key absent — treating as round 0");
  }

  // B-8: `round` (physical per-cycle verdict counter) is a distinct field from
  // `no_go_round` (qualifying-only backstop). Its absence means legacy
  // review.json — strike classification and the rounds_audit completeness
  // check are skipped entirely (below, in main()), not just defaulted.
  let round = null;
  let legacyRound = false;
  if (Object.prototype.hasOwnProperty.call(review, "round")) {
    if (typeof review.round !== "number" || !Number.isFinite(review.round) || review.round < 0) {
      fail(2, "review.json field round must be a non-negative number");
      return;
    }
    round = review.round;
  } else {
    legacyRound = true;
    warnings.push("legacy review.json: round key absent — skipping strike and rounds_audit checks (B-8)");
  }

  const cycle = deriveCycle(review);
  const findings = coerceFindings(review);

  // B-7: A-11 re-keyed to the physical `round` counter (retiring the old
  // no_go_round-based form) — warn when round is absent/0 while a finding
  // carries first_seen_round > 0 (unverifiable carry-forward inconsistency).
  if ((round === null || round === 0) && findings.some((f) => typeof f.first_seen_round === "number" && f.first_seen_round > 0)) {
    warnings.push(
      "inconsistency: round is absent/0 but a finding carries first_seen_round > 0 (unverifiable carry-forward, A-11/B-7)"
    );
  }

  return { findings, warnings, noGoRound, legacyNoGoRound, round, legacyRound, cycle };
}

// ── structural finding checks ────────────────────────────────────────────
// FR-012/013/025/027 (ratchet obligation + unencodable-finding escalation)
// and the FIX-2 hardening: a RESOLVED HIGH+ finding with missing/non-numeric
// round provenance must not silently escape the ratchet.
function computeFindingViolations(findings) {
  const violations = [];

  for (const f of findings) {
    if (!f || typeof f !== "object") continue;
    if (!HIGH_PLUS.has(f.severity)) continue;

    const fingerprint = findingFingerprint(f);
    const accepted = f.status === "ACCEPTED";

    if (f.status === "RESOLVED" && !accepted) {
      const firstSeenValid = typeof f.first_seen_round === "number" && Number.isFinite(f.first_seen_round);
      const resolvedValid = typeof f.resolved_round === "number" && Number.isFinite(f.resolved_round);

      if (!firstSeenValid || !resolvedValid) {
        // Missing/non-numeric round provenance — cannot prove this was (or
        // wasn't) a same-round raise+resolve, so it cannot be waved through.
        violations.push({
          type: "missing-round-provenance",
          cause: "unencodable-finding",
          fingerprint,
          detail:
            "HIGH+ finding RESOLVED but missing/non-numeric first_seen_round or resolved_round — cannot verify ratchet exemption",
        });
      } else if (f.resolved_round > f.first_seen_round && (f.check === null || f.check === undefined)) {
        // FR-012/FR-025: crossed a loop-back round with no linked check.
        violations.push({
          type: "resolved-without-check",
          cause: "unencodable-finding",
          fingerprint,
          detail: "HIGH+ finding RESOLVED across a loop-back round with no linked check (FR-012)",
        });
      }
    }

    // FR-027: unencodable + not ACCEPTED -> design-not-converging.
    if (f.check_encodable === false && !accepted) {
      violations.push({
        type: "unencodable-finding",
        cause: "unencodable-finding",
        fingerprint,
        detail: "HIGH+ finding marked check_encodable: false and not human-ACCEPTED (FR-027)",
      });
    }
  }

  return violations;
}

// ── red-before-green verification (full mode only) ──────────────────────
// FR-035..FR-039. Returns { violations, checks, anyInconclusive }.
function runRedGreenVerification(findings, args) {
  const violations = [];
  const checks = [];
  let anyInconclusive = false;
  if (args.static) return { violations, checks, anyInconclusive };

  const repoRoot = process.cwd();
  for (const f of findings) {
    if (!f || typeof f !== "object") continue;
    if (!f.check || typeof f.check !== "string") continue;

    const outcome = verifyFinding(f, repoRoot, args.timeout * 1000);
    checks.push(outcome.checkEntry);
    violations.push(...outcome.violations);
    if (outcome.inconclusive) anyInconclusive = true;
  }

  return { violations, checks, anyInconclusive };
}

// US-1/US-3 (B-8): only evaluated when `round` is present — legacy
// review.json skips strike classification and the audit check entirely.
// Extracted from main() (FIX-2) — one function, one responsibility: derive
// this round's strike state plus any streak/audit violations and warnings.
function evaluateStrikesAndAudit(review, round, legacyRound, findings, strikesRequired) {
  if (legacyRound) return { currentRoundStrike: null, violations: [], warnings: [] };

  const { strikeByRound, currentStrike } = computeStrikeMap(review, round, findings);
  const violations = [];

  const streakViolation = computeStreakViolation(strikeByRound, round, strikesRequired);
  if (streakViolation) violations.push(streakViolation);

  const auditViolation = computeMissingAuditViolation(review, round);
  if (auditViolation) violations.push(auditViolation);

  const warnings = computeMissingStrikeWarnings(review, round); // FIX-4
  return { currentRoundStrike: currentStrike, violations, warnings };
}

// Assembles the JSON report (extracted from main(), FIX-2). `cause` is
// derived here so it always reflects the final, fully-accumulated
// violations list. `trace` (FR-176) is emitted on every exit code, including
// exit 2/legacy-skip — the anti-stale-script signal roster-review checks
// exactly as it checks `config.strikes`.
function buildReport({ args, noGoRound, legacyNoGoRound, round, legacyRound, currentRoundStrike, warnings, violations, checks, trace }) {
  return {
    mode: args.static ? "static" : "full",
    no_go_round: noGoRound,
    max_rounds: args.maxRounds,
    legacy_no_go_round: legacyNoGoRound,
    round,
    legacy_round: legacyRound,
    current_round_strike: currentRoundStrike,
    config: { max_rounds: args.maxRounds, strikes: args.strikes, static: args.static },
    cause: selectCause(violations),
    warnings,
    violations,
    checks,
    trace,
  };
}

// FR-059/B-5 exit precedence (extracted from main(), FIX-2), extended
// FR-173/r5-trace-enforcement: an inconclusive red/green run OR a
// current-cycle malformed trace line (traceDegraded) — both fail-closed (2)
// — outranks a design violation (1), which outranks a process-incomplete-
// only report (3), which outranks a clean pass (0). `traceDegraded`
// defaulting to falsy on an undefined argument preserves backward
// compatibility for any caller that predates this parameter.
function decideExit(report, anyInconclusive, traceDegraded) {
  if (anyInconclusive || traceDegraded) return 2;
  const hasDesignViolation = report.violations.some((v) => v.cause !== "process-incomplete");
  if (hasDesignViolation) return 1;
  const hasProcessIncompleteOnly = report.violations.some((v) => v.cause === "process-incomplete");
  if (hasProcessIncompleteOnly) return 3;
  return 0;
}

// ── main ─────────────────────────────────────────────────────────────────
function main() {
  const { args, review, findings, warnings, noGoRound, legacyNoGoRound, round, legacyRound, cycle, reviewAbsPath } =
    validateArgsAndReview(process.argv.slice(2));

  const violations = [];

  // FR-026: cap violation.
  if (noGoRound >= args.maxRounds) {
    violations.push({
      type: "round-cap",
      cause: "round-cap",
      detail: `no_go_round (${noGoRound}) >= max-rounds (${args.maxRounds})`,
    });
  }

  violations.push(...computeFindingViolations(findings));

  const strikeAudit = evaluateStrikesAndAudit(review, round, legacyRound, findings, args.strikes);
  violations.push(...strikeAudit.violations);
  warnings.push(...strikeAudit.warnings);
  warnings.push(...computeCrossRuntimeWarnings(review));

  const redGreen = runRedGreenVerification(findings, args);
  violations.push(...redGreen.violations);

  const trace = evaluateTrace({ review, round, cycle, legacyRound, reviewAbsPath });
  if (trace.fail) fail(trace.fail.code, trace.fail.message);
  violations.push(...trace.violations);
  warnings.push(...trace.warnings);

  const report = buildReport({
    args,
    noGoRound,
    legacyNoGoRound,
    round,
    legacyRound,
    currentRoundStrike: strikeAudit.currentRoundStrike,
    warnings,
    violations,
    checks: redGreen.checks,
    trace: trace.traceBlock,
  });

  // Always emit the JSON report, even on exit 0 — unlike check-scope-diff.sh's
  // silent-on-clean convention, roster-review must merge `checks[]`
  // (red_verified/check_blob) back into review.json on every verdict,
  // including a clean GO round (A-1/A-2 gate-before-write order).
  process.stdout.write(JSON.stringify(report, null, 2) + "\n");
  process.exit(decideExit(report, redGreen.anyInconclusive, trace.degraded));
}

if (require.main === module) {
  main();
}

module.exports = { parseArgs, isFullSha, evaluateStrikesAndAudit, buildReport, decideExit };
