// scripts/lib/review-convergence-rules.js — CommonJS
// Pure, dependency-free rule functions for the review-fanout-convergence
// mechanics (spec: specs/review-fanout-convergence.md FR-050..FR-085,
// Amendments B-1..B-9) — split out of scripts/check-review-convergence.js
// (review finding FIX-1) so each file stays under the repo's 500-line limit.
//
// Responsibility boundary: this module owns two-strike novel-finding
// escalation (US-1), delta-scoped loop-back rounds_audit completeness
// (US-3), the cross-runtime circuit breaker's structural warning (US-2),
// and top-level cause selection (FR-059). The main script keeps CLI parsing,
// orchestration, and red/green check verification.
"use strict";

const HIGH_PLUS = new Set(["CRITICAL", "HIGH"]);

// ── two-strike novel-finding escalation (US-1, B-1) ──────────────────────
// A physical round r >= 2 is a strike iff it contains >=1 novel HIGH+
// non-scope finding (first_seen_round == r) that is neither same-round-
// resolved nor ACCEPTED (FR-053). Round 1 never strikes (FR-053, AC-4).
function isNovelStrikeFinding(f, round) {
  if (!f || typeof f !== "object") return false;
  if (!HIGH_PLUS.has(f.severity)) return false;
  if (f.category === "scope") return false; // EC-4: scope findings never strike
  if (f.status === "ACCEPTED") return false; // EC-1
  if (typeof f.first_seen_round !== "number" || f.first_seen_round !== round) return false;
  if (f.status === "RESOLVED" && f.resolved_round === f.first_seen_round) return false; // same-round-resolved
  return true;
}

// E-4 (clarified in review, review-v2-corrections.md): a round also strikes
// when it reopened >=1 HIGH+ finding (regardless of which round first raised
// it) — a regression-heavy loop-back round must not be invisible to
// two-strike escalation just because nothing is "novel" this round. Carries
// the SAME two guards as isNovelStrikeFinding: ACCEPTED wins over reopen —
// the ACCEPT-permanence contract (roster-review.md's "acceptance permanently
// waives the invariant") means a human-waived finding must not strike even
// if it was mechanically reopened; and scope-category findings never strike
// (EC-4), reopened or not.
function isReopenedStrikeFinding(f, round) {
  if (!f || typeof f !== "object") return false;
  if (!HIGH_PLUS.has(f.severity)) return false;
  if (f.category === "scope") return false; // EC-4
  if (f.status === "ACCEPTED") return false; // ACCEPT-permanence wins over reopen
  return f.reopened_at_round === round;
}

// E-1: a valid streak_override is scoped to the CURRENT round only — stale
// the moment the next verdict increments `round` (round-cap escalation is
// never overridable and is untouched by this check).
function isValidStreakOverride(review, currentRound) {
  const o = review && review.streak_override;
  return !!o && typeof o === "object" && o.round === currentRound && o.by === "human";
}

// B-1: past strikes are read from rounds_audit[].strike — journaled by
// roster-review, never recomputed here. Only the CURRENT round's strike is
// computed fresh, from findings, since that is point-in-time correct at gate
// time — UNLESS a valid current-round streak_override is present (E-1), in
// which case the journaled/forced `strike: false` is respected instead of
// recomputing (the override's whole purpose is to force this round to not
// strike). Returns a round -> strike boolean map covering every round seen.
// FIX-5: this non-recomputation invariant is what
// scripts/check-review-convergence-rules.test.js's "streak read from
// rounds_audit[].strike is not recomputed" fixture proves — a late ACCEPT of
// a round-2 finding must not erase a strike already journaled for round 2.
function computeStrikeMap(review, currentRound, findings) {
  const strikeByRound = new Map();
  const roundsAudit = Array.isArray(review.rounds_audit) ? review.rounds_audit : [];
  for (const entry of roundsAudit) {
    if (entry && typeof entry.round === "number" && typeof entry.strike === "boolean") {
      strikeByRound.set(entry.round, entry.strike);
    }
  }
  const overridden = isValidStreakOverride(review, currentRound);
  const currentStrike =
    !overridden &&
    currentRound >= 2 &&
    findings.some((f) => isNovelStrikeFinding(f, currentRound) || isReopenedStrikeFinding(f, currentRound));
  strikeByRound.set(currentRound, currentStrike);
  return { strikeByRound, currentStrike };
}

// FR-054/FR-055: violation when the last `strikesRequired` consecutive
// physical rounds (all >= 2) each classify as a strike; a strike-free round
// resets the streak (non-consecutive strikes never accumulate).
function computeStreakViolation(strikeByRound, currentRound, strikesRequired) {
  if (currentRound < 2) return null;
  for (let r = currentRound; r > currentRound - strikesRequired; r--) {
    if (r < 2 || strikeByRound.get(r) !== true) return null;
  }
  return {
    type: "novel-finding-streak",
    cause: "novel-finding-streak",
    detail: `${strikesRequired} consecutive round(s) (>= round 2) each classified as a strike, ending at round ${currentRound}`,
  };
}

// FIX-4 (LOW hardening): a PAST rounds_audit entry (2 <= round < currentRound)
// lacking a boolean `strike` field silently resets the streak in
// computeStrikeMap above (Map.get() returns undefined, never `true`) — this
// warning makes that prose-discipline dependency on roster-review visible
// instead of a silent no-op.
function computeMissingStrikeWarnings(review, currentRound) {
  const warnings = [];
  const roundsAudit = Array.isArray(review.rounds_audit) ? review.rounds_audit : [];
  for (const entry of roundsAudit) {
    if (!entry || typeof entry.round !== "number") continue;
    if (entry.round < 2 || entry.round >= currentRound) continue;
    if (typeof entry.strike !== "boolean") {
      warnings.push(
        `rounds_audit entry for round ${entry.round} lacks a boolean strike field — the streak silently resets on this omission`
      );
    }
  }
  return warnings;
}

// ── delta-scoped loop-back rounds_audit completeness (US-3, B-8) ────────
// FR-078/FR-079: on round >= 2, the current round's rounds_audit entry must
// be present and complete (reviewed_sha, fix_sha|fix_sha_reason,
// non-empty specialists_run with non-empty selection_reason each). Missing
// or incomplete -> process-incomplete (exit 3), never routed (B-5).
function computeMissingAuditViolation(review, currentRound) {
  if (currentRound < 2) return null;
  const roundsAudit = Array.isArray(review.rounds_audit) ? review.rounds_audit : [];
  const entry = roundsAudit.find((e) => e && e.round === currentRound);
  const incomplete = (detail) => ({ type: "missing-loopback-audit", cause: "process-incomplete", detail });

  if (!entry) return incomplete(`no rounds_audit entry for round ${currentRound}`);
  if (typeof entry.reviewed_sha === "undefined") {
    return incomplete(`rounds_audit entry for round ${currentRound} missing reviewed_sha`);
  }
  if (typeof entry.fix_sha === "undefined") {
    return incomplete(`rounds_audit entry for round ${currentRound} missing fix_sha`);
  }
  if (entry.fix_sha === null && (typeof entry.fix_sha_reason !== "string" || entry.fix_sha_reason.trim() === "")) {
    // EC-8: dirty tree -> fix_sha null + fix_sha_reason passes with a flag;
    // null without a reason is incomplete.
    return incomplete(`rounds_audit entry for round ${currentRound} has null fix_sha without fix_sha_reason`);
  }
  if (!Array.isArray(entry.specialists_run) || entry.specialists_run.length === 0) {
    return incomplete(`rounds_audit entry for round ${currentRound} has empty specialists_run`);
  }
  for (const s of entry.specialists_run) {
    if (!s || typeof s.selection_reason !== "string" || s.selection_reason.trim() === "") {
      return incomplete(`rounds_audit entry for round ${currentRound} has an empty selection_reason`);
    }
  }
  return null;
}

// ── cross-runtime circuit breaker structural warning (US-2, FR-069) ─────
// The breaker's mechanical teeth are limited to this structural warning plus
// the config echo (B-9, accepted residual R-4) — probe-once/discard/no-retry
// compliance is prose-level, enforced by roster-review.
function computeCrossRuntimeWarnings(review) {
  const warnings = [];
  const crossRuntime = review.cross_runtime;
  if (!crossRuntime || typeof crossRuntime !== "object" || Array.isArray(crossRuntime)) return warnings;
  for (const [name, entry] of Object.entries(crossRuntime)) {
    if (!entry || entry.status !== "degraded") continue;
    if (!entry.reason) warnings.push(`cross_runtime.${name}: degraded entry missing reason`);
    if (!entry.config_digest) warnings.push(`cross_runtime.${name}: degraded entry missing config_digest`);
  }
  return warnings;
}

// FR-059/B-5, extended FR-174 (r5-trace-enforcement): precedence
// unencodable-finding > unattested-invocation > novel-finding-streak >
// round-cap; process-incomplete is never a top-level cause.
function selectCause(violations) {
  const causes = new Set(violations.map((v) => v.cause));
  if (causes.has("unencodable-finding")) return "unencodable-finding";
  if (causes.has("unattested-invocation")) return "unattested-invocation";
  if (causes.has("novel-finding-streak")) return "novel-finding-streak";
  if (causes.has("round-cap")) return "round-cap";
  return null;
}

module.exports = {
  HIGH_PLUS,
  isNovelStrikeFinding,
  isReopenedStrikeFinding,
  isValidStreakOverride,
  computeStrikeMap,
  computeStreakViolation,
  computeMissingStrikeWarnings,
  computeMissingAuditViolation,
  computeCrossRuntimeWarnings,
  selectCause,
};
