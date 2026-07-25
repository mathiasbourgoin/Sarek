// scripts/lib/review/review-trace-rules.js — CommonJS, pure (no filesystem I/O).
//
// Trace obligation/coverage/correspondence/malformed-classification rule
// functions (spec: specs/r5-trace-enforcement.md FR-169..174/176/180,
// US-3/US-4). Split out so scripts/check-review-convergence.js gains only
// thin dispatch (FR-178) — I/O (reading the trace file/journal) stays in the
// gate; every classification decision lives here, unit-tested directly.
//
// Responsibility boundary: this module owns (1) whether the current round is
// trace-obligated at all (deriveObligation, FR-169), (2) classifying raw
// trace-file lines into current-cycle-valid / degraded / warned
// (classifyLines, FR-173/C-4), (3) required-event coverage and
// claimed-invocation correspondence (computeRequiredCoverage, FR-170/171),
// (4) cross-runtime journal corroboration (computeCrossRuntimeCorroboration,
// FR-172/D2 — journal field `digest` vs review.json field `config_digest`,
// same value, different key), (5) the omit-everything warning (FR-180), and
// (6) the report's anti-stale-script `trace` block (buildTraceBlock, FR-176).
"use strict";

const path = require("path");
const { compile } = require("./finding-schema");

const TRACE_SCHEMA_PATH = path.resolve(__dirname, "..", "..", "..", "schema", "review-trace.schema.json");

let traceValidator = null;
function getTraceValidator() {
  if (!traceValidator) traceValidator = compile(require(TRACE_SCHEMA_PATH));
  return traceValidator;
}

// ── obligation (FR-169, US-4) ────────────────────────────────────────────
// obligated iff the trace file exists OR the current round's rounds_audit
// entry carries trace_schema_version. Neither holds (or `round` key absent
// entirely, B-8 legacy) -> skip all trace checks with a warning.
function deriveObligation({ traceFileExists, roundsAuditEntry, hasRoundKey }) {
  if (!hasRoundKey) {
    return { obligated: false, skip: true, skipReason: "legacy review.json: round key absent — trace checks skipped (B-8)" };
  }
  const hasVersion = !!roundsAuditEntry && typeof roundsAuditEntry.trace_schema_version === "string";
  if (traceFileExists || hasVersion) {
    return { obligated: true, skip: false, skipReason: null };
  }
  return {
    obligated: false,
    skip: true,
    skipReason: "round predates the trace mechanism — trace checks skipped (B-8)",
  };
}

// ── line classification (FR-173, C-4) ────────────────────────────────────
// Splits raw trace-file lines (strings) into: `current` (schema-valid lines
// matching (task, cycle==currentCycle)), `degraded` (true iff ANY current-
// cycle line is unparseable/schema-invalid — fail-closed, exit 2), and
// `warnings` (prior-cycle corruption, task mismatch — never block).
function classifyLines(rawLines, task, currentCycle) {
  const validator = getTraceValidator();
  const current = [];
  const warnings = [];
  let degraded = false;

  for (const raw of rawLines) {
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      // Unparseable at all -> current-cycle-or-unknown -> fail closed (FR-173:
      // "or which is unparseable at all").
      degraded = true;
      continue;
    }

    const { valid, errors } = validator.validate(parsed);
    if (!valid) {
      if (typeof parsed.cycle === "number" && parsed.cycle !== currentCycle) {
        warnings.push(`prior-cycle malformed trace line (cycle ${parsed.cycle}): ${errors.join("; ")}`);
      } else {
        degraded = true;
      }
      continue;
    }

    if (parsed.cycle !== currentCycle) continue; // prior-cycle valid line — not degraded, not counted
    if (parsed.task !== task) {
      warnings.push(`trace line task "${parsed.task}" does not match task slug "${task}" — not counted (EC-5)`);
      continue;
    }
    current.push(parsed);
  }

  return { current, degraded, warnings };
}

// ── required-event coverage + correspondence (FR-170/171) ───────────────
// Zero current-cycle-and-round lines on an obligated round -> a single
// missing-trace violation (process-incomplete, exit 3). >=1 line but a
// claimed invocation lacks its corresponding line -> one unattested-
// invocation violation per claim (exit 1).
function computeRequiredCoverage({ currentLines, currentRound, mode, specialistsRun, normalizedBy }) {
  const roundLines = currentLines.filter((l) => l.round === currentRound);

  if (roundLines.length === 0) {
    return [{ type: "missing-trace", cause: "process-incomplete", detail: `no trace lines for round ${currentRound}` }];
  }

  const violations = [];
  const hasEvent = (event, actor) =>
    roundLines.some((l) => l.event === event && (actor === undefined || l.actor === actor));

  if (normalizedBy && !hasEvent("normalizer")) {
    violations.push(unattested(`normalizer claimed via normalized_by ("${normalizedBy}") with no matching normalizer trace line`));
  }

  if (mode === "full" && !hasEvent("scope-gate")) {
    violations.push(unattested("Full-mode round with no scope-gate trace line"));
  }

  for (const s of specialistsRun || []) {
    const name = s && s.name;
    if (!name) continue;
    if (!hasEvent("specialist", name)) {
      violations.push(unattested(`specialists_run claims "${name}" with no matching specialist trace line`));
    }
  }

  return violations;
}

function unattested(detail) {
  return { type: "unattested-invocation", cause: "unattested-invocation", detail };
}

// ── cross-runtime corroboration (FR-172, D2) ─────────────────────────────
// D2 (binding): journal writes `digest`; review.json's cross_runtime entries
// write `config_digest` — same VALUE, different key. Match
// cross_runtime[rt].config_digest against journalEntry.digest — never
// config_digest against config_digest.
function computeCrossRuntimeCorroboration({ crossRuntime, journalEntries, currentRound, currentCycle }) {
  const violations = [];
  if (!crossRuntime || typeof crossRuntime !== "object") return violations;

  const CHECKED_STATUSES = new Set(["healthy", "degraded", "skipped-human"]);
  for (const [runtime, entry] of Object.entries(crossRuntime)) {
    if (!entry || entry.round !== currentRound || !CHECKED_STATUSES.has(entry.status)) continue;

    const match = (journalEntries || []).some(
      (j) => j && j.runtime === runtime && j.digest === entry.config_digest && j.cycle === currentCycle
    );
    if (!match) {
      violations.push(unattested(`cross_runtime.${runtime} claims round ${currentRound} status "${entry.status}" with no matching journal entry (runtime, digest, cycle)`));
    }
  }
  return violations;
}

// ── omit-everything warning (FR-180, C-2) ────────────────────────────────
// Non-legacy round (round present) with NEITHER trace obligation NOR a
// normalized_by stamp -> loud warning (never a violation in v1).
function computeOmitEverythingWarning({ hasRoundKey, obligated, normalizedBy }) {
  if (!hasRoundKey) return null;
  if (obligated || normalizedBy) return null;
  return "round carries no trace obligation and no normalized_by stamp — the omit-everything posture (C-2); confirm the reviewer and normalizer actually ran (FR-180)";
}

// ── report anti-stale-script signal (FR-176) ─────────────────────────────
function buildTraceBlock({ obligated, currentLines, skip }) {
  return {
    obligated,
    lines_seen: currentLines.length,
    schema_version: currentLines.length > 0 ? currentLines[0].schema_version : null,
    skipped: !!skip,
  };
}

module.exports = {
  deriveObligation,
  classifyLines,
  computeRequiredCoverage,
  computeCrossRuntimeCorroboration,
  computeOmitEverythingWarning,
  buildTraceBlock,
};
