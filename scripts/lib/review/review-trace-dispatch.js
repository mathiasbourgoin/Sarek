// scripts/lib/review/review-trace-dispatch.js — CommonJS.
//
// I/O-facing orchestration for the trace verification dispatch (spec:
// specs/r5-trace-enforcement.md FR-169..176/180, US-3, C-5/FR-178 line-
// budget contingency). Split out of scripts/check-review-convergence.js
// (same "split by responsibility" principle as redgreen-scratch.js, FIX-1
// follow-on) so the main gate file stays under the repo's 500-line limit as
// the trace dispatch is added.
//
// Responsibility boundary: this module owns validating review.task (D4),
// deriving the sibling trace/journal paths from path.dirname(reviewAbsPath)
// (D4 — never suffix-stripping), reading both files READ-ONLY
// (FR-163/FR-164 — this module MUST NOT write to either), and orchestrating
// the pure classification/coverage/corroboration calls in
// scripts/lib/review/review-trace-rules.js. It never calls process.exit
// itself — an invalid task slug or an I/O read failure is reported back to
// the caller (scripts/check-review-convergence.js) as a `fail` descriptor,
// which decides how to exit; this keeps the module testable without mocking
// process.exit and keeps "thin dispatch" (FR-178) literal in the gate file.
"use strict";

const fs = require("fs");
const path = require("path");
const { validSlug } = require("../xruntime/xruntime-journal");
const {
  deriveObligation,
  classifyLines,
  computeRequiredCoverage,
  computeCrossRuntimeCorroboration,
  computeOmitEverythingWarning,
  buildTraceBlock,
} = require("./review-trace-rules");

// Reads a JSONL file's non-empty lines. Absent file -> empty (not an error);
// an existing-but-unreadable file -> readError (caller fails closed, D4).
function readJsonlLines(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return { lines: [], readError: false };
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    return { lines: raw.split("\n").filter((l) => l.trim() !== ""), readError: false };
  } catch (e) {
    return { lines: [], readError: true };
  }
}

function safeParseJson(raw) {
  try {
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

// D4/A-2: does the review directory already hold ANY *-review-trace.jsonl?
// Used to fail closed when the round is trace-obligated by a file that exists
// on disk but `task` is blanked/invalid — otherwise a lying reviewer could
// erase `task`, make the gate unable to locate the file, and disguise an
// obligated round as a benign legacy skip (the D4 evasion the review caught).
function hasOrphanTraceFile(reviewAbsPath) {
  try {
    return fs
      .readdirSync(path.dirname(reviewAbsPath))
      .some((name) => /-review-trace\.jsonl$/.test(name));
  } catch (e) {
    return false;
  }
}

// D4: derives the two sibling paths from path.dirname(reviewAbsPath) — NEVER
// suffix-stripping (sidesteps a `.json.draft` vs `-review.json` mismatch).
function deriveTracePaths(reviewAbsPath, task) {
  const baseDir = path.dirname(reviewAbsPath);
  return {
    tracePath: path.join(baseDir, `${task}-review-trace.jsonl`),
    journalPath: path.join(baseDir, `${task}-xruntime.jsonl`),
  };
}

// Classification + coverage + cross-runtime corroboration once obligation +
// I/O are settled (function-length discipline split from evaluateTrace).
function evaluateObligatedTrace({ review, round, cycle, rawTraceLines, rawJournalLines, roundsAuditEntry, task }) {
  const { current, degraded, warnings } = classifyLines(rawTraceLines, task, cycle);
  const traceBlock = buildTraceBlock({ obligated: true, currentLines: current, skip: false });
  if (degraded) return { violations: [], warnings, traceBlock, degraded: true };

  const journalEntries = rawJournalLines.map(safeParseJson).filter(Boolean);
  const violations = computeRequiredCoverage({
    currentLines: current,
    currentRound: round,
    mode: review.mode,
    specialistsRun: (roundsAuditEntry && roundsAuditEntry.specialists_run) || [],
    normalizedBy: review.normalized_by || null,
  }).concat(
    computeCrossRuntimeCorroboration({
      crossRuntime: review.cross_runtime,
      journalEntries,
      currentRound: round,
      currentCycle: cycle,
    })
  );

  return { violations, warnings, traceBlock, degraded: false };
}

// Orchestrates trace obligation + coverage + cross-runtime corroboration for
// the current round. Returns either `{ fail: { code, message } }` (caller
// must exit) or `{ violations, warnings, traceBlock, degraded }`.
//
// D4/FR-179 ordering note: task-slug validation is REQUIRED (fail-closed)
// only once the round is ALREADY obligated via the trace_schema_version
// prong (a round that positively claims trace-awareness) — that is the
// actual evasion vector D4 closes: a reviewer stamping trace_schema_version
// while corrupting/omitting task to dodge file inspection. When obligation
// would otherwise rest SOLELY on the file-existence prong, an invalid/absent
// task means no legitimately-produced trace file could be named under it in
// the first place (the producer CLI requires --task to write anything), so
// treating that prong as unresolvable-false and falling through to the B-8
// skip is not a new evasion — it is what lets the ~40 pre-existing
// round-based fixtures in check-review-convergence-rules.test.js (predating
// this feature, none of which set task or trace_schema_version) keep passing
// unmodified (FR-179), while a genuinely trace-obligated round still fails
// closed on a corrupted task.
function evaluateTrace({ review, round, cycle, legacyRound, reviewAbsPath }) {
  const hasRoundKey = !legacyRound;
  const roundsAudit = Array.isArray(review.rounds_audit) ? review.rounds_audit : [];
  const roundsAuditEntry = roundsAudit.find((e) => e && e.round === round) || null;
  const hasVersion = !!roundsAuditEntry && typeof roundsAuditEntry.trace_schema_version === "string";
  const taskValid = typeof review.task === "string" && validSlug(review.task);

  // Fail closed on an obligated round with a blank/invalid task. Obligation
  // here comes from EITHER the version prong (round positively claims trace
  // awareness) OR an existing *-review-trace.jsonl in the review dir (A-2: a
  // blanked task must not disguise a round whose evidence file already exists
  // as a legacy skip — closes the D4 evasion). A round with neither prong and
  // no task stays a legacy B-8 skip, keeping the pre-existing task-less
  // round-based fixtures green (FR-179).
  if (hasRoundKey && !taskValid && (hasVersion || hasOrphanTraceFile(reviewAbsPath))) {
    return {
      fail: {
        code: 2,
        message:
          "review.json field task must be a non-empty slug — required to derive the trace/journal paths for a trace-obligated round (D4/A-2)",
      },
    };
  }

  const task = review.task;
  const { tracePath, journalPath } =
    hasRoundKey && taskValid ? deriveTracePaths(reviewAbsPath, task) : { tracePath: null, journalPath: null };

  const traceFileExists = !!tracePath && fs.existsSync(tracePath);
  const obligation = deriveObligation({ traceFileExists, roundsAuditEntry, hasRoundKey });
  if (obligation.skip) {
    const warnings = [obligation.skipReason];
    const omitWarning = computeOmitEverythingWarning({ hasRoundKey, obligated: false, normalizedBy: review.normalized_by || null });
    if (omitWarning) warnings.push(omitWarning);
    return { violations: [], warnings, traceBlock: buildTraceBlock({ obligated: false, currentLines: [], skip: true }), degraded: false };
  }

  const { lines: rawTraceLines, readError: traceReadError } = readJsonlLines(tracePath);
  const { lines: rawJournalLines, readError: journalReadError } = readJsonlLines(journalPath);
  if (traceReadError || journalReadError) {
    return {
      violations: [],
      warnings: [],
      traceBlock: buildTraceBlock({ obligated: true, currentLines: [], skip: false }),
      degraded: true,
    };
  }

  return evaluateObligatedTrace({ review, round, cycle, rawTraceLines, rawJournalLines, roundsAuditEntry, task });
}

module.exports = { evaluateTrace, readJsonlLines, deriveTracePaths };
