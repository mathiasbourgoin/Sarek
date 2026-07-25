#!/usr/bin/env node
// scripts/review-normalize.js — CommonJS, read-only w.r.t. the repository.
//
// H-05 review-result normalizer (spec: specs/review-skill-slimming.md US-2,
// FR-099..108, Amendment D-1 reobservations[]; specs/review-v2-corrections.md
// INV-1/2/5, Amendments E-2/E-3/E-4/E-7). Validates new-round specialist
// findings against schema/review-finding.schema.json, carries prior-round
// ledger entries forward byte-identical (FR-101), mechanically merges exact
// duplicates by SEMANTIC identity (FR-104/INV-1 — a v1 fingerprint collision
// alone never merges away a distinct finding) and surfaces probable
// duplicates for owner adjudication (FR-105), canonicalizes AND deduplicates
// cross-runtime findings within their own augment-only array (INV-5/E-7,
// never merged into primary), classifies ledger re-reports into
// reobserved/reopen/pending-check dispositions (INV-2/E-2/E-4 — the
// normalizer proposes, roster-review acts), and reports schema-invalid input
// in `rejected[]` (FR-100) — never silently dropped.
//
// Usage:
//   node scripts/review-normalize.js [<finding-file.json> ...] [--ledger <path>]
//     [--round <n>] [--cycle <n>] [--task <slug>] [--gate-report <path>] [--prior <path>]
//
// Each positional file is a JSON array of specialist finding objects; with no
// positional files, findings are read from stdin (a JSON array; empty stdin
// is treated as an empty array, EC-8). `--ledger <path>` is the prior
// cumulative ledger (a JSON array of previously-persisted findings); absent
// means no ledger (first round). `--round <n>` is passed by roster-review to
// stamp reobservations — omitted means `round: null`. `--gate-report <path>`
// is the PRIOR round's persisted briefs/<task>-gate-report.json (E-2) — its
// absence fails closed (a RESOLVED, check-linked ledger entry re-reporting
// without a gate report to consult classifies "reopen", never "reobserved").
// `--prior <path>` is the prior FULL briefs/<task>-review.json ENVELOPE
// (distinct from `--ledger`, which is only its `findings` array) — when both
// `--round` and `--prior` are given, the caller's `--round` is cross-checked
// against `scripts/lib/review/review-lifecycle.js`'s `deriveRoundState(prior)` and a
// mismatch is reported in `warnings[]` (never a hard failure — advisory only,
// review finding FIX-1). `--task <slug>` and `--cycle <n>` (spec:
// specs/r5-trace-enforcement.md FR-166, US-2) are used ONLY to self-append
// this invocation's own `normalizer` trace line to
// briefs/<task>-review-trace.jsonl BEFORE stdout is emitted — never consumed
// by normalize()'s own (pure) logic; omitting either means no self-append
// (unchanged from before this feature). An append failure surfaces as a
// warning in `warnings[]`, never a changed exit code or normalization result.
//
// Output: single JSON object on stdout —
//   { findings, cross_runtime_findings, probable_duplicates, rejected,
//     reobservations, dispositions: { reopened, pending_check }, warnings,
//     stats, normalizer_version }
// Exit: 0 on success (including empty input); 2 on usage/degraded input.
"use strict";

const fs = require("fs");
const path = require("path");
const { loadFindingSchema } = require("./lib/review/finding-schema");
const {
  canonicalFingerprint,
  hasV2Fields,
  computeFingerprintV2,
  computeFid,
  isCrossRuntime,
  mergeExactDuplicates,
  computeProbableDuplicates,
  computeLedgerProbableDuplicates,
  splitReobservations,
} = require("./lib/review/normalize-rules");
const { deriveRoundState } = require("./lib/review/review-lifecycle");
const { appendTraceLine } = require("./lib/review/review-trace");

const NORMALIZER_VERSION = "2.0.0";
const TRACE_SCHEMA_VERSION = "1.0";

function fail(code, message) {
  process.stderr.write(`review-normalize: ${message}\n`);
  process.exit(code);
}

function parseArgs(argv) {
  const out = { files: [], ledgerPath: null, round: null, gateReportPath: null, priorPath: null, task: null, cycle: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--ledger") {
      out.ledgerPath = argv[++i];
      if (out.ledgerPath === undefined) fail(2, "--ledger requires a path argument");
    } else if (a === "--round") {
      const raw = argv[++i];
      const n = parseInt(raw, 10);
      if (Number.isNaN(n)) fail(2, "--round requires an integer argument");
      out.round = n;
    } else if (a === "--cycle") {
      // r5-trace-enforcement (FR-166): only used to stamp this invocation's
      // self-appended trace line — never consumed by normalize()'s own logic.
      const raw = argv[++i];
      const n = parseInt(raw, 10);
      if (Number.isNaN(n)) fail(2, "--cycle requires an integer argument");
      out.cycle = n;
    } else if (a === "--task") {
      out.task = argv[++i];
      if (out.task === undefined) fail(2, "--task requires a value");
    } else if (a === "--gate-report") {
      out.gateReportPath = argv[++i];
      if (out.gateReportPath === undefined) fail(2, "--gate-report requires a path argument");
    } else if (a === "--prior") {
      out.priorPath = argv[++i];
      if (out.priorPath === undefined) fail(2, "--prior requires a path argument");
    } else if (a.startsWith("--")) {
      fail(2, `unknown flag: ${a}`);
    } else {
      out.files.push(a);
    }
  }
  return out;
}

function readJsonArrayFile(filePath, label) {
  const resolved = path.resolve(process.cwd(), filePath);
  if (!fs.existsSync(resolved)) fail(2, `${label} not found: ${filePath}`);
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(resolved, "utf8"));
  } catch (e) {
    fail(2, `${label} is not valid JSON: ${e.message}`);
    return;
  }
  if (!Array.isArray(parsed)) fail(2, `${label} must be a JSON array: ${filePath}`);
  return parsed;
}

function readJsonObjectFile(filePath, label) {
  const resolved = path.resolve(process.cwd(), filePath);
  if (!fs.existsSync(resolved)) return null; // absent gate report -> fail-closed handling in classifyDisposition
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(resolved, "utf8"));
  } catch (e) {
    fail(2, `${label} is not valid JSON: ${e.message}`);
    return;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) fail(2, `${label} must be a JSON object: ${filePath}`);
  return parsed;
}

function readStdinArray() {
  let raw;
  try {
    raw = fs.readFileSync(0, "utf8");
  } catch (e) {
    return []; // no stdin attached (e.g. interactive TTY with nothing piped)
  }
  const trimmed = raw.trim();
  if (trimmed === "") return []; // EC-8: empty input -> empty array
  let parsed;
  try {
    parsed = JSON.parse(trimmed);
  } catch (e) {
    fail(2, `stdin is not valid JSON: ${e.message}`);
    return;
  }
  if (!Array.isArray(parsed)) fail(2, "stdin must be a JSON array");
  return parsed;
}

function readNewFindings(files) {
  if (files.length === 0) return readStdinArray();
  let all = [];
  for (const f of files) all = all.concat(readJsonArrayFile(f, "finding file"));
  return all;
}

// Validates each candidate against the canonical schema, splitting into
// `valid` (kept) and `rejected` (schema-invalid, with a reason — FR-100).
function validateFindings(candidates) {
  const validator = loadFindingSchema();
  const valid = [];
  const rejected = [];
  for (const f of candidates) {
    const { valid: ok, errors } = validator.validate(f);
    if (ok) valid.push(f);
    else rejected.push({ finding: f, reason: errors.join("; ") });
  }
  return { valid, rejected };
}

// Canonicalizes fingerprint (FR-102, always recomputed — never trusted from
// input), computes fingerprint_v2 where applicable (FR-103), and computes
// `fid` (E-3, always — the fid namespace does not depend on v2 fields being
// present). Returns NEW objects — inputs are never mutated. Used for BOTH the
// primary pipeline and the cross-runtime augment-only array (INV-5/E-7): the
// same canonical-identity rules apply to both.
function canonicalizeFindings(findings) {
  return findings.map((f) => {
    const canon = Object.assign({}, f, { fingerprint: canonicalFingerprint(f) });
    if (hasV2Fields(canon)) canon.fingerprint_v2 = computeFingerprintV2(canon);
    canon.fid = computeFid(canon);
    if (canon.status === undefined) canon.status = "OPEN"; // FIX-C (CGF-6, INV-C)
    return canon;
  });
}

// FIX-1 (review): cross-checks the caller-supplied `--round` against
// scripts/lib/review/review-lifecycle.js's own derivation from the prior envelope —
// the normalizer never recomputes the lifecycle rule itself, it only asks
// the witness and reports a drift. `priorReview` absent -> no check (nothing
// to cross-check against, not an error).
function checkRoundConsistency(round, priorReview) {
  if (round === null || round === undefined || !priorReview) return null;
  const derived = deriveRoundState(priorReview);
  if (derived.round === null || derived.round === round) return null;
  return (
    `round consistency: caller passed --round ${round} but ` +
    `scripts/lib/review/review-lifecycle.js derived ${derived.round} from the prior verdict — ` +
    "the two disagree; roster-review's own round derivation may be stale (FIX-1)"
  );
}

function buildStats({ input, rejected, crossRuntime, reobservations, reopened, pendingCheck, merged, ledger, probableDuplicates }) {
  return {
    input,
    rejected,
    cross_runtime: crossRuntime,
    reobservations,
    reopened,
    pending_check: pendingCheck,
    merged,
    carried_forward: ledger,
    probable_duplicates: probableDuplicates,
  };
}

// Splits validated findings into primary vs. cross-runtime, canonicalizing
// both (fingerprint/fingerprint_v2/fid). INV-5/E-7: cross-runtime findings
// are ALSO deduplicated within their own augment-only array here, at intake
// — never merged into primary `findings`; "never rewritten" (roster-review's
// augment-only contract) applies to what happens after this point.
function partitionAndCanonicalize(valid) {
  const crossRuntimeFindings = [];
  const primaryCandidates = [];
  for (const f of valid) (isCrossRuntime(f) ? crossRuntimeFindings : primaryCandidates).push(f);

  const primary = canonicalizeFindings(primaryCandidates);
  const dedupedCrossRuntime = mergeExactDuplicates(canonicalizeFindings(crossRuntimeFindings));
  return { primary, dedupedCrossRuntime };
}

// Pure orchestration (no I/O) — exported so tests can exercise the full
// pipeline without going through argv/stdin/exit.
function normalize({ newFindings, ledger, round, gateReport, priorReview }) {
  const ledgerArr = Array.isArray(ledger) ? ledger : [];
  const warnings = [];
  const roundWarning = checkRoundConsistency(round === undefined ? null : round, priorReview || null);
  if (roundWarning) warnings.push(roundWarning);

  const { valid, rejected } = validateFindings(newFindings);
  const { primary, dedupedCrossRuntime } = partitionAndCanonicalize(valid);

  // INV-2/E-4: a re-report matching a ledger entry is disposed by
  // splitReobservations() into reobserved/reopen/pending-check — a RESOLVED
  // entry is suppressed to metadata ONLY when its linked check is confirmed
  // green on the current tree via the supplied gate report; a resolved entry
  // with no check, or no gate report to consult, is a regression (reopen),
  // never silently reduced to noise.
  const { reobservations, reopened, pendingCheck, genuinelyNew } = splitReobservations(primary, ledgerArr, round, gateReport);

  const settled = mergeExactDuplicates(genuinelyNew);
  const probableDuplicates = computeProbableDuplicates(settled).concat(
    computeLedgerProbableDuplicates(ledgerArr, settled)
  );
  const findings = ledgerArr.concat(settled);

  return {
    findings,
    cross_runtime_findings: dedupedCrossRuntime,
    probable_duplicates: probableDuplicates,
    rejected,
    reobservations,
    dispositions: { reopened, pending_check: pendingCheck },
    warnings,
    stats: buildStats({
      input: newFindings.length,
      rejected: rejected.length,
      crossRuntime: dedupedCrossRuntime.length,
      reobservations: reobservations.length,
      reopened: reopened.length,
      pendingCheck: pendingCheck.length,
      merged: settled.length,
      ledger: ledgerArr.length,
      probableDuplicates: probableDuplicates.length,
    }),
    normalizer_version: NORMALIZER_VERSION,
  };
}

// FR-166 (US-2): self-appends this invocation's own `normalizer` trace line
// BEFORE stdout is emitted (append-before-report) — reusing (not editing)
// scripts/lib/xruntime/xruntime-journal.js's mechanics via the sibling
// scripts/lib/review/review-trace.js module (never the journal file itself,
// INV-4). Only appends when --task, --round, AND --cycle are ALL given — a
// line stamped with a guessed/sentinel cycle would silently mismatch the
// gate's (cycle, round) scoping (scripts/lib/review/review-trace-rules.js's
// classifyLines) and defeat the append's purpose; an older invocation
// predating these flags simply appends nothing, unchanged from before this
// feature. An append failure is surfaced as a warning in the result — never
// a silent success (FR-096 pattern) — and NEVER changes the exit code or
// normalization output (C-9).
function selfAppendTrace(args, result) {
  if (args.task === null || args.round === null || args.cycle === null) return;
  const entry = {
    schema_version: TRACE_SCHEMA_VERSION,
    ts: new Date().toISOString(),
    task: args.task,
    round: args.round,
    cycle: args.cycle,
    event: "normalizer",
    actor: "review-normalize.js",
    outcome: "ran",
  };
  const appended = appendTraceLine(process.cwd(), args.task, entry);
  if (!appended.ok) {
    result.warnings.push(`review-trace append failed (normalizer self-append, FR-166): ${appended.error}`);
  }
}

function main(argv) {
  const args = parseArgs(argv);
  const newFindings = readNewFindings(args.files);
  const ledger = args.ledgerPath ? readJsonArrayFile(args.ledgerPath, "--ledger file") : [];
  const gateReport = args.gateReportPath ? readJsonObjectFile(args.gateReportPath, "--gate-report file") : null;
  const priorReview = args.priorPath ? readJsonObjectFile(args.priorPath, "--prior file") : null;
  const result = normalize({ newFindings, ledger, round: args.round, gateReport, priorReview });
  selfAppendTrace(args, result);
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
  process.exit(0);
}

module.exports = { normalize, parseArgs, NORMALIZER_VERSION, selfAppendTrace };

if (require.main === module) {
  main(process.argv.slice(2));
}
