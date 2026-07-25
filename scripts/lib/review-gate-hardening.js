// scripts/lib/review-gate-hardening.js — CommonJS, repo-local tool (NOT part
// of the upstream review-tool bundle; deliberately absent from
// scripts/review-bundle.manifest.json, and it sits directly in scripts/lib/
// rather than the bundle-owned scripts/lib/review/, exactly like its sibling
// scripts/lib/review-verdict-rules.js).
//
// WHY THIS EXISTS
//
// scripts/check-review-convergence.js is the review pipeline's anti-vacuity
// gate, and it was itself vacuous. A verdict that under-populates the keys the
// gate reads does not fail the gate — it removes the gate's ability to check
// anything, and the resulting run is byte-identical to a genuine pass at every
// point a caller looks (exit 0, `violations: []`, `cause: null`). That is the
// "a gate that cannot fail" defect: a green that means "nothing was checked",
// not "everything checked out".
//
// schema/review-json-schema.md §"Prose/enforcer discrepancies" recorded eleven
// of these (D1..D11) as *known and unresolved*. This module resolves them. Each
// rule below is labelled with its D-number where it has one, plus the three
// undocumented holes found while auditing the same file (G3/G5/G13).
//
// THE TWO SHAPES OF FAILURE, and why the split matters
//
//   fatal      → exit 2, degraded input. Used when the verdict has removed the
//                gate's ability to decide. There is no honest verdict to give,
//                so the gate refuses to give one. `fatal` is a string message;
//                the caller performs the exit.
//   violation  → exit 1 or 3 via the existing precedence. Used when the gate
//                CAN decide and the answer is "no".
//
// The distinction is the whole point: previously every one of these landed in
// `warnings[]` on an exit-0 run, i.e. in the one channel that a passing gate
// lets a caller ignore.
//
// RECORDED SKIPS (the --allow-legacy contract)
//
// Two rules (G3 `round`, G4 `no_go_round`) guard genuinely pre-schema fixtures
// that must keep passing. They are NOT silently tolerated: the caller must pass
// --allow-legacy, and the gate then stamps `legacy_skip_authorized: true` into
// its report. An authorized skip is a *recorded* skip — visible in the
// artifact, and refused downstream by review-verdict-assemble.js --write-strike,
// because a strike computed by a gate that skipped its own checks must never be
// journaled as if it had been computed. An unrecorded skip is indistinguishable
// from a pass; a recorded one is not.
"use strict";

const { validSlug } = require("./xruntime/xruntime-journal");

// Bumped whenever a gate is added or its verdict changes. Echoed into the gate
// report as `hardening.version` so a consumer can detect a check-review-
// convergence.js that predates this module, exactly as it detects a stale
// script via `config.strikes` and `trace`.
const HARDENING_VERSION = "1.0.0";

const HIGH_PLUS = new Set(["CRITICAL", "HIGH"]);
// schema/review-finding.schema.json §properties.status. The gate only ever
// tested `=== "RESOLVED"` / `=== "ACCEPTED"`, so any other string silently
// took the neither-branch (D7).
const FINDING_STATUSES = new Set(["OPEN", "RESOLVED", "ACCEPTED"]);
// schema/review-json-schema.md §`mode`. Verdict mode — NOT the gate report's
// own `mode`, whose enum is static|full (D5).
const VERDICT_MODES = new Set(["express", "fast", "full"]);
const VERDICT_STATUSES = new Set(["GO", "NO-GO"]);
const NO_GO_TYPES = new Set([
  "out-of-scope-change",
  "spec-ac-failure",
  "cross-runtime-finding",
  "design-not-converging",
  "review-integrity-failure",
]);
const NO_GO_CAUSES = new Set(["unencodable-finding", "unattested-invocation", "novel-finding-streak", "round-cap"]);
// schema/review-json-schema.md §`cross_runtime`. Prose lists five statuses and
// the gate validated none of them, so an unrecognised status fell out of the
// corroboration set and was never attested (D9's neighbour).
const CROSS_RUNTIME_STATUSES = new Set(["healthy", "degraded", "skipped-degraded", "skipped-human", "blocked"]);

function has(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function fingerprintOf(finding) {
  if (!finding || typeof finding !== "object") return "?";
  return typeof finding.fingerprint === "string"
    ? finding.fingerprint
    : `${finding.path || "?"}:${finding.line ?? 0}:${finding.category || "?"}`;
}

function isOpen(finding) {
  return finding.status !== "RESOLVED" && finding.status !== "ACCEPTED";
}

// ── G3 (P3 core) + G4 (D10): the two absent-key vacuity paths ────────────
// `round` absent removed strike classification, rounds_audit completeness AND
// every trace check. `no_go_round` absent defaulted to 0, so the round cap
// could never fire. Both used to be warnings on an exit-0 run.
function checkLegacyAuthorization({ legacyRound, legacyNoGoRound, allowLegacy }) {
  if (allowLegacy) return null;
  if (legacyRound) {
    return (
      "review.json has no `round` key — the gate would skip strike classification, the rounds_audit " +
      "completeness check and every trace check, then exit 0 with violations: [] (B-8 vacuity, " +
      "schema/review-json-schema.md §round). Assemble the verdict with scripts/review-verdict-assemble.js, " +
      "or pass --allow-legacy to authorize the skip — it is then RECORDED as legacy_skip_authorized in " +
      "the report and --write-strike refuses that report."
    );
  }
  if (legacyNoGoRound) {
    return (
      "review.json has no `no_go_round` key — it defaults to 0, so the round-cap violation can never " +
      "fire (D10, schema/review-json-schema.md §no_go_round). Assemble the verdict with " +
      "scripts/review-verdict-assemble.js, or pass --allow-legacy to authorize the skip."
    );
  }
  return null;
}

// ── G13: blind coercions ─────────────────────────────────────────────────
// `rounds_audit` and `cross_runtime` were both read as
// `Array.isArray(x) ? x : []` / `typeof x === "object" || return`. A wrong type
// therefore silently emptied the check rather than failing it — the same shape
// as the `findings` hole already closed by FIX-A/CGF-1.
function checkContainerTypes(review) {
  if (has(review, "rounds_audit") && !Array.isArray(review.rounds_audit)) {
    return "review.json field rounds_audit is present but not an array (degraded input) — it was silently read as [], which empties the past-strike map and the loop-back completeness check";
  }
  if (
    has(review, "cross_runtime") &&
    review.cross_runtime !== null &&
    (typeof review.cross_runtime !== "object" || Array.isArray(review.cross_runtime))
  ) {
    return "review.json field cross_runtime is present but not an object (degraded input) — it was silently ignored, which drops every cross-runtime corroboration check";
  }
  if (has(review, "cross_runtime_findings") && !Array.isArray(review.cross_runtime_findings)) {
    return "review.json field cross_runtime_findings is present but not an array (degraded input)";
  }
  return null;
}

// ── G8 (D5) / G7 (D2) / G9 (D4) / G10 (D11) / G11 (D6): envelope keys ────
// Each of these is a key whose malformed value used to read to the gate
// exactly like an omission, and whose omission used to remove an obligation
// rather than fail one.
function checkEnvelopeKeys(review, { legacyRound }) {
  // G8/D5: `mode: "Full"` (or the gate report's own `mode: "static"` copied in
  // by mistake) reads to the gate as "no scope-gate trace line required".
  if (has(review, "mode") && !VERDICT_MODES.has(review.mode)) {
    return (
      `review.json field mode is ${JSON.stringify(review.mode)}, not one of ${[...VERDICT_MODES].join(" | ")} (D5). ` +
      "Any other value silently drops the scope-gate trace obligation. Note the name collision: the gate " +
      "REPORT's `mode` is static|full and is a different field — do not copy one into the other."
    );
  }
  // G7/D2: the routing key the gate never read. A typo routes nowhere.
  if (has(review, "status") && !VERDICT_STATUSES.has(review.status)) {
    return `review.json field status is ${JSON.stringify(review.status)}, not one of ${[...VERDICT_STATUSES].join(" | ")}`;
  }
  const noGoReasonFatal = checkNoGoReason(review);
  if (noGoReasonFatal) return noGoReasonFatal;

  if (legacyRound) return null; // pre-schema fixtures: already authorized above

  // G9/D4: `task` was load-bearing only once the round was trace-obligated;
  // on any other round an absent/invalid slug fell through to the B-8 skip.
  if (typeof review.task !== "string" || !validSlug(review.task)) {
    return `review.json field task is ${JSON.stringify(review.task)}, not a valid slug — it derives the trace and journal sibling paths (D4)`;
  }
  // G10/D11: absent or non-numeric `cycle` silently became null, which filters
  // every numerically-stamped trace line out as prior-cycle.
  if (!has(review, "cycle")) {
    return "review.json has no `cycle` key — every numerically-stamped trace line then classifies as prior-cycle, so a trace-obligated round reports missing-trace even with the lines on disk (D11)";
  }
  if (typeof review.cycle !== "number" || !Number.isFinite(review.cycle) || review.cycle < 1) {
    return `review.json field cycle is ${JSON.stringify(review.cycle)} — must be a finite number >= 1 (D11)`;
  }
  // G11/D6: without normalization, fingerprints are not stable, so "novel
  // finding" is uncomputable and strike classification is meaningless. That
  // used to be a conditional warning.
  if (typeof review.normalized_by !== "string" || review.normalized_by.trim() === "") {
    return (
      "review.json field normalized_by is absent or empty — the findings were never run through " +
      "scripts/review-normalize.js, so fingerprints are not stable and `novel finding` is uncomputable. " +
      "Strike classification on an unnormalized verdict is not a result (D6)."
    );
  }
  return null;
}

function checkNoGoReason(review) {
  const reason = review.no_go_reason;
  if (review.status === "NO-GO" && (reason === null || reason === undefined)) {
    return "review.json has status NO-GO with no no_go_reason — no_go_reason.type is the routing key, and roster-run falls through to `none` without it (D2)";
  }
  if (reason === null || reason === undefined) return null;
  if (typeof reason !== "object" || Array.isArray(reason)) {
    return `review.json field no_go_reason is ${Array.isArray(reason) ? "an array" : typeof reason}, not an object or null (D2)`;
  }
  if (!NO_GO_TYPES.has(reason.type)) {
    return `review.json field no_go_reason.type is ${JSON.stringify(reason.type)}, not one of ${[...NO_GO_TYPES].join(" | ")} — an unrecognised type routes nowhere (D2)`;
  }
  if (has(reason, "cause") && reason.cause !== null && !NO_GO_CAUSES.has(reason.cause)) {
    return `review.json field no_go_reason.cause is ${JSON.stringify(reason.cause)}, not one of ${[...NO_GO_CAUSES].join(" | ")} (D2)`;
  }
  return null;
}

// ── G6 (D7): finding status enum ─────────────────────────────────────────
// The gate tested `=== "RESOLVED"` and `=== "ACCEPTED"` and nothing else, so
// `status: "OPEN-FOR-HUMAN"` (live in this repo) took neither branch. The
// dangerous direction is a near-miss of "ACCEPTED", which reads as un-waived —
// but a near-miss of "RESOLVED" skips the ratchet entirely.
function checkFindingStatuses(findings, label) {
  for (const finding of findings) {
    if (!finding || typeof finding !== "object") continue;
    if (!has(finding, "status")) continue;
    if (!FINDING_STATUSES.has(finding.status)) {
      return (
        `${label} entry ${fingerprintOf(finding)} has status ${JSON.stringify(finding.status)}, not one of ` +
        `${[...FINDING_STATUSES].join(" | ")} (schema/review-finding.schema.json, D7) — the gate only ever ` +
        "compares against RESOLVED and ACCEPTED, so any other value silently escapes both the ratchet and the waiver"
      );
    }
  }
  return null;
}

// ── G1 (D3, named): rounds_audit[].strike must be a boolean ──────────────
// computeStrikeMap() only records `typeof entry.strike === "boolean"`, so a
// null never lands in the map, Map.get() returns undefined, and
// computeStreakViolation()'s `!== true` test resets the streak. A verdict with
// `strike: null` on every round escapes novel-finding escalation completely
// while the report prints current_round_strike: true next to violations: [].
// This was a warning, and only for PAST rounds — a null on the current round
// produced no signal at all.
function checkStrikeBooleans(review, currentRound) {
  if (currentRound === null) return null;
  const roundsAudit = Array.isArray(review.rounds_audit) ? review.rounds_audit : [];
  for (const entry of roundsAudit) {
    if (!entry || typeof entry.round !== "number") continue;
    if (entry.round < 2) continue; // round 1 never strikes
    const isCurrent = entry.round === currentRound;
    if (isCurrent && !has(entry, "strike")) continue; // absent on the draft is correct — the gate writes it
    if (typeof entry.strike === "boolean") continue;
    return (
      `rounds_audit entry for round ${entry.round} has strike ${JSON.stringify(entry.strike)} — it must be a ` +
      "boolean (D3). A non-boolean never enters the strike map, so computeStreakViolation() reads it as " +
      "not-a-strike and the novel-finding streak silently resets. Re-gate that round and journal its strike " +
      "with `review-verdict-assemble.js --write-strike`; on the round being gated now, leave `strike` ABSENT."
    );
  }
  return null;
}

// ── G12 (D9): cross_runtime entry shape ──────────────────────────────────
// The verdict persists `config_digest`; the journal persists the same value
// under `digest`. Writing the journal's key into the verdict yields a silent
// corroboration miss. Nothing validated `status` either, so an unrecognised
// value simply fell out of the corroborated set.
function checkCrossRuntime(review) {
  const crossRuntime = review.cross_runtime;
  if (!crossRuntime || typeof crossRuntime !== "object" || Array.isArray(crossRuntime)) return null;
  for (const [name, entry] of Object.entries(crossRuntime)) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return `cross_runtime.${name} is not an object (degraded input)`;
    }
    if (has(entry, "digest") && !has(entry, "config_digest")) {
      return (
        `cross_runtime.${name} carries \`digest\` but not \`config_digest\` (D9). The key asymmetry is ` +
        "deliberate and load-bearing: the VERDICT writes config_digest, the JOURNAL writes digest, and the " +
        "gate matches cross_runtime[rt].config_digest against journalEntry.digest. As written, corroboration " +
        "silently finds nothing."
      );
    }
    if (has(entry, "status") && !CROSS_RUNTIME_STATUSES.has(entry.status)) {
      return (
        `cross_runtime.${name}.status is ${JSON.stringify(entry.status)}, not one of ` +
        `${[...CROSS_RUNTIME_STATUSES].join(" | ")} — an unrecognised status is never corroborated and never ` +
        "warned about, so a runtime that did not run reads exactly like one that did"
      );
    }
  }
  return null;
}

// ── G5: rounds_audit gaps erase the streak ───────────────────────────────
// Undocumented, found auditing §round: computeStrikeMap() only knows rounds
// that HAVE an entry and computeStreakViolation() stops at the first round that
// is not `true`, so jumping round 3 → round 7 erases the streak with no
// warning at all (computeMissingStrikeWarnings only inspects entries that
// exist). Repairable by journaling the missing round, hence process-incomplete.
function computeAuditGapViolations(review, currentRound) {
  if (currentRound === null || currentRound < 3) return [];
  const roundsAudit = Array.isArray(review.rounds_audit) ? review.rounds_audit : [];
  const seen = new Set(roundsAudit.filter((e) => e && typeof e.round === "number").map((e) => e.round));
  const violations = [];
  for (let r = 2; r < currentRound; r++) {
    if (seen.has(r)) continue;
    violations.push({
      type: "missing-past-audit",
      cause: "process-incomplete",
      detail:
        `no rounds_audit entry for round ${r} (rounds 2..${currentRound - 1} must all be journaled). ` +
        "A gap is invisible to computeStrikeMap() and resets the novel-finding streak silently.",
    });
  }
  return violations;
}

// ── G2 (D1, named): cross_runtime_findings has GO authority and was unread ─
// roster-review: "any cross_runtime_findings entry that is CRITICAL or HIGH
// (OPEN) sets status: NO-GO". The gate never opened the array. The prose escape
// hatch is to mirror the entry into `findings` so it enters the ratchet — and
// forgetting the mirror was unenforced, on the array that carries the findings
// the primary reviewer missed.
function computeCrossRuntimeFindingViolations(review, findings) {
  const crossRuntimeFindings = Array.isArray(review.cross_runtime_findings) ? review.cross_runtime_findings : [];
  if (crossRuntimeFindings.length === 0) return [];

  const mirrored = new Set();
  for (const finding of findings) {
    if (!finding || typeof finding !== "object") continue;
    mirrored.add(fingerprintOf(finding));
    if (typeof finding.fingerprint_v2 === "string") mirrored.add(finding.fingerprint_v2);
  }

  const violations = [];
  for (const finding of crossRuntimeFindings) {
    if (!finding || typeof finding !== "object") continue;
    if (!HIGH_PLUS.has(finding.severity)) continue;
    if (!isOpen(finding)) continue;
    const fingerprint = fingerprintOf(finding);
    if (mirrored.has(fingerprint)) continue;
    violations.push({
      type: "unmirrored-cross-runtime-finding",
      cause: "unencodable-finding",
      fingerprint,
      detail:
        `${finding.severity} OPEN cross_runtime_findings entry is not mirrored into findings[] (D1). ` +
        "roster-review gives it GO authority, but only the findings[] mirror puts it under the ratchet, " +
        "the red/green obligation and strike classification.",
    });
  }
  return violations;
}

// ── G7 (D2, second half): a GO verdict that the gate is failing ──────────
// The gate never read `status`, so a verdict could assert GO while the gate
// simultaneously reported a design violation, and the two artifacts disagreed
// with nothing to reconcile them. Evaluated last, over the fully accumulated
// violation list.
function computeGoConsistencyViolation(review, violations) {
  if (review.status !== "GO") return null;
  const design = violations.filter((v) => v.cause && v.cause !== "process-incomplete");
  if (design.length === 0) return null;
  return {
    type: "go-with-design-violation",
    cause: design[0].cause,
    detail:
      `verdict claims status GO while the gate reports ${design.length} design violation(s) ` +
      `(${[...new Set(design.map((v) => v.type))].join(", ")}) — the gate's answer is authoritative (D2)`,
  };
}

// ── entry point ──────────────────────────────────────────────────────────
// Returns { fatal, violations } — `fatal` is a message the caller must turn
// into exit 2, `violations` join the existing precedence machinery. Ordered
// most-fundamental first so the message a caller sees names the root cause and
// not a downstream symptom of it.
function evaluateHardening({ review, round, legacyRound, legacyNoGoRound, allowLegacy, findings }) {
  const fatal =
    checkContainerTypes(review) ||
    checkLegacyAuthorization({ legacyRound, legacyNoGoRound, allowLegacy }) ||
    checkEnvelopeKeys(review, { legacyRound }) ||
    checkFindingStatuses(findings, "findings") ||
    checkFindingStatuses(Array.isArray(review.cross_runtime_findings) ? review.cross_runtime_findings : [], "cross_runtime_findings") ||
    checkStrikeBooleans(review, round) ||
    checkCrossRuntime(review);
  if (fatal) return { fatal, violations: [] };

  return {
    fatal: null,
    violations: computeAuditGapViolations(review, round).concat(computeCrossRuntimeFindingViolations(review, findings)),
  };
}

module.exports = {
  HARDENING_VERSION,
  FINDING_STATUSES,
  VERDICT_MODES,
  VERDICT_STATUSES,
  NO_GO_TYPES,
  NO_GO_CAUSES,
  CROSS_RUNTIME_STATUSES,
  checkLegacyAuthorization,
  checkContainerTypes,
  checkEnvelopeKeys,
  checkNoGoReason,
  checkFindingStatuses,
  checkStrikeBooleans,
  checkCrossRuntime,
  computeAuditGapViolations,
  computeCrossRuntimeFindingViolations,
  computeGoConsistencyViolation,
  evaluateHardening,
};
