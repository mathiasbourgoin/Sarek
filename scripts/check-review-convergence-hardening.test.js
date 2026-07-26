#!/usr/bin/env node
// scripts/check-review-convergence-hardening.test.js — zero-dependency,
// node-runnable:
//
//   node scripts/check-review-convergence-hardening.test.js
//
// Every case here constructs an input that SHOULD be rejected and proves the
// REAL scripts/check-review-convergence.js rejects it, with the right exit code
// and a message that names the hole. A gate nobody has watched fail is exactly
// the bug this suite exists to prevent, so assertions are on the refusal, never
// merely on "not exit 0".
//
// The control case (BASE) is load-bearing in the other direction: it proves the
// fixture passes cleanly, so each rejection below is attributable to the single
// field that case mutates and not to the fixture being broken all along.
//
// Coverage map — schema/review-json-schema.md §"Prose/enforcer discrepancies":
//
//   G1  D3   rounds_audit[].strike must be a boolean          (named in the task)
//   G2  D1   cross_runtime_findings is read at last           (named in the task)
//   G3  —    absent `round` is refused, not silently skipped
//   G4  D10  absent `no_go_round` is refused
//   G5  —    a rounds_audit gap erases the streak
//   G6  D7   finding `status` enum
//   G7  D2   `status` / `no_go_reason` routing keys
//   G8  D5   verdict `mode` enum (and the gate-report `mode` collision)
//   G9  D4   `task` is load-bearing on every non-legacy round
//   G10 D11  `cycle` presence and type
//   G11 D6   `normalized_by` — an unnormalized verdict has no stable fingerprints
//   G12 D9   cross_runtime config_digest/digest asymmetry + status enum
//   G13 —    blind container coercions (rounds_audit / cross_runtime)
//
//   D8 is deliberately absent: it records the enforcer being STRICTER than the
//   prose (missing round provenance is already a violation), which is not a
//   hole. It is resolved by documentation, not by a gate.
//
// Repo-local test. It requires the upstream review-tool bundle to be installed
// (scripts/check-review-convergence.js and scripts/lib/review/*); if it is not,
// this file exits non-zero rather than reporting a skip as a pass.
"use strict";

const assert = require("assert");
const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const GATE = path.join(REPO, "scripts", "check-review-convergence.js");

if (!fs.existsSync(GATE)) {
  process.stderr.write(
    `check-review-convergence-hardening.test.js: ${path.relative(REPO, GATE)} is not installed.\n` +
      "This is a FAILURE, not a skip: the contract under test is unverified. Install the review-tool\n" +
      "bundle (scripts/REVIEW-BUNDLE.md) and re-run.\n"
  );
  process.exit(2);
}

let failures = 0;
let passes = 0;

function check(name, fn) {
  try {
    fn();
    passes += 1;
    process.stdout.write(`  ok   ${name}\n`);
  } catch (e) {
    failures += 1;
    process.stdout.write(`  FAIL ${name}\n       ${e.message.split("\n")[0]}\n`);
  }
}

let scratchDir = null;
function scratch() {
  if (!scratchDir) scratchDir = fs.mkdtempSync(path.join(os.tmpdir(), "gate-hardening-"));
  return scratchDir;
}

let seq = 0;
function writeVerdict(verdict) {
  seq += 1;
  const dir = path.join(scratch(), `case-${seq}`);
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${verdict.task || "unnamed"}-review.json`);
  fs.writeFileSync(file, JSON.stringify(verdict, null, 2));
  return file;
}

function runGate(verdictPath, args = []) {
  const r = spawnSync(process.execPath, [GATE, verdictPath, "--max-rounds", "5", "--strikes", "2", "--static", ...args], {
    cwd: REPO,
    encoding: "utf8",
  });
  return { status: r.status, stdout: r.stdout || "", stderr: r.stderr || "" };
}

// Asserts a fail-closed refusal: exit 2, NO report on stdout (a report is what
// a caller persists and treats as a result), and a stderr message matching
// `pattern`. The empty-stdout assertion matters as much as the exit code — the
// original defect was a report that looked exactly like a passing one.
function rejects(verdict, pattern, args = []) {
  const r = runGate(writeVerdict(verdict), args);
  assert.strictEqual(r.status, 2, `expected exit 2, got ${r.status}. stderr: ${r.stderr}`);
  assert.strictEqual(r.stdout, "", "an input-rejection must emit no gate report");
  assert.match(r.stderr, pattern, `message did not name the hole. stderr: ${r.stderr}`);
  return r;
}

// Asserts the gate reported (exit 0/1/3) and returns the parsed report.
function reports(verdict, args = []) {
  const r = runGate(writeVerdict(verdict), args);
  assert.notStrictEqual(r.stdout, "", `expected a gate report, got exit ${r.status}. stderr: ${r.stderr}`);
  return { status: r.status, report: JSON.parse(r.stdout) };
}

function auditEntry(round, extra = {}) {
  return Object.assign(
    {
      round,
      reviewed_sha: `reviewed-${round}`,
      fix_sha: `fix-${round}`,
      specialists_run: [{ name: "reviewer", selection_reason: "always (owner)" }],
    },
    extra
  );
}

// The §"Minimal valid verdict" shape, minus `trace_schema_version` so the round
// carries no trace obligation (no trace file exists beside these fixtures) and
// the base case is a clean exit 0.
function base(overrides = {}) {
  return Object.assign(
    {
      task: "gate-hardening-selftest",
      date: "2026-07-25T10:00:22.000Z",
      status: "GO",
      mode: "full",
      round: 1,
      cycle: 1,
      no_go_round: 0,
      auto_fixes_applied: 0,
      findings: [],
      cross_runtime_findings: [],
      cross_runtime: {},
      rounds_audit: [auditEntry(1)],
      no_go_reason: null,
      summary: "No findings.",
      escalation_needed: false,
      escalation_reason: null,
      normalized_by: "2.0.0",
    },
    overrides
  );
}

function finding(overrides = {}) {
  return Object.assign(
    {
      severity: "HIGH",
      confidence: 4,
      path: "src/thing.ml",
      line: 42,
      category: "correctness",
      summary: "a HIGH finding",
      evidence: "src/thing.ml:42",
      fix: "fix it",
      fingerprint: "src/thing.ml:42:correctness",
      specialist: "reviewer",
      status: "OPEN",
      first_seen_round: 1,
    },
    overrides
  );
}

process.stdout.write("check-review-convergence hardening (fail-closed)\n");

// ── control ──────────────────────────────────────────────────────────────
check("CONTROL: the base fixture passes cleanly (exit 0, no violations)", () => {
  const { status, report } = reports(base());
  assert.strictEqual(status, 0, `base fixture is not clean: ${JSON.stringify(report.violations)}`);
  assert.deepStrictEqual(report.violations, []);
  assert.strictEqual(report.legacy_skip_authorized, false);
  assert.strictEqual(typeof report.hardening.version, "string", "report must carry hardening.version (stale-script signal)");
});

// ── G1 / D3 — the named all-null-strike escape ───────────────────────────
check("G1 rejects strike:null on a gated past round", () => {
  rejects(
    base({
      round: 3,
      rounds_audit: [auditEntry(1, { strike: false }), auditEntry(2, { strike: null }), auditEntry(3)],
    }),
    /rounds_audit entry for round 2 has strike null/
  );
});

check("G1 rejects a non-boolean strike written onto the round being gated", () => {
  rejects(
    base({ round: 2, rounds_audit: [auditEntry(1, { strike: false }), auditEntry(2, { strike: null })] }),
    /rounds_audit entry for round 2 has strike null/
  );
});

check("G1 rejects strike: \"false\" — the string, not the boolean", () => {
  rejects(
    base({ round: 3, rounds_audit: [auditEntry(1, { strike: false }), auditEntry(2, { strike: "false" }), auditEntry(3)] }),
    /has strike "false"/
  );
});

check("G1 CONTROL: strike ABSENT on the round being gated is correct, not rejected", () => {
  // The pipeline requires exactly this: the gate computes the strike, so the
  // draft must not pre-empt it. A rule that rejected this would be unusable.
  const { status } = reports(base({ round: 2, rounds_audit: [auditEntry(1, { strike: false }), auditEntry(2)] }));
  assert.strictEqual(status, 0);
});

// ── G2 / D1 — the named unread cross_runtime_findings array ──────────────
check("G2 rejects a HIGH OPEN cross_runtime finding that was never mirrored into findings[]", () => {
  const { status, report } = reports(base({ status: "NO-GO", no_go_reason: { type: "cross-runtime-finding" }, cross_runtime_findings: [finding()] }));
  assert.strictEqual(status, 1);
  const violation = report.violations.find((v) => v.type === "unmirrored-cross-runtime-finding");
  assert.ok(violation, `expected unmirrored-cross-runtime-finding, got ${JSON.stringify(report.violations)}`);
  assert.strictEqual(violation.cause, "unencodable-finding");
  assert.strictEqual(report.cause, "unencodable-finding");
});

check("G2 CONTROL: the same finding mirrored into findings[] is accepted", () => {
  const { status, report } = reports(base({ cross_runtime_findings: [finding()], findings: [finding()] }));
  assert.strictEqual(status, 0, JSON.stringify(report.violations));
});

check("G2 applies the ratchet to cross_runtime_findings, not only to findings[]", () => {
  // A RESOLVED HIGH+ with no round provenance is an unencodable-finding
  // violation in findings[]. Before D1 it was invisible in this array.
  const { status, report } = reports(
    base({
      status: "NO-GO",
      no_go_reason: { type: "cross-runtime-finding" },
      cross_runtime_findings: [finding({ status: "RESOLVED", first_seen_round: undefined })],
    })
  );
  assert.strictEqual(status, 1);
  assert.ok(report.violations.some((v) => v.type === "missing-round-provenance"), JSON.stringify(report.violations));
});

check("G2 rejects a non-array cross_runtime_findings", () => {
  rejects(base({ cross_runtime_findings: {} }), /cross_runtime_findings is present but not an array/);
});

check("G2 CONTROL: a MEDIUM cross-runtime finding does not need mirroring", () => {
  const { status } = reports(base({ cross_runtime_findings: [finding({ severity: "MEDIUM" })] }));
  assert.strictEqual(status, 0);
});

// ── G3 — absent `round`, the P3 vacuity core ─────────────────────────────
check("G3 rejects a verdict with no `round` key", () => {
  const verdict = base();
  delete verdict.round;
  rejects(verdict, /no `round` key/);
});

check("G3 the --allow-legacy skip is authorized AND recorded, never silent", () => {
  const verdict = base();
  delete verdict.round;
  const { status, report } = reports(verdict, ["--allow-legacy"]);
  assert.strictEqual(status, 0);
  assert.strictEqual(report.legacy_skip_authorized, true, "an authorized skip must be visible in the artifact");
  assert.strictEqual(report.current_round_strike, null);
});

// ── G4 / D10 — absent `no_go_round`, the quieter vacuity path ────────────
check("G4 rejects a verdict with no `no_go_round` key (the cap could never fire)", () => {
  const verdict = base();
  delete verdict.no_go_round;
  rejects(verdict, /no `no_go_round` key/);
});

// ── G5 — a rounds_audit gap erases the streak silently ───────────────────
check("G5 rejects a rounds_audit that skips a past round", () => {
  const { status, report } = reports(
    base({
      round: 4,
      rounds_audit: [auditEntry(1, { strike: false }), auditEntry(3, { strike: true }), auditEntry(4)],
    })
  );
  assert.strictEqual(status, 3, "a missing past entry is repairable — process-incomplete, never routed");
  const violation = report.violations.find((v) => v.type === "missing-past-audit");
  assert.ok(violation, JSON.stringify(report.violations));
  assert.match(violation.detail, /round 2/);
  assert.strictEqual(report.cause, null, "process-incomplete is never a top-level cause");
});

// ── G6 / D7 — finding status enum ────────────────────────────────────────
check("G6 rejects a finding status outside OPEN|RESOLVED|ACCEPTED", () => {
  // Live in this repo: briefs/make-tests-actually-run-review.json carries
  // status "OPEN-FOR-HUMAN", which matches neither of the gate's two tests.
  rejects(base({ findings: [finding({ status: "OPEN-FOR-HUMAN" })] }), /has status "OPEN-FOR-HUMAN"/);
});

check("G6 rejects a near-miss of ACCEPTED, which would read as un-waived", () => {
  rejects(base({ findings: [finding({ status: "ACCEPTED-BY-HUMAN" })] }), /findings entry .* has status "ACCEPTED-BY-HUMAN"/);
});

check("G6 applies the status enum to cross_runtime_findings too", () => {
  rejects(
    base({ cross_runtime_findings: [finding({ status: "open" })] }),
    /cross_runtime_findings entry .* has status "open"/
  );
});

// ── G7 / D2 — the routing keys the gate never read ───────────────────────
check("G7 rejects a typo'd no_go_reason.type (it would route nowhere)", () => {
  rejects(
    base({ status: "NO-GO", no_go_reason: { type: "design-not-convering" } }),
    /no_go_reason.type is "design-not-convering"/
  );
});

check("G7 rejects a NO-GO verdict with no no_go_reason at all", () => {
  rejects(base({ status: "NO-GO", no_go_reason: null }), /status NO-GO with no no_go_reason/);
});

check("G7 rejects a status outside GO|NO-GO", () => {
  rejects(base({ status: "PASS" }), /field status is "PASS"/);
});

check("G7 rejects a no_go_reason.cause outside the closed cause enum", () => {
  rejects(
    base({ status: "NO-GO", no_go_reason: { type: "design-not-converging", cause: "process-incomplete" } }),
    /no_go_reason.cause is "process-incomplete"/
  );
});

check("G7 flags a verdict asserting GO while the gate reports a design violation", () => {
  const { status, report } = reports(base({ status: "GO", findings: [finding({ check_encodable: false })] }));
  assert.strictEqual(status, 1);
  assert.ok(report.violations.some((v) => v.type === "go-with-design-violation"), JSON.stringify(report.violations));
});

// ── G8 / D5 — the mode enum and the report-mode name collision ───────────
check("G8 rejects a capitalised mode typo (it silently drops the scope-gate obligation)", () => {
  rejects(base({ mode: "Full" }), /field mode is "Full"/);
});

check("G8 rejects the gate REPORT's mode enum copied into the verdict", () => {
  const r = rejects(base({ mode: "static" }), /field mode is "static"/);
  assert.match(r.stderr, /name collision/, "the message must explain the collision, not just reject");
});

// ── G9 / D4 — task is load-bearing on every non-legacy round ─────────────
check("G9 rejects an absent task on a non-legacy round", () => {
  const verdict = base();
  delete verdict.task;
  rejects(verdict, /has no `task` key/);
});

check("G9 rejects a path-traversing task slug", () => {
  rejects(base({ task: "../escape" }), /not a valid slug/);
});

// ── G10 / D11 — cycle presence and type ──────────────────────────────────
check("G10 rejects an absent cycle", () => {
  const verdict = base();
  delete verdict.cycle;
  rejects(verdict, /no `cycle` key/);
});

check("G10 rejects a stringly-typed cycle (it silently became null)", () => {
  rejects(base({ cycle: "1" }), /field cycle is "1"/);
});

// ── G11 / D6 — an unnormalized verdict has no stable fingerprints ────────
check("G11 rejects a verdict that never went through review-normalize.js", () => {
  const verdict = base();
  delete verdict.normalized_by;
  const r = rejects(verdict, /has no `normalized_by` key/);
  assert.match(r.stderr, /`novel finding` is uncomputable/, "the message must say why, not just which key");
});

check("G11 rejects an empty normalized_by", () => {
  rejects(base({ normalized_by: "   " }), /field normalized_by is "   ", not a non-empty string/);
});

// ── G12 / D9 — cross_runtime key asymmetry and status enum ───────────────
check("G12 rejects the journal's `digest` key written into the verdict", () => {
  const r = rejects(
    base({ cross_runtime: { codex: { status: "healthy", digest: "abc123", round: 1 } } }),
    /carries `digest` but not `config_digest`/
  );
  assert.match(r.stderr, /corroboration\s+silently finds nothing/);
});

check("G12 rejects an unrecognised cross_runtime status (never corroborated, never warned)", () => {
  rejects(base({ cross_runtime: { codex: { status: "ok", config_digest: "abc123" } } }), /status is "ok"/);
});

check("G12 CONTROL: the documented five statuses are accepted", () => {
  for (const status of ["healthy", "degraded", "skipped-degraded", "skipped-human", "blocked"]) {
    const { status: exit } = reports(base({ cross_runtime: { codex: { status, config_digest: "abc", reason: "r" } } }));
    assert.notStrictEqual(exit, 2, `status ${status} must not be degraded input`);
  }
});

// ── G13 — blind container coercions ──────────────────────────────────────
check("G13 rejects a non-array rounds_audit (it was silently read as [])", () => {
  rejects(base({ rounds_audit: {} }), /rounds_audit is present but not an array/);
});

check("G13 rejects a non-object cross_runtime (it was silently ignored)", () => {
  rejects(base({ cross_runtime: [] }), /cross_runtime is present but not an object/);
});

// ── G14/G15 (review finding F4) — absence-blindness in the hardening itself ─
//
// The first version of this module guarded `mode` and `status` with
// `has(review, key) && !VALID.has(value)`: it closed the malformed half and
// left the omission half open, three lines under a comment describing these as
// keys "whose omission used to remove an obligation rather than fail one".
// Measured at the time:
//
//   mode: "Full"      -> exit 2, "not one of express | fast | full (D5)"
//   mode key OMITTED  -> exit 0, violations: []
//
// Every case below is paired: the malformed form was ALREADY rejected, so the
// omitted form is the one that proves the hole is shut. Three of the five keys
// (`cycle`, `task`, `normalized_by`) did check absence, which is exactly why
// hand-written guards looked complete — hence the table in the module.

check("F4 rejects an OMITTED mode, not just a misspelled one", () => {
  // The dropped obligation is real: review-trace-rules.js requires the
  // scope-gate trace line only when mode === "full".
  rejects(base({ mode: "Full" }), /field mode is "Full"/); // was already caught
  const verdict = base();
  delete verdict.mode;
  const r = rejects(verdict, /has no `mode` key/); // was exit 0
  assert.match(r.stderr, /DROPS that obligation rather than failing it/);
});

check("F4 rejects an OMITTED status, which disarmed the GO-consistency check", () => {
  // With status present, an unmirrored HIGH yields BOTH violations.
  const withStatus = reports(base({ status: "GO", cross_runtime_findings: [finding()] }));
  assert.deepStrictEqual(
    withStatus.report.violations.map((v) => v.type).sort(),
    ["go-with-design-violation", "unmirrored-cross-runtime-finding"]
  );
  // Deleting the key used to silently drop go-with-design-violation.
  const verdict = base({ cross_runtime_findings: [finding()] });
  delete verdict.status;
  rejects(verdict, /has no `status` key/);
});

check("F4 rejects an OMITTED cross_runtime_findings, not just a non-array one", () => {
  rejects(base({ cross_runtime_findings: {} }), /not an array/); // was already caught
  const verdict = base();
  delete verdict.cross_runtime_findings;
  rejects(verdict, /has no `cross_runtime_findings` key/); // was exit 0
});

check("F4 rejects cross_runtime: null — the carve-out its own message condemned", () => {
  // checkContainerTypes had an explicit `!== null` exemption, granting null
  // precisely the "silently ignored" treatment the message warns about.
  const r = rejects(base({ cross_runtime: null }), /field cross_runtime is null/);
  assert.match(r.stderr, /drops every cross-runtime corroboration check/);
});

check("F4 rejects an OMITTED rounds_audit and an OMITTED cross_runtime", () => {
  const noAudit = base();
  delete noAudit.rounds_audit;
  rejects(noAudit, /has no `rounds_audit` key/);

  const noXruntime = base();
  delete noXruntime.cross_runtime;
  rejects(noXruntime, /has no `cross_runtime` key/);
});

check("F4 rejects an OMITTED findings array", () => {
  const verdict = base();
  delete verdict.findings;
  rejects(verdict, /has no `findings` key/);
});

check("F4 rejects a cross_runtime entry with NO status", () => {
  // computeCrossRuntimeCorroboration only attests healthy|degraded|skipped-human,
  // so a statusless entry is never corroborated — the omission bought the same
  // silence an unrecognised value would have.
  rejects(base({ cross_runtime: { codex: { config_digest: "abc", round: 1 } } }), /status is absent/);
});

check("G14 rejects a null / non-object rounds_audit ELEMENT", () => {
  // The container type was checked; the element types were not. Every consumer
  // opens with `if (!e || typeof e !== "object") continue`.
  const audit = (extra) => base({ round: 3, rounds_audit: [auditEntry(1, { strike: false }), extra, auditEntry(3)] });
  rejects(audit(null), /rounds_audit\[1\] is not an object/);
  rejects(audit("x"), /rounds_audit\[1\] is not an object/);
});

check("G14 rejects a rounds_audit element with no numeric round", () => {
  rejects(
    base({ round: 3, rounds_audit: [auditEntry(1, { strike: false }), { reviewed_sha: "a", strike: true }, auditEntry(3)] }),
    /rounds_audit\[1\] has no numeric `round`/
  );
});

check("G14 rejects a null element in findings and in cross_runtime_findings", () => {
  rejects(base({ findings: [null] }), /findings\[0\] is not an object/);
  rejects(base({ cross_runtime_findings: [finding(), "x"] }), /cross_runtime_findings\[1\] is not an object/);
});

check("F4 CONTROL: every required key present and well-formed still passes", () => {
  // The base fixture carries all nine table keys. If this ever goes red, the
  // table has grown a requirement the assembler does not emit.
  const { status } = reports(base());
  assert.strictEqual(status, 0);
});

// ── teardown ─────────────────────────────────────────────────────────────
if (scratchDir) fs.rmSync(scratchDir, { recursive: true, force: true });

process.stdout.write(`\n${passes} passed, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
