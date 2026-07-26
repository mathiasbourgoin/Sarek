#!/usr/bin/env node
// scripts/check-mandated-steps.test.js — zero-dependency, node-runnable:
//
//   node scripts/check-mandated-steps.test.js
//
// Same discipline as scripts/check-review-convergence-hardening.test.js: each
// case constructs the ledger that SHOULD be rejected and proves the real
// scripts/check-mandated-steps.js rejects it with the right exit code and a
// message that names the missing or malformed record.
//
// The three scenarios that motivated the tool are reproduced literally:
//   - preflight never run across a day of implementation work
//   - a human gate skipped under a standing autonomy delegation
//   - a specialist disabled by its own breaker
"use strict";

const assert = require("assert");
const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const TOOL = path.join(REPO, "scripts", "check-mandated-steps.js");
const TASK = "steps-selftest";

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

function scratch() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mandated-steps-"));
  fs.mkdirSync(path.join(dir, "briefs"));
  return dir;
}

function run(argv, cwd) {
  const r = spawnSync(process.execPath, [TOOL, ...argv], { cwd, encoding: "utf8" });
  return { status: r.status, stdout: r.stdout || "", stderr: r.stderr || "" };
}

function record(dir, argv) {
  return run(["--record", "--task", TASK, ...argv], dir);
}

function checkPhase(dir, phase, extra = []) {
  return run(["--task", TASK, "--phase", phase, ...extra], dir);
}

function writeLedger(dir, lines) {
  fs.writeFileSync(path.join(dir, "briefs", `${TASK}-steps.jsonl`), lines.map((l) => (typeof l === "string" ? l : JSON.stringify(l))).join("\n") + "\n");
}

process.stdout.write("check-mandated-steps\n");

// ── the core rule: absence is a failure ──────────────────────────────────
check("an absent ledger fails, listing every mandated step (never exit 0)", () => {
  const dir = scratch();
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1, "no ledger at all must never read as a pass");
  assert.match(r.stderr, /no record for mandated step "preflight"/);
  assert.strictEqual(JSON.parse(r.stdout).ledger_exists, false);
});

check("SCENARIO: a full implement phase with preflight never run fails", () => {
  // The literal friction: an entire day of implementation work with
  // /roster-doctor preflight never invoked, and nothing in the record saying so.
  const dir = scratch();
  assert.strictEqual(record(dir, ["--step", "scope-gate", "--outcome", "ran", "--result", "PASS", "--actor", "agent"]).status, 0);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /no record for mandated step "preflight"/);
  assert.match(r.stderr, /indistinguishable from one that ran and passed/);
});

check("recording the preflight as RAN satisfies the requirement", () => {
  const dir = scratch();
  assert.strictEqual(record(dir, ["--step", "preflight", "--outcome", "ran", "--result", "READY", "--actor", "agent"]).status, 0);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 0, r.stderr);
});

check("recording the preflight as SKIPPED with a reason also satisfies it", () => {
  // Skipping is allowed. Skipping silently is not.
  const dir = scratch();
  const w = record(dir, ["--step", "preflight", "--outcome", "skipped", "--actor", "agent", "--reason", "no build or test in this doc-only phase"]);
  assert.strictEqual(w.status, 0);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 0, r.stderr);
  assert.match(JSON.parse(w.stdout).record.reason, /doc-only/);
});

check("a preflight that ran and found NOT-READY is recordable evidence, and needs a reason", () => {
  const dir = scratch();
  const bad = record(dir, ["--step", "preflight", "--outcome", "ran", "--result", "NOT-READY", "--actor", "agent"]);
  assert.strictEqual(bad.status, 0, "the writer accepts it; the checker is what demands the reason");
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /reports NOT-READY with no `reason`/);
});

// ── unreasoned skips ─────────────────────────────────────────────────────
check("the writer refuses --outcome skipped with no --reason", () => {
  const dir = scratch();
  const r = record(dir, ["--step", "preflight", "--outcome", "skipped", "--actor", "agent"]);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /requires --reason/);
});

check("the checker refuses a hand-written skip with no reason", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "skipped", actor: "agent" }]);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /outcome "skipped" with no `reason`/);
});

check("the checker refuses an empty-string reason", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "skipped", actor: "agent", reason: "   " }]);
  assert.strictEqual(checkPhase(dir, "implement").status, 1);
});

// ── SCENARIO: the human gate under a standing delegation ─────────────────
check("SCENARIO: an agent may not record the human gate as RAN", () => {
  const dir = scratch();
  writeLedger(dir, [
    { ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" },
    { ts: "2026-07-25T10:01:00Z", task: TASK, step: "human-gate", outcome: "ran", result: "PASS", actor: "agent" },
  ]);
  const r = checkPhase(dir, "ship");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /this step is human-only/);
  assert.match(r.stderr, /never journaled as a human's/);
});

check("SCENARIO: the same gate skipped under a recorded delegation passes", () => {
  const dir = scratch();
  writeLedger(dir, [
    { ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" },
    {
      ts: "2026-07-25T10:01:00Z",
      task: TASK,
      step: "human-gate",
      outcome: "skipped",
      actor: "agent",
      reason: "standing autonomy delegation for session 2026-07-25; user AFK",
    },
  ]);
  const r = checkPhase(dir, "ship");
  assert.strictEqual(r.status, 0, r.stderr);
});

check("a human-answered gate is recordable as RAN by a human actor", () => {
  const dir = scratch();
  writeLedger(dir, [
    { ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" },
    { ts: "2026-07-25T10:01:00Z", task: TASK, step: "human-gate", outcome: "ran", result: "PASS", actor: "human" },
  ]);
  assert.strictEqual(checkPhase(dir, "ship").status, 0);
});

// ── SCENARIO: the breaker-disabled cross-runtime pass ────────────────────
check("SCENARIO: cross-runtime breaker-skip is recordable, via explicit --require", () => {
  // `review` mandates neither scope-gate nor xruntime by default — the trace
  // mechanism owns both, and the measurement in mandated-steps-rules.js showed
  // requiring them here would fire on nearly every round. They stay opt-in.
  const dir = scratch();
  writeLedger(dir, [
    { ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" },
  ]);
  const missing = run(["--task", TASK, "--require", "xruntime"], dir);
  assert.strictEqual(missing.status, 1);
  assert.match(missing.stderr, /no record for mandated step "xruntime"/);

  assert.strictEqual(
    record(dir, [
      "--step", "xruntime", "--outcome", "skipped", "--actor", "agent",
      "--reason", "circuit breaker: runtime degraded this cycle with unchanged digest",
    ]).status,
    0
  );
  assert.strictEqual(run(["--task", TASK, "--require", "xruntime"], dir).status, 0);
});

check("MEASURED: `review` mandates nothing by default (the trace mechanism owns it)", () => {
  // Red-on-mutation for the measurement decision itself: if someone re-adds
  // scope-gate/xruntime to the review defaults, this goes red and they have to
  // read why. A ledger with only a preflight record must satisfy `review`.
  const dir = scratch();
  writeLedger(dir, [
    { ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" },
  ]);
  const r = checkPhase(dir, "review");
  assert.strictEqual(r.status, 0, `review must not mandate steps the trace mechanism owns: ${r.stderr}`);
  assert.deepStrictEqual(JSON.parse(r.stdout).required, []);
});

check("a degraded specialist is repeatable — several may be recorded in one phase", () => {
  const dir = scratch();
  writeLedger(dir, [
    { ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" },
    { ts: "2026-07-25T10:00:30Z", task: TASK, step: "degraded-specialist", outcome: "skipped", actor: "agent", reason: "type-design-analyzer: spawn-error" },
    { ts: "2026-07-25T10:00:40Z", task: TASK, step: "degraded-specialist", outcome: "skipped", actor: "agent", reason: "silent-failure-hunter: timeout" },
  ]);
  assert.strictEqual(checkPhase(dir, "qa").status, 0);
});

// ── the typo hole: a near-miss must not stand in for the real step ───────
check("a typo'd step id is rejected AND leaves the real step missing", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: TASK, step: "prefligt", outcome: "ran", result: "READY", actor: "agent" }]);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /unknown step "prefligt"/);
  assert.match(r.stderr, /no record for mandated step "preflight"/, "the typo must not satisfy the requirement it resembles");
});

check("the writer refuses an unknown --step outright", () => {
  const dir = scratch();
  const r = record(dir, ["--step", "preflght", "--outcome", "ran", "--result", "READY", "--actor", "agent"]);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /not a known step/);
});

// ── enum + duplicate + integrity rules ───────────────────────────────────
check("an outcome outside ran|skipped is rejected", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "partial", actor: "agent" }]);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /has outcome "partial"/);
});

check("an unattributable record (no actor) is rejected", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", result: "READY" }]);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /a skip must be attributable/);
});

check("a `ran` record with no result is rejected — it must say what it found", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "ran", actor: "agent" }]);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /must say what it found/);
});

check("a record with no timestamp is rejected", () => {
  const dir = scratch();
  writeLedger(dir, [{ task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" }]);
  assert.match(checkPhase(dir, "implement").stderr, /has no `ts` timestamp/);
});

check("two records for one step are rejected — one of them is not evidence", () => {
  const dir = scratch();
  writeLedger(dir, [
    { ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "skipped", actor: "agent", reason: "in a hurry" },
    { ts: "2026-07-25T10:05:00Z", task: TASK, step: "preflight", outcome: "ran", result: "READY", actor: "agent" },
  ]);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /is recorded twice/);
});

check("a record stamped with another task is rejected", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: "some-other-task", step: "preflight", outcome: "ran", result: "READY", actor: "agent" }]);
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 1);
  assert.match(r.stderr, /is stamped task "some-other-task"/);
});

// ── the checker's own fail-closed behaviour ──────────────────────────────
check("a malformed ledger line is exit 2, never a silently dropped record", () => {
  const dir = scratch();
  fs.writeFileSync(path.join(dir, "briefs", `${TASK}-steps.jsonl`), '{"step":"preflight"\n');
  const r = checkPhase(dir, "implement");
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /is not valid JSON/);
});

check("an unknown phase is rejected, not defaulted to `nothing required`", () => {
  const dir = scratch();
  const r = checkPhase(dir, "implemnt");
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /unknown phase "implemnt"/);
  assert.match(r.stderr, /Refusing to default/);
});

check("--require names an unknown step and is rejected", () => {
  const dir = scratch();
  const r = run(["--task", TASK, "--require", "preflight,nonsense"], dir);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /unknown step\(s\): nonsense/);
});

check("a phase with no mandated steps still validates the records it has", () => {
  const dir = scratch();
  writeLedger(dir, [{ ts: "2026-07-25T10:00:22Z", task: TASK, step: "preflight", outcome: "skipped", actor: "agent" }]);
  const r = checkPhase(dir, "spec");
  assert.strictEqual(r.status, 1, "an empty requirement set must not make the checker vacuous");
  assert.match(r.stderr, /with no `reason`/);
});

process.stdout.write(`\n${passes} passed, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
