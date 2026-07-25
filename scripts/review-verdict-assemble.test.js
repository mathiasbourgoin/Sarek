#!/usr/bin/env node
// scripts/review-verdict-assemble.test.js — zero-dependency, node-runnable.
//
//   node scripts/review-verdict-assemble.test.js
//
// Exercises scripts/review-verdict-assemble.js end-to-end against the REAL
// scripts/check-review-convergence.js — the point is not that the assembler's
// own units work, it is that the verdicts it emits make the gate run its real
// checks instead of taking the B-8 legacy skip. Each case below therefore
// asserts on the gate's exit code and top-level `cause`, and the round-cap /
// novel-finding-streak cases are red-on-mutation: flipping one `strike` to
// null (or dropping `round`) changes the asserted outcome.
//
// Repo-local test (the upstream review-tool bundle does not own it).
"use strict";

const assert = require("assert");
const { execFileSync, spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const ASSEMBLE = path.join(REPO, "scripts", "review-verdict-assemble.js");
const GATE = path.join(REPO, "scripts", "check-review-convergence.js");
const TRACE = path.join(REPO, "scripts", "lib", "review", "review-trace.js");
const TASK = "assemble-selftest";

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
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "assemble-test-"));
  fs.mkdirSync(path.join(dir, "briefs"));
  return dir;
}

function highFinding(round) {
  return {
    severity: "HIGH",
    confidence: 4,
    path: `src/round${round}.ml`,
    line: round * 10,
    category: "correctness",
    summary: `novel HIGH finding first raised in round ${round}`,
    evidence: `src/round${round}.ml line ${round * 10} — quoted code`,
    fix: "fix it",
    fingerprint: `src/round${round}.ml:${round * 10}:correctness`,
    specialist: "reviewer",
    first_seen_round: round,
    status: "OPEN",
  };
}

function run(cmd, argv, cwd) {
  const r = spawnSync(process.execPath, [cmd, ...argv], { cwd, encoding: "utf8" });
  return { status: r.status, stdout: r.stdout || "", stderr: r.stderr || "" };
}

function assembleRound(dir, round, extra = []) {
  const findingsFile = path.join("briefs", `reviewer-r${round}.json`);
  fs.writeFileSync(path.join(dir, findingsFile), JSON.stringify([highFinding(round)]));
  return run(
    ASSEMBLE,
    [
      "--task", TASK,
      "--round", String(round),
      "--cycle", "1",
      "--status", "NO-GO",
      "--mode", "full",
      "--reviewed-sha", `reviewed-${round}`,
      "--fix-sha", `fix-${round}`,
      "--specialist", `reviewer=always (owner) on round ${round}`,
      "--no-go-round", String(round),
      "--no-go-type", "design-not-converging",
      "--findings", findingsFile,
      ...extra,
    ],
    dir
  );
}

// The scope-gate and specialist trace lines the gate demands for a Full-mode
// round. Appended here because the corresponding tools really did run in the
// scenario under test — never fabricate one for a tool that did not run
// (FR-177/C-3); this test is the only place a synthetic append is legitimate.
function appendTraces(dir, round) {
  for (const [event, actor] of [["scope-gate", "check-scope-diff.sh"], ["specialist", "reviewer"]]) {
    execFileSync(process.execPath, [
      TRACE, "--root", dir, "--task", TASK, "--round", String(round),
      "--cycle", "1", "--event", event, "--actor", actor, "--outcome", "ran",
    ]);
  }
}

function gate(verdictPath, args = []) {
  const r = spawnSync(process.execPath, [GATE, verdictPath, "--max-rounds", "5", "--strikes", "2", "--static", ...args], {
    cwd: REPO,
    encoding: "utf8",
  });
  return { status: r.status, report: JSON.parse(r.stdout), raw: r.stdout };
}

// Drives round 1..`upto` through assemble → traces → gate → --write-strike →
// promote, returning the last gate result and the persisted verdict path.
function driveRounds(dir, upto) {
  const draft = path.join(dir, "briefs", `${TASK}-review.json.draft`);
  const final = path.join(dir, "briefs", `${TASK}-review.json`);
  const report = path.join(dir, "briefs", `${TASK}-gate-report.json`);
  let last = null;
  for (let round = 1; round <= upto; round++) {
    const a = assembleRound(dir, round);
    assert.strictEqual(a.status, 0, `assemble round ${round} failed: ${a.stderr}`);
    appendTraces(dir, round);
    last = gate(draft);
    fs.writeFileSync(report, last.raw);
    const w = run(ASSEMBLE, ["--write-strike", "--verdict", draft, "--gate-report", report], dir);
    assert.strictEqual(w.status, 0, `write-strike round ${round} failed: ${w.stderr}`);
    fs.renameSync(draft, final);
  }
  return { last, final };
}

process.stdout.write("review-verdict-assemble\n");

// ── the anti-vacuity contract ────────────────────────────────────────────
check("refuses to emit a verdict without --round (the B-8 vacuous-gate hole)", () => {
  const dir = scratch();
  const r = run(ASSEMBLE, ["--task", TASK, "--cycle", "1", "--status", "GO", "--reviewed-sha", "a",
    "--fix-sha", "b", "--specialist", "reviewer=always"], dir);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /--round <n> is required/);
});

check("gate runs its REAL checks on an assembled verdict — no legacy skip", () => {
  const dir = scratch();
  const { last } = driveRounds(dir, 1);
  assert.strictEqual(last.report.legacy_round, false, "gate took the legacy path");
  assert.strictEqual(last.report.legacy_no_go_round, false);
  assert.strictEqual(last.report.trace.obligated, true, "trace checks were skipped");
  assert.strictEqual(last.report.trace.skipped, false);
  assert.deepStrictEqual(last.report.warnings, [], `unexpected warnings: ${JSON.stringify(last.report.warnings)}`);
  assert.strictEqual(last.status, 0);
  assert.strictEqual(last.report.current_round_strike, false, "round 1 must never strike");
});

check("round 1 emits no `strike` key until the gate reports it, then a boolean", () => {
  const dir = scratch();
  const draft = path.join(dir, "briefs", `${TASK}-review.json.draft`);
  assert.strictEqual(assembleRound(dir, 1).status, 0);
  const pre = JSON.parse(fs.readFileSync(draft, "utf8"));
  assert.ok(!("strike" in pre.rounds_audit[0]), "`strike` must be ABSENT pre-gate, never null");

  appendTraces(dir, 1);
  const report = path.join(dir, "briefs", `${TASK}-gate-report.json`);
  fs.writeFileSync(report, gate(draft).raw);
  assert.strictEqual(run(ASSEMBLE, ["--write-strike", "--verdict", draft, "--gate-report", report], dir).status, 0);
  assert.strictEqual(JSON.parse(fs.readFileSync(draft, "utf8")).rounds_audit[0].strike, false);
});

check("--write-strike refuses a non-boolean current_round_strike (never writes null)", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 1);
  const bogus = path.join(dir, "briefs", "legacy-report.json");
  fs.writeFileSync(bogus, JSON.stringify({ round: 1, current_round_strike: null }));
  const r = run(ASSEMBLE, ["--write-strike", "--verdict", final, "--gate-report", bogus], dir);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /not a boolean/);
});

// ── strike escalation, red-on-mutation ───────────────────────────────────
check("round 2 strikes but does not yet violate (round 1 can never strike)", () => {
  const dir = scratch();
  const { last } = driveRounds(dir, 2);
  assert.strictEqual(last.report.current_round_strike, true);
  assert.strictEqual(last.report.cause, null);
  assert.strictEqual(last.status, 0);
});

check("two consecutive striking rounds escalate: cause novel-finding-streak, exit 1", () => {
  const dir = scratch();
  const { last } = driveRounds(dir, 3);
  assert.strictEqual(last.report.cause, "novel-finding-streak");
  assert.strictEqual(last.status, 1);
});

check("five novel rounds: round-cap AND streak both fire; streak wins on precedence", () => {
  const dir = scratch();
  const { last, final } = driveRounds(dir, 5);
  assert.strictEqual(last.status, 1);
  const types = last.report.violations.map((v) => v.type).sort();
  assert.deepStrictEqual(types, ["novel-finding-streak", "round-cap"]);
  // FR-059/B-5 precedence: novel-finding-streak outranks round-cap, so
  // `cause` is NOT "round-cap" even though the cap is violated.
  assert.strictEqual(last.report.cause, "novel-finding-streak");
  const audit = JSON.parse(fs.readFileSync(final, "utf8")).rounds_audit;
  assert.deepStrictEqual(audit.map((e) => e.strike), [false, true, true, true, true]);
});

check("MUTATION: strike:null on a past round silently resets the streak (round-cap only)", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 5);
  const verdict = JSON.parse(fs.readFileSync(final, "utf8"));
  verdict.rounds_audit.find((e) => e.round === 4).strike = null;
  fs.writeFileSync(final, JSON.stringify(verdict, null, 2));

  const mutated = gate(final);
  assert.strictEqual(mutated.report.cause, "round-cap", "the streak must have been reset by the null");
  assert.deepStrictEqual(mutated.report.violations.map((v) => v.type), ["round-cap"]);
  assert.match(mutated.report.warnings.join("\n"), /round 4 lacks a boolean strike field/);
  assert.strictEqual(mutated.status, 1);
});

check("MUTATION: all-null strikes below the cap escape the gate entirely (exit 0)", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 5);
  const verdict = JSON.parse(fs.readFileSync(final, "utf8"));
  for (const entry of verdict.rounds_audit) entry.strike = null;
  verdict.no_go_round = 2; // below --max-rounds, as in the real five-round PR
  fs.writeFileSync(final, JSON.stringify(verdict, null, 2));

  const escaped = gate(final);
  // Documented defect (schema/review-json-schema.md §strike): five striking
  // rounds, `current_round_strike: true`, and the gate still reports no
  // violation. Warnings only. This is why `strike` must be a boolean.
  assert.strictEqual(escaped.report.current_round_strike, true);
  assert.deepStrictEqual(escaped.report.violations, []);
  assert.strictEqual(escaped.report.cause, null);
  assert.strictEqual(escaped.status, 0);
  assert.strictEqual(escaped.report.warnings.length, 3);
});

check("MUTATION: dropping `round` makes the gate skip strike/audit/trace checks (exit 0)", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 5);
  const verdict = JSON.parse(fs.readFileSync(final, "utf8"));
  delete verdict.round;
  fs.writeFileSync(final, JSON.stringify(verdict, null, 2));

  const skipped = gate(final);
  assert.strictEqual(skipped.report.legacy_round, true);
  assert.strictEqual(skipped.report.current_round_strike, null);
  assert.match(skipped.report.warnings.join("\n"), /round key absent — skipping strike and rounds_audit checks/);
  assert.strictEqual(skipped.status, 1, "only round-cap survives, via no_go_round");
});

// ── carry-forward + delegation contracts ─────────────────────────────────
check("rounds_audit is carried forward append-only and prior entries are untouched", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 3);
  const audit = JSON.parse(fs.readFileSync(final, "utf8")).rounds_audit;
  assert.deepStrictEqual(audit.map((e) => e.round), [1, 2, 3]);
  assert.strictEqual(audit[0].reviewed_sha, "reviewed-1", "round 1's entry was rewritten");
  assert.strictEqual(audit[1].reviewed_sha, "reviewed-2");
  for (const entry of audit) assert.strictEqual(entry.trace_schema_version, "1.0");
});

check("refuses to re-emit a round the prior verdict already journaled", () => {
  const dir = scratch();
  driveRounds(dir, 2);
  const r = assembleRound(dir, 2);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /append-only/);
});

check("findings are cumulative and carry forward across rounds", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 3);
  const verdict = JSON.parse(fs.readFileSync(final, "utf8"));
  assert.strictEqual(verdict.findings.length, 3);
  assert.deepStrictEqual(verdict.findings.map((f) => f.first_seen_round), [1, 2, 3]);
  for (const f of verdict.findings) assert.ok(typeof f.fid === "string" && f.fid.length > 0, "normalizer did not stamp fid");
  assert.strictEqual(typeof verdict.normalized_by, "string");
});

check("rejects the verdict ENVELOPE where review-normalize.js needs a findings ARRAY", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 1);
  const r = run(ASSEMBLE, ["--task", "other", "--round", "1", "--cycle", "1", "--status", "GO",
    "--reviewed-sha", "a", "--fix-sha", "b", "--specialist", "reviewer=always",
    "--findings", path.relative(dir, final)], dir);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /JSON ARRAY of finding objects, not a verdict envelope/);
});

check("fails closed on a schema-invalid finding rather than dropping it", () => {
  const dir = scratch();
  const bad = path.join("briefs", "bad.json");
  fs.writeFileSync(path.join(dir, bad), JSON.stringify([{ severity: "HIGH", summary: "no other required keys" }]));
  const r = run(ASSEMBLE, ["--task", TASK, "--round", "1", "--cycle", "1", "--status", "GO",
    "--reviewed-sha", "a", "--fix-sha", "b", "--specialist", "reviewer=always", "--findings", bad], dir);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /rejected as schema-invalid/);
});

check("requires exactly one of --fix-sha / --fix-sha-reason", () => {
  const dir = scratch();
  const base = ["--task", TASK, "--round", "1", "--cycle", "1", "--status", "GO",
    "--reviewed-sha", "a", "--specialist", "reviewer=always"];
  assert.strictEqual(run(ASSEMBLE, base, dir).status, 2);
  assert.strictEqual(run(ASSEMBLE, [...base, "--fix-sha", "b", "--fix-sha-reason", "dirty-tree"], dir).status, 2);
  const ok = run(ASSEMBLE, [...base, "--fix-sha-reason", "dirty-tree"], dir);
  assert.strictEqual(ok.status, 0, ok.stderr);
  const entry = JSON.parse(fs.readFileSync(path.join(dir, "briefs", `${TASK}-review.json.draft`), "utf8")).rounds_audit[0];
  assert.strictEqual(entry.fix_sha, null);
  assert.strictEqual(entry.fix_sha_reason, "dirty-tree");
});

check("requires --no-go-round on NO-GO and forbids a non-zero one on GO", () => {
  const dir = scratch();
  const base = ["--task", TASK, "--round", "1", "--cycle", "1", "--reviewed-sha", "a",
    "--fix-sha", "b", "--specialist", "reviewer=always"];
  assert.match(run(ASSEMBLE, [...base, "--status", "NO-GO", "--no-go-type", "design-not-converging"], dir).stderr,
    /--no-go-round <n> is required/);
  assert.match(run(ASSEMBLE, [...base, "--status", "GO", "--no-go-round", "3"], dir).stderr,
    /must be 0 \(or omitted\) on a GO verdict/);
});

check("every specialist entry carries a non-empty selection_reason", () => {
  const dir = scratch();
  const base = ["--task", TASK, "--round", "1", "--cycle", "1", "--status", "GO",
    "--reviewed-sha", "a", "--fix-sha", "b"];
  assert.match(run(ASSEMBLE, base, dir).stderr, /at least one --specialist/);
  assert.match(run(ASSEMBLE, [...base, "--specialist", "reviewer="], dir).stderr, /empty name or selection_reason/);
  assert.match(run(ASSEMBLE, [...base, "--specialist", "reviewer"], dir).stderr, /<name>=<selection_reason>/);
});

process.stdout.write(`\n${passes} passed, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
