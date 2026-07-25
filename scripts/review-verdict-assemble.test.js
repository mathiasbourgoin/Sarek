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

// Runs the gate WITHOUT parsing its stdout. Required for the fail-closed
// cases: an input-rejection exit 2 deliberately emits no report at all, so
// JSON.parse()ing it would mask the refusal as a parse error.
function gateRaw(verdictPath, args = []) {
  const r = spawnSync(process.execPath, [GATE, verdictPath, "--max-rounds", "5", "--strikes", "2", "--static", ...args], {
    cwd: REPO,
    encoding: "utf8",
  });
  return { status: r.status, stdout: r.stdout || "", stderr: r.stderr || "" };
}

function gate(verdictPath, args = []) {
  const r = gateRaw(verdictPath, args);
  assert.notStrictEqual(r.stdout, "", `gate emitted no report (exit ${r.status}): ${r.stderr}`);
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

// The three cases below used to assert the DEFECT — they are the executable
// record of what "a gate that cannot fail" looked like here, and each now
// asserts the refusal that replaced it. Read the old expectations in git
// history alongside these: same mutation, opposite verdict.
check("MUTATION: strike:null on a past round is now refused, not warned about", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 5);
  const verdict = JSON.parse(fs.readFileSync(final, "utf8"));
  verdict.rounds_audit.find((e) => e.round === 4).strike = null;
  fs.writeFileSync(final, JSON.stringify(verdict, null, 2));

  // WAS: exit 1, cause "round-cap", the streak silently reset by the null, and
  // the only signal a warning inside a report nobody reads.
  const mutated = gateRaw(final);
  assert.strictEqual(mutated.status, 2, "a non-boolean strike is degraded input, not a warning");
  assert.match(mutated.stderr, /rounds_audit entry for round 4 has strike null/);
  assert.match(mutated.stderr, /silently resets/);
});

check("MUTATION: all-null strikes below the cap are refused (was: total escape, exit 0)", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 5);
  const verdict = JSON.parse(fs.readFileSync(final, "utf8"));
  for (const entry of verdict.rounds_audit) entry.strike = null;
  verdict.no_go_round = 2; // below --max-rounds, as in the real five-round PR
  fs.writeFileSync(final, JSON.stringify(verdict, null, 2));

  // WAS the headline defect (schema/review-json-schema.md §strike): five
  // striking rounds, `current_round_strike: true`, `violations: []`, exit 0.
  const escaped = gateRaw(final);
  assert.strictEqual(escaped.status, 2);
  assert.strictEqual(escaped.stdout, "", "an input-rejection exit 2 emits no report to mistake for a pass");
  assert.match(escaped.stderr, /must be a\s+boolean/);
});

check("MUTATION: dropping `round` is refused unless the skip is explicitly authorized", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 5);
  const verdict = JSON.parse(fs.readFileSync(final, "utf8"));
  delete verdict.round;
  fs.writeFileSync(final, JSON.stringify(verdict, null, 2));

  // WAS: exit 1 via the round cap alone — the skipped strike/audit/trace checks
  // contributed nothing, and below the cap the same mutation exited 0.
  const skipped = gateRaw(final);
  assert.strictEqual(skipped.status, 2);
  assert.match(skipped.stderr, /no `round` key/);
  assert.match(skipped.stderr, /--allow-legacy/);

  // The escape hatch exists, but it leaves a mark: authorized, recorded, and
  // refused downstream.
  const authorized = gate(final, ["--allow-legacy"]);
  assert.strictEqual(authorized.report.legacy_skip_authorized, true);
  assert.strictEqual(authorized.report.current_round_strike, null);

  const dirReport = path.join(dir, "briefs", `${TASK}-authorized-report.json`);
  fs.writeFileSync(dirReport, authorized.raw);
  const w = run(ASSEMBLE, ["--write-strike", "--verdict", final, "--gate-report", dirReport], dir);
  assert.strictEqual(w.status, 2, "a strike from a legacy-skipped gate must never be journaled");
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

// ── fail-closed on the inputs that decide WHICH gate checks run ──────────
// Each of the four below is a lever that leaves the verdict schema-valid while
// switching a gate check off. The assembler must refuse the lever, not emit a
// verdict the gate will silently under-check.

check("--write-strike refuses a gate report from a stale gate script (no config.strikes / no trace)", () => {
  const dir = scratch();
  const { final } = driveRounds(dir, 1);
  const stale = path.join(dir, "briefs", "stale-report.json");

  // Boolean strike, right round — everything the old checks looked at. Only
  // the absent `config`/`trace` reveal a gate that predates the current rules.
  fs.writeFileSync(stale, JSON.stringify({ round: 1, current_round_strike: false }));
  const noConfig = run(ASSEMBLE, ["--write-strike", "--verdict", final, "--gate-report", stale], dir);
  assert.strictEqual(noConfig.status, 2);
  assert.match(noConfig.stderr, /no numeric config\.strikes/);

  fs.writeFileSync(stale, JSON.stringify({ round: 1, current_round_strike: false, config: { strikes: 2 } }));
  const noTrace = run(ASSEMBLE, ["--write-strike", "--verdict", final, "--gate-report", stale], dir);
  assert.strictEqual(noTrace.status, 2);
  assert.match(noTrace.stderr, /no `trace` block/);

  // and nothing was written: the persisted strike is still the real gate's
  // `false`, not a value copied out of the stale report.
  assert.strictEqual(JSON.parse(fs.readFileSync(final, "utf8")).rounds_audit[0].strike, false);
});

check("refuses a SKIPPED round — the hole would erase the streak", () => {
  const dir = scratch();
  driveRounds(dir, 2);
  const r = assembleRound(dir, 4); // lifecycle derives 3
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /--round 4 but .*derives 3/);
  assert.match(r.stderr, /erases the novel-finding streak/);
  assert.strictEqual(fs.existsSync(path.join(dir, "briefs", `${TASK}-review.json.draft`)), false,
    "a refused round must not leave a draft behind");
});

check("refuses a --no-go-round that does not hold or advance by one", () => {
  const dir = scratch();
  driveRounds(dir, 2); // prior no_go_round === 2
  const jump = assembleRound(dir, 3, ["--no-go-round", "9"]);
  assert.strictEqual(jump.status, 2);
  assert.match(jump.stderr, /does not follow the prior verdict's 2/);

  const frozen = assembleRound(dir, 3, ["--no-go-round", "1"]);
  assert.strictEqual(frozen.status, 2, "a counter that goes backwards disables the round cap");
  assert.match(frozen.stderr, /disables the round-cap backstop/);

  // holding is legitimate (a NO-GO round that does not qualify)
  assert.strictEqual(assembleRound(dir, 3, ["--no-go-round", "2"]).status, 0);
});

check("refuses the trace-obligation levers: unknown --mode, non-1.0 --trace-schema-version", () => {
  const dir = scratch();
  const bogusMode = assembleRound(dir, 1, ["--mode", "Full"]);
  assert.strictEqual(bogusMode.status, 2, "'Full' !== 'full' — the gate would drop the scope-gate obligation");
  assert.match(bogusMode.stderr, /--mode must be one of express \| fast \| full/);

  const noTrace = assembleRound(dir, 1, ["--trace-schema-version", "none"]);
  assert.strictEqual(noTrace.status, 2);
  assert.match(noTrace.stderr, /--trace-schema-version must be "1\.0"/);

  // and the stamp is unconditional on a round that does assemble
  assert.strictEqual(assembleRound(dir, 1).status, 0);
  const entry = JSON.parse(fs.readFileSync(path.join(dir, "briefs", `${TASK}-review.json.draft`), "utf8")).rounds_audit[0];
  assert.strictEqual(entry.trace_schema_version, "1.0");
});

check("--allow-rejected no longer exists: a schema-invalid finding is never droppable", () => {
  const dir = scratch();
  const bad = path.join("briefs", "bad.json");
  fs.writeFileSync(path.join(dir, bad), JSON.stringify([{ severity: "HIGH", summary: "no other required keys" }]));
  const r = run(ASSEMBLE, ["--task", TASK, "--round", "1", "--cycle", "1", "--status", "GO",
    "--reviewed-sha", "a", "--fix-sha", "b", "--specialist", "reviewer=always",
    "--findings", bad, "--allow-rejected"], dir);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /--allow-rejected was removed/);
  assert.strictEqual(fs.existsSync(path.join(dir, "briefs", `${TASK}-review.json.draft`)), false);
});

check("refuses to carry forward a past round whose --write-strike never ran", () => {
  const dir = scratch();
  const final = path.join(dir, "briefs", `${TASK}-review.json`);
  driveRounds(dir, 2);

  // Simulate the two-phase protocol being half-applied: round 2 was gated but
  // its strike was never written back. computeStrikeMap() cannot see the entry,
  // so the streak resets across it and the gate's only signal is a warning on
  // a run that may exit 0.
  const v = JSON.parse(fs.readFileSync(final, "utf8"));
  delete v.rounds_audit.find((e) => e.round === 2).strike;
  fs.writeFileSync(final, JSON.stringify(v, null, 2));

  const r = assembleRound(dir, 3);
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /round 2 has no boolean `strike`/);
  assert.match(r.stderr, /silently resets the novel-finding streak/);
});

// ── findings must survive intake, and must be classifiable ───────────────

check("applies the normalizer's REOPENED disposition so E-4 can actually fire", () => {
  const dir = scratch();
  const final = path.join(dir, "briefs", `${TASK}-review.json`);
  driveRounds(dir, 1);

  // Mark round 1's HIGH as RESOLVED with no linked check. INV-2: a RESOLVED
  // entry with no check can never be verified, so re-reporting it next round
  // is always a reopen, never carry-forward noise.
  const v1 = JSON.parse(fs.readFileSync(final, "utf8"));
  v1.findings[0].status = "RESOLVED";
  v1.findings[0].resolved_round = 1;
  fs.writeFileSync(final, JSON.stringify(v1, null, 2));

  // Round 2 re-reports the SAME finding (same fingerprint), not a novel one.
  const reReport = [Object.assign(highFinding(1), { first_seen_round: 1 })];
  fs.writeFileSync(path.join(dir, "briefs", "reviewer-r2.json"), JSON.stringify(reReport));
  const a = run(ASSEMBLE, ["--task", TASK, "--round", "2", "--cycle", "1", "--status", "NO-GO",
    "--mode", "full", "--reviewed-sha", "reviewed-2", "--fix-sha", "fix-2",
    "--specialist", "reviewer=always (owner) on round 2", "--no-go-round", "2",
    "--no-go-type", "design-not-converging", "--findings", path.join("briefs", "reviewer-r2.json")], dir);
  assert.strictEqual(a.status, 0, a.stderr);
  assert.match(a.stderr, /REOPENED .* applied to findings/);

  const draft = JSON.parse(fs.readFileSync(path.join(dir, "briefs", `${TASK}-review.json.draft`), "utf8"));
  const f = draft.findings.find((x) => x.fingerprint === reReport[0].fingerprint);
  assert.strictEqual(f.status, "OPEN", "the reopened body must replace the RESOLVED ledger entry");
  assert.strictEqual(f.reopened_at_round, 2);

  // and the gate can now see it: isReopenedStrikeFinding() keys on exactly this.
  appendTraces(dir, 2);
  const g = gate(path.join(dir, "briefs", `${TASK}-review.json.draft`));
  assert.strictEqual(g.report.current_round_strike, true,
    "a reopened HIGH must strike (E-4) — it never can if dispositions are dropped");
});

check("refuses a CRITICAL/HIGH finding with no first_seen_round (silently unstrikeable)", () => {
  const dir = scratch();
  const bare = Object.assign(highFinding(1), {});
  delete bare.first_seen_round;
  fs.writeFileSync(path.join(dir, "briefs", "bare.json"), JSON.stringify([bare]));
  const r = run(ASSEMBLE, ["--task", TASK, "--round", "1", "--cycle", "1", "--status", "NO-GO",
    "--reviewed-sha", "a", "--fix-sha", "b", "--specialist", "reviewer=always",
    "--no-go-round", "1", "--no-go-type", "design-not-converging",
    "--findings", path.join("briefs", "bare.json")], dir);
  // The normalizer does NOT stamp first_seen_round, and isNovelStrikeFinding()
  // requires it — so without this refusal the round can never strike at all.
  assert.strictEqual(r.status, 2);
  assert.match(r.stderr, /missing or future first_seen_round/);
});

check("refuses GO while a CRITICAL/HIGH is still OPEN", () => {
  const dir = scratch();
  fs.writeFileSync(path.join(dir, "briefs", "open.json"), JSON.stringify([highFinding(1)]));
  const r = run(ASSEMBLE, ["--task", TASK, "--round", "1", "--cycle", "1", "--status", "GO",
    "--reviewed-sha", "a", "--fix-sha", "b", "--specialist", "reviewer=always",
    "--findings", path.join("briefs", "open.json")], dir);
  assert.strictEqual(r.status, 2, "the gate never reads `status`, so nothing else would catch this");
  assert.match(r.stderr, /open CRITICAL\/HIGH finding/);
});

check("carries cross_runtime_findings forward within a cycle (augment-only)", () => {
  const dir = scratch();
  const final = path.join(dir, "briefs", `${TASK}-review.json`);
  driveRounds(dir, 1);
  const v1 = JSON.parse(fs.readFileSync(final, "utf8"));
  v1.cross_runtime_findings = [Object.assign(highFinding(1), {
    specialist: "codex-xruntime", fingerprint: "src/xr.ml:1:correctness", path: "src/xr.ml", line: 1,
  })];
  fs.writeFileSync(final, JSON.stringify(v1, null, 2));

  assert.strictEqual(assembleRound(dir, 2).status, 0);
  const draft = JSON.parse(fs.readFileSync(path.join(dir, "briefs", `${TASK}-review.json.draft`), "utf8"));
  assert.strictEqual(draft.cross_runtime_findings.length, 1,
    "round 2 supplied no cross-runtime findings, so round 1's must still be there");
  assert.strictEqual(draft.cross_runtime_findings[0].fingerprint, "src/xr.ml:1:correctness");
});

// ── the remaining caller-controlled enums ────────────────────────────────
check("refuses an invalid --task slug (gate exit 2, and it composes the briefs/ paths)", () => {
  const dir = scratch();
  const base = ["--round", "1", "--cycle", "1", "--status", "GO", "--reviewed-sha", "a",
    "--fix-sha", "b", "--specialist", "reviewer=always"];
  for (const bad of ["My Task", "../pwned"]) {
    const r = run(ASSEMBLE, ["--task", bad, ...base], dir);
    assert.strictEqual(r.status, 2, `--task ${JSON.stringify(bad)} was accepted`);
    assert.match(r.stderr, /is not a valid slug/);
  }
  assert.strictEqual(fs.existsSync(path.join(dir, "..", "pwned-review.json.draft")), false);
});

check("refuses a typo'd --no-go-type / --no-go-cause and a non-human streak override", () => {
  const dir = scratch();
  const base = ["--task", TASK, "--round", "1", "--cycle", "1", "--reviewed-sha", "a",
    "--fix-sha", "b", "--specialist", "reviewer=always"];
  const noGo = [...base, "--status", "NO-GO", "--no-go-round", "1"];

  assert.match(run(ASSEMBLE, [...noGo, "--no-go-type", "desgin-not-converging"], dir).stderr,
    /--no-go-type must be one of/);
  assert.match(run(ASSEMBLE, [...noGo, "--no-go-type", "design-not-converging",
    "--no-go-cause", "process-incomplete"], dir).stderr, /--no-go-cause must be one of/);
  assert.match(run(ASSEMBLE, [...base, "--status", "GO", "--streak-override-by", "reviewer-agent"], dir).stderr,
    /--streak-override-by must be "human"/);
});

process.stdout.write(`\n${passes} passed, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
