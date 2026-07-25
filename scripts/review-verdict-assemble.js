#!/usr/bin/env node
// scripts/review-verdict-assemble.js — CommonJS, repo-local tool (NOT part of
// the upstream review-tool bundle; it is absent from
// scripts/review-bundle.manifest.json by design and survives a bundle upgrade).
//
// Assembles a schema-valid briefs/<task>-review.json draft so no verdict is
// ever hand-derived. Contract: schema/review-json-schema.md (the same document
// scripts/lib/review/*.js and skills/pipeline/roster-review.md cite).
//
// WHY THIS EXISTS: check-review-convergence.js silently degrades to a no-op on
// a verdict that lacks `round` ("legacy review.json: round key absent —
// skipping strike and rounds_audit checks (B-8)", violations: [], exit 0). A
// hand-written verdict under-populates exactly the keys the gate needs, so the
// anti-vacuity gate was itself vacuous. This tool refuses to emit a verdict
// without `round`, and never emits `strike: null` (which silently resets
// novel-finding-streak detection — see schema/review-json-schema.md §"strike").
//
// FAIL-CLOSED ON OBLIGATION, not just on shape. Emitting a schema-valid
// envelope is not enough: several of its fields are what the gate consults to
// decide WHETHER to run a check at all (`round`, `cycle`, `no_go_round`,
// `mode`, `trace_schema_version`, and the completeness of `findings`). A wrong
// one buys silence, not a wrong answer — so every one of them is validated
// against the lifecycle witness or a closed enum here, never passed through.
//
// PIPELINE ORDER (also at the top of the schema doc; load-bearing, because
// without normalization first "novel finding" is uncomputable — fingerprints
// are not yet stable):
//
//   specialists emit findings  →  normalize the ARRAY  →  assemble the verdict
//     →  run the gate  →  write strikes back  →  route on the verdict
//
// This tool owns steps 2, 3 and 5, delegating step 2 wholesale to
// scripts/review-normalize.js (fingerprinting is never reimplemented here).
//
// TWO PHASES — because `strike` is only knowable AFTER the gate reports it:
//
//   1) assemble:     --task <slug> --round <n> --cycle <n> --status <GO|NO-GO>
//                    --reviewed-sha <sha> (--fix-sha <sha> | --fix-sha-reason
//                    <text>) --specialist <name>=<reason> [...]
//                    [--findings <file.json> ...] [--out <path>] ...
//      → writes the draft with NO `strike` key on the current round's entry
//        (absent, never null — a null is silently accepted downstream and
//        resets the streak once the round becomes a past round).
//
//   2) write-strike: --write-strike --verdict <draft> --gate-report <json>
//      → copies the gate's boolean `current_round_strike` into the entry for
//        the report's own `round`. Refuses anything non-boolean, a mismatched
//        round, or a report from a stale gate script.
//
// Between them, run the gate exactly as roster-review §5.5 prescribes:
//   node scripts/check-review-convergence.js <draft> --max-rounds N --strikes N
//
// The rule layer — the closed enums, the lifecycle-continuity and append-only
// rules, disposition application, envelope construction — lives in
// scripts/lib/review-verdict-rules.js (repo-local too, and likewise absent
// from the bundle manifest). This file keeps CLI parsing, I/O, and
// orchestration.
//
// Exit codes: 0 success; 2 usage or degraded input (never a half-written
// verdict).
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const { deriveRoundState } = require("./lib/review/review-lifecycle");
const {
  TRACE_SCHEMA_VERSION,
  fail,
  warn,
  parseCount,
  validateAssembleArgs,
  parseSpecialists,
  assertGateReportIsCurrent,
  carryForwardRoundsAudit,
  buildAuditEntry,
  buildVerdict,
  reportNormalizerSideChannels,
  applyDispositions,
  assertStrikeableFindings,
  carryForwardCrossRuntime,
  assertGoIsClean,
  enforceLifecycle,
} = require("./lib/review-verdict-rules");

const NORMALIZER = path.resolve(__dirname, "review-normalize.js");

// ── argument parsing ─────────────────────────────────────────────────────
const VALUE_FLAGS = {
  "--task": "task",
  "--round": "round",
  "--cycle": "cycle",
  "--status": "status",
  "--mode": "mode",
  "--reviewed-sha": "reviewedSha",
  "--fix-sha": "fixSha",
  "--fix-sha-reason": "fixShaReason",
  "--no-go-round": "noGoRound",
  "--no-go-type": "noGoType",
  "--no-go-cause": "noGoCause",
  "--summary": "summary",
  "--auto-fixes": "autoFixes",
  "--escalation-reason": "escalationReason",
  "--streak-override-by": "streakOverrideBy",
  "--trace-schema-version": "traceSchemaVersion",
  "--prior": "prior",
  "--out": "out",
  "--gate-report": "gateReport",
  "--verdict": "verdict",
};

function emptyArgs() {
  return {
    findings: [],
    specialists: [],
    failedAcs: [],
    traceSchemaVersion: TRACE_SCHEMA_VERSION,
    mode: "full",
    autoFixes: "0",
    writeStrike: false,
    escalationNeeded: false,
  };
}

function parseArgs(argv) {
  const out = emptyArgs();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--findings") out.findings.push(requireValue(argv, ++i, a));
    else if (a === "--specialist") out.specialists.push(requireValue(argv, ++i, a));
    else if (a === "--failed-ac") out.failedAcs.push(requireValue(argv, ++i, a));
    else if (a === "--write-strike") out.writeStrike = true;
    else if (a === "--escalation-needed") out.escalationNeeded = true;
    else if (a === "--allow-rejected") {
      fail("--allow-rejected was removed: a schema-invalid finding is never droppable. Fix the finding against schema/review-finding.schema.json instead.");
    } else if (VALUE_FLAGS[a]) out[VALUE_FLAGS[a]] = requireValue(argv, ++i, a);
    else fail(`unknown flag or stray argument: ${a}`);
  }
  return out;
}

function requireValue(argv, index, flag) {
  const value = argv[index];
  if (value === undefined || value.startsWith("--")) fail(`${flag} requires a value`);
  return value;
}

function readJsonFile(filePath, label) {
  const resolved = path.resolve(process.cwd(), filePath);
  if (!fs.existsSync(resolved)) fail(`${label} not found: ${filePath}`);
  try {
    return JSON.parse(fs.readFileSync(resolved, "utf8"));
  } catch (e) {
    fail(`${label} is not valid JSON: ${e.message}`);
  }
}

function runWriteStrike(args) {
  if (!args.verdict) fail("--write-strike requires --verdict <path>");
  if (!args.gateReport) fail("--write-strike requires --gate-report <path>");

  const verdictPath = path.resolve(process.cwd(), args.verdict);
  const verdict = readJsonFile(args.verdict, "--verdict file");
  const report = readJsonFile(args.gateReport, "--gate-report file");

  if (typeof report.current_round_strike !== "boolean") {
    fail(
      `gate report field current_round_strike is ${JSON.stringify(report.current_round_strike)}, not a boolean — ` +
        "refusing to write a non-boolean strike (a null strike silently resets novel-finding-streak detection). " +
        "A null here means the gate took the B-8 legacy path: the verdict it gated had no `round` key."
    );
  }
  if (typeof report.round !== "number") fail("gate report field round is not a number — is this the report for this verdict?");
  if (verdict.round !== report.round) {
    fail(`gate report round (${report.round}) does not match verdict round (${verdict.round}) — wrong report file`);
  }
  assertGateReportIsCurrent(report);

  const entry = (Array.isArray(verdict.rounds_audit) ? verdict.rounds_audit : []).find((e) => e && e.round === report.round);
  if (!entry) fail(`no rounds_audit entry for round ${report.round} — assemble the draft first`);

  entry.strike = report.current_round_strike;
  fs.writeFileSync(verdictPath, `${JSON.stringify(verdict, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify({ ok: true, verdict: args.verdict, round: report.round, strike: entry.strike }, null, 2)}\n`
  );
}

// ── normalization (delegated, never reimplemented) ───────────────────────
// review-normalize.js takes each positional file as a JSON **ARRAY of finding
// objects** — NOT the verdict envelope. Passing a verdict yields "finding file
// must be a JSON array"; nothing at its call site says so, which is exactly
// the friction this wrapper removes.
function runNormalizer({ findingFiles, ledger, round, cycle, task, priorPath, gateReportPath }) {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "review-verdict-"));
  const ledgerPath = path.join(scratch, "ledger.json");
  fs.writeFileSync(ledgerPath, JSON.stringify(ledger));

  const argv = findingFiles.slice();
  argv.push("--ledger", ledgerPath, "--round", String(round), "--cycle", String(cycle), "--task", task);
  if (priorPath) argv.push("--prior", priorPath);
  if (gateReportPath && fs.existsSync(path.resolve(process.cwd(), gateReportPath))) {
    argv.push("--gate-report", gateReportPath);
  }

  let stdout;
  try {
    stdout = execFileSync(process.execPath, [NORMALIZER, ...argv], { encoding: "utf8", cwd: process.cwd() });
  } catch (e) {
    fail(`scripts/review-normalize.js failed (exit ${e.status}): ${String(e.stderr || e.message).trim()}`);
  } finally {
    fs.rmSync(scratch, { recursive: true, force: true });
  }

  try {
    return JSON.parse(stdout);
  } catch (e) {
    fail(`scripts/review-normalize.js emitted unparseable stdout: ${e.message}`);
  }
}

// Each positional --findings file must already be a JSON array of finding
// objects. Checked here so the failure names the offending file instead of
// surfacing as the normalizer's context-free "finding file must be a JSON array".
function assertFindingFilesAreArrays(files) {
  for (const file of files) {
    const parsed = readJsonFile(file, `--findings file ${file}`);
    if (!Array.isArray(parsed)) {
      fail(`--findings ${file} is a JSON ${parsed === null ? "null" : typeof parsed} — review-normalize.js requires a JSON ARRAY of finding objects, not a verdict envelope`);
    }
  }
}

function runAssemble(args) {
  validateAssembleArgs(args);
  const round = parseCount(args.round, "--round");
  const cycle = parseCount(args.cycle, "--cycle");
  if (round < 1) fail("--round must be >= 1 (round 1 is the first round of a cycle)");
  if (cycle < 1) fail("--cycle must be >= 1");

  const priorPath = args.prior || `briefs/${args.task}-review.json`;
  const priorExists = fs.existsSync(path.resolve(process.cwd(), priorPath));
  const prior = priorExists ? readJsonFile(priorPath, "--prior file") : null;
  const derived = deriveRoundState(prior);

  // Carry-forward is validated BEFORE the normalizer runs so an append-only
  // violation fails fast instead of after a full normalization pass. It also
  // runs BEFORE enforceLifecycle() so that re-emitting an already-journaled
  // round reports the specific append-only refusal rather than the generic
  // round-mismatch one.
  const roundsAudit = carryForwardRoundsAudit(derived.roundsAudit, round).concat([
    buildAuditEntry(args, round, parseSpecialists(args.specialists)),
  ]);
  enforceLifecycle(prior, derived, round, cycle, args);

  assertFindingFilesAreArrays(args.findings);
  const normalized = runNormalizer({
    findingFiles: args.findings,
    ledger: Array.isArray(prior && prior.findings) && !derived.freshCycle ? prior.findings : [],
    round,
    cycle,
    task: args.task,
    priorPath: priorExists ? priorPath : null,
    gateReportPath: args.gateReport || `briefs/${args.task}-gate-report.json`,
  });
  reportNormalizerSideChannels(normalized);
  applyDispositions(normalized);
  assertStrikeableFindings(normalized.findings, round);
  assertGoIsClean(args, normalized.findings);

  const verdict = buildVerdict({
    args,
    round,
    cycle,
    normalized,
    roundsAudit,
    crossRuntime: derived.crossRuntime,
    crossRuntimeFindings: carryForwardCrossRuntime(prior, derived.freshCycle, normalized.cross_runtime_findings),
  });

  const outPath = path.resolve(process.cwd(), args.out || `briefs/${args.task}-review.json.draft`);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(verdict, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify(
      {
        ok: true,
        draft: path.relative(process.cwd(), outPath),
        round,
        cycle,
        findings: verdict.findings.length,
        cross_runtime_findings: verdict.cross_runtime_findings.length,
        strike: "absent — run the gate, then --write-strike",
        next: "node scripts/check-review-convergence.js <draft> --max-rounds <n> --strikes <n>",
      },
      null,
      2
    )}\n`
  );
}

function main(argv) {
  const args = parseArgs(argv);
  if (args.writeStrike) runWriteStrike(args);
  else runAssemble(args);
  process.exit(0);
}

module.exports = { parseArgs, parseSpecialists, carryForwardRoundsAudit, buildAuditEntry, buildVerdict, main };

if (require.main === module) {
  main(process.argv.slice(2));
}
