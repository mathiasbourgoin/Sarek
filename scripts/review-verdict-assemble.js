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
// PIPELINE ORDER (also documented at the top of the schema doc — the order is
// load-bearing: without normalization first, "novel finding" is uncomputable
// because fingerprints are not stable):
//
//   specialists emit findings  →  normalize the ARRAY  →  assemble the verdict
//     →  run the gate  →  write strikes back  →  route on the verdict
//
// This tool owns steps 2, 3 and 5. It delegates step 2 wholesale to
// scripts/review-normalize.js (fingerprinting is never reimplemented here).
//
// TWO PHASES — because `strike` is only knowable AFTER the gate reports it:
//
//   1) assemble:      node scripts/review-verdict-assemble.js --task <slug> --round <n>
//                       --cycle <n> --status <GO|NO-GO> --reviewed-sha <sha>
//                       (--fix-sha <sha> | --fix-sha-reason <text>)
//                       --specialist <name>=<selection_reason> [...]
//                       [--findings <file.json> ...] [--out <path>] ...
//      → writes the draft with NO `strike` key on the current round's entry
//        (absent, never null — a null would be silently accepted here and
//        would silently reset the streak once the round becomes a past round).
//
//   2) write-strike:  node scripts/review-verdict-assemble.js --write-strike
//                       --verdict <draft path> --gate-report <gate stdout json>
//      → copies the gate's boolean `current_round_strike` into the entry for
//        the gate report's own `round`. Refuses anything non-boolean.
//
// Between them, run the gate exactly as roster-review §5.5 prescribes:
//   node scripts/check-review-convergence.js <draft> --max-rounds N --strikes N
//
// Exit codes: 0 success; 2 usage or degraded input (fail-closed, never a
// half-written verdict).
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const { deriveRoundState } = require("./lib/review/review-lifecycle");

const NORMALIZER = path.resolve(__dirname, "review-normalize.js");
const TRACE_SCHEMA_VERSION = "1.0";
const STATUSES = new Set(["GO", "NO-GO"]);

function fail(message) {
  process.stderr.write(`review-verdict-assemble: ${message}\n`);
  process.exit(2);
}

function warn(message) {
  process.stderr.write(`review-verdict-assemble: warning: ${message}\n`);
}

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
    allowRejected: false,
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
    else if (a === "--allow-rejected") out.allowRejected = true;
    else if (VALUE_FLAGS[a]) out[VALUE_FLAGS[a]] = requireValue(argv, ++i, a);
    else fail(`unknown flag or stray argument: ${a}`);
  }
  return out;
}

function requireValue(argv, index, flag) {
  const value = argv[index];
  if (value === undefined || value.startsWith("--")) fail(`${flag} requires a value`);
  return value;
}

function parseCount(raw, flag) {
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0) fail(`${flag} must be a non-negative integer (got ${JSON.stringify(raw)})`);
  return n;
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

// ── phase 2: write the gate-reported strike back ────────────────────────
// `strike` is the one field this tool cannot compute: it is the gate's own
// output. Refusing a non-boolean here is what keeps `strike: null` — which
// makes computeStrikeMap()'s Map.get() return undefined and thus silently
// resets the novel-finding streak — out of every verdict this tool touches.
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

  const entry = (Array.isArray(verdict.rounds_audit) ? verdict.rounds_audit : []).find((e) => e && e.round === report.round);
  if (!entry) fail(`no rounds_audit entry for round ${report.round} — assemble the draft first`);

  entry.strike = report.current_round_strike;
  fs.writeFileSync(verdictPath, `${JSON.stringify(verdict, null, 2)}\n`);
  process.stdout.write(
    `${JSON.stringify({ ok: true, verdict: args.verdict, round: report.round, strike: entry.strike }, null, 2)}\n`
  );
}

// ── phase 1 validation ───────────────────────────────────────────────────
function validateAssembleArgs(args) {
  if (!args.task) fail("--task <slug> is required");
  if (args.round === undefined) {
    fail("--round <n> is required — a verdict without `round` makes check-review-convergence.js skip strike and rounds_audit checks entirely (B-8), so the gate passes vacuously. That refusal is the whole point of this tool.");
  }
  if (args.cycle === undefined) fail("--cycle <n> is required (the trace mechanism scopes lines by (cycle, round))");
  if (!STATUSES.has(args.status)) fail(`--status must be one of ${[...STATUSES].join(" | ")}`);
  if (!args.reviewedSha) fail("--reviewed-sha <sha> is required (rounds_audit completeness, FR-078)");
  if (!args.fixSha === !args.fixShaReason) {
    fail("exactly one of --fix-sha <sha> or --fix-sha-reason <text> is required (EC-8: a dirty tree records null + a reason, never a guessed sha)");
  }
  if (args.specialists.length === 0) {
    fail("at least one --specialist <name>=<selection_reason> is required — \"why did this specialist run this round\" must always be answerable");
  }
  if (args.status === "NO-GO" && !args.noGoType) fail("--no-go-type is required on a NO-GO verdict");
  if (args.status === "GO" && args.noGoType) fail("--no-go-type is meaningless on a GO verdict");
  if (args.status === "GO" && args.noGoRound !== undefined && parseCount(args.noGoRound, "--no-go-round") !== 0) {
    fail("--no-go-round must be 0 (or omitted) on a GO verdict — no_go_round resets on GO");
  }
  if (args.status === "NO-GO" && args.noGoRound === undefined) {
    fail("--no-go-round <n> is required on a NO-GO verdict — it is the round-cap backstop counter and is NOT derivable from `round` (the two are separate counters with separate reset rules)");
  }
}

function parseSpecialists(specs) {
  return specs.map((raw) => {
    const idx = raw.indexOf("=");
    if (idx <= 0) fail(`--specialist must be <name>=<selection_reason> (got ${JSON.stringify(raw)})`);
    const name = raw.slice(0, idx).trim();
    const selection_reason = raw.slice(idx + 1).trim();
    if (!name || !selection_reason) fail(`--specialist ${JSON.stringify(raw)} has an empty name or selection_reason`);
    return { name, selection_reason };
  });
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
      fail(`--findings ${file} is a JSON ${Array.isArray(parsed) ? "array" : typeof parsed} — review-normalize.js requires a JSON ARRAY of finding objects, not a verdict envelope`);
    }
  }
}

// ── rounds_audit carry-forward (append-only) ─────────────────────────────
// Prior entries are copied through verbatim; a prior entry for the round being
// assembled is a refusal, not an overwrite — that would mean the round counter
// never advanced (scripts/lib/review/review-lifecycle.js owns the bump).
function carryForwardRoundsAudit(priorEntries, round) {
  const carried = [];
  for (const entry of priorEntries) {
    if (!entry || typeof entry !== "object") fail("prior rounds_audit contains a non-object entry — refusing to carry forward unverifiable state");
    if (entry.round === round) {
      fail(`prior verdict already has a rounds_audit entry for round ${round} — rounds_audit is append-only. Bump the round via \`node scripts/lib/review/review-lifecycle.js --prior <verdict>\` instead of rewriting it.`);
    }
    if (typeof entry.round === "number" && entry.round > round) {
      fail(`prior verdict has a rounds_audit entry for round ${entry.round}, which is ahead of --round ${round} — refusing to assemble a verdict that would look like it went backwards`);
    }
    carried.push(entry);
  }
  return carried;
}

function buildAuditEntry(args, round, specialists) {
  const entry = { round, reviewed_sha: args.reviewedSha };
  entry.fix_sha = args.fixSha ? args.fixSha : null;
  if (!args.fixSha) entry.fix_sha_reason = args.fixShaReason;
  entry.specialists_run = specialists;
  if (args.traceSchemaVersion !== "none") entry.trace_schema_version = args.traceSchemaVersion;
  // NOTE: `strike` is deliberately ABSENT, not null. It is written by
  // --write-strike after the gate reports current_round_strike.
  return entry;
}

function buildNoGoReason(args) {
  if (args.status === "GO") return null;
  const reason = { type: args.noGoType };
  if (args.noGoCause) reason.cause = args.noGoCause;
  if (args.failedAcs.length > 0) reason.failed_acs = args.failedAcs;
  return reason;
}

function buildVerdict({ args, round, cycle, normalized, roundsAudit, crossRuntime }) {
  const verdict = {
    task: args.task,
    date: new Date().toISOString(),
    status: args.status,
    mode: args.mode,
    round,
    cycle,
    no_go_round: args.noGoRound === undefined ? 0 : parseCount(args.noGoRound, "--no-go-round"),
    auto_fixes_applied: parseCount(args.autoFixes, "--auto-fixes"),
    findings: normalized.findings,
    cross_runtime_findings: normalized.cross_runtime_findings,
    cross_runtime: crossRuntime,
    rounds_audit: roundsAudit,
    no_go_reason: buildNoGoReason(args),
    summary: args.summary || "",
    escalation_needed: args.escalationNeeded,
    escalation_reason: args.escalationReason || null,
    normalized_by: normalized.normalizer_version,
  };
  if (args.streakOverrideBy) verdict.streak_override = { round, by: args.streakOverrideBy };
  return verdict;
}

// Surfaces the normalizer's non-verdict outputs (they have no home in
// review.json and must not vanish silently, FR-100).
function reportNormalizerSideChannels(args, normalized) {
  for (const w of normalized.warnings || []) warn(`review-normalize: ${w}`);
  for (const d of normalized.probable_duplicates || []) {
    warn(`review-normalize: probable duplicate needs owner adjudication: ${JSON.stringify(d)}`);
  }
  const rejected = normalized.rejected || [];
  if (rejected.length === 0) return;
  for (const r of rejected) warn(`review-normalize: REJECTED (schema-invalid) finding: ${r.reason}`);
  if (!args.allowRejected) {
    fail(`${rejected.length} finding(s) were rejected as schema-invalid by review-normalize.js — fix them against schema/review-finding.schema.json, or pass --allow-rejected to drop them deliberately`);
  }
}

// Cross-checks --round/--cycle against the lifecycle witness rather than
// re-deriving the rule here. Advisory (a warning), matching review-normalize.js's
// own treatment of the same disagreement.
function checkLifecycle(prior, round, cycle) {
  const derived = deriveRoundState(prior);
  if (derived.round !== null && derived.round !== round) {
    warn(`--round ${round} but scripts/lib/review/review-lifecycle.js derives ${derived.round} from the prior verdict`);
  }
  if (derived.cycle !== null && derived.cycle !== cycle) {
    warn(`--cycle ${cycle} but scripts/lib/review/review-lifecycle.js derives ${derived.cycle} from the prior verdict`);
  }
  return derived;
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
  const derived = checkLifecycle(prior, round, cycle);

  // Carry-forward is validated BEFORE the normalizer runs so an append-only
  // violation fails fast instead of after a full normalization pass.
  const roundsAudit = carryForwardRoundsAudit(derived.roundsAudit, round).concat([
    buildAuditEntry(args, round, parseSpecialists(args.specialists)),
  ]);

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
  reportNormalizerSideChannels(args, normalized);

  const verdict = buildVerdict({
    args,
    round,
    cycle,
    normalized,
    roundsAudit,
    crossRuntime: derived.crossRuntime,
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
