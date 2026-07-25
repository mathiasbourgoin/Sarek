#!/usr/bin/env node
// scripts/check-mandated-steps.js — CommonJS, repo-local tool.
//
// Records and verifies the mandated-step ledger, `briefs/<task>-steps.jsonl`.
// Rules live in scripts/lib/mandated-steps-rules.js; this file is CLI, I/O and
// orchestration only.
//
// WHY: a mandated step that is skipped without a record is indistinguishable
// from one that ran and passed. Preflight, human gates and degraded specialists
// could all be skipped silently. Making the skip *recordable* is half of it;
// making the ABSENCE of a required record a failure is the half that has teeth.
//
// TWO MODES
//
//   record — append one schema-valid record (never hand-write the JSONL):
//     node scripts/check-mandated-steps.js --record --task <slug> \
//       --step preflight --outcome ran --result READY --actor agent
//     node scripts/check-mandated-steps.js --record --task <slug> \
//       --step human-gate --outcome skipped --actor agent \
//       --reason "standing autonomy delegation, session 2026-07-25"
//
//   check — verify the ledger against what the phase mandates:
//     node scripts/check-mandated-steps.js --task <slug> --phase implement
//
// Exit contract:
//   0 = every mandated step has a valid record
//   1 = a mandated step has no record, or a record is invalid — the gate's "no"
//   2 = usage error / unreadable or malformed ledger (fail closed)
//
// An absent ledger is exit 1 with every requirement listed, never exit 0: "no
// file" is the strongest possible form of "nothing was recorded".
"use strict";

const fs = require("fs");
const path = require("path");
const { evaluate, STEPS, OUTCOMES, ACTORS, RESULTS } = require("./lib/mandated-steps-rules");

const VALUE_FLAGS = {
  "--task": "task",
  "--phase": "phase",
  "--ledger": "ledger",
  "--step": "step",
  "--outcome": "outcome",
  "--result": "result",
  "--actor": "actor",
  "--reason": "reason",
  "--require": "require",
};

function usage(message) {
  process.stderr.write(`check-mandated-steps: ${message}\n`);
  process.exit(2);
}

function parseArgs(argv) {
  const args = { record: false };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === "--record") {
      args.record = true;
    } else if (Object.prototype.hasOwnProperty.call(VALUE_FLAGS, flag)) {
      const value = argv[++i];
      if (value === undefined || value.startsWith("--")) usage(`${flag} requires a value`);
      args[VALUE_FLAGS[flag]] = value;
    } else {
      usage(`unknown flag or stray argument: ${flag}`);
    }
  }
  return args;
}

function ledgerPath(args) {
  if (args.ledger) return path.resolve(process.cwd(), args.ledger);
  if (!args.task) usage("--task <slug> is required (or pass --ledger <path>)");
  return path.resolve(process.cwd(), "briefs", `${args.task}-steps.jsonl`);
}

// A malformed line is exit 2, never a skipped line. Silently dropping an
// unparseable record would reintroduce the defect on the checker's own side.
function readLedger(file) {
  if (!fs.existsSync(file)) return [];
  let raw;
  try {
    raw = fs.readFileSync(file, "utf8");
  } catch (e) {
    usage(`cannot read ledger ${file}: ${e.message}`);
  }
  const records = [];
  raw.split("\n").forEach((line, index) => {
    if (line.trim() === "") return;
    try {
      records.push(JSON.parse(line));
    } catch (e) {
      usage(`ledger line ${index + 1} is not valid JSON: ${e.message}`);
    }
  });
  return records;
}

function runRecord(args) {
  if (!args.task) usage("--record requires --task <slug>");
  if (!args.step) usage("--record requires --step <id>");
  if (!Object.prototype.hasOwnProperty.call(STEPS, args.step)) {
    usage(`--step ${JSON.stringify(args.step)} is not a known step — known: ${Object.keys(STEPS).join(", ")}`);
  }
  if (!OUTCOMES.has(args.outcome)) usage(`--outcome must be one of ${[...OUTCOMES].join(" | ")}`);
  if (!ACTORS.has(args.actor)) usage(`--actor must be one of ${[...ACTORS].join(" | ")}`);
  if (args.outcome === "skipped" && !(args.reason && args.reason.trim() !== "")) {
    usage("--outcome skipped requires --reason <text> — an unreasoned skip is exactly what this ledger exists to prevent");
  }
  if (args.outcome === "ran" && !RESULTS.has(args.result)) {
    usage(`--outcome ran requires --result <${[...RESULTS].join("|")}>`);
  }

  const record = {
    ts: new Date().toISOString(),
    task: args.task,
    step: args.step,
    outcome: args.outcome,
    actor: args.actor,
  };
  if (args.outcome === "ran") record.result = args.result;
  if (args.reason) record.reason = args.reason;
  if (args.phase) record.phase = args.phase;

  const file = ledgerPath(args);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, `${JSON.stringify(record)}\n`);
  process.stdout.write(`${JSON.stringify({ ok: true, ledger: path.relative(process.cwd(), file), record }, null, 2)}\n`);
  process.exit(0);
}

function runCheck(args) {
  if (!args.phase && !args.require) usage("--phase <name> is required (or --require <step,step>)");
  const file = ledgerPath(args);
  const records = readLedger(file);
  const result = evaluate({
    records,
    phase: args.phase,
    require: args.require ? args.require.split(",").map((s) => s.trim()).filter(Boolean) : null,
    task: args.task,
  });
  if (result.usageError) usage(result.usageError);

  const report = {
    ledger: path.relative(process.cwd(), file),
    ledger_exists: fs.existsSync(file),
    phase: args.phase || null,
    required: result.required,
    records: records.length,
    missing: result.missing,
    invalid: result.invalid,
    ok: result.ok,
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!result.ok) {
    for (const message of result.missing.concat(result.invalid)) {
      process.stderr.write(`check-mandated-steps: ${message}\n`);
    }
    process.exit(1);
  }
  process.exit(0);
}

function main(argv) {
  const args = parseArgs(argv);
  if (args.record) runRecord(args);
  else runCheck(args);
}

module.exports = { parseArgs, readLedger, main };

if (require.main === module) {
  main(process.argv.slice(2));
}
