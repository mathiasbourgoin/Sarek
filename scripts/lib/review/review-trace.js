#!/usr/bin/env node
// scripts/lib/review/review-trace.js — CommonJS.
//
// Producer-side append for briefs/<task>-review-trace.jsonl (spec:
// specs/r5-trace-enforcement.md FR-160..166, US-2). Replicates the
// append-forever JSONL mechanics of scripts/lib/xruntime/xruntime-journal.js's
// appendJournalLine (FR-166: "reusing or extracting") as a SIBLING module —
// deliberately NOT editing xruntime-journal.js itself, so
// readLatestJournalEntry/shouldRefuseDegraded stay byte-for-byte what they
// read today (INV-4, zero risk). The trace file is a distinct artifact from
// the xruntime journal (never the same file, never merged — see the spec's
// "which file carries the trace" clarification).
//
// Usage (thin CLI, for deterministic append from skill prose):
//   node scripts/lib/review/review-trace.js --task <slug> --round <n>
//     --cycle <n> --event <scope-gate|normalizer|specialist|cross-runtime>
//     --actor <name> --outcome <ran|skipped> [--detail <string>] [--digest <string>]
//     [--root <dir>]
// Prints `{"ok":true}` or `{"ok":false,"error":"..."}` on stdout; exit 0 on a
// successful append, exit 2 on a usage/append error (never silently healthy,
// FR-096 pattern carried over from the journal helper).
"use strict";

const fs = require("fs");
const path = require("path");

const SCHEMA_VERSION = "1.0";

function tracePath(root, task) {
  return path.resolve(root, "briefs", `${task}-review-trace.jsonl`);
}

// Append-before-report: appends exactly one JSON line, mkdir-ing briefs/ if
// needed. Append failure is reported to the caller (never silently healthy,
// FR-096 mirror) — the caller decides how to surface it (a warning, never a
// change in exit code, per FR-166).
function appendTraceLine(root, task, entry) {
  const line = JSON.stringify(entry);
  try {
    fs.mkdirSync(path.resolve(root, "briefs"), { recursive: true });
    fs.appendFileSync(tracePath(root, task), line + "\n");
  } catch (e) {
    return { ok: false, error: e.message, line };
  }
  return { ok: true, line };
}

function parseArgs(argv) {
  const out = { root: process.cwd(), unknownFlag: null, missing: [] };
  const FLAGS = { "--task": "task", "--round": "round", "--cycle": "cycle", "--event": "event", "--actor": "actor", "--outcome": "outcome", "--detail": "detail", "--digest": "digest", "--root": "root" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const key = FLAGS[a];
    if (!key) {
      out.unknownFlag = a;
      continue;
    }
    out[key] = argv[++i];
  }
  return out;
}

function validateRequired(args) {
  for (const key of ["task", "round", "cycle", "event", "actor", "outcome"]) {
    if (args[key] === undefined) args.missing.push(key);
  }
  return args.missing;
}

function fail(message) {
  process.stdout.write(JSON.stringify({ ok: false, error: message }) + "\n");
  process.exit(2);
}

function main(argv) {
  const args = parseArgs(argv);
  if (args.unknownFlag) fail(`unknown flag: ${args.unknownFlag}`);
  const missing = validateRequired(args);
  if (missing.length) fail(`missing required flag(s): ${missing.map((k) => `--${k}`).join(", ")}`);

  const round = parseInt(args.round, 10);
  const cycle = parseInt(args.cycle, 10);
  if (Number.isNaN(round) || Number.isNaN(cycle)) fail("--round and --cycle must be integers");

  const entry = {
    schema_version: SCHEMA_VERSION,
    ts: new Date().toISOString(),
    task: args.task,
    round,
    cycle,
    event: args.event,
    actor: args.actor,
    outcome: args.outcome,
  };
  if (args.detail !== undefined) entry.detail = args.detail;
  if (args.digest !== undefined) entry.digest = args.digest;

  const result = appendTraceLine(args.root, args.task, entry);
  if (!result.ok) fail(result.error);
  process.stdout.write(JSON.stringify({ ok: true }) + "\n");
  process.exit(0);
}

module.exports = { tracePath, appendTraceLine, SCHEMA_VERSION };

if (require.main === module) {
  main(process.argv.slice(2));
}
