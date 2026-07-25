#!/usr/bin/env node
// scripts/lib/review-lifecycle.js — CommonJS, pure functions + a thin CLI.
//
// The two-event round/cycle lifecycle for roster-review (specs/review-v2-
// corrections.md INV-3, Amendments E-5/E-6). This module is the executable
// witness for the prose contract in skills/pipeline/roster-review.md §5.5 —
// literally, not just by reference: roster-review shells out to the CLI
// below at draft composition instead of re-deriving the rule in prose, and
// scripts/review-normalize.js requires this module directly to cross-check
// a caller-supplied `--round` (review finding FIX-1: this module was
// previously unwired — only its own test imported it).
//
//   1) a persisted GO verdict retains its cycle-final state (round,
//      rounds_audit, cross_runtime) — it is never reset in place, so it stays
//      auditable.
//   2) the NEXT review cycle then initializes fresh: round 1, full fan-out,
//      a fresh rounds_audit, a new cross-runtime probe, and cycle + 1.
//
// No I/O in deriveRoundState() itself — only the CLI below reads a file.
//
// CLI usage: node scripts/lib/review-lifecycle.js --prior <review.json path>
//   Prints `{round, cycle, fresh_cycle}` (snake_case — the review.json
//   envelope's own field naming) to stdout. Absent path (no prior file,
//   fresh task) is legitimate input, not an error, and derives the fresh-
//   cycle state exactly like an in-process `deriveRoundState(null)` call.
//   Exit 0 on success; 2 on a --prior path that exists but is not valid JSON
//   (unverifiable prior state must not silently print a guessed answer).
"use strict";

const fs = require("fs");

function isFreshCycle(prior) {
  return !prior || prior.status === "GO";
}

// Derives this draft's round/cycle/carry-forward state from the prior
// persisted verdict (or its absence). Pure — never mutates `prior`.
//
//   - absent prior, or prior.status === "GO"  -> fresh cycle: round 1,
//     cycle = prior.cycle + 1 (or 1 if no prior/no prior cycle), empty
//     rounds_audit, empty cross_runtime.
//   - prior.status === "NO-GO", numeric prior.round -> round = prior.round + 1,
//     cycle carried forward unchanged, rounds_audit/cross_runtime carried
//     forward.
//   - prior.status === "NO-GO", round absent (legacy) -> stay legacy for this
//     cycle (round: null, legacyRound: true) — carries rounds_audit/
//     cross_runtime forward as-is.
function deriveRoundState(prior) {
  if (isFreshCycle(prior)) {
    const priorCycle = prior && typeof prior.cycle === "number" ? prior.cycle : 0;
    return {
      round: 1,
      cycle: priorCycle + 1,
      roundsAudit: [],
      crossRuntime: {},
      legacyRound: false,
      freshCycle: true,
    };
  }

  const roundsAudit = Array.isArray(prior.rounds_audit) ? prior.rounds_audit : [];
  const crossRuntime = prior.cross_runtime && typeof prior.cross_runtime === "object" ? prior.cross_runtime : {};
  const cycle = typeof prior.cycle === "number" ? prior.cycle : null;

  if (typeof prior.round !== "number") {
    return { round: null, cycle, roundsAudit, crossRuntime, legacyRound: true, freshCycle: false };
  }

  return { round: prior.round + 1, cycle, roundsAudit, crossRuntime, legacyRound: false, freshCycle: false };
}

// ── CLI ──────────────────────────────────────────────────────────────────

function fail(code, message) {
  process.stderr.write(`review-lifecycle: ${message}\n`);
  process.exit(code);
}

function parseArgs(argv) {
  const out = { priorPath: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--prior") out.priorPath = argv[++i];
  }
  return out;
}

// Reads --prior <path> if given. Absent path/flag = legitimately no prior
// verdict (fresh task). A path that exists but fails to parse is unverifiable
// state — fail closed (exit 2) rather than silently deriving a fresh cycle
// that might be wrong.
function readPrior(priorPath) {
  if (!priorPath) return null;
  if (!fs.existsSync(priorPath)) return null; // no prior file yet — legitimate fresh-task input
  try {
    return JSON.parse(fs.readFileSync(priorPath, "utf8"));
  } catch (e) {
    fail(2, `--prior file exists but is not valid JSON: ${priorPath}`);
  }
}

function main(argv) {
  const args = parseArgs(argv);
  const prior = readPrior(args.priorPath);
  const state = deriveRoundState(prior);
  process.stdout.write(JSON.stringify({ round: state.round, cycle: state.cycle, fresh_cycle: state.freshCycle }) + "\n");
  process.exit(0);
}

module.exports = { isFreshCycle, deriveRoundState, main };

if (require.main === module) {
  main(process.argv.slice(2));
}
