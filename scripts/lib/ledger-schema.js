// scripts/lib/ledger-schema.js — CommonJS, zero dependencies.
//
// The pipeline durable-state ledger (briefs/<task>-state.json) as an
// executable schema.
//
// Backlog #103. Upstream roster encodes these rules as a jq predicate embedded
// in roster-run.md prose. That predicate lives in an untracked skill file, so
// in a fresh clone the ledger has no schema at all and a malformed ledger
// surfaces as a mysterious resume bug several phases later. This is the same
// rules, tracked, runnable without jq, and cross-checkable against the jq
// predicate when the skill file is present (see roster-implement-posthook.sh).
"use strict";

const SEQ = {
  express: ["implement", "review", "ship"],
  fast: ["implement", "review", "qa", "ship"],
  full: ["question", "research", "intake", "spec", "plan", "implement", "review", "qa", "ship"],
};

const VOCAB = {
  question: ["COMPLETED"],
  research: ["COMPLETED"],
  intake: ["VALIDATED"],
  spec: ["VALIDATED", "SKIPPED", "BOUNCED"],
  plan: ["COMPLETED"],
  implement: ["COMPLETED", "PARTIAL"],
  review: ["GO", "NO-GO"],
  qa: ["GO", "NO-GO"],
  ship: ["COMPLETED", "BLOCKED"],
};

// WHAT THIS DELIBERATELY DOES NOT CHECK: phase ORDER.
//
// The obvious next rule is "a phase may not appear before its predecessors in
// SEQ". Measured against the 34 real ledgers in briefs/, a rule of that shape
// flags 7 of them — `research > spec > plan` (no intake), `intake > plan` (no
// spec), and so on. Full-mode runs legitimately skip optional phases, and the
// ledger records nothing that distinguishes a phase deliberately skipped from
// one erroneously missing. A rule that fires on a fifth of valid input is not
// a gate; it is something people learn to ignore.
//
// The specific failure this was meant to catch — a phase that did not run,
// with the stale ledger handed to the next phase anyway — is caught precisely
// instead, by the caller asserting which phase must be the LAST event
// (roster-implement-posthook.sh --expect-phase). That has no false positives
// by construction: the hook runs only after that phase.

// Returns { valid, errors[] }. Every rule reports its own message: "the ledger
// is invalid" is useless to whoever has to repair it by hand.
function validateLedger(ledger, task) {
  const errors = [];
  const bad = (m) => errors.push(m);

  if (ledger === null || typeof ledger !== "object" || Array.isArray(ledger)) {
    return { valid: false, errors: ["ledger is not a JSON object"] };
  }

  if (typeof task === "string" && ledger.task !== task) {
    bad(`task mismatch: ledger.task is ${JSON.stringify(ledger.task)}, expected ${JSON.stringify(task)}`);
  }

  const seq = SEQ[ledger.mode];
  if (!seq) {
    bad(`mode must be one of ${Object.keys(SEQ).join("/")}, got ${JSON.stringify(ledger.mode)}`);
  }

  if (typeof ledger.current_phase !== "string") {
    bad(`current_phase must be a string, got ${JSON.stringify(ledger.current_phase)}`);
  }

  if (!Array.isArray(ledger.events) || ledger.events.length === 0) {
    bad("events must be a non-empty array");
    return { valid: errors.length === 0, errors };
  }

  ledger.events.forEach((e, i) => {
    const at = `events[${i}]`;
    if (e === null || typeof e !== "object" || Array.isArray(e)) {
      bad(`${at} is not an object`);
      return;
    }
    if (typeof e.phase !== "string") {
      bad(`${at}.phase must be a string, got ${JSON.stringify(e.phase)}`);
      return;
    }
    const allowed = VOCAB[e.phase];
    if (!allowed) {
      bad(`${at}.phase is not a known phase: ${JSON.stringify(e.phase)}`);
    } else if (!allowed.includes(e.outcome)) {
      bad(`${at}: outcome ${JSON.stringify(e.outcome)} is not legal for phase ${e.phase} (legal: ${allowed.join("/")})`);
    }
    // `reason` is optional, but when present it must be a string — a boolean
    // `reason: false` is the historical defect this rule exists for.
    if (Object.prototype.hasOwnProperty.call(e, "reason") && typeof e.reason !== "string") {
      bad(`${at}.reason must be a string when present, got ${JSON.stringify(e.reason)}`);
    }
  });

  const last = ledger.events[ledger.events.length - 1];
  if (last && typeof last === "object" && last.phase !== ledger.current_phase) {
    bad(`current_phase (${JSON.stringify(ledger.current_phase)}) does not match the last event's phase (${JSON.stringify(last && last.phase)})`);
  }

  if (seq && typeof ledger.current_phase === "string" && !seq.includes(ledger.current_phase)) {
    bad(`current_phase ${JSON.stringify(ledger.current_phase)} is not part of the ${ledger.mode} sequence (${seq.join(" > ")})`);
  }

  return { valid: errors.length === 0, errors };
}

// Asserts the ledger's LAST event is for `phase`. The failure this exists for:
// /roster-implement dies before appending its event, and the post-hook then
// validates the stale pre-implement ledger, prints OK, and hands off to review
// — the "inexplicable skipped phase, discovered several phases later" that the
// hook is supposed to prevent, produced by the hook itself.
function expectLatestPhase(ledger, phase) {
  const errors = [];
  if (!ledger || !Array.isArray(ledger.events) || ledger.events.length === 0) {
    return { valid: false, errors: ["ledger has no events, so no phase can have completed"] };
  }
  const last = ledger.events[ledger.events.length - 1];
  const seen = ledger.events.filter((e) => e && e.phase === phase).length;

  if (!last || last.phase !== phase) {
    errors.push(
      `the latest ledger event is ${JSON.stringify(last && last.phase)}, expected ${JSON.stringify(phase)}` +
        (seen === 0
          ? ` — there is no ${phase} event anywhere in this ledger, so ${phase} did not run (or died before recording it). This is a STALE ledger; do not hand it to the next phase.`
          : ` — the most recent ${phase} event is not the latest event, so this ledger has moved on since.`)
    );
  }
  if (typeof ledger.current_phase === "string" && ledger.current_phase !== phase) {
    errors.push(`current_phase is ${JSON.stringify(ledger.current_phase)}, expected ${JSON.stringify(phase)}`);
  }
  return { valid: errors.length === 0, errors };
}

module.exports = { SEQ, VOCAB, validateLedger, expectLatestPhase };
