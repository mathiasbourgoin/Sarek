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

module.exports = { SEQ, VOCAB, validateLedger };
