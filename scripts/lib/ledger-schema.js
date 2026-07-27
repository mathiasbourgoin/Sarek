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

// SKIPPED is legal for EVERY phase (backlog-135), not just `spec`.
//
// Before this, `spec` was the only phase with a skip vocabulary, and even that
// was mostly written into briefs/<task>-spec.md rather than the ledger — no
// -state.json in the 48-ledger corpus contained a SKIPPED event at all. Every
// other phase could only be skipped by not appearing, which is byte-identical
// to a phase that was supposed to run and did not.
//
// The rule is the one the mandated-step ledger already applies one level down
// (scripts/lib/mandated-steps-rules.js, schema/mandated-steps-schema.md:29 —
// "Skipping is allowed. Skipping silently is not."): skipping is permitted,
// skipping without saying so is not. A SKIPPED event therefore carries a
// `reason` and a `by`, and both are REQUIRED — see requiredSkipFields below.
// Whether a reason is a *good* reason stays a human judgement; what is
// enforced is that one exists and is attributable.
//
// KNOWN CONSEQUENCE for the jq cross-check. roster-implement-posthook.sh
// compares this schema's verdict against the LEDGER_SCHEMA jq predicate in
// .harness/skills/roster-run.md, which is untracked and carries the OLD
// vocabulary. The first ledger to record a non-spec skip will make the two
// disagree, and the hook will correctly report SCHEMA DRIFT. That is the
// detector working, not a regression: the predicate in the skill file needs
// the same SKIPPED vocabulary. It cannot be changed here because it lives
// outside the repository.
const SKIPPED = "SKIPPED";

const VOCAB = {
  question: ["COMPLETED", SKIPPED],
  research: ["COMPLETED", SKIPPED],
  intake: ["VALIDATED", SKIPPED],
  spec: ["VALIDATED", SKIPPED, "BOUNCED"],
  plan: ["COMPLETED", SKIPPED],
  implement: ["COMPLETED", "PARTIAL", SKIPPED],
  review: ["GO", "NO-GO", SKIPPED],
  qa: ["GO", "NO-GO", SKIPPED],
  ship: ["COMPLETED", "BLOCKED", SKIPPED],
};

// PHASE ORDER: still not enforced here, but no longer unenforceable.
//
// The note this replaces read: "the ledger records nothing that distinguishes
// a phase deliberately skipped from one erroneously missing. A rule that fires
// on a fifth of valid input is not a gate; it is something people learn to
// ignore." Measured against the 34 ledgers of the day, a naive order rule
// flagged 7 — `research > spec > plan` (no intake), `intake > plan` (no spec).
//
// That was a statement about the SCHEMA, not about the rule. With a recordable
// skip, the seven ledgers that a phase-order gate would have libelled can say
// what they did, and the distinction the gate needs becomes expressible:
//
//     absent        -> the phase did not run and nobody said it wouldn't
//     SKIPPED       -> the phase did not run, on purpose, for a stated reason,
//                      decided by a named actor
//     any other     -> the phase ran
//
// `phaseGaps` below computes exactly that distinction. It is deliberately a
// REPORTING function and not a gate: this change makes enforcement possible,
// and the corpus has to be migrated before enforcement is honest. Turning it
// into a gate today would fail the 48 existing ledgers, none of which record
// a skip, for a defect they predate. See the header of phaseGaps for what an
// enforcer built on it would then be able to check.
//
// The specific failure the missing order-check was meant to catch — a phase
// that did not run, with the stale ledger handed to the next phase anyway —
// remains caught precisely by the caller asserting which phase must be the
// LAST event (roster-implement-posthook.sh --expect-phase). That has no false
// positives by construction: the hook runs only after that phase.

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
    // ...except on a SKIPPED event, where it is the entire point (backlog-135).
    //
    // A skip record with no reason and no actor records strictly less than the
    // omission it replaces: it says the phase did not run, which was already
    // visible, and drops the "on purpose, and here is who decided" that is the
    // only reason to write it down. Blank and whitespace-only are rejected for
    // the same reason a missing field is — `reason: ""` is not a reason.
    if (e.outcome === SKIPPED) {
      for (const field of ["reason", "by"]) {
        const v = e[field];
        if (typeof v !== "string" || v.trim() === "") {
          bad(
            `${at}: a SKIPPED ${e.phase} event must record a non-empty ${field} — ` +
              `got ${JSON.stringify(v)}. Skipping is allowed; skipping silently is not, ` +
              `so a skip states why (reason) and who decided (by).`
          );
        }
      }
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

// Classifies every phase the ledger's mode requires up to and including
// current_phase (backlog-135). REPORTING ONLY — nothing calls this as a gate.
//
// Returns { ran[], skipped[], missing[], unknownMode }:
//
//   ran      — a non-SKIPPED event exists for the phase
//   skipped  — a SKIPPED event exists (and, since validateLedger passed, it
//              carries a reason and a `by`)
//   missing  — neither. Before backlog-135 this bucket and `skipped` were the
//              same bucket, which is why phase-order enforcement was blocked.
//
// WHAT AN ENFORCER BUILT ON THIS COULD NOW CHECK, and could not before:
//
//   1. "No phase of the declared mode is unaccounted for." `missing` being
//      non-empty is now a real finding rather than an ambiguity: a full-mode
//      run that reaches `ship` with no `spec` event either skipped spec on
//      purpose (and can say so) or lost it. Previously indistinguishable.
//   2. "A skip is attributable." Already enforced by validateLedger, but an
//      order gate can additionally require that skips of gate-like phases
//      (review, qa) name a human `by`, the way mandated-steps-rules.js treats
//      humanOnly steps.
//   3. "Phases did not run out of order." The original rule — a phase may not
//      appear before its SEQ predecessors — with `skipped` predecessors
//      counted as satisfied. That is the rule measured at 7 false positives in
//      34 ledgers; those 7 are exactly the shape a skip record now describes.
//   4. "current_phase is reachable." current_phase can be required to be the
//      first phase after the last accounted-for one, catching a ledger that
//      jumped.
//
// MEASURED, the way this repo requires a proposed rule to be measured before
// it ships (schema/mandated-steps-schema.md:117-119). Against the 48 real
// ledgers in briefs/ at the time of writing:
//
//   * validateLedger's verdict changed on ZERO of them. The vocabulary change
//     is strictly additive and no existing ledger becomes invalid. (15 were
//     already invalid before this change and are still invalid after, for
//     unrelated pre-existing reasons.)
//   * phaseGaps reports unaccounted-for phases in 9 of 48 — question 9,
//     intake 6, research 4, spec 3, plan 1. That is the same order as the
//     "7 of 34" that blocked the order rule originally, and it is why the
//     enforcer is still not wired: those 9 predate skip records and would be
//     failed for a defect they could not have avoided.
//
// The migration this waits on is therefore a policy call — grandfather by
// date, or enforce only on ledgers created after this schema — not a schema
// one, which is why it is not made here.
function phaseGaps(ledger) {
  const seq = SEQ[ledger && ledger.mode];
  if (!seq) return { ran: [], skipped: [], missing: [], unknownMode: true };

  const events = Array.isArray(ledger.events) ? ledger.events : [];
  // Only phases up to current_phase are due. A full-mode run sitting at
  // `plan` has not failed to `ship`; it has not got there yet.
  const upto = seq.indexOf(ledger.current_phase);
  const due = upto === -1 ? seq : seq.slice(0, upto + 1);

  const ran = [];
  const skipped = [];
  const missing = [];
  for (const phase of due) {
    const forPhase = events.filter((e) => e && e.phase === phase);
    if (forPhase.some((e) => e.outcome !== SKIPPED)) ran.push(phase);
    else if (forPhase.length > 0) skipped.push(phase);
    else missing.push(phase);
  }
  return { ran, skipped, missing, unknownMode: false };
}

module.exports = { SEQ, VOCAB, SKIPPED, validateLedger, expectLatestPhase, phaseGaps };
