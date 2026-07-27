#!/usr/bin/env node
// SPDX-License-Identifier: CECILL-B
// SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
//
// Covering test for scripts/lib/ledger-schema.js.
//
// The schema had no test of its own — it was exercised only incidentally, via
// roster-implement-posthook.test.sh, which cares about the hook rather than
// the rules. backlog-135 adds a rule whose entire value is that it REJECTS
// something, so it gets a test that watches it reject.
//
// Every case here is a proof of rejection or a paired positive control. A
// validator that accepts everything and a validator that is correct are
// indistinguishable from a green run, which is the failure mode this
// repository keeps finding.
"use strict";

const {
  SEQ,
  VOCAB,
  SKIPPED,
  validateLedger,
  expectLatestPhase,
  phaseGaps,
} = require("./lib/ledger-schema.js");

let pass = 0;
let fail = 0;

function check(desc, got, want) {
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  if (g === w) {
    console.log(`  PASS: ${desc}`);
    pass += 1;
  } else {
    console.log(`  FAIL: ${desc} -- expected ${w}, got ${g}`);
    fail += 1;
  }
}

// Asserts invalid AND that some error mentions `needle`. "It went red" is not
// enough: a rule that rejects for the wrong reason still rejects, and would
// keep passing this test after the rule it is named for was deleted.
function checkRejects(desc, ledger, needle) {
  const r = validateLedger(ledger, ledger.task);
  const hit = r.errors.some((e) => e.includes(needle));
  check(`${desc} (rejected)`, r.valid, false);
  check(`${desc} (for the right reason: ${JSON.stringify(needle)})`, hit, true);
}

console.log("ledger-schema.js covering test");

// A minimal well-formed ledger, used as the base for mutation below.
const base = () => ({
  task: "demo",
  mode: "full",
  current_phase: "implement",
  events: [
    { phase: "intake", outcome: "VALIDATED", by: "roster-intake" },
    { phase: "plan", outcome: "COMPLETED", by: "roster-plan" },
    { phase: "implement", outcome: "COMPLETED", by: "roster-implement" },
  ],
});

// ── Positive control ────────────────────────────────────────────────────────
// Without this, every rejection below could be a validator that says no to
// everything.
check("a well-formed ledger validates", validateLedger(base(), "demo").valid, true);

// ── backlog-135: a skip is recordable, and only when it says what and who ───

// The load-bearing case. Before this change `intake` had no SKIPPED in its
// vocabulary, so the only way to express "we deliberately did not do intake"
// was to omit the event — indistinguishable from losing it.
const withSkip = base();
withSkip.events[0] = {
  phase: "intake",
  outcome: SKIPPED,
  by: "mathias",
  reason: "hygiene task, scope fixed by the backlog item; no intake needed",
};
check(
  "a skip WITH a reason and a `by` validates",
  validateLedger(withSkip, "demo").valid,
  true
);

// Every phase, not just spec. The old vocabulary allowed SKIPPED on `spec`
// alone, which is why the corpus expresses skips by omission everywhere else.
for (const phase of Object.keys(VOCAB)) {
  check(
    `SKIPPED is legal for phase ${phase}`,
    VOCAB[phase].includes(SKIPPED),
    true
  );
}

// A skip that does not say why records less than the omission it replaces.
const noReason = base();
noReason.events[0] = { phase: "intake", outcome: SKIPPED, by: "mathias" };
checkRejects("a skip with no reason", noReason, "non-empty reason");

const noBy = base();
noBy.events[0] = { phase: "intake", outcome: SKIPPED, reason: "not needed" };
checkRejects("a skip with no `by`", noBy, "non-empty by");

// Blank and whitespace-only are the obvious way to satisfy a presence check
// while recording nothing, so they are rejected like a missing field.
for (const [label, value] of [["empty", ""], ["whitespace-only", "   \t "]]) {
  const blank = base();
  blank.events[0] = { phase: "intake", outcome: SKIPPED, by: "x", reason: value };
  checkRejects(`a skip with a ${label} reason`, blank, "non-empty reason");

  const blankBy = base();
  blankBy.events[0] = { phase: "intake", outcome: SKIPPED, by: value, reason: "x" };
  checkRejects(`a skip with a ${label} \`by\``, blankBy, "non-empty by");
}

// Non-string reason/by must not slip through the trim().
const numericBy = base();
numericBy.events[0] = { phase: "intake", outcome: SKIPPED, by: 7, reason: "x" };
checkRejects("a skip with a non-string `by`", numericBy, "non-empty by");

// The reason/by requirement must apply ONLY to skips: a normal event has never
// been required to carry a `by`, and 48 real ledgers depend on that.
const noByRan = base();
delete noByRan.events[0].by;
check(
  "a non-skipped event still needs no `by`",
  validateLedger(noByRan, "demo").valid,
  true
);

// ── Pre-existing rules, still enforced ──────────────────────────────────────

const badOutcome = base();
badOutcome.events[1].outcome = "GO";
checkRejects("an outcome illegal for its phase", badOutcome, "is not legal for phase plan");

const badReason = base();
badReason.events[1].reason = false;
checkRejects("a non-string reason on a normal event", badReason, "reason must be a string");

const badMode = base();
badMode.mode = "turbo";
checkRejects("an unknown mode", badMode, "mode must be one of");

const mismatched = base();
mismatched.current_phase = "review";
checkRejects("current_phase not matching the last event", mismatched, "does not match the last event");

const noEvents = base();
noEvents.events = [];
checkRejects("an empty events array", noEvents, "non-empty array");

const unknownPhase = base();
unknownPhase.events[1].phase = "deploy";
checkRejects("an unknown phase", unknownPhase, "not a known phase");

// ── expectLatestPhase, unchanged behaviour ──────────────────────────────────

check(
  "expectLatestPhase accepts the matching latest phase",
  expectLatestPhase(base(), "implement").valid,
  true
);
check(
  "expectLatestPhase rejects a stale ledger",
  expectLatestPhase(base(), "review").valid,
  false
);
check(
  "expectLatestPhase names a phase absent entirely",
  expectLatestPhase(base(), "review").errors.some((e) => e.includes("STALE ledger")),
  true
);

// ── phaseGaps: the distinction backlog-135 exists to create ─────────────────

// The whole point, in one assertion pair: two ledgers that reach the same
// phase, one having skipped `spec` on purpose and one having lost it, must be
// told apart. Before this change both looked exactly like the second.
const gapLedger = {
  task: "demo",
  mode: "full",
  current_phase: "plan",
  events: [
    { phase: "question", outcome: "COMPLETED" },
    { phase: "research", outcome: "COMPLETED" },
    { phase: "intake", outcome: "VALIDATED" },
    { phase: "spec", outcome: SKIPPED, by: "mathias", reason: "no API change" },
    { phase: "plan", outcome: "COMPLETED" },
  ],
};
check("phaseGaps: a declared skip lands in `skipped`", phaseGaps(gapLedger).skipped, ["spec"]);
check("phaseGaps: nothing is missing when the skip is declared", phaseGaps(gapLedger).missing, []);

const lostSpec = JSON.parse(JSON.stringify(gapLedger));
lostSpec.events.splice(3, 1);
check("phaseGaps: an omitted phase lands in `missing`", phaseGaps(lostSpec).missing, ["spec"]);
check("phaseGaps: and NOT in `skipped`", phaseGaps(lostSpec).skipped, []);

// Only phases up to current_phase are due — a run sitting at `plan` has not
// failed to ship, it has not got there yet. An enforcer without this would
// fire on every in-flight ledger.
check(
  "phaseGaps: phases after current_phase are not due",
  phaseGaps(gapLedger).missing.includes("ship"),
  false
);
check("phaseGaps: ran lists the phases that ran", phaseGaps(gapLedger).ran, [
  "question",
  "research",
  "intake",
  "plan",
]);

// A phase with both a skip and a real event counts as ran — re-entry after a
// skip is a real sequence (spec SKIPPED, then bounced back and specced).
const skipThenRan = JSON.parse(JSON.stringify(gapLedger));
skipThenRan.events.splice(4, 0, { phase: "spec", outcome: "VALIDATED" });
check("phaseGaps: a phase later actually run counts as ran", phaseGaps(skipThenRan).ran.includes("spec"), true);

check("phaseGaps: an unknown mode reports unknownMode", phaseGaps({ mode: "turbo" }).unknownMode, true);

// Express mode has a shorter sequence; a phase it never runs is not missing.
check(
  "phaseGaps: express mode does not miss `spec`",
  phaseGaps({
    mode: "express",
    current_phase: "ship",
    events: [
      { phase: "implement", outcome: "COMPLETED" },
      { phase: "review", outcome: "GO" },
      { phase: "ship", outcome: "COMPLETED" },
    ],
  }).missing,
  []
);
check("SEQ still declares all three modes", Object.keys(SEQ).sort(), ["express", "fast", "full"]);

console.log();
console.log(`passed: ${pass}   failed: ${fail}`);
if (fail !== 0) process.exit(1);
console.log("OK: ledger-schema.js records skips with a reason and an actor,");
console.log("    rejects a skip that states neither, and can tell a declared");
console.log("    skip from an omitted phase (backlog-135)");
