// scripts/lib/mandated-steps-rules.js — CommonJS, repo-local, pure rules.
//
// THE DEFECT THIS CLOSES
//
// A mandated pipeline step that is skipped without leaving a record is
// indistinguishable from one that ran and passed. Nothing in the artifacts
// separates the two, so the skip reads as a pass forever after. Three real
// instances, all from the same friction log:
//
//   - roster-run Step 4 mandates `/roster-doctor preflight` before any phase
//     that builds or tests. It was not run across an entire day of
//     implementation work, and the environment had three real problems it
//     would have surfaced (stale main, wedged Docker daemon, half-upgraded
//     opam switch). Nothing recorded that it had not run.
//   - the roster-ship human gate was skipped under a standing autonomy
//     delegation. Legitimate — but it went into the record as though a human
//     had answered, which is exactly the distinction the gate exists to keep.
//   - the cross-runtime pass, mandatory on every Fast/Full PR, was disabled by
//     its own circuit breaker on a caller-side fault; the round then looked
//     like a round on which cross-runtime simply had nothing to say.
//
// This is the same defect as the review gate's silent degradation, one level
// up: a green that means "not checked", not "checked out".
//
// THE RULE
//
// Every mandated step emits exactly one record. `skipped` is a legal outcome —
// the point is not to forbid skipping, it is to make a skip *say so*, in
// writing, with a reason and an actor. The absence of a required record is
// then a failure, because absence is the only thing that can no longer be
// confused with success.
//
// Deliberately NOT enforced: whether a `reason` is a *good* reason. That is a
// human judgement. What is enforced is that one exists, that it is attributable,
// and that an agent's decision is never recorded as a human's.
"use strict";

const OUTCOMES = new Set(["ran", "skipped"]);
const ACTORS = new Set(["human", "agent"]);
// `ran` steps additionally report what they found. NOT-READY is a legitimate,
// recordable result — a preflight that ran and found the environment broken is
// evidence, and must not be indistinguishable from one that found it healthy.
const RESULTS = new Set(["READY", "NOT-READY", "PASS", "FAIL", "N/A"]);

// The mandated-step catalogue. A step id absent from here is rejected rather
// than accepted, so a typo can never stand in for the step it resembles —
// `prefligt` recorded, `preflight` still missing, is a failure, not a pass.
const STEPS = {
  preflight: {
    description: "/roster-doctor preflight — required before any phase that builds or tests (roster-run Step 4)",
  },
  "human-gate": {
    description: "the human validation quiz (rules/human-validation.md)",
    // A gate whose whole purpose is human judgement cannot be recorded as
    // having *run* by an agent. Under a standing delegation the honest record
    // is outcome: skipped, actor: agent, reason: <the delegation>.
    humanOnly: true,
  },
  xruntime: {
    description: "the cross-runtime review pass (mandatory on Fast/Full PRs)",
  },
  "scope-gate": {
    description: "check-scope-diff.sh — the out-of-scope-change gate",
  },
  "degraded-specialist": {
    description: "a conditional specialist that was selected but could not run",
    repeatable: true,
  },
};

// Which steps each phase mandates. `--require` overrides this; an unknown
// phase is rejected, never defaulted to "nothing required" (that default would
// itself be a gate that cannot fail — a typo'd phase would demand nothing).
const PHASE_REQUIREMENTS = {
  intake: [],
  question: [],
  research: [],
  spec: [],
  plan: ["human-gate"],
  implement: ["preflight"],
  review: ["preflight", "scope-gate", "xruntime"],
  qa: ["preflight"],
  ship: ["preflight", "human-gate"],
};

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim() !== "";
}

// Validates one record in isolation. Returns an error string or null.
function validateRecord(record, index) {
  const where = `record ${index + 1}`;
  if (!record || typeof record !== "object" || Array.isArray(record)) return `${where} is not a JSON object`;
  if (!isNonEmptyString(record.step)) return `${where} has no \`step\``;
  if (!Object.prototype.hasOwnProperty.call(STEPS, record.step)) {
    return `${where} names an unknown step ${JSON.stringify(record.step)} — known steps: ${Object.keys(STEPS).join(", ")}. An unrecognised id cannot satisfy a requirement, so a typo here leaves the real step unrecorded.`;
  }
  if (!OUTCOMES.has(record.outcome)) {
    return `${where} (${record.step}) has outcome ${JSON.stringify(record.outcome)}, not one of ${[...OUTCOMES].join(" | ")}`;
  }
  if (!ACTORS.has(record.actor)) {
    return `${where} (${record.step}) has actor ${JSON.stringify(record.actor)}, not one of ${[...ACTORS].join(" | ")} — a skip must be attributable`;
  }
  if (!isNonEmptyString(record.ts)) return `${where} (${record.step}) has no \`ts\` timestamp`;

  if (record.outcome === "skipped") {
    if (!isNonEmptyString(record.reason)) {
      return `${where} (${record.step}) is outcome "skipped" with no \`reason\` — an unreasoned skip is the thing this check exists to stop`;
    }
    return null;
  }

  // outcome === "ran"
  if (!RESULTS.has(record.result)) {
    return `${where} (${record.step}) is outcome "ran" with result ${JSON.stringify(record.result)}, not one of ${[...RESULTS].join(" | ")} — a step that ran must say what it found`;
  }
  if ((record.result === "NOT-READY" || record.result === "FAIL") && !isNonEmptyString(record.reason)) {
    return `${where} (${record.step}) reports ${record.result} with no \`reason\``;
  }
  if (STEPS[record.step].humanOnly && record.actor !== "human") {
    return `${where} (${record.step}) is recorded as outcome "ran" by actor ${JSON.stringify(record.actor)} — this step is human-only. An agent proceeding under a standing delegation must record outcome "skipped" with the delegation as its reason, so an agent's decision is never journaled as a human's.`;
  }
  return null;
}

// Cross-record rules: one record per step (except explicitly repeatable ones),
// and every record belongs to the task under check.
function validateRecordSet(records, task) {
  const errors = [];
  const seen = new Map();
  records.forEach((record, index) => {
    const error = validateRecord(record, index);
    if (error) {
      errors.push(error);
      return;
    }
    if (task && record.task !== undefined && record.task !== task) {
      errors.push(`record ${index + 1} (${record.step}) is stamped task ${JSON.stringify(record.task)}, not ${JSON.stringify(task)}`);
      return;
    }
    if (STEPS[record.step].repeatable) return;
    if (seen.has(record.step)) {
      errors.push(
        `step ${JSON.stringify(record.step)} is recorded twice (records ${seen.get(record.step) + 1} and ${index + 1}) — two records for one step means one of them is not evidence about this run`
      );
      return;
    }
    seen.set(record.step, index);
  });
  return errors;
}

function resolveRequirements({ phase, require: explicit }) {
  if (Array.isArray(explicit) && explicit.length > 0) {
    const unknown = explicit.filter((step) => !Object.prototype.hasOwnProperty.call(STEPS, step));
    if (unknown.length > 0) return { error: `--require names unknown step(s): ${unknown.join(", ")}` };
    return { required: explicit };
  }
  if (!Object.prototype.hasOwnProperty.call(PHASE_REQUIREMENTS, phase)) {
    return {
      error: `unknown phase ${JSON.stringify(phase)} — known phases: ${Object.keys(PHASE_REQUIREMENTS).join(", ")}. Refusing to default an unrecognised phase to "nothing required".`,
    };
  }
  return { required: PHASE_REQUIREMENTS[phase] };
}

// THE load-bearing rule: a required step with no record is a failure. Not a
// warning, not an empty list — a failure with the step named.
function computeMissingRequirements(records, required) {
  const recorded = new Set(records.filter((r) => r && typeof r.step === "string").map((r) => r.step));
  return required
    .filter((step) => !recorded.has(step))
    .map(
      (step) =>
        `no record for mandated step ${JSON.stringify(step)} (${STEPS[step].description}). An unrecorded step is indistinguishable from one that ran and passed; if it was skipped, record it as skipped with a reason.`
    );
}

function evaluate({ records, phase, require: explicit, task }) {
  const resolved = resolveRequirements({ phase, require: explicit });
  if (resolved.error) return { usageError: resolved.error };

  const invalid = validateRecordSet(records, task);
  const missing = computeMissingRequirements(records, resolved.required);
  return {
    usageError: null,
    required: resolved.required,
    invalid,
    missing,
    ok: invalid.length === 0 && missing.length === 0,
  };
}

module.exports = {
  OUTCOMES,
  ACTORS,
  RESULTS,
  STEPS,
  PHASE_REQUIREMENTS,
  validateRecord,
  validateRecordSet,
  resolveRequirements,
  computeMissingRequirements,
  evaluate,
};
