// scripts/lib/xruntime-classify.js — CommonJS.
//
// Fully mechanical classification for scripts/xruntime-review.js (FR-088..092,
// Amendment D-3; specs/review-v2-corrections.md INV-6). No model judgment
// anywhere in this module — exit-code corroboration, byte inspection, and
// schema validation only.
"use strict";

const { loadFindingSchema } = require("../review/finding-schema");

// INV-6: a spawn-LAYER failure (the OS/runtime never started the subprocess
// at all — E2BIG argv-too-large, ENOENT missing binary, EACCES, ...) is a
// distinct, pre-runtime failure class. It must never be conflated with
// "empty-output" (which implies the runtime DID execute and produced
// nothing) — that would misattribute a transport failure to the model and
// trip the breaker for the wrong reason. ETIMEDOUT is excluded: that is the
// wrapper-level timeout backstop, classified via classifyExitCode's own
// corroborated-timeout path instead.
function isSpawnError(result) {
  return !!(result && result.error && result.error.code && result.error.code !== "ETIMEDOUT");
}

// D-3: exit 3 classifies `tree-mutation` ONLY when stderr carries the
// wrapper's deterministic TREE-MUTATED marker; exit 124 classifies `timeout`
// ONLY when the measured duration corroborates it. An uncorroborated exit
// code (e.g. exit 3 with no marker — should not happen with an unmodified
// wrapper, but the helper must not assume) falls through to output
// inspection rather than trusting the bare exit code.
function classifyExitCode(exitCode, stderr, durationS, timeoutS) {
  if (exitCode === 3 && /TREE-MUTATED/.test(stderr || "")) return "tree-mutation";
  if (exitCode === 124 && typeof durationS === "number" && durationS >= timeoutS) return "timeout";
  return null;
}

// Fence-aware JSON extraction: prefers the last ```json fenced block (a
// banner or preamble may precede it, EC-5); falls back to parsing the whole
// trimmed stdout.
function extractJson(stdout) {
  const fences = [...stdout.matchAll(/```json\s*([\s\S]*?)```/g)];
  const candidate = fences.length ? fences[fences.length - 1][1] : stdout;
  try {
    return { ok: true, value: JSON.parse(candidate.trim()) };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

// A schema-valid findings array (empty included) is the only shape that
// classifies healthy — anything else (wrong root type, any element failing
// the canonical finding schema) is non-conforming.
function validateFindingsArray(candidate) {
  if (!Array.isArray(candidate)) return false;
  const validator = loadFindingSchema();
  return candidate.every((f) => validator.validate(f).valid);
}

// #102: fault attribution. Every non-healthy outcome is blamed on exactly one
// side, and only a `runtime` fault may arm the circuit breaker.
//
//   runtime — the runtime or its environment misbehaved: it never started
//             (spawn-error), never finished (timeout), mutated the tree
//             (tree-mutation), or ran and returned nothing at all
//             (empty-output). Re-probing it at the same digest is wasted work,
//             which is what the breaker exists to prevent.
//
//   caller  — the runtime ran and answered; the answer did not satisfy a
//             contract the CALLER is responsible for stating
//             (non-conforming-output). The runtime is not known to be
//             unhealthy, so suppressing every later probe punishes unrelated
//             work for one bad prompt. Fix the prompt — see
//             `xruntime-review.js --emit-contract` — and probe again.
const FAULT_BY_OUTCOME = {
  "spawn-error": "runtime",
  timeout: "runtime",
  "tree-mutation": "runtime",
  "empty-output": "runtime",
  "non-conforming-output": "caller",
};

function faultFor(outcome) {
  return FAULT_BY_OUTCOME[outcome] || "runtime";
}

function classifyOutput(stdout) {
  const trimmed = (stdout || "").trim();
  if (trimmed === "") return { outcome: "empty-output", fault: "runtime" };
  const parsed = extractJson(trimmed);
  if (!parsed.ok || !validateFindingsArray(parsed.value)) {
    return {
      outcome: "non-conforming-output",
      fault: "caller",
      excerpt: trimmed.slice(0, 500),
    };
  }
  return { outcome: "healthy", findings: parsed.value };
}

// Top-level classification: exit-code corroboration takes precedence over
// output inspection (FR-088), which runs only when the exit code is
// uncorroborated.
function classify({ exitCode, stderr, durationS, timeoutS, stdout }) {
  const corroborated = classifyExitCode(exitCode, stderr, durationS, timeoutS);
  if (corroborated) return { outcome: corroborated, fault: faultFor(corroborated) };
  return classifyOutput(stdout);
}

module.exports = {
  classifyExitCode,
  extractJson,
  validateFindingsArray,
  classifyOutput,
  classify,
  isSpawnError,
  faultFor,
  FAULT_BY_OUTCOME,
};
