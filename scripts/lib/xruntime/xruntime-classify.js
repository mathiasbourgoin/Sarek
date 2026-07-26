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
  "runtime-error": "runtime",
  "non-conforming-output": "caller",
};

function faultFor(outcome) {
  return FAULT_BY_OUTCOME[outcome] || "runtime";
}

// Fault values are derived through faultFor() rather than written inline:
// FAULT_BY_OUTCOME is meant to be the single mechanical source of truth for
// attribution, and a second copy of the mapping is how that guarantee decays.
function classifyOutput(stdout) {
  const trimmed = (stdout || "").trim();
  if (trimmed === "") return { outcome: "empty-output", fault: faultFor("empty-output") };
  const parsed = extractJson(trimmed);
  if (!parsed.ok || !validateFindingsArray(parsed.value)) {
    return {
      outcome: "non-conforming-output",
      fault: faultFor("non-conforming-output"),
      excerpt: trimmed.slice(0, 500),
    };
  }
  return { outcome: "healthy", findings: parsed.value };
}

// Top-level classification: exit-code corroboration takes precedence over
// output inspection (FR-088), which runs only when the exit code is
// uncorroborated.
//
// EXIT STATUS IS CHECKED BEFORE OUTPUT SHAPE, and a nonzero exit is always a
// runtime fault. Without this, fault attribution had a hole big enough to
// drive a dying runtime through: classifyExitCode corroborates only exit 3
// (with the TREE-MUTATED marker) and exit 124 (with a corroborating
// duration), so exit 1 / 127 / 137 / 139 fell through to classifyOutput —
// which never looks at the exit code. xruntime-exec.sh merges stderr into
// stdout, so a crashed runtime's death rattle ("command not found",
// a stack trace, a partial banner) is not a findings array, and the run was
// therefore attributed to the CALLER and never armed the breaker. A runtime
// that dies on every invocation would have been re-probed forever.
//
// An uncorroborated nonzero exit is deliberately NOT promoted to a specific
// cause: `runtime-error` claims only that the process failed, never which
// way. That keeps D-3's rule intact — a bare exit 3 still does not get to
// assert `tree-mutation` without the marker — while refusing to let the
// failure be read as an answer.
function classify({ exitCode, stderr, durationS, timeoutS, stdout }) {
  const corroborated = classifyExitCode(exitCode, stderr, durationS, timeoutS);
  if (corroborated) return { outcome: corroborated, fault: faultFor(corroborated) };

  if (exitCode !== 0) {
    const merged = ((stdout || "") + (stderr || "")).trim();
    return {
      outcome: "runtime-error",
      fault: faultFor("runtime-error"),
      exitCode,
      excerpt: merged.slice(0, 500),
    };
  }

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
