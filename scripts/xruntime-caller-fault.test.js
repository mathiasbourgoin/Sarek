#!/usr/bin/env node
// Tests for #102 — a non-conforming OUTPUT is a caller fault and must not arm
// the runtime circuit breaker.
//
// Every assertion here has a positive control: the same scenario with a
// RUNTIME fault must still arm the breaker. Without that pair, "the breaker
// did not fire" is indistinguishable from a breaker that no longer works.
"use strict";

const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const { classify, faultFor } = require(path.join(ROOT, "scripts/lib/review/../xruntime/xruntime-classify"));
const { shouldRefuseDegraded } = require(path.join(ROOT, "scripts/lib/xruntime/xruntime-journal"));

let pass = 0;
const results = [];
function check(name, fn) {
  try {
    fn();
    pass += 1;
    results.push(`  ok — ${name}`);
  } catch (e) {
    results.push(`  FAIL — ${name}\n      ${e.message}`);
    process.exitCode = 1;
  }
}

// ── unit: fault attribution ─────────────────────────────────────────────────
check("malformed output is attributed to the caller", () => {
  const c = classify({ exitCode: 0, stderr: "", durationS: 1, timeoutS: 480, stdout: "here you go: not json" });
  assert.strictEqual(c.outcome, "non-conforming-output");
  assert.strictEqual(c.fault, "caller");
});

check("a findings array that fails the schema is still a caller fault", () => {
  const c = classify({
    exitCode: 0, stderr: "", durationS: 1, timeoutS: 480,
    stdout: '```json\n[{"nope": true}]\n```',
  });
  assert.strictEqual(c.outcome, "non-conforming-output");
  assert.strictEqual(c.fault, "caller");
});

check("empty output remains a RUNTIME fault (positive control)", () => {
  const c = classify({ exitCode: 0, stderr: "", durationS: 1, timeoutS: 480, stdout: "   " });
  assert.strictEqual(c.outcome, "empty-output");
  assert.strictEqual(c.fault, "runtime");
});

check("timeout and tree-mutation remain RUNTIME faults", () => {
  assert.strictEqual(faultFor("timeout"), "runtime");
  assert.strictEqual(faultFor("tree-mutation"), "runtime");
  assert.strictEqual(faultFor("spawn-error"), "runtime");
});

check("a well-formed empty findings array is healthy, not degraded", () => {
  const c = classify({ exitCode: 0, stderr: "", durationS: 1, timeoutS: 480, stdout: "[]" });
  assert.strictEqual(c.outcome, "healthy");
});

// ── unit: the breaker itself ────────────────────────────────────────────────
const journalBase = { outcome: "degraded", cycle: 7, runtime: "opencode", digest: "d1" };

check("caller-fault journal entry does NOT arm the breaker", () => {
  const refuse = shouldRefuseDegraded({
    reviewJson: null,
    journalEntry: { ...journalBase, reason: "non-conforming-output", fault: "caller" },
    runtime: "opencode", digest: "d1", humanRetry: false, currentCycle: 7,
  });
  assert.strictEqual(refuse, false);
});

check("runtime-fault journal entry DOES arm the breaker (positive control)", () => {
  const refuse = shouldRefuseDegraded({
    reviewJson: null,
    journalEntry: { ...journalBase, reason: "empty-output", fault: "runtime" },
    runtime: "opencode", digest: "d1", humanRetry: false, currentCycle: 7,
  });
  assert.strictEqual(refuse, true);
});

check("a pre-#102 entry with no fault key still arms the breaker", () => {
  const refuse = shouldRefuseDegraded({
    reviewJson: null,
    journalEntry: { ...journalBase, reason: "empty-output" },
    runtime: "opencode", digest: "d1", humanRetry: false, currentCycle: 7,
  });
  assert.strictEqual(refuse, true);
});

check("caller-fault cross_runtime entry in a NO-GO verdict does NOT arm the breaker", () => {
  const reviewJson = {
    status: "NO-GO",
    cross_runtime: { opencode: { status: "degraded", config_digest: "d1", fault: "caller" } },
  };
  assert.strictEqual(
    shouldRefuseDegraded({ reviewJson, journalEntry: null, runtime: "opencode", digest: "d1", humanRetry: false, currentCycle: null }),
    false
  );
});

check("runtime-fault cross_runtime entry in a NO-GO verdict DOES arm it (positive control)", () => {
  const reviewJson = {
    status: "NO-GO",
    cross_runtime: { opencode: { status: "degraded", config_digest: "d1", fault: "runtime" } },
  };
  assert.strictEqual(
    shouldRefuseDegraded({ reviewJson, journalEntry: null, runtime: "opencode", digest: "d1", humanRetry: false, currentCycle: null }),
    true
  );
});

// ── end-to-end through the real CLI ─────────────────────────────────────────
// Two consecutive probes in the same cycle. The question is only ever whether
// the SECOND one ran or was suppressed.
function probeTwice(stubBin) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "xruntime-e2e-"));
  fs.mkdirSync(path.join(dir, "briefs"));
  const promptFile = path.join(dir, "prompt.txt");
  fs.writeFileSync(promptFile, "find bugs");
  const run = () =>
    spawnSync(
      "node",
      [
        path.join(ROOT, "scripts/xruntime-review.js"), "opencode",
        "--task", "faulttest", "--prompt-file", promptFile,
        "--cycle", "3", "--round", "1", "--timeout", "20",
      ],
      { cwd: dir, encoding: "utf8", env: { ...process.env, XRUNTIME_BIN: stubBin } }
    );
  const first = run();
  const second = run();
  const parse = (r) => JSON.parse(r.stdout.trim().split("\n").pop());
  return { dir, first: parse(first), second: parse(second) };
}

check("E2E: a malformed answer does not suppress the next probe", () => {
  // /bin/echo prints its own argv — valid execution, output that is not a
  // findings array. Exactly the caller-fault shape.
  const { first, second } = probeTwice("/bin/echo");
  assert.strictEqual(first.status, "degraded", `first status: ${first.status}`);
  assert.strictEqual(first.reason, "non-conforming-output");
  assert.strictEqual(first.fault, "caller");
  assert.ok(/--emit-contract/.test(first.remedy || ""), "remedy should point at --emit-contract");
  assert.notStrictEqual(second.status, "skipped-degraded", "breaker was armed by a caller fault");
  assert.strictEqual(second.reason, "non-conforming-output");
});

check("E2E positive control: a silent runtime DOES suppress the next probe", () => {
  // /bin/true exits 0 having written nothing — the runtime ran and produced
  // nothing, a genuine runtime failure.
  const { first, second } = probeTwice("/bin/true");
  assert.strictEqual(first.reason, "empty-output");
  assert.strictEqual(first.fault, "runtime");
  assert.strictEqual(second.status, "skipped-degraded", "breaker failed to arm on a runtime fault");
});

check("E2E: --emit-contract prints the contract and invokes nothing", () => {
  const r = spawnSync("node", [path.join(ROOT, "scripts/xruntime-review.js"), "--emit-contract"], {
    encoding: "utf8",
  });
  assert.strictEqual(r.status, 0);
  assert.ok(/Output contract \(machine-parsed/.test(r.stdout));
  assert.ok(/specialist/.test(r.stdout), "contract must state the specialist field");
  assert.ok(/\[\]/.test(r.stdout), "contract must say an empty array is a success");
});

console.log(results.join("\n"));
console.log(`\nxruntime-caller-fault.test: ${pass}/${results.length} passed`);
