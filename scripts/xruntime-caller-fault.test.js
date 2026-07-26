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

// ── F1 regression: a dying runtime must never be attributed to the caller ───
// Every assertion above this block passes exitCode 0. That was the coverage
// hole: fault attribution was only ever exercised on a runtime that exited
// cleanly, so the path where a CRASHED runtime's output gets judged against
// the caller's output contract was never tested at all.
for (const code of [1, 2, 126, 127, 137, 139]) {
  check(`exit ${code} with junk on stdout is a RUNTIME fault, not a caller fault`, () => {
    const c = classify({
      exitCode: code,
      stderr: "",
      durationS: 0.2,
      timeoutS: 480,
      // xruntime-exec.sh merges stderr into stdout, so this is what a crashed
      // runtime actually looks like to the classifier.
      stdout: "opencode: command not found\nnode:internal/errors\n    throw err;\n",
    });
    assert.strictEqual(c.fault, "runtime", `exit ${code} attributed to ${c.fault}`);
    assert.strictEqual(c.outcome, "runtime-error");
    assert.strictEqual(c.exitCode, code);
    assert.ok(/command not found/.test(c.excerpt || ""), "excerpt should carry the wreckage");
  });
}

check("a crashed runtime that happens to emit a valid findings array is still a runtime fault", () => {
  // Output shape must not rescue a nonzero exit: a process that died after
  // printing something well-formed did not complete its work.
  const c = classify({ exitCode: 1, stderr: "", durationS: 0.2, timeoutS: 480, stdout: "[]" });
  assert.strictEqual(c.outcome, "runtime-error");
  assert.strictEqual(c.fault, "runtime");
});

check("an uncorroborated exit 3 is a generic runtime-error, never tree-mutation", () => {
  // D-3 says a bare exit code may not assert a specific cause. It may still
  // assert that the process failed.
  const c = classify({ exitCode: 3, stderr: "no marker here", durationS: 1, timeoutS: 480, stdout: "junk" });
  assert.strictEqual(c.outcome, "runtime-error");
  assert.strictEqual(c.fault, "runtime");
});

check("an uncorroborated exit 124 is a generic runtime-error, never timeout", () => {
  const c = classify({ exitCode: 124, stderr: "", durationS: 0.5, timeoutS: 480, stdout: "junk" });
  assert.strictEqual(c.outcome, "runtime-error");
  assert.strictEqual(c.fault, "runtime");
});

check("positive control: corroborated exit 3 and 124 keep their specific causes", () => {
  const mutated = classify({ exitCode: 3, stderr: "xruntime-exec: TREE-MUTATED — ...", durationS: 1, timeoutS: 480, stdout: "" });
  assert.strictEqual(mutated.outcome, "tree-mutation");
  assert.strictEqual(mutated.fault, "runtime");
  const timedOut = classify({ exitCode: 124, stderr: "", durationS: 480, timeoutS: 480, stdout: "" });
  assert.strictEqual(timedOut.outcome, "timeout");
  assert.strictEqual(timedOut.fault, "runtime");
});

check("exit 0 with junk output is still a CALLER fault (the #102 case is preserved)", () => {
  const c = classify({ exitCode: 0, stderr: "", durationS: 1, timeoutS: 480, stdout: "here you go: not json" });
  assert.strictEqual(c.outcome, "non-conforming-output");
  assert.strictEqual(c.fault, "caller");
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

// ── journal corruption must not poison the whole task forever ───────────────
const { readLatestJournalEntry } = require(path.join(ROOT, "scripts/lib/xruntime/xruntime-journal"));

function journalFixture(lines) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "xruntime-journal-"));
  fs.mkdirSync(path.join(dir, "briefs"));
  fs.writeFileSync(path.join(dir, "briefs", "jt-xruntime.jsonl"), lines.join("\n") + "\n");
  return dir;
}
const J = (o) => JSON.stringify(o);

check("historical corruption does not block a lookup in a later cycle", () => {
  const dir = journalFixture([
    J({ runtime: "opencode", digest: "old", outcome: "degraded", cycle: 1 }),
    "{{{ corrupted line from cycle 1 }}}",
    J({ runtime: "opencode", digest: "old", outcome: "healthy", cycle: 1 }),
    J({ runtime: "opencode", digest: "new", outcome: "healthy", cycle: 5 }),
  ]);
  const e = readLatestJournalEntry(dir, "jt", "opencode", "new", 5);
  assert.ok(e && !e.malformed, "a clean current-cycle match must be returned");
  assert.strictEqual(e.outcome, "healthy");
});

check("a brand-new digest is not permanently blocked by old corruption (the escape hatch)", () => {
  // This is the poison case: a runtime UPGRADE produces a digest that has
  // never appeared, so the scan runs off the end of the file, past the old
  // corruption, every single time.
  const dir = journalFixture([
    J({ runtime: "opencode", digest: "d1", outcome: "degraded", cycle: 1 }),
    "}}} truncated write from a crash two months ago",
    J({ runtime: "opencode", digest: "d1", outcome: "degraded", cycle: 2 }),
  ]);
  const e = readLatestJournalEntry(dir, "jt", "opencode", "upgraded-digest", 7);
  assert.strictEqual(e, null, "an upgraded runtime must be probeable, not blocked by stale corruption");
});

check("corruption INSIDE the current cycle still fails closed (positive control)", () => {
  const dir = journalFixture([
    J({ runtime: "opencode", digest: "d1", outcome: "healthy", cycle: 3 }),
    "%%% crash-before-persist, current cycle",
  ]);
  const e = readLatestJournalEntry(dir, "jt", "opencode", "d1", 3);
  assert.ok(e && e.malformed, "current-cycle corruption must still block");
  assert.strictEqual(e.reason, "malformed-journal");
});

check("with an unknown cycle the strict pre-existing fail-closed behaviour is preserved", () => {
  const dir = journalFixture([
    J({ runtime: "opencode", digest: "d1", outcome: "degraded", cycle: 1 }),
    "### corrupt",
  ]);
  const e = readLatestJournalEntry(dir, "jt", "opencode", "never-seen", null);
  assert.ok(e && e.malformed, "unknown cycle must assume the worst");
});

// ── lifecycle: an unrecognized verdict status must not be read as NO-GO ─────
const { deriveRoundState } = require(path.join(ROOT, "scripts/lib/review/review-lifecycle"));

check("a GO prior still starts a fresh cycle (positive control)", () => {
  const s = deriveRoundState({ status: "GO", cycle: 2, round: 4 });
  assert.strictEqual(s.round, 1);
  assert.strictEqual(s.cycle, 3);
  assert.strictEqual(s.freshCycle, true);
});

check("a NO-GO prior still continues the cycle (positive control)", () => {
  const s = deriveRoundState({ status: "NO-GO", cycle: 2, round: 4 });
  assert.strictEqual(s.round, 5);
  assert.strictEqual(s.cycle, 2);
});

check("an unrecognized status is refused, not silently continued as NO-GO", () => {
  for (const bad of ["GOO", "", "no-go", null, 0, { }]) {
    assert.throws(
      () => deriveRoundState({ status: bad, cycle: 2, round: 4, rounds_audit: [{ round: 4 }] }),
      /expected "GO" or "NO-GO"/,
      `status ${JSON.stringify(bad)} was accepted`
    );
  }
});

// ── version probe: never hash "nothing" as if it were a version ─────────────
const { probeVersion, computeDigest } = require(path.join(ROOT, "scripts/lib/xruntime/xruntime-digest"));

check("a missing runtime binary yields a placeholder digest, not a hash of empty output", () => {
  const r = computeDigest("ghostrt", "/nonexistent/definitely-not-a-runtime", "read-only");
  assert.strictEqual(r.digest, "ghostrt:version-unavailable");
  assert.strictEqual(r.versionProbeTimedOut, true);
  assert.ok(/spawn-error:ENOENT/.test(r.versionProbeReason || ""), `reason was ${r.versionProbeReason}`);
});

check("the probe reports a missing binary as a spawn error, not as a timeout", () => {
  const p = probeVersion("/nonexistent/definitely-not-a-runtime");
  assert.strictEqual(p.timedOut, false, "ENOENT is not a timeout");
  assert.strictEqual(p.unavailable, true);
});

check("positive control: a working binary still produces a real hashed digest", () => {
  const r = computeDigest("echo", "/bin/echo", "read-only");
  assert.ok(/^echo:[a-f0-9]{16}$/.test(r.digest), `digest was ${r.digest}`);
  assert.strictEqual(r.versionProbeTimedOut, false);
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

check("E2E: a runtime that CRASHES suppresses the next probe (F1 regression)", () => {
  // The two stubs used elsewhere in this file both exit 0. A dying runtime is
  // a different path entirely and had no E2E coverage — which is how the
  // regression reached the branch.
  const binDir = fs.mkdtempSync(path.join(os.tmpdir(), "xruntime-stub-"));
  const stub = path.join(binDir, "dying-runtime");
  fs.writeFileSync(stub, "#!/bin/sh\necho 'opencode: fatal: cannot open display' >&2\nexit 1\n", { mode: 0o755 });
  try {
    const { first, second } = probeTwice(stub);
    assert.strictEqual(first.status, "degraded");
    assert.strictEqual(first.reason, "runtime-error", `first reason: ${first.reason}`);
    assert.strictEqual(first.fault, "runtime", "a crashed runtime must not be blamed on the caller");
    assert.ok(!first.remedy, "a runtime crash must not offer the caller-fault remedy");
    assert.strictEqual(second.status, "skipped-degraded", "breaker failed to arm on a crashed runtime");
  } finally {
    fs.rmSync(binDir, { recursive: true, force: true });
  }
});

check("E2E positive control: a silent runtime DOES suppress the next probe", () => {
  // /bin/true exits 0 having written nothing — the runtime ran and produced
  // nothing, a genuine runtime failure.
  const { first, second } = probeTwice("/bin/true");
  assert.strictEqual(first.reason, "empty-output");
  assert.strictEqual(first.fault, "runtime");
  assert.strictEqual(second.status, "skipped-degraded", "breaker failed to arm on a runtime fault");
});

// ── the breaker must STAY armed, not alternate ──────────────────────────────
// Every earlier E2E here inspected probe #2 only, which structurally cannot
// see an every-other-probe failure: it attests a property that holds at N=2
// and says nothing about N=3. These run a sequence and assert the whole shape.
function probeSequence(stubBin, n, extraArgsAt = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "xruntime-seq-"));
  fs.mkdirSync(path.join(dir, "briefs"));
  const promptFile = path.join(dir, "prompt.txt");
  fs.writeFileSync(promptFile, "find bugs");
  const out = [];
  for (let i = 1; i <= n; i++) {
    const args = [
      path.join(ROOT, "scripts/xruntime-review.js"), "opencode",
      "--task", "seq", "--prompt-file", promptFile,
      "--cycle", "3", "--round", "1", "--timeout", "20",
      ...(extraArgsAt[i] || []),
    ];
    const r = spawnSync("node", args, {
      cwd: dir, encoding: "utf8",
      env: { ...process.env, XRUNTIME_BIN: typeof stubBin === "function" ? stubBin(i) : stubBin },
    });
    out.push(JSON.parse(r.stdout.trim().split("\n").pop()).status);
  }
  fs.rmSync(dir, { recursive: true, force: true });
  return out;
}

function writeStub(dir, name, body) {
  const p = path.join(dir, name);
  fs.writeFileSync(p, body, { mode: 0o755 });
  return p;
}

check("six probes against a silent runtime: armed once, then refused every time", () => {
  const seq = probeSequence("/bin/true", 6);
  assert.strictEqual(seq[0], "degraded", `first probe: ${seq[0]}`);
  const rest = seq.slice(1);
  assert.ok(
    rest.every((s) => s === "skipped-degraded"),
    `breaker alternated instead of staying armed: ${seq.join(", ")}`
  );
});

check("five probes against an ABSENT binary: the breaker arms (version probe must not bypass it)", () => {
  const seq = probeSequence("/nonexistent/definitely-not-a-runtime", 5);
  assert.strictEqual(seq[0], "degraded");
  assert.ok(
    seq.slice(1).every((s) => s === "skipped-degraded"),
    `a runtime that cannot start was re-probed: ${seq.join(", ")}`
  );
});

check("positive control: a HEALTHY runtime is probed every time, never refused", () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "xruntime-stubs-"));
  try {
    const stub = writeStub(d, "healthy",
      '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "stub 1.0"; exit 0; fi\necho "[]"\n');
    const seq = probeSequence(stub, 6);
    assert.ok(seq.every((s) => s === "healthy"), `breaker over-armed on a healthy runtime: ${seq.join(", ")}`);
  } finally {
    fs.rmSync(d, { recursive: true, force: true });
  }
});

check("positive control: --human-retry bypasses an armed breaker", () => {
  const seq = probeSequence("/bin/true", 3, { 3: ["--human-retry"] });
  assert.deepStrictEqual(seq, ["degraded", "skipped-degraded", "degraded"], seq.join(", "));
});

check("positive control: a runtime VERSION change re-probes (the escape hatch)", () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "xruntime-stubs-"));
  try {
    const mk = (v) => writeStub(d, `v${v}`,
      `#!/bin/sh\nif [ "$1" = "--version" ]; then echo "stub ${v}"; exit 0; fi\nexit 0\n`);
    const a = mk(1), b = mk(2);
    const seq = probeSequence((i) => (i === 3 ? b : a), 3);
    assert.deepStrictEqual(seq, ["degraded", "skipped-degraded", "degraded"], seq.join(", "));
  } finally {
    fs.rmSync(d, { recursive: true, force: true });
  }
});

check("a refusal record does not clear a prior arm in the journal lookup", () => {
  const dir = journalFixture([
    J({ runtime: "opencode", digest: "d1", outcome: "degraded", reason: "empty-output", cycle: 3, fault: "runtime" }),
    J({ runtime: "opencode", digest: "d1", outcome: "skipped-degraded", cycle: 3 }),
  ]);
  const e = readLatestJournalEntry(dir, "jt", "opencode", "d1", 3);
  assert.strictEqual(e.outcome, "degraded", "the refusal record shadowed the degradation");
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
