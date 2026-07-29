#!/usr/bin/env node
"use strict";
/**
 * Covering test for check-review-finding-tests.js (backlog-164).
 *
 * Offline throughout: the subject accepts --threads/--files so every case is a
 * fixture, and CI needs no network and no GitHub token. The live path is
 * exercised by hand against real PRs; what is pinned here is the DECISION table,
 * including every fail-closed branch — because "verified nothing" is the failure
 * mode this gate is most likely to have.
 */

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const SUBJECT = path.join(__dirname, "check-review-finding-tests.js");
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "finding-tests-"));
let pass = 0;
let fail = 0;

/* A refusal must be the REFUSAL, not merely exit 1 — an uncaught exception also
   exits 1, so asserting the status alone lets a crash satisfy a case whose
   subject is the refusal path. Found by mutating the refusal away: the code then
   threw on an undefined escape, every status-only case still "passed", and only
   the id-naming assertion noticed. */
function checkRefused(desc, r, id) {
  const ok = r.status === 1 && /REFUSED:/.test(r.out) && (!id || r.out.includes(id));
  if (ok) {
    console.log(`  PASS: ${desc}`);
    pass++;
  } else {
    console.log(
      `  FAIL: ${desc} -- status=${r.status} refused=${/REFUSED:/.test(r.out)} ` +
        `names-id=${!id || r.out.includes(id)}`
    );
    fail++;
  }
}

function check(desc, got, want) {
  if (String(got) === String(want)) {
    console.log(`  PASS: ${desc}`);
    pass++;
  } else {
    console.log(`  FAIL: ${desc} -- expected ${want}, got ${got}`);
    fail++;
  }
}

function write(name, content) {
  const p = path.join(TMP, name);
  fs.writeFileSync(p, content);
  return p;
}

/** A GraphQL-shaped document with the given threads. */
function threadsDoc(threads) {
  return JSON.stringify({
    data: {
      repository: {
        pullRequest: {
          reviewThreads: {
            nodes: threads.map((t) => ({
              isResolved: t.resolved !== false,
              path: t.path || "src/thing.ml",
              resolvedBy: { login: t.resolvedBy ?? "coderabbitai[bot]" },
              comments: {
                nodes: [
                  {
                    author: { login: t.author ?? "coderabbitai" },
                    body: t.id
                      ? `some finding\n<!-- cr-comment:v1:${t.id} -->`
                      : "a finding with no id marker",
                  },
                ],
              },
            })),
          },
        },
      },
    },
  });
}

function run(args) {
  const r = spawnSync("node", [SUBJECT, ...args], { encoding: "utf8" });
  return { status: r.status, out: (r.stdout || "") + (r.stderr || "") };
}

console.log("check-review-finding-tests.js covering test");

// ── the core refusal ─────────────────────────────────────────────────────────
const oneFinding = write("t1.json", threadsDoc([{ id: "aaa111" }]));
const noTests = write("f1.txt", "sarek/ppx/Sarek_lower_ir.ml\nREADME.md\n");
const emptyEsc = write("e-empty.tsv", "# nothing recorded\n");

let r = run(["1", "--threads", oneFinding, "--files", noTests, "--escapes", emptyEsc]);
checkRefused("a resolved finding with NO test path and no escape is REFUSED", r, "aaa111");

// The positive control. Without it, "refuses a fix with no test" and "refuses
// every PR" are the same observation, and the second makes the gate unusable
// rather than strict.
const withTests = write(
  "f2.txt",
  "sarek/ppx/Sarek_lower_ir.ml\nsarek/tests/unit/test_thing.ml\n"
);
r = run(["1", "--threads", oneFinding, "--files", withTests, "--escapes", emptyEsc]);
check("  (control) the same finding WITH a test path is accepted", r.status, 0);

// Each test-path shape the predicate claims to recognise, asserted individually
// so dropping one arm fails on that arm rather than somewhere vague.
for (const p of [
  "sarek/tests/unit/test_x.ml",
  "spoc/test/thing.ml",
  "scripts/foo.test.js",
  "scripts/test-suite-counts.test.sh",
  "scripts/prove-red-fixtures/some-log.txt",
]) {
  const f = write(`f-${p.replace(/[^a-z0-9]/gi, "_")}.txt`, `src/a.ml\n${p}\n`);
  r = run(["1", "--threads", oneFinding, "--files", f, "--escapes", emptyEsc]);
  check(`  test path recognised: ${p}`, r.status, 0);
}

// A path that merely CONTAINS the word must not count, or the gate is satisfied
// by any file with "test" in its name.
const notATest = write("f-nota.txt", "src/latest_results.ml\nsrc/contest.ml\n");
r = run(["1", "--threads", oneFinding, "--files", notATest, "--escapes", emptyEsc]);
checkRefused("  (control) 'latest'/'contest' are NOT test paths", r, "aaa111");

// ── the recorded escape ──────────────────────────────────────────────────────
const goodEsc = write("e-ok.tsv", "aaa111\ta doc-only correction\tmathias\n");
r = run(["1", "--threads", oneFinding, "--files", noTests, "--escapes", goodEsc]);
check("a recorded escape excuses the finding", r.status, 0);
check("  and the output shows the reason and owner", /doc-only correction.*mathias/.test(r.out), true);

const wrongIdEsc = write("e-wrong.tsv", "bbb222\tunrelated\tmathias\n");
r = run(["1", "--threads", oneFinding, "--files", noTests, "--escapes", wrongIdEsc]);
checkRefused("  an escape for a DIFFERENT id does not excuse it", r, "aaa111");

// A finding with no cr-comment id can never be excused by id, so it must refuse
// rather than fall through as excused-by-default.
const noId = write("t-noid.json", threadsDoc([{ id: null }]));
r = run(["1", "--threads", noId, "--files", noTests, "--escapes", goodEsc]);
checkRefused("a finding with no cr-comment id cannot be excused", r, null);

// ── malformed escapes are exit 2, not a silent excuse ────────────────────────
r = run(["1", "--threads", oneFinding, "--files", noTests,
         "--escapes", write("e-2f.tsv", "aaa111\tonly two fields\n")]);
check("an escape with 2 fields is a usage error (2), not an excuse", r.status, 2);

r = run(["1", "--threads", oneFinding, "--files", noTests,
         "--escapes", write("e-blank.tsv", "aaa111\t\tmathias\n")]);
check("an escape with a BLANK reason is a usage error (2)", r.status, 2);

r = run(["1", "--threads", oneFinding, "--files", noTests,
         "--escapes", write("e-noby.tsv", "aaa111\ta reason\t\n")]);
check("an escape with a blank owner is a usage error (2)", r.status, 2);

// ── fail-closed on absent / unusable data ────────────────────────────────────
// The trap this repo keeps hitting: no data must never read as "nothing wrong".
r = run(["1", "--threads", write("t-empty.json", "{}"), "--files", noTests]);
check("a response with no reviewThreads array is exit 2, NOT 0", r.status, 2);

r = run(["1", "--threads", write("t-garbage.json", "not json at all"), "--files", noTests]);
check("unparseable thread data is exit 2", r.status, 2);

// ── zero findings says so, rather than looking like a verified pass ──────────
r = run(["1", "--threads", write("t-none.json", threadsDoc([])), "--files", noTests]);
check("zero resolved findings exits 0", r.status, 0);
check("  and SAYS nothing was verified", /NOTHING TO VERIFY/.test(r.out), true);

// ── scope: only CodeRabbit's own findings, resolved by CodeRabbit ────────────
// A HUMAN-resolved thread is self-attestation and out of scope; counting it
// would make the gate demand tests for things nobody verified were fixed.
r = run(["1", "--threads", write("t-human.json", threadsDoc([{ id: "ccc333", resolvedBy: "mathiasbourgoin" }])),
         "--files", noTests, "--escapes", emptyEsc]);
check("a HUMAN-resolved thread is out of scope (exit 0)", r.status, 0);

r = run(["1", "--threads", write("t-other.json", threadsDoc([{ id: "ddd444", author: "someone-else" }])),
         "--files", noTests, "--escapes", emptyEsc]);
check("a non-CodeRabbit thread is out of scope (exit 0)", r.status, 0);

// An UNRESOLVED CodeRabbit finding is not yet a fix, so it is not this gate's
// business — the review itself is still blocking.
r = run(["1", "--threads", write("t-unres.json", threadsDoc([{ id: "eee555", resolved: false }])),
         "--files", noTests, "--escapes", emptyEsc]);
check("an UNRESOLVED finding is out of scope (exit 0)", r.status, 0);

// ── usage ────────────────────────────────────────────────────────────────────
check("no PR number is a usage error", run([]).status, 2);
check("--threads without --files is a usage error",
      run(["1", "--threads", oneFinding]).status, 2);

fs.rmSync(TMP, { recursive: true, force: true });
console.log(`\npassed: ${pass}   failed: ${fail}`);
process.exit(fail === 0 ? 0 : 1);
