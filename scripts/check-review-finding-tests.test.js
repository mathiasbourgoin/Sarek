#!/usr/bin/env node
"use strict";
/**
 * Covering test for check-review-finding-tests.js (backlog-164).
 *
 * No network and no GitHub token, throughout. Most cases go through the subject's
 * own --threads/--files fixture path; the PAGINATION cases cannot, because a
 * fixture is one GraphQL document and paging is by definition about the second
 * one, so those drive the live cursor loop with a stub `gh` on PATH (see stubGh).
 * What is pinned here is the DECISION table, including every fail-closed branch —
 * because "verified nothing" is the failure mode this gate is most likely to have,
 * and a truncated read that looks clean is exactly that failure.
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

/**
 * A GraphQL-shaped document with the given threads.
 *
 * `pageInfo` defaults to a COMPLETE connection. It is present by default because
 * the subject requires it: a real response always carries it, so its absence
 * means the document is hand-rolled or the schema moved, and in neither case can
 * completeness be concluded. Pass `pageInfo: null` to omit it entirely, or an
 * explicit object to declare truncation.
 */
function threadsDoc(threads, pageInfo = { hasNextPage: false, endCursor: null }) {
  return JSON.stringify({
    data: {
      repository: {
        pullRequest: {
          reviewThreads: {
            ...(pageInfo === null ? {} : { pageInfo }),
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

function run(args, extraPath) {
  const env = extraPath
    ? { ...process.env, PATH: `${extraPath}${path.delimiter}${process.env.PATH}` }
    : process.env;
  const r = spawnSync("node", [SUBJECT, ...args], { encoding: "utf8", env });
  return { status: r.status, out: (r.stdout || "") + (r.stderr || "") };
}

/**
 * A directory holding a stub `gh` that the subject will find on PATH.
 *
 * Multi-page accumulation cannot be expressed as a fixture — a fixture is ONE
 * GraphQL document, i.e. one page, and the offline path has nowhere to fetch a
 * second from (which is itself a case below). So the paging cases drive the LIVE
 * code path with a fake `gh`: same execFileSync, same cursor loop, no network.
 * `page2` is the shell body used when the query carries `after:`, so a test can
 * make page 2 return threads, return junk, or fail outright.
 */
function stubGh(name, { page1, page2, files }) {
  const dir = path.join(TMP, name);
  fs.mkdirSync(dir, { recursive: true });
  // A JSON document contains no single quote, so single-quoting it is safe and
  // keeps the whole branch on one line — a heredoc cannot be used inside a `case`
  // arm without its terminator colliding with the `;;`.
  const emit = (body) =>
    body.startsWith("!")
      ? body.slice(1) // a raw shell fragment: let the stub fail or misbehave
      : `printf '%s' '${body}'`;
  fs.writeFileSync(
    path.join(dir, "gh"),
    `#!/bin/sh
case "$*" in
  *graphql*)
    case "$*" in
      *'after:'*) ${emit(page2)} ;;
      *) ${emit(page1)} ;;
    esac
    ;;
  *) printf '%s\\n' ${files.map((f) => `'${f}'`).join(" ")} ;;
esac
`,
    { mode: 0o755 }
  );
  return dir;
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
// The genuinely-empty case must NOT borrow the human-resolved wording, or the
// two branches below stop being distinguishable in the direction that matters.
check("  and does not claim a human resolved anything", /resolved by a\s+human/.test(r.out), false);
check("  and reports zero out-of-scope", /0 human-resolved/.test(r.out), true);

// ── scope: only CodeRabbit's own findings, resolved by CodeRabbit ────────────
// A HUMAN-resolved thread is self-attestation and out of scope to ENFORCE;
// counting it as in-scope would make the gate demand tests for things nobody
// verified were fixed.
//
// But the exit code is not the whole claim. This branch used to print "0
// resolved CodeRabbit finding(s)" and "no resolved CodeRabbit findings on this
// PR" while one existed and the author had resolved it -- a false statement
// about the PR, reachable by exactly the actor the gate constrains. The
// assertions below are on the OUTPUT, because a scope decision that reads as an
// all-clear is the defect, and status 0 is correct in both branches and so
// cannot tell them apart.
r = run(["1", "--threads", write("t-human.json", threadsDoc([{ id: "ccc333", resolvedBy: "mathiasbourgoin" }])),
         "--files", noTests, "--escapes", emptyEsc]);
check("a HUMAN-resolved thread is out of scope (exit 0)", r.status, 0);
check("  and is NOT reported as zero findings present",
      /0 resolved CodeRabbit finding\(s\), /.test(r.out), false);
check("  and is counted as out-of-scope rather than dropped",
      /1 human-resolved \(out of scope\)/.test(r.out), true);
check("  and the clean line says a human resolved it, not that there were none",
      /resolved by a\s+human/.test(r.out), true);
check("  and does not print the no-findings-at-all wording",
      /no resolved CodeRabbit findings on this PR/.test(r.out), false);
check("  and says the omission is deliberate", /DELIBERATELY NOT CHECKED/.test(r.out), true);
check("  and NAMES the finding it is not checking", /ccc333/.test(r.out), true);
check("  and names who resolved it", /mathiasbourgoin/.test(r.out), true);

r = run(["1", "--threads", write("t-other.json", threadsDoc([{ id: "ddd444", author: "someone-else" }])),
         "--files", noTests, "--escapes", emptyEsc]);
check("a non-CodeRabbit thread is out of scope (exit 0)", r.status, 0);
// A thread that is not a CodeRabbit finding is not under-reported by being
// absent -- there is no review claim to misstate -- so it must NOT be counted
// into the human-resolved tally, or that tally stops meaning what it says.
check("  and is not counted as a human-resolved CodeRabbit finding",
      /0 human-resolved/.test(r.out), true);

// An UNRESOLVED CodeRabbit finding is not yet a fix, so it is not this gate's
// business — the review itself is still blocking.
r = run(["1", "--threads", write("t-unres.json", threadsDoc([{ id: "eee555", resolved: false }])),
         "--files", noTests, "--escapes", emptyEsc]);
check("an UNRESOLVED finding is out of scope (exit 0)", r.status, 0);

// ── pagination ───────────────────────────────────────────────────────────────
// The hole this section exists for: reviewThreads(first:100) with no cursor loop
// silently truncates, so a resolved finding on page 2 was invisible and the gate
// printed NOTHING TO VERIFY over it. Every case here fails against the
// single-page version.
const NO_TEST_FILES = ["src/a.ml", "README.md"];

// Accumulation. The ONLY finding lives on page 2; page 1 is a full page of
// out-of-scope threads. If the walk stops at page 1 the subject sees zero
// findings and exits 0 saying NOTHING TO VERIFY, so this case is the direct
// negation of the bug.
const pageOneFiller = Array.from({ length: 100 }, (_, i) => ({
  id: `fa11e${i}`,
  resolved: false,
}));
r = run(
  ["1", "--escapes", emptyEsc],
  stubGh("s-accum", {
    page1: threadsDoc(pageOneFiller, { hasNextPage: true, endCursor: "CUR1" }),
    page2: threadsDoc([{ id: "2f00da" }]),
    files: NO_TEST_FILES,
  })
);
checkRefused("a finding found only on PAGE 2 is seen", r, "2f00da");
check("  and the walk did not stop at page 1", /NOTHING TO VERIFY/.test(r.out), false);

// The same shape, but the page-2 finding is the one that must be EXCUSED. A gate
// that only ever refuses harder on more data is not the same as one that reads
// the data.
r = run(
  ["1", "--escapes", write("e-p2.tsv", "2f00da\ta doc-only correction\tmathias\n")],
  stubGh("s-accum2", {
    page1: threadsDoc(pageOneFiller, { hasNextPage: true, endCursor: "CUR1" }),
    page2: threadsDoc([{ id: "2f00da" }]),
    files: NO_TEST_FILES,
  })
);
check("  a page-2 finding can be excused by its recorded escape", r.status, 0);
check("  and the page-2 id is what the output names", /2f00da/.test(r.out), true);

// Fail closed on a walk that cannot COMPLETE. Each of these would otherwise be
// reported as a verdict over a partial read.
// `says` is per-case rather than one shared regex: a shared "NOTHING was
// verified" assertion would be wider than any single branch and would keep
// passing if two distinct causes collapsed into one message.
const partialCases = {
  "a page-2 query that FAILS": {
    page2: "!echo 'gh: boom' >&2; exit 1",
    says: /the page after CUR1 failed/,
  },
  "hasNextPage:true with a NULL endCursor": {
    page1: threadsDoc(pageOneFiller, { hasNextPage: true, endCursor: null }),
    says: /no endCursor to follow/,
  },
  "hasNextPage:true with an absent endCursor": {
    page1: threadsDoc(pageOneFiller, { hasNextPage: true }),
    says: /no endCursor to follow/,
  },
  // Takes the pre-existing empty-input trap rather than the pagination refusal,
  // and must name the page — otherwise "the connection ended here" and "page 2
  // came back unusable" read the same.
  "a page 2 with no nodes array": { page2: "{}", says: /on page 2/ },
  "a cursor that does not advance": {
    page2: threadsDoc([{ id: "2f00da" }], { hasNextPage: true, endCursor: "CUR1" }),
    says: /not advancing/,
  },
};
for (const [desc, over] of Object.entries(partialCases)) {
  r = run(
    ["1", "--escapes", emptyEsc],
    stubGh(`s-${desc.replace(/[^a-z0-9]/gi, "_")}`, {
      page1: threadsDoc(pageOneFiller, { hasNextPage: true, endCursor: "CUR1" }),
      page2: threadsDoc([{ id: "2f00da" }]),
      files: NO_TEST_FILES,
      ...over,
    })
  );
  check(`incomplete pagination fails closed: ${desc} is exit 2`, r.status, 2);
  check(
    `  and says WHICH incompleteness: ${desc}`,
    over.says.test(r.out),
    true
  );
}

// A page budget must end as a refusal, not as an unbounded loop or a verdict.
// Every page advances its cursor and every page claims another, so the only way
// out is the budget.
r = run(
  ["1", "--escapes", emptyEsc],
  stubGh("s-budget", {
    page1: threadsDoc(pageOneFiller, { hasNextPage: true, endCursor: "CUR1" }),
    // A shell fragment: a fresh cursor each call, so the walk never terminates
    // on its own and never repeats a cursor either.
    page2:
      "!n=$(date +%s%N); printf '%s' '" +
      threadsDoc([], { hasNextPage: true, endCursor: "__CUR__" }).replace(
        "__CUR__",
        "'\"$n\"'"
      ) +
      "'",
    files: NO_TEST_FILES,
  })
);
check("the page budget is a refusal, not a loop or a verdict", r.status, 2);
check("  and names the budget", /page budget/.test(r.out), true);

// The zero-files trap on the LIVE path. Untested until now, and it is what caught
// a real defect while this section was being written: the REST payload names the
// field `filename`, so `--jq '.[].path'` returned one BLANK line per file and a
// `wc -l` sanity check still said 294. Blank paths are indistinguishable from a
// dropped list, and the only thing standing between that and a confident verdict
// over an empty diff is this trap.
r = run(
  ["1", "--escapes", emptyEsc],
  stubGh("s-nofiles", {
    page1: threadsDoc([{ id: "aaa111" }]),
    page2: "!exit 1",
    files: [],
  })
);
check("a live file list that comes back EMPTY is exit 2, not a verdict", r.status, 2);
check("  and refuses to verify against nothing", /against nothing/.test(r.out), true);

// The offline path, whose honest answer is "I cannot".
r = run([
  "1",
  "--threads",
  write("t-truncated.json", threadsDoc([{ id: "aaa111" }], { hasNextPage: true, endCursor: "CUR1" })),
  "--files",
  noTests,
  "--escapes",
  emptyEsc,
]);
check("a fixture declaring hasNextPage:true is exit 2, not a verdict on page 1", r.status, 2);
check("  and says why the offline path cannot continue", /nowhere to fetch page 2/.test(r.out), true);

// A document with no pageInfo at all cannot be told apart from a truncated one,
// so it is not accepted either.
r = run([
  "1",
  "--threads",
  write("t-nopageinfo.json", threadsDoc([{ id: "aaa111" }], null)),
  "--files",
  noTests,
  "--escapes",
  emptyEsc,
]);
check("a document with NO pageInfo is exit 2", r.status, 2);
check("  and says pageInfo is what is missing", /pageInfo/.test(r.out), true);

// ── usage ────────────────────────────────────────────────────────────────────
check("no PR number is a usage error", run([]).status, 2);
check("--threads without --files is a usage error",
      run(["1", "--threads", oneFinding]).status, 2);

fs.rmSync(TMP, { recursive: true, force: true });
console.log(`\npassed: ${pass}   failed: ${fail}`);
process.exit(fail === 0 ? 0 : 1);
