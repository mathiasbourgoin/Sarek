#!/usr/bin/env node
"use strict";
/**
 * A RESOLVED CodeRabbit finding must arrive with a test, or a recorded escape
 * (backlog-164).
 *
 * WHY THIS EXISTS
 *
 * CodeRabbit found something real on six consecutive pull requests in one day
 * whose CI was fully green. Each was fixed, and whether a regression test came
 * with the fix was decided by whoever happened to be writing it — a habit, not a
 * gate. A habit that has to hold on the twentieth tired iteration is the thing
 * this repo replaces with a check.
 *
 * WHAT MAKES IT CHECKABLE
 *
 * Measured, not assumed (the API question was the reason this item sat open):
 *   - `isResolved` IS readable, via GraphQL only — REST does not expose it.
 *   - `resolvedBy.login` is `coderabbitai[bot]` on every resolved thread
 *     inspected, i.e. resolution is THIRD-PARTY VERIFICATION that the fix
 *     landed, not the author's own say-so. That is what makes it a usable
 *     trigger; if resolution were self-attested it would be worthless here.
 *   - resolution is NOT a side effect of outdating: threads exist that are
 *     resolved-and-not-outdated and unresolved-and-not-outdated.
 *   - each finding body carries a stable `cr-comment:v1:<hex>` id, which is the
 *     escape key. Line numbers are NOT usable — `line` comes back null.
 *
 * THE CEILING, STATED RATHER THAN IMPLIED
 *
 * This checks that the pull request TOUCHED A TEST PATH. That is weaker than
 * "a test that fails without the fix", and deliberately so: the strong form is
 * prove-red, which needs a per-finding red command and is far more expensive.
 * Touching a comment in a test file satisfies this check. So it is a floor
 * against the fix-with-no-test-at-all case, not proof of coverage — and it must
 * not be described in a commit message or a report as though it were.
 *
 * FAIL-CLOSED, because the failure modes here are all "verified nothing":
 *   - API returned no data / auth failed        -> exit 2, never 0
 *   - resolved findings exist and no test path  -> exit 1 unless every finding
 *                                                  id has an escape
 *   - zero resolved findings                    -> exit 0, but SAYS so, so a
 *                                                  clean line cannot be read as
 *                                                  "the findings were checked"
 *
 * Usage:
 *   node scripts/check-review-finding-tests.js <pr-number> [--repo owner/name]
 *        [--escapes scripts/review-finding-escapes.tsv]
 *        [--threads FILE --files FILE]   (offline: pre-fetched JSON, for tests)
 *
 * Escape file format (tab-separated, `#` comments allowed):
 *   <cr-comment-id>\t<reason>\t<by>
 * All three fields are mandatory and non-blank — an escape without a reason or
 * an owner is the "silent skip" this repo already refuses elsewhere.
 */

const fs = require("fs");
const { execFileSync } = require("child_process");

const TEST_PATH_RE =
  /(^|\/)tests?\//i.source +
  "|" +
  /(^|\/)test_[^/]*$/i.source +
  "|" +
  /[._-]test\.[a-z]+$/i.source +
  "|" +
  /\.test\.[a-z]+$/i.source +
  "|" +
  /(^|\/)prove-red-fixtures\//i.source;
const IS_TEST_PATH = new RegExp(TEST_PATH_RE, "i");

function usage(msg) {
  console.error(`ERROR: ${msg}`);
  console.error(
    "usage: check-review-finding-tests.js <pr> [--repo o/n] [--escapes FILE] " +
      "[--threads FILE --files FILE]"
  );
  process.exit(2);
}

function parseArgs(argv) {
  const o = {
    pr: null,
    repo: "mathiasbourgoin/Sarek",
    escapes: "scripts/review-finding-escapes.tsv",
    threads: null,
    files: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--repo") o.repo = argv[++i];
    else if (a === "--escapes") o.escapes = argv[++i];
    else if (a === "--threads") o.threads = argv[++i];
    else if (a === "--files") o.files = argv[++i];
    else if (/^\d+$/.test(a)) o.pr = a;
    else usage(`unknown argument: ${a}`);
  }
  if (!o.pr) usage("a PR number is required");
  if ((o.threads && !o.files) || (o.files && !o.threads))
    usage("--threads and --files must be given together");
  return o;
}

/** Resolved CodeRabbit findings, as [{id, path}]. Fails closed on no data. */
function fetchThreads(opt) {
  let raw;
  if (opt.threads) {
    raw = fs.readFileSync(opt.threads, "utf8");
  } else {
    const [owner, name] = opt.repo.split("/");
    if (!owner || !name) usage(`--repo must be owner/name, got ${opt.repo}`);
    const q = `{repository(owner:"${owner}",name:"${name}"){pullRequest(number:${opt.pr}){reviewThreads(first:100){nodes{isResolved path resolvedBy{login} comments(first:1){nodes{author{login} body}}}}}}}`;
    try {
      raw = execFileSync("gh", ["api", "graphql", "-f", `query=${q}`], {
        encoding: "utf8",
        maxBuffer: 64 * 1024 * 1024,
      });
    } catch (e) {
      console.error(
        `ERROR: the GraphQL query failed, so NOTHING was verified: ${
          e.stderr || e.message
        }`
      );
      process.exit(2);
    }
  }
  let doc;
  try {
    doc = JSON.parse(raw);
  } catch (e) {
    console.error(`ERROR: thread data is not JSON, so nothing was verified: ${e.message}`);
    process.exit(2);
  }
  const nodes = doc?.data?.repository?.pullRequest?.reviewThreads?.nodes;
  if (!Array.isArray(nodes)) {
    // The empty-input trap: an auth failure or a schema change gives a document
    // with no nodes array, which must not read as "no findings".
    console.error(
      "ERROR: no reviewThreads array in the response — auth failure, wrong PR, " +
        "or a schema change. Refusing to report a verdict."
    );
    process.exit(2);
  }
  const findings = [];
  for (const n of nodes) {
    if (!n || n.isResolved !== true) continue;
    const c = n.comments?.nodes?.[0];
    const author = c?.author?.login || "";
    // Only CodeRabbit's own findings, resolved by CodeRabbit itself. A thread a
    // human resolved is self-attestation and is out of scope for this gate.
    if (!/^coderabbitai/i.test(author)) continue;
    if (!/^coderabbitai/i.test(n.resolvedBy?.login || "")) continue;
    const m = /cr-comment:v1:([0-9a-f]+)/.exec(c?.body || "");
    findings.push({ id: m ? m[1] : null, path: n.path || "(no path)" });
  }
  return findings;
}

function fetchFiles(opt) {
  if (opt.files) {
    const raw = fs.readFileSync(opt.files, "utf8");
    return raw.split("\n").map((s) => s.trim()).filter(Boolean);
  }
  try {
    const out = execFileSync(
      "gh",
      ["pr", "view", opt.pr, "--repo", opt.repo, "--json", "files", "--jq", ".files[].path"],
      { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }
    );
    const files = out.split("\n").map((s) => s.trim()).filter(Boolean);
    if (files.length === 0) {
      console.error("ERROR: the PR reports zero changed files — refusing to verify against nothing.");
      process.exit(2);
    }
    return files;
  } catch (e) {
    console.error(`ERROR: could not list changed files: ${e.stderr || e.message}`);
    process.exit(2);
  }
}

/** id -> {reason, by}. Rejects a malformed or blank-field entry loudly. */
function readEscapes(file) {
  const map = new Map();
  if (!fs.existsSync(file)) return map;
  const lines = fs.readFileSync(file, "utf8").split("\n");
  lines.forEach((line, i) => {
    const s = line.trim();
    if (!s || s.startsWith("#")) return;
    const parts = line.split("\t");
    if (parts.length !== 3) {
      console.error(
        `ERROR: ${file}:${i + 1} has ${parts.length} tab-separated field(s), expected 3 ` +
          `(<id>\\t<reason>\\t<by>). A half-written escape must not silently excuse a finding.`
      );
      process.exit(2);
    }
    const [id, reason, by] = parts.map((p) => p.trim());
    if (!id || !reason || !by) {
      console.error(
        `ERROR: ${file}:${i + 1} has a blank field. An escape needs an id, a REASON and an OWNER — ` +
          `a blank reason is the silent skip this check exists to prevent.`
      );
      process.exit(2);
    }
    map.set(id, { reason, by });
  });
  return map;
}

function main() {
  const opt = parseArgs(process.argv.slice(2));
  const findings = fetchThreads(opt);
  const files = fetchFiles(opt);
  const escapes = readEscapes(opt.escapes);
  const testPaths = files.filter((f) => IS_TEST_PATH.test(f));

  console.log(`PR #${opt.pr}: ${findings.length} resolved CodeRabbit finding(s), ` +
              `${files.length} changed file(s), ${testPaths.length} test path(s)`);

  if (findings.length === 0) {
    // Says so explicitly: a clean line here means "nothing to check", NOT
    // "the findings were checked and carried tests".
    console.log("NOTHING TO VERIFY: no resolved CodeRabbit findings on this PR.");
    process.exit(0);
  }

  if (testPaths.length > 0) {
    console.log(`OK: the fix touches test path(s): ${testPaths.join(", ")}`);
    console.log(
      "NOTE (ceiling): this asserts a test path was TOUCHED, not that a test " +
        "fails without the fix. The strong form is prove-red."
    );
    process.exit(0);
  }

  // No test path. Every finding now needs its own recorded escape.
  const unexcused = findings.filter((f) => !f.id || !escapes.has(f.id));
  if (unexcused.length === 0) {
    console.log("OK: no test path, but every finding has a recorded escape:");
    for (const f of findings) {
      const e = escapes.get(f.id);
      console.log(`  ${f.id} (${f.path}) — ${e.reason} [by ${e.by}]`);
    }
    process.exit(0);
  }

  console.error(
    `\nREFUSED: ${unexcused.length} resolved finding(s) with no test path in the ` +
      `PR and no recorded escape:`
  );
  for (const f of unexcused) {
    console.error(`  ${f.id || "(no cr-comment id)"}  ${f.path}`);
  }
  console.error(
    `\nEither add a test, or record why one is not warranted in ${opt.escapes}:\n` +
      `  <cr-comment-id>\\t<reason>\\t<your-name>\n` +
      `Legitimate reasons exist (a doc correction; a finding you REFUTED rather ` +
      `than fixed) — they just have to be written down rather than assumed.`
  );
  process.exit(1);
}

main();
