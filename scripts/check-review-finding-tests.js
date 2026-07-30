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
 *   - a PAGED read that could not be completed  -> exit 2 (see below)
 *
 * PAGINATION, which was a live hole rather than a hypothetical one
 *
 * `reviewThreads(first:100)` with no cursor loop silently truncates: a finding
 * that lives on page 2 is invisible, and the gate then prints NOTHING TO VERIFY
 * or a clean pass while a real unexcused finding sits there. That is precisely
 * the "verified nothing that reads as verified something" shape the FAIL-CLOSED
 * list above exists to forbid, so this now follows pageInfo.hasNextPage /
 * endCursor with `after:` until the connection is exhausted, and refuses (exit 2)
 * if the walk cannot COMPLETE — a page that errors, a missing or malformed
 * pageInfo, hasNextPage:true with no endCursor to follow, or the page budget
 * being hit. A partial read must never be able to look like a clean one.
 *
 * Measured, not assumed, on the same axis for the changed-file list:
 * `gh pr view <pr> --json files` returned exactly 100 paths for PR #388, whose
 * real diff is 294 files; `gh api .../pulls/<pr>/files --paginate` returned all
 * 294. The truncated list is the DANGEROUS direction here — dropping the one
 * test path in the tail turns an accepted PR into a REFUSED one, but worse, it
 * makes the verdict depend on where in the diff a file happens to sort. So the
 * file list comes from the paginated REST endpoint.
 *
 * HOW A FIXTURE EXPRESSES PAGES (offline `--threads FILE`)
 *
 * It does not, deliberately. A fixture is ONE GraphQL document, i.e. one page,
 * and the offline path cannot fetch a second one — there is nothing to fetch
 * from. So: the fixture must carry a pageInfo (a real response always does; its
 * absence means the document is hand-rolled or the schema moved, and either way
 * nothing can be concluded), and if that pageInfo says hasNextPage:true the
 * offline path refuses with exit 2 rather than reporting on the visible page.
 * Multi-page ACCUMULATION is therefore tested through the online path with a
 * stub `gh` on PATH, not through the fixture.
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

/* 100 threads per page. The budget exists so a cursor that never advances ends
   as a REFUSAL rather than as an unbounded loop; 50 pages is 5000 threads, far
   past any real review, so hitting it means something is wrong, not that the PR
   is unusually chatty. */
const THREADS_PER_PAGE = 100;
const MAX_THREAD_PAGES = 50;

/** Refuse, naming the page, because a partial read is a read of nothing. */
function refusePartial(msg) {
  console.error(
    `ERROR: the review-thread pagination could not be completed, so NOTHING was ` +
      `verified: ${msg}`
  );
  console.error(
    "A partially-read thread list is not a clean one: a resolved finding on an " +
      "unread page would be invisible here and the gate would pass. Refusing to " +
      "report a verdict."
  );
  process.exit(2);
}

/** One page of the reviewThreads connection, as a parsed document. */
function fetchThreadPage(opt, after) {
  const [owner, name] = opt.repo.split("/");
  if (!owner || !name) usage(`--repo must be owner/name, got ${opt.repo}`);
  const cursor = after ? `,after:"${after}"` : "";
  // comments(first:1) is INTENTIONAL and is not the same bug as the thread
  // connection was: only the FIRST comment of a thread is CodeRabbit's finding
  // body, which is where the author login and the cr-comment id live. Replies
  // are human discussion and are never read, so there is nothing on a second
  // page of comments for this gate to miss. Do not "fix" it into a cursor loop.
  const q =
    `{repository(owner:"${owner}",name:"${name}"){pullRequest(number:${opt.pr})` +
    `{reviewThreads(first:${THREADS_PER_PAGE}${cursor})` +
    `{pageInfo{hasNextPage endCursor}` +
    `nodes{isResolved path resolvedBy{login} comments(first:1){nodes{author{login} body}}}}}}}`;
  let raw;
  try {
    raw = execFileSync("gh", ["api", "graphql", "-f", `query=${q}`], {
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (e) {
    if (after) refusePartial(`the query for the page after ${after} failed: ${e.stderr || e.message}`);
    console.error(
      `ERROR: the GraphQL query failed, so NOTHING was verified: ${
        e.stderr || e.message
      }`
    );
    process.exit(2);
  }
  return parseThreadDoc(raw);
}

function parseThreadDoc(raw) {
  try {
    return JSON.parse(raw);
  } catch (e) {
    console.error(`ERROR: thread data is not JSON, so nothing was verified: ${e.message}`);
    process.exit(2);
  }
}

/**
 * Every reviewThreads node on the PR, accumulated across pages.
 *
 * The empty-input trap is applied PER PAGE: a page whose nodes are not an array
 * is an auth failure or a schema change, and must not read as "the connection
 * ended here".
 */
function fetchThreadNodes(opt) {
  const all = [];
  let after = null;
  for (let page = 1; page <= MAX_THREAD_PAGES; page++) {
    const doc = opt.threads
      ? parseThreadDoc(fs.readFileSync(opt.threads, "utf8"))
      : fetchThreadPage(opt, after);
    const conn = doc?.data?.repository?.pullRequest?.reviewThreads;
    const nodes = conn?.nodes;
    if (!Array.isArray(nodes)) {
      // The empty-input trap: an auth failure or a schema change gives a document
      // with no nodes array, which must not read as "no findings".
      console.error(
        "ERROR: no reviewThreads array in the response — auth failure, wrong PR, " +
          "or a schema change. Refusing to report a verdict." +
          (page > 1 ? ` (on page ${page})` : "")
      );
      process.exit(2);
    }
    all.push(...nodes);

    const pi = conn.pageInfo;
    if (!pi || typeof pi !== "object" || typeof pi.hasNextPage !== "boolean")
      refusePartial(
        `page ${page} carries no usable pageInfo{hasNextPage endCursor}. Without it ` +
          `there is no way to tell a complete connection from a truncated one` +
          (opt.threads ? ", and a hand-rolled fixture must not be trusted to be complete" : "")
      );
    if (!pi.hasNextPage) return all;
    if (opt.threads)
      refusePartial(
        `the offline fixture declares hasNextPage:true, and the offline path has ` +
          `nowhere to fetch page 2 from — a fixture is one GraphQL document, i.e. ` +
          `one page. Use a fixture whose connection is complete, or exercise ` +
          `multi-page accumulation through the live path`
      );
    if (typeof pi.endCursor !== "string" || pi.endCursor === "")
      refusePartial(
        `page ${page} says hasNextPage:true but gives no endCursor to follow, so ` +
          `the remaining threads are unreachable`
      );
    if (pi.endCursor === after)
      refusePartial(
        `page ${page} returned the same endCursor as the previous page (${after}), ` +
          `so the walk is not advancing`
      );
    after = pi.endCursor;
  }
  refusePartial(
    `the page budget of ${MAX_THREAD_PAGES} pages (${
      MAX_THREAD_PAGES * THREADS_PER_PAGE
    } threads) was exhausted and the connection still reports more`
  );
}

/**
 * Resolved CodeRabbit findings, split by WHO resolved them. Fails closed on no
 * data. Returns {findings, humanResolved}, both [{id, path}] (humanResolved
 * entries also carry resolvedBy).
 *
 * `humanResolved` is returned rather than dropped, and that is the whole point
 * of the split. Both lists are out of this gate's enforcement scope in the same
 * way — only CodeRabbit-resolved threads are third-party attestation that the
 * fix landed — but a dropped list made the gate MISREPORT the PR: with one
 * CodeRabbit finding that the author had resolved, it printed "0 resolved
 * CodeRabbit finding(s)" and "no resolved CodeRabbit findings on this PR", and
 * both are false. There was one. A scope decision has to read as a scope
 * decision; stated as a count of zero it reads as an all-clear, and the actor
 * who reaches this branch is exactly the actor the gate constrains — an author
 * resolving a CodeRabbit thread is routine, not exotic.
 */
function fetchThreads(opt) {
  const nodes = fetchThreadNodes(opt);
  const findings = [];
  const humanResolved = [];
  for (const n of nodes) {
    if (!n || n.isResolved !== true) continue;
    const c = n.comments?.nodes?.[0];
    const author = c?.author?.login || "";
    // Not a CodeRabbit finding at all: never this gate's subject, and not
    // reported, because there is nothing about the review to misstate.
    if (!/^coderabbitai/i.test(author)) continue;
    const m = /cr-comment:v1:([0-9a-f]+)/.exec(c?.body || "");
    const rec = { id: m ? m[1] : null, path: n.path || "(no path)" };
    // A CodeRabbit finding resolved by anyone but CodeRabbit is
    // self-attestation: out of scope to ENFORCE, but counted and named so the
    // clean line cannot be read as "there were none".
    const resolver = n.resolvedBy?.login || "";
    if (!/^coderabbitai/i.test(resolver)) {
      humanResolved.push({ ...rec, resolvedBy: resolver || "(unknown)" });
      continue;
    }
    findings.push(rec);
  }
  return { findings, humanResolved };
}

function fetchFiles(opt) {
  if (opt.files) {
    const raw = fs.readFileSync(opt.files, "utf8");
    return raw.split("\n").map((s) => s.trim()).filter(Boolean);
  }
  // NOT `gh pr view --json files`: that connection is capped at 100 and gives no
  // signal that it truncated. MEASURED on PR #388 (294 changed files) —
  // `gh pr view 388 --json files --jq '.files[].path'` printed 100 paths, while
  // `gh api .../pulls/388/files --paginate --jq '.[].filename'` printed all 294,
  // and the 294 are a strict superset of the 100. --paginate walks the Link header
  // to exhaustion and exits nonzero if any page fails, which is caught below, so a
  // short read cannot be mistaken for the whole diff.
  //
  // `.filename`, not `.path`: the REST representation of a changed file names the
  // field `filename` (only the GraphQL/`gh pr view` shape calls it `path`). Worth
  // stating because getting it wrong is SILENT in the counting sense — `.[].path`
  // over the REST payload yields one empty line per file, so a naive `wc -l` still
  // reports 294 while every path is blank. What caught it was the zero-files trap
  // below, which is the reason that trap is there.
  try {
    const [owner, name] = opt.repo.split("/");
    if (!owner || !name) usage(`--repo must be owner/name, got ${opt.repo}`);
    const out = execFileSync(
      "gh",
      [
        "api",
        `repos/${owner}/${name}/pulls/${opt.pr}/files`,
        "--paginate",
        "--jq",
        ".[].filename",
      ],
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
  const { findings, humanResolved } = fetchThreads(opt);
  const files = fetchFiles(opt);
  const escapes = readEscapes(opt.escapes);
  const testPaths = files.filter((f) => IS_TEST_PATH.test(f));

  console.log(`PR #${opt.pr}: ${findings.length} resolved CodeRabbit finding(s) in scope, ` +
              `${humanResolved.length} human-resolved (out of scope), ` +
              `${files.length} changed file(s), ${testPaths.length} test path(s)`);

  if (findings.length === 0) {
    // Says so explicitly: a clean line here means "nothing to check", NOT
    // "the findings were checked and carried tests". And the two reasons a PR
    // can have nothing to check are NOT the same fact, so they do not share a
    // sentence — see fetchThreads. Distinguishing them is asserted in
    // check-review-finding-tests.test.js, not left to this comment.
    if (humanResolved.length > 0) {
      console.log(
        `NOTHING TO VERIFY: no CodeRabbit-resolved findings, but ` +
          `${humanResolved.length} CodeRabbit finding(s) on this PR were resolved by a ` +
          `human and are DELIBERATELY NOT CHECKED (self-attestation, not ` +
          `third-party confirmation the fix landed):`
      );
      for (const f of humanResolved) {
        console.log(
          `  ${f.id || "(no cr-comment id)"}  ${f.path} — resolved by ${f.resolvedBy}`
        );
      }
    } else {
      console.log("NOTHING TO VERIFY: no resolved CodeRabbit findings on this PR.");
    }
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
