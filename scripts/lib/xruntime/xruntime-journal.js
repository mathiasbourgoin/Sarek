// scripts/lib/xruntime-journal.js — CommonJS.
//
// Journal + persisted-state I/O for scripts/xruntime-review.js
// (FR-095..098, Amendment D-2, D-8; specs/review-v2-corrections.md
// INV-4/E-5/E-8). The journal is append-forever per task
// (briefs/<task>-xruntime.jsonl); review.json state is read-only here — the
// helper reads only the PERSISTED briefs/<task>-review.json, never a
// `.draft` file (V-1).
"use strict";

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const SLUG_RE = /^[a-z0-9-]+$/;

function validSlug(task) {
  return typeof task === "string" && SLUG_RE.test(task);
}

function journalPath(root, task) {
  return path.resolve(root, "briefs", `${task}-xruntime.jsonl`);
}

function reviewJsonPath(root, task) {
  return path.resolve(root, "briefs", `${task}-review.json`);
}

// D-8: warn (stderr) when briefs/ is not actually git-ignored in this repo —
// a consumer repo without the ignore would see the append-forever journal as
// a standing untracked file visible to the scope gate. Non-blocking.
function warnIfBriefsNotIgnored(root) {
  try {
    execSync("git check-ignore briefs/", { cwd: root, stdio: "ignore" });
  } catch (e) {
    process.stderr.write(
      "xruntime-review: warning — briefs/ is not git-ignored in this repo; the append-forever " +
        "journal will be a standing untracked file visible to the scope gate (D-8).\n"
    );
  }
}

// FR-095/096: append exactly one JSON line per invocation, after — never
// during — the wrapper subprocess. Journal-append failure is reported to the
// caller (never silently healthy, FR-096).
function appendJournalLine(root, task, entry) {
  const line = JSON.stringify(entry);
  try {
    fs.mkdirSync(path.resolve(root, "briefs"), { recursive: true });
    fs.appendFileSync(journalPath(root, task), line + "\n");
  } catch (e) {
    return { ok: false, error: e.message, line };
  }
  return { ok: true, line };
}

// E-8: reads the persisted review.json, distinguishing three states instead
// of collapsing "absent" and "malformed" into one silent `null`. INV-4: a
// malformed verdict is explicit degraded-input — the caller (xruntime-review.js)
// fails closed on "malformed" rather than treating it as "no prior state".
//   - "absent": no file — legitimately no prior cycle yet.
//   - "malformed": file exists but is not parseable JSON — unverifiable
//     round state, never silently treated as fresh.
//   - "valid": parsed successfully; `value` carries the parsed object.
function readReviewJson(root, task) {
  const p = reviewJsonPath(root, task);
  if (!fs.existsSync(p)) return { state: "absent", value: null };
  try {
    return { state: "valid", value: JSON.parse(fs.readFileSync(p, "utf8")) };
  } catch (e) {
    return { state: "malformed", value: null };
  }
}

// INV-4/E-5: scans the append-forever journal for the LAST entry matching
// this exact (runtime, digest) pair — the crash-before-persist enforcement
// input. Absent/unreadable journal or no match -> null (caller treats that
// as "nothing to refuse against"). A malformed line is fail-closed: its
// runtime/digest cannot be authenticated, so it may be the very degradation
// record the breaker needs to enforce.
function readLatestJournalEntry(root, task, runtime, digest) {
  const p = journalPath(root, task);
  if (!fs.existsSync(p)) return null;
  let lines;
  try {
    lines = fs
      .readFileSync(p, "utf8")
      .trim()
      .split("\n")
      .filter(Boolean);
  } catch (e) {
    return null;
  }
  for (let i = lines.length - 1; i >= 0; i--) {
    let entry;
    try {
      entry = JSON.parse(lines[i]);
    } catch (e) {
      return { malformed: true, reason: "malformed-journal" };
    }
    if (entry.runtime === runtime && entry.digest === digest) {
      if (entry.outcome === "blocked" && entry.reason === "malformed-journal") {
        return { malformed: true, reason: "malformed-journal" };
      }
      return entry;
    }
  }
  return null;
}

// D-2/INV-4/E-5: refuses a repeat probe when EITHER of two independent
// enforcement inputs shows an unchanged-digest degraded state:
//   (1) the persisted review.json (legacy path, D-2 unchanged): a mid-cycle
//       NO-GO with this runtime's cross_runtime entry degraded at this exact
//       digest. A persisted GO or an absent file means a fresh cycle, so a
//       prior degraded state there is stale (O-2).
//   (2) the journal (INV-4, new): the LAST journal entry for this exact
//       (runtime, digest) degraded IN THE SAME CYCLE as the current
//       invocation — this is what makes a crash-before-persist refuse a
//       repeat probe even though no review.json was ever written to check
//       against. A journal entry from a PRIOR cycle is stale (deterministic
//       separation) and never refuses. `currentCycle` is supplied by the
//       caller (roster-review passes --cycle) — when unknown (null), this
//       branch never fires (nothing to prove "same cycle" against; falls
//       back to (1) only, preserving pre-E-5 behavior).
// `--human-retry` always bypasses both.
function shouldRefuseDegraded({ reviewJson, journalEntry, runtime, digest, humanRetry, currentCycle }) {
  if (humanRetry) return false;

  if (reviewJson && reviewJson.status === "NO-GO") {
    const entry = reviewJson.cross_runtime && reviewJson.cross_runtime[runtime];
    if (entry && entry.status === "degraded" && entry.config_digest === digest) return true;
  }

  if (
    journalEntry &&
    journalEntry.outcome === "degraded" &&
    currentCycle !== null &&
    currentCycle !== undefined &&
    journalEntry.cycle === currentCycle
  ) {
    return true;
  }

  return false;
}

module.exports = {
  validSlug,
  journalPath,
  reviewJsonPath,
  warnIfBriefsNotIgnored,
  appendJournalLine,
  readReviewJson,
  readLatestJournalEntry,
  shouldRefuseDegraded,
};
