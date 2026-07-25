// scripts/lib/review-verdict-rules.js — CommonJS, repo-local.
//
// The rule + validation layer of scripts/review-verdict-assemble.js, split out
// so each file stays under the repo's 500-line limit (the same split
// scripts/lib/review/review-convergence-rules.js made out of
// check-review-convergence.js). Contract: schema/review-json-schema.md.
//
// Responsibility boundary: this module owns WHAT a valid verdict is — the
// closed enums, the lifecycle-continuity rules, the rounds_audit append-only
// rules, the normalizer-disposition application, and envelope construction.
// review-verdict-assemble.js keeps CLI parsing, file I/O, and orchestration.
//
// Repo-local like its caller: deliberately absent from
// scripts/review-bundle.manifest.json, so a bundle upgrade neither overwrites
// nor flags it.
//
// `fail()` exits the process (2) rather than throwing: every caller here is a
// fail-closed refusal on the way to writing a verdict, and a half-written
// verdict is the outcome this whole tool exists to prevent.
"use strict";

const { buildLedgerIndex, findLedgerEntry } = require("./review/normalize-rules");
const { validSlug } = require("./xruntime/xruntime-journal");
const TRACE_SCHEMA_VERSION = "1.0";
const STATUSES = new Set(["GO", "NO-GO"]);
// schema/review-json-schema.md §`mode`. The gate reads `mode` for exactly one
// thing — `mode === "full"` obligates a scope-gate trace line — so ANY other
// string, including a typo, silently drops that obligation.
const MODES = new Set(["express", "fast", "full"]);
// schema/review-json-schema.md §`no_go_reason`. Closed enums in the prose,
// validated by nothing (D2) — a typo routes nowhere. The gate never reads
// them, so this is the only place the typo can be caught.
const NO_GO_TYPES = new Set([
  "out-of-scope-change",
  "spec-ac-failure",
  "cross-runtime-finding",
  "design-not-converging",
  "review-integrity-failure",
]);
const NO_GO_CAUSES = new Set(["unencodable-finding", "unattested-invocation", "novel-finding-streak", "round-cap"]);
const HIGH_PLUS = new Set(["CRITICAL", "HIGH"]);

function fail(message) {
  process.stderr.write(`review-verdict-assemble: ${message}\n`);
  process.exit(2);
}

function warn(message) {
  process.stderr.write(`review-verdict-assemble: warning: ${message}\n`);
}

function parseCount(raw, flag) {
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0) fail(`${flag} must be a non-negative integer (got ${JSON.stringify(raw)})`);
  return n;
}

// ── phase 1 validation ───────────────────────────────────────────────────
function validateAssembleArgs(args) {
  if (!args.task) fail("--task <slug> is required");
  // Same validator the gate uses. An invalid slug is not cosmetic: the gate
  // hard-fails exit 2 on a trace-obligated round (and every round this tool
  // assembles is obligated), and `--task` also composes the default --prior /
  // --out paths, so `../x` would write outside briefs/.
  if (!validSlug(args.task)) {
    fail(`--task ${JSON.stringify(args.task)} is not a valid slug (validSlug(), scripts/lib/xruntime/xruntime-journal.js) — the gate exits 2 on a trace-obligated round with an unusable task, and the slug composes the briefs/ paths`);
  }
  if (args.round === undefined) {
    fail("--round <n> is required — a verdict without `round` makes check-review-convergence.js skip strike and rounds_audit checks entirely (B-8), so the gate passes vacuously. That refusal is the whole point of this tool.");
  }
  if (args.cycle === undefined) fail("--cycle <n> is required (the trace mechanism scopes lines by (cycle, round))");
  if (!STATUSES.has(args.status)) fail(`--status must be one of ${[...STATUSES].join(" | ")}`);
  if (!args.reviewedSha) fail("--reviewed-sha <sha> is required (rounds_audit completeness, FR-078)");
  if (!args.fixSha === !args.fixShaReason) {
    fail("exactly one of --fix-sha <sha> or --fix-sha-reason <text> is required (EC-8: a dirty tree records null + a reason, never a guessed sha)");
  }
  if (args.specialists.length === 0) {
    fail("at least one --specialist <name>=<selection_reason> is required — \"why did this specialist run this round\" must always be answerable");
  }
  if (args.status === "NO-GO" && !args.noGoType) fail("--no-go-type is required on a NO-GO verdict");
  if (args.status === "GO" && args.noGoType) fail("--no-go-type is meaningless on a GO verdict");
  if (args.noGoType && !NO_GO_TYPES.has(args.noGoType)) {
    fail(`--no-go-type must be one of ${[...NO_GO_TYPES].join(" | ")} (got ${JSON.stringify(args.noGoType)}) — it is the routing key, and roster-run falls through on an unrecognised value (D2)`);
  }
  if (args.noGoCause && !NO_GO_CAUSES.has(args.noGoCause)) {
    fail(`--no-go-cause must be one of ${[...NO_GO_CAUSES].join(" | ")} (got ${JSON.stringify(args.noGoCause)}) — it mirrors the gate report's top-level cause, and process-incomplete is never one`);
  }
  // schema/review-json-schema.md §streak_override: the gate honours an
  // override only when `by === "human"` (isValidStreakOverride). Any other
  // actor writes an override that does nothing while the verdict claims one —
  // a misleading artifact in the one place a human audits a waiver.
  if (args.streakOverrideBy && args.streakOverrideBy !== "human") {
    fail(`--streak-override-by must be "human" (got ${JSON.stringify(args.streakOverrideBy)}) — the gate ignores any other actor, so the override would be recorded but inert`);
  }
  if (args.status === "GO" && args.noGoRound !== undefined && parseCount(args.noGoRound, "--no-go-round") !== 0) {
    fail("--no-go-round must be 0 (or omitted) on a GO verdict — no_go_round resets on GO");
  }
  if (args.status === "NO-GO" && args.noGoRound === undefined) {
    fail("--no-go-round <n> is required on a NO-GO verdict — it is the round-cap backstop counter and is NOT derivable from `round` (the two are separate counters with separate reset rules)");
  }
  // Both of these select which gate checks run at all, so neither may be a
  // free-form string. An unrecognised --mode reads to the gate exactly like an
  // absent one (no scope-gate obligation); a trace_schema_version other than
  // the one the trace mechanism speaks — including omitting the key — leaves
  // the round un-obligated, so the specialist/scope trace checks never fire
  // while the verdict still looks schema-valid.
  if (!MODES.has(args.mode)) {
    fail(`--mode must be one of ${[...MODES].join(" | ")} (got ${JSON.stringify(args.mode)}) — an unrecognised mode drops the Full-mode scope-gate trace obligation silently`);
  }
  if (args.traceSchemaVersion !== TRACE_SCHEMA_VERSION) {
    fail(`--trace-schema-version must be ${JSON.stringify(TRACE_SCHEMA_VERSION)} (got ${JSON.stringify(args.traceSchemaVersion)}) — a newly assembled round always commits to the trace checks; omitting the stamp is only legitimate for a round that genuinely predates the trace mechanism, which this tool cannot assemble`);
  }
}

function parseSpecialists(specs) {
  return specs.map((raw) => {
    const idx = raw.indexOf("=");
    if (idx <= 0) fail(`--specialist must be <name>=<selection_reason> (got ${JSON.stringify(raw)})`);
    const name = raw.slice(0, idx).trim();
    const selection_reason = raw.slice(idx + 1).trim();
    if (!name || !selection_reason) fail(`--specialist ${JSON.stringify(raw)} has an empty name or selection_reason`);
    return { name, selection_reason };
  });
}

// ── phase 2: write the gate-reported strike back ────────────────────────
// `strike` is the one field this tool cannot compute: it is the gate's own
// output. Refusing a non-boolean here is what keeps `strike: null` — which
// makes computeStrikeMap()'s Map.get() return undefined and thus silently
// resets the novel-finding streak — out of every verdict this tool touches.
// `config` ({max_rounds, strikes, static}) and `trace` (FR-176) are the gate's
// own documented anti-stale-script signal, emitted on every exit code it
// reports on — including the legacy skip (see the header of
// check-review-convergence.js). A report missing either came from a gate
// predating the current strike semantics, so its boolean
// `current_round_strike` was computed under other rules: a stale `false` is
// indistinguishable from a genuine one once journaled, and silently breaks the
// streak the next round evaluates.
//
// (An input-rejection exit 2 emits no report at all — there is nothing to
// write back from on that path, and nothing should be invented.)
function assertGateReportIsCurrent(report) {
  const stale = (what) =>
    fail(`gate report has ${what} — it came from a stale check-review-convergence.js (the current gate emits both config and trace on every exit code it reports on, including the legacy skip). Re-run the gate with the current script; refusing to journal a strike computed under unknown rules.`);
  if (!report.config || typeof report.config !== "object" || typeof report.config.strikes !== "number") {
    stale("no numeric config.strikes");
  }
  if (!report.trace || typeof report.trace !== "object") stale("no `trace` block (FR-176)");
  if (!report.hardening || typeof report.hardening.version !== "string") {
    stale("no `hardening.version` (scripts/lib/review-gate-hardening.js)");
  }
  // A gate run that was authorized to SKIP its own checks is not evidence that
  // those checks passed. `legacy_skip_authorized` is the recorded-skip marker
  // the gate stamps when --allow-legacy suppressed the round / no_go_round
  // fail-closed refusal; journaling a strike from such a report would convert
  // a recorded skip back into an indistinguishable pass, which is precisely
  // the defect the flag exists to avoid.
  if (report.legacy_skip_authorized === true) {
    fail(
      "gate report has legacy_skip_authorized: true — that run was authorized (via --allow-legacy) to SKIP " +
        "strike classification and/or the round cap, so its strike is not a computed result. Refusing to journal " +
        "it. Assemble the verdict properly (scripts/review-verdict-assemble.js) and re-gate without --allow-legacy."
    );
  }
}

// ── rounds_audit carry-forward (append-only) ─────────────────────────────
// Prior entries are copied through verbatim; a prior entry for the round being
// assembled is a refusal, not an overwrite — that would mean the round counter
// never advanced (scripts/lib/review/review-lifecycle.js owns the bump).
function carryForwardRoundsAudit(priorEntries, round) {
  const carried = [];
  for (const entry of priorEntries) {
    if (!entry || typeof entry !== "object") fail("prior rounds_audit contains a non-object entry — refusing to carry forward unverifiable state");
    if (entry.round === round) {
      fail(`prior verdict already has a rounds_audit entry for round ${round} — rounds_audit is append-only, and this round was already journaled. If the prior gate run exited 3, repair the draft and RE-GATE that round (never bump \`round\`); only a completed round advances the counter.`);
    }
    if (typeof entry.round === "number" && entry.round > round) {
      fail(`prior verdict has a rounds_audit entry for round ${entry.round}, which is ahead of --round ${round} — refusing to assemble a verdict that would look like it went backwards`);
    }
    // A carried entry for round >= 2 with no boolean `strike` is a round whose
    // phase 2 (--write-strike) never ran. computeStrikeMap() cannot see it, so
    // the streak resets across it and the gate's only signal is a warning on a
    // run that may well exit 0 — the exact silent pass this tool exists to
    // prevent. Refusing here is the only place the two-phase protocol has
    // teeth. (Round 1 never strikes, so its entry is exempt.)
    if (typeof entry.round === "number" && entry.round >= 2 && typeof entry.strike !== "boolean") {
      fail(`prior rounds_audit entry for round ${entry.round} has no boolean \`strike\` (${JSON.stringify(entry.strike)}) — that round was never completed with --write-strike, and carrying it forward silently resets the novel-finding streak. Re-gate round ${entry.round} and write its strike before assembling round ${round}.`);
    }
    carried.push(entry);
  }
  return carried;
}

function buildAuditEntry(args, round, specialists) {
  const entry = { round, reviewed_sha: args.reviewedSha };
  entry.fix_sha = args.fixSha ? args.fixSha : null;
  if (!args.fixSha) entry.fix_sha_reason = args.fixShaReason;
  entry.specialists_run = specialists;
  // Always stamped: validateAssembleArgs() has already pinned it to "1.0", and
  // the stamp is what obligates the round to the trace checks.
  entry.trace_schema_version = args.traceSchemaVersion;
  // NOTE: `strike` is deliberately ABSENT, not null. It is written by
  // --write-strike after the gate reports current_round_strike.
  return entry;
}

function buildNoGoReason(args) {
  if (args.status === "GO") return null;
  const reason = { type: args.noGoType };
  if (args.noGoCause) reason.cause = args.noGoCause;
  if (args.failedAcs.length > 0) reason.failed_acs = args.failedAcs;
  return reason;
}

function buildVerdict({ args, round, cycle, normalized, roundsAudit, crossRuntime, crossRuntimeFindings }) {
  const verdict = {
    task: args.task,
    date: new Date().toISOString(),
    status: args.status,
    mode: args.mode,
    round,
    cycle,
    no_go_round: args.noGoRound === undefined ? 0 : parseCount(args.noGoRound, "--no-go-round"),
    auto_fixes_applied: parseCount(args.autoFixes, "--auto-fixes"),
    findings: normalized.findings,
    cross_runtime_findings: crossRuntimeFindings,
    cross_runtime: crossRuntime,
    rounds_audit: roundsAudit,
    no_go_reason: buildNoGoReason(args),
    summary: args.summary || "",
    escalation_needed: args.escalationNeeded,
    escalation_reason: args.escalationReason || null,
    normalized_by: normalized.normalizer_version,
  };
  if (args.streakOverrideBy) verdict.streak_override = { round, by: args.streakOverrideBy };
  return verdict;
}

// Surfaces the normalizer's non-verdict outputs (they have no home in
// review.json and must not vanish silently, FR-100).
function reportNormalizerSideChannels(normalized) {
  for (const w of normalized.warnings || []) warn(`review-normalize: ${w}`);
  for (const d of normalized.probable_duplicates || []) {
    warn(`review-normalize: probable duplicate needs owner adjudication: ${JSON.stringify(d)}`);
  }
  const rejected = normalized.rejected || [];
  if (rejected.length === 0) return;
  for (const r of rejected) warn(`review-normalize: REJECTED (schema-invalid) finding: ${r.reason}`);
  // Unconditional — there is deliberately no escape hatch. A malformed
  // HIGH/CRITICAL that never reaches `findings` is one the gate cannot
  // classify as a novel strike, so the round does not strike, the streak does
  // not accumulate, and the verdict looks clean. The stderr warnings above are
  // not durable, so "warn and continue" would be silent loss, fail-open.
  fail(`${rejected.length} finding(s) were rejected as schema-invalid by review-normalize.js — fix them against schema/review-finding.schema.json. There is deliberately no flag to drop them: a dropped HIGH/CRITICAL cannot strike, so the gate would report a clean round it never saw.`);
}

// The normalizer PROPOSES dispositions and never mutates ledger status itself
// (single-executor principle) — so somebody downstream must apply them, and
// this tool is that somebody. `normalize()` returns
// `findings = ledger.concat(genuinelyNew)`: a re-reported RESOLVED finding is
// NOT in `findings`, it is in `dispositions.reopened`. Ignore that and
// `isReopenedStrikeFinding()` (`f.reopened_at_round === round`) can never fire
// on anything this tool emits, so E-4 is dead and a regression-heavy loop-back
// round is invisible to two-strike escalation.
//
// Matching reuses the normalizer's own buildLedgerIndex/findLedgerEntry so the
// identity rule cannot drift from the one that produced the disposition.
function applyDispositions(normalized) {
  const dispositions = normalized.dispositions || {};
  const pending = Array.isArray(dispositions.pending_check) ? dispositions.pending_check : [];
  const reopened = Array.isArray(dispositions.reopened) ? dispositions.reopened : [];

  for (const r of normalized.reobservations || []) {
    warn(`review-normalize: reobserved (carry-forward noise, not a fresh finding): ${r.fingerprint}`);
  }

  // pending-check is genuinely undecidable until THIS round's gate run has
  // produced a report — the normalizer says so explicitly. Applying it either
  // way would be a guess, and dropping it loses a possible regression, so it
  // is a refusal with the repair spelled out.
  if (pending.length > 0) {
    fail(
      `${pending.length} finding(s) are in the normalizer's pending-check disposition: a RESOLVED ledger entry with a linked check that the supplied gate report has no entry for, re-reported this round. Whether it is a regression is only knowable from a gate report that covers that check. Re-gate the prior round so briefs/<task>-gate-report.json carries the check, then re-assemble. Fingerprints: ${pending.map((p) => p.fingerprint).join(", ")}`
    );
  }

  if (reopened.length === 0) return;
  const index = buildLedgerIndex(normalized.findings);
  for (const body of reopened) {
    const entry = findLedgerEntry(body, index);
    if (!entry) {
      fail(`normalizer reopened ${body.fingerprint} but no matching entry is in the normalized findings — refusing to emit a verdict that would drop a reopened HIGH+ finding`);
    }
    Object.assign(entry, body);
    warn(`review-normalize: REOPENED ${body.fingerprint} (was RESOLVED in round ${body.reopened_from_round}) — applied to findings, so it counts toward this round's strike (E-4)`);
  }
}

// `first_seen_round` is NOT stamped by the normalizer — canonicalizeFindings()
// rewrites only fingerprint/fingerprint_v2/fid/status, and
// schema/review-finding.schema.json does not require the field. But
// isNovelStrikeFinding() returns false unless it is numeric and equals the
// round, so a HIGH+ finding that arrives without it can NEVER strike: the
// round is silently unstrikeable and the streak never accumulates. Scoped to
// HIGH+ because that is exactly the set the strike rule looks at.
function assertStrikeableFindings(findings, round) {
  const unclassifiable = findings.filter(
    (f) => f && HIGH_PLUS.has(f.severity) && (typeof f.first_seen_round !== "number" || f.first_seen_round > round)
  );
  if (unclassifiable.length === 0) return;
  fail(
    `${unclassifiable.length} CRITICAL/HIGH finding(s) have a missing or future first_seen_round: ` +
      `${unclassifiable.map((f) => `${f.fingerprint} (${JSON.stringify(f.first_seen_round)})`).join(", ")}. ` +
      "isNovelStrikeFinding() requires a numeric first_seen_round equal to the round, so these can never strike — " +
      `the round would be silently unstrikeable. Stamp first_seen_round: ${round} on findings raised this round.`
  );
}

// INV-5/E-7: cross_runtime_findings is augment-only and never rewritten after
// intake — but the normalizer only ever sees THIS round's input files, so
// taking its output verbatim drops every prior round's entries. Carried within
// a cycle, reset on a fresh one (like `findings`).
function carryForwardCrossRuntime(prior, freshCycle, current) {
  const priorEntries = !freshCycle && prior && Array.isArray(prior.cross_runtime_findings) ? prior.cross_runtime_findings : [];
  const seen = new Set(priorEntries.map((f) => (f && (f.fid || f.fingerprint)) || ""));
  const appended = current.filter((f) => !seen.has((f && (f.fid || f.fingerprint)) || ""));
  return priorEntries.concat(appended);
}

// roster-review's Output Contract: an OPEN CRITICAL/HIGH forces NO-GO. The
// gate never reads `status`, so nothing downstream catches a GO that carries
// one — and deriveRoundState() then treats the next cycle as fresh and resets
// the ledger to [], deleting the finding from the only file that held it.
function assertGoIsClean(args, findings) {
  if (args.status !== "GO") return;
  const open = findings.filter((f) => f && HIGH_PLUS.has(f.severity) && f.status !== "RESOLVED" && f.status !== "ACCEPTED");
  if (open.length === 0) return;
  fail(
    `--status GO with ${open.length} open CRITICAL/HIGH finding(s): ${open.map((f) => f.fingerprint).join(", ")}. ` +
      "An open HIGH+ forces NO-GO (roster-review Output Contract). The gate does not read `status`, so this would " +
      "pass; and the next cycle resets findings to [] on a GO, deleting them. RESOLVE or human-ACCEPT them first."
  );
}

// Cross-checks --round/--cycle/--no-go-round against the lifecycle witness
// rather than re-deriving the rule here. FAIL-CLOSED, not advisory: all three
// are inputs the gate uses to decide how much to check, so a wrong one buys
// silence rather than a wrong answer.
//
//   --round  computeStreakViolation() walks `round, round-1, …` for
//     consecutive `true`s, and computeStrikeMap() only knows rounds that have
//     a rounds_audit entry. Jumping 3 -> 7 leaves 4-6 absent, so
//     `strikeByRound.get(r) !== true` short-circuits and the streak is erased
//     — silently, since computeMissingStrikeWarnings() only inspects entries
//     that EXIST. carryForwardRoundsAudit() already refuses a repeated or
//     backwards round; this refuses the forward jump.
//   --cycle  scopes trace lines by (cycle, round); a wrong one hides them.
//   --no-go-round  the round-cap backstop, caller-supplied and never derived,
//     so a value that never advances never trips the cap.
//
// Legacy exception: a prior NO-GO verdict with no `round` derives `null` — the
// state this tool exists to migrate away from, unverifiable, so it warns.
function enforceLifecycle(prior, derived, round, cycle, args) {
  if (derived.round === null) {
    warn(
      "prior verdict has no `round` key (legacy) — round continuity cannot be verified against " +
        "scripts/lib/review/review-lifecycle.js. This round re-establishes it."
    );
  } else if (derived.round !== round) {
    fail(
      `--round ${round} but scripts/lib/review/review-lifecycle.js derives ${derived.round} from the prior verdict. ` +
        "Rounds are physical and consecutive: a skipped round leaves a hole in rounds_audit that erases the " +
        "novel-finding streak (computeStreakViolation walks consecutive rounds and stops at the first non-true)."
    );
  }
  if (derived.cycle !== null && derived.cycle !== cycle) {
    fail(
      `--cycle ${cycle} but scripts/lib/review/review-lifecycle.js derives ${derived.cycle} from the prior verdict — ` +
        "the trace mechanism scopes lines by (cycle, round), so a wrong cycle hides this round's trace lines."
    );
  }
  enforceNoGoRound(prior, derived, args);
}

// no_go_round counts only QUALIFYING (NO-GO) rounds and resets to 0 on GO, so
// it is not derivable from `round`. What IS checkable is its step: relative to
// the prior verdict it may only stay put or advance by one, and a fresh cycle
// (no prior, or a prior GO) starts from 0.
function enforceNoGoRound(prior, derived, args) {
  if (args.status === "GO") return; // already pinned to 0 by validateAssembleArgs
  const priorNoGoRound = !derived.freshCycle && prior && typeof prior.no_go_round === "number" ? prior.no_go_round : 0;
  const noGoRound = parseCount(args.noGoRound, "--no-go-round");
  if (noGoRound !== priorNoGoRound && noGoRound !== priorNoGoRound + 1) {
    fail(
      `--no-go-round ${noGoRound} does not follow the prior verdict's ${priorNoGoRound} — it may only hold or ` +
        "advance by one per NO-GO verdict. A no_go_round that never advances never reaches --max-rounds, which " +
        "disables the round-cap backstop entirely."
    );
  }
}
module.exports = {
  TRACE_SCHEMA_VERSION,
  STATUSES,
  MODES,
  NO_GO_TYPES,
  NO_GO_CAUSES,
  HIGH_PLUS,
  fail,
  warn,
  parseCount,
  validateAssembleArgs,
  parseSpecialists,
  assertGateReportIsCurrent,
  carryForwardRoundsAudit,
  buildAuditEntry,
  buildNoGoReason,
  buildVerdict,
  reportNormalizerSideChannels,
  applyDispositions,
  assertStrikeableFindings,
  carryForwardCrossRuntime,
  assertGoIsClean,
  enforceLifecycle,
};
