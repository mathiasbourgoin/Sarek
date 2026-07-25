// scripts/lib/normalize-rules.js — CommonJS, pure functions.
//
// Fingerprinting, exact-dedup, probable-duplicate, and re-observation rules for
// scripts/review-normalize.js (spec: specs/review-skill-slimming.md US-2,
// FR-099..108, Amendment D-1; specs/review-v2-corrections.md INV-1/INV-2/INV-5,
// Amendments E-3/E-4/E-7). No I/O here — scripts/review-normalize.js owns
// reading files/stdin and writing stdout.
"use strict";

const crypto = require("crypto");

const SEVERITY_RANK = { CRITICAL: 5, HIGH: 4, MEDIUM: 3, LOW: 2, INFO: 1 };

// FR-102: fingerprint v1 = path:line:category, null line -> 0. This is the
// PRIMARY identity — always recomputed here, never trusted from specialist
// input, so a specialist's own (possibly non-conforming) `fingerprint` field
// can never desync the merge/dedup pipeline from the canonical form.
function canonicalFingerprint(finding) {
  const line = finding.line === null || finding.line === undefined ? 0 : finding.line;
  return `${finding.path}:${line}:${finding.category}`;
}

// FR-103: fingerprint_v2 computed only when the finding already carries
// boundary/invariant/failure_mode — never required of any input.
function hasV2Fields(finding) {
  return finding.boundary !== undefined || finding.invariant !== undefined || finding.failure_mode !== undefined;
}

function computeFingerprintV2(finding) {
  return [finding.boundary || "", finding.invariant || "", finding.failure_mode || ""].join("|");
}

// E-3: fid = fingerprint + "#" + sha8(normalized summary) — the addressable
// identity for reobservation matching, probable-duplicate records, and gate
// checks[] keying. Requires `finding.fingerprint` to already be canonical
// (caller computes fid AFTER canonicalFingerprint). v1 `fingerprint` is
// unchanged for compatibility — fid is purely additive.
function computeFid(finding) {
  const normalizedSummary = normalizeSummary(finding.summary);
  const hash = crypto.createHash("sha256").update(normalizedSummary).digest("hex").slice(0, 8);
  return `${finding.fingerprint}#${hash}`;
}

function normalizeSummary(summary) {
  return (summary || "").trim().toLowerCase().replace(/\s+/g, " ");
}

// FR-106 scope guard: cross-runtime findings never enter the primary
// merge/dedup/reobservation pipeline — they are routed to cross_runtime_findings
// (augment-only elsewhere in roster-review) but per INV-5/E-7 they ARE
// canonicalized and deduplicated within that augment-only array (see
// scripts/review-normalize.js) so an untrusted model-provided fingerprint can
// never survive into anything the ratchet mirrors (FR-015).
function isCrossRuntime(finding) {
  return typeof finding.specialist === "string" && /-xruntime$/.test(finding.specialist);
}

function canonicalLine(finding) {
  return finding.line === null || finding.line === undefined ? 0 : finding.line;
}

// INV-1: two findings sharing a v1 fingerprint but differing in `summary` (or
// any present v2 semantic field) are NEVER exact-merged — a v1 collision
// alone (same path:line:category) never proves it is the same defect. Exact
// duplicate requires byte-identical summary AND, when either side carries a
// v2 field, byte-identical boundary/invariant/failure_mode too.
function isExactDuplicatePair(a, b) {
  if (a.summary !== b.summary) return false;
  if (hasV2Fields(a) || hasV2Fields(b)) {
    if ((a.boundary || "") !== (b.boundary || "")) return false;
    if ((a.invariant || "") !== (b.invariant || "")) return false;
    if ((a.failure_mode || "") !== (b.failure_mode || "")) return false;
  }
  return true;
}

// FR-104/INV-1: merges one fingerprint-sharing group into a single survivor
// only when every non-anchor member is an exact duplicate of the anchor (see
// isExactDuplicatePair — semantic match, not merely "both have a real line");
// otherwise the anchor stays alone and everything else is a leftover, which
// the caller re-exposes for probable-duplicate detection (a leftover shares
// path+category+line with its survivor, i.e. delta 0 — always in the
// probable window).
function partitionExactGroup(group) {
  if (group.length === 1) return { merged: group, leftover: [] };
  const anchor = group[0];
  const merged = group.filter((f) => isExactDuplicatePair(anchor, f));
  const leftover = group.filter((f) => !isExactDuplicatePair(anchor, f));
  return merged.length > 1 ? { merged, leftover } : { merged: [anchor], leftover: group.slice(1) };
}

function pickSurvivor(group) {
  let best = group[0];
  for (let i = 1; i < group.length; i++) {
    const candidate = group[i];
    const rankDiff = (SEVERITY_RANK[candidate.severity] || 0) - (SEVERITY_RANK[best.severity] || 0);
    if (rankDiff > 0) best = candidate;
    else if (rankDiff === 0 && (candidate.evidence || "").length > (best.evidence || "").length) best = candidate;
  }
  return best;
}

// Merges a fingerprint-group into one survivor carrying `convergence: [...]`
// (the contributing specialists, in input order, deduplicated).
function mergeGroup(group) {
  const survivor = Object.assign({}, pickSurvivor(group));
  const specialists = [];
  for (const f of group) if (!specialists.includes(f.specialist)) specialists.push(f.specialist);
  if (specialists.length > 1) survivor.convergence = specialists;
  return survivor;
}

// FR-104/INV-1: groups findings by fingerprint, merges exact duplicates within
// each group, and returns the settled findings list (survivors + any
// unmerged leftovers) alongside every leftover so the caller can still run
// probable-duplicate detection over them.
function mergeExactDuplicates(findings) {
  const byFingerprint = new Map();
  for (const f of findings) {
    const key = f.fingerprint;
    if (!byFingerprint.has(key)) byFingerprint.set(key, []);
    byFingerprint.get(key).push(f);
  }

  const settled = [];
  for (const group of byFingerprint.values()) {
    const { merged, leftover } = partitionExactGroup(group);
    settled.push(mergeGroup(merged));
    settled.push(...leftover);
  }
  return settled;
}

// FR-105: non-exact findings sharing path+category with |line delta| <= 3 are
// listed for owner adjudication, never auto-merged. Operates over the
// SETTLED list (post exact-merge) so an already-merged group contributes
// only its single survivor to this pass.
function probableDuplicateRecord(a, b) {
  if (a.path !== b.path || a.category !== b.category) return null;
  const delta = Math.abs(canonicalLine(a) - canonicalLine(b));
  if (delta > 3) return null;
  return {
    path: a.path,
    category: a.category,
    line_delta: delta,
    a: { fingerprint: a.fingerprint, fid: a.fid || null, specialist: a.specialist, line: a.line },
    b: { fingerprint: b.fingerprint, fid: b.fid || null, specialist: b.specialist, line: b.line },
  };
}

function computeProbableDuplicates(settledFindings) {
  const probable = [];
  for (let i = 0; i < settledFindings.length; i++) {
    for (let j = i + 1; j < settledFindings.length; j++) {
      const record = probableDuplicateRecord(settledFindings[i], settledFindings[j]);
      if (record) probable.push(record);
    }
  }
  return probable;
}

// INV-1: carried ledger rows have already been adjudicated against each
// other, so only compare them to genuinely new survivors. This surfaces a
// fail-closed identity miss without re-emitting old ledger↔ledger noise.
function computeLedgerProbableDuplicates(ledger, freshFindings) {
  const probable = [];
  for (const a of ledger) {
    if (!a || typeof a !== "object") continue;
    for (const b of freshFindings) {
      const record = probableDuplicateRecord(a, b);
      if (record) probable.push(record);
    }
  }
  return probable;
}

// E-3/INV-1: indexes the prior cumulative ledger by semantic `fid`. Legacy
// entries that predate the field are migrated in memory when their summary
// is available. The coarse v1 fingerprint index remains available only for
// callers whose incoming finding also predates fid; a modern fid miss must
// never fall back and suppress a distinct finding at the same location.
function buildLedgerIndex(ledger) {
  const byFid = new Map();
  const byFingerprint = new Map();
  for (const entry of ledger) {
    if (!entry || typeof entry !== "object") continue;
    const semanticFid =
      entry.fid ||
      (entry.fingerprint && typeof entry.summary === "string" ? computeFid(entry) : null);
    if (semanticFid) {
      if (!byFid.has(semanticFid)) byFid.set(semanticFid, []);
      byFid.get(semanticFid).push(entry);
    }
    if (entry.fingerprint && !byFingerprint.has(entry.fingerprint)) byFingerprint.set(entry.fingerprint, entry);
  }
  return { byFid, byFingerprint };
}

function sameSemanticIdentity(a, b) {
  if (a.fingerprint !== b.fingerprint) return false;
  if (normalizeSummary(a.summary) !== normalizeSummary(b.summary)) return false;
  if (hasV2Fields(a) || hasV2Fields(b)) {
    if ((a.boundary || "") !== (b.boundary || "")) return false;
    if ((a.invariant || "") !== (b.invariant || "")) return false;
    if ((a.failure_mode || "") !== (b.failure_mode || "")) return false;
  }
  return true;
}

function findLedgerEntry(f, index) {
  if (f.fid) {
    const candidates = index.byFid.get(f.fid) || [];
    return candidates.find((entry) => sameSemanticIdentity(f, entry)) || null;
  }
  if (f.fingerprint && index.byFingerprint.has(f.fingerprint)) return index.byFingerprint.get(f.fingerprint);
  return null;
}

// E-2: indexes a persisted gate report's checks[] by (check, fid) with a
// fingerprint fallback — the same keying the gate itself uses (E-3).
function buildGateCheckIndex(gateReport) {
  const map = new Map();
  if (!gateReport || !Array.isArray(gateReport.checks)) return map;
  for (const c of gateReport.checks) {
    if (!c || typeof c.check !== "string") continue;
    const key = `${c.check}#${c.fid || c.fingerprint || ""}`;
    map.set(key, c);
  }
  return map;
}

// INV-2: classifies a ledger match into one of three dispositions.
//   - "reobserved": ledger entry is not RESOLVED (no regression risk — plain
//     carry-forward noise), OR it IS RESOLVED with a check that the latest
//     gate report shows red_verified: true on the current tree.
//   - "reopen": RESOLVED with no linked check at all (can never be verified
//     -> always a regression, never metadata, per INV-2's explicit rule) OR
//     RESOLVED with a check the gate report shows as failed/unverified, OR
//     RESOLVED with a check but NO gate report was supplied at all (fail
//     closed — "Resolution of the intake open question").
//   - "pending-check": RESOLVED with a check, a gate report WAS supplied, but
//     that report has no entry for this exact (check, fid) yet — the check
//     may have been linked only this round. The skill resolves this after
//     THIS round's gate run, from this round's freshly persisted report.
function classifyDisposition(ledgerEntry, gateCheckIndex, gateReportProvided) {
  if (ledgerEntry.status !== "RESOLVED") return "reobserved";
  if (!ledgerEntry.check) return "reopen";
  if (!gateReportProvided) return "reopen";
  const key = `${ledgerEntry.check}#${ledgerEntry.fid || ledgerEntry.fingerprint || ""}`;
  const checkEntry = gateCheckIndex.get(key);
  if (!checkEntry) return "pending-check";
  return checkEntry.red_verified === true ? "reobserved" : "reopen";
}

// D-1/E-4: separates ledger-matching new findings by disposition. "reobserved"
// findings are reduced to metadata (never merged, never a fresh finding,
// never dropped — unchanged from the original D-1 contract). "reopen" and
// "pending-check" findings carry their FULL re-observed body forward (INV-2:
// "the full re-observed finding body is always preserved") plus enough
// provenance for the skill to act — the normalizer proposes, it never itself
// mutates ledger status (single-executor principle for gate/report reads).
function splitReobservations(findings, ledger, round, gateReport) {
  const index = buildLedgerIndex(ledger);
  const gateCheckIndex = buildGateCheckIndex(gateReport);
  const reobservations = [];
  const reopened = [];
  const pendingCheck = [];
  const genuinelyNew = [];
  const stampedRound = round === undefined ? null : round;

  for (const f of findings) {
    const ledgerEntry = findLedgerEntry(f, index);
    if (!ledgerEntry) {
      genuinelyNew.push(f);
      continue;
    }

    const disposition = classifyDisposition(ledgerEntry, gateCheckIndex, !!gateReport);
    if (disposition === "reobserved") {
      reobservations.push({ fingerprint: f.fingerprint, fid: f.fid || null, specialist: f.specialist, round: stampedRound });
    } else if (disposition === "reopen") {
      reopened.push(
        Object.assign({}, f, {
          status: "OPEN",
          resolved_round: null,
          reopened_from_round: ledgerEntry.resolved_round,
          reopened_at_round: stampedRound,
        })
      );
    } else {
      pendingCheck.push(Object.assign({}, f, { pending_check: ledgerEntry.check }));
    }
  }
  return { reobservations, reopened, pendingCheck, genuinelyNew };
}

module.exports = {
  SEVERITY_RANK,
  canonicalFingerprint,
  hasV2Fields,
  computeFingerprintV2,
  computeFid,
  isCrossRuntime,
  canonicalLine,
  isExactDuplicatePair,
  mergeExactDuplicates,
  computeProbableDuplicates,
  computeLedgerProbableDuplicates,
  buildLedgerIndex,
  sameSemanticIdentity,
  findLedgerEntry,
  buildGateCheckIndex,
  classifyDisposition,
  splitReobservations,
};
