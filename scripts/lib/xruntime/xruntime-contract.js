// scripts/lib/xruntime/xruntime-contract.js — CommonJS.
//
// The output contract the cross-runtime probe requires, as data.
//
// Backlog #102. A non-conforming OUTPUT is a CALLER fault, not a runtime
// failure: the runtime started, ran, and answered — the answer just did not
// match a contract the caller never stated. Treating that as runtime
// degradation armed the circuit breaker and suppressed probes for unrelated
// work. Four parallel cross-runtime passes degraded `opencode` on all four PRs
// at once; the breaker read self-inflicted contention and prompt drift as a
// broken runtime.
//
// Two halves to the fix, and this module is the first: make the contract
// something a caller can obtain mechanically (`--emit-contract`) instead of
// re-describing in prose per prompt. The second half lives in
// xruntime-classify.js / xruntime-journal.js: fault attribution, so a
// caller-fault outcome never arms the breaker.
"use strict";

const CONTRACT_VERSION = "1";

// Appended verbatim to a probe prompt. Deliberately terse and imperative:
// every clause here is a condition xruntime-classify.js mechanically checks,
// so anything softer would describe a rule that is not actually enforced.
const OUTPUT_CONTRACT = `## Output contract (machine-parsed — v${CONTRACT_VERSION})

Reply with a JSON array and nothing that could be mistaken for one.

- The reply MUST contain a JSON array as its last \`\`\`json fenced block, or be
  a bare JSON array with no other text.
- The array holds finding objects conforming to schema/review-finding.schema.json.
- An empty array \`[]\` is the correct answer when you found nothing. It is a
  successful result, not a failure — do not explain instead of emitting it.
- Every finding sets \`specialist\` to "<runtime>-xruntime".
- No prose before or after the array unless it precedes the final fenced block.
- Report behavior, not phrasing. A disagreement about wording is not a finding.

Output that does not parse under these rules is discarded as a CALLER error:
the run is not retried, and it does not count against the runtime.`;

module.exports = { CONTRACT_VERSION, OUTPUT_CONTRACT };
