# Review-tool bundle — consumer notes

This directory carries a small set of files distributed from the
[roster](https://github.com/mathiasbourgoin/roster) repo: the deterministic tools
`roster-review`/`roster-qa` depend on, plus their shared libraries and schema.

**These files are upstream-owned and generated.** `scripts/review-bundle.manifest.json` is the
sole sentinel — it lists every file in the bundle with its expected sha256. **Do not hand-edit
any bundle file or the manifest.** A local edit will be detected as "modified" on the next
verify/upgrade/remove and handled conservatively (skipped with a warning, or refused outright) —
see the recovery guidance those commands print if that happens.

## Reachability — which half of "survives a fresh clone" is actually fixed

Tracking these files guarantees a fresh clone **has** every gate. It does not guarantee anything
**runs** them. Most are invoked from roster skill prose under `.harness/skills/`,
`.claude/commands/` and `.agents/skills/`, all of which are deliberately machine-local. So a clone
can hold a complete, hash-verified bundle that sits inert until `/recruit` installs the roster.

**The decision: keep the callers local.** Tracking the skills would fork the roster install into
this repo — a second upgrade obligation on top of the bundle's, on large prose files that churn on
every roster release, owned by a different upstream. The cost is worse than the gap.

**What makes the gap safe is that it is measured, not assumed.**
`scripts/check-review-bundle-tracked.sh` computes transitive reachability from the roots a fresh
clone executes by itself (CI workflows and the `Makefile`) and reports every bundle tool no tracked
caller reaches. Documentation mentions do not count as callers — only executable carriers form
edges. Today that list is:

| Tool | Reached from CI? | Why |
|---|---|---|
| `review-bundle-verify.js` | yes | `check-review-bundle-tracked.sh`, run by CI |
| `xruntime-review.js`, `xruntime-exec.sh` | yes | via `xruntime-caller-fault.test.js` in CI |
| `review-normalize.js` | no | pipeline-phase tool; needs a task and `briefs/` state |
| `check-review-convergence.js` | no | same |
| `check-scope-diff.sh` | no | same |
| `review-verdict-assemble.js` | no | its `.test.js` runs in CI; the tool itself is phase-driven |

If a future change wires one of these into CI, the report shrinks on its own. If someone adds a
bundle tool that nothing ever runs, it appears here instead of being silently inert.

## Local patches (this consumer)

The rule above is "do not hand-edit". Where SPOC has had to, the divergence is declared rather
than hidden: `review-bundle.manifest.json` carries `channel: "local-patched"`, a
`bundle_version` suffixed `+spoc.N`, and a `local_patches[]` entry naming every touched path, the
reason, and the test that covers it. `review-bundle-verify.js` still passes, because the manifest
hashes were regenerated — the manifest remains an honest sentinel against *undeclared* drift.

An upgrade will overwrite these files. Re-apply every `local_patches[]` entry with
`reapply_on_upgrade: true` afterwards and re-run its tests, or the fix silently regresses.

| Ref | What | Why it could not wait for upstream |
|-----|------|-----------------------------------|
| #102 + F1 | Fault attribution on cross-runtime outcomes (`fault: "runtime" \| "caller"`), a nonzero exit always being a runtime fault, `--emit-contract`, `xruntime-contract.js`, accurate version-probe cause reporting, bounded journal fail-close | A malformed probe *output* armed the runtime circuit breaker as though the runtime had failed, suppressing probes for unrelated work — and, in the first cut of that fix, a *crashed* runtime was blamed on the caller and never armed the breaker at all. Covered by `scripts/xruntime-caller-fault.test.js`. |
| CodeRabbit | `deriveRoundState` refuses an unrecognized prior verdict status | It was read as a NO-GO continuation, which increments the round and carries `rounds_audit`/`cross_runtime` forward out of a verdict whose validity is unknown — manufacturing attested-looking state from unverifiable input. |

**These declarations are enforced, not decorative.** Each entry names `markers` — strings that
exist only while the patch is applied — and `check-review-bundle-tracked.sh` fails if any marker is
missing. An upgrade that overwrites a patched file regenerates its sha256, so
`review-bundle-verify` stays green on a tree where the fix is gone; the marker is what notices.
A patch that declares `reapply_on_upgrade` without markers is itself a failure.

## Commands

Run verification from the consumer repo root. Install, upgrade, and removal remain owned by the
external `review-bundle-install.sh` bootstrapper; that lifecycle script is intentionally not
installed into the consumer bundle.

```bash
# Verify the installed bundle is complete and unmodified (no network calls).
node scripts/review-bundle-verify.js

# Fetch the lifecycle installer from a trusted roster raw URL.
# Replace OWNER, REPOSITORY, and REF with the trusted source coordinates.
RAW_PREFIX='https://raw.githubusercontent.com/OWNER/REPOSITORY/REF'
INSTALLER=$(mktemp)
trap 'rm -f "$INSTALLER"' EXIT
curl -fsSL "$RAW_PREFIX/scripts/review-bundle-install.sh" -o "$INSTALLER"

# Install (first time) or upgrade (already installed).
bash "$INSTALLER" install --from-raw "$RAW_PREFIX"
bash "$INSTALLER" upgrade --from-raw "$RAW_PREFIX"

# Remove the bundle (the shared wrapper, scripts/xruntime-exec.sh, is kept — other
# tools may still depend on it).
bash "$INSTALLER" remove
```

Full details, including collision handling and `--force`: see the header comment of
`scripts/review-bundle-install.sh`, and specs/review-tool-distribution.md in the roster repo.
