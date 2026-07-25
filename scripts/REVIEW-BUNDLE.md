# Review-tool bundle — consumer notes

This directory carries a small set of files distributed from the
[roster](https://github.com/mathiasbourgoin/roster) repo: the deterministic tools
`roster-review`/`roster-qa` depend on, plus their shared libraries and schema.

**These files are upstream-owned and generated.** `scripts/review-bundle.manifest.json` is the
sole sentinel — it lists every file in the bundle with its expected sha256. **Do not hand-edit
any bundle file or the manifest.** A local edit will be detected as "modified" on the next
verify/upgrade/remove and handled conservatively (skipped with a warning, or refused outright) —
see the recovery guidance those commands print if that happens.

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
| #102 | Fault attribution on cross-runtime outcomes (`fault: "runtime" \| "caller"`), `--emit-contract`, `scripts/lib/xruntime/xruntime-contract.js` | A malformed probe *output* armed the runtime circuit breaker as though the runtime had failed, which then suppressed probes for unrelated work. Covered by `scripts/xruntime-caller-fault.test.js`. |

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
