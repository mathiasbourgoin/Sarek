// scripts/lib/xruntime-digest.js — CommonJS.
//
// config_digest computation for scripts/xruntime-review.js (FR-093, FR-094).
// Hashes the runtime name, its `--version` output (10s timeout), and the
// sandbox-mode flag — deliberately excludes the review timeout value and any
// prompt/diff content (including either would change the digest every round
// and void the probe-once/no-retry rule).
"use strict";

const crypto = require("crypto");
const { spawnSync } = require("child_process");

// Default 10s per FR-094. Overridable via XRUNTIME_VERSION_PROBE_TIMEOUT_MS
// strictly as a test-speed hook (mirrors xruntime-exec.sh's own XRUNTIME_BIN
// testing hook) — production behavior is the spec's 10s default.
const VERSION_PROBE_TIMEOUT_MS = parseInt(process.env.XRUNTIME_VERSION_PROBE_TIMEOUT_MS, 10) || 10000;

// A hang classifies as degraded `version-probe-timeout` (FR-094) with a
// placeholder digest `<runtime>:version-unavailable` — never a real hash of
// unavailable output.
function probeVersion(runtimeBin) {
  const result = spawnSync(runtimeBin, ["--version"], {
    timeout: VERSION_PROBE_TIMEOUT_MS,
    encoding: "utf8",
  });
  const timedOut = !!(result.error && result.error.code === "ETIMEDOUT");
  const output = timedOut ? "" : (result.stdout || "") + (result.stderr || "");
  return { timedOut, output };
}

function computeDigests(runtimeName, runtimeBin, sandboxFlags) {
  const probe = probeVersion(runtimeBin);
  if (probe.timedOut) {
    return {
      digests: Object.fromEntries(sandboxFlags.map((flag) => [flag, `${runtimeName}:version-unavailable`])),
      versionProbeTimedOut: true,
    };
  }
  const digests = Object.fromEntries(
    sandboxFlags.map((sandboxFlag) => {
      const hash = crypto
        .createHash("sha256")
        .update(`${runtimeName}:${probe.output}:${sandboxFlag}`)
        .digest("hex")
        .slice(0, 16);
      return [sandboxFlag, `${runtimeName}:${hash}`];
    })
  );
  return { digests, versionProbeTimedOut: false };
}

function computeDigest(runtimeName, runtimeBin, sandboxFlag) {
  const result = computeDigests(runtimeName, runtimeBin, [sandboxFlag]);
  return { digest: result.digests[sandboxFlag], versionProbeTimedOut: result.versionProbeTimedOut };
}

module.exports = { computeDigest, computeDigests, probeVersion, VERSION_PROBE_TIMEOUT_MS };
