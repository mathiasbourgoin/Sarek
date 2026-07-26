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
  // A spawn-LAYER failure (ENOENT for a runtime that is not installed, EACCES,
  // EAGAIN) leaves stdout/stderr null. Hashing that produced a perfectly
  // stable digest of the empty string — a digest that LOOKS like a real
  // version fingerprint and compares equal across every such failure, so the
  // breaker could arm at, and later release from, a fingerprint that never
  // described a runtime at all. Treat it like the timeout case: an explicit
  // placeholder, never a hash of nothing.
  const spawnFailed = !!(result.error && !timedOut);
  const unavailable = timedOut || spawnFailed;
  const output = unavailable ? "" : (result.stdout || "") + (result.stderr || "");
  // The CAUSE is reported distinctly from the EFFECT. Both produce the
  // placeholder digest, but calling a missing binary a "timeout" is the same
  // misattribution this module exists to avoid.
  const reason = timedOut
    ? "version-probe-timeout"
    : spawnFailed
      ? `version-probe-spawn-error:${result.error.code}`
      : null;
  return { timedOut, unavailable, reason, output };
}

function computeDigests(runtimeName, runtimeBin, sandboxFlags) {
  const probe = probeVersion(runtimeBin);
  if (probe.unavailable) {
    return {
      digests: Object.fromEntries(sandboxFlags.map((flag) => [flag, `${runtimeName}:version-unavailable`])),
      versionProbeTimedOut: true, // retained: existing callers gate on this name
      versionProbeReason: probe.reason,
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
  return { digests, versionProbeTimedOut: false, versionProbeReason: null };
}

function computeDigest(runtimeName, runtimeBin, sandboxFlag) {
  const result = computeDigests(runtimeName, runtimeBin, [sandboxFlag]);
  return {
    digest: result.digests[sandboxFlag],
    versionProbeTimedOut: result.versionProbeTimedOut,
    versionProbeReason: result.versionProbeReason,
  };
}

module.exports = { computeDigest, computeDigests, probeVersion, VERSION_PROBE_TIMEOUT_MS };
