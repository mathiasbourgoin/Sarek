#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-B
# Copyright (c) 2012-2026 Mathias Bourgoin
#
# Refuse machine-identifying data in tracked files.
#
# WHY (backlog-168): benchmark output carried `system.hostname`, `system.kernel`
# and host `system.memory_gb`. Three personal machines ended up named in 263
# committed files -- in the FILENAMES as well as the payloads -- and in the
# published gh-pages dashboard. The producer is fixed; this stops it coming back.
#
# Three independent checks, because the leak had three independent shapes and
# closing one would not have caught the others:
#
#   1. PAYLOAD  -- a benchmark JSON carrying a removed field.
#   2. PATH     -- a tracked filename that looks like <host>_<bench>_<size>_<ts>,
#                  which is exactly the shape the 263 files had.
#   3. PRODUCER -- source that writes a removed field, or shells out to
#                  `hostname` outside the one sanctioned call site.
#
# Scope is `git ls-files`, not the filesystem: an untracked scratch file is not
# a disclosure, and a tracked one is -- regardless of what is on disk.
#
# Exit: 0 clean - 1 identifier found - 2 the check could not run (fails closed).

set -uo pipefail

cd "$(dirname "$0")/.." || { echo "::error::cannot reach repo root" >&2; exit 2; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "::error::not a git work tree -- this check reads 'git ls-files'" >&2
  exit 2
fi

DEIDENT="scripts/deidentify-benchmark-results.py"
[ -f "$DEIDENT" ] || { echo "::error::$DEIDENT missing" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "::error::python3 required" >&2; exit 2; }

failed=0

# --- 1. payloads -----------------------------------------------------------
# Reuse the scrubber's own --check so the gate and the fixer can never disagree
# about what counts as an identifier.
mapfile -t payloads < <(git ls-files -- 'benchmarks/results/*.json' \
                                       'gh-pages/benchmarks/data/*.json' 2>/dev/null)
if [ ${#payloads[@]} -gt 0 ]; then
  if ! python3 "$DEIDENT" --check "${payloads[@]}"; then
    failed=1
  fi
else
  echo "note: no tracked benchmark payloads (that is the expected state after"
  echo "      backlog-168 dropped the historical results)"
fi

# --- 2. paths --------------------------------------------------------------
# The 263 files were named <hostname>_<benchmark>_<size>_<ISO timestamp>.json.
# A machine LABEL is allowed in that position; a hostname is not, and the two
# are not distinguishable by pattern -- so require the label to be one of the
# derived <os>-<vendor> forms the producer can actually emit.
#
# Deliberately NOT a blocklist of the three known hostnames: that would pass a
# fourth machine, which is how this class survives.
bad_paths=$(git ls-files -- 'benchmarks/results/*' 2>/dev/null \
  | grep -E '/[^/]+_[a-z_0-9]+_[0-9]+_[0-9]{4}-[0-9]{2}-[0-9]{2}' \
  | grep -vE '/(linux|darwin|windows)-(nvidia|amd|intel|apple|unknown)_' || true)
if [ -n "$bad_paths" ]; then
  echo "::error::tracked result path(s) are not named after a derived machine label:"
  printf '  %s\n' $bad_paths
  echo "  Expected <os>-<vendor>_<benchmark>_<size>_<timestamp>.json."
  echo "  If that leading token is a hostname, it must not be committed."
  failed=1
fi

# --- 3. producer -----------------------------------------------------------
# `hostname` may be read at exactly one place: the check that REFUSES an
# operator override equal to the hostname. Anywhere else is a reintroduction.
SANCTIONED='benchmarks/system_info.ml'
producer_hits=$(git grep -nI -E 'open_process_in "hostname"|Unix\.gethostname' \
                  -- '*.ml' 2>/dev/null | grep -v "^${SANCTIONED}:" || true)
if [ -n "$producer_hits" ]; then
  echo "::error::source reads the hostname outside ${SANCTIONED}:"
  printf '  %s\n' "$producer_hits"
  failed=1
fi

# A JSON writer emitting a removed field. Matches the emit shape, not prose, so
# the comments explaining the removal do not trip it.
emit_hits=$(git grep -nI -E '\("(hostname|kernel)", *`String' -- '*.ml' 2>/dev/null || true)
if [ -n "$emit_hits" ]; then
  echo "::error::source emits a field removed by backlog-168:"
  printf '  %s\n' "$emit_hits"
  failed=1
fi

# The CSV header is a separate surface -- it leaked independently of the JSON.
csv_hits=$(git grep -nI -E '"(benchmark,timestamp,hostname|hostname,timestamp)' \
             -- '*.ml' 2>/dev/null || true)
if [ -n "$csv_hits" ]; then
  echo "::error::a CSV header still declares a hostname column:"
  printf '  %s\n' "$csv_hits"
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  echo ""
  echo "Fix: run scripts/deidentify-benchmark-results.py on the payload(s), or"
  echo "     drop the identifying field from the producer. See backlog-168."
  exit 1
fi

echo "OK -- ${#payloads[@]} tracked payload(s), no machine identifier in payloads, paths or producers"
exit 0
