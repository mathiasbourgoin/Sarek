#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Runs ci/assert-toolchain.sh against a HERMETIC stub toolchain (backlog-176).
#
# WHY THIS WRAPPER EXISTS. ci/assert-toolchain.sh is green only inside the CI
# docker image: on a developer host it exits 1 because ptxas, nvdisasm and the
# CUDA header tree are absent (measured: 3 of 11 checks fail). scripts/prove-red.sh
# refuses a subject whose unmutated baseline is not green -- "every red below it
# would prove nothing" -- so the gate could never have a red path without a way
# to reach green off-image. That, and not neglect, is why it has gone two audits
# without one.
#
# WHY HERMETIC, AND NOT JUST "STUBS FIRST ON PATH". An earlier revision of this
# file stubbed only ptxas/nvdisasm/cuda_fp16.h and prepended them to the host
# PATH, leaving clang, glslangValidator, naga and libnvrtc to the host on the
# stated theory that faking a tool that is really there weakens the baseline for
# nothing. Both halves of that were wrong, and each was measured:
#
#   1. THE BASELINE WAS NOT REPRODUCIBLE WHERE IT RUNS. prove-red.sh is invoked
#      by the `build` job of .github/workflows/ci.yml, on a bare ubuntu-latest
#      runner -- NOT inside ci/Dockerfile's image. That runner has no naga, no
#      glslangValidator and no loadable libnvrtc, so the unstubbed baseline is
#      red on arrival there and prove-red.sh exits 2 for the whole repository.
#      "Genuinely present on a normal developer host" described one machine, not
#      the one that has to agree.
#   2. A PRESERVED HOST PATH DEFEATS THE MUTATIONS. `rm -f .../bin/ptxas` only
#      makes ptxas absent if nothing else on PATH provides it. On any host with a
#      CUDA toolkit the deletion is invisible, the gate stays green, and the
#      mutation is credited a red it did not earn -- the exact shape prove-red.sh
#      exists to catch (CodeRabbit, PR #384).
#
# So PATH is rebuilt from nothing: this fixture's bin/ first, then a private
# directory holding symlinks to a FIXED allowlist of base utilities the subject
# needs (bash, cat, chmod, find, grep, head, mktemp, rm). No tool under assertion
# is in that allowlist, so this fixture is their only possible source and a
# deleted stub is genuinely absent.
#
# WHAT THIS DOES NOT CLAIM. A green run here says the gate's DECISION LOGIC is
# sound: it counts checks, it accumulates failures, it trips on drift. It says
# nothing about whether the real CUDA toolkit works, which only the CI image can
# answer -- that is the `assert-toolchain` container step in ci.yml, a different
# assertion rather than a weaker copy of this one. The mutations are what make
# the green meaningful.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# Derived from this script's own location (fixture is <root>/scripts/prove-red-fixtures/
# assert-toolchain), so the wrapper works from any cwd and inside prove-red's
# scratch world without needing an env var the spec format cannot set.
root=$(cd "$here/../../.." && pwd)
subject="$root/ci/assert-toolchain.sh"

# Exit 2, never 1, on every refusal below: prove-red.sh reads 1 as "the gate
# failed as promised" and 2 as "the mechanism could not produce evidence".
if [ ! -x "$subject" ]; then
  echo "run-with-stubs: subject missing or not executable: $subject" >&2
  echo "Is ci/assert-toolchain.sh in the spec block's copy: set, and +x in git?" >&2
  exit 2
fi

# THE ONE HOST PATH THE SUBJECT HARDCODES. ci/assert-toolchain.sh searches for
# cuda_fp16.h under BOTH "${CUDA_PATH:-/usr/local/cuda}" AND a literal
# /usr/local/cuda, so on a host with a toolkit installed there, deleting this
# fixture's header would leave that check green and the cuda-header-tree-missing
# mutation would be credited a red it did not earn. PATH cannot hide an absolute
# path, so refuse rather than produce a mutation that cannot fail.
if [ -n "$(find -L /usr/local/cuda -name cuda_fp16.h 2>/dev/null | head -1)" ]; then
  echo "run-with-stubs: REFUSING to run. A real cuda_fp16.h exists under" >&2
  echo "/usr/local/cuda, which ci/assert-toolchain.sh searches unconditionally," >&2
  echo "so the cuda-header-tree-missing mutation could not go red on this host." >&2
  exit 2
fi

sysbin=$(mktemp -d) || exit 2

# The allowlist is exhaustive for the subject as written, and deliberately short.
# A utility the subject starts using and that is not listed here makes the
# BASELINE red on every host at once, which is loud and immediate; the
# alternative -- inheriting the host PATH -- is what made the mutations inert.
missing=""
for util in bash cat chmod find grep head mktemp rm; do
  p=$(type -P "$util" 2>/dev/null) || p=""
  if [ -z "$p" ]; then
    missing="$missing $util"
    continue
  fi
  ln -s "$p" "$sysbin/$util" || { rm -rf "$sysbin"; exit 2; }
done
if [ -n "$missing" ]; then
  echo "run-with-stubs: host is missing base utilities:$missing" >&2
  rm -rf "$sysbin"
  exit 2
fi

export PATH="$here/bin:$sysbin"
export CUDA_PATH="$here/cuda"

# Not exec: the temp directory above has to be removed, and an exec'd process
# never reaches a trap that would do it.
"$subject"
rc=$?
rm -rf "$sysbin"
exit "$rc"
