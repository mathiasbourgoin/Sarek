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
#      CUDA toolkit the deletion is invisible; whether the gate then reaches 13/13
#      depends on that host also having naga, glslangValidator and a loadable
#      libnvrtc, which is what was observed here (CodeRabbit, PR #384). The
#      invisibility is the general fact, the 13/13 was one machine. Being exact
#      about which way that fails:
#      prove-red.sh reads exit 0 as `DID NOT FAIL` and exits 1, so the symptom is
#      a spurious red for the whole repository on a CUDA host rather than a
#      credited red. Either way the mutation's verdict is a property of the
#      machine, which is what disqualifies it as evidence.
#
# So PATH is rebuilt from nothing: this fixture's bin/ first, then a private
# directory holding symlinks to a FIXED allowlist of base utilities -- bash, cat,
# find, grep, head, mktemp and rm for ci/assert-toolchain.sh, plus chmod, which
# this fixture's own bin/cc needs and the subject never calls. No tool under
# assertion is in that allowlist, so this fixture is their only possible source
# and a deleted stub is genuinely absent.
#
# "HERMETIC" HERE MEANS PATH-RESOLVED, and the difference is worth stating rather
# than leaving to be discovered. Absolute paths are not mediated by PATH and are
# not isolated: the shebangs reach /usr/bin/env, the NVRTC probe bin/cc writes
# runs /bin/sh, and the subject's own cuda_fp16.h search names /usr/local/cuda --
# that last one is the reason for the refusal below. The wrapper itself also runs
# host commands outside the sandbox, deliberately -- it has to build the sandbox
# with something -- and the list is exhaustive because a partial one is the defect
# this note exists to avoid: dirname and mktemp and ln while constructing it, find
# and head in the refusal below, and rm on the way out.
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
# mutation could not go red at all. PATH cannot hide an absolute path, so refuse
# rather than produce a mutation that cannot fail.
#
# THE BLAST RADIUS IS THE RUN'S VERDICT, not just this fixture. prove-red.sh
# dies on the first non-green baseline, so on a developer machine with a toolkit
# at /usr/local/cuda the whole run exits 2. Measured rather than assumed, because
# an earlier revision of this comment claimed all 7 subjects and 38 mutations are
# lost and that is not what happens: prove-red scans scripts/ before ci/, so the
# six subjects ahead of this one have already run and printed their verdicts. What
# is lost is this subject's baseline and its 5 mutations, plus the run's overall
# verdict. That is the intended trade -- a loud stop beats one silently inert
# mutation -- but it is stated because a gate that fails the run is a gate people
# disable. CI is unaffected: the `build` job is ubuntu-latest only, which ships
# no CUDA toolkit.
if [ -n "$(find -L /usr/local/cuda -name cuda_fp16.h 2>/dev/null | head -1)" ]; then
  echo "run-with-stubs: REFUSING to run. A real cuda_fp16.h exists under" >&2
  echo "/usr/local/cuda, which ci/assert-toolchain.sh searches unconditionally," >&2
  echo "so the cuda-header-tree-missing mutation could not go red on this host." >&2
  exit 2
fi

sysbin=$(mktemp -d) || exit 2

# Exhaustive for the PATH-resolved commands ci/assert-toolchain.sh runs (see the
# note on what "hermetic" means above; absolute paths are a separate matter), and
# deliberately short. A utility the subject starts using and that is not listed
# here makes the BASELINE red on every host at once, which is loud and immediate;
# the alternative -- inheriting the host PATH -- is what made the mutations inert.
# chmod is here for this fixture's own bin/cc, not for the subject.
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
