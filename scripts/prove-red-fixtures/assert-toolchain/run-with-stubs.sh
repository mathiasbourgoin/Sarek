#!/usr/bin/env bash
# Runs ci/assert-toolchain.sh against a hermetic stub toolchain (backlog-176).
#
# WHY THIS WRAPPER EXISTS. ci/assert-toolchain.sh is green only inside the CI
# docker image: on a developer host it exits 1 because ptxas, nvdisasm and the
# CUDA header tree are absent (measured: 3 of 11 checks fail). scripts/prove-red.sh
# refuses a subject whose unmutated baseline is not green -- "every red below it
# would prove nothing" -- so the gate could never have a red path without a way
# to reach green off-image. That, and not neglect, is why it has gone two audits
# without one.
#
# The stubs beside this file supply exactly the contracts the gate depends on and
# no more; the wrapper sets PATH and CUDA_PATH so the gate finds them. Everything
# else the gate probes (clang, glslangValidator, naga) is genuinely present on a
# normal developer host and is deliberately NOT stubbed -- faking a tool that is
# really there would weaken the baseline for no gain.
#
# WHAT THIS DOES NOT CLAIM. A green run here says the gate's DECISION LOGIC is
# sound: it counts checks, it accumulates failures, it trips on drift. It says
# nothing about whether the real CUDA toolkit works, which only the CI image can
# answer. The mutations are what make the green meaningful.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export PATH="$here/bin:$PATH"
export CUDA_PATH="$here/cuda"
# Derived from this script's own location (fixture is <root>/scripts/prove-red-fixtures/
# assert-toolchain), so the wrapper works from any cwd and inside prove-red's
# scratch world without needing an env var the spec format cannot set.
root=$(cd "$here/../../.." && pwd)
exec "$root/ci/assert-toolchain.sh"
