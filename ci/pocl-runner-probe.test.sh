#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Covering test for ci/pocl-runner-probe.sh (backlog-99).
#
# The probe is informational and exits 0 by design, which is exactly the shape
# that rots unnoticed: nothing downstream reads its exit code, so a probe that
# silently stopped measuring would keep printing a verdict nobody could tell
# from a real one. What has to hold is that the VERDICT tracks the stages, and
# in particular that no combination of failures can produce POCL-WORKS.
#
# It runs on any machine, with or without pocl. The stage results come from a
# stub compiler that emits chosen PROBE lines, so the classifier is exercised
# against outcomes this host cannot actually produce -- including the two that
# matter most and that no CI runner may ever exhibit: a pocl that compiles
# invalid source, and a pocl that enumerates but cannot build.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/pocl-runner-probe.sh"
[ -x "$PROBE" ] || { echo "FAIL: $PROBE not found or not executable"; exit 2; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/pocl-probe-test.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0
check() {
  local desc="$1" got="$2" want="$3"
  if [ "$got" = "$want" ]; then
    echo "  PASS: $desc"; pass=$((pass + 1))
  else
    echo "  FAIL: $desc -- expected '$want', got '$got'"; fail=$((fail + 1))
  fi
}

# A stub `cc`: ignores the source, emits a probe binary that prints whatever
# PROBE lines $STAGES holds. `exit 7` makes it a stub that FAILS to build.
mkdir -p "$TMP/bin"
cat > "$TMP/bin/stubcc" <<'STUB'
#!/usr/bin/env bash
[ "${STUB_CC_FAILS:-0}" = "1" ] && { echo "fatal error: CL/cl.h: No such file"; exit 1; }
out=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out="$2"; shift 2;; *) shift;; esac
done
{ echo '#!/usr/bin/env bash'; printf '%s\n' "$STAGES" | while IFS= read -r l; do
    [ -n "$l" ] && echo "echo '$l'"; done; } > "$out"
chmod +x "$out"
STUB
chmod +x "$TMP/bin/stubcc"

# An empty vendors dir: stage A must report fail rather than reading the host's
# real /etc/OpenCL/vendors, which would make this test's result depend on the
# machine running it.
mkdir -p "$TMP/vendors"

run_probe() {
  STAGES="$1" \
  STUB_CC_FAILS="${2:-0}" \
  CC="$TMP/bin/stubcc" \
  OCL_ICD_VENDORS="$TMP/vendors" \
  POCL_PROBE_WORKDIR="$TMP/w$RANDOM" \
    "$PROBE" ${3:-} 2>&1
}
verdict_of() { echo "$1" | /usr/bin/grep '^POCL_PROBE_VERDICT=' | /usr/bin/cut -d= -f2; }

ALL_GOOD='PROBE enumerate pass
PROBE compile pass
PROBE execute pass
PROBE reject pass'

echo "ci/pocl-runner-probe.sh covering test"

# Positive control first. Without it every assertion below could be satisfied
# by a script that always says POCL-ABSENT.
out="$(run_probe "$ALL_GOOD")"
check "all stages pass -> POCL-WORKS" "$(verdict_of "$out")" "POCL-WORKS"

# The load-bearing negatives. Each drops exactly one stage.
out="$(run_probe 'PROBE enumerate pass
PROBE compile pass
PROBE execute pass
PROBE reject fail')"
check "compiles invalid source -> POCL-UNTRUSTWORTHY" \
  "$(verdict_of "$out")" "POCL-UNTRUSTWORTHY"

out="$(run_probe 'PROBE enumerate pass
PROBE compile fail')"
check "enumerates but cannot compile -> POCL-ENUMERATES-ONLY (the backlog-79 shape)" \
  "$(verdict_of "$out")" "POCL-ENUMERATES-ONLY"

out="$(run_probe 'PROBE enumerate fail')"
check "no pocl device -> POCL-ABSENT" "$(verdict_of "$out")" "POCL-ABSENT"

# A build that succeeds is not a kernel that ran. This is the assertion that
# stops the probe from being downgraded to a compile check later.
out="$(run_probe 'PROBE enumerate pass
PROBE compile pass
PROBE execute fail
PROBE reject pass')"
check "compiles but does not run -> not POCL-WORKS" \
  "$([ "$(verdict_of "$out")" = "POCL-WORKS" ] && echo works || echo "not-works")" \
  "not-works"

# An unrun stage must never read as a passed one -- the mechanism this whole
# probe is a reaction to. A compiler that cannot build the probe leaves every
# stage `unknown`, and unknown is not pass.
out="$(run_probe "$ALL_GOOD" 1)"
check "probe that fails to BUILD -> POCL-ABSENT, not POCL-WORKS" \
  "$(verdict_of "$out")" "POCL-ABSENT"
check "unbuilt probe marks stages unbuilt, not pass" \
  "$(echo "$out" | /usr/bin/grep -c 'compile *unbuilt')" "1"

# Stage A is independent of the kernel stages and must read the vendors dir it
# is told about, not the host's.
check "empty vendors dir -> icd fail" \
  "$(echo "$out" | /usr/bin/grep -c 'icd *fail')" "1"
mkdir -p "$TMP/vendors2" && echo "libpocl.so.2" > "$TMP/vendors2/pocl.icd"
out2="$(STAGES="$ALL_GOOD" CC="$TMP/bin/stubcc" OCL_ICD_VENDORS="$TMP/vendors2" \
  POCL_PROBE_WORKDIR="$TMP/wv" "$PROBE" 2>&1)"
check "a pocl.icd present -> icd pass" \
  "$(echo "$out2" | /usr/bin/grep -c 'icd *pass')" "1"

# --strict turns the verdict into an exit code. The probe is non-gating in CI,
# so this is the only place its exit code is meaningful -- and the only way a
# future promotion to a gate can be trusted.
run_probe "$ALL_GOOD" 0 --strict >/dev/null 2>&1
check "--strict exits 0 when pocl works" "$?" "0"
run_probe 'PROBE enumerate fail' 0 --strict >/dev/null 2>&1
check "--strict exits non-zero when it does not" \
  "$([ "$?" -ne 0 ] && echo nonzero || echo zero)" "nonzero"
run_probe 'PROBE enumerate fail' >/dev/null 2>&1
check "without --strict the probe stays non-gating (exit 0)" "$?" "0"

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || exit 1
echo "OK: the pocl probe's verdict tracks its stages, an unrun stage never"
echo "    reads as a pass, and only a compiled+executed+validating pocl"
echo "    reports POCL-WORKS (backlog-99)"
