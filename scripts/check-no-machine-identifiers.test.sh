#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-B
# Copyright (c) 2012-2026 Mathias Bourgoin
#
# Red-path harness for check-no-machine-identifiers.sh (backlog-168).
#
# The gate has FIVE independent red shapes, because the leak had five: an
# identifying payload, an identifying filename, a producer shelling out to
# `hostname`, a producer re-emitting the JSON field, and a CSV header regaining
# the column. A harness that exercised one would have let the other four back
# in -- which is how this class survived in the first place.
#
# Each case runs in a THROWAWAY git repo: the gate reads `git ls-files`, so the
# fixtures must actually be tracked, and doing that in the real repo would mean
# mutating its index. Nothing here touches the working repository.
#
# Exit: 0 all cases behaved - 1 a case did not - 2 setup failure (fails closed).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
GATE_SRC="$REPO_ROOT/scripts/check-no-machine-identifiers.sh"
DEIDENT_SRC="$REPO_ROOT/scripts/deidentify-benchmark-results.py"
for f in "$GATE_SRC" "$DEIDENT_SRC"; do
  [ -f "$f" ] || { echo "::error::missing $f" >&2; exit 2; }
done
command -v git >/dev/null 2>&1 || { echo "::error::git required" >&2; exit 2; }

pass=0
fail=0

# Build a minimal tracked repo containing the gate, the scrubber, and whatever
# fixture files the case needs, then run the gate inside it.
#   $1 name  $2 expected exit  $3 expected substring in output
# Remaining args: alternating <relative path> <content> pairs.
check() {
  local name="$1" want_exit="$2" want_text="$3"; shift 3
  local tmp
  tmp="$(mktemp -d)" || { echo "::error::mktemp failed" >&2; exit 2; }

  mkdir -p "$tmp/scripts"
  cp "$GATE_SRC" "$tmp/scripts/check-no-machine-identifiers.sh"
  cp "$DEIDENT_SRC" "$tmp/scripts/deidentify-benchmark-results.py"

  while [ "$#" -gt 0 ]; do
    local path="$1" content="$2"; shift 2
    mkdir -p "$tmp/$(dirname "$path")"
    printf '%s\n' "$content" > "$tmp/$path"
  done

  (
    cd "$tmp" || exit 2
    git init -q .
    git add -A
  ) >/dev/null 2>&1 || { echo "::error::[$name] fixture repo setup failed" >&2; rm -rf "$tmp"; exit 2; }

  local out got
  out="$(bash "$tmp/scripts/check-no-machine-identifiers.sh" 2>&1)"
  got=$?

  if [ "$got" -eq "$want_exit" ] && printf '%s' "$out" | grep -qF "$want_text"; then
    echo "PASS $name (exit $got)"
    pass=$((pass + 1))
  else
    echo "FAIL $name -- wanted exit $want_exit containing '$want_text'"
    echo "     got exit $got:"
    printf '%s\n' "$out" | sed 's/^/       /'
    fail=$((fail + 1))
  fi
  rm -rf "$tmp"
}

CLEAN_PAYLOAD='{"benchmark":{"name":"vector_add"},"system":{"machine":"linux-amd","os":"Linux","cpu":{"model":"X","cores":1,"threads":1},"devices":[]},"results":[]}'
DIRTY_PAYLOAD='{"benchmark":{"name":"vector_add"},"system":{"hostname":"myhost","os":"Linux","kernel":"6.1.0","cpu":{"model":"X","cores":1,"threads":1},"devices":[]},"results":[]}'
GOOD_NAME='benchmarks/results/linux-amd_vector_add_1024_2026-07-30T00-00-00.json'
BAD_NAME='benchmarks/results/myhost_vector_add_1024_2026-07-30T00-00-00.json'

# --- green baselines -------------------------------------------------------
# Pinned as green so a future tightening cannot start refusing legitimate data
# and be mistaken for a working gate.
check "green: no payloads at all is clean" 0 "no machine identifier" \
  "README.md" "placeholder"

check "green: a properly labelled payload and path" 0 "no machine identifier" \
  "$GOOD_NAME" "$CLEAN_PAYLOAD"

# --- red 1: payload --------------------------------------------------------
check "red: payload carries hostname and kernel" 1 "carries hostname, kernel" \
  "$GOOD_NAME" "$DIRTY_PAYLOAD"

# --- red 2: filename, with a payload that is ALREADY clean -----------------
# The distinguishing case: scrubbing payloads does not fix a filename, and this
# is the shape the 263 committed files actually had.
check "red: filename is a hostname though the payload is clean" 1 \
  "not named after a derived machine label" \
  "$BAD_NAME" "$CLEAN_PAYLOAD"

# --- red 3: producer reads the hostname ------------------------------------
check "red: source shells out to hostname outside system_info.ml" 1 \
  "reads the hostname outside" \
  "benchmarks/leak.ml" 'let h () = Unix.open_process_in "hostname"'

check "red: source uses Unix.gethostname" 1 "reads the hostname outside" \
  "benchmarks/leak.ml" 'let h () = Unix.gethostname ()'

# --- red 4: producer re-emits the JSON field -------------------------------
check "red: JSON writer emits a hostname field" 1 \
  "emits a field removed by backlog-168" \
  "benchmarks/out.ml" '("hostname", `String info.machine);'

check "red: JSON writer emits a kernel field" 1 \
  "emits a field removed by backlog-168" \
  "benchmarks/out.ml" '("kernel", `String info.kernel);'

# --- red 5: CSV header regains the column ----------------------------------
# A separate surface: the CSV leaked independently of the JSON, so a JSON-only
# check would have passed this.
check "red: CSV header declares a hostname column" 1 \
  "CSV header still declares a hostname column" \
  "benchmarks/out.ml" '  "benchmark,timestamp,hostname,device_id\n"'

# --- the sanctioned call site must NOT trip the producer check -------------
# system_info.ml legitimately reads the hostname, to REFUSE an override equal
# to it. If this were red, the gate would forbid its own safety check.
check "green: system_info.ml may read the hostname" 0 "no machine identifier" \
  "benchmarks/system_info.ml" 'let x () = Unix.open_process_in "hostname"'

# --- fails closed ----------------------------------------------------------
tmp="$(mktemp -d)" || exit 2
mkdir -p "$tmp/scripts"
cp "$GATE_SRC" "$tmp/scripts/check-no-machine-identifiers.sh"
cp "$DEIDENT_SRC" "$tmp/scripts/deidentify-benchmark-results.py"
# No `git init`: not a work tree.
out="$(bash "$tmp/scripts/check-no-machine-identifiers.sh" 2>&1)"; got=$?
if [ "$got" -eq 2 ] && printf '%s' "$out" | grep -qF "not a git work tree"; then
  echo "PASS red: outside a git work tree is exit 2, not a pass (exit $got)"
  pass=$((pass + 1))
else
  echo "FAIL red: outside a git work tree -- wanted exit 2, got $got: $out"
  fail=$((fail + 1))
fi
rm -rf "$tmp"

# Missing scrubber must also fail closed, not silently skip the payload check.
tmp="$(mktemp -d)" || exit 2
mkdir -p "$tmp/scripts"
cp "$GATE_SRC" "$tmp/scripts/check-no-machine-identifiers.sh"
(cd "$tmp" && git init -q . && git add -A) >/dev/null 2>&1
out="$(bash "$tmp/scripts/check-no-machine-identifiers.sh" 2>&1)"; got=$?
if [ "$got" -eq 2 ]; then
  echo "PASS red: missing scrubber is exit 2 (exit $got)"
  pass=$((pass + 1))
else
  echo "FAIL red: missing scrubber -- wanted exit 2, got $got: $out"
  fail=$((fail + 1))
fi
rm -rf "$tmp"

# --- the scrubber must not rewrite an already-clean payload ----------------
# It re-serializes (indent=2, sort_keys=True), so an unconditional write turned
# every already-clean file it was pointed at into a pure-formatting diff. Only
# a scrub that actually stripped or relabelled something may touch the file.
tmp="$(mktemp -d)" || exit 2
printf '%s\n' '{
    "system": {
        "machine": "linux-nvidia",
  "os": "Linux",
 "devices": [{"name":"NVIDIA RTX 3090"}]
    },
 "results": []
}' > "$tmp/clean.json"
cp "$tmp/clean.json" "$tmp/clean.before"
out="$(python3 "$DEIDENT_SRC" "$tmp/clean.json" 2>&1)"; got=$?
if [ "$got" -eq 0 ] && cmp -s "$tmp/clean.before" "$tmp/clean.json"; then
  echo "PASS green: an already-clean payload is left byte-identical"
  pass=$((pass + 1))
else
  echo "FAIL green: already-clean payload was rewritten (exit $got)"
  diff "$tmp/clean.before" "$tmp/clean.json" | sed 's/^/       /'
  fail=$((fail + 1))
fi
# The other polarity, so the skip cannot be a blanket "never write": a payload
# that DOES carry an identifier must still be rewritten, and end up clean.
printf '%s\n' '{"system":{"hostname":"myhost","kernel":"6.1.0","os":"Linux",
 "devices":[{"name":"NVIDIA RTX 3090"}]},"results":[]}' > "$tmp/dirty.json"
cp "$tmp/dirty.json" "$tmp/dirty.before"
out="$(python3 "$DEIDENT_SRC" "$tmp/dirty.json" 2>&1)"; got=$?
if [ "$got" -eq 0 ] && ! cmp -s "$tmp/dirty.before" "$tmp/dirty.json" \
   && python3 "$DEIDENT_SRC" --check "$tmp/dirty.json" >/dev/null 2>&1; then
  echo "PASS red: an identifying payload is still rewritten, and comes out clean"
  pass=$((pass + 1))
else
  echo "FAIL red: identifying payload was not rewritten/cleaned (exit $got): $out"
  fail=$((fail + 1))
fi
# The scrubber RELABELS, so its label derivation must agree with the producer's
# (System_info.gpu_vendor_of). While they disagreed, a correctly labelled
# Apple-Silicon payload was rewritten darwin-apple -> darwin-unknown: the
# scrubber corrupted a clean file and desynchronized it from its own filename,
# which carries the producer's label. The GPU's name equalling the CPU's is the
# case that split them, so that is the case pinned.
printf '%s\n' '{
  "results": [],
  "system": {
    "cpu": {
      "cores": 12,
      "model": "Apple M4 Max",
      "threads": 12
    },
    "devices": [
      {
        "framework": "Metal",
        "name": "Apple M4 Max"
      }
    ],
    "machine": "darwin-apple",
    "os": "Darwin"
  }
}' > "$tmp/apple.json"
cp "$tmp/apple.json" "$tmp/apple.before"
out="$(python3 "$DEIDENT_SRC" "$tmp/apple.json" 2>&1)"; got=$?
if [ "$got" -eq 0 ] && cmp -s "$tmp/apple.before" "$tmp/apple.json"; then
  echo "PASS green: scrubber agrees with the producer on darwin-apple"
  pass=$((pass + 1))
else
  echo "FAIL green: scrubber relabelled a correct darwin-apple payload (exit $got)"
  diff "$tmp/apple.before" "$tmp/apple.json" | sed 's/^/       /'
  fail=$((fail + 1))
fi
rm -rf "$tmp"

# --- the machine label must be filename-safe before it reaches a glob -------
# run_all_benchmarks.sh interpolates system.machine into an `rm` glob. The
# label is operator-settable via SAREK_BENCH_MACHINE (get_machine_label only
# refuses the hostname), so `*` and `../..` are reachable. Pin the predicate
# the script guards with -- these are the shapes that must not pass.
#
# The pattern is EXTRACTED from the script rather than restated here. A copy
# would let the two drift: someone loosens the script, the test keeps asserting
# the old pattern and stays green while `*` starts passing again.
RUNNER="$REPO_ROOT/benchmarks/run_all_benchmarks.sh"
[ -f "$RUNNER" ] || { echo "::error::missing $RUNNER" >&2; exit 2; }
LABEL_RE="$(grep -oE "grep -qE '[^']+'" "$RUNNER" | head -1 | sed "s/.*'\(.*\)'/\1/")"
if [ -z "$LABEL_RE" ]; then
  echo "::error::could not extract the machine-label pattern from $RUNNER --" >&2
  echo "::error::the guard may have been removed or reworded; failing closed" >&2
  exit 2
fi
echo "note: machine-label pattern extracted from run_all_benchmarks.sh: $LABEL_RE"
label_case() {
  local label="$1" want="$2"   # want: safe | unsafe
  local got="unsafe"
  printf '%s' "$label" | grep -qE "$LABEL_RE" && got="safe"
  if [ "$got" = "$want" ]; then
    echo "PASS label '$label' is $want"
    pass=$((pass + 1))
  else
    echo "FAIL label '$label' -- wanted $want, got $got"
    fail=$((fail + 1))
  fi
}
# Derived labels the producer actually emits must keep working, or the cleanup
# silently stops cleaning and results accumulate under a duplicate key.
label_case "linux-nvidia" safe
label_case "darwin-apple" safe
label_case "linux-unknown" safe
# The destructive shapes.
label_case "*"              unsafe   # would expand to every machine's results
label_case "../../etc"      unsafe   # would leave benchmarks/results entirely
label_case "linux nvidia"   unsafe   # word-splits into two glob arguments
label_case "-rf"            unsafe   # would be read as an option without rm --
label_case ""               unsafe

echo ""
if [ "$fail" -ne 0 ]; then
  echo "check-no-machine-identifiers.test.sh: $fail case(s) FAILED, $pass passed"
  exit 1
fi
echo "check-no-machine-identifiers.test.sh: all $pass cases passed"
exit 0
