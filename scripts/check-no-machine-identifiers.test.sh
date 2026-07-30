#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-B
# Copyright (c) 2012-2026 Mathias Bourgoin
#
# Red-path harness for check-no-machine-identifiers.sh (backlog-168, -168b).
#
# The gate has FIVE independent red shapes, because the leak had five: an
# identifying payload, an identifying filename, a producer shelling out to
# `hostname`, a producer re-emitting the JSON field, and a CSV header regaining
# the column. A harness that exercised one would have let the other four back
# in -- which is how this class survived in the first place.
#
# It also covers the label SHAPE itself (backlog-168b), which is enforced in two
# languages -- the gate's sourced bash regex and the producer's OCaml validation
# of SAREK_BENCH_MACHINE. Those are not compared by a comment saying they must
# agree: both are EXECUTED here over one shared case table, and the producer's
# override policy is executed too, because the label an operator may set and the
# label that may be committed diverging is precisely the bug -168b fixed.
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
SHAPE_SRC="$REPO_ROOT/scripts/machine-label-shape.sh"
LABEL_ML="$REPO_ROOT/benchmarks/machine_label.ml"
for f in "$GATE_SRC" "$DEIDENT_SRC" "$SHAPE_SRC" "$LABEL_ML"; do
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
  cp "$SHAPE_SRC" "$tmp/scripts/machine-label-shape.sh"

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
# The disambiguating-suffix shapes. A fleet with two same-hardware machines is
# the reason SAREK_BENCH_MACHINE exists; before the suffix was admitted, the
# label it needs to set produced filenames this gate refused to commit -- the
# override worked everywhere except for its own purpose.
SUFFIX_NAME='benchmarks/results/linux-amd-b_vector_add_1024_2026-07-30T00-00-00.json'
SUFFIX_PAYLOAD='{"benchmark":{"name":"vector_add"},"system":{"machine":"linux-amd-b","os":"Linux","cpu":{"model":"X","cores":1,"threads":1},"devices":[]},"results":[]}'
LONG_SUFFIX_NAME='benchmarks/results/linux-amd-123456789_vector_add_1024_2026-07-30T00-00-00.json'
UPPER_SUFFIX_NAME='benchmarks/results/linux-amd-B_vector_add_1024_2026-07-30T00-00-00.json'
HOSTNAME_NAME='benchmarks/results/drangleic_vector_add_1024_2026-07-30T00-00-00.json'
# A hostname-named file under a DIRECTORY that carries a label-shaped name. The
# allowlist used to be the label alone, unanchored, so this path matched it two
# components away from the filename it was supposed to be vouching for and the
# hostname was committable (CodeRabbit, PR #389).
DIR_DISGUISE_NAME='benchmarks/results/linux-amd_/drangleic_vector_add_1024_2026-07-30T00-00-00.json'
# The other polarity of that anchoring, so binding the label to the filename
# cannot become "the label must be in the top directory": a correctly named file
# nested one level down stays committable.
NESTED_GOOD_NAME='benchmarks/results/archive/linux-amd_vector_add_1024_2026-07-30T00-00-00.json'

# --- green baselines -------------------------------------------------------
# Pinned as green so a future tightening cannot start refusing legitimate data
# and be mistaken for a working gate.
check "green: no payloads at all is clean" 0 "no machine identifier" \
  "README.md" "placeholder"

check "green: a properly labelled payload and path" 0 "no machine identifier" \
  "$GOOD_NAME" "$CLEAN_PAYLOAD"

check "green: a label carrying the disambiguating suffix is committable" 0 \
  "no machine identifier" \
  "$SUFFIX_NAME" "$SUFFIX_PAYLOAD"

# --- red 2b: the suffix is BOUNDED, and bounded to [a-z0-9] ----------------
# The suffix is the one free-form field in the label, so its bound is what
# stops it becoming somewhere a hostname can ride along. Both polarities of
# the bound are pinned, since a widening that admitted these would be
# invisible against the green above.
check "red: suffix over the length bound" 1 \
  "not named after a derived machine label" \
  "$LONG_SUFFIX_NAME" "$CLEAN_PAYLOAD"

check "red: suffix with a character outside [a-z0-9]" 1 \
  "not named after a derived machine label" \
  "$UPPER_SUFFIX_NAME" "$CLEAN_PAYLOAD"

# The whole point of admitting a suffix was NOT to admit this. A bare hostname
# has no <os>-<vendor> prefix, and the suffix cannot supply one.
check "red: a bare hostname is still refused after the suffix was admitted" 1 \
  "not named after a derived machine label" \
  "$HOSTNAME_NAME" "$CLEAN_PAYLOAD"

# --- red 2c: the allowlist is bound to the FILENAME, not to the path ---------
# A label anywhere in the path used to excuse the filename, so a directory named
# `linux-amd_` vouched for a hostname beside it.
check "red: a label-shaped directory does not excuse a hostname filename" 1 \
  "not named after a derived machine label" \
  "$DIR_DISGUISE_NAME" "$CLEAN_PAYLOAD"

check "green: a correctly labelled file in a subdirectory is still committable" 0 \
  "no machine identifier" \
  "$NESTED_GOOD_NAME" "$CLEAN_PAYLOAD"

# --- fails closed on a missing shape definition ----------------------------
# The shape now lives in scripts/machine-label-shape.sh so the gate and the
# producer cannot drift. A gate that cannot read the shape cannot tell a label
# from a hostname, so it must not report a pass.
tmp_noshape="$(mktemp -d)" || exit 2
mkdir -p "$tmp_noshape/scripts"
cp "$GATE_SRC" "$tmp_noshape/scripts/check-no-machine-identifiers.sh"
cp "$DEIDENT_SRC" "$tmp_noshape/scripts/deidentify-benchmark-results.py"
(cd "$tmp_noshape" && git init -q . && git add -A) >/dev/null 2>&1 \
  || { echo "::error::fixture repo setup failed (missing-shape case)" >&2
       rm -rf "$tmp_noshape"; exit 2; }
out="$(bash "$tmp_noshape/scripts/check-no-machine-identifiers.sh" 2>&1)"; got=$?
if [ "$got" -eq 2 ] && printf '%s' "$out" | grep -qF "cannot decide what a legal machine label is"; then
  echo "PASS red: missing machine-label-shape.sh is exit 2 (exit $got)"
  pass=$((pass + 1))
else
  echo "FAIL red: missing machine-label-shape.sh -- wanted exit 2, got $got: $out"
  fail=$((fail + 1))
fi
rm -rf "$tmp_noshape"

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
# The suffixed label too: this predicate decides whether OLD RESULTS get cleaned
# up, so refusing the suffix would mean the one machine that needs the
# disambiguator is the one whose stale results accumulate under its key.
label_case "linux-amd-b" safe
label_case "linux-amd-rack12" safe
# Still bounded -- looser than the commit shape on the os/vendor tokens, not on
# the suffix, so this cannot become a free-form field either.
label_case "linux-amd-123456789" unsafe
label_case "linux-amd-B" unsafe
label_case "linux-amd-a-b" unsafe
# The destructive shapes.
label_case "*"              unsafe   # would expand to every machine's results
label_case "../../etc"      unsafe   # would leave benchmarks/results entirely
label_case "linux nvidia"   unsafe   # word-splits into two glob arguments
label_case "-rf"            unsafe   # would be read as an option without rm --
label_case ""               unsafe

# --- the scrubber must not relabel a legal suffix away ----------------------
# It RELABELS, and it derives the label from hardware -- which has no way of
# knowing about an operator-set suffix. Left alone it rewrote linux-amd-b ->
# linux-amd: it strips the only thing separating two same-hardware machines and
# desynchronizes the payload from its own filename, which still carries the
# suffix. Same defect class as the darwin-apple relabelling above.
tmp="$(mktemp -d)" || exit 2
printf '%s\n' '{
  "results": [],
  "system": {
    "cpu": {
      "cores": 16,
      "model": "AMD Ryzen 9 7950X 16-Core Processor",
      "threads": 32
    },
    "devices": [
      {
        "framework": "HIP",
        "name": "Radeon RX 7900 XTX"
      }
    ],
    "machine": "linux-amd-b",
    "os": "Linux"
  }
}' > "$tmp/suffix.json"
cp "$tmp/suffix.json" "$tmp/suffix.before"
out="$(python3 "$DEIDENT_SRC" "$tmp/suffix.json" 2>&1)"; got=$?
if [ "$got" -eq 0 ] && cmp -s "$tmp/suffix.before" "$tmp/suffix.json"; then
  echo "PASS green: scrubber keeps a legal disambiguating suffix"
  pass=$((pass + 1))
else
  echo "FAIL green: scrubber relabelled a suffixed payload (exit $got)"
  diff "$tmp/suffix.before" "$tmp/suffix.json" | sed 's/^/       /'
  fail=$((fail + 1))
fi
# The other polarity, so "keep the suffix" cannot become "keep anything": a
# label whose suffix is NOT legal must still be relabelled to the derived one.
printf '%s\n' '{"system":{"machine":"linux-amd-NOTLEGAL9","os":"Linux",
 "cpu":{"model":"AMD Ryzen 9 7950X 16-Core Processor","cores":16,"threads":32},
 "devices":[{"framework":"HIP","name":"Radeon RX 7900 XTX"}]},"results":[]}' \
  > "$tmp/badsuffix.json"
out="$(python3 "$DEIDENT_SRC" "$tmp/badsuffix.json" 2>&1)"; got=$?
if [ "$got" -eq 0 ] && grep -q '"machine": "linux-amd"' "$tmp/badsuffix.json"; then
  echo "PASS red: scrubber relabels a suffix that is not legal"
  pass=$((pass + 1))
else
  echo "FAIL red: an illegal suffix survived the scrubber (exit $got): $out"
  sed 's/^/       /' "$tmp/badsuffix.json"
  fail=$((fail + 1))
fi
rm -rf "$tmp"

# --- one shape, two implementations, compared by EXECUTION ------------------
# The label shape is enforced twice: by this gate, on what may be committed
# (scripts/machine-label-shape.sh, sourced), and by the producer, on what an
# operator may set through SAREK_BENCH_MACHINE (benchmarks/machine_label.ml).
# They MUST agree -- an override the producer accepts and the gate later
# refuses is exactly the contradiction this change fixes, and it is invisible
# until commit time.
#
# So they are not merely commented as "keep these in sync". Both are run over
# ONE case table below, and the literal patterns are compared too. A comment
# cannot fail; this can.
# shellcheck source=scripts/machine-label-shape.sh
. "$SHAPE_SRC" || { echo "::error::cannot source $SHAPE_SRC" >&2; exit 2; }
[ -n "${MACHINE_LABEL_SHAPE:-}" ] \
  || { echo "::error::$SHAPE_SRC defined no MACHINE_LABEL_SHAPE" >&2; exit 2; }
command -v ocaml >/dev/null 2>&1 \
  || { echo "::error::ocaml required to run benchmarks/machine_label.ml" >&2; exit 2; }

# 1. the two pattern literals -----------------------------------------------
# The first string literal at or after `let shape_doc`, so ocamlformat moving it
# onto its own line cannot turn this into a silent no-extraction (which would
# hit the fail-closed branch below rather than pass, but still stop comparing).
ML_SHAPE_DOC="$(awk '
  /^let shape_doc/ { seen = 1 }
  seen && match($0, /"[^"]*"/) { print substr($0, RSTART + 1, RLENGTH - 2); exit }
' "$LABEL_ML")"
if [ -z "$ML_SHAPE_DOC" ]; then
  echo "::error::could not extract shape_doc from $LABEL_ML -- it may have been" >&2
  echo "::error::renamed or reformatted; failing closed rather than skipping" >&2
  exit 2
fi
if [ "$ML_SHAPE_DOC" = "$MACHINE_LABEL_SHAPE" ]; then
  echo "PASS green: machine_label.ml's shape_doc is byte-identical to MACHINE_LABEL_SHAPE"
  pass=$((pass + 1))
else
  echo "FAIL the two statements of the label shape have drifted:"
  echo "       bash: $MACHINE_LABEL_SHAPE"
  echo "       ocaml: $ML_SHAPE_DOC"
  fail=$((fail + 1))
fi

# 2. the two implementations, over one table --------------------------------
# `<empty>` stands for the empty label, which cannot be a line of its own here.
SHAPE_CASES="
linux-amd safe
darwin-apple safe
windows-nvidia safe
linux-unknown safe
linux-nvidia-b safe
linux-amd-2 safe
linux-amd-lab2 safe
linux-amd-r2d2 safe
linux-amd-12345678 safe
linux-amd-123456789 unsafe
linux-amd- unsafe
linux-amd-B unsafe
linux-amd-a_b unsafe
linux-amd-a.b unsafe
linux-amd-a-b unsafe
linux-amd-b- unsafe
drangleic unsafe
myhost unsafe
freebsd-nvidia unsafe
linux-banana unsafe
linux unsafe
* unsafe
../../etc unsafe
<empty> unsafe
"
DRIVERS="$(mktemp -d)" || exit 2
# machine_label.ml is Stdlib-only precisely so it can be #use'd here: this runs
# the SHIPPING implementation, not a restatement of it.
cat > "$DRIVERS/shape_driver.ml" <<OCAML_DRIVER
#use "$LABEL_ML";;
let () =
  try
    while true do
      let raw = input_line stdin in
      let label = if raw = "<empty>" then "" else raw in
      print_string
        (raw ^ " " ^ (if is_wellformed label then "safe" else "unsafe") ^ "\n")
    done
  with End_of_file -> ()
OCAML_DRIVER
OCAML_OUT="$(printf '%s\n' "$SHAPE_CASES" | grep -v '^[[:space:]]*$' | awk '{print $1}' \
  | ocaml "$DRIVERS/shape_driver.ml" 2>&1)"
ocaml_status=$?
if [ "$ocaml_status" -ne 0 ]; then
  echo "::error::running $LABEL_ML failed (exit $ocaml_status):" >&2
  printf '%s\n' "$OCAML_OUT" >&2
  exit 2
fi
while read -r raw want; do
  [ -n "$raw" ] || continue
  label="$raw"; [ "$raw" = "<empty>" ] && label=""
  bash_got="unsafe"
  printf '%s' "$label" | grep -qE "$MACHINE_LABEL_SHAPE" && bash_got="safe"
  ocaml_got="$(printf '%s\n' "$OCAML_OUT" | awk -v k="$raw" '$1 == k {print $2}' | head -1)"
  if [ "$bash_got" = "$want" ] && [ "$ocaml_got" = "$want" ]; then
    echo "PASS shape '$raw' is $want in both implementations"
    pass=$((pass + 1))
  else
    echo "FAIL shape '$raw' -- wanted $want; bash said ${bash_got:-<none>}," \
         "ocaml said ${ocaml_got:-<none>}"
    fail=$((fail + 1))
  fi
done <<EOF
$(printf '%s\n' "$SHAPE_CASES" | grep -v '^[[:space:]]*$')
EOF

# --- the producer's override policy, executed ------------------------------
# Shape validation was ADDED to the override; the hostname refusal was already
# there and is load-bearing. Both polarities are pinned, and so is their ORDER:
# a hostname that happens to have a legal shape must still be refused as a
# hostname, or adding shape validation would have opened a way round the very
# refusal it was added next to.
cat > "$DRIVERS/resolve_driver.ml" <<OCAML_DRIVER
#use "$LABEL_ML";;
(* Which refusal fired is part of the assertion, not decoration: "refused" alone
   would let the hostname case be satisfied by the shape check, and then a
   hostname of legal shape would pass with nothing red anywhere. *)
let contains haystack needle =
  let n = String.length needle and l = String.length haystack in
  let rec go i = i + n <= l && (String.sub haystack i n = needle || go (i + 1)) in
  go 0

let case name ~derived ~override ~hostname =
  let outcome =
    try "ok:" ^ resolve ~derived ~override ~hostname
    with Failure m ->
      if contains m "hostname. That is the identifier" then "refused:hostname"
      else if contains m "does not have the machine-label shape" then
        "refused:shape"
      else if contains m "but not one for THIS machine" then "refused:base"
      else "refused:unrecognised-message"
  in
  print_string (name ^ " " ^ outcome ^ "\n")

let host h () = h

(* The hostname is read only to refuse an override. With no override there is
   nothing to compare it against, and reading it anyway would put a subprocess
   -- and the identifier itself -- on a path that has no use for either. *)
let must_not_be_read () = failwith "the hostname was read with no override set"

let () =
  case "no-override" ~derived:"linux-amd" ~override:None ~hostname:must_not_be_read ;
  case "empty-override" ~derived:"linux-amd" ~override:(Some "") ~hostname:must_not_be_read ;
  case "suffixed" ~derived:"linux-amd" ~override:(Some "linux-amd-b") ~hostname:(host "drangleic") ;
  case "suffixed-padded" ~derived:"linux-amd" ~override:(Some " linux-amd-b ") ~hostname:(host "drangleic") ;
  case "hostname" ~derived:"linux-amd" ~override:(Some "drangleic") ~hostname:(host "drangleic") ;
  case "hostname-cased" ~derived:"linux-amd" ~override:(Some " Drangleic ") ~hostname:(host "drangleic") ;
  case "hostname-shaped-like-a-label" ~derived:"linux-amd" ~override:(Some "linux-amd") ~hostname:(host "linux-amd") ;
  case "bare-word" ~derived:"linux-amd" ~override:(Some "workstation") ~hostname:(host "drangleic") ;
  case "suffix-too-long" ~derived:"linux-amd" ~override:(Some "linux-amd-123456789") ~hostname:(host "drangleic") ;
  case "suffix-illegal-char" ~derived:"linux-amd" ~override:(Some "linux-amd-B") ~hostname:(host "drangleic") ;
  case "unknown-vendor" ~derived:"linux-amd" ~override:(Some "linux-banana") ~hostname:(host "drangleic") ;
  case "glob" ~derived:"linux-amd" ~override:(Some "*") ~hostname:(host "drangleic") ;
  (* Well-formed, and refused anyway: the scrubber derives the label from the
     payload's hardware, so a different BASE comes back relabelled while the
     filename keeps the operator's -- the payload would disagree with its own
     name. Both polarities: a different vendor, a different os, and the same
     base with a suffix still accepted (see `suffixed` above). *)
  case "other-vendor-base" ~derived:"linux-amd" ~override:(Some "linux-intel-b") ~hostname:(host "drangleic") ;
  case "other-os-base" ~derived:"linux-amd" ~override:(Some "darwin-amd") ~hostname:(host "drangleic") ;
  (* The derived label is NOT shape-checked (an OS outside the enumeration still
     produces results), so a suffix on such a base must be refused by the SHAPE
     rule -- being a legitimate variant of the derived label does not make it
     committable. *)
  case "unshaped-derived-suffixed" ~derived:"freebsd-nvidia" ~override:(Some "freebsd-nvidia-b") ~hostname:(host "drangleic") ;
  case "unshaped-derived-exact" ~derived:"freebsd-nvidia" ~override:(Some "freebsd-nvidia") ~hostname:(host "drangleic")
OCAML_DRIVER
RESOLVE_OUT="$(ocaml "$DRIVERS/resolve_driver.ml" 2>&1)"
resolve_status=$?
if [ "$resolve_status" -ne 0 ]; then
  echo "::error::running the override policy in $LABEL_ML failed (exit $resolve_status):" >&2
  printf '%s\n' "$RESOLVE_OUT" >&2
  exit 2
fi
resolve_case() {
  local name="$1" want="$2"
  local got
  got="$(printf '%s\n' "$RESOLVE_OUT" | awk -v k="$name" '$1 == k {print $2}' | head -1)"
  if [ "$got" = "$want" ]; then
    echo "PASS override $name -> $want"
    pass=$((pass + 1))
  else
    echo "FAIL override $name -- wanted $want, got ${got:-<none>}"
    printf '%s\n' "$RESOLVE_OUT" | sed 's/^/       /'
    fail=$((fail + 1))
  fi
}
resolve_case no-override                  "ok:linux-amd"
resolve_case empty-override               "ok:linux-amd"
resolve_case suffixed                     "ok:linux-amd-b"
resolve_case suffixed-padded              "ok:linux-amd-b"
resolve_case hostname                     "refused:hostname"
resolve_case hostname-cased               "refused:hostname"
resolve_case hostname-shaped-like-a-label "refused:hostname"
resolve_case bare-word                    "refused:shape"
resolve_case suffix-too-long              "refused:shape"
resolve_case suffix-illegal-char          "refused:shape"
resolve_case unknown-vendor               "refused:shape"
resolve_case glob                         "refused:shape"
resolve_case other-vendor-base            "refused:base"
resolve_case other-os-base                "refused:base"
resolve_case unshaped-derived-suffixed    "refused:shape"
resolve_case unshaped-derived-exact       "refused:shape"
rm -rf "$DRIVERS"

echo ""
if [ "$fail" -ne 0 ]; then
  echo "check-no-machine-identifiers.test.sh: $fail case(s) FAILED, $pass passed"
  exit 1
fi
echo "check-no-machine-identifiers.test.sh: all $pass cases passed"
exit 0
