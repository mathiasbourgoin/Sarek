#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# A workflow step must not invoke the OCaml toolchain on the HOST inside a job
# that never provisions one.
#
# WHY THIS EXISTS. backlog-186's HIP compile-gate failed twice, in two different
# ways, and no gate in this repository could see either one.
#
#   1. The step was inserted immediately BEFORE the previous step's `run:` line,
#      so that step lost its action and this one got two `run:` keys. YAML keeps
#      the last silently, so the step NAMED for the HIP gate actually ran
#      check-negative-case-coverage.sh. check-workflow-steps.sh now covers that.
#
#   2. The fix restored the intended command -- but as a HOST `run:`:
#
#          - name: Compile-gate the HIP backend and its tests
#            run: dune build @sarek-hip/all
#
#      copying the shape of the step directly above it, whose own comment says
#      it "needs no toolchain, so it runs on the host". This one does need the
#      toolchain. ci.yml's `build` job provisions none on the host: there is no
#      `ocaml/setup-ocaml` anywhere in it, every OCaml action goes through
#      `docker run ... spoc-ci:latest`. The step died at
#      `dune: command not found`, exit 127, and ~1.9k lines of backend plus 6
#      test executables were STILL outside every automated path -- the same net
#      effect as failure 1, reached by a different route.
#
# Neither check-workflow-steps.sh nor check-test-alias-coverage.sh can see this
# class. The first reads step STRUCTURE and is entirely happy with one well
# formed `run:` key; the second tracks `(alias ...)` rule aliases, and
# `sarek-hip` is a DIRECTORY alias that is not one of them -- measured: splitting
# that command across a line continuation leaves alias-coverage at exit 0. The
# only thing that recorded the dependency at all was prose, in
# scripts/unwired-targets.tsv, where the `alias::e2e-hip` exemption justifies
# itself with "until then ci.yml compile-gates the same targets via
# @sarek-hip/all". A gate whose justification rests on a step that no gate checks
# is the shape this repository keeps paying for.
#
# WHY IT IS NOT ENOUGH THAT EXIT 127 IS LOUD. It was loud here, and that is
# luck, not design: `dune` and `opam` are absent from the runner image, so they
# announce themselves. `make` is NOT -- it ships on ubuntu-latest. A host-side
# `make e2e-fast` in this job would run the real Makefile against a tree with no
# switch, and a recipe that short-circuits on a missing prerequisite exits 0.
# That is a vacuous pass, not a crash, and it is the same class.
#
# WHAT IT CHECKS, per job, in every .github/workflows/*.yml:
#   - does the job provision a host OCaml toolchain? (`uses: ocaml/setup-ocaml`)
#   - if NOT, no host-side `run:` in it may invoke a toolchain command at a
#     COMMAND POSITION. `./scripts/check-dune-dir-visibility.sh` is a script
#     whose name contains "dune"; it is not an invocation of dune, and a
#     substring test would false-positive on it.
#
# A `run:` body is treated as containerised for the extent of a `docker run`
# invocation -- its backslash-continuation chain and the quoted `bash -lc '...'`
# string it carries. Anything outside that extent is host-side.
#
# Exit codes: 0 = no host-side toolchain use in a job without a host toolchain,
# 1 = a step invokes the toolchain where none exists,
# 2 = cannot run (no workflows, no python3) -- fail closed.

set -uo pipefail

git rev-parse --show-toplevel >/dev/null 2>&1 || {
  echo "check-workflow-toolchain-env: not inside a git work tree" >&2
  exit 2
}
cd "$(git rev-parse --show-toplevel)" || exit 2

command -v python3 >/dev/null 2>&1 || {
  echo "check-workflow-toolchain-env: python3 not found" >&2
  exit 2
}

python3 - <<'PY'
import glob, re, sys

files = sorted(glob.glob(".github/workflows/*.yml") + glob.glob(".github/workflows/*.yaml"))
if not files:
    print("check-workflow-toolchain-env: no workflow files found", file=sys.stderr)
    sys.exit(2)

# Commands that need an OCaml switch on PATH. `make` is here because this repo's
# Makefile targets drive dune, so a host-side `make` in a container-only job is
# the silent variant of the same defect.
TOOLCHAIN = ("dune", "opam", "ocaml", "ocamlc", "ocamlopt", "ocamlfind",
             "ocamllex", "ocamlyacc", "odoc", "make")
# A command POSITION: start of line, or just after a shell separator. This is
# what distinguishes `dune build` from `./scripts/check-dune-dir-visibility.sh`.
CMD_AT = re.compile(
    r"(?:^|[\n;&|(`]|\$\(|&&|\|\|)\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    r"(" + "|".join(TOOLCHAIN) + r")(?![\w./-])"
)
HOST_TOOLCHAIN_ACTION = re.compile(r"uses:\s*ocaml/setup-ocaml")

problems = []
jobs_seen = 0
host_steps_seen = 0   # host `run:` steps actually EXAMINED
runs_seen = 0         # every `run:` step recognised, exempt job or not
exempt_jobs = 0


def run_bodies(lines, start, end):
    """Yield (lineno, body) for each `run:` step in lines[start:end]."""
    i = start
    while i < end:
        m = re.match(r"^(\s*)run:\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        ind, val = len(m.group(1)), m.group(2).strip()
        if val in ("|", ">", "|-", ">-", "|+", ">+"):
            body, j = [], i + 1
            while j < end:
                ln = lines[j]
                if ln.strip() and (len(ln) - len(ln.lstrip())) <= ind:
                    break
                body.append(ln)
                j += 1
            yield i + 1, "\n".join(body)
            i = j
        else:
            yield i + 1, val
            i += 1


def strip_container_regions(body):
    """Blank the extent of every `docker run` invocation; keep line count."""
    out = []
    in_docker = False
    in_quote = False          # inside the bash -lc '...' string
    for ln in body.split("\n"):
        if not in_docker and re.search(r"(?:^|[\s;&|])docker\s+run\b", ln):
            in_docker = True
        if in_docker:
            out.append("")
            # A single quote count that is odd flips in/out of the quoted script.
            if ln.count("'") % 2 == 1:
                in_quote = not in_quote
            if not in_quote and not ln.rstrip().endswith("\\"):
                in_docker = False
            continue
        out.append(ln)
    return "\n".join(out)


for f in files:
    lines = open(f, encoding="utf-8").read().splitlines()
    # Top-level `jobs:` then each 2-space-indented job key.
    job_starts = []
    in_jobs = False
    for i, ln in enumerate(lines):
        if re.match(r"^jobs:\s*$", ln):
            in_jobs = True
            continue
        if in_jobs:
            if ln.strip() and not ln.startswith(" "):
                in_jobs = False
                continue
            jm = re.match(r"^  ([A-Za-z_][\w-]*):\s*$", ln)
            if jm:
                job_starts.append((i, jm.group(1)))
    if not job_starts:
        print(f"check-workflow-toolchain-env: {f} declares no jobs this scanner "
              f"can see — refusing to report it clean", file=sys.stderr)
        sys.exit(2)
    bounds = [(s, n, (job_starts[k + 1][0] if k + 1 < len(job_starts) else len(lines)))
              for k, (s, n) in enumerate(job_starts)]
    for start, name, end in bounds:
        jobs_seen += 1
        job_text = "\n".join(lines[start:end])
        if HOST_TOOLCHAIN_ACTION.search(job_text):
            # A host switch exists, so host-side dune/opam is correct here --
            # docs.yml and deploy-pr-preview.yml genuinely do this. Count the
            # steps anyway: they are coverage this gate deliberately does not
            # provide, and the green line below names the number so the
            # exemption is auditable rather than invisible.
            exempt_jobs += 1
            runs_seen += sum(1 for _ in run_bodies(lines, start, end))
            continue
        for lineno, body in run_bodies(lines, start, end):
            runs_seen += 1
            host = strip_container_regions(body)
            # Comments in a shell body are not invocations.
            host = re.sub(r"#[^\n]*", "", host)
            if not host.strip():
                continue
            host_steps_seen += 1
            for m in CMD_AT.finditer(host):
                problems.append(
                    f"{f}:{lineno}: job {name!r} runs `{m.group(1)}` on the HOST, "
                    f"but the job provisions no host OCaml toolchain "
                    f"(no ocaml/setup-ocaml) — every toolchain action here must go "
                    f"through `docker run ... spoc-ci:latest`"
                )

for p in problems:
    print(p)

if problems:
    print()
    print(f"{len(problems)} step(s) invoke the toolchain in a job that has none.")
    print("`dune`/`opam` fail loudly at exit 127; `make` exists on the runner and")
    print("can exit 0 having built nothing, which is the same defect unobserved.")
    sys.exit(1)

# Same anti-vacuity rule as check-workflow-steps.sh: a run that inspected no
# host step has verified nothing, and must not print a green.
# The counter that matters here is `runs_seen`, not `host_steps_seen`. A tree in
# which every job provisions its own switch has NO step for this gate to examine,
# and reporting that clean is correct rather than vacuous -- the exemption is
# named in the line below and is auditable. What would be vacuous is recognising
# no job or no `run:` step at all, i.e. failing to parse the workflows.
if jobs_seen == 0 or runs_seen == 0:
    print(
        f"check-workflow-toolchain-env: examined {len(files)} file(s), "
        f"{jobs_seen} job(s) and recognised {runs_seen} `run:` step(s).",
        file=sys.stderr,
    )
    print(
        "  Exit 2, not 0: with no step in view this gate is guarding nothing, or "
        "the workflows use a form it cannot parse. Neither is a pass.",
        file=sys.stderr,
    )
    sys.exit(2)

print(
    f"check-workflow-toolchain-env: OK — examined {host_steps_seen} host `run:` "
    f"step(s) of {runs_seen} across {jobs_seen} job(s) in {len(files)} workflow "
    f"file(s); none invokes the toolchain in a job that does not provision one "
    f"({exempt_jobs} job(s) skipped as ocaml/setup-ocaml provisions a host switch)"
)
PY
