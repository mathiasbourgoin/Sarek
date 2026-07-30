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
#   - does the job provision a host OCaml toolchain? (`uses: ocaml/setup-ocaml`,
#     or an in-job `apt-get install` of a toolchain package -- see below)
#   - if NOT, no host-side `run:` in it may invoke a toolchain command at a
#     COMMAND POSITION. `./scripts/check-dune-dir-visibility.sh` is a script
#     whose name contains "dune"; it is not an invocation of dune, and a
#     substring test would false-positive on it.
#
# APT PROVISIONING, and why it counts. `ocaml/setup-ocaml` is not the only way a
# job can put a compiler on the host PATH, and treating it as the only way made
# this gate collide with a correct step rather than catch a defect: ci.yml's
# build job installs `ocaml-nox` with apt so that
# scripts/check-no-machine-identifiers.test.sh can drive benchmarks/machine_label.ml
# through the toplevel -- the cross-language anti-drift check between a bash
# regex and `Machine_label.is_wellformed`. That job really does provision the
# toolchain; it just does not do it with the action this gate first knew about.
#
# The alternative -- moving the harness into spoc-ci:latest -- was rejected with
# measurement, and the measurement was RE-TAKEN here rather than trusted:
# ci/Dockerfile installs `opam` (line 35) and NO compiler package and NO python3,
# so `ocaml` exists in that image only after "Build CI image" (ci.yml step ~326)
# and "Build SPOC packages" (~361) have created the switch in .opam-ci. The
# harness runs at ~178, ahead of both. Containerising it would put a fail-closed
# gate behind the two slowest steps in the job and add an undeclared dependency
# on the oneAPI base image happening to carry a python3.
#
# So: a host `run:` step counts as PROVISIONING tool T when, in that one step,
#   1. an `apt-get install` (or `apt install`) at a command position names a
#      package that ships T -- per APT_PROVIDES below, which is a fail-closed
#      allow-list: a package it does not know provisions nothing; and
#   2. AFTER that install, at a command position, some tool of that package's
#      set is version-asserted (`-version` / `--version` / `-vnum`).
#
# Condition 2 is not decoration and it is what keeps this from being an
# exemption. `apt-get install` exiting 0 is not the same claim as T being
# runnable on PATH, and a gate that accepted the install alone would accept a
# job whose PATH is wrong -- reintroducing the class in a new place. ci.yml's
# own step says exactly this in a comment above its `ocaml -version`. One
# assertion per package set is enough: it proves the install materialised and
# that PATH resolves it. WHICH binaries a package ships is a distribution fact,
# not a claim the workflow gets to make, so the rest of the set follows.
#
# Scope of the provisioning is ORDERED and job-local: T is available to the
# remainder of the provisioning step (from the install onward) and to every
# later step of the SAME job. A use of T earlier in that step, or in an earlier
# step, is still a finding -- otherwise "the job installs it somewhere" would
# excuse a step that runs before the install, which is the exit-127 shape again.
# Nothing crosses a job boundary: each job provisions its own runner.
#
# This is deliberately NOT a line-number or step-name exemption. A line number
# rots on the next edit above it, and a step-name exemption would hide the next
# real instance in a step that happens to be named the same.
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

# Fail-closed allow-list: apt package -> the toolchain binaries it ships. A
# package absent from this table provisions NOTHING, so `apt-get install jq`
# cannot launder a host `dune`. Sets are per-package because one version
# assertion licenses the whole set (see the header): what a .deb contains is a
# distribution fact, not something the workflow asserts.
APT_PROVIDES = {
    "ocaml-nox": {"ocaml", "ocamlc", "ocamlopt", "ocamllex", "ocamlyacc"},
    "ocaml": {"ocaml", "ocamlc", "ocamlopt", "ocamllex", "ocamlyacc"},
    "ocaml-findlib": {"ocamlfind"},
    "opam": {"opam"},
    "dune": {"dune"},
    "odoc": {"odoc"},
    "make": {"make"},
    "build-essential": {"make"},
}
# `apt-get install` / `apt install` at a command position, with the flags and
# the package list that follow it on the same logical command.
APT_INSTALL = re.compile(
    r"(?:^|[\n;&|(`]|\$\(|&&|\|\|)\s*(?:sudo\s+)?apt(?:-get)?\s+"
    r"(?:-[^\s]+\s+)*install\b([^\n;&|]*)"
)
# A version assertion, at a command position, for a specific tool.
def version_assertion(tool):
    return re.compile(
        r"(?:^|[\n;&|(`]|\$\(|&&|\|\|)\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
        + re.escape(tool) + r"\s+--?(?:version|vnum)\b"
    )


def apt_provisions(host_body):
    """Tools this ONE host body provisions, as {tool: offset-it-becomes-live}.

    A package's tool set is admitted only once some tool of that set is
    version-asserted after the install -- the install alone is a claim, not a
    demonstration that the binary resolves on PATH.
    """
    live = {}
    for m in APT_INSTALL.finditer(host_body):
        tail = m.group(1)
        # Drop flags and `apt-get`'s own options; the rest are package names.
        # A continuation (`\` at end of line) keeps the package list going.
        args = tail
        end = m.end()
        while args.rstrip().endswith("\\"):
            nl = host_body.find("\n", end)
            if nl == -1:
                break
            nxt_end = host_body.find("\n", nl + 1)
            nxt = host_body[nl + 1:nxt_end if nxt_end != -1 else len(host_body)]
            args = args.rstrip().rstrip("\\") + " " + nxt
            end = nxt_end if nxt_end != -1 else len(host_body)
        pkgs = [a for a in args.replace("\\", " ").split() if not a.startswith("-")]
        for pkg in pkgs:
            provided = APT_PROVIDES.get(pkg)
            if not provided:
                continue
            # Some tool of the set must be version-asserted after this install.
            asserted_at = None
            for tool in sorted(provided):
                am = version_assertion(tool).search(host_body, m.end())
                if am and (asserted_at is None or am.start() < asserted_at):
                    asserted_at = am.start()
            if asserted_at is None:
                continue
            for tool in provided:
                if tool not in live or m.start() < live[tool]:
                    live[tool] = m.start()
    return live

problems = []
jobs_seen = 0
host_steps_seen = 0   # host `run:` steps actually EXAMINED
runs_seen = 0         # every `run:` step recognised, exempt job or not
exempt_jobs = 0
apt_provisioned_steps = 0  # host steps that themselves provision via apt


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
        # Tools an earlier step of THIS job put on the host PATH with apt.
        # Job-local and ordered: reset per job, grows as steps are walked.
        provisioned = set()
        for lineno, body in run_bodies(lines, start, end):
            runs_seen += 1
            host = strip_container_regions(body)
            # Comments in a shell body are not invocations.
            host = re.sub(r"#[^\n]*", "", host)
            if not host.strip():
                continue
            host_steps_seen += 1
            # Provisioning this step performs, with the offset at which each
            # tool becomes live inside it.
            here = apt_provisions(host)
            if here:
                apt_provisioned_steps += 1
            for m in CMD_AT.finditer(host):
                tool = m.group(1)
                if tool in provisioned:
                    continue
                if tool in here and m.start() > here[tool]:
                    # Installed by an `apt-get install` earlier in this very
                    # step, and version-asserted after it.
                    continue
                how = (
                    "no ocaml/setup-ocaml, and no earlier `apt-get install` of a "
                    "package that ships it followed by a version assertion"
                )
                if tool in here:
                    how = (
                        f"the `apt-get install` that provides `{tool}` comes LATER "
                        f"in this same step, so at this point it is not on PATH"
                    )
                problems.append(
                    f"{f}:{lineno}: job {name!r} runs `{tool}` on the HOST, "
                    f"but the job provisions no host OCaml toolchain "
                    f"({how}) — every toolchain action here must go "
                    f"through `docker run ... spoc-ci:latest`"
                )
            provisioned |= set(here)

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
    f"({exempt_jobs} job(s) skipped as ocaml/setup-ocaml provisions a host switch; "
    f"{apt_provisioned_steps} step(s) provision a host tool via apt-get + a "
    f"version assertion)"
)
PY
