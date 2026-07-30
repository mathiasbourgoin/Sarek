#!/usr/bin/env bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
# Every repo-relative path cited in a source comment OR in a tracked markdown
# document must resolve to a TRACKED file, and every commit sha cited in one
# must be REACHABLE from a remote branch.
#
# WHY THIS EXISTS. Four source files cited design notes under `roster/` — a
# working directory that was never published, so no clone has ever had it:
#
#   sarek/ppx/Sarek_tag_erasure.ml:13
#     "See roster/ptx-limits-campaign/L14-static-tag-erasure.md."
#
# A reader following that pointer finds nothing, and cannot tell from the
# comment whether the file was renamed, deleted, or never shipped. The citations
# survived a doc sweep precisely because nothing reads comments. Running this
# gate for the first time found six MORE of the same shape, under `briefs/`,
# in files nobody had connected to the four.
#
# TRACKED, NOT PRESENT. Resolution is against `git ls-files`, never the
# filesystem. That distinction is the whole point: `roster/` and `briefs/` exist
# on the machine that wrote these comments and in no clone, so a filesystem
# check would pass here AND pass in CI while every reader hit a dead end.
#
# THREE THINGS IT DELIBERATELY DOES NOT FLAG, each measured against this tree
# (26 raw candidates -> 10 real findings, 0 false positives):
#
#  1. Documentation placeholders — `path/to/Sarek_gemm.ml`, `.../Sarek_df64.ml`.
#     Five usage examples write a path shape, not a path.
#  2. Ancestor-relative references — `theories/PtxLayout.v` cited from
#     formal/codegen-ptx/test/ means formal/codegen-ptx/theories/PtxLayout.v.
#     A citation resolves from the repo root or ANY ancestor of the citing file,
#     which is how a reader resolves it.
#  3. OCaml quoted-string literals -- {| ... |} and {id| ... |id}. 46 files here
#     use them to hold raw PTX/CUDA source, which contains both "(*" sequences
#     and real-looking paths. Unhandled, a {| would open a phantom comment and
#     swallow the code after it into the scan (CodeRabbit, PR #387).
#  4. PTX mnemonics that lex like paths — `ld/st.shared` is not a shell script.
#     The extension must end the token; without that guard `st.shared` matched
#     `st.sh` and three prose lines read as dangling citations.
#
# COMMENTS ONLY. The scan sees only OCaml comment text; string literals and
# code are blanked out first (newlines preserved, so reported line numbers stay
# real). Without that, a legitimate string such as
#
#     let fixture = "fixtures/missing.md" in ...
#
# reads as a dangling citation and fails CI over a path that was never a
# citation at all — the gate blocking a correct change. Raised by CodeRabbit on
# PR #387; the string-literal boundary is pinned by a prove-red case below.
#
# WRAPPED CITATIONS ARE UNWRAPPED FIRST. ocamlformat breaks a long comment
# mid-path, and one of the four roster/ citations that motivated this gate was
# written that way:
#
#     * flows through the queue. See roster/ptx-limits-campaign/L16-dynamic-
#     * parallelism.md for the CDP-vs-worklist rationale
#
# A line-oriented scan sees no `.md` on either line and reports nothing — the
# gate would have missed the very citation it exists to catch. Comment
# continuations are therefore joined before matching.
#
# An earlier version of this gate had a pre-filter requiring the first path
# component to name a tracked directory. It is gone: it made the gate blind to
# a citation whose whole directory is missing — the roster/ case, i.e. exactly
# what the gate is for. The check now stands on the three exclusions above.
#
# Exit codes: 0 = every citation resolves, 1 = a citation dangles,
# 2 = cannot run (not a git tree, no sources, no python3) — fail closed rather
# than scan nothing and report success.

set -uo pipefail

git rev-parse --show-toplevel >/dev/null 2>&1 || {
  echo "check-cited-paths-exist: not inside a git work tree" >&2
  exit 2
}
cd "$(git rev-parse --show-toplevel)" || exit 2

command -v python3 >/dev/null 2>&1 || {
  echo "check-cited-paths-exist: python3 not found" >&2
  exit 2
}

python3 - <<'PY'
import os, re, subprocess, sys, urllib.parse


def ls(*args):
    out = subprocess.run(["git", "ls-files", *args], capture_output=True, text=True)
    if out.returncode != 0:
        print("check-cited-paths-exist: git ls-files failed", file=sys.stderr)
        sys.exit(2)
    return [l for l in out.stdout.splitlines() if l]


sources = ls("*.ml", "*.mli")
if not sources:
    print("check-cited-paths-exist: no tracked .ml/.mli files found", file=sys.stderr)
    sys.exit(2)
tracked = set(ls())

# The prose trees. Every markdown file tracked at the repo root or under one of
# these, and nothing else. The four excluded trees are excluded for a stated
# reason, not by oversight:
#
#   benchmarks/  machine-written result write-ups, full of device paths and
#                driver hashes that are not citations of anything in this repo;
#   .github/     issue and PR templates, whose "paths" are instructions to the
#                person filling the form;
#   sarek*/ spoc/ per-directory READMEs — in scope for a later pass, left out
#                here only to keep this gate's first landing triageable.
#
# formal/ IS in scope: `formal/type-safety/STATUS.md` carried one of the two
# symptoms of the unreachable-sha defect below (the same false sentence as
# docs/plans/FORMAL-ROADMAP.md), so excluding it would have made the gate blind
# to half of the very instance that motivated it.
MD_ROOTS = ("docs/", "kb/", "specs/", "gh-pages/", "formal/")
md_sources = [
    f for f in ls("*.md") if "/" not in f or f.startswith(MD_ROOTS)
]
if not md_sources:
    print("check-cited-paths-exist: no tracked .md files found", file=sys.stderr)
    sys.exit(2)

# The trailing (?![A-Za-z0-9]) is load-bearing: without it "ld/st.shared" in a
# PTX comment matches "st.sh" and reads as a missing shell script.
CITATION = re.compile(
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\.(?:ml|mli|v|md|sh|json|ya?ml)(?![A-Za-z0-9])"
)
PLACEHOLDER = re.compile(r"(?:^|/)(?:path/to|\.\.\.)/")


def comments_only(src):
    """Blank everything that is not inside an OCaml (* ... *) comment.

    Newlines are preserved so line numbers in findings match the real file.
    OCaml comments nest, and a string literal may contain "(*" or "*)", so this
    is a small state machine rather than a regex: tracking string state is what
    keeps a comment-looking substring inside a literal from opening a comment,
    and tracking depth is what stops a nested close from ending the outer one.
    """
    out = []
    i, n = 0, len(src)
    depth = 0          # comment nesting depth; 0 = not in a comment
    in_string = False  # inside "..." (only tracked outside comments)
    quoted_close = None  # inside {id|...|id}: the exact closing delimiter
    # OCaml CHARACTER literals. `Buffer.add_char buf '"'` would otherwise flip
    # in_string on and never off, blanking every comment for the rest of the
    # file. Two tracked files already contain that exact token
    # (sarek/codegen/Sarek_wgsl_abi.ml:53, benchmarks/to_csv.ml), so the gate
    # was measurably blind over their tails while reporting full coverage.
    CHAR_LIT = re.compile(r"'(?:\\.|[^'\\])'")
    # OCaml quoted-string literals. 46 files here use them, and they hold raw
    # PTX/CUDA text that contains "(*" and paths -- so an unhandled {| would
    # open a phantom comment and swallow real code into the scan.
    QUOTED_OPEN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_']*)?\|")
    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if quoted_close is not None:
            # Opaque text until the matching |id}. Blank it; keep newlines.
            if src.startswith(quoted_close, i):
                out.append(" " * len(quoted_close))
                i += len(quoted_close)
                quoted_close = None
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if depth == 0 and in_string:
            # Blank the literal's contents; an escaped quote does not close it.
            if c == "\\" and i + 1 < n:
                out.append("\n" if src[i + 1] == "\n" else " ")
                out.append(" ")
                i += 2
                continue
            if c == '"':
                in_string = False
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if depth == 0 and c == "'":
            m = CHAR_LIT.match(src, i)
            if m:
                out.append(" " * (m.end() - i))
                i = m.end()
                continue
        if depth == 0 and c == '"':
            in_string = True
            out.append(" ")
            i += 1
            continue
        if depth == 0 and c == "{":
            m = QUOTED_OPEN.match(src, i)
            if m:
                quoted_close = "|" + (m.group(1) or "") + "}"
                out.append(" " * (m.end() - i))
                i = m.end()
                continue
        if two == "(*":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if two == "*)" and depth > 0:
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth > 0:
            # Inside a comment: keep the text. A string literal inside a comment
            # is still comment text, and citations legitimately appear in [".."].
            out.append(c)
        else:
            out.append("\n" if c == "\n" else " ")
        i += 1
    return "".join(out)


def _unwrap_paths(text):
    """Join a line broken mid-path, never mid-prose.

    A trailing "-" is only a continuation if the token it ends already contains
    a "/" -- that is what distinguishes `roster/ptx-limits-campaign/L16-` from
    the English word `device-`.
    """
    out, lines = [], text.split("\n")
    i = 0
    while i < len(lines):
        cur = lines[i]
        while cur.endswith("-") and i + 1 < len(lines):
            tok = cur.split()[-1] if cur.split() else ""
            if "/" not in tok:
                break
            nxt = re.sub(r"^\s*\*?\s*", "", lines[i + 1])
            cur = cur + nxt
            i += 1
        out.append(cur)
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Markdown. A SEPARATE extractor, not the OCaml state machine above: in a .md
# file there is no comment/code distinction to recover, the citation forms are
# markdown's own (inline links, reference definitions, backticked spans), and
# the one construct that must be blanked -- a fenced code block -- is delimited
# by lines, not by nesting.
# ---------------------------------------------------------------------------

FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def md_prose(src):
    """Blank fenced code blocks; keep everything else. Newlines preserved.

    Fences are blanked because that is where a markdown file keeps the things
    that LOOK like citations and are not: shell transcripts, `git log` output,
    generated ledger dumps, lockfile digests, sample tool output. A path or a
    hash inside a fence is a specimen being displayed, not a pointer a reader is
    invited to follow. The cost of this exclusion is real and accepted -- a
    genuine citation written only inside a fence is not checked -- and it is the
    reason inline backticked spans OUTSIDE fences are scanned instead of being
    lumped in with "code".
    """
    out, close = [], None
    for line in src.split("\n"):
        if close is None:
            m = FENCE.match(line)
            if m:
                close = m.group(1)[0] * 3
                out.append("")
                continue
            out.append(line)
        else:
            # Any fence of the same character at least as long as the opener
            # closes it. Tildes never close backticks and vice versa.
            m = FENCE.match(line)
            out.append("")
            if m and m.group(1)[0] * 3 == close:
                close = None
    return "\n".join(out)


# [text](target), with optional <> wrapping and an optional "title".
MD_LINK = re.compile(r"\[[^\]\n]*\]\(\s*<?([^)\s>]*)>?(?:\s+\"[^\"\n]*\")?\s*\)")
# A reference definition: [label]: target
MD_REFDEF = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*<?([^\s>]+)>?", re.MULTILINE)
# An inline code span. Backticked paths are the dominant citation form in this
# repo's docs ("see `sarek/codegen/Sarek_ir.ml`") and are invisible to the two
# link forms above.
MD_CODESPAN = re.compile(r"`([^`\n]+)`")


def md_link_target(raw):
    """Normalise a link target, or None if it does not address this repo.

    Excluded, each deliberately:
      - absolute URLs and mailto:/tel: -- another host's problem;
      - pure fragments (#section) -- same document;
      - site-absolute targets (/backends/) -- gh-pages Jekyll permalinks, which
        address the published SITE, not a path in the working tree;
      - template placeholders ({{ ... }}).
    An anchor or query on a real path is STRIPPED rather than excluded, so
    `docs/x.md#section` is checked as `docs/x.md`: the fragment is the reader's
    problem, the file is the gate's.
    """
    t = raw.strip()
    if not t or t.startswith(("#", "/", "{")):
        return None
    if "://" in t or t.startswith(("mailto:", "tel:")):
        return None
    t = t.split("#", 1)[0].split("?", 1)[0]
    t = urllib.parse.unquote(t)
    if not t or t.startswith(("#", "{")):
        return None
    return t


def resolves(citation, citing_file):
    """A citation resolves from the repo root or any ancestor of its file."""
    if citation in tracked:
        return True
    d = os.path.dirname(citing_file)
    while True:
        if os.path.normpath(os.path.join(d, citation)) in tracked:
            return True
        if not d:
            return False
        d = os.path.dirname(d)


# A tracked directory has no entry of its own in `git ls-files`, so a link to
# one is resolved by prefix.
tracked_dirs = set()
for _p in tracked:
    _d = os.path.dirname(_p)
    while _d:
        tracked_dirs.add(_d)
        _d = os.path.dirname(_d)


def resolves_any(citation, citing_file):
    """As `resolves`, but for a LINK target.

    Two extra forms, both of which a link may legitimately address and a
    backticked path may not:

      - a directory (`../benchmarks/`);
      - the RENDERED page of a markdown source. gh-pages is a Jekyll site, so
        `[backends](backends.html)` addresses the page built from
        `gh-pages/docs/backends.md`. Thirteen such links exist and every one is
        correct; demanding a tracked `.html` would fail them all and teach the
        next author that this gate is noise.
    """
    if resolves(citation, citing_file):
        return True
    if citation.endswith(".html") and resolves(
        citation[: -len(".html")] + ".md", citing_file
    ):
        return True
    cand = citation.rstrip("/")
    if not cand:
        return False
    if cand in tracked_dirs:
        return True
    d = os.path.dirname(citing_file)
    while True:
        if os.path.normpath(os.path.join(d, cand)) in tracked_dirs:
            return True
        if not d:
            return False
        d = os.path.dirname(d)


# Exempt paths: things that are path-shaped and legitimately not in this repo
# (upstream sources, generated trees, external docs). Every sibling gate has
# such a channel; without one this gate hard-fails a comment like
# "Mirrors OCaml stdlib typing/typecore.ml behaviour" with no remedy but to
# reword until no path-shaped token survives.
#
# The same channel carries sha exemptions. A row's first field is either a bare
# token (exempt everywhere) or `citing/file.md::token`, which exempts it in that
# ONE file. The scoped form exists because of the sha half: several documents
# now quote an unreachable sha in order to say it is unreachable, and a bare
# exemption for such a token would also wave through a NEW wrong citation of it
# somewhere else.
EXEMPT_FILE = "scripts/cited-paths-exempt.tsv"
exempt = set()
exempt_scoped = set()
try:
    for line in open(EXEMPT_FILE, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1].strip():
            print(
                f"check-cited-paths-exist: {EXEMPT_FILE}: every exemption needs a "
                f"TAB-separated reason -- {line!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        key = parts[0].strip()
        if "::" in key:
            f_, _, tok = key.partition("::")
            exempt_scoped.add((f_.strip(), tok.strip()))
        else:
            exempt.add(key)
except FileNotFoundError:
    pass


def is_exempt(token, citing_file):
    return token in exempt or (citing_file, token) in exempt_scoped


def read_source(f):
    try:
        return open(f, encoding="utf-8", errors="replace").read()
    except OSError as e:
        # NOT `continue`: skipping a source it could not read and then printing
        # "every path ... resolves" is a vacuous pass over a tree it did not
        # examine. Refuse instead.
        print(f"check-cited-paths-exist: cannot read {f}: {e}", file=sys.stderr)
        sys.exit(2)


def lineno_of(raw, token):
    """First line carrying the token, or None."""
    return next(
        (i for i, line in enumerate(raw.splitlines(), 1) if token in line), None
    )


dangling = []
for f in sources:
    try:
        raw = open(f, encoding="utf-8", errors="replace").read()
    except OSError as e:
        # NOT `continue`: skipping a source it could not read and then printing
        # "every path ... resolves" is a vacuous pass over a tree it did not
        # examine. Refuse instead.
        print(f"check-cited-paths-exist: cannot read {f}: {e}", file=sys.stderr)
        sys.exit(2)
    # Comments only, THEN join a comment line broken mid-token: a trailing "-"
    # followed by the next line's comment prefix is one path, not two. Line
    # numbers are reported from the ORIGINAL text, so a finding still points at
    # a real line.
    # Join a citation ocamlformat broke mid-path -- but ONLY when the trailing
    # token already looks like a path, i.e. contains a "/". Without that guard
    # the rule joins any hyphenated prose: "The lookup is device-\n * specific/
    # notes.md aware" fabricated a finding for `device-specific/notes.md`, a
    # path nobody wrote, and reported it at line "?" because it matches no
    # single line. ocamlformat wraps hyphenated prose constantly.
    text = _unwrap_paths(comments_only(raw))
    for c in sorted(set(CITATION.findall(text))):
        if "://" in c or "github.com/" in c:
            continue
        if PLACEHOLDER.search(c) or c.startswith("..."):
            continue
        if is_exempt(c, f):
            continue
        if resolves(c, f):
            continue
        # Report against the ORIGINAL text so the line number is real. A
        # wrapped citation matches no single line, so fall back to the line
        # carrying its first segment.
        lines = raw.splitlines()
        lineno = next(
            (i for i, line in enumerate(lines, 1) if c in line), None
        ) or next(
            (i for i, line in enumerate(lines, 1) if c.split("/")[0] + "/" in line),
            None,
        )
        dangling.append((f, lineno, c))

# --- markdown paths --------------------------------------------------------
# Same resolution rule (tracked, not present), different extraction. Three
# forms, in the order a reader meets them.
md_text = {}
for f in md_sources:
    raw = read_source(f)
    # md_prose blanks fenced lines to "" rather than dropping them, so an offset
    # into `prose` still has the ORIGINAL line number -- findings point at a real
    # line even when the same path is cited more than once in a file.
    prose = md_prose(raw)
    md_text[f] = (raw, prose)
    cited = []  # (kind, token, offset in prose)
    for rx in (MD_LINK, MD_REFDEF):
        for m in rx.finditer(prose):
            t = md_link_target(m.group(1))
            if t:
                cited.append(("link", t, m.start(1)))
    # A backticked span is prose, so it is filtered the same way an OCaml
    # comment is: only path-SHAPED tokens count, and the placeholder and
    # PTX-mnemonic exclusions still apply.
    for m in MD_CODESPAN.finditer(prose):
        span = m.group(1)
        for cm in CITATION.finditer(span):
            # An ABSOLUTE or HOME-relative path is not a repo-relative citation.
            # The regex cannot start a match at "/", so `/home/mathias/dev/SPOC/
            # sarek/...` pasted from a transcript arrives here as `home/mathias/
            # ...` and reads as a dangling citation of a directory named "home",
            # and `~/.claude/skills/formal-apparatus/SKILL.md` as one of a
            # directory named ".claude". Machine-local absolute paths are
            # scripts/check-no-machine-identifiers.sh's business; a path under
            # the agent harness in $HOME is nobody's, and neither is this gate's.
            #
            # `$SCRIPT_DIR/add-license-headers.sh` is the same shape for a
            # different reason: the leading token is a shell variable, so the
            # citation is a COMMAND, and the file it names lives wherever the
            # variable points.
            if cm.start() > 0 and span[cm.start() - 1] in "/$":
                continue
            cited.append(("code", cm.group(0), m.start(1) + cm.start()))
    for kind, c, off in sorted(set(cited)):
        if "://" in c or "github.com/" in c:
            continue
        if PLACEHOLDER.search(c) or c.startswith("..."):
            continue
        if is_exempt(c, f):
            continue
        # A link may address a directory; a backticked path must be a file --
        # `kb/` as a bare word in prose is not a citation anybody follows, and
        # accepting directories there would make every "under sarek/codegen/"
        # mention resolve for free.
        if resolves_any(c, f) if kind == "link" else resolves(c, f):
            continue
        dangling.append((f, prose.count("\n", 0, off) + 1, c))

for f, lineno, c in dangling:
    print(f"{f}:{lineno if lineno else '?'}: cites '{c}', which is not a tracked file")

# --- cited shas must be REACHABLE, not merely present ----------------------
# The defect is not a missing object. `d72a2e6a`, `fbfb3656`, `f6c14c2a` and
# `1da95861` were all cited as evidence in docs/plans/FORMAL-ROADMAP.md and
# formal/type-safety/STATUS.md, and all four EXIST in the workstation clone that
# wrote them -- they were real commits, orphaned by a rebase. Each is contained
# in ZERO remote branches (a reachable sibling is in 120+), so no fresh clone
# can resolve any of them: the citation reads as verifiable provenance and is
# not. An existence check alone would have passed all four. A fifth, 4b15a323,
# was produced BY the fix for the first three -- an agent cited a sha its own
# rebase had just orphaned -- which is why this is a gate and not a review note.
#
# Range 7..40 also excludes sha256 digests by construction: a 64-hex token fails
# the trailing boundary, so lockfile and artifact digests never enter.
SHA = re.compile(r"(?<![0-9A-Za-z])([0-9a-f]{7,40})(?![0-9A-Za-z])")


def sha_candidates(text):
    """Hex tokens that a reader would take for a commit sha."""
    for m in SHA.finditer(text):
        tok = m.group(1)
        # An all-digit run is a number (a line count, a byte size), not a sha.
        if tok.isdigit():
            continue
        i, j = m.start(1), m.end(1)
        # Part of a longer compound token: `deadbee.md`, `abc1234-rc1`, a hex
        # colour. The trailing boundary above only rejects letters and digits,
        # so the separators have to be rejected here.
        if i > 0 and text[i - 1] in "#":
            continue
        if i >= 2 and text[i - 2].isalnum() and text[i - 1] in "._-/":
            continue
        if j + 1 < len(text) and text[j] in "._-/" and text[j + 1].isalnum():
            continue
        yield tok, m.start(1)


def git_ok(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True
    )


sha_cites = []  # (file, lineno, token)
for f in md_sources:
    raw, prose = md_text[f]
    seen = set()
    for tok, off in sha_candidates(prose):
        if tok in seen or is_exempt(tok, f):
            continue
        seen.add(tok)
        sha_cites.append((f, prose.count("\n", 0, off) + 1, tok))

# FAIL CLOSED. `git branch -r --contains` on a shallow or single-branch clone
# reports nothing for EVERY sha, reachable or not, so the check would either
# fail the whole build or -- if someone "fixed" that by treating empty as
# fine -- pass everything while examining nothing. That second outcome is the
# vacuous-green failure this repository keeps closing (backlog-152: about twenty
# PRs merged green onto a main that was already red, because a pull_request
# checkout had no origin/main and the gate read it as "no differences"). So the
# preconditions are ASSERTED, and their absence is exit 2 -- never a pass, and
# never a silent skip. Asserted only when there is at least one sha to check, so
# that a repository which cites none (and the per-case fixtures in the red-path
# harness) is not forced to have a remote it does not need.
if sha_cites:
    shallow = git_ok("rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0 or shallow.stdout.strip() != "false":
        print(
            "check-cited-paths-exist: this is a SHALLOW clone, in which "
            "`git branch -r --contains` reports nothing for every sha -- the sha "
            "reachability check cannot run. Check out with fetch-depth: 0.",
            file=sys.stderr,
        )
        sys.exit(2)
    remotes = git_ok("branch", "-r", "--format=%(refname)")
    if remotes.returncode != 0 or not remotes.stdout.strip():
        print(
            "check-cited-paths-exist: no remote-tracking branches "
            "(refs/remotes/*) in this clone, so reachability is unmeasurable and "
            "every cited sha would look unreachable. Fetch the remote refs "
            "(actions/checkout with fetch-depth: 0) before running this gate.",
            file=sys.stderr,
        )
        sys.exit(2)

unreachable = []
for f, lineno, tok in sha_cites:
    if git_ok("cat-file", "-e", f"{tok}^{{commit}}").returncode != 0:
        unreachable.append((f, lineno, tok, "no such commit in this repository"))
        continue
    # REMOTE BRANCHES ONLY -- tags deliberately do not count. `actions/checkout`
    # passes `--no-tags` unless `fetch-tags: true` is set, so `git tag
    # --contains` is non-empty on the workstation and empty in CI: counting tags
    # would make the verdict depend on which machine ran the gate, which is the
    # same asymmetry as backlog-152 (same bytes, same job name, opposite
    # verdicts). It also mattered here: `d72a2e6a` is contained in the pushed
    # tag `archive/master` and in no branch, so a tag-counting version of this
    # check would have called one of the four motivating instances fine.
    if not git_ok("branch", "-r", "--contains", tok).stdout.strip():
        unreachable.append(
            (f, lineno, tok, "exists locally but is in ZERO remote branches")
        )

for f, lineno, tok, why in unreachable:
    print(f"{f}:{lineno if lineno else '?'}: cites commit '{tok}': {why}")

if dangling or unreachable:
    print()
    if dangling:
        print(f"{len(dangling)} citation(s) point at a path no reader can open.")
        print("Fix the path, reword so it is not a bare path, or add the path to")
        print(f"{EXEMPT_FILE} with a TAB and a reason if it legitimately")
        print("lives outside this repository.")
    if unreachable:
        print(f"{len(unreachable)} cited commit(s) cannot be resolved in a fresh clone.")
        print("Cite a sha that is an ancestor of a pushed branch (check with")
        print("`git branch -r --contains <sha>`), or, if the document quotes the")
        print(f"sha in order to say it is unreachable, add `<file>::<sha>` to")
        print(f"{EXEMPT_FILE} with a TAB and that reason.")
    sys.exit(1)

print(
    "check-cited-paths-exist: OK — every repo-relative path cited in "
    f"{len(sources)} tracked .ml/.mli files and {len(md_sources)} tracked .md "
    f"files resolves to a tracked file, and all {len(sha_cites)} cited commit "
    "sha(s) are reachable from a remote branch"
)
PY
