---
title: Properties and invariants
last-updated: 2026-07-27
status: live-doctrine
owner: human
schema-version: 2
---

# Properties

This file states the invariants this project holds itself to. It has two halves and
they are not equally trustworthy, so they are kept apart:

- the **`code-intel` block** below, which `scripts/check-kb-properties.sh` executes on
  every CI run, and
- the **prose sections**, which nothing executes.

Anything that can be moved from the second half to the first should be. A property in
prose is a property that will drift.

## The property this project learned the hard way

> **A gate that cannot fail is worse than no gate.**

Not a slogan — a measured recurrence. The 2026-07-27 skill-health report records **12
new instances of the `gate-vacuous` class in a single session**, and 9 of the 12 were
caught only by a manual habit that nothing required. The shapes seen in this repository,
each of which actually happened here:

| Shape | Instance |
| --- | --- |
| A checker exists, exits non-zero on `main`, and no workflow or `Makefile` target invokes it | `check-license-headers.sh` before backlog-137 — the fourth such find |
| A directory is outside dune's traversal, so its tests do not build, do not run, do not format-check, and nothing goes red | `sarek/sarek/test/`'s seven executables behind a stale `(dirs ir_extract)` (backlog-147) |
| An assertion accepts `Pexp_extension _` as a pass, so it asserts nothing | the 4 `kirc_ast_quoting` cases deleted in backlog-78 |
| A finder ends in `2>/dev/null`, so deleting a whole declared root makes the gate report success about a tree it never read | `add-license-headers.sh` before backlog-137 |
| A pattern silently drops part of its input, and the total it prints is believed | `[0-9]+ tests run` missing Alcotest's singular `1 test run.` — 15 suites, 11 cases (backlog-147) |
| A validation pipeline returns `Ok` on an invalid shader because it runs no validator at all | `codegen_golden` ran neither `glslang` nor `naga`; the identical hole had already been closed on PTX by the `ptxas` gate and nobody generalised it |
| The anti-vacuity gate is itself vacuous | `check-review-convergence.js` with no `round` key returned `violations: []`, exit 0 — five review rounds ran without it ever routing back |
| A test can only probe names it was told about, so a name added to one backend and to no list leaves it green | the arm-parity matrix, hence `check-arm-parity-coverage.sh` (backlog-94) |
| A sweep is skipped for an unrelated exclusion and reports zero | the fp64 predicate that was in no path of the OpenCL gate (`fp-contraction-policy.md` §10.10) |
| A calibration is pinned at one hand-computed input, and a broken oracle passes the pin | `test_opencl_f16_tripwire`'s oracle pinned at `x = 1.0` (§11.5) |
| A count-only sweep reports `1 / 63488` and nothing else, hiding *which* value was wrong | RADV returning `-0` for `0.0 - x` at `x = +0` (§13.6) |
| An empty declared set turns a strict check into a permissive one | any `exempt`/roots list that no longer names anything |
| An emitter names one concept twice — declaring it under one spelling and using it under another — and nothing compares its two halves | `Sarek_ir_glsl` declared `<v>_len` and used `sarek_<v>_length` for `EArrayLen`; unreachable from surface Sarek (no `len` primitive exists), so no test, example or user could observe it (backlog-156) |

The operative rule, and the reason `scripts/check-kb-properties.sh` exists at all:

> **Prove red before trusting green.** A checker is not evidence until it has been
> mutated and observed to fail *with the message it promises*. A positive control is not
> optional — without it, "went red" and "is always red" are the same observation.

Nine of those twelve were caught by a manual habit that nothing required, so as of
backlog-151 the rule is mechanical for the checkers that opt in:
`scripts/prove-red.sh` applies each declared mutation to a scratch copy and asserts the
exact exit code and the exact message. What it does *not* do is claim to cover every
checker — see "Reading the block" below.

## The machine-checked block

```code-intel
{"id": "KB-GATE-INVENTORY", "type": "gate-inventory-complete", "description": "Every scripts/* and ci/* a fresh clone executes from CI or the Makefile is declared below, or exempted with a stated reason. Adding a gate to CI without a row here fails. Both directories are scanned because gates do land outside scripts/ — the pocl runner probe and its covering test are under ci/ — and an inventory that knows one directory is complete about a set it chose rather than about what CI runs.", "check": {"carriers": [".github/workflows/ci.yml", "Makefile"], "exempt_manifest": "scripts/review-bundle.manifest.json", "exempt": ["scripts/coverage-unit.sh", "scripts/gpu-bench-check.sh", "ci/Dockerfile"]}}
{"id": "KB-GATE-SELF", "type": "gate-red-path", "description": "This checker itself. A gate that validates other gates and is not itself proven able to fail is the joke version of this file.", "check": {"tool": "scripts/check-kb-properties.sh", "red_path": "scripts/check-kb-properties.test.sh"}}
{"id": "KB-GATE-PROVE-RED", "type": "gate-red-path", "description": "Executes the mutation declared beside each subject checker and requires the exact exit code and message it promises, with a mandatory positive control first. This is the step KB-GATE-SELF cannot take: `gate-red-path` above enforces that a red-path test is DECLARED, and a declared red-path test that asserts nothing satisfies it perfectly. Its own spec block breaks a committed fixture gate in the four ways it could lie, so a version of it that credited an immune checker fails rather than printing a second green.", "check": {"tool": "scripts/prove-red.sh", "red_path": "scripts/prove-red.test.sh"}}
{"id": "KB-GATE-ALCOTEST", "type": "gate-red-path", "description": "An unregistered Alcotest case compiles and the suite reports green having not run it.", "check": {"tool": "scripts/check-alcotest-registration.js", "red_path": "scripts/check-alcotest-registration.test.js"}}
{"id": "KB-GATE-DUNE-VISIBILITY", "type": "gate-red-path", "description": "A (dirs ...) stanza can put a whole directory outside dune's traversal; to dune there is then nothing there to fail.", "check": {"tool": "scripts/check-dune-dir-visibility.sh", "red_path": "scripts/check-dune-dir-visibility.test.sh"}}
{"id": "KB-GATE-LICENSE", "type": "gate-red-path", "description": "The pre-backlog-137 finder ended in 2>/dev/null, so a deleted root produced 'All license headers are up-to-date!' about a tree it had not read.", "check": {"tool": "scripts/check-license-headers.sh", "red_path": "scripts/check-license-headers.test.sh"}}
{"id": "KB-GATE-BUNDLE-TRACKED", "type": "gate-red-path", "description": "An untracked review-tool bundle verifies perfectly on the workstation that has it; only a fresh-clone check notices.", "check": {"tool": "scripts/check-review-bundle-tracked.sh", "red_path": "scripts/check-review-bundle-tracked.test.sh"}}
{"id": "KB-GATE-ARM-PARITY", "type": "gate-red-path", "description": "Lexical companion to test_backend_arm_parity.ml: a name added to one backend's arm and to no list is invisible to the behavioural test. Its red path is a prove-red-spec block beside it rather than a .test.sh, because the three mutations that matter are edits to a scratch copy of the five backend sources and of the matrix — which is what prove-red.sh does — and not a harness. Two of its three internal refusals are now observed firing; the third (scanned != len(backends)) is unreachable while the loop body ends in `scanned += 1`, so mutating it would mutate a tautology.", "check": {"tool": "scripts/check-arm-parity-coverage.sh", "red_path": "scripts/prove-red.sh"}}
{"id": "KB-GATE-ALIAS-COVERAGE", "type": "gate-red-path", "description": "Guards that every test dune file is reachable from a runtest alias. Its red path is a prove-red-spec block beside it (backlog-151): an unwired executable is exit 1, and all THREE anti-vacuity refusals are exit 2, observed: a missing dune file, a file that parses to zero declared executables, and (added with DIRECTION 3) a scanned root with no invoker file at all, since a reachability question with nothing to search is unanswerable rather than answered yes. DIRECTION 3 itself checks that every declared rule alias is actually BUILT by some dune build-shaped command; an earlier version matched a bare '@name' substring anywhere in the invoker text, which an `echo 'we do not build @e2e-fixture'` satisfied, and matched by prefix so a future @e2e would have been satisfied by the existing @e2e-hip. Both are now refused, both observed. It remains true that alias coverage assumes dune can see the file it audits, which is what check-dune-dir-visibility.sh backstops.", "check": {"tool": "scripts/check-test-alias-coverage.sh", "red_path": "scripts/prove-red.sh"}}
{"id": "KB-GATE-NEGATIVE-CASE-COVERAGE", "type": "gate-red-path", "description": "Every negative-compile case declared in sarek/tests/negative/dune must be asserted by the Makefile test_negative recipe. The two halves live apart — dune declares a profile-gated library per case, the exact-stderr greps live in the Makefile — and nothing connected them, so a case added to dune with no assertion is never built and therefore never runs. That hole is invisible by construction: a negative case is EXPECTED to fail, so its absence looks exactly like its success. Both directions are checked (declared-not-asserted is a silent hole; asserted-not-declared makes the build fail for a reason unrelated to the property and the message grep report something misleading), and an empty set on either side is exit 2 rather than a vacuous pass. Deliberately NOT in scope: whether each asserted message is the right one for its case, which no lexical gate can settle — the residual coincidental-substring risk and test_superstep_diverged's documented non-blocking KNOWN-ISSUE fallback are both recorded in kb/sarek/tests/negative.md. Red path is a prove-red-spec block beside it (backlog-165); all three mutations observed firing.", "check": {"tool": "scripts/check-negative-case-coverage.sh", "red_path": "scripts/prove-red.sh"}}
{"id": "KB-GATE-WORKFLOW-STEPS", "type": "gate-red-path", "description": "Every workflow step has exactly one of run:/uses: and no repeated key, in BLOCK form and in single-line FLOW form, and a run that recognises zero steps is exit 2 rather than a pass. Both of those qualifications were added after an adversarial review: the first version could not see a flow mapping at all (`- {name: x, run: a, run: b}` matched nothing, so the step was neither checked nor counted) and then reported \"OK — 0 steps\" on a file of them. A multi-line flow mapping is REFUSED rather than skipped, since a step the gate cannot analyse must not read as well-formed. Child-key indentation is derived from where the first key sits, not assumed to be dash+2, which had false-positived valid `-   name:` alignment with a message that also said the step had no action when it did. A step was inserted into ci.yml immediately BEFORE another step's run: line, which left that step with no action and gave the new one two. YAML keeps the LAST duplicate silently, so the step NAMED 'Compile-gate the HIP backend' actually ran check-negative-case-coverage.sh and `dune build @sarek-hip/all` -- the entire deliverable of backlog-186, ~1.9k lines of backend and 6 test executables -- never executed once, while the CI log showed a green check under the correct name. The gate itself was never wrong (exit 0 locally); it was wired to nothing. Nothing here read the workflow STRUCTURALLY: ocamlformat does not, dune does not, and GitHub's own parser accepted the duplicate key rather than rejecting it. Dependency-free by necessity rather than taste -- PyYAML's safe_load SILENTLY drops duplicate keys, so parsing with it would have reported this very tree clean; the library cannot see the defect it would be used to find. Two exclusions, both forced by measurement (first run: 5 findings, all spurious, 0 real): nested mappings (with: {name: coverage} is an artifact name, not a second step name) and block scalars (deploy-pr-preview.yml embeds JavaScript under script: | whose object literals carry owner:/repo:/body:). Red path is a .test.sh rather than a prove-red-spec block because the subject locates the repo with git rev-parse and prove-red.sh's scratch has no .git, so no green baseline is possible there; its first red case is the ACTUAL misordered insertion rather than a synthetic edit, and a second case asserts the DEAD step is named too, since a fix that restored the HIP action but left the first step actionless would still be broken.", "check": {"tool": "scripts/check-workflow-steps.sh", "red_path": "scripts/check-workflow-steps.test.sh"}}
{"id": "KB-GATE-TVAR-ALLOCATOR", "type": "gate-red-path", "description": "The type-variable id space has exactly ONE allocator. Sarek_typer had one site building a tvar with fresh_var_id -- the TERM-variable counter in Sarek_typed_ast -- while every other tvar uses fresh_tvar_id from Sarek_types. Two separate Atomics both starting at 0, so the space was not injective. That stopped being cosmetic when float_literal_ids and numeric_required_ids landed: both are keyed on tvar ids and consulted INSIDE unify as set membership, so a leaked term id could be spuriously present in one of them and REJECT A LEGAL PROGRAM (reachable for any kernel with a recursive local function). Why a SOURCE gate and not a test: the defect makes unify reject during type INFERENCE, so the kernel never reaches lowering and backlog-184's lowered-IR uniqueness assertion is structurally blind to it -- the review that recommended one assertion for H2+H3+tailrec was wrong on the H2 third. It is also arithmetic-dependent (the two counters must have drifted into collision), so a runtime test would be a coin flip on counter state rather than a gate. Both polarities are checked: no Unbound-with-fresh_var_id anywhere (the defect), AND at least one Unbound-with-fresh_tvar_id still exists (without which the gate would go green on a tree where tvar construction had been renamed or moved out of its SOURCES list -- nothing to check is not the same as the invariant holding). A missing source is exit 2, not a pass. Two prose invariants that asserted this while it was false -- Sarek_lower_ir's 'There is ONE id allocator' and Sarek_types' 'IDs are monotonic within a kernel' -- were restated to name this gate as the enforcement, because a premise stated only in a comment is what failed here.", "check": {"tool": "scripts/check-tvar-id-allocator.sh", "red_path": "scripts/prove-red.sh"}}
{"id": "KB-GATE-README-REPO-LINKS", "type": "gate-red-path", "description": "README's CI badge and Actions links must name THIS repository. Line 5 pointed at mathiasbourgoin/SPOC for the whole life of the Sarek rework while every other link in the file pointed at Sarek, so the single most-read line in the repo reported a different project's build status. Nothing caught it because a badge is an IMAGE: it renders, so a wrong URL looks exactly like a right one. The expected owner/repo is DERIVED (GITHUB_REPOSITORY in Actions, else the git remote) and never written into the gate -- a hardcoded name would be a second copy of the fact being protected, going stale on any rename. Scoped to CI surfaces only: gh-pages/docs/publications.md links a dozen talk PDFs under mathiasbourgoin/SPOC that genuinely live there, and rewriting those would break them. Red path is a .test.sh rather than a prove-red-spec block because the subject derives its expectation from GITHUB_REPOSITORY or the git remote, neither of which exists in prove-red.sh's bare mktemp scratch, so the mandatory green baseline could not be established there; the harness supplies each source explicitly and checks that BOTH are honoured, including that GITHUB_REPOSITORY wins.", "check": {"tool": "scripts/check-readme-repo-links.sh", "red_path": "scripts/check-readme-repo-links.test.sh"}}
{"id": "KB-GATE-CITED-PATHS", "type": "gate-red-path", "description": "Every repo-relative path cited in a tracked .ml/.mli comment must resolve to a TRACKED file. Four comments cited design notes under roster/, a working directory never published, so no clone has ever had them; running the gate for the first time found six MORE of the same shape under briefs/, in files nobody had connected to the four. Resolution is against git ls-files, NOT the filesystem, and that distinction is the entire point: roster/ and briefs/ exist on the machine that wrote those comments and in no clone, so a presence check would have passed locally AND in CI while every reader hit a dead end. Three exclusions, each forced by measurement (26 raw candidates -> 10 real findings, 0 false positives): documentation placeholders (path/to/, ...), ancestor-relative refs (theories/PtxLayout.v cited from formal/codegen-ptx/test/), and PTX mnemonics that lex like paths (ld/st.shared matched st.sh until the extension was anchored to the token end). It scans ONLY comment text -- string literals and code are blanked, newlines preserved so line numbers stay real -- because otherwise a legitimate literal such as \"fixtures/missing.md\" fails CI over a path that was never a citation, and a gate blocking a correct change is worse than the drift it catches (CodeRabbit, PR #387). An earlier version had a pre-filter requiring the first path component to name a tracked directory; it was removed because it made the gate blind to a citation whose whole directory is missing, i.e. exactly the roster/ case it exists for. Red path is a .test.sh rather than a prove-red-spec block because the subject needs a git tree and prove-red.sh's scratch is a bare mktemp directory where it would exit 2, so no green baseline is possible; the harness builds a throwaway repo per case and its load-bearing pair is the SAME path asserted found in a comment and ignored in a string literal -- if the comment scanner had blanked everything, the first would be green and the gate inert.", "check": {"tool": "scripts/check-cited-paths-exist.sh", "red_path": "scripts/check-cited-paths-exist.test.sh"}}
{"id": "KB-GATE-FORMAL-ADMIT", "type": "gate-red-path", "description": "Fast lexical tripwire for `Admitted.` in the Rocq sources.", "check": {"tool": "scripts/check-formal-proofs.sh", "red_path": null, "reason": "Deliberately uncovered at this layer: a grep is not a proof and this script does not claim to be the guarantee. The machine-checked guarantee is the formal-proofs job, which rebuilds every .v from scratch, and the ledger/axiom-allowlist gate beside it does have a red-path test (KB-GATE-PROOF-LEDGER)."}}
{"id": "KB-GATE-OPAM-CLEAN", "type": "gate-red-path", "description": "Guards against a `make opam` regression that dirties the tree.", "check": {"tool": "scripts/check-opam-clean.sh", "red_path": null, "reason": "No red-path test yet. Lowest-value of the uncovered four: it compares a generated file against the tree, so it fails loudly and locally rather than silently, and it has no declared-coverage set that could empty out."}}
{"id": "KB-GATE-TOOLCHAIN-ASSERT", "type": "gate-red-path", "description": "Every codegen gate self-skips when its tool is absent (ptxas, glslangValidator, naga), so a SKIP is how the whole codegen-validation story silently disappears. This asserts the tools are present AND runnable, and carries an fp64 positive control so the skip cannot become CI's normal outcome.", "check": {"tool": "ci/assert-toolchain.sh", "red_path": null, "reason": "No red-path test yet, and it is the highest-value of the uncovered gates: its subject is exactly the shape this file is about. It does hard-fail rather than warn (nvdisasm section), and it carries its own positive control, but neither has been observed firing. Best candidate for the first application of the approved scripts/prove-red.sh."}}
{"id": "KB-GATE-POCL-PROBE", "type": "gate-red-path", "description": "Measures whether pocl can compile a kernel on a bare runner. Informational and exits 0 by design, which is the shape that rots unnoticed: nothing reads its exit code, so a probe that stopped measuring would keep printing a verdict indistinguishable from a real one.", "check": {"tool": "ci/pocl-runner-probe.sh", "red_path": "ci/pocl-runner-probe.test.sh"}}
{"id": "KB-GATE-PROOF-LEDGER", "type": "gate-red-path", "description": "Proof-ledger / axiom-allowlist gate for the Rocq development.", "check": {"tool": "scripts/check-proof-ledger.py", "red_path": "scripts/check-proof-ledger.test.sh", "invocation": "manual", "reason": "The gate runs inside the formal-proofs container job via its own driver; its red-path test runs in the fast job because it synthesises the ledgers and needs no Rocq. Putting the enforcement's red-path coverage behind a 50-second container pull is how a gate ends up untested."}}
{"id": "KB-GATE-SUITE-COUNTS", "type": "gate-red-path", "description": "The canonical test-suite counting rule. '0 FAIL of N' is exactly as trustworthy as N -- and only if the run that produced N actually completed: a build failure part-way through leaves a plausible partial total (measured: 45 suites, past the floor, off a `dune test` that exited 1), so a failed run is exit 5 and failing cases are exit 4, never a silent 0.", "check": {"tool": "scripts/test-suite-counts.sh", "red_path": "scripts/test-suite-counts.test.sh", "invocation": "manual", "reason": "The tool consumes a `dune test` log and is run by an operator or an agent reporting a total; CI has no reason to run it. What CI does run is the covering test, which is where the counting RULE is pinned."}}
{"id": "KB-GATE-WORKTREE-BOOTSTRAP", "type": "gate-red-path", "description": "Per-agent worktree bootstrap, including the scratchpad namespacing that a basename collision defeated.", "check": {"tool": "scripts/agent-worktree-bootstrap.sh", "red_path": "scripts/agent-worktree-bootstrap.test.sh", "invocation": "manual", "reason": "Invoked by an agent when it starts work, not by CI. Its test runs in CI because the bootstrap's failure mode — an agent silently sharing the main checkout — is invisible until work is lost."}}
{"id": "KB-GATE-IMPLEMENT-POSTHOOK", "type": "gate-red-path", "description": "roster-implement post-hook: writes the ledger row that records what a phase did.", "check": {"tool": "scripts/roster-implement-posthook.sh", "red_path": "scripts/roster-implement-posthook.test.sh", "invocation": "manual", "reason": "Phase-driven; it needs a task and briefs/ state, neither of which exists on a CI checkout. Its rules are covered by the test, which does run."}}
{"id": "KB-GATE-LEDGER-SCHEMA", "type": "gate-red-path", "description": "The canonical ledger schema. Its skip record exists to REJECT a skip that names no reason and no actor.", "check": {"tool": "scripts/lib/ledger-schema.js", "red_path": "scripts/ledger-schema.test.js", "invocation": "manual", "reason": "A library, not an entry point — it is loaded by the phase tools. Before backlog-135 it was exercised only incidentally by the post-hook's test, which cares about the hook rather than the rules; the covering test is what CI runs."}}
{"id": "KB-GATE-MANDATED-STEPS", "type": "gate-red-path", "description": "Checks that a phase recorded the steps its skill mandates.", "check": {"tool": "scripts/check-mandated-steps.js", "red_path": "scripts/check-mandated-steps.test.js", "invocation": "manual", "reason": "Phase-driven: it reads briefs/ state that a CI checkout does not have. The test runs in both CI and the Makefile harness target."}}
{"id": "KB-GATE-REVIEW-VERDICT", "type": "gate-red-path", "description": "Assembles the review verdict; refuses an unrecognized prior verdict status rather than reading it as a NO-GO continuation.", "check": {"tool": "scripts/review-verdict-assemble.js", "red_path": "scripts/review-verdict-assemble.test.js", "invocation": "manual", "reason": "Phase-driven. The behaviour that matters — never manufacturing attested-looking state from unverifiable input — is asserted by the test, which CI runs."}}
{"id": "KB-GATE-REVIEW-CONVERGENCE", "type": "gate-red-path", "description": "The anti-vacuity gate that was itself vacuous: no `round` key gave violations: [] and exit 0.", "check": {"tool": "scripts/check-review-convergence.js", "red_path": "scripts/check-review-convergence-hardening.test.js", "invocation": "manual", "reason": "A review-bundle member, phase-driven, and REVIEW-BUNDLE.md records that CI does not reach it. Its hardening test is a declared local patch and runs in CI, so the fail-closed behaviour is attested even though the tool is not exercised here."}}
{"id": "KB-GATE-REVIEW-FINDING-TESTS", "type": "gate-red-path", "description": "A RESOLVED CodeRabbit finding must arrive with a test, or a recorded escape (backlog-164). CodeRabbit found something real on six consecutive fully-green PRs in one day and whether a regression test came with the fix was a habit. SCOPE, because the claim is narrower than the name: it asserts the PR TOUCHED A TEST PATH -- touching a comment in a test file satisfies it -- so it is a floor against fix-with-no-test-at-all, NOT proof of coverage. The strong form is prove-red per finding and is not built. Usable at all only because resolvedBy is coderabbitai[bot], i.e. third-party verification the fix landed rather than the author's own say-so; a self-attested resolution would make it worthless. A human-resolved CodeRabbit finding is therefore out of scope to ENFORCE, but it is COUNTED AND NAMED rather than dropped: reporting it as \"0 resolved CodeRabbit finding(s)\" and \"no resolved CodeRabbit findings on this PR\" was false -- there was one -- and a scope decision stated as a count of zero reads as an all-clear, on a branch reachable by exactly the actor the gate constrains. The clean line now distinguishes no-CodeRabbit-threads-at-all from N-present-resolved-by-a-human, and the covering test asserts that OUTPUT, not just exit 0, since status 0 is correct in both branches and so cannot tell them apart. Its own covering test found that a crash also exits 1, so the refusal cases assert the REFUSED marker and the finding id, not just the status. PAGINATION was a live hole of exactly the kind the gate exists to close: reviewThreads(first:100) with no cursor loop meant a resolved finding on page 2 was invisible and the gate printed NOTHING TO VERIFY over it, so the walk now follows pageInfo.hasNextPage/endCursor and REFUSES (exit 2) if it cannot complete -- errored page, missing/malformed pageInfo, hasNextPage with no endCursor, a non-advancing cursor, or the page budget. Same axis, measured: gh pr view --json files returned 100 paths for PR #388 whose diff is 294 files, so the changed-file list comes from gh api .../pulls/N/files --paginate. The paging cases drive the live cursor loop with a stub gh on PATH, since a fixture is one document and cannot express a second page.", "check": {"tool": "scripts/check-review-finding-tests.js", "red_path": "scripts/check-review-finding-tests.test.js"}}
{"id": "KB-GATE-XRUNTIME", "type": "gate-red-path", "description": "Cross-runtime review dispatch and its fault attribution: a nonzero exit is always a runtime fault, and a malformed probe output must not arm the breaker as though the runtime had failed.", "check": {"tool": "scripts/xruntime-review.js", "red_path": "scripts/xruntime-caller-fault.test.js", "invocation": "manual", "reason": "A review-bundle member invoked from roster-review prose. REVIEW-BUNDLE.md records it as CI-reachable only via this test, which is the covering test for the #102 + F1 local patch."}}
{"id": "PROP-CAP-UNKNOWN-DOES-NOT-PERMIT", "type": "grep-present", "description": "Sarek_capability.permits is a three-valued match in which Unknown does not permit. A two-valued answer forces an unprobed device into a bucket, and `not unsupported` puts it in the permitted one every time. Every defect that motivated the capability model was something permitted by default. docs/design/capability-model.md §3.", "check": {"file": "spoc/ir/Sarek_capability.ml", "literal": "| Unknown _ -> false"}}
{"id": "PROP-HIP-FP-CONTRACT-OFF", "type": "grep-present", "description": "hiprtc contracts a*b+c by default, and additionally fuses an f32 multiply into an f32->f16 narrowing (v_fma_mixlo_f16). -ffp-contract=off is one of the two required defences and must stay in Hip_rtc.base_options. docs/fp-contraction-policy.md §2, §9.6.", "check": {"file": "sarek-hip/Hip_rtc.ml", "literal": "-ffp-contract=off"}}
{"id": "PROP-HIP-BARRIER-AMD-SCOPED", "type": "grep-present", "description": "The second required AMD defence: an opacity barrier on every narrowing's argument, emitted only under the AMD toolchain guard. The non-AMD arm is a bare identity. docs/fp-contraction-policy.md §11.4, §11.4a.", "check": {"file": "sarek/codegen/Sarek_ir_cuda.ml", "literal": "asm volatile(\"\" : \"+v\"(x));"}}
{"id": "PROP-METAL-FP-CONTRACT-PRAGMA", "type": "grep-present", "description": "Metal contracts a*b+c under every compile option measured, including mathMode=Safe (8773/8773 on Apple M4). The pragma is the only measured defence and must appear in every generated kernel. docs/fp-contraction-policy.md §2, §10.5.", "check": {"file": "sarek/codegen/Sarek_ir_metal.ml", "literal": "#pragma METAL fp contract(off)"}}
{"id": "PROP-GLSL-PRECISE-ON-FLOAT-LOCALS", "type": "grep-present", "description": "Every f32/f64 GLSL local carries `precise`, which glslang lowers to SPIR-V NoContraction. The invariant is 'never delete it', NOT 'it protects these shapes' — on RADV the ISA is opcode-identical with and without, and §6's stronger claim was retracted. It is also what makes S_f32_mul_then_absorb_add the f16 model the shipped codegen runs under. docs/fp-contraction-policy.md §2, §6, §13.4.", "check": {"file": "sarek/codegen/Sarek_ir_glsl.ml", "literal": "| TFloat32 | TFloat64 -> Buffer.add_string buf \"precise \""}}
{"id": "PROP-DF64-MUL-RN-VIA-FMA", "type": "grep-present", "description": "No compile option turns off FP_CONTRACT in OpenCL C, and -fmad=true is nvrtc's default. Sarek_df64 defends by construction instead: mul_rn routes every product through fma, leaving no fusable multiply. Measured effect on CUDA: df64 mul 5.92e-08 -> 9.07e-15. docs/fp-contraction-policy.md §2, §10.8.", "check": {"file": "sarek/Sarek_df64/Sarek_df64.ml", "literal": "mul_rn (a : float32) (b : float32) : float32 = fma a b 0.0"}}
{"id": "PROP-SUITE-COUNT-SINGULAR-TOLERANT", "type": "grep-present", "description": "Alcotest singularises: a suite running exactly one case prints '1 test run.'. A plural-only pattern dropped 15 suites and 11 cases, and two agents reported different totals for the same commit. `tests?` is the whole point of the pattern.", "check": {"file": "scripts/test-suite-counts.sh", "literal": "tests? run"}}
{"id": "PROP-GLOSSARY-CAPABILITY-KINDS", "type": "grep-present", "description": "glossary.md names the six capability kinds and states that Toolchain_semantic must be able to override a device saying yes. If the constructor is renamed, the glossary silently describes a vocabulary nobody uses; this is the anchor that makes that red instead.", "check": {"file": "spoc/ir/Sarek_capability.mli", "literal": "| Toolchain_semantic"}}
{"id": "PROP-GLOSSARY-F16-MODEL-NAMES", "type": "grep-present", "description": "S_drop_intermediate_narrowing is named in the model set precisely so it can be excluded: it sits at 1 ulp like every admitted member, and the only instrument keeping it out is that it is not on the admitted list. glossary.md documents that; this anchors the name to the source.", "check": {"file": "tools/f16_model_set/f16_model_set.ml", "literal": "S_drop_intermediate_narrowing"}}
{"id": "PROP-NO-SILENT-F64-NARROWING", "type": "grep-absent", "description": "SCOPE: this catches the HISTORICAL REGRESSION, not the class. It is a literal grep, so only this exact spelling is caught — `TFloat64 -> \"float\"` without the leading pipe, or with different spacing, or reached through a helper, walks past it. That is not hypothetical: a reviewer re-running the mutation independently first wrote `function TFloat64 -> \"float\" | _ -> \"\"` and the check stayed green, correctly. Do not read this as \"no backend may map a 64-bit float to a 32-bit device type\" — that claim is broader than the check, and a reader who takes it that way will over-trust it. What it does pin: until the capability model landed, Metal's arm was `| TFloat64 -> \"float\"` with a comment saying Metal has no double — a silent halving of precision with no refusal anywhere on the path. The fix was a refusal (Sarek_capability.float64_absent_metal, kind Backend_structural), not a widening, and this stops that one arm coming back. The general property is carried by the capability model and stated in prose below. docs/design/capability-model.md, docs/fp-contraction-policy.md §10.13.", "check": {"paths": ["sarek/codegen"], "suffixes": [".ml"], "literal": "| TFloat64 -> \"float\""}}
{"id": "KB-GATE-NO-MACHINE-IDENTIFIERS", "type": "gate-red-path", "description": "Refuses machine-identifying data in tracked files (backlog-168). Three surfaces, because the leak had three and closing one would not have caught the others: a benchmark payload carrying hostname/kernel/host-memory_gb, a tracked result FILENAME whose leading token is not a legal machine label (the 263 committed files leaked through their paths, which scrubbing payloads does not fix), and a producer that either shells out to hostname outside the single sanctioned call site in benchmarks/system_info.ml or re-emits a removed field into the JSON or the CSV header (the CSV leaked independently of the JSON). The path check is a LABEL-SHAPE allowlist, not a blocklist of the three known hostnames -- a blocklist would pass a fourth machine, which is how this class survives. The shape is <os>-<vendor>[-<suffix>] with os and vendor ENUMERATED and the suffix optional, 1-8 chars of [a-z0-9]: the enumerated prefix is what refuses a bare hostname (`drangleic` cannot acquire one, and the optional suffix cannot supply it), and the suffix exists because two boxes with the same OS and GPU vendor derive an identical label and merge in the dedup key -- before it was admitted (backlog-168b) the SAREK_BENCH_MACHINE override that exists to separate them produced filenames this gate refused, so it worked everywhere except for its own purpose. The shape is DEFINED ONCE, in scripts/machine-label-shape.sh, which this gate sources and which benchmarks/machine_label.ml restates for its validation of the operator override; the red path executes both implementations over one case table rather than trusting a comment that they agree. The path allowlist is anchored to the FILENAME, not to the path: composed with the shared `_<benchmark>_<size>_<timestamp>` tail, so a label-shaped DIRECTORY cannot vouch for a hostname-named file beside it. And a shape-legal override is not sufficient -- it must also be this machine's derived label, or that label plus a suffix, because scripts/deidentify-benchmark-results.py recomputes the label from the payload's own hardware and would relabel a different base back while the filenames kept the operator's version, leaving each payload disagreeing with its own name. Fails closed at exit 2 outside a git work tree, without the scrubber, or without that shape definition, since a check that cannot read the tracked set or cannot tell a label from a hostname is not a clean result. It does NOT remove hardware models: cpu.model and device names are the subject matter of a benchmark, so the line drawn here is identity and patch level, not hardware. It also does not reach git history or an already-deployed site; both were handled once, by hand, and neither is re-checked by anything.", "check": {"tool": "scripts/check-no-machine-identifiers.sh", "red_path": "scripts/check-no-machine-identifiers.test.sh"}}
```

### Reading the block

- `gate-red-path` enforces **declaration** completeness, not **coverage** completeness. A
  gate with no red-path test passes — but only by saying so, here, with a reason. That is
  a deliberately weaker contract than "every gate is proven able to fail"; the strong
  version fails today on three gates (`check-formal-proofs.sh`, `check-opam-clean.sh`,
  `ci/assert-toolchain.sh`), and a gate that is red on arrival gets disabled rather than
  fixed. The strong part is `KB-GATE-INVENTORY`: a gate cannot reach CI without a row
  saying which of the two it is.
- **A declared red path is not an executed one.** A `red_path` naming a test that asserts
  nothing satisfies the row above perfectly, which is why `scripts/prove-red.sh`
  (backlog-151) exists: it applies the mutation declared *beside* a subject checker and
  requires the exact exit code and the exact message that checker promises, after a
  mandatory positive control. Coverage is opt-in, one deliberate edit at a time, and
  pinned by `EXPECTED_SUBJECTS` so it cannot shrink in silence — the same
  declaration-over-coverage trade as the bullet above, and for the same reason. A
  `red_path` of `scripts/prove-red.sh` means the evidence is a spec block beside the gate
  rather than a `.test.sh` beside it.
- `invocation: "manual"` says CI runs the covering test but not the tool. It is a real
  weakening and it always carries a reason.
- Review-bundle members are exempt by **delegation**, not by opinion:
  `scripts/check-review-bundle-tracked.sh` computes and publishes their reachability and
  `scripts/REVIEW-BUNDLE.md` records which of them CI does not reach and why. The
  exemption names the manifest rather than re-listing paths, so it cannot rot separately
  from it.
- Prose-reading auditors skip this block (`schema/kb-schema.md`); it is executed, not
  read.
- **`grep-present` and `grep-absent` pin a spelling, not a property.** Both match a
  literal. A declaration catches the exact string it names and nothing else: different
  whitespace, a missing leading `|`, the same construct reached through a helper, all
  walk past it. So each is a **regression pin on the instance that actually occurred**,
  and the description says which instance. Read as general properties they will be
  over-trusted — which is not hypothetical, it is how a reviewer's first mutation of
  `PROP-NO-SILENT-F64-NARROWING` stayed correctly green. Where the general property
  matters it is stated in prose below and held by a type or a test, not by a grep. The
  right instinct on reading one of these is *"this exact regression cannot come back"*,
  never *"this class is impossible"*.

## Properties held in prose, and why they are not in the block yet

Each of these is a real invariant with a real defect behind it. None is in the block
because the check I could write for it today would be weaker than the test that already
holds it, and a weak duplicate gate is worse than none — it produces a second green.

- **The evidence tier must be recorded next to every claim, and a claim may not assert
  more than its tier supports.** `docs/fp-contraction-policy.md` §2 defines the ladder
  (`executed` / `machine-code` / `compiler-output` / `by-construction` / `unverified`;
  see `glossary.md`). The rule that bites: *"a capability claim whose evidence tier is
  weaker than `executed` should not also assert how the violation manifests"*
  (`capability-model.md` §5). Not machine-checkable without parsing prose.
- **Every zero needs a liveness control.** A sweep reporting 0 disagreements is
  indistinguishable from a sweep that did not run. `fp-contraction-policy.md` requires a
  positive control beside every null, and the f16 probes require a host calibration that
  must pass *before any device number is printed*. Held by the probes themselves.
- **A capability question is answered by kind, not by a boolean.** `Backend_structural`
  is decidable with no device; `Device_optional` needs one; `Toolchain_semantic` can only
  be measured and **must be able to override a device saying yes**; `Policy` is a verdict
  revised by a decision, not by a measurement; `Flag_legality` is accepted by runtimes
  that lack the bit, so acceptance is not evidence of support. `capability-model.md` §2.
- **A width mismatch with a correct in-language lowering is a codegen bug, not a missing
  capability.** `capability-model.md` §5.1. The admission test: *does a correct lowering
  exist in the target language?*
- **No backend may silently map a 64-bit float to a 32-bit device type.** The general
  form of `PROP-NO-SILENT-F64-NARROWING`, which pins only the one spelling that
  regressed. Where the target language genuinely has no `double`, the answer is a
  *refusal* — `Sarek_capability.float64_absent_metal`, kind `Backend_structural`,
  refused statically with no device needed — never a widening and never a quiet
  substitution. Held by the capability model and its tests
  (`spoc/ir/test/test_sarek_capability.ml`), not by the grep.
- **A width sweep must cover three edges, not two.** `source -> IR` and
  `IR -> device type` were each swept while the third edge — `IR <-> host scalar_kind
  -> exec-arg dispatch` — had no sweep, and that is the edge the bytes actually cross.
  `test_host_ir_width_agreement.ml` closes it by pinning
  `Vector.elem_size = Sarek_ir_layout.scalar_size` (the same denominator the backend
  sweep uses, so the three segments form one chain rather than three self-consistent
  fragments) and by requiring every host scalar kind to complete an exec-arg round trip
  or appear in a pinned refusal list. `width-addition-cost.md` §3.
- **Half of a new width is invisible to the compiler, and it is the expensive half.**
  Measured, not estimated: a usable scalar width touches **40** production files;
  adding the constructors to all four addable vocabularies forces **21** of them, over
  **seven** build rounds. The other **20** are the FFI layer, host storage, exec-arg
  dispatch, the conversion primitives and the interpreter narrowing — and every f16
  soundness bug lived there, none in a match arm. A blast-radius estimate built by
  grepping exhaustive matches is accurate about the 21 and says nothing about the 20.
  `width-addition-cost.md` §1, §2.
- **A pair-match with a catch-all is a dispatch table no constructor can force.**
  `Execute.exec_arg_of_vector`'s `get` matches on `Vector.kind` (total, so a new host
  kind forces an arm) while its `set`, forty lines away, matches on a
  `(typed_value, kind)` pair ending in a catch-all (so nothing does). f16 shipped with
  the first and not the second, and could not run on the Interpreter device at all.
  Only a sweep can see this. `width-addition-cost.md` §2.1.
- **A stdlib affordance one width enjoys is not inherited by its sibling.**
  `Bigarray.Float16` exists and gives f16 free 2-byte storage, free round-on-store and a
  float-typed `Vector.get`/`set`; the `Bigarray.kind` GADT is closed and has **no**
  bfloat16 (executed, OCaml 5.3.0). So bf16 is "f16 again with a different `cvt`" from
  the IR downward and a storage-layer redesign from the IR upward — the generalisation,
  one level up, of the Ctypes-mirror lesson in `f16-dsl-element-type.md` §3.3.
  `width-addition-cost.md` §5.1.
- **The capability table and a codegen-correctness sweep are complementary instruments,
  neither subsuming the other.** A defect found by one is not evidence about the other,
  and "we have a capability model now" is not a reason to stop sweeping.
  `capability-model.md` §5.
- **f16 relaxation is an allowlist, not a lifting.** A driver nobody has swept keeps
  today's refusal automatically, with no new decision required.
  `f16-relaxed-accuracy.md` §1.5. This is the same property as
  `PROP-CAP-UNKNOWN-DOES-NOT-PERMIT` seen from the numerics side.
- **Name the regime when quoting the f16 contract.** "Sarek's f16 contract" is ambiguous:
  Regime A (scalar f16) accepts on exact agreement with a named model set, Regime B
  (cooperative matrix) on a derived numeric bound. The two answers differ in kind, not
  degree. `f16-relaxed-accuracy.md` §1.6.
- **A model set whose members coincide on the swept inputs discriminates nothing.** Eight
  of the twenty f16 shapes are non-discriminating — every policy is the same function
  there — so a table of twenty zeros would read as twenty confirmations.
  `docs/measurements/f16-shapes-2026-07-27/README.md`.
- **Refer to internal tracker items as `backlog-NN`, never `#NN`.** GitHub auto-links a
  bare `#NN` to whatever holds that number on a shared counter now past 330.
  `CONTRIBUTING.md`. Mechanically checkable in principle; not written because the
  false-positive surface (legitimate `PR #275`) needs care and no defect has come from it
  since the convention landed.

## Escalation

`.claude/rules/escalation.md` binds agents to ask before any action **listed in this file
as requiring human approval**. Today that list is:

- Changing the `code-intel` block above in a way that **removes** a declaration, or that
  turns a `red_path` into `null`. Adding declarations needs no approval.
- Editing any review-bundle member (`scripts/review-bundle.manifest.json` `files[]`)
  outside the declared `local_patches[]` process.
- Removing `precise` from GLSL float locals, `-ffp-contract=off` from hiprtc, the Metal
  contraction pragma, or the AMD opacity barrier. Each is a measured defence, and each
  measurement is preserved under `docs/measurements/` or in
  `docs/fp-contraction-policy.md`.
