# Formal Verif Autopilot — Workflow Feedback Log

## Tick 0 — [T2-F02] Environment-threaded is_varying in Rocq spec for F-02

**Verdict**: GO | **Committed**: true | **Evolve**: CHANGE

**Friction points**:
- Tick 0 was a re-evaluation tick: T2-F02 was already committed (da7ae592) before the workflow ran, so Execute phase ran on a task that was already done. The workflow correctly detected this and promoted T1-LATEX as current task, but the PLAN.md had to be created from scratch since it was never committed — meaning tick 0 spent planning budget reconstructing state rather than advancing work.
- The pre-existing QCheck exit-code bug in test_convergence_extraction.ml is a latent false-positive in L4: the conformance link reports PASS even when properties fail. This is confirmed non-blocking for T2-F02 but means L4 has been giving a false PASS signal on extraction conformance throughout the project. The cost is undetected extraction regressions.

**Workflow improvements**:
- Add a standing DOCS-SYNC gate check at the start of Phase 2 (Execute): before picking a task, verify proof-ledger.json theorem count matches the actual theorem count in ConvergenceSpec.v, and flag drift as a hygiene subtask to address in the same tick if present. Tick 0 discovered both proof-ledger.json (11 vs 16 theorems) and ConvergenceSafetySpec.tex drift only implicitly; explicit pre-task drift detection would have surfaced T1-LEDGER and T1-LATEX as ready work immediately without requiring post-hoc re-evaluation.
- After a task is discovered DONE on re-evaluation (as T2-F02 was this tick), emit a short summary of what state was inferred and what was skipped so the Phase 5 feedback log accurately reflects what actually ran vs what was reconstructed from git history.

**Skill improvements**:
- The reviewer correctly identified that test_convergence_extraction.ml exits 0 on QCheck failure (binary exits with unit, not the test result), making dune runtest blind to extraction-conformance regressions. The apparatus skill's L4 conformance check policy should note this as a known class of false-passing tests: 'if the test binary returns unit rather than an exit code derived from QCheck_base_runner.run_tests, dune runtest will not detect failures; verify the exit code plumbing before trusting L4 PASS verdicts on new test files.'
- The reviewer noted that is_varying_in_env/check_env are not under the extraction CMBT link and that STATUS.md acknowledges this gap under Next candidates. The apparatus skill should encode a policy: whenever a new spec function is added that is not yet extracted, L2 must explicitly record the function name as a pending extraction gap in the apparatus check result, not just note it informally in STATUS.md.

---

## Tick 0 — [T2-WARP] WarpConvergence error class in ConvergenceSpec.v
**(2026-06-11)** | Gate: BLOCK | Committed: false | Evolve: CHANGE

**Baseline issues**: none

**Friction**:
- vo_sha256 integrity is entirely invisible to the apparatus pipeline check (P4 only checks count); the Fable reviewer is the only safety net, which means a fabricated/stale hash passes all script-level gates and reaches commit-stage before being caught.
- Execute agent has no explicit instruction on the correct computation sequence for vo_sha256 (coqc first, sha256sum the output .vo, then write ledger), so any shortcut — copying a prior value, computing before the final edit, or guessing — is silent and undetected until manual review.
- Section-numbering gaps are not checked by any automated step; the Execute and Gate phases both passed despite the 17→19→20 sequence, and the reviewer raised it as a minor finding only.
- Shallow theorem quantification (warp_varying_if_flags quantifies only the else-branch; proof is independent of the quantified variable) passes coqchk cleanly and is therefore invisible to all apparatus pipeline checks — it is caught only by an adversarial semantic reviewer, not by the gate machinery.

**Workflow improvements**:
- Phase 2 (Execute) prompt must explicitly state how to compute vo_sha256: after coqc completes, run `sha256sum theories/ConvergenceSpec.vo | awk '{print $1}'` and record that exact value in proof-ledger.json — never copy a prior value or fabricate one.
- Phase 3 (Gate) runApparatusCheck P4 check must be upgraded from a theorem-count check to also verify vo_sha256: run `sha256sum ${FORMAL}/theories/ConvergenceSpec.vo | awk '{print $1}'` and compare against the ledger's recorded value; fail if they differ.
- Phase 3 (Gate) independent Fable review prompt should explicitly request: (a) re-run `sha256sum theories/ConvergenceSpec.vo` and compare to ledger's vo_sha256, and (b) verify section numbers are consecutive with no gaps — these are currently unchecked by the apparatus check and fell through to the reviewer as a surprise.
- Phase 2 (Execute) should include a section-numbering consistency check instruction: after renaming or adding any section, run `grep -n '^## [0-9]' theories/ConvergenceSpec.v` and confirm the sequence is strictly consecutive with no gaps before returning.

**Skill improvements**:
- The formal-apparatus SKILL.md's /formal-check P4 rule (proof-ledger.json in sync) should be extended to include a vo_sha256 integrity check: 'run `sha256sum theories/<ProjectSpec>.vo` and compare against the ledger's vo_sha256 field; FAIL if they differ or if the ledger was written before the final coqc run'.
- Add a new apparatus memory `feedback_proof_ledger_hash_integrity.md`: the vo_sha256 field must be computed from the .vo produced by the FINAL coqc invocation on the shipped source; any other value (copied from a prior build, computed on a pre-edit .vo, or fabricated) is a stale-artifact error; the check is: sha256sum after coqc, then write proof-ledger.json, in that order — never the reverse.
- The apparatus check prompt in the skill should specify that P4 is a two-part check: (1) theorem count match, (2) sha256sum of the .vo matches the ledger's vo_sha256 field — both must pass for P4 to be PASS.

---
