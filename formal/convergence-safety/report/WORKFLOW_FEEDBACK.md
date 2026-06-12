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
