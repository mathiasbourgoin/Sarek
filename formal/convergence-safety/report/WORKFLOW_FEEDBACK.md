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

## Tick 0 — [T2-RETURN] TEReturn early-return barrier-skip
**(2026-06-12)** | Gate: GO | Committed: true | Evolve: CHANGE

**Baseline issues**: none

**Friction**:
- proof-ledger.json shipped without a `build_invocation` field despite the apparatus memory explicitly requiring it. The field was added to the memory (feedback_proof_ledger_artifact_integrity.md) during the T2-WARP tick, but the Execute agent for T2-RETURN was not instructed to write it, the P4 apparatus check did not enforce its presence, and the reviewer labeled its absence as Informational rather than a gate gap. Three layers all missed the same enforcement.
- The first-pass reviewer's `0fdf1107` hash mismatch (vs canonical `41e0d70a`) is a direct consequence of missing `build_invocation`: the reviewer used `-R . ConvergenceSpec` from inside the directory, producing a different `.vo` byte sequence. The tick reached COMMITTED with this unresolved, relying only on the second reviewer's Informational label to not block it. The assurance chain held by luck of reviewer calibration, not by gate enforcement.
- Issue 3 (circular conformance-test clause) is carried as 'now acceptable' because extraction-side tests independently exercise EReturn. This resolution is correct, but the workflow has no mechanism to record that a known limitation is tracked and the stated remedy is in place — the future reviewer will re-raise it unless it is documented in a findings note or STATUS.md. No friction with the workflow itself here, but a documentation gap for future ticks.

**Workflow improvements**:
- Phase 3 (Gate) runApparatusCheck P4 check must be upgraded to verify two things: (1) theorem count matches spec (already done), and (2) a `build_invocation` field exists in proof-ledger.json; if absent, P4 must FAIL — the T2-RETURN reviewer correctly identified the missing field, but P4 passed anyway because the script only checks count.
- Phase 2 (Execute) prompt must explicitly instruct the agent to record a `build_invocation` field in proof-ledger.json alongside `vo_sha256`, stating the exact compiler subcommand (`rocq compile -R theories ConvergenceSpec`), the path-qualifier flags, and the working directory (project root). The T2-RETURN proof-ledger.json shipped without this field, which is the direct cause of the first-pass reviewer's hash mismatch (they used a different mapping).
- Phase 3 (Gate) independent Fable review prompt should add an explicit checklist item: 'Read the `build_invocation` field from proof-ledger.json. If absent, REVISE — the field is mandatory and its absence is not informational. If present, confirm the hash in `vo_sha256` is reproducible using that exact invocation.' The T2-RETURN reviewer correctly surfaced this but labeled it Informational; the prompt did not guide them to treat absence as a gate-level gap.
- Phase 1 (Formal-Check) baseline prompt should include a `build_invocation` presence check under P4: grep proof-ledger.json for the `build_invocation` key and report FAIL with message 'ledger missing build_invocation' if absent. This would have surfaced the gap at baseline rather than at review time.

**Skill improvements**:
- The apparatus skill's `feedback_proof_ledger_artifact_integrity.md` already covers failure mode 4 (build-invocation mismatch) correctly and even documents a `build_invocation` field requirement. The gap is that the apparatus check prompt in the skill (P4 rule in /formal-check) does not yet include the presence check for `build_invocation`. Update the P4 rule in the skill's /formal-check policy to: 'P4 is a two-part check: (1) theorem count in ledger matches `grep -c "^Theorem" spec.v`; (2) `build_invocation` field exists in proof-ledger.json — FAIL if absent, because without it the vo_sha256 cannot be independently reproduced.' The memory is correct; the check procedure needs to be extended to match.
- The apparatus skill's reviewer guidance should be updated to state that a missing `build_invocation` field is a gate-level failure, not an informational note. The T2-RETURN reviewer found it but labeled it Informational; the skill's reviewer duty section should explicitly say: 'A ledger without build_invocation MUST be returned as REVISE, not Informational — the hash is unreproducible without it and the assurance chain is broken.' This corrects the miscalibrated verdict without requiring a script change.

---

## Tick 0 — [T3-S2] Uniformity soundness of is_varying_in_env (semantic grounding of EVary)
**(2026-06-13)** | Gate: BLOCK | Committed: false | Evolve: CHANGE

**Baseline issues**: none

**Friction**:
- BLOCK on documentation alone: Issues 1-3 (compile, coqchk, ledger) were all RESOLVED. The BLOCK was caused entirely by Issues 4 (LaTeX) and 5 (STATUS.md). The proof work is correct and complete; the tick is blocked from committing correct proofs because of documentation that the Execute agent was not instructed to update. This is a recurring pattern (T2-WARP had similar documentation drift friction).
- STATUS.md stale by design: STATUS.md was 'Last updated 2026-06-12 (T2-RETURN)' and said '20/20 theorems proven' and 'Next: T3-GATE' — while the proof-ledger.json already had 28 entries from this same tick. The two documents were written by different agents in different phases, and neither phase checked for consistency between them.
- LaTeX out-of-scope note contradicts delivered work: lines 111-114 of ConvergenceSafetySpec.tex state 'Semantic soundness of is_varying (needs texpr execution semantics) ... out of scope'. T3-S2 delivered exactly this via ConvergenceSemantics.v. The Execute agent never read the LaTeX to check whether its task falsified any prior out-of-scope claim. There is no instruction in the Execute prompt to do so.
- The baseline PASS signal is misleading: the baseline apparatus check returned PASS despite STATUS.md being wrong (20 vs 28 theorems) and LaTeX having a stale out-of-scope claim. These are P3 and P2 failures that the baseline agent either missed or rated non-critical. A PASS baseline that conceals two documentation gaps gives Execute a false sense that documentation is already clean — the agent then focuses only on the Rocq proof work.

**Workflow improvements**:
- Phase 3 (Gate): the script must gate on `gateCheck.latexInSync` and `gateCheck.statusInSync` in addition to `gateCheck.passed && gateCheck.coqchkPass`. Currently the gate advances to Fable review even when documentation is known-stale, pushing an avoidable BLOCK onto the adversarial reviewer. The fix: after the apparatus re-check, add `if (!gateCheck.latexInSync || !gateCheck.statusInSync) { trigger the link-fix agent with an explicit documentation-only mandate before the Fable review }`. This converts a reviewer-caught BLOCK into a recoverable Gate-phase fix.
- Phase 2 (Execute): the Execute prompt lists STATUS.md as needing an update 'if theorem count, test counts, or finding status changed', but does not mention ConvergenceSafetySpec.tex or the LaTeX out-of-scope notes. Add a mandatory checklist item: 'If any theorem was added or any prior out-of-scope claim was falsified, update both STATUS.md AND ConvergenceSafetySpec.tex — remove or revise the out-of-scope note that the delivered work resolves.' The T3-S2 Execute completed 7 new theorems without touching LaTeX, which means the LaTeX file still describes semantic soundness as out-of-scope at the moment T3-S2 delivered it.
- Phase 2 (Execute): the FILES block in the Execute prompt does not list `theories/ConvergenceSemantics.v` because it did not exist before T3-S2. When a task creates a new theory file rather than editing an existing one, the prompt's static file list is wrong. Add a dynamic note: 'If the task approach references a new .v file not in this list, include it explicitly — the apparatus chain must cover all compiled .v files, not just the files from prior ticks.'
- Phase 1 (Formal-Check) baseline check: the baseline reported PASS despite STATUS.md being dated 2026-06-12 and covering only 20 theorems (ledger already had 28 from the T3-S2 prior run). This means the baseline apparatus check's P3/P4 either did not cross-check STATUS.md theorem count against the ledger, or rated the discrepancy non-critical. P3 must compare STATUS.md 'Total: N theorems' against the actual ledger theorem count, not just against ConvergenceSpec.v. The STATUS.md and proof-ledger.json can diverge independently of the spec.

**Skill improvements**:
- The apparatus skill's `/formal-check` P2 rule (LaTeX spec in sync) checks that `superstep_outer_diverged_error` appears in the LaTeX — this is a T1-era spot-check. The rule should be generalized: 'For every theorem tier present in proof-ledger.json, at least one theorem of that tier must appear in the LaTeX catalog. If T3 theorems exist in the ledger with no T3 entry in the LaTeX, P2 = FAIL.' This would have caught the T3-S2 LaTeX gap automatically.
- Add a new apparatus memory `feedback_out_of_scope_notes_must_be_updated.md`: when a task delivers work that was previously classified as out-of-scope in a spec document (LaTeX, STATUS.md, or ASSUMPTIONS.md), the Execute agent must update the out-of-scope note in the same task — not defer it. A theorem proven and an out-of-scope claim that still describes it as future work are contradictory artifacts that undermine the spec document's trustworthiness. The apparatus check's P2 should flag any theorem tier present in the ledger that still appears only under an 'out-of-scope' label in the LaTeX.
- The apparatus skill's reviewer duty section should add: 'Check whether the task creates a new theory file (e.g. ConvergenceSemantics.v). If so, verify the file appears in the CMBT chain: (a) it is compiled by the Makefile, (b) coqchk covers it, (c) STATUS.md references it, and (d) the LaTeX references at least its main theorem. A new .v file not covered by all four checks is a gap in the apparatus chain, not just a documentation gap.'

---
