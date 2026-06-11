// formal-verif-autopilot.js
//
// Autonomous formal-verification plan driver for convergence-safety.
// Inspired by the bounty-autopilot pattern from ~/dev/bounty-skills.
//
// GOAL: each tick executes one formal task, enforces the FULL apparatus
// pipeline (all 7 CMBT links + /formal-check equivalent), and uses the
// resulting friction to evolve the apparatus skill itself.
//
// TICK STRUCTURE:
//   Phase 0 Adapt        — read plan + prior feedback; pick current task; re-adapt if needed
//   Phase 1 Formal-Check — baseline run of full apparatus pipeline (before Execute)
//   Phase 2 Execute      — implement the current task (no commit)
//   Phase 3 Gate         — re-run full apparatus pipeline + coqchk + Fable review until GO
//   Phase 4 Evolve       — propose apparatus skill / methodology-backlog update; Fable review until GO
//   Phase 5 Feedback     — write friction log; self-adapt this script for next tick
//
// The workflow self-patches its own script via args.scriptPath at the end of each tick.

export const meta = {
  name: 'formal-verif-autopilot',
  description: 'Autonomous Rocq formal-verification with full apparatus pipeline + skill evolution',
  phases: [
    { title: 'Adapt',         detail: 'Read plan state, pick task, re-adapt workflow' },
    { title: 'Formal-Check',  detail: 'Baseline: run full apparatus pipeline (7 links)' },
    { title: 'Execute',       detail: 'Implement the current formal task (no commit)' },
    { title: 'Gate',          detail: 'Full apparatus re-check + coqchk + Fable review until GO' },
    { title: 'Evolve',        detail: 'Update apparatus skill + methodology backlog; Fable review until GO' },
    { title: 'Feedback',      detail: 'Write friction log; self-adapt script for next tick' },
  ],
}

// ── Paths ─────────────────────────────────────────────────────────────────────
const SPOC         = '/home/mathias/dev/SPOC'
const FORMAL       = `${SPOC}/formal/convergence-safety`
const SKILL_ROOT   = '/home/mathias/.claude/skills/formal-apparatus'
const PLAN_FILE    = `${FORMAL}/PLAN.md`
const FEEDBACK_LOG = `${FORMAL}/report/WORKFLOW_FEEDBACK.md`
const METHODOLOGY_BACKLOG = '/home/mathias/.claude/projects/-home-mathias-dev-bounty-skills/memory/methodology-upgrade-sources.md'

const tick       = (args && args.tick != null) ? args.tick : 0
const scriptPath = (args && args.scriptPath) || null

// ── Schemas ───────────────────────────────────────────────────────────────────

const PLAN_SCHEMA = {
  type: 'object',
  required: ['tick', 'openTasks', 'currentTask', 'blockers', 'workflowAdaptations'],
  properties: {
    tick: { type: 'number' },
    openTasks: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'title', 'tier', 'status'],
        properties: {
          id:        { type: 'string' },
          title:     { type: 'string' },
          tier:      { type: 'string' },
          status:    { type: 'string', enum: ['open', 'in-progress', 'blocked', 'done'] },
          blockedBy: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    currentTask: {
      type: 'object',
      required: ['id', 'title', 'approach', 'expectedDeliverables'],
      properties: {
        id:                   { type: 'string' },
        title:                { type: 'string' },
        approach:             { type: 'string' },
        expectedDeliverables: { type: 'array', items: { type: 'string' } },
      },
    },
    blockers:            { type: 'array', items: { type: 'string' } },
    workflowAdaptations: { type: 'array', items: { type: 'string' } },
  },
}

// Full apparatus pipeline check result (mirrors /formal-check output)
const APPARATUS_CHECK_SCHEMA = {
  type: 'object',
  required: ['passed', 'links', 'policyViolations', 'admitsFound', 'coqchkPass', 'summary'],
  properties: {
    passed:           { type: 'boolean' },
    links: {
      type: 'object',
      properties: {
        L1_spec:          { type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
        L2_extraction:    { type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
        L3_zero_admits:   { type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
        L4_conformance:   { type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
        L5_coverage_probe:{ type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
        L6_findings:      { type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
        L7_benchmarks:    { type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
      },
    },
    policyViolations: { type: 'array', items: { type: 'string' } },
    admitsFound:      { type: 'array', items: { type: 'string' } },
    coqchkPass:       { type: 'boolean' },
    latexInSync:      { type: 'boolean' },
    statusInSync:     { type: 'boolean' },
    summary:          { type: 'string' },
  },
}

const TASK_RESULT_SCHEMA = {
  type: 'object',
  required: ['taskId', 'status', 'changes', 'commitMessage'],
  properties: {
    taskId:        { type: 'string' },
    status:        { type: 'string', enum: ['complete', 'partial', 'blocked', 'skipped'] },
    changes:       { type: 'array', items: { type: 'string' } },
    commitMessage: { type: 'string' },
    nextTaskId:    { type: 'string' },
    notes:         { type: 'string' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['verdict', 'issues', 'requiredChanges'],
  properties: {
    verdict:         { type: 'string', enum: ['GO', 'REVISE', 'BLOCK'] },
    issues:          { type: 'array', items: { type: 'string' } },
    requiredChanges: { type: 'array', items: { type: 'string' } },
    positives:       { type: 'array', items: { type: 'string' } },
  },
}

const EVOLVE_SCHEMA = {
  type: 'object',
  required: ['verdict', 'rationale'],
  properties: {
    verdict:              { type: 'string', enum: ['CHANGE', 'NO_CHANGE'] },
    skillFile:            { type: 'string' },
    changeType:           { type: 'string', enum: ['new-memory', 'update-memory', 'update-policy', 'update-skill-doc', 'methodology-backlog'] },
    rationale:            { type: 'string' },
    proposedContent:      { type: 'string' },
    methodologyBacklogItem: { type: 'string' },
  },
}

const FEEDBACK_SCHEMA = {
  type: 'object',
  required: ['workflowImprovements', 'skillImprovements', 'frictionPoints', 'scriptChangesNeeded'],
  properties: {
    workflowImprovements: { type: 'array', items: { type: 'string' } },
    skillImprovements:    { type: 'array', items: { type: 'string' } },
    frictionPoints:       { type: 'array', items: { type: 'string' } },
    scriptChangesNeeded:  { type: 'boolean' },
    scriptChangeSummary:  { type: 'string' },
  },
}

// ── Reusable: full apparatus pipeline check ───────────────────────────────────
// Mirrors /formal-check. Called both before Execute (baseline) and after (gate).
const runApparatusCheck = async (label) => agent(
  `You are running a full apparatus pipeline check on the convergence-safety project.
  This mirrors /formal-check from the formal-apparatus skill.

  Project root: ${FORMAL}
  Skill root:   ${SKILL_ROOT}

  Run ALL of the following checks and report each as PASS / FAIL / SKIP:

  === CMBT COMPLETENESS CHAIN (7 links) ===

  L1 — Spec compiles:
    \`cd ${FORMAL} && coqc theories/ConvergenceSpec.v 2>&1; echo "coqc:$?"\`
    PASS iff exit 0 and no errors.

  L2 — Extraction exists and compiles:
    \`ls ${FORMAL}/extraction/ConvergenceSafetyExtraction.v 2>/dev/null && echo "exists"\`
    \`ls ${FORMAL}/extraction/ConvergenceModel.ml 2>/dev/null && echo "model-exists"\`
    \`cd ${SPOC} && dune build formal/convergence-safety/extraction/ 2>&1; echo "dune:$?"\`
    PASS iff both files exist and dune succeeds.

  L3 — Zero admits / axioms:
    \`grep -n "Admitted\\." ${FORMAL}/theories/ConvergenceSpec.v; echo "admitted-done"\`
    \`grep -n "admit\\b" ${FORMAL}/theories/ConvergenceSpec.v; echo "admit-done"\`
    \`cd ${FORMAL} && coqc -Q theories ConvergenceSpec theories/ConvergenceSpec.v && coqchk -Q theories ConvergenceSpec ConvergenceSpec.ConvergenceSpec 2>&1 | tail -3; echo "coqchk:$?"\`
    PASS iff 0 Admitted / admit occurrences AND coqchk exits 0 AND output contains "Modules were successfully checked".

  L4 — Conformance tests pass:
    \`cd ${SPOC} && dune runtest formal/convergence-safety/test/ 2>&1 | tail -20; echo "dune-test:$?"\`
    PASS iff exit 0.

  L5 — Coverage probe exists:
    \`ls ${FORMAL}/test_helpers/coverage_probe.ml 2>/dev/null && echo "exists"\`
    PASS iff file exists.

  L6 — Findings documented:
    \`ls ${FORMAL}/findings/DIVERGENCE_FINDINGS.md && ls ${FORMAL}/findings/UNCOVERED_EDGE_CASES.md 2>/dev/null && echo "exists"\`
    Check that F-01 and F-02 are mentioned: \`grep -c "F-01\\|F-02" ${FORMAL}/findings/DIVERGENCE_FINDINGS.md\`
    PASS iff both files exist.

  L7 — Benchmarks / dual-axis coverage table:
    \`cat ${FORMAL}/report/BENCHMARKS.md | head -20\`
    PASS iff BENCHMARKS.md exists and has a coverage table (even if populated with "n/a").

  === POLICY CHECKS ===

  P1 — No dangling policy cross-references:
    \`grep -r "policy/" ${FORMAL}/*.md ${FORMAL}/report/*.md 2>/dev/null | grep -v ".md:.*policy/" | head -5\`

  P2 — LaTeX spec in sync with theorems:
    Read ${FORMAL}/ConvergenceSafetySpec.tex theorem list and compare against
    \`grep "^Theorem\\|^Lemma" ${FORMAL}/theories/ConvergenceSpec.v\`
    Check that superstep_outer_diverged_error appears in the LaTeX.

  P3 — STATUS.md theorem count in sync:
    \`grep "Total.*theorems" ${FORMAL}/STATUS.md\`
    Compare against: \`grep -c "^Theorem" ${FORMAL}/theories/ConvergenceSpec.v\`

  P4 — proof-ledger.json in sync:
    \`cat ${FORMAL}/proof-ledger.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('theorems',[])),'theorems in ledger')" 2>/dev/null || echo "ledger-parse-error"\`
    PASS if theorem count matches spec.

  P5 — SCREEN-FIDELITY (test files call real Sarek code, not hand-reimplemented logic):
    \`grep -l "Sarek_convergence\\|check_expr" ${FORMAL}/test/*.ml 2>/dev/null\`
    PASS iff live test files import Sarek_convergence.

  === RETURN ===
  Set passed = true only if ALL 7 CMBT links are PASS and no P-violations are critical.
  Populate policyViolations with any P-check failures.
  latexInSync = true if P2 passes. statusInSync = true if P3+P4 pass.
  summary = one sentence.`,
  { phase: label, label: `apparatus-check-${label}`, schema: APPARATUS_CHECK_SCHEMA }
)

// ── Phase 0: Adapt ────────────────────────────────────────────────────────────
phase('Adapt')
log(`Tick ${tick} — adapting plan`)

const lastFeedback = tick > 0
  ? `Also read ${FEEDBACK_LOG} for what the previous tick recommended. Honor those adaptations.`
  : 'Tick 0 — no prior feedback.'

const planState = await agent(
  `You are the formal verification plan manager for convergence-safety.

  Tick: ${tick}
  SPOC root: ${SPOC}
  Formal project: ${FORMAL}

  READ ALL:
  1. \`cat ${FORMAL}/STATUS.md\`
  2. \`cd ${SPOC} && git log --oneline -8 formal/convergence-safety/ 2>&1\`
  3. \`cd ${SPOC} && git status --short 2>&1\`
  4. \`gh pr list --repo mathiasbourgoin/Sarek --state open --json number,title 2>/dev/null || echo "gh unavailable"\`
  5. \`cat ${PLAN_FILE} 2>/dev/null || echo "PLAN.md not found"\`
  ${lastFeedback}

  FULL ROADMAP (seed openTasks in this order; skip tasks already DONE per STATUS.md):

  T1 — Spec completeness (T1 tier; prerequisite: PR #182 merged):
  - T1A-CONF  : Extend QCheck conformance tests in test_convergence_conformance.ml to cover ESuperstep
                (blocked by PR #182 merge; add ≥3 new QCheck properties for ESuperstep false/true cases)
  - T1-LATEX  : Reconcile ConvergenceSafetySpec.tex — add superstep_outer_diverged_error theorem entry;
                remove stale TESuperstep elision note; verify all 12 theorems listed
  - T1-LEDGER : Sync proof-ledger.json — add superstep_outer_diverged_error; update theorem count to 12

  T2 — Spec soundness gaps (new Rocq definitions required):
  - T2-F02    : Env-threaded is_varying. Add VarEnv (nat list representing varying vars), is_varying_env,
                update check to thread VarEnv, add EVar→ELet propagation, add theorem
                let_alias_diverged_error (env-aware: if let x = EVary in EIf (EVar x) (EBarrier) then error),
                update all proofs. NOTE: ConvergenceSpec.v was already partially edited by a prior agent
                (EVar and ELet with nat var_id added) — read the current state before assuming anything.
  - T2-WARP   : WarpConvergence error class. Extend error inductive (WarpError), add warp-tagged check,
                add theorem warp_diverged_error.
  - T2-RETURN : TEReturn early-return barrier-skip. Add EReturn constructor, model barrier-skip,
                add safety theorem.

  T3 — Human gate then semantic soundness:
  - T3-GATE   : HUMAN DECISION — stop here and surface: "All T1+T2 done. Confirm T3-SEMANTIC in scope."
  - T3-SEMANTIC: Semantic soundness (months; separate sub-project)

  Hygiene (every tick):
  - DOCS-SYNC : STATUS.md / ASSUMPTIONS.md / proof-ledger.json drift — always check; always fix if drifted

  BLOCKER RULES:
  - T1A-CONF.blockedBy gets "HARD: PR #182 not merged" only if PR is still open
  - Top-level blockers[] = blockers for the SELECTED currentTask only
  - "HARD:" prefix = currentTask cannot execute this tick
  - T3-GATE always surfaces as a human gate (workflow stops)

  Pick currentTask = first task whose blockedBy has no "HARD:" entries.
  Write updated plan to ${PLAN_FILE}.`,
  { phase: 'Adapt', label: 'plan-adapt', schema: PLAN_SCHEMA }
)

log(`Task: [${planState.currentTask.id}] ${planState.currentTask.title}`)
if (planState.blockers.length) log(`Blockers: ${planState.blockers.join(' | ')}`)
if (planState.workflowAdaptations.length) log(`Adaptations: ${planState.workflowAdaptations.join('; ')}`)

const currentIsT3Gate = planState.currentTask.id === 'T3-GATE'
const hardBlocked = currentIsT3Gate || planState.blockers.some(b => b.startsWith('HARD:'))

if (hardBlocked) {
  const reason = currentIsT3Gate
    ? 'HUMAN-GATE: All T1+T2 tasks complete. Confirm T3-SEMANTIC (semantic soundness) is in scope before proceeding.'
    : planState.blockers.find(b => b.startsWith('HARD:'))
  log(`Hard blocker — returning early: ${reason}`)
  return { tick, verdict: 'HARD-BLOCKED', reason, planState }
}

// ── Phase 1: Formal-Check (baseline) ─────────────────────────────────────────
phase('Formal-Check')
log('Baseline apparatus check (before Execute)')

const baselineCheck = await runApparatusCheck('Formal-Check')
log(`Baseline: ${baselineCheck.passed ? 'ALL PASS' : 'ISSUES'} | coqchk:${baselineCheck.coqchkPass} | latex:${baselineCheck.latexInSync} | status:${baselineCheck.statusInSync}`)

const baselineFailedLinks = Object.entries(baselineCheck.links || {})
  .filter(([, v]) => v === 'FAIL').map(([k]) => k)
if (baselineFailedLinks.length) log(`Baseline failing links: ${baselineFailedLinks.join(', ')}`)

// ── Phase 2: Execute ──────────────────────────────────────────────────────────
phase('Execute')

const taskResult = await agent(
  `You are executing a formal verification task for the convergence-safety project.

  Task ID: ${planState.currentTask.id}
  Title:   ${planState.currentTask.title}
  Approach: ${planState.currentTask.approach}
  Expected deliverables: ${JSON.stringify(planState.currentTask.expectedDeliverables)}

  IMPORTANT — CURRENT SPEC STATE:
  The spec at ${FORMAL}/theories/ConvergenceSpec.v has been partially modified by a prior agent.
  READ IT CAREFULLY before making any edits. Specifically:
  - EVar : nat -> expr was added
  - ELet was changed to ELet : nat -> expr -> expr -> expr
  - The existing proofs may or may not compile — verify with coqc first.
  Run \`cd ${FORMAL} && coqc theories/ConvergenceSpec.v 2>&1 | head -5; echo "coqc:$?"\` first.

  APPARATUS PIPELINE (all 7 links must pass when you are done):
  L1 coqc clean | L2 extraction builds | L3 0 admits/axioms + coqchk |
  L4 conformance tests pass | L5 coverage_probe.ml exists | L6 findings/ present | L7 BENCHMARKS.md exists

  FILES:
  - Spec:         ${FORMAL}/theories/ConvergenceSpec.v
  - Extraction:   ${FORMAL}/extraction/ConvergenceSafetyExtraction.v
  - Conformance:  ${FORMAL}/test/test_convergence_conformance.ml
  - Live CMBT:    ${FORMAL}/test/test_convergence_live.ml
  - Extraction tests: ${FORMAL}/test/test_convergence_extraction.ml
  - LaTeX:        ${FORMAL}/ConvergenceSafetySpec.tex
  - Proof ledger: ${FORMAL}/proof-ledger.json
  - STATUS.md:    ${FORMAL}/STATUS.md

  EXECUTION PROTOCOL:
  1. Read relevant source files first (do not assume prior state)
  2. Make changes according to the task approach
  3. After each Rocq edit, run coqc to verify it compiles
  4. After all Rocq changes: run coqchk to confirm 0 axioms
  5. After any test file changes: run \`cd ${SPOC} && dune runtest formal/convergence-safety/ 2>&1 | tail -20; echo "dune:$?"\`
  6. Update STATUS.md if theorem count, test counts, or finding status changed
  7. DO NOT COMMIT in this phase

  Return the list of files changed and a concise commit message (type(scope): description).`,
  { phase: 'Execute', label: `exec-${planState.currentTask.id}`, schema: TASK_RESULT_SCHEMA }
)

log(`Execute: ${taskResult.status} | ${taskResult.changes.length} file(s) changed`)

// ── Phase 3: Gate ─────────────────────────────────────────────────────────────
phase('Gate')

let gateVerdict = 'SKIP'
let reviewResult = null
let gateCheck = null

if (taskResult.status === 'complete' || taskResult.status === 'partial') {

  // Step 3a: Full apparatus re-check (all 7 links)
  gateCheck = await runApparatusCheck('Gate')
  const failedLinks = Object.entries(gateCheck.links || {})
    .filter(([, v]) => v === 'FAIL').map(([k]) => k)

  log(`Gate check: ${gateCheck.passed ? 'ALL PASS' : 'FAIL [' + failedLinks.join(',') + ']'} | coqchk:${gateCheck.coqchkPass}`)

  if (!gateCheck.passed || !gateCheck.coqchkPass) {
    // One fix attempt for failed links before giving up on this tick
    const linkFixResult = await agent(
      `The apparatus gate check failed. Fix ALL failing items.

      Failed CMBT links: ${JSON.stringify(failedLinks)}
      Policy violations: ${JSON.stringify(gateCheck.policyViolations)}
      Admits found: ${JSON.stringify(gateCheck.admitsFound)}
      coqchk pass: ${gateCheck.coqchkPass}
      Summary: ${gateCheck.summary}

      Project: ${FORMAL}
      Read the relevant files, fix every failing link, then re-run the checks.
      Verify: \`cd ${FORMAL} && coqc theories/ConvergenceSpec.v 2>&1; echo "coqc:$?"\`
      \`cd ${SPOC} && dune runtest formal/convergence-safety/test/ 2>&1 | tail -10; echo "dune:$?"\`
      Do NOT commit.`,
      { phase: 'Gate', label: 'link-fix', schema: TASK_RESULT_SCHEMA }
    )

    // Re-check after fix attempt
    gateCheck = await runApparatusCheck('Gate')
    const remainingFails = Object.entries(gateCheck.links || {})
      .filter(([, v]) => v === 'FAIL').map(([k]) => k)
    log(`Gate re-check: ${gateCheck.passed ? 'ALL PASS' : 'STILL FAILING [' + remainingFails.join(',') + ']'}`)

    if (!gateCheck.passed || !gateCheck.coqchkPass) {
      gateVerdict = 'FAIL-PIPELINE'
    }
  }

  if (gateVerdict !== 'FAIL-PIPELINE') {
    // Step 3b: Independent Fable review — fresh context
    reviewResult = await agent(
      `You are an independent Rocq/Coq formal verification reviewer. Cold review — you did NOT write this.

      Project: ${FORMAL}
      Apparatus skill: ${SKILL_ROOT}
      Task executed: ${JSON.stringify(planState.currentTask)}
      Claimed changes: ${JSON.stringify(taskResult.changes)}
      Apparatus check result: ${JSON.stringify(gateCheck.links)}

      REVIEW PROTOCOL:
      1. \`grep -n "^Theorem\\|^Lemma" ${FORMAL}/theories/ConvergenceSpec.v\` — list all theorems
      2. \`grep -n "Admitted\\.\\|admit\\b" ${FORMAL}/theories/ConvergenceSpec.v\` — check admits
      3. Read the changed theorem proofs — verify every inductive case is handled
         (EVar, ELet with 3 args, ESuperstep must appear in EVERY expr_list_rect proof)
      4. Check that any new QCheck property actually tests the intended semantic
      5. Check that STATUS.md theorem count matches actual theorem count
      6. Check that LaTeX theorem list mentions any new theorem added this tick
      7. Check that proof-ledger.json has an entry for each theorem

      Be adversarial. REVISE with exact file+line if wrong. GO only if genuinely correct.
      BLOCK if a fundamental issue (unsound proof, wrong spec semantics) cannot be fixed this tick.`,
      { phase: 'Gate', label: 'fable-review-1', model: 'fable', schema: REVIEW_SCHEMA }
    )

    log(`Fable review 1: ${reviewResult.verdict} — ${reviewResult.issues.length} issue(s)`)

    if (reviewResult.verdict === 'GO') {
      gateVerdict = 'GO'

    } else if (reviewResult.verdict === 'REVISE') {
      const fixResult = await agent(
        `Apply ALL reviewer-required fixes.

        Issues: ${JSON.stringify(reviewResult.issues)}
        Required changes: ${JSON.stringify(reviewResult.requiredChanges)}
        Project: ${FORMAL}

        Fix everything, then verify:
        \`cd ${FORMAL} && coqc theories/ConvergenceSpec.v 2>&1; echo "coqc:$?"\`
        \`cd ${SPOC} && dune runtest formal/convergence-safety/test/ 2>&1 | tail -5; echo "dune:$?"\``,
        { phase: 'Gate', label: 'fix-revise', schema: TASK_RESULT_SCHEMA }
      )

      // Second Fable review — no more REVISE after this
      const review2 = await agent(
        `Independent Rocq reviewer — second pass. Cold review.
        Prior issues: ${JSON.stringify(reviewResult.issues)}
        Claimed fixes: ${JSON.stringify(fixResult.changes)}
        Read ${FORMAL}/theories/ConvergenceSpec.v and verify ALL prior issues are resolved.
        GO if resolved. BLOCK if any remain (no further REVISE rounds).`,
        { phase: 'Gate', label: 'fable-review-2', model: 'fable', schema: REVIEW_SCHEMA }
      )
      log(`Fable review 2: ${review2.verdict}`)
      reviewResult = review2
      gateVerdict = review2.verdict === 'GO' ? 'GO' : 'BLOCK'

    } else {
      gateVerdict = 'BLOCK'
    }
  }

} else if (taskResult.status === 'blocked') {
  gateVerdict = 'SKIP-BLOCKED'
} else {
  gateVerdict = 'SKIP-' + taskResult.status.toUpperCase()
}

log(`Gate: ${gateVerdict}`)

// Commit only on GO
let committed = false
let commitSha = null

if (gateVerdict === 'GO') {
  const commitResult = await agent(
    `Commit and push the gate-cleared formal verification changes.

    SPOC root: ${SPOC}
    Branch: formal/convergence-safety-phase1a (confirm with git branch --show-current)

    1. Stage ONLY convergence-safety files (never .opam-ci/, never test_dft.ml):
       \`cd ${SPOC} && git status --short | grep "formal/convergence-safety" | grep -v "test_dft"\`
       Stage each modified file individually.
    2. Commit with exactly this message (include Co-Authored-By):
    \`\`\`
    cd ${SPOC} && git commit -m "$(cat <<'CMSG'
${taskResult.commitMessage}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
CMSG
)"
    \`\`\`
    3. Push: \`cd ${SPOC} && git push 2>&1 | tail -3\`
    4. Return { committed: true/false, sha: "short-sha", pushOk: true/false }`,
    { phase: 'Gate', label: 'commit', schema: {
      type: 'object', required: ['committed'],
      properties: { committed: { type: 'boolean' }, sha: { type: 'string' }, pushOk: { type: 'boolean' } }
    }}
  )
  committed = commitResult.committed
  commitSha = commitResult.sha
  log(`Commit: ${committed ? 'OK sha=' + (commitSha || '?') : 'FAILED'}`)
}

// ── Phase 4: Evolve ───────────────────────────────────────────────────────────
phase('Evolve')

const frictionData = {
  tick,
  taskId:           planState.currentTask.id,
  taskStatus:       taskResult.status,
  gateVerdict,
  baselineLinks:    baselineCheck.links,
  gateLinks:        gateCheck ? gateCheck.links : {},
  policyViolations: gateCheck ? gateCheck.policyViolations : [],
  reviewIssues:     reviewResult ? reviewResult.issues : [],
  reviewPositives:  reviewResult ? (reviewResult.positives || []) : [],
  latexInSync:      gateCheck ? gateCheck.latexInSync : null,
  statusInSync:     gateCheck ? gateCheck.statusInSync : null,
  committed,
}

const evolveProposal = await agent(
  `You are a formal-apparatus skill improvement agent for this experiment.

  EXPERIMENT GOAL: each tick's friction improves the apparatus skill itself.
  The workflow and the apparatus skill co-evolve via this Evolve phase.

  Apparatus skill: ${SKILL_ROOT}
  This tick's friction: ${JSON.stringify(frictionData)}

  READ ALL of the following before proposing:
  1. \`ls ${SKILL_ROOT}/memory/\` — all memory file names
  2. Read the 2-3 most relevant memory files based on friction (especially
     feedback_rocq_proof_methodology.md, feedback_formal_project_docs.md,
     feedback_cmbt_methodology.md if relevant to this tick's issues)
  3. Read ${SKILL_ROOT}/STATUS.md (apparatus version + known gaps)
  4. \`cat ${METHODOLOGY_BACKLOG}\` — the methodology upgrade backlog
     (these are pre-approved upgrades waiting to be folded into the apparatus)
  5. \`cat ${FEEDBACK_LOG} 2>/dev/null | tail -60\` — recent feedback trends

  DECISION LOGIC:
  A. First check the methodology-upgrade-sources.md backlog.
     If ANY backlog item maps to this tick's friction (e.g. the tick hit an independence
     gap → item 3 "orchestrator+fresh-reviewer independence" is relevant), propose folding
     that item into the apparatus skill as a domain-neutral memory or policy update.
     Mark the backlog item as folded in your proposal.

  B. If no backlog item maps, check if this tick's friction reveals a new gap not
     already covered by existing memories. If yes, propose a NEW memory file.

  C. If friction is fully covered by existing memories and no backlog item maps:
     return verdict "NO_CHANGE" (conservative — avoid noise).

  PROPOSAL RULES:
  - ONE proposed change per tick maximum
  - Domain-neutral: no project names (SPOC, Sarek, convergence-safety) in generic apparatus files
  - If proposing a backlog fold-in: set changeType = "methodology-backlog", include the
    backlog item text in methodologyBacklogItem
  - Memory frontmatter: name, description, metadata.type (feedback/user/project/reference)
  - Complete proposed content (including frontmatter) if changeType is memory-related
  - Under 400 lines`,
  { phase: 'Evolve', label: 'evolve-propose', schema: EVOLVE_SCHEMA }
)

log(`Evolve: ${evolveProposal.verdict}${evolveProposal.skillFile ? ' → ' + evolveProposal.skillFile : ''}${evolveProposal.changeType === 'methodology-backlog' ? ' [backlog fold-in]' : ''}`)

if (evolveProposal.verdict === 'CHANGE') {
  let approvedContent = null
  let currentContent = evolveProposal.proposedContent
  let attempt = 0

  while (!approvedContent && attempt < 3) {
    attempt++

    const evolveReview = await agent(
      `You are reviewing a proposed apparatus skill update. COLD REVIEW — you did NOT propose this.

      Target file: ${evolveProposal.skillFile}
      Change type: ${evolveProposal.changeType}
      Rationale: ${evolveProposal.rationale}
      ${evolveProposal.changeType === 'methodology-backlog' ? 'Backlog item being folded: ' + evolveProposal.methodologyBacklogItem : ''}

      Proposed content:
      ===BEGIN===
      ${currentContent}
      ===END===

      Read current target: \`cat "${evolveProposal.skillFile}" 2>/dev/null || echo "NEW FILE"\`
      Check for duplicates: \`grep -rl "${evolveProposal.skillFile ? evolveProposal.skillFile.split('/').pop().replace('.md','') : 'NONE'}" ${SKILL_ROOT}/memory/ 2>/dev/null | head -3\`

      REVIEW CRITERIA (ALL must pass for GO):
      1. Domain-neutral: no project-specific names (SPOC/Sarek/convergence-safety/tezos) in generic apparatus files
      2. Non-duplicative: doesn't repeat content from an existing memory
      3. Rationale is backed by actual tick friction (not hypothetical)
      4. Correct frontmatter (name, description, metadata.type fields present and valid)
      5. A future Claude would apply this correctly in a fresh context
      6. Under 400 lines
      7. If backlog fold-in: the item is folded verbatim as domain-neutral guidance, no source attribution

      GO = apply as-is. REVISE = exact required changes. BLOCK = do not apply.`,
      { phase: 'Evolve', label: `evolve-review-${attempt}`, model: 'fable', schema: REVIEW_SCHEMA }
    )

    log(`Evolve Fable review ${attempt}: ${evolveReview.verdict}`)

    if (evolveReview.verdict === 'GO') {
      approvedContent = currentContent

      // Apply
      await agent(
        `Apply the approved apparatus skill update.

        Target file: ${evolveProposal.skillFile}
        Content:
        ===BEGIN===
        ${approvedContent}
        ===END===

        Steps:
        1. Write the file (create or overwrite as appropriate)
        2. If it is a NEW file in ${SKILL_ROOT}/memory/ and a MEMORY.md index exists there,
           append one line: "- [Title](filename.md) — one-line description"
           to \`${SKILL_ROOT}/memory/MEMORY.md\` (if that file exists)
        3. If changeType is "methodology-backlog", also append a note to
           ${METHODOLOGY_BACKLOG} marking the folded item:
           "**[FOLDED tick ${tick}]** — folded into ${evolveProposal.skillFile}"
        4. Confirm all writes`,
        { phase: 'Evolve', label: 'evolve-apply' }
      )
      log(`Skill update applied: ${evolveProposal.skillFile}`)

    } else if (evolveReview.verdict === 'REVISE' && attempt < 3) {
      const revised = await agent(
        `Revise the skill update per reviewer feedback.
        Target: ${evolveProposal.skillFile}
        Current content:
        ===BEGIN===
        ${currentContent}
        ===END===
        Issues: ${JSON.stringify(evolveReview.issues)}
        Required changes: ${JSON.stringify(evolveReview.requiredChanges)}
        Return ONLY the complete revised content.`,
        { phase: 'Evolve', label: `evolve-revise-${attempt}`,
          schema: { type: 'object', required: ['content'], properties: { content: { type: 'string' } } } }
      )
      currentContent = revised.content
    } else {
      log(`Skill update blocked at attempt ${attempt}`)
      break
    }
  }

  if (!approvedContent) log(`Evolve: no skill change applied this tick`)
}

// ── Phase 5: Feedback + Self-Adaptation ──────────────────────────────────────
phase('Feedback')

const feedback = await agent(
  `Write workflow feedback for tick ${tick} of the formal-verif-autopilot experiment.

  Tick summary:
  - Task: [${planState.currentTask.id}] ${planState.currentTask.title}
  - Execute status: ${taskResult.status}
  - Baseline apparatus check: ${baselineCheck.passed ? 'PASS' : 'ISSUES: ' + JSON.stringify(baselineCheck.links)}
  - Gate: ${gateVerdict}
  - Committed: ${committed}
  - Review issues: ${JSON.stringify(reviewResult ? reviewResult.issues : [])}
  - Evolve: ${evolveProposal.verdict} ${evolveProposal.changeType === 'methodology-backlog' ? '[backlog fold-in]' : ''}
  - Blockers: ${JSON.stringify(planState.blockers)}

  EXPERIMENT CONTEXT: the workflow and apparatus skill co-evolve. Feedback should be actionable:
  identify what the workflow's phase structure missed, what the apparatus skill lacked,
  and what concrete script change (file/phase/line) would fix it.

  For scriptChangesNeeded: true ONLY if a specific, named change to this .js script
  would improve the next tick. Describe it as: "In Phase X, change Y to Z because A."
  Vague improvements are NOT script changes.`,
  { phase: 'Feedback', label: 'feedback', schema: FEEDBACK_SCHEMA }
)

// Append to feedback log
await agent(
  `Append this tick's feedback entry to ${FEEDBACK_LOG}.
  Create if missing with heading: # Formal Verif Autopilot — Workflow Feedback Log

  Append:

  ## Tick ${tick} — [${planState.currentTask.id}] ${planState.currentTask.title}
  **${new Date().toISOString().slice(0,10)}** | Gate: ${gateVerdict} | Committed: ${committed} | Evolve: ${evolveProposal.verdict}

  **Baseline issues**: ${baselineCheck.passed ? 'none' : JSON.stringify(Object.entries(baselineCheck.links || {}).filter(([,v])=>v==='FAIL').map(([k])=>k))}

  **Friction**:
  ${feedback.frictionPoints.map(f => '- ' + f).join('\n') || '- none'}

  **Workflow improvements**:
  ${feedback.workflowImprovements.map(i => '- ' + i).join('\n') || '- none'}

  **Skill improvements**:
  ${feedback.skillImprovements.map(i => '- ' + i).join('\n') || '- none'}

  ---`,
  { phase: 'Feedback', label: 'feedback-write' }
)

// Self-adapt script if concrete change identified
const selfAdapt = (feedback.scriptChangesNeeded && scriptPath && feedback.scriptChangeSummary)
  ? await agent(
    `Self-adapt the workflow script based on concrete feedback.
    Script: ${scriptPath}
    Feedback: ${feedback.scriptChangeSummary}
    Tick: ${tick}

    1. Read the script
    2. Apply ONLY the specific changes described — no scope creep
    3. Validate: meta block intact, JS syntactically valid, no phase() calls removed
    4. Write back to ${scriptPath}
    Return { changed: true/false, rationale: "what changed + why" }`,
    { phase: 'Feedback', label: 'self-adapt', schema: {
      type: 'object', required: ['changed', 'rationale'],
      properties: { changed: { type: 'boolean' }, rationale: { type: 'string' } }
    }}
  )
  : { changed: false, rationale: scriptPath ? 'no concrete script change requested' : 'no scriptPath provided' }

log(`Self-adapt: ${selfAdapt.changed ? 'updated — ' + selfAdapt.rationale : 'no change'}`)
log(`Done: ${feedback.workflowImprovements.length} workflow + ${feedback.skillImprovements.length} skill suggestions`)

// ── Return ────────────────────────────────────────────────────────────────────
const verdict =
  committed               ? 'COMMITTED'
  : gateVerdict === 'GO'  ? 'GATE-PASS-NOT-COMMITTED'
  : gateVerdict.startsWith('SKIP') ? 'SKIPPED'
  : 'GATE-FAIL'

return {
  tick,
  verdict,
  taskId:       planState.currentTask.id,
  taskStatus:   taskResult.status,
  gateVerdict,
  committed,
  commitSha,
  evolveVerdict:  evolveProposal.verdict,
  evolveType:     evolveProposal.changeType,
  scriptAdapted:  selfAdapt.changed,
  nextTaskId:     taskResult.nextTaskId,
  baselineLinks:  baselineCheck.links,
  gateLinks:      gateCheck ? gateCheck.links : {},
  feedback: {
    workflow: feedback.workflowImprovements.length,
    skill:    feedback.skillImprovements.length,
    friction: feedback.frictionPoints.length,
  },
}
