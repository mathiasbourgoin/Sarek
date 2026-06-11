// formal-verif-autopilot.js
//
// Autonomous formal-verification plan driver for convergence-safety.
// Inspired by the bounty-autopilot pattern from ~/dev/bounty-skills.
//
// STRUCTURE (one tick = one bounded sweep):
//   Phase 0 Adapt    — read plan state, re-adapt this workflow script if needed
//   Phase 1 Execute  — execute the current open formal task
//   Phase 2 Gate     — coqchk 0-axiom gate + independent Fable review until GO
//   Phase 3 Evolve   — propose skill update → Fable review loop until GO
//   Phase 4 Feedback — write friction notes, self-adapt script for next tick
//
// Run with:   Workflow({ scriptPath: "<this file>", args: { tick: 0 } })
// Next tick:  Workflow({ scriptPath: "<this file>", args: { tick: 1 } })
// The workflow self-patches its own script at the end of each tick (Feedback phase).

export const meta = {
  name: 'formal-verif-autopilot',
  description: 'Autonomous Rocq formal-verification plan execution with skill evolution',
  phases: [
    { title: 'Adapt',    detail: 'Read plan state, re-adapt workflow for this tick' },
    { title: 'Execute',  detail: 'Execute the selected open formal task' },
    { title: 'Gate',     detail: 'coqchk 0-axiom gate + independent Fable review until GO' },
    { title: 'Evolve',   detail: 'Update formal-apparatus skill, Fable review loop until GO' },
    { title: 'Feedback', detail: 'Write improvement notes, self-adapt script for next tick' },
  ],
}

// ── Constants ─────────────────────────────────────────────────────────────────
const SPOC        = '/home/mathias/dev/SPOC'
const FORMAL      = `${SPOC}/formal/convergence-safety`
const SKILL_ROOT  = '/home/mathias/.claude/skills/formal-apparatus'
const PLAN_FILE   = `${FORMAL}/PLAN.md`
const FEEDBACK_LOG = `${FORMAL}/report/WORKFLOW_FEEDBACK.md`

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
          id:         { type: 'string' },
          title:      { type: 'string' },
          tier:       { type: 'string' },
          status:     { type: 'string', enum: ['open', 'in-progress', 'blocked', 'done'] },
          blockedBy:  { type: 'array', items: { type: 'string' } },
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

const TASK_RESULT_SCHEMA = {
  type: 'object',
  required: ['taskId', 'status', 'changes', 'coqchkPass', 'commitMessage'],
  properties: {
    taskId:         { type: 'string' },
    status:         { type: 'string', enum: ['complete', 'partial', 'blocked', 'skipped'] },
    changes:        { type: 'array', items: { type: 'string' } },
    coqchkPass:     { type: 'boolean' },
    coqchkOutput:   { type: 'string' },
    testResults:    { type: 'string' },
    commitMessage:  { type: 'string' },
    nextTaskId:     { type: 'string' },
    notes:          { type: 'string' },
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

const EVOLVE_PROPOSAL_SCHEMA = {
  type: 'object',
  required: ['verdict', 'rationale'],
  properties: {
    verdict:         { type: 'string', enum: ['CHANGE', 'NO_CHANGE'] },
    skillFile:       { type: 'string' },
    rationale:       { type: 'string' },
    proposedContent: { type: 'string' },
    changeType:      { type: 'string', enum: ['new-memory', 'update-memory', 'update-policy', 'update-skill-doc'] },
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

const SELF_ADAPT_SCHEMA = {
  type: 'object',
  required: ['changed', 'rationale'],
  properties: {
    changed:       { type: 'boolean' },
    rationale:     { type: 'string' },
    newScriptPath: { type: 'string' },
  },
}

// ── Phase 0: Adapt ────────────────────────────────────────────────────────────
phase('Adapt')
log(`Tick ${tick} — reading plan state`)

const lastFeedbackHint = tick > 0
  ? `Also read ${FEEDBACK_LOG} for what the previous tick recommended.`
  : 'This is tick 0 — no prior feedback.'

const planState = await agent(
  `You are the formal verification plan manager for convergence-safety.

  Tick: ${tick}
  SPOC root: ${SPOC}
  Formal project: ${FORMAL}

  READ ALL of the following:
  1. \`cat ${FORMAL}/STATUS.md\`
  2. \`cd ${SPOC} && git log --oneline -8 2>&1\`
  3. \`cd ${SPOC} && git status 2>&1\`
  4. \`gh pr list --repo mathiasbourgoin/Sarek --state open --json number,title,state 2>/dev/null || echo "gh unavailable"\`
  5. \`cat ${PLAN_FILE} 2>/dev/null || echo "PLAN.md not found"\`
  ${lastFeedbackHint}

  KNOWN ROADMAP — use this to seed openTasks in priority order:
  (Check current STATUS.md to see which are already done; skip those.)

  T1 — Spec completeness (all non-admitted, 0-axiom):
  - T1A-CONF   : Extend QCheck conformance tests to cover ESuperstep in abstract OCaml model
                 (blocked by PR #182 merge; add ≥3 new properties testing ESuperstep false/true)
  - T1-LATEX   : LaTeX spec (ConvergenceSafetySpec.tex) reconciliation — update theorem list to
                 include superstep_outer_diverged_error; remove stale elision note for TESuperstep
  - T1-LEDGER  : proof-ledger.json sync — add superstep_outer_diverged_error entry; update theorem
                 count; verify all fields match STATUS.md

  T2 — Spec soundness gaps (requires new Rocq definitions):
  - T2-F02     : Env-threaded is_varying in Rocq spec. Add VarEnv type (var→bool finite map),
                 is_varying_in_env, update check to thread env, update ELet/TEVar cases,
                 add theorem is_varying_env_monotone, update all inductive proofs.
  - T2-WARP    : WarpConvergence error class. Extend error inductive (WarpError),
                 add warp_check function for WarpConvergence-tagged primitives,
                 add theorem warp_diverged_error (analogous to superstep_outer_diverged_error).
  - T2-RETURN  : TEReturn early-return analysis. Add EReturn constructor, model barrier-skip
                 through early-return in diverged branch. Add safety theorem.

  T3 — Semantic soundness (human gate required before starting):
  - T3-GATE    : Human decision point — confirm T3-SEMANTIC is in scope before proceeding
  - T3-SEMANTIC: Semantic soundness with eval relation for texpr (months; separate sub-project)

  Hygiene (run at every tick):
  - DOCS-SYNC  : STATUS.md / ASSUMPTIONS.md / proof-ledger.json drift check and sync

  BLOCKER RULES:
  - If PR #182 is still open (not merged), set T1A-CONF.blockedBy = ["HARD: PR #182 not merged"]
  - The TOP-LEVEL blockers array is ONLY for the currentTask's blockers (tasks already dequeued)
  - Prefix with "HARD:" in blockers only when the SELECTED currentTask cannot execute this tick

  For currentTask: pick the first task whose blockedBy contains no "HARD:" entries.
  For workflowAdaptations: note any changes vs the previous tick plan.

  Write the updated plan to ${PLAN_FILE}.`,
  { phase: 'Adapt', label: 'plan-adapt', schema: PLAN_SCHEMA }
)

log(`Current task: [${planState.currentTask.id}] ${planState.currentTask.title}`)
if (planState.blockers.length > 0) log(`Blockers: ${planState.blockers.join(' | ')}`)
if (planState.workflowAdaptations.length > 0) log(`Adaptations: ${planState.workflowAdaptations.join('; ')}`)

// HARD-block only when the CURRENT task cannot execute — not when a backlog item is blocked
// T3-GATE is a human decision point — always hard-block on it so the loop stops for review
const currentIsT3Gate = planState.currentTask.id === 'T3-GATE'
const hardBlocked = currentIsT3Gate || planState.blockers.some(b => b.startsWith('HARD:'))
if (hardBlocked) {
  const reason = currentIsT3Gate
    ? 'HUMAN-GATE: All T1+T2 tasks complete. Confirm T3-SEMANTIC (semantic soundness) is in scope before proceeding.'
    : planState.blockers.find(b => b.startsWith('HARD:'))
  log(`Hard blocker — returning early`)
  return { tick, verdict: 'HARD-BLOCKED', reason, planState }
}

// ── Phase 1: Execute ──────────────────────────────────────────────────────────
phase('Execute')

const taskResult = await agent(
  `You are executing a formal verification task for the convergence-safety Rocq project.

  Task ID: ${planState.currentTask.id}
  Title:   ${planState.currentTask.title}
  Approach: ${planState.currentTask.approach}
  Expected deliverables: ${JSON.stringify(planState.currentTask.expectedDeliverables)}

  SPOC root: ${SPOC}
  Rocq spec: ${FORMAL}/theories/ConvergenceSpec.v
  Conformance tests: ${FORMAL}/test/test_convergence_conformance.ml
  Extraction tests:  ${FORMAL}/test/test_convergence_extraction.ml
  Live CMBT tests:   ${FORMAL}/test/test_convergence_live.ml
  STATUS.md: ${FORMAL}/STATUS.md

  EXECUTION PROTOCOL:
  1. Read all relevant source files before editing
  2. Make changes according to the task approach
  3. Verify Rocq compilation:
     \`cd ${FORMAL} && coqc theories/ConvergenceSpec.v 2>&1; echo "coqc:$?"\`
  4. If Rocq spec changed, run coqchk:
     \`cd ${FORMAL} && coqc -Q theories ConvergenceSpec theories/ConvergenceSpec.v && coqchk -Q theories ConvergenceSpec ConvergenceSpec.ConvergenceSpec 2>&1 | tail -4; echo "coqchk:$?"\`
  5. If OCaml tests changed: \`cd ${SPOC} && dune runtest 2>&1 | tail -20; echo "dune:$?"\`
  6. If STATUS.md counts changed, update it

  RULES:
  - Do NOT commit in this phase
  - coqchkPass = true ONLY if output contains "Modules were successfully checked" AND exit 0
  - If the task is "wait for PR merge": status = "blocked", changes = []
  - Commit message format: "type(scope): short description" (no Co-Authored-By here)`,
  { phase: 'Execute', label: `exec-${planState.currentTask.id}`, schema: TASK_RESULT_SCHEMA }
)

log(`Execution: ${taskResult.status} | coqchk: ${taskResult.coqchkPass} | ${taskResult.changes.length} change(s)`)

// ── Phase 2: Gate ─────────────────────────────────────────────────────────────
phase('Gate')

let gateVerdict = 'SKIP'
let reviewResult = null

if (taskResult.status === 'complete') {

  if (!taskResult.coqchkPass) {
    gateVerdict = 'FAIL-COQCHK'
    log(`Gate FAIL: coqchk did not pass — ${taskResult.coqchkOutput || 'no output'}`)

  } else {
    // Independent Fable review — fresh context, no memory of the Execute phase
    reviewResult = await agent(
      `You are an independent Rocq/Coq formal verification reviewer.
      You have NO context from the implementation phase. Cold review only.

      Project: ${FORMAL}
      Task just executed: ${JSON.stringify(planState.currentTask)}
      Claimed changes: ${JSON.stringify(taskResult.changes)}

      REVIEW PROTOCOL:
      1. Read ${FORMAL}/theories/ConvergenceSpec.v
      2. Run \`cd ${FORMAL} && coqc theories/ConvergenceSpec.v 2>&1; echo "exit:$?"\`
      3. Check for admits: \`grep -n "Admitted\\." ${FORMAL}/theories/ConvergenceSpec.v; echo "admitted-check-done"\`
      4. For each new or changed proof: verify all inductive cases are handled
         (especially ESuperstep — added Phase 1a — must appear in every expr_list_rect proof)
      5. Check STATUS.md is consistent with the spec (theorem count, coqchk status)
      6. Check any changed test file actually exercises the modified spec definitions

      Be adversarial. If something is wrong: REVISE with exact file+line fixes needed.
      Only GO if everything is genuinely correct and complete.`,
      { phase: 'Gate', label: 'fable-review-1', model: 'fable', schema: REVIEW_SCHEMA }
    )

    log(`Fable review 1: ${reviewResult.verdict} — ${reviewResult.issues.length} issues`)

    if (reviewResult.verdict === 'GO') {
      gateVerdict = 'GO'

    } else if (reviewResult.verdict === 'REVISE') {
      // One fix attempt before second review
      const fixResult = await agent(
        `Apply the reviewer-required fixes to the formal work.

        Issues: ${JSON.stringify(reviewResult.issues)}
        Required changes: ${JSON.stringify(reviewResult.requiredChanges)}
        Files: ${FORMAL}/theories/ConvergenceSpec.v (and any test files listed)

        Fix all required changes then re-run coqchk:
        \`cd ${FORMAL} && coqc -Q theories ConvergenceSpec theories/ConvergenceSpec.v && coqchk -Q theories ConvergenceSpec ConvergenceSpec.ConvergenceSpec 2>&1 | tail -4; echo "coqchk:$?"\`

        Return updated changes and coqchk result.`,
        { phase: 'Gate', label: 'fix-1', schema: TASK_RESULT_SCHEMA }
      )

      if (fixResult.coqchkPass) {
        const review2 = await agent(
          `You are an independent Rocq reviewer, second pass. Cold review.

          The spec at ${FORMAL}/theories/ConvergenceSpec.v was revised.
          Prior issues: ${JSON.stringify(reviewResult.issues)}
          Claimed fixes: ${JSON.stringify(fixResult.changes)}

          1. Read the spec
          2. Run coqc to verify
          3. Check ALL prior issues are resolved
          4. GO if all resolved. BLOCK if any remain (no more REVISE after 2 rounds).`,
          { phase: 'Gate', label: 'fable-review-2', model: 'fable', schema: REVIEW_SCHEMA }
        )
        log(`Fable review 2: ${review2.verdict}`)
        reviewResult = review2
        gateVerdict = review2.verdict === 'GO' ? 'GO' : 'BLOCK'
      } else {
        gateVerdict = 'FAIL-COQCHK-AFTER-FIX'
      }

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
if (gateVerdict === 'GO') {
  const commitResult = await agent(
    `Commit and push the gate-cleared formal verification changes.

    SPOC root: ${SPOC}
    1. \`cd ${SPOC} && git branch --show-current\` — confirm branch
    2. \`cd ${SPOC} && git status\` — see what changed
    3. Stage ONLY formal files (never test_dft.ml, never .opam-ci/):
       Add: formal/convergence-safety/theories/*.v formal/convergence-safety/theories/*.vo
            formal/convergence-safety/theories/*.glob formal/convergence-safety/theories/.*.aux
            formal/convergence-safety/STATUS.md formal/convergence-safety/ASSUMPTIONS.md
            formal/convergence-safety/PLAN.md formal/convergence-safety/test/*.ml
            (only the ones git status shows as modified)
    4. Commit:
       \`cd ${SPOC} && git commit -m "$(cat <<'COMMITMSG'\n${taskResult.commitMessage}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\nCOMMITMSG\n)"\`
    5. Push: \`cd ${SPOC} && git push 2>&1\`
    6. Return { committed: true/false, sha: "...", pushOutput: "..." }`,
    { phase: 'Gate', label: 'commit', schema: {
      type: 'object', required: ['committed'],
      properties: { committed: { type: 'boolean' }, sha: { type: 'string' }, pushOutput: { type: 'string' } }
    }}
  )
  committed = commitResult.committed
  log(`Commit: ${committed ? 'OK sha=' + (commitResult.sha || '?') : 'FAILED'}`)
}

// ── Phase 3: Evolve ───────────────────────────────────────────────────────────
phase('Evolve')

const frictionData = {
  tick,
  taskId:        planState.currentTask.id,
  taskStatus:    taskResult.status,
  gateVerdict,
  reviewIssues:  reviewResult ? reviewResult.issues   : [],
  reviewPositives: reviewResult ? reviewResult.positives : [],
  changes:       taskResult.changes.length,
  notes:         taskResult.notes || '',
}

const evolveProposal = await agent(
  `You are a formal-apparatus skill improvement agent.

  Skill root: ${SKILL_ROOT}
  This tick's friction data: ${JSON.stringify(frictionData)}

  READ:
  1. \`ls ${SKILL_ROOT}/memory/\` — all memory file names
  2. Read 2-3 of the most relevant memory files based on the friction data
  3. Read ${SKILL_ROOT}/SKILL.md sections that relate to the friction
  4. \`cat ${FEEDBACK_LOG} 2>/dev/null | tail -40\` — recent feedback trends

  DECIDE:
  - Did this tick reveal a genuine skill gap not already covered by a memory?
  - If YES: propose a targeted change (new memory, update existing memory, or update policy)
  - If NO: return verdict "NO_CHANGE" (preferred — be conservative)

  CONSTRAINTS:
  - ONE proposed change maximum per tick
  - Domain-neutral: no project names (SPOC, Sarek, convergence-safety) in generic skill files
  - Memory frontmatter schema: name + description + metadata.type (user/feedback/project/reference)
  - If proposing a NEW memory file: provide complete content including frontmatter
  - If updating an existing memory: read it first, then provide complete new content`,
  { phase: 'Evolve', label: 'evolve-propose', schema: EVOLVE_PROPOSAL_SCHEMA }
)

log(`Evolve: ${evolveProposal.verdict}${evolveProposal.skillFile ? ' → ' + evolveProposal.skillFile : ''}`)

if (evolveProposal.verdict === 'CHANGE') {
  let approvedContent = null
  let currentContent = evolveProposal.proposedContent
  let attempt = 0

  while (!approvedContent && attempt < 3) {
    attempt++

    const evolveReview = await agent(
      `You are reviewing a proposed formal-apparatus skill update. Cold review — you did NOT propose this.

      Target file: ${evolveProposal.skillFile}
      Change type: ${evolveProposal.changeType}
      Rationale: ${evolveProposal.rationale}

      Proposed content:
      ===BEGIN===
      ${currentContent}
      ===END===

      Read the current file: \`cat ${evolveProposal.skillFile} 2>/dev/null || echo "NEW FILE"\`
      Also check for conflicts: \`ls ${SKILL_ROOT}/memory/ | xargs grep -l "${evolveProposal.skillFile ? evolveProposal.skillFile.split('/').pop().replace('.md','') : 'none'}" 2>/dev/null\`

      REVIEW CRITERIA (all must pass for GO):
      1. Domain-neutral: no project-specific names in generic apparatus files
      2. Non-duplicative: doesn't repeat an existing memory
      3. Rationale is backed by actual tick friction (not hypothetical)
      4. Correct frontmatter (name, description, metadata.type fields present)
      5. Future Claude would apply this correctly in a fresh context
      6. Under 400 lines (memories must be concise)

      GO = apply as-is. REVISE = give specific required changes. BLOCK = do not apply.`,
      { phase: 'Evolve', label: `evolve-review-${attempt}`, model: 'fable', schema: REVIEW_SCHEMA }
    )

    log(`Evolve review ${attempt}: ${evolveReview.verdict}`)

    if (evolveReview.verdict === 'GO') {
      approvedContent = currentContent
      // Apply the change
      await agent(
        `Apply the approved skill file update.

        File to write: ${evolveProposal.skillFile}
        Content:
        ===BEGIN===
        ${approvedContent}
        ===END===

        Steps:
        1. Write the file using the Write tool
        2. If this is a NEW file (not an update), also append a line to
           \`ls ${SKILL_ROOT}/memory/ | grep -q MEMORY.md && echo "MEMORY.md exists"\`
           — if MEMORY.md exists, append: "- [FileName](filename.md) — one-line hook"
        3. Confirm both writes succeeded`,
        { phase: 'Evolve', label: `evolve-apply` }
      )
      log(`Skill update applied: ${evolveProposal.skillFile}`)

    } else if (evolveReview.verdict === 'REVISE' && attempt < 3) {
      const revised = await agent(
        `Revise the skill update per Fable reviewer feedback.

        Target: ${evolveProposal.skillFile}
        Rationale: ${evolveProposal.rationale}
        Current content:
        ===BEGIN===
        ${currentContent}
        ===END===
        Issues: ${JSON.stringify(evolveReview.issues)}
        Required changes: ${JSON.stringify(evolveReview.requiredChanges)}

        Produce the COMPLETE revised content only (no commentary).`,
        { phase: 'Evolve', label: `evolve-revise-${attempt}`,
          schema: { type: 'object', required: ['content'], properties: { content: { type: 'string' } } } }
      )
      currentContent = revised.content

    } else {
      log(`Skill update blocked at attempt ${attempt}`)
      break
    }
  }

  if (!approvedContent) log(`Evolve: no skill change applied (blocked or max attempts)`)
}

// ── Phase 4: Feedback + Self-Adaptation ──────────────────────────────────────
phase('Feedback')

const feedback = await agent(
  `Write a concise workflow feedback note for tick ${tick}.

  Tick summary:
  - Task: [${planState.currentTask.id}] ${planState.currentTask.title}
  - Status: ${taskResult.status}
  - Gate: ${gateVerdict}
  - Committed: ${committed}
  - Review issues: ${JSON.stringify(reviewResult ? reviewResult.issues : [])}
  - Evolve: ${evolveProposal.verdict}
  - Blockers hit: ${JSON.stringify(planState.blockers)}
  - Workflow adaptations noted: ${JSON.stringify(planState.workflowAdaptations)}

  For "scriptChangesNeeded": true ONLY if a concrete, specific change to THIS workflow
  script would meaningfully improve the next tick. "Be more thorough" is NOT a script change.
  A script change must be describable as: "in phase X, add/change Y to Z because A".`,
  { phase: 'Feedback', label: 'feedback', schema: FEEDBACK_SCHEMA }
)

// Append to feedback log
await agent(
  `Append this tick's feedback to ${FEEDBACK_LOG}.

  Create the file if it doesn't exist with heading:
  # Formal Verif Autopilot — Workflow Feedback Log

  Then append:

  ## Tick ${tick} — [${planState.currentTask.id}] ${planState.currentTask.title}

  **Verdict**: ${gateVerdict} | **Committed**: ${committed} | **Evolve**: ${evolveProposal.verdict}

  **Friction points**:
  ${feedback.frictionPoints.map(f => '- ' + f).join('\n') || '- none'}

  **Workflow improvements**:
  ${feedback.workflowImprovements.map(i => '- ' + i).join('\n') || '- none'}

  **Skill improvements**:
  ${feedback.skillImprovements.map(i => '- ' + i).join('\n') || '- none'}

  ---`,
  { phase: 'Feedback', label: 'feedback-write' }
)

// Self-adapt: update this script if feedback identifies concrete improvements
const selfAdapt = (feedback.scriptChangesNeeded && scriptPath && feedback.scriptChangeSummary)
  ? await agent(
    `Self-adapt the workflow script based on concrete feedback.

    Script location: ${scriptPath}
    Feedback: ${feedback.scriptChangeSummary}
    Tick: ${tick}

    1. Read the script at ${scriptPath}
    2. Apply ONLY the specific changes described in the feedback
    3. Validate: the meta block must remain intact, JS must be syntactically valid
    4. Write the updated script back to ${scriptPath}
    5. Return { changed: true, rationale: "<what changed and why>", newScriptPath: "${scriptPath}" }

    GUARD: if the feedback is vague or the change would break the script structure, return
    { changed: false, rationale: "change too risky or too vague" }`,
    { phase: 'Feedback', label: 'self-adapt', schema: SELF_ADAPT_SCHEMA }
  )
  : { changed: false, rationale: scriptPath ? 'no concrete script change requested' : 'no scriptPath — pass args.scriptPath to enable self-adaptation' }

log(`Self-adapt: ${selfAdapt.changed ? 'script updated — ' + selfAdapt.rationale : 'no change'}`)
log(`Feedback done: ${feedback.workflowImprovements.length} workflow + ${feedback.skillImprovements.length} skill suggestions`)

// ── Return ────────────────────────────────────────────────────────────────────
const verdict =
  committed          ? 'COMMITTED'
  : gateVerdict === 'GO' ? 'GATE-PASS-NOT-COMMITTED'
  : gateVerdict.startsWith('SKIP') ? 'SKIPPED'
  : gateVerdict.startsWith('FAIL') || gateVerdict === 'BLOCK' ? 'GATE-FAIL'
  : 'HARD-BLOCKED'

return {
  tick,
  verdict,
  taskId:       planState.currentTask.id,
  taskTitle:    planState.currentTask.title,
  taskStatus:   taskResult.status,
  gateVerdict,
  committed,
  evolveVerdict: evolveProposal.verdict,
  scriptAdapted: selfAdapt.changed,
  scriptPath:   selfAdapt.newScriptPath || scriptPath,
  nextTaskId:   taskResult.nextTaskId,
  feedback: {
    workflow: feedback.workflowImprovements.length,
    skill:    feedback.skillImprovements.length,
    friction: feedback.frictionPoints.length,
  },
}
