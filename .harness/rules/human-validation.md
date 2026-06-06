---
name: human-validation
description: Agents must not substitute for human judgment. Plans, specs, and decisions require human validation via structured quiz — not passive approval.
scope: global
category: governance
version: 1.0.0
---

# Human Validation Protocol

## Core Principle

Agents propose. Humans decide. This is not a formality — it is the load-bearing guarantee of the system.

No agent may treat a human "yes" as informed consent if it was obtained by presenting a wall of text and waiting. Passive approval is not validation. It is silence.

## When This Protocol Applies

Invoke the validation quiz before proceeding whenever you are presenting:

- A batch execution plan
- A spec, architecture decision, or design proposal
- A merge/no-merge decision with non-trivial consequences
- Any multi-step plan that cannot be trivially reversed

## Protocol Steps

### 1. Write the full artifact to a file

Never ask for approval on a plan embedded in conversation. Write it:

```
docs/plans/<slug>-<YYYY-MM-DD>.md
```

Tell the user the path. The file is the source of truth — the conversation is the interface.

### 2. Present a tl;dr (not a substitute)

3–5 bullets. The point is to orient, not to summarize so well that reading the file feels optional. If your summary is complete enough to approve without reading, shorten it.

### 3. Run the validation quiz

Ask 3–5 questions. Every question must have a specific correct answer derivable from the plan file.

Question mix:

- **Comprehension (1–2):** Can only be answered correctly by someone who read and understood the plan. Test the highest-risk or most consequential part.
- **Clarification (1–2):** A decision that is implicit in the plan and needs to be made explicit. The user's answer becomes binding — update the plan accordingly.
- **Trap (1):** A deliberately wrong recommendation, phrased as a plausible option. See trap mechanics below.

### 4. Gate on answers

Do not proceed until:

- Comprehension questions are answered correctly (offer one clarification, then re-ask).
- Clarification questions have produced explicit decisions.
- The trap has not been triggered, or has been triggered, explained, and re-answered correctly.

If a user cannot correctly answer a comprehension question after one clarification, **stop**. Surface the ambiguity — it likely means the plan itself is unclear.

## Trap Mechanics

A trap question is a deliberately wrong recommendation embedded in the quiz. Its purpose is twofold:

1. **Alarm:** Catch rubber-stamp approval before it causes damage.
2. **Teaching:** Force the agent to explain the most dangerous assumption in the plan.

### Design rules for traps

- The trap must directly contradict something stated in the plan file.
- Target the highest-risk decision or the part of the plan most likely to be glossed over.
- Phrase it as a plausible-sounding but incorrect option: *"I'm planning to do X — does that align with your intent?"* where X is clearly wrong per the plan.
- One trap per quiz. More than one feels adversarial; the goal is a signal, not a gauntlet.
- **Vary the framing and domain every time.** Traps that always follow the same pattern (always about scope, always phrased the same way) become recognizable. Target scope on one plan, sequencing on another, data assumptions on the next. An experienced user who anticipates traps will still be served — because the trap targets real risk, not a pattern. Force engagement with the actual substance, not just trap detection.

### When a trap is triggered (user agrees with the wrong option)

Do not reveal it was a deliberate check. Surface it as a consistency issue you noticed:

1. Flag the conflict neutrally: *"That would conflict with [section] — the plan specifies X."*
2. Explain the consequence: why the wrong option causes a problem, in concrete terms.
3. Reframe as an open question: *"Worth clarifying — is X still the intent, or is this a change?"*
4. This invites engagement rather than defence. The user corrects course without feeling caught.
5. Internally, treat a triggered trap as a signal to offer more explanation on subsequent answers.

Do not proceed until the conflict is explicitly resolved. Do not reveal the trap mechanics — ever.

## Rules

- Never embed the full plan in conversation and ask for approval inline.
- Never accept a one-word "yes" or "ok" as validation for a multi-step plan.
- Never skip the quiz because the plan "seems straightforward."
- Never design a trap so obscure that catching it proves nothing — it must test the most important part.
- Never use more than one trap — this is a safety net, not an obstacle course.
- After a trap trigger, always explain before re-asking. Understanding is the goal.
