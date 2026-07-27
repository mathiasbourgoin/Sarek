(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** What the INTERPRETER provides, stated by the evaluator that provides it
    (backlog-154).

    {1 The defect this exists to close}

    [Execute.check_device_capabilities] is applied on all three [Execute.run]
    dispatch paths. [Execute.run_interpreter_vectors] — the entry point the
    cross-backend numeric oracle actually goes through — did not apply it. The
    two answers disagreed, and the disagreement was observed on this
    workstation, in one run of [test_coopmat_integer_e2e]:

    {v
      OK  launch gate refuses on CPU Interpreter (Sequential)
      OK  launch gate refuses on CPU Interpreter (Parallel, 32 cores)
      ...
      OK  16 tiles, all 65536 u8 operand pairs, bit-identical to the interpreter
    v}

    The gate refuses the Interpreter for the very kernel the interpreter then
    computes 65536 correct results for. The refusal is not wrong on its own
    terms: [Framework_sig.capabilities.coopmat] is [None] for this backend,
    {!Sarek_coopmat.verdict} maps [None] to {!Sarek_capability.Unknown}, and
    [Unknown] does not permit. Every step is the capability model working as
    designed.

    What that means is worth stating plainly, because it is the actual finding:
    {b the bypass is load-bearing.} The oracle is reachable only because the
    second entry point skips the gate. Adding the gate to
    [run_interpreter_vectors] without first fixing what the interpreter
    ADVERTISES would not close a hole, it would break the oracle.

    {1 What an interpreter should advertise}

    Neither of the two obvious answers survives.

    [Unknown] — the status quo — is wrong. [Unknown] means "we could not probe".
    For hardware that is an honest confession, and refusing on it is right. For
    an interpreter there is nothing to probe: the answer is in the evaluator,
    and reading the evaluator IS the probe. Reporting [Unknown] for a question
    whose answer is in the same repository is not conservatism; it is declining
    to answer, and here it declines on behalf of the one backend every numeric
    claim in the project is checked against.

    "Supports everything" is wrong too, and not only in principle. It is
    factually false: {!Sarek_ir_interp_eval.coopmat_refuse_float} refuses float
    cooperative-matrix accumulation, and refuses it for a reason —
    [SPV_KHR_cooperative_matrix] leaves the order of the k+1 additions to the
    implementation, so there is no single value a strict oracle could compare
    against. An interpreter that claimed everything would validate the backends
    against shapes it cannot evaluate and configurations no device offers.

    The rule that survives both is:
    {b an interpreter advertises exactly what its evaluator implements, as a
       by-construction measurement of that evaluator, and never as "unprobed".}
    It is falsifiable — every claim below names the code that makes it true — it
    cannot over-claim, and it never refuses something the oracle can actually
    do.

    {1 Why this is a predicate and not a [device_support] record}

    {!Framework_sig.capabilities.coopmat} models support as a finite
    [ds_configs] list, which is the right shape for hardware: a device
    enumerates what it advertises, and
    [vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR] hands you the list.

    The interpreter is not enumerable. Every invocation holds the whole matrix
    ({!Sarek_ir_interp_value.env}.[coopmats]), so it evaluates integer
    configurations at ANY shape — an infinite set that no [ds_configs] can
    state. That mismatch, and not an oversight, is why [coopmat = None] was the
    only available answer at the plugin: the record could not express the truth,
    so it expressed nothing.

    Hence a predicate. This is a general observation about the capability model
    rather than a fact about cooperative matrices, and it is written down here
    because the same wall is waiting for the Native backend and for any future
    software target. *)

(** The wide element types the interpreter evaluates.

    Every entry is [By_construction] — point at the code:

    - [Float64]: OCaml floats are binary64; [VFloat64] carries them unchanged.
    - [Int64]: [Int64.t] is native; [VInt64] carries it unchanged.
    - [Float16]: {!Sarek_float16} implements binary16 round-to-nearest-even, and
      [Sarek_ir_interp.interp_array_to_vector] rounds the writeback through a
      [Bigarray.Float16] cell so "cast then store" and "store" agree.

    {b [Float16] is the second instance of the same gap}, and it is already
    loaded. [Interpreter_plugin_base] and [Native_plugin_base] both report
    [device_features = [Float64; Int64]], omitting an f16 they both implement.
    Nothing refuses today only because [Execute.check_device_capabilities]
    excludes [Float16] from its [gated] list — deliberately, and for an
    unrelated reason (no backend probes shaderFloat16 yet). The day that list
    widens with the f16 probe, [run] will refuse an f16 kernel on the
    Interpreter that [run_interpreter_vectors] runs correctly: coopmat's defect
    exactly, in a second capability, waiting on a change already planned. *)
val device_features : Sarek_ir_analysis.feature list

(** [coopmat_verdict cfg] judges a cooperative-matrix configuration against the
    interpreter's evaluator.

    [Available] for integer component types at any shape:
    [SPV_KHR_cooperative_matrix] states integer accumulation is exact at the
    precision of the result type, so a strict oracle has a single right answer
    to give.

    {!Sarek_capability.Unavailable} for any float component type, with
    [cap_kind = Policy]: the interpreter could produce a plausible number, and
    refuses to, because the specification leaves the addition order to the
    implementation. [Policy] rather than [Backend_structural] is the honest kind
    — this is a decision about what an oracle may claim, revisable by a
    decision, not a property of OCaml. It mirrors
    {!Sarek_ir_interp_eval.coopmat_refuse_float}, which is where a kernel that
    slips past this gate lands anyway; the difference is that this one fires
    before any partial writeback, and names the capability. *)
val coopmat_verdict : Sarek_coopmat.config -> Sarek_capability.verdict

(** [first_refusal ir] is the first capability [ir] requires that the
    interpreter does not provide, or [None] when it provides all of them.

    Written in terms of {!Sarek_capability.permits} so an [Unknown] cannot leak
    through as permitted.

    The WIDTH features ([Float64], [Float16], [Int64]) are judged against
    {!device_features}, all three of them — which is safe HERE, and not in
    [Execute.check_device_capabilities], precisely because {!device_features} is
    a by-construction claim about this evaluator rather than an unwritten device
    probe. That is why the interpreter gate can be stricter than the device gate
    instead of looser.

    [Coopmat] is a {!Sarek_ir_analysis.feature} too but is deliberately NOT
    judged that way: it is decided per configuration by {!coopmat_verdict}.
    Treating it as a boolean is wrong in both directions — absent, it refuses
    the integer matrices this evaluator computes; present, it permits the float
    ones it refuses. Both were observed while writing this: gating it by
    membership made the integer positive control fail with
    [the device advertises no cooperative-matrix support].

    Returns a verdict rather than raising: the caller renders and raises through
    its own error type, so the diagnostic reads the same whichever entry point
    produced it. *)
val first_refusal : Sarek_ir_types.kernel -> Sarek_capability.verdict option
