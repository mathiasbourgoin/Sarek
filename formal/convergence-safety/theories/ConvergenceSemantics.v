(******************************************************************************)
(* ConvergenceSemantics.v — T3-S1
 *
 * Semantic domain and fuel-indexed big-step evaluator with barrier traces.
 * Requires ConvergenceSpec (frozen; 20 theorems, 0 admits, 0 axioms).
 *
 * Design decisions (from PLAN.md T3-SEMANTIC global decisions):
 * 1. New file; ConvergenceSpec.v stays frozen.
 * 2. Big-step with fuel: total Fixpoint returning option; EWhile nontermination
 *    absorbed by None.
 * 3. vary_val as Section parameter (Variable), not an axiom — preserves 0-axiom
 *    invariant.
 * 4. Safety = barrier-trace uniformity (not operational deadlock modelling).
 * 5. Outcome type (ONorm | ORet) from day one so EReturn never forces a refactor.
 *
 * T3-S1 specific decisions:
 * - ESuperstep emits implicit EvBarrier for both dv=true and dv=false (the
 *   runtime barrier exists regardless of the analysis flag; the checker difference
 *   is in what it flags, not in what the hardware executes).
 * - ORet short-circuits ESeq and EApp (subsequent elements are not evaluated).
 * - EReturn wraps the evaluated outcome as ORet.
 ******************************************************************************)

From Stdlib Require Import List Arith Lia.
Import ListNotations.
From ConvergenceSpec Require Import ConvergenceSpec.

(* ===== 1. Semantic domain ===== *)

(** Thread identifier *)
Definition tid := nat.

(** Runtime value — nat (booleans via 0 = false, nonzero = true;
    no value sum type needed at this abstraction level) *)
Definition value := nat.

(** Value environment: association list mapping variable id to value *)
Definition venv := list (nat * value).

Definition venv_lookup (rho : venv) (x : nat) : value :=
  match find (fun p => Nat.eqb (fst p) x) rho with
  | Some (_, v) => v
  | None        => 0
  end.

Definition venv_extend (rho : venv) (x : nat) (v : value) : venv :=
  (x, v) :: rho.

(** Synchronisation events emitted during evaluation *)
Inductive event : Type :=
  | EvBarrier : event   (* barrier synchronisation point *)
  | EvWarp    : event.  (* warp-collective call site *)

(** Trace: sequence of events emitted by a thread *)
Definition trace := list event.

(** Evaluation outcome:
    ONorm v — normal completion with value v
    ORet  v — early return with value v (EReturn constructor) *)
Inductive outcome : Type :=
  | ONorm : value -> outcome
  | ORet  : value -> outcome.

(* ===== 2. Helper: eval_seq and eval_args (hoisted for Fixpoint termination) ===== *)

(* These are defined inside the eval Fixpoint below; we use local fix notation. *)

(* ===== 3. Evaluator — Section with vary_val as parameter ===== *)

Section Evaluator.

(** vary_val maps each tid to the value of EVary (thread-private varying value).
    This is a Section Variable so all theorems in this section quantify over it
    when the Section closes, preserving the 0-axiom invariant. *)
Variable vary_val : tid -> value.

(** eval fuel t rho e
    Big-step evaluator, fuel-indexed for totality.
    Returns None when fuel is exhausted (nontermination).
    Returns Some (outcome, trace) on termination. *)

Fixpoint eval (fuel : nat) (t : tid) (rho : venv) (e : expr)
    : option (outcome * trace) :=
  match fuel with
  | O => None
  | S fuel' =>
    match e with
    | ELit       => Some (ONorm 0, [])
    | EVary      => Some (ONorm (vary_val t), [])
    | EBarrier   => Some (ONorm 0, [EvBarrier])
    | EWarpPoint => Some (ONorm 0, [EvWarp])
    | EVar x     => Some (ONorm (venv_lookup rho x), [])
    | EUnop e1   =>
        match eval fuel' t rho e1 with
        | Some (ORet v, tr) => Some (ORet v, tr)
        | Some (ONorm v, tr) => Some (ONorm v, tr)
        | None              => None
        end
    | EBinop e1 e2 =>
        match eval fuel' t rho e1 with
        | Some (ORet v, tr)    => Some (ORet v, tr)
        | Some (ONorm v1, tr1) =>
            match eval fuel' t rho e2 with
            | Some (ONorm v2, tr2) => Some (ONorm (v1 + v2), tr1 ++ tr2)
            | Some (ORet v2, tr2)  => Some (ORet v2, tr1 ++ tr2)
            | None                 => None
            end
        | None => None
        end
    | EIf cond e_then e_else =>
        match eval fuel' t rho cond with
        | Some (ORet v, tr)     => Some (ORet v, tr)
        | Some (ONorm cv, tr_c) =>
            let branch := if Nat.eqb cv 0 then e_else else e_then in
            match eval fuel' t rho branch with
            | Some (o, tr_b) => Some (o, tr_c ++ tr_b)
            | None           => None
            end
        | None => None
        end
    | EWhile cond body =>
        match eval fuel' t rho cond with
        | Some (ORet v, tr)     => Some (ORet v, tr)
        | Some (ONorm cv, tr_c) =>
            if Nat.eqb cv 0
            then Some (ONorm 0, tr_c)
            else
              match eval fuel' t rho body with
              | Some (ORet v, tr_b)  => Some (ORet v, tr_c ++ tr_b)
              | Some (ONorm _, tr_b) =>
                  match eval fuel' t rho (EWhile cond body) with
                  | Some (o, tr_loop) => Some (o, tr_c ++ tr_b ++ tr_loop)
                  | None              => None
                  end
              | None => None
              end
        | None => None
        end
    | EFor lo hi body =>
        match eval fuel' t rho lo with
        | Some (ORet v, tr)       => Some (ORet v, tr)
        | Some (ONorm lo_v, tr_lo) =>
            match eval fuel' t rho hi with
            | Some (ORet v, tr)         => Some (ORet v, tr_lo ++ tr)
            | Some (ONorm hi_v, tr_hi) =>
                if Nat.leb hi_v lo_v
                then Some (ONorm 0, tr_lo ++ tr_hi)
                else
                  let steps := hi_v - lo_v in
                  (fix loop (k : nat) (acc_tr : trace) : option (outcome * trace) :=
                    match k with
                    | O    => Some (ONorm 0, acc_tr)
                    | S k' =>
                        match eval fuel' t rho body with
                        | Some (ORet v, tr_b)  => Some (ORet v, acc_tr ++ tr_b)
                        | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                        | None                 => None
                        end
                    end) steps (tr_lo ++ tr_hi)
            | None => None
            end
        | None => None
        end
    | ESeq es =>
        (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
          match xs with
          | []      => Some (ONorm 0, acc_tr)
          | x :: rest =>
              match eval fuel' t rho x with
              | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
              | None               => None
              end
          end) es []
    | ELet x e_val body =>
        match eval fuel' t rho e_val with
        | Some (ORet v, tr)    => Some (ORet v, tr)
        | Some (ONorm v, tr_v) =>
            let rho' := venv_extend rho x v in
            match eval fuel' t rho' body with
            | Some (o, tr_b) => Some (o, tr_v ++ tr_b)
            | None           => None
            end
        | None => None
        end
    | ESuperstep _dv body cont =>
        (* Implicit end-of-superstep barrier emitted for both dv=true and dv=false.
           The runtime barrier always exists; dv affects only static checking. *)
        match eval fuel' t rho body with
        | Some (ORet v, tr_b)  => Some (ORet v, tr_b ++ [EvBarrier])
        | Some (ONorm _, tr_b) =>
            match eval fuel' t rho cont with
            | Some (o, tr_c) => Some (o, tr_b ++ [EvBarrier] ++ tr_c)
            | None           => None
            end
        | None => None
        end
    | EApp args =>
        (fix eval_args (xs : list expr) (acc_tr : trace) (last_v : value)
            : option (outcome * trace) :=
          match xs with
          | []      => Some (ONorm last_v, acc_tr)
          | x :: rest =>
              match eval fuel' t rho x with
              | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
              | None               => None
              end
          end) args [] 0
    | EReturn e_inner =>
        match eval fuel' t rho e_inner with
        | Some (ONorm v, tr) => Some (ORet v, tr)
        | other              => other
        end
    end
  end.

(* ===== 4. Auxiliary: accumulator-shift lemmas for eval_seq ===== *)

(** eval_seq_none_iff: eval_seq xs acc returns None iff eval_seq xs acc' returns None
    (None is acc-independent since it comes only from eval returning None). *)

Lemma eval_seq_none_iff :
  forall fuel' t rho xs acc1 acc2,
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None
          end
      end) xs acc1 = None <->
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None
          end
      end) xs acc2 = None.
Proof.
  intros fuel' t rho.
  induction xs as [| x xs' IHxs].
  - intros acc1 acc2. simpl. split; intro H; discriminate.
  - intros acc1 acc2. simpl.
    destruct (eval fuel' t rho x) as [[[xv|xv] xtr]|] eqn:Hx.
    + apply IHxs.
    + split; intro H; discriminate.
    + split; intro H; exact H.
Qed.

(** AUXILIARY: accumulator-shift lemmas (written out explicitly to avoid
    internal-fix opacity issues). Branch ordering notes:
    - destruct (eval...) as [[[xv|xv] xtr]|]:
        first  + : ONorm xv (first constructor of outcome)
        second + : ORet  xv (second constructor of outcome)
        third  + : None
    - eval_seq: ONorm hits recursive arm; ORet hits terminal arm.
    - eval_args: ONorm hits recursive arm; ORet hits terminal arm. *)

(** eval_seq_shift_nil: eval_seq xs [] = Some (o, tr) implies
    eval_seq xs pref = Some (o, pref ++ tr) for any prefix pref.
    Proved by induction on xs using eval_seq_none_iff for the contradiction case. *)

Lemma eval_seq_shift_nil :
  forall fuel' t rho xs pref o tr,
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None
          end
      end) xs [] = Some (o, tr) ->
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None
          end
      end) xs pref = Some (o, pref ++ tr).
Proof.
  intros fuel' t rho.
  (* We prove the stronger general version by induction on xs, then specialize.
     General: eval_seq xs base = Some (o, tr) →
              eval_seq xs (extra ++ base) = Some (o, extra ++ tr). *)
  assert (Hgen :
    forall xs base extra o tr,
      (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
        match xs0 with
        | []      => Some (ONorm 0, acc_tr)
        | x :: rest =>
            match eval fuel' t rho x with
            | Some (ORet v, tr0) => Some (ORet v, acc_tr ++ tr0)
            | Some (ONorm _, tr0) => eval_seq rest (acc_tr ++ tr0)
            | None               => None
            end
        end) xs base = Some (o, tr) ->
      (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
        match xs0 with
        | []      => Some (ONorm 0, acc_tr)
        | x :: rest =>
            match eval fuel' t rho x with
            | Some (ORet v, tr0) => Some (ORet v, acc_tr ++ tr0)
            | Some (ONorm _, tr0) => eval_seq rest (acc_tr ++ tr0)
            | None               => None
            end
        end) xs (extra ++ base) = Some (o, extra ++ tr)).
  { induction xs as [| x xs' IHxs].
    - intros base extra o tr H. simpl in H. injection H as Ho Htr. subst.
      simpl. reflexivity.
    - intros base extra o tr H. simpl in H. simpl.
      destruct (eval fuel' t rho x) as [[[xv|xv] xtr]|] eqn:Hx.
      + (* ONorm xv (first constructor): H = eval_seq xs' (base ++ xtr) = Some (o, tr) *)
        rewrite <- app_assoc. apply IHxs. exact H.
      + (* ORet xv (second constructor): H = Some (ORet xv, base ++ xtr) = Some (o, tr) *)
        injection H as Ho Htr. subst.
        f_equal. f_equal. rewrite app_assoc. reflexivity.
      + discriminate. }
  intros xs pref o tr H.
  pose proof (Hgen xs [] pref o tr) as Hshift.
  simpl in Hshift. rewrite app_nil_r in Hshift.
  apply Hshift. exact H.
Qed.

(** eval_seq_none_shift: eval_seq xs acc1 = None implies eval_seq xs acc2 = None.
    None is acc-independent (comes from eval returning None). *)

Lemma eval_seq_none_shift :
  forall fuel' t rho xs acc1 acc2,
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None
          end
      end) xs acc1 = None ->
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None
          end
      end) xs acc2 = None.
Proof.
  intros fuel' t rho.
  induction xs as [| x xs' IHxs].
  - intros acc1 acc2 H. simpl in H. discriminate.
  - intros acc1 acc2 H. simpl in H. simpl.
    destruct (eval fuel' t rho x) as [[[xv|xv] xtr]|] eqn:Hx.
    + (* ONorm xv: H = eval_seq xs' (acc1 ++ xtr) = None; recurse *)
      apply IHxs with (acc1 := acc1 ++ xtr). exact H.
    + (* ORet xv: H = Some (ORet xv, acc1 ++ xtr) = None; contradiction *)
      discriminate.
    + (* None: both return None *)
      reflexivity.
Qed.

(** eval_args_shift_nil and eval_args_none_shift: analogous for eval_args. *)

Lemma eval_args_shift_nil :
  forall fuel' t rho xs pref lv o tr,
    (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
        : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm last_v, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
          | None               => None
          end
      end) xs [] lv = Some (o, tr) ->
    (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
        : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm last_v, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
          | None               => None
          end
      end) xs pref lv = Some (o, pref ++ tr).
Proof.
  intros fuel' t rho.
  assert (Hgen :
    forall xs base extra lv o tr,
      (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
          : option (outcome * trace) :=
        match xs0 with
        | []      => Some (ONorm last_v, acc_tr)
        | x :: rest =>
            match eval fuel' t rho x with
            | Some (ORet v, tr0) => Some (ORet v, acc_tr ++ tr0)
            | Some (ONorm v, tr0) => eval_args rest (acc_tr ++ tr0) v
            | None               => None
            end
        end) xs base lv = Some (o, tr) ->
      (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
          : option (outcome * trace) :=
        match xs0 with
        | []      => Some (ONorm last_v, acc_tr)
        | x :: rest =>
            match eval fuel' t rho x with
            | Some (ORet v, tr0) => Some (ORet v, acc_tr ++ tr0)
            | Some (ONorm v, tr0) => eval_args rest (acc_tr ++ tr0) v
            | None               => None
            end
        end) xs (extra ++ base) lv = Some (o, extra ++ tr)).
  { induction xs as [| x xs' IHxs].
    - intros base extra lv o tr H. simpl in H. injection H as Ho Htr. subst.
      simpl. reflexivity.
    - intros base extra lv o tr H. simpl in H. simpl.
      destruct (eval fuel' t rho x) as [[[xv|xv] xtr]|] eqn:Hx.
      + (* ONorm xv: recursive *)
        rewrite <- app_assoc. apply IHxs. exact H.
      + (* ORet xv: terminal *)
        injection H as Ho Htr. subst.
        f_equal. f_equal. rewrite app_assoc. reflexivity.
      + discriminate. }
  intros xs pref lv o tr H.
  pose proof (Hgen xs [] pref lv o tr) as Hshift.
  simpl in Hshift. rewrite app_nil_r in Hshift.
  apply Hshift. exact H.
Qed.

Lemma eval_args_none_shift :
  forall fuel' t rho xs acc1 acc2 lv,
    (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
        : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm last_v, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
          | None               => None
          end
      end) xs acc1 lv = None ->
    (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
        : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm last_v, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
          | None               => None
          end
      end) xs acc2 lv = None.
Proof.
  intros fuel' t rho.
  induction xs as [| x xs' IHxs].
  - intros acc1 acc2 lv H. simpl in H. discriminate.
  - intros acc1 acc2 lv H. simpl in H. simpl.
    destruct (eval fuel' t rho x) as [[[xv|xv] xtr]|] eqn:Hx.
    + (* ONorm xv: recursive *)
      apply IHxs with (acc1 := acc1 ++ xtr). exact H.
    + (* ORet xv: H = Some (...) = None: contradiction *)
      discriminate.
    + (* None *)
      reflexivity.
Qed.

(** for_loop_fixed: the EFor inner loop parameterised by a single body result.
    Because `eval` is a pure function (same inputs → same outputs), every iteration
    calls `eval n t rho body` and obtains the same result; this helper captures that
    fixed result as an explicit parameter, making the monotonicity proof tractable. *)
Fixpoint for_loop_fixed
    (body : option (outcome * trace)) (k : nat) (acc : trace)
    : option (outcome * trace) :=
  match k with
  | O    => Some (ONorm 0, acc)
  | S k' =>
      match body with
      | Some (ORet v, tr_b)  => Some (ORet v, acc ++ tr_b)
      | Some (ONorm _, tr_b) => for_loop_fixed body k' (acc ++ tr_b)
      | None                 => None end
  end.

(** The EFor loop as written in `eval` (using `eval n t rho e3` inline) is
    definitionally equal to `for_loop_fixed (eval n t rho e3) k acc`. *)
Lemma for_loop_eq :
  forall (n : nat) (t : tid) (rho : venv) (e3 : expr) (k : nat) (acc : trace),
    (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
      match k0 with
      | O => Some (ONorm 0, acc_tr)
      | S k' =>
          match eval n t rho e3 with
          | Some (ORet v, tr_b)  => Some (ORet v, acc_tr ++ tr_b)
          | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
          | None                 => None end
      end) k acc = for_loop_fixed (eval n t rho e3) k acc.
Proof.
  intros n t rho e3.
  induction k as [| k' IHk].
  - intros acc. reflexivity.
  - intros acc.
    destruct (eval n t rho e3) as [[[bv|bv] trb]|] eqn:Hbody.
    + simpl. rewrite IHk. reflexivity.
    + simpl. reflexivity.
    + simpl. reflexivity.
Qed.

(** for_loop_fixed_mono: if body_n implies body_Sn, the fixed loop is monotone. *)
Lemma for_loop_fixed_mono :
  forall (body_n body_Sn : option (outcome * trace))
    (Hmono : forall r, body_n = Some r -> body_Sn = Some r)
    (k : nat) (acc : trace) (r : outcome * trace),
    for_loop_fixed body_n k acc = Some r ->
    for_loop_fixed body_Sn k acc = Some r.
Proof.
  intros body_n body_Sn Hmono.
  induction k as [| k' IHk].
  - intros acc r H. exact H.
  - intros acc r H.
    simpl in H.
    destruct body_n as [[[bv|bv] trb]|] eqn:Hbody_n.
    + (* body_n = Some (ONorm bv, trb) — Hmono now has body_n substituted *)
      (* HSn : body_Sn = Some (ONorm bv, trb);
         H   : for_loop_fixed (Some (ONorm bv, trb)) k' (acc ++ trb) = Some r *)
      pose proof (Hmono _ eq_refl) as HSn.
      simpl. rewrite HSn. simpl. exact H.
    + (* body_n = Some (ORet bv, trb) *)
      pose proof (Hmono _ eq_refl) as HSn.
      simpl. rewrite HSn. simpl. exact H.
    + (* body_n = None *)
      discriminate.
Qed.

(* ===== 5. Theorem: eval_fuel_monotone ===== *)

(** for_loop_none_propagates: if the loop body always returns None (because
    eval returned None), then the loop with k >= 1 iterations also returns None.
    Used in the EFor case of eval_fuel_monotone. *)
Lemma for_loop_none_propagates :
  forall k acc,
    k >= 1 ->
    (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
      match k0 with
      | 0    => Some (ONorm 0, acc_tr)
      | S _  => None
      end) k acc = None.
Proof.
  intros k acc Hk. destruct k. lia. simpl. reflexivity.
Qed.

(** for_loop_mono: if the body evaluator is monotone in fuel (eval n → eval S n),
    then the for-loop accumulator is also monotone.
    Proof: rewrite both loops as `for_loop_fixed (eval ? t rho e3)` using
    `for_loop_eq`, then apply `for_loop_fixed_mono`. *)
Lemma for_loop_mono :
  forall (n : nat) (t : tid) (rho : venv) (e3 : expr)
    (step_mono : forall (r : outcome * trace),
                 eval n t rho e3 = Some r -> eval (S n) t rho e3 = Some r)
    (k : nat) (acc : trace) (r : outcome * trace),
    (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
      match k0 with
      | O => Some (ONorm 0, acc_tr)
      | S k' =>
          match eval n t rho e3 with
          | Some (ORet v, tr_b)  => Some (ORet v, acc_tr ++ tr_b)
          | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
          | None                 => None end
      end) k acc = Some r ->
    (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
      match k0 with
      | O => Some (ONorm 0, acc_tr)
      | S k' =>
          match eval (S n) t rho e3 with
          | Some (ORet v, tr_b)  => Some (ORet v, acc_tr ++ tr_b)
          | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
          | None                 => None end
      end) k acc = Some r.
Proof.
  intros n t rho e3 step_mono k acc r H.
  rewrite for_loop_eq in H.
  rewrite for_loop_eq.
  exact (for_loop_fixed_mono (eval n t rho e3) (eval (S n) t rho e3)
           step_mono k acc r H).
Qed.

(** eval_seq_mono: eval_seq with fuel n lifted to fuel S n under step_mono.
    Uses `change` to unfold the goal one step without triggering `simpl`'s
    expansion of `eval (S n) t rho x` (which opens a case split on `x`). *)
Lemma eval_seq_mono :
  forall (n : nat) (t : tid) (rho : venv)
    (step_mono : forall (e : expr) (r : outcome * trace),
                 eval n t rho e = Some r -> eval (S n) t rho e = Some r)
    (xs : list expr) (acc : trace) (r : outcome * trace),
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval n t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None end
      end) xs acc = Some r ->
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval (S n) t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None end
      end) xs acc = Some r.
Proof.
  intros n t rho step_mono.
  induction xs as [| x xs' IHxs].
  - intros acc r H. exact H.
  - intros acc r H.
    destruct (eval n t rho x) as [[[xv|xv] xtr]|] eqn:Hx; simpl in H; try discriminate.
    + pose proof (step_mono x (ONorm xv, xtr) Hx) as IHx.
      change (match eval (S n) t rho x with
        | Some (ORet v, tr)  => Some (ORet v, acc ++ tr)
        | Some (ONorm _, tr) =>
            (fix eval_seq xs0 acc_tr :=
              match xs0 with
              | []      => Some (ONorm 0, acc_tr)
              | x0 :: rest =>
                  match eval (S n) t rho x0 with
                  | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                  | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
                  | None               => None end
              end) xs' (acc ++ tr)
        | None => None end = Some r).
      rewrite IHx. apply IHxs. exact H.
    + pose proof (step_mono x (ORet xv, xtr) Hx) as IHx.
      change (match eval (S n) t rho x with
        | Some (ORet v, tr)  => Some (ORet v, acc ++ tr)
        | Some (ONorm _, tr) =>
            (fix eval_seq xs0 acc_tr :=
              match xs0 with
              | []      => Some (ONorm 0, acc_tr)
              | x0 :: rest =>
                  match eval (S n) t rho x0 with
                  | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                  | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
                  | None               => None end
              end) xs' (acc ++ tr)
        | None => None end = Some r).
      rewrite IHx. exact H.
Qed.

(** eval_args_mono: eval_args with fuel n lifted to fuel S n under step_mono.
    Same technique as eval_seq_mono. *)
Lemma eval_args_mono :
  forall (n : nat) (t : tid) (rho : venv)
    (step_mono : forall (e : expr) (r : outcome * trace),
                 eval n t rho e = Some r -> eval (S n) t rho e = Some r)
    (xs : list expr) (acc : trace) (lv : value) (r : outcome * trace),
    (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
        : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm last_v, acc_tr)
      | x :: rest =>
          match eval n t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
          | None               => None end
      end) xs acc lv = Some r ->
    (fix eval_args (xs0 : list expr) (acc_tr : trace) (last_v : value)
        : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm last_v, acc_tr)
      | x :: rest =>
          match eval (S n) t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
          | None               => None end
      end) xs acc lv = Some r.
Proof.
  intros n t rho step_mono.
  induction xs as [| x xs' IHxs].
  - intros acc lv r H. exact H.
  - intros acc lv r H.
    destruct (eval n t rho x) as [[[xv|xv] xtr]|] eqn:Hx; simpl in H; try discriminate.
    + pose proof (step_mono x (ONorm xv, xtr) Hx) as IHx.
      change (match eval (S n) t rho x with
        | Some (ORet v, tr)  => Some (ORet v, acc ++ tr)
        | Some (ONorm v, tr) =>
            (fix eval_args xs0 acc_tr last_v :=
              match xs0 with
              | []      => Some (ONorm last_v, acc_tr)
              | x0 :: rest =>
                  match eval (S n) t rho x0 with
                  | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                  | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
                  | None               => None end
              end) xs' (acc ++ tr) v
        | None => None end = Some r).
      rewrite IHx. apply IHxs. exact H.
    + pose proof (step_mono x (ORet xv, xtr) Hx) as IHx.
      change (match eval (S n) t rho x with
        | Some (ORet v, tr)  => Some (ORet v, acc ++ tr)
        | Some (ONorm v, tr) =>
            (fix eval_args xs0 acc_tr last_v :=
              match xs0 with
              | []      => Some (ONorm last_v, acc_tr)
              | x0 :: rest =>
                  match eval (S n) t rho x0 with
                  | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                  | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
                  | None               => None end
              end) xs' (acc ++ tr) v
        | None => None end = Some r).
      rewrite IHx. exact H.
Qed.

(** eval_fuel_monotone: if eval succeeds at fuel n, it succeeds with the same
    result at fuel S n. This is the standard sanity lemma for fuel-indexed
    evaluators; all later theorems rely on it. *)

Theorem eval_fuel_monotone :
  forall n t rho e r,
    eval n t rho e = Some r ->
    eval (S n) t rho e = Some r.
Proof.
  (* Strategy: induction on fuel n.
     IH : forall t rho e r, eval n' t rho e = Some r -> eval (S n') t rho e = Some r.
     For each constructor, simpl in H unfolds eval (S n') one step (exposes eval n').
     For the goal eval (S (S n')) we use:
       assert (Hstep : eval (S (S n')) ... = <match eval (S n') ...>) by reflexivity
     which is closed by definitional equality. Then rewrite IH results into the goal.
     This avoids simpl on the goal (which would case-split on abstract sub-expressions).
     EWhile's recursive self-call is handled by IH applied to (EWhile cond body).
     EFor's inner loop is handled by induction on k once eval n' body is fixed.
     ESeq/EApp inner loops are handled by induction on the list using IH for each element. *)
  induction n as [| n' IH].
  - intros t rho e r H. simpl in H. discriminate.
  - intros t rho e r H.
    (* After destruct e, the constructor argument names (from ConvergenceSpec.v):
       EVar n; EBinop e1 e2; EUnop e; EIf e1 e2 e3; EWhile e1 e2; EFor e1 e2 e3;
       ESeq l; ELet n e1 e2; ESuperstep b e1 e2; EApp l; EReturn e *)
    destruct e.
    (* 1: ELit *) + exact H.
    (* 2: EVary *) + exact H.
    (* 3: EBarrier *) + exact H.
    (* 4: EWarpPoint *) + exact H.
    (* 5: EVar n *) + exact H.
    (* 6: EBinop e1 e2 *)
    + simpl in H.
      destruct (eval n' t rho e1) as [[[v1|v1] tr1]|] eqn:He1; try discriminate.
      * pose proof (IH t rho e1 (ONorm v1, tr1) He1) as IH1.
        destruct (eval n' t rho e2) as [[[v2|v2] tr2]|] eqn:He2; try discriminate.
        -- pose proof (IH t rho e2 (ONorm v2, tr2) He2) as IH2.
           assert (Hstep : eval (S (S n')) t rho (EBinop e1 e2) =
             match eval (S n') t rho e1 with
             | Some (ORet v, tr) => Some (ORet v, tr)
             | Some (ONorm v1', tr1') =>
                 match eval (S n') t rho e2 with
                 | Some (ONorm v2', tr2') => Some (ONorm (v1' + v2'), tr1' ++ tr2')
                 | Some (ORet v2', tr2') => Some (ORet v2', tr1' ++ tr2')
                 | None => None end
             | None => None end) by reflexivity.
           rewrite Hstep. rewrite IH1. rewrite IH2. exact H.
        -- pose proof (IH t rho e2 (ORet v2, tr2) He2) as IH2.
           assert (Hstep : eval (S (S n')) t rho (EBinop e1 e2) =
             match eval (S n') t rho e1 with
             | Some (ORet v, tr) => Some (ORet v, tr)
             | Some (ONorm v1', tr1') =>
                 match eval (S n') t rho e2 with
                 | Some (ONorm v2', tr2') => Some (ONorm (v1' + v2'), tr1' ++ tr2')
                 | Some (ORet v2', tr2') => Some (ORet v2', tr1' ++ tr2')
                 | None => None end
             | None => None end) by reflexivity.
           rewrite Hstep. rewrite IH1. rewrite IH2. exact H.
      * pose proof (IH t rho e1 (ORet v1, tr1) He1) as IH1.
        assert (Hstep : eval (S (S n')) t rho (EBinop e1 e2) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm v1', tr1') =>
              match eval (S n') t rho e2 with
              | Some (ONorm v2', tr2') => Some (ONorm (v1' + v2'), tr1' ++ tr2')
              | Some (ORet v2', tr2') => Some (ORet v2', tr1' ++ tr2')
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IH1. exact H.
    (* 7: EUnop e (subexpr is named 'e' after destruct) *)
    + simpl in H.
      destruct (eval n' t rho e) as [[[v|v] tr]|] eqn:He1; try discriminate.
      * pose proof (IH t rho e (ONorm v, tr) He1) as IH1.
        assert (Hstep : eval (S (S n')) t rho (EUnop e) =
          match eval (S n') t rho e with
          | Some (ORet v', tr') => Some (ORet v', tr')
          | Some (ONorm v', tr') => Some (ONorm v', tr')
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IH1. exact H.
      * pose proof (IH t rho e (ORet v, tr) He1) as IH1.
        assert (Hstep : eval (S (S n')) t rho (EUnop e) =
          match eval (S n') t rho e with
          | Some (ORet v', tr') => Some (ORet v', tr')
          | Some (ONorm v', tr') => Some (ONorm v', tr')
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IH1. exact H.
    (* 8: EIf e1 e2 e3 (cond, then, else) *)
    + simpl in H.
      destruct (eval n' t rho e1) as [[[cv|cv] trc]|] eqn:Hcond; try discriminate.
      * pose proof (IH t rho e1 (ONorm cv, trc) Hcond) as IHc.
        destruct (eval n' t rho (if Nat.eqb cv 0 then e3 else e2))
                 as [[o trb]|] eqn:Hbr; try discriminate.
        pose proof (IH t rho (if Nat.eqb cv 0 then e3 else e2) (o, trb) Hbr) as IHb.
        assert (Hstep : eval (S (S n')) t rho (EIf e1 e2 e3) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm cv', trc') =>
              match eval (S n') t rho (if Nat.eqb cv' 0 then e3 else e2) with
              | Some (o', trb') => Some (o', trc' ++ trb')
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHc. rewrite IHb. exact H.
      * pose proof (IH t rho e1 (ORet cv, trc) Hcond) as IHc.
        assert (Hstep : eval (S (S n')) t rho (EIf e1 e2 e3) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm cv', trc') =>
              match eval (S n') t rho (if Nat.eqb cv' 0 then e3 else e2) with
              | Some (o', trb') => Some (o', trc' ++ trb')
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHc. exact H.
    (* 9: EWhile e1 e2 (cond, body) *)
    + simpl in H.
      destruct (eval n' t rho e1) as [[[cv|cv] trc]|] eqn:Hcond; try discriminate.
      * pose proof (IH t rho e1 (ONorm cv, trc) Hcond) as IHc.
        assert (Hstep : eval (S (S n')) t rho (EWhile e1 e2) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm cv', trc') =>
              if Nat.eqb cv' 0 then Some (ONorm 0, trc')
              else
                match eval (S n') t rho e2 with
                | Some (ORet v, trb) => Some (ORet v, trc' ++ trb)
                | Some (ONorm _, trb) =>
                    match eval (S n') t rho (EWhile e1 e2) with
                    | Some (o, trl) => Some (o, trc' ++ trb ++ trl)
                    | None => None end
                | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHc.
        destruct (Nat.eqb cv 0); [exact H |].
        destruct (eval n' t rho e2) as [[[bv|bv] trb]|] eqn:Hbody; try discriminate.
        -- pose proof (IH t rho e2 (ONorm bv, trb) Hbody) as IHb.
           rewrite IHb.
           destruct (eval n' t rho (EWhile e1 e2)) as [[ow trl]|] eqn:Hloop;
             try discriminate.
           pose proof (IH t rho (EWhile e1 e2) (ow, trl) Hloop) as IHloop.
           rewrite IHloop. exact H.
        -- pose proof (IH t rho e2 (ORet bv, trb) Hbody) as IHb.
           rewrite IHb. exact H.
      * pose proof (IH t rho e1 (ORet cv, trc) Hcond) as IHc.
        assert (Hstep : eval (S (S n')) t rho (EWhile e1 e2) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm cv', trc') =>
              if Nat.eqb cv' 0 then Some (ONorm 0, trc')
              else
                match eval (S n') t rho e2 with
                | Some (ORet v, trb) => Some (ORet v, trc' ++ trb)
                | Some (ONorm _, trb) =>
                    match eval (S n') t rho (EWhile e1 e2) with
                    | Some (o, trl) => Some (o, trc' ++ trb ++ trl)
                    | None => None end
                | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHc. exact H.
    (* 10: EFor e1 e2 e3 (lo, hi, body) *)
    + simpl in H.
      destruct (eval n' t rho e1) as [[[lv|lv] trlo]|] eqn:Hlo; try discriminate.
      * pose proof (IH t rho e1 (ONorm lv, trlo) Hlo) as IHl.
        destruct (eval n' t rho e2) as [[[hv|hv] trhi]|] eqn:Hhi; try discriminate.
        -- pose proof (IH t rho e2 (ONorm hv, trhi) Hhi) as IHh.
           assert (Hstep : eval (S (S n')) t rho (EFor e1 e2 e3) =
             match eval (S n') t rho e1 with
             | Some (ORet v, tr) => Some (ORet v, tr)
             | Some (ONorm lo_v, tr_lo) =>
                 match eval (S n') t rho e2 with
                 | Some (ORet v, tr) => Some (ORet v, tr_lo ++ tr)
                 | Some (ONorm hi_v, tr_hi) =>
                     if Nat.leb hi_v lo_v
                     then Some (ONorm 0, tr_lo ++ tr_hi)
                     else
                       (fix loop (k : nat) (acc_tr : trace) : option (outcome * trace) :=
                         match k with
                         | O => Some (ONorm 0, acc_tr)
                         | S k' =>
                             match eval (S n') t rho e3 with
                             | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                             | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                             | None => None end
                         end) (hi_v - lo_v) (tr_lo ++ tr_hi)
                 | None => None end
             | None => None end) by reflexivity.
           rewrite Hstep. rewrite IHl. rewrite IHh. simpl.
           destruct (Nat.leb hv lv); [exact H |]. simpl in H.
           apply (for_loop_mono n' t rho e3
             (fun r' Hr' => IH t rho e3 r' Hr')
             (hv - lv) (trlo ++ trhi)).
           exact H.
        -- pose proof (IH t rho e2 (ORet hv, trhi) Hhi) as IHh.
           assert (Hstep : eval (S (S n')) t rho (EFor e1 e2 e3) =
             match eval (S n') t rho e1 with
             | Some (ORet v, tr) => Some (ORet v, tr)
             | Some (ONorm lo_v, tr_lo) =>
                 match eval (S n') t rho e2 with
                 | Some (ORet v, tr) => Some (ORet v, tr_lo ++ tr)
                 | Some (ONorm hi_v, tr_hi) =>
                     if Nat.leb hi_v lo_v then Some (ONorm 0, tr_lo ++ tr_hi)
                     else
                       (fix loop (k : nat) (acc_tr : trace) : option (outcome * trace) :=
                         match k with
                         | O => Some (ONorm 0, acc_tr)
                         | S k' =>
                             match eval (S n') t rho e3 with
                             | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                             | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                             | None => None end
                         end) (hi_v - lo_v) (tr_lo ++ tr_hi)
                 | None => None end
             | None => None end) by reflexivity.
           rewrite Hstep. rewrite IHl. rewrite IHh. exact H.
      * pose proof (IH t rho e1 (ORet lv, trlo) Hlo) as IHl.
        assert (Hstep : eval (S (S n')) t rho (EFor e1 e2 e3) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm lo_v, tr_lo) =>
              match eval (S n') t rho e2 with
              | Some (ORet v, tr) => Some (ORet v, tr_lo ++ tr)
              | Some (ONorm hi_v, tr_hi) =>
                  if Nat.leb hi_v lo_v then Some (ONorm 0, tr_lo ++ tr_hi)
                  else
                    (fix loop (k : nat) (acc_tr : trace) : option (outcome * trace) :=
                      match k with
                      | O => Some (ONorm 0, acc_tr)
                      | S k' =>
                          match eval (S n') t rho e3 with
                          | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                          | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                          | None => None end
                      end) (hi_v - lo_v) (tr_lo ++ tr_hi)
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHl. exact H.
    (* 11: ESeq l *)
    + simpl in H.
      assert (Hstep : eval (S (S n')) t rho (ESeq l) =
        (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
          match xs with
          | [] => Some (ONorm 0, acc_tr)
          | x :: rest =>
              match eval (S n') t rho x with
              | Some (ORet v, tr) => Some (ORet v, acc_tr ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
              | None => None end
          end) l []) by reflexivity.
      rewrite Hstep.
      apply (eval_seq_mono n' t rho (fun e r0 He0 => IH t rho e r0 He0) l [] r).
      exact H.
    (* 12: ELet n e1 e2 *)
    + simpl in H.
      destruct (eval n' t rho e1) as [[[ev|ev] trev]|] eqn:Hev; try discriminate.
      * pose proof (IH t rho e1 (ONorm ev, trev) Hev) as IHe.
        (* Use rho_ext := venv_extend rho n ev so the goal doesn't have a let *)
        (* Introduce rho_ext for the extended environment to keep goal/H in sync *)
        set (rho_ext := venv_extend rho n ev) in H |- *.
        assert (Hstep : eval (S (S n')) t rho (ELet n e1 e2) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm v, tr_v) =>
              match eval (S n') t (venv_extend rho n v) e2 with
              | Some (o, tr_b) => Some (o, tr_v ++ tr_b)
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHe.
        destruct (eval n' t rho_ext e2) as [[ob trb]|] eqn:Hbody;
          try discriminate.
        pose proof (IH t rho_ext e2 (ob, trb) Hbody) as IHb.
        fold rho_ext. rewrite IHb. exact H.
      * pose proof (IH t rho e1 (ORet ev, trev) Hev) as IHe.
        assert (Hstep : eval (S (S n')) t rho (ELet n e1 e2) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr) => Some (ORet v, tr)
          | Some (ONorm v, tr_v) =>
              let rho' := venv_extend rho n v in
              match eval (S n') t rho' e2 with
              | Some (o, tr_b) => Some (o, tr_v ++ tr_b)
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHe. exact H.
    (* 13: ESuperstep b e1 e2 (dv_flag, body, cont) *)
    + simpl in H.
      destruct (eval n' t rho e1) as [[[bv|bv] trb]|] eqn:Hbody; try discriminate.
      * pose proof (IH t rho e1 (ONorm bv, trb) Hbody) as IHb.
        assert (Hstep : eval (S (S n')) t rho (ESuperstep b e1 e2) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr_b) => Some (ORet v, tr_b ++ [EvBarrier])
          | Some (ONorm _, tr_b) =>
              match eval (S n') t rho e2 with
              | Some (o, tr_c) => Some (o, tr_b ++ [EvBarrier] ++ tr_c)
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHb.
        destruct (eval n' t rho e2) as [[oc trc]|] eqn:Hcont; try discriminate.
        pose proof (IH t rho e2 (oc, trc) Hcont) as IHc. rewrite IHc. exact H.
      * pose proof (IH t rho e1 (ORet bv, trb) Hbody) as IHb.
        assert (Hstep : eval (S (S n')) t rho (ESuperstep b e1 e2) =
          match eval (S n') t rho e1 with
          | Some (ORet v, tr_b) => Some (ORet v, tr_b ++ [EvBarrier])
          | Some (ONorm _, tr_b) =>
              match eval (S n') t rho e2 with
              | Some (o, tr_c) => Some (o, tr_b ++ [EvBarrier] ++ tr_c)
              | None => None end
          | None => None end) by reflexivity.
        rewrite Hstep. rewrite IHb. exact H.
    (* 14: EApp l *)
    + simpl in H.
      assert (Hstep : eval (S (S n')) t rho (EApp l) =
        (fix eval_args (xs : list expr) (acc_tr : trace) (last_v : value)
            : option (outcome * trace) :=
          match xs with
          | [] => Some (ONorm last_v, acc_tr)
          | x :: rest =>
              match eval (S n') t rho x with
              | Some (ORet v, tr) => Some (ORet v, acc_tr ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
              | None => None end
          end) l [] 0) by reflexivity.
      rewrite Hstep.
      apply (eval_args_mono n' t rho (fun e r0 He0 => IH t rho e r0 He0) l [] 0 r).
      exact H.
    (* 15: EReturn e (subexpr is 'e' after destruct) *)
    + simpl in H.
      destruct (eval n' t rho e) as [[[iv|iv] tri]|] eqn:Hei.
      * pose proof (IH t rho e (ONorm iv, tri) Hei) as IHi.
        assert (Hstep : eval (S (S n')) t rho (EReturn e) =
          match eval (S n') t rho e with
          | Some (ONorm v, tr) => Some (ORet v, tr)
          | other => other end) by reflexivity.
        rewrite Hstep. rewrite IHi. exact H.
      * pose proof (IH t rho e (ORet iv, tri) Hei) as IHi.
        assert (Hstep : eval (S (S n')) t rho (EReturn e) =
          match eval (S n') t rho e with
          | Some (ONorm v, tr) => Some (ORet v, tr)
          | other => other end) by reflexivity.
        rewrite Hstep. rewrite IHi. exact H.
      * discriminate.
Qed.

(** eval_seq_concat_acc: the key inner lemma for eval_app_seq_compose.
    eval_seq es1 acc = Some (ONorm v1, tr1) together with eval_seq es2 [] = Some (o2, tr2)
    implies eval_seq (es1 ++ es2) acc = Some (o2, tr1 ++ tr2).
    Proved by induction on es1 with acc, v1, tr1 universally quantified,
    so the IH gives the right generality for the recursive case. *)
Lemma eval_seq_concat_acc :
  forall (fuel' : nat) (t : tid) (rho : venv) (es1 es2 : list expr)
    (o2 : outcome) (tr2 : trace),
    (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None end
      end) es2 [] = Some (o2, tr2) ->
    forall (v1 : value) (tr1 : trace) (acc : trace),
    (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None end
      end) es1 acc = Some (ONorm v1, tr1) ->
    (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval fuel' t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None end
      end) (es1 ++ es2) acc = Some (o2, tr1 ++ tr2).
Proof.
  intros fuel' t rho es1 es2 o2 tr2 H2.
  induction es1 as [| e es1' IHes1].
  - (* es1 = [] *)
    intros v1 tr1 acc H1.
    simpl in H1. injection H1 as Hv Htr. subst v1. rewrite <- Htr.
    simpl. exact (eval_seq_shift_nil fuel' t rho es2 acc o2 tr2 H2).
  - (* es1 = e :: es1' *)
    intros v1 tr1 acc H1.
    simpl in H1. simpl.
    destruct (eval fuel' t rho e) as [[[ev|ev] etr]|] eqn:He.
    + (* ONorm ev etr: H1 = eval_seq es1' (acc ++ etr) = Some (ONorm v1, tr1) *)
      apply IHes1 with (v1 := v1) (tr1 := tr1) (acc := acc ++ etr). exact H1.
    + (* ORet ev etr: H1 = Some (ORet ev, acc ++ etr) = Some (ONorm v1, tr1): contradiction *)
      injection H1 as Hco. discriminate.
    + discriminate.
Qed.

(* ===== 6. Theorem: eval_app_seq_compose ===== *)

(** eval_app_seq_compose: trace homomorphism over ESeq.
    Evaluating ESeq (es1 ++ es2) produces a trace that is the concatenation
    of the traces from ESeq es1 (when es1 produces ONorm) and ESeq es2.
    This is the semantic mirror of check_seq_hom.

    Precondition: es1 produces ONorm (not ORet) — if any element of es1 returns
    early, ESeq short-circuits and the composition does not apply. *)

Theorem eval_app_seq_compose :
  forall fuel t rho es1 es2 v1 tr1 o2 tr2,
    eval fuel t rho (ESeq es1) = Some (ONorm v1, tr1) ->
    eval fuel t rho (ESeq es2) = Some (o2, tr2) ->
    eval fuel t rho (ESeq (es1 ++ es2)) = Some (o2, tr1 ++ tr2).
Proof.
  intros fuel t rho es1 es2 v1 tr1 o2 tr2 H1 H2.
  destruct fuel as [| fuel'].
  - simpl in H1. discriminate.
  - simpl in H1. simpl in H2. simpl.
    (* H1 : eval_seq es1 [] = Some (ONorm v1, tr1) *)
    (* H2 : eval_seq es2 [] = Some (o2, tr2)       *)
    (* Goal : eval_seq (es1 ++ es2) [] = Some (o2, tr1 ++ tr2) *)
    exact (eval_seq_concat_acc fuel' t rho es1 es2 o2 tr2 H2 v1 tr1 [] H1).
Qed.

End Evaluator.
