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

From Stdlib Require Import Bool List Arith Lia.
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

(******************************************************************************)
(* T3-S2 — Uniformity soundness of is_strongly_uniform
 *
 * We define is_strongly_uniform, a semantically-correct uniformity predicate
 * stricter than is_varying_in_env: for ELet x v b it additionally requires v
 * to be uniform (correcting a soundness gap in is_varying_in_env).
 *
 * Main theorem (not_varying_uniform): is_var_free e = true, env_agrees g rho1 rho2,
 * is_strongly_uniform g e = true → eval fuel t1 rho1 e = eval fuel t2 rho2 e.
 ******************************************************************************)

(* ----------------------------------------------------------------------- *)
(* 7.1  venv helpers                                                        *)
(* ----------------------------------------------------------------------- *)

Lemma venv_lookup_extend_same :
  forall (rho : venv) (x : nat) (v : value),
    venv_lookup (venv_extend rho x v) x = v.
Proof.
  intros rho x v. unfold venv_lookup, venv_extend. simpl.
  rewrite Nat.eqb_refl. reflexivity.
Qed.

Lemma venv_lookup_extend_diff :
  forall (rho : venv) (x y : nat) (v : value),
    x <> y ->
    venv_lookup (venv_extend rho y v) x = venv_lookup rho x.
Proof.
  intros rho x y v Hne. unfold venv_lookup, venv_extend. simpl.
  destruct (Nat.eqb y x) eqn:Heq.
  - apply Nat.eqb_eq in Heq. symmetry in Heq. contradiction.
  - reflexivity.
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.2  Env helpers                                                         *)
(* ----------------------------------------------------------------------- *)

Lemma env_lookup_extend_same_key :
  forall (env : Env) (x : nat) (v : bool),
    env_lookup (env_extend env x v) x = v.
Proof.
  intros env x v. unfold env_lookup, env_extend. simpl.
  rewrite Nat.eqb_refl. reflexivity.
Qed.

Lemma env_lookup_extend_diff_key :
  forall (env : Env) (x y : nat) (v : bool),
    x <> y ->
    env_lookup (env_extend env y v) x = env_lookup env x.
Proof.
  intros env x y v Hne. unfold env_lookup, env_extend. simpl.
  destruct (Nat.eqb y x) eqn:Heq.
  - apply Nat.eqb_eq in Heq. symmetry in Heq. contradiction.
  - reflexivity.
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.3  env_agrees — uniform-variable agreement predicate                  *)
(* ----------------------------------------------------------------------- *)

(** env_agrees env rho1 rho2 = true means: for every explicitly tracked
    variable x in env with env_lookup env x = false (non-varying), rho1 and
    rho2 agree on x. Variables not tracked in env (defaulting to false via
    env_lookup) are only constrained when they appear in the env list.
    Note: is_var_free e = true is required as a precondition for
    not_varying_uniform to exclude the EVar case where the default
    env_lookup [] x = false would otherwise be unprovable. *)
Fixpoint env_agrees (env : Env) (rho1 rho2 : venv) : bool :=
  match env with
  | []           => true
  | (x, v) :: rest =>
      (v || Nat.eqb (venv_lookup rho1 x) (venv_lookup rho2 x)) &&
      env_agrees rest rho1 rho2
  end.

(** env_agrees_extend: env_agrees is preserved by extending both env and
    both rhos with the same variable x and the same value w. *)
Lemma env_agrees_extend :
  forall (env : Env) (rho1 rho2 : venv) (x : nat) (w : value) (flag : bool),
    env_agrees env rho1 rho2 = true ->
    env_agrees (env_extend env x flag) (venv_extend rho1 x w) (venv_extend rho2 x w) = true.
Proof.
  intros env rho1 rho2 x w flag Ha.
  unfold env_extend. simpl.
  rewrite venv_lookup_extend_same. rewrite venv_lookup_extend_same.
  rewrite Nat.eqb_refl. rewrite Bool.orb_true_r. simpl.
  (* Remaining goal: env_agrees env (venv_extend rho1 x w) (venv_extend rho2 x w) = true *)
  induction env as [| [y vy] rest IH].
  - simpl. reflexivity.
  - simpl in Ha. apply andb_true_iff in Ha as [Hh Ht].
    simpl. apply andb_true_iff. split.
    + destruct (Nat.eq_dec y x) as [Heqyx | Hneyx].
      * subst y.
        rewrite venv_lookup_extend_same. rewrite venv_lookup_extend_same.
        rewrite Nat.eqb_refl. rewrite Bool.orb_true_r. reflexivity.
      * rewrite (venv_lookup_extend_diff rho1 y x w Hneyx).
        rewrite (venv_lookup_extend_diff rho2 y x w Hneyx).
        exact Hh.
    + exact (IH Ht).
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.5  is_strongly_uniform — semantically correct uniformity predicate      *)
(* ----------------------------------------------------------------------- *)

(** is_strongly_uniform env e = true means:
    e's evaluation is independent of vary_val t AND of the values of
    variables that are varying in env.  Unlike is_varying_in_env, for
    ELet x v b this predicate also requires v to be uniform, ensuring
    the binder's evaluation cannot introduce a trace dependency on tid. *)
Fixpoint is_strongly_uniform (env : Env) (e : expr) : bool :=
  match e with
  | EVary              => false
  | ELit | EBarrier | EWarpPoint => true
  | EVar x             => negb (env_lookup env x)
  | EBinop a b         => is_strongly_uniform env a && is_strongly_uniform env b
  | EUnop e1           => is_strongly_uniform env e1
  | EIf c et ef        =>
      is_strongly_uniform env c &&
      is_strongly_uniform env et &&
      is_strongly_uniform env ef
  | EWhile c b         =>
      is_strongly_uniform env c &&
      is_strongly_uniform env b
  | EFor lo hi b       =>
      is_strongly_uniform env lo &&
      is_strongly_uniform env hi &&
      is_strongly_uniform env b
  | ESeq es            => forallb (is_strongly_uniform env) es
  | ELet x v b        =>
      let vv := negb (is_strongly_uniform env v) in
      is_strongly_uniform env v &&
      is_strongly_uniform (env_extend env x vv) b
  | ESuperstep _ body cont =>
      is_strongly_uniform env body && is_strongly_uniform env cont
  | EApp args          => forallb (is_strongly_uniform env) args
  | EReturn e1         => is_strongly_uniform env e1
  end.

(* ----------------------------------------------------------------------- *)
(* 7.6  is_var_free — structural EVar-free check (needed before theorem)    *)
(* ----------------------------------------------------------------------- *)

(** is_var_free e = true means e contains no EVar constructors *)
Fixpoint is_var_free (e : expr) : bool :=
  match e with
  | EVar _             => false
  | ELit | EVary | EBarrier | EWarpPoint => true
  | EBinop a b         => is_var_free a && is_var_free b
  | EUnop e1           => is_var_free e1
  | EIf c et ef        => is_var_free c && is_var_free et && is_var_free ef
  | EWhile c b         => is_var_free c && is_var_free b
  | EFor lo hi b       => is_var_free lo && is_var_free hi && is_var_free b
  | ESeq es            => forallb is_var_free es
  | ELet x v b         => is_var_free v && is_var_free b
  | ESuperstep _ body cont => is_var_free body && is_var_free cont
  | EApp args          => forallb is_var_free args
  | EReturn e1         => is_var_free e1
  end.

(* ----------------------------------------------------------------------- *)
(* 7.7  not_varying_uniform — main uniformity theorem                        *)
(* ----------------------------------------------------------------------- *)

(** Auxiliary: uniform lists eval identically for ESeq (eval_seq inline fix).
    eval_seq is a local fix inside eval, so we replicate its structure here. *)
Section Evaluator.
Variable vary_val : tid -> value.

Lemma uniform_eval_seq :
  forall fuel' g es t1 t2 rho1 rho2 acc_tr,
    (forall e, is_var_free e = true -> env_agrees g rho1 rho2 = true ->
               is_strongly_uniform g e = true ->
               eval vary_val fuel' t1 rho1 e = eval vary_val fuel' t2 rho2 e) ->
    forallb is_var_free es = true ->
    env_agrees g rho1 rho2 = true ->
    forallb (is_strongly_uniform g) es = true ->
    (fix eval_seq (xs : list expr) (acc : trace) : option (outcome * trace) :=
      match xs with
      | [] => Some (ONorm 0, acc)
      | x :: rest =>
          match eval vary_val fuel' t1 rho1 x with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
          | None => None
          end
      end) es acc_tr =
    (fix eval_seq (xs : list expr) (acc : trace) : option (outcome * trace) :=
      match xs with
      | [] => Some (ONorm 0, acc)
      | x :: rest =>
          match eval vary_val fuel' t2 rho2 x with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
          | None => None
          end
      end) es acc_tr.
Proof.
  intros fuel' g es t1 t2 rho1 rho2.
  induction es as [| h tl IHtl].
  - intros acc_tr _ _ _ _. reflexivity.
  - intros acc_tr IH Hf Hagr Hu.
    simpl in Hf. apply andb_true_iff in Hf as [Hfh Hftl].
    simpl in Hu. apply andb_true_iff in Hu as [Huh Hutl].
    simpl.
    rewrite (IH h Hfh Hagr Huh).
    destruct (eval vary_val fuel' t2 rho2 h) as [[[hv|hv] tr_h]|]; try reflexivity.
    exact (IHtl (acc_tr ++ tr_h) IH Hftl Hagr Hutl).
Qed.

(** Auxiliary: uniform body evals identically for EFor's loop. *)
Lemma uniform_eval_loop :
  forall fuel' body t1 t2 rho1 rho2 k acc_tr,
    eval vary_val fuel' t1 rho1 body = eval vary_val fuel' t2 rho2 body ->
    (fix loop (n : nat) (acc : trace) : option (outcome * trace) :=
      match n with
      | O => Some (ONorm 0, acc)
      | S n' =>
          match eval vary_val fuel' t1 rho1 body with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm _, tr) => loop n' (acc ++ tr)
          | None => None
          end
      end) k acc_tr =
    (fix loop (n : nat) (acc : trace) : option (outcome * trace) :=
      match n with
      | O => Some (ONorm 0, acc)
      | S n' =>
          match eval vary_val fuel' t2 rho2 body with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm _, tr) => loop n' (acc ++ tr)
          | None => None
          end
      end) k acc_tr.
Proof.
  intros fuel' body t1 t2 rho1 rho2 k acc_tr Hbody.
  induction k as [| k' IHk].
  - reflexivity.
  - simpl. rewrite Hbody. reflexivity.
Qed.

(** Auxiliary: uniform lists eval identically for EApp (eval_args inline fix). *)
Lemma uniform_eval_args :
  forall fuel' g es t1 t2 rho1 rho2 acc_tr last_v,
    (forall e, is_var_free e = true -> env_agrees g rho1 rho2 = true ->
               is_strongly_uniform g e = true ->
               eval vary_val fuel' t1 rho1 e = eval vary_val fuel' t2 rho2 e) ->
    forallb is_var_free es = true ->
    env_agrees g rho1 rho2 = true ->
    forallb (is_strongly_uniform g) es = true ->
    (fix eval_args (xs : list expr) (acc : trace) (lv : value)
        : option (outcome * trace) :=
      match xs with
      | [] => Some (ONorm lv, acc)
      | x :: rest =>
          match eval vary_val fuel' t1 rho1 x with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
          | None => None
          end
      end) es acc_tr last_v =
    (fix eval_args (xs : list expr) (acc : trace) (lv : value)
        : option (outcome * trace) :=
      match xs with
      | [] => Some (ONorm lv, acc)
      | x :: rest =>
          match eval vary_val fuel' t2 rho2 x with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
          | None => None
          end
      end) es acc_tr last_v.
Proof.
  intros fuel' g es t1 t2 rho1 rho2.
  induction es as [| h tl IHtl].
  - intros acc_tr last_v _ _ _ _. reflexivity.
  - intros acc_tr last_v IH Hf Hagr Hu.
    simpl in Hf. apply andb_true_iff in Hf as [Hfh Hftl].
    simpl in Hu. apply andb_true_iff in Hu as [Huh Hutl].
    simpl.
    rewrite (IH h Hfh Hagr Huh).
    destruct (eval vary_val fuel' t2 rho2 h) as [[[hv|hv] tr_h]|]; try reflexivity.
    exact (IHtl (acc_tr ++ tr_h) hv IH Hftl Hagr Hutl).
Qed.

(** not_varying_uniform: if is_var_free e = true, env_agrees g rho1 rho2, and
    is_strongly_uniform g e = true, then eval is uniform across thread ids and envs.

    The is_var_free precondition excludes EVar: for EVar x, env_lookup [] x = false
    (non-varying) but env_agrees [] rho1 rho2 = true does not constrain rho1/rho2,
    so the theorem would be false without this precondition.

    The ELet case uses env_agrees_extend to lift agreement through venv_extend.
    The ESeq/EApp cases use the uniform_eval_seq/uniform_eval_args helpers. *)

Theorem not_varying_uniform :
  forall fuel g e t1 t2 rho1 rho2,
    is_var_free e = true ->
    env_agrees g rho1 rho2 = true ->
    is_strongly_uniform g e = true ->
    eval vary_val fuel t1 rho1 e = eval vary_val fuel t2 rho2 e.
Proof.
  induction fuel as [| fuel' IHfuel]; intros g e t1 t2 rho1 rho2 Hf Hagr Hu.
  - destruct e; reflexivity.
  - destruct e.
    + (* ELit *)    reflexivity.
    + (* EVary *)   simpl in Hu. discriminate.
    + (* EBarrier *) reflexivity.
    + (* EWarpPoint *) reflexivity.
    + (* EVar x — excluded by is_var_free *)
      simpl in Hf. discriminate.
    + (* EBinop a b *)
      simpl in Hf. apply andb_true_iff in Hf as [Hfa Hfb].
      simpl in Hu. apply andb_true_iff in Hu as [Hua Hub].
      simpl.
      rewrite (IHfuel g e1 t1 t2 rho1 rho2 Hfa Hagr Hua).
      rewrite (IHfuel g e2 t1 t2 rho1 rho2 Hfb Hagr Hub). reflexivity.
    + (* EUnop eu *)
      simpl in Hf. simpl in Hu. simpl.
      rewrite (IHfuel g e t1 t2 rho1 rho2 Hf Hagr Hu). reflexivity.
    + (* EIf c et ef *)
      simpl in Hf. apply andb_true_iff in Hf as [Hfct Hfef].
      apply andb_true_iff in Hfct as [Hfc Hft].
      simpl in Hu. apply andb_true_iff in Hu as [Huct Huef].
      apply andb_true_iff in Huct as [Huc Hut].
      simpl.
      rewrite (IHfuel g e1 t1 t2 rho1 rho2 Hfc Hagr Huc).
      destruct (eval vary_val fuel' t2 rho2 e1) as [[[cv|cv] ctr]|]; try reflexivity.
      simpl. destruct (Nat.eqb cv 0) eqn:Hbr.
      ++ rewrite (IHfuel g e3 t1 t2 rho1 rho2 Hfef Hagr Huef). reflexivity.
      ++ rewrite (IHfuel g e2 t1 t2 rho1 rho2 Hft Hagr Hut). reflexivity.
    + (* EWhile c b *)
      simpl in Hf. apply andb_true_iff in Hf as [Hfc Hfb].
      simpl in Hu. apply andb_true_iff in Hu as [Huc Hub].
      simpl.
      rewrite (IHfuel g e1 t1 t2 rho1 rho2 Hfc Hagr Huc).
      destruct (eval vary_val fuel' t2 rho2 e1) as [[[cv|cv] ctr]|]; try reflexivity.
      destruct (Nat.eqb cv 0) eqn:Hcv; try reflexivity.
      rewrite (IHfuel g e2 t1 t2 rho1 rho2 Hfb Hagr Hub).
      destruct (eval vary_val fuel' t2 rho2 e2) as [[[bv|bv] btr]|]; try reflexivity.
      (* recursive EWhile call *)
      rewrite (IHfuel g (EWhile e1 e2) t1 t2 rho1 rho2).
      ++ reflexivity.
      ++ simpl. apply andb_true_iff. exact (conj Hfc Hfb).
      ++ exact Hagr.
      ++ simpl. apply andb_true_iff. exact (conj Huc Hub).
    + (* EFor lo hi b *)
      simpl in Hf. apply andb_true_iff in Hf as [Hflohi Hfb].
      apply andb_true_iff in Hflohi as [Hflo Hfhi].
      simpl in Hu. apply andb_true_iff in Hu as [Hulo_hi Hub].
      apply andb_true_iff in Hulo_hi as [Hulo Huhi].
      simpl.
      rewrite (IHfuel g e1 t1 t2 rho1 rho2 Hflo Hagr Hulo).
      destruct (eval vary_val fuel' t2 rho2 e1) as [[[lv|lv] ltr]|]; try reflexivity.
      rewrite (IHfuel g e2 t1 t2 rho1 rho2 Hfhi Hagr Huhi).
      destruct (eval vary_val fuel' t2 rho2 e2) as [[[hv|hv] htr]|]; try reflexivity.
      destruct (Nat.leb hv lv) eqn:Hle; try reflexivity.
      apply (uniform_eval_loop fuel' e3 t1 t2 rho1 rho2 (hv - lv) (ltr ++ htr)).
      exact (IHfuel g e3 t1 t2 rho1 rho2 Hfb Hagr Hub).
    + (* ESeq es — use uniform_eval_seq helper *)
      simpl in Hf. simpl in Hu. simpl.
      apply (uniform_eval_seq fuel' g l t1 t2 rho1 rho2 []).
      * intros e0 Hfe Hagre Hue. exact (IHfuel g e0 t1 t2 rho1 rho2 Hfe Hagre Hue).
      * exact Hf. * exact Hagr. * exact Hu.
    + (* ELet x v b *)
      simpl in Hf. apply andb_true_iff in Hf as [Hfv Hfb].
      simpl in Hu. apply andb_true_iff in Hu as [Huv Hub].
      simpl.
      rewrite (IHfuel g e1 t1 t2 rho1 rho2 Hfv Hagr Huv).
      destruct (eval vary_val fuel' t2 rho2 e1) as [[[w|w] trv]|]; try reflexivity.
      rewrite (IHfuel (env_extend g n (negb (is_strongly_uniform g e1)))
               e2 t1 t2 (venv_extend rho1 n w) (venv_extend rho2 n w)
               Hfb
               (env_agrees_extend g rho1 rho2 n w (negb (is_strongly_uniform g e1)) Hagr)
               Hub).
      reflexivity.
    + (* ESuperstep dv body cont *)
      simpl in Hf. apply andb_true_iff in Hf as [Hfbd Hfct].
      simpl in Hu. apply andb_true_iff in Hu as [Hubd Huct].
      simpl.
      rewrite (IHfuel g e1 t1 t2 rho1 rho2 Hfbd Hagr Hubd).
      destruct (eval vary_val fuel' t2 rho2 e1) as [[[sv|sv] str]|]; try reflexivity.
      rewrite (IHfuel g e2 t1 t2 rho1 rho2 Hfct Hagr Huct). reflexivity.
    + (* EApp args — use uniform_eval_args helper *)
      simpl in Hf. simpl in Hu. simpl.
      apply (uniform_eval_args fuel' g l t1 t2 rho1 rho2 [] 0).
      * intros e0 Hfe Hagre Hue. exact (IHfuel g e0 t1 t2 rho1 rho2 Hfe Hagre Hue).
      * exact Hf. * exact Hagr. * exact Hu.
    + (* EReturn er *)
      simpl in Hf. simpl in Hu. simpl.
      rewrite (IHfuel g e t1 t2 rho1 rho2 Hf Hagr Hu). reflexivity.
Qed.

(** not_varying_uniform_args: list version derived from not_varying_uniform. *)
Lemma not_varying_uniform_args :
  forall fuel g es t1 t2 rho1 rho2 acc_tr last_v,
    forallb is_var_free es = true ->
    env_agrees g rho1 rho2 = true ->
    forallb (is_strongly_uniform g) es = true ->
    (fix eval_args (xs : list expr) (acc : trace) (lv : value)
        : option (outcome * trace) :=
      match xs with
      | [] => Some (ONorm lv, acc)
      | x :: rest =>
          match eval vary_val fuel t1 rho1 x with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
          | None => None
          end
      end) es acc_tr last_v =
    (fix eval_args (xs : list expr) (acc : trace) (lv : value)
        : option (outcome * trace) :=
      match xs with
      | [] => Some (ONorm lv, acc)
      | x :: rest =>
          match eval vary_val fuel t2 rho2 x with
          | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
          | None => None
          end
      end) es acc_tr last_v.
Proof.
  intros fuel g es t1 t2 rho1 rho2 acc_tr last_v Hf Hagr Hu.
  apply (uniform_eval_args fuel g es t1 t2 rho1 rho2 acc_tr last_v).
  - intros e Hfe Hagre Hue. exact (not_varying_uniform fuel g e t1 t2 rho1 rho2 Hfe Hagre Hue).
  - exact Hf. - exact Hagr. - exact Hu.
Qed.

End Evaluator.

(* ----------------------------------------------------------------------- *)
(* 7.8  is_strongly_uniform_impl_is_not_varying                             *)
(* ----------------------------------------------------------------------- *)

(** is_strongly_uniform implies ¬is_varying_in_env              *)
Lemma is_strongly_uniform_impl_is_not_varying :
  forall env e,
    is_strongly_uniform env e = true ->
    is_varying_in_env env e = false.
Proof.
  intros env e. revert env.
  apply (expr_list_rect
    (fun e => forall env, is_strongly_uniform env e = true -> is_varying_in_env env e = false)
    (fun es => forall env, forallb (is_strongly_uniform env) es = true ->
               existsb (is_varying_in_env env) es = false)).
  (* ELit *) - intros env _. reflexivity.
  (* EVary *) - intros env Hu. simpl in Hu. discriminate.
  (* EBarrier *) - intros env _. reflexivity.
  (* EWarpPoint *) - intros env _. reflexivity.
  (* EVar x *)
  - intros x env Hu. simpl in Hu. simpl.
    destruct (env_lookup env x) eqn:Hel; simpl in Hu.
    + discriminate.
    + reflexivity.
  (* EBinop *)
  - intros a b IHa IHb env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Hua Hub].
    simpl. rewrite (IHa env Hua). rewrite (IHb env Hub). reflexivity.
  (* EUnop *)
  - intros eu IHeu env Hu. simpl in Hu. simpl. exact (IHeu env Hu).
  (* EIf *)
  - intros c t0 el IHc IHt0 IHel env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Huct Huel].
    apply andb_true_iff in Huct as [Huc Hut0].
    simpl. rewrite (IHc env Huc). rewrite (IHt0 env Hut0). rewrite (IHel env Huel). reflexivity.
  (* EWhile *)
  - intros c b IHc IHb env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Huc Hub].
    simpl. rewrite (IHc env Huc). rewrite (IHb env Hub). reflexivity.
  (* EFor *)
  - intros lo hi b IHlo IHhi IHb env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Hlohi Hub].
    apply andb_true_iff in Hlohi as [Hulo Huhi].
    simpl. rewrite (IHlo env Hulo). rewrite (IHhi env Huhi). rewrite (IHb env Hub). reflexivity.
  (* ESeq *)
  - intros es IHes env Hu. simpl in Hu. simpl. exact (IHes env Hu).
  (* ELet x v b *)
  - intros x v b IHv IHb env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Huv Hub].
    simpl. rewrite (IHv env Huv).
    rewrite Huv in Hub. simpl in Hub.
    exact (IHb (env_extend env x false) Hub).
  (* ESuperstep *)
  - intros dv body cont IHbd IHct env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Hubd Huct].
    simpl. rewrite (IHbd env Hubd). rewrite (IHct env Huct). reflexivity.
  (* EApp *)
  - intros args IHargs env Hu. simpl in Hu. simpl. exact (IHargs env Hu).
  (* EReturn *)
  - intros er IHer env Hu. simpl in Hu. simpl. exact (IHer env Hu).
  (* Plist [] *) - intros env _. reflexivity.
  (* Plist h :: tl *)
  - intros h tl IHh IHtl env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Huh Hutl].
    simpl. rewrite (IHh env Huh). exact (IHtl env Hutl).
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.9  var_free_env_irrelevant                                             *)
(* ----------------------------------------------------------------------- *)

(** var_free_env_irrelevant: for var-free expressions,
    is_strongly_uniform does not depend on the env (no EVar to look up). *)
Lemma var_free_env_irrelevant :
  forall e env1 env2,
    is_var_free e = true ->
    is_strongly_uniform env1 e = is_strongly_uniform env2 e.
Proof.
  intro e.
  apply (expr_list_rect
    (fun ex => forall env1 env2, is_var_free ex = true ->
               is_strongly_uniform env1 ex = is_strongly_uniform env2 ex)
    (fun bs => forall env1 env2,
               forallb is_var_free bs = true ->
               forallb (is_strongly_uniform env1) bs = forallb (is_strongly_uniform env2) bs)).
  (* ELit *) - intros env1 env2 _. reflexivity.
  (* EVary *) - intros env1 env2 _. reflexivity.
  (* EBarrier *) - intros env1 env2 _. reflexivity.
  (* EWarpPoint *) - intros env1 env2 _. reflexivity.
  (* EVar n0 — impossible *)
  - intros n0 env1 env2 Hf. simpl in Hf. discriminate.
  (* EBinop *)
  - intros a b IHa IHb env1 env2 Hf.
    simpl in Hf. apply andb_true_iff in Hf as [Ha Hb].
    simpl. rewrite (IHa env1 env2 Ha). rewrite (IHb env1 env2 Hb). reflexivity.
  (* EUnop *)
  - intros eu IHeu env1 env2 Hf. simpl in Hf. simpl. exact (IHeu env1 env2 Hf).
  (* EIf *)
  - intros c t0 el IHc IHt0 IHel env1 env2 Hf.
    simpl in Hf. apply andb_true_iff in Hf as [Hfet Hfel].
    apply andb_true_iff in Hfet as [Hfc Hft0].
    simpl. rewrite (IHc env1 env2 Hfc). rewrite (IHt0 env1 env2 Hft0).
    rewrite (IHel env1 env2 Hfel). reflexivity.
  (* EWhile *)
  - intros c b IHc IHb env1 env2 Hf.
    simpl in Hf. apply andb_true_iff in Hf as [Hfc Hfb].
    simpl. rewrite (IHc env1 env2 Hfc). rewrite (IHb env1 env2 Hfb). reflexivity.
  (* EFor *)
  - intros lo hi b IHlo IHhi IHb env1 env2 Hf.
    simpl in Hf. apply andb_true_iff in Hf as [Hfbod Hfb].
    apply andb_true_iff in Hfbod as [Hflo Hfhi].
    simpl. rewrite (IHlo env1 env2 Hflo). rewrite (IHhi env1 env2 Hfhi).
    rewrite (IHb env1 env2 Hfb). reflexivity.
  (* ESeq *)
  - intros es IHes env1 env2 Hf. simpl in *. exact (IHes env1 env2 Hf).
  (* ELet x0 v b *)
  - intros x0 v b IHv IHb env1 env2 Hf.
    simpl in Hf. apply andb_true_iff in Hf as [Hfv Hfb].
    simpl. rewrite (IHv env1 env2 Hfv).
    destruct (is_strongly_uniform env2 v) eqn:Huv2.
    + simpl.
      apply (IHb (env_extend env1 x0 false) (env_extend env2 x0 false) Hfb).
    + simpl. reflexivity.
  (* ESuperstep *)
  - intros dv bd ct IHbd IHct env1 env2 Hf.
    simpl in Hf. apply andb_true_iff in Hf as [Hfb Hfc].
    simpl. rewrite (IHbd env1 env2 Hfb). rewrite (IHct env1 env2 Hfc). reflexivity.
  (* EApp *)
  - intros args IHargs env1 env2 Hf. simpl in *. exact (IHargs env1 env2 Hf).
  (* EReturn *)
  - intros er IHer env1 env2 Hf. simpl in Hf. simpl. exact (IHer env1 env2 Hf).
  (* Plist [] *) - intros env1 env2 _. reflexivity.
  (* Plist (h :: tl) *)
  - intros h tl IHh IHtl env1 env2 Hf.
    simpl in Hf. apply andb_true_iff in Hf as [Hfh Hftl].
    simpl. rewrite (IHh env1 env2 Hfh). rewrite (IHtl env1 env2 Hftl). reflexivity.
Qed.

(** is_strongly_uniform_env_irrelevant: alias for var_free_env_irrelevant *)
Lemma is_strongly_uniform_env_irrelevant :
  forall e env1 env2,
    is_var_free e = true ->
    is_strongly_uniform env1 e = is_strongly_uniform env2 e.
Proof.
  exact var_free_env_irrelevant.
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.10  var_free_is_strongly_uniform_empty                                 *)
(* ----------------------------------------------------------------------- *)

(** var_free_is_strongly_uniform_empty: var-free + non-varying
    implies is_strongly_uniform [] e = true. *)
Lemma var_free_is_strongly_uniform_empty :
  forall e,
    is_var_free e = true ->
    is_varying e = false ->
    is_strongly_uniform [] e = true.
Proof.
  intro e.
  apply (expr_list_rect
    (fun e1 => is_var_free e1 = true -> is_varying e1 = false -> is_strongly_uniform [] e1 = true)
    (fun es => forallb is_var_free es = true ->
               existsb is_varying es = false ->
               forallb (is_strongly_uniform []) es = true)).
  (* ELit *) - intros _ _. reflexivity.
  (* EVary *) - intros _ Hv. simpl in Hv. discriminate.
  (* EBarrier *) - intros _ _. reflexivity.
  (* EWarpPoint *) - intros _ _. reflexivity.
  (* EVar n0 — is_var_free = false, excluded *)
  - intros n0 Hf. simpl in Hf. discriminate.
  (* EBinop a b *)
  - intros a b IHa IHb Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfa Hfb].
    simpl in Hv. apply orb_false_iff in Hv as [Hva Hvb].
    simpl. apply andb_true_iff. exact (conj (IHa Hfa Hva) (IHb Hfb Hvb)).
  (* EUnop eu *)
  - intros eu IHeu Hf Hv. simpl in Hf. simpl in Hv. simpl. exact (IHeu Hf Hv).
  (* EIf c t el *)
  - intros c t el IHc IHt IHel Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfct Hfel].
    apply andb_true_iff in Hfct as [Hfc Hft].
    simpl in Hv. apply orb_false_iff in Hv as [Hvctel Hvnel].
    apply orb_false_iff in Hvctel as [Hvc Hvt].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHc Hfc Hvc) (IHt Hft Hvt)).
    + exact (IHel Hfel Hvnel).
  (* EWhile c b *)
  - intros c b IHc IHb Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfc Hfb].
    simpl in Hv. apply orb_false_iff in Hv as [Hvc Hvb].
    simpl. apply andb_true_iff. exact (conj (IHc Hfc Hvc) (IHb Hfb Hvb)).
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hflohi Hfb].
    apply andb_true_iff in Hflohi as [Hflo Hfhi].
    simpl in Hv. apply orb_false_iff in Hv as [Hvlohi Hvb].
    apply orb_false_iff in Hvlohi as [Hvlo Hvhi].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHlo Hflo Hvlo) (IHhi Hfhi Hvhi)).
    + exact (IHb Hfb Hvb).
  (* ESeq es *)
  - intros es IHes Hf Hv. simpl. exact (IHes Hf Hv).
  (* ELet x v b *)
  - intros x v b IHv IHb Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfv Hfb].
    simpl in Hv. apply orb_false_iff in Hv as [Hvv Hvb].
    assert (Huv : is_strongly_uniform [] v = true). { exact (IHv Hfv Hvv). }
    assert (Hnv : negb (is_strongly_uniform [] v) = false). { rewrite Huv. reflexivity. }
    simpl. apply andb_true_iff. split.
    + exact Huv.
    + rewrite Hnv.
      rewrite (var_free_env_irrelevant b (env_extend [] x false) [] Hfb).
      exact (IHb Hfb Hvb).
  (* ESuperstep dv body cont *)
  - intros dv body cont IHbd IHct Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfbd Hfct].
    simpl in Hv. apply orb_false_iff in Hv as [Hvbd Hvct].
    simpl. apply andb_true_iff. exact (conj (IHbd Hfbd Hvbd) (IHct Hfct Hvct)).
  (* EApp args *)
  - intros args IHargs Hf Hv. simpl. exact (IHargs Hf Hv).
  (* EReturn er *)
  - intros er IHer Hf Hv. simpl in Hf. simpl in Hv. simpl. exact (IHer Hf Hv).
  (* Plist [] *) - intros _ _. reflexivity.
  (* Plist h :: tl *)
  - intros h tl IHh IHtl Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfh Hftl].
    simpl in Hv. apply orb_false_iff in Hv as [Hvh Hvtl].
    simpl. apply andb_true_iff. exact (conj (IHh Hfh Hvh) (IHtl Hftl Hvtl)).
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.11  closed_uniform                                                      *)
(* ----------------------------------------------------------------------- *)

(** closed_uniform: var-free + non-varying implies eval is fully uniform
    across all thread ids and variable environments. *)
Section Evaluator.
Variable vary_val : tid -> value.

Corollary closed_uniform :
  forall e fuel t1 t2 rho1 rho2,
    is_var_free e = true ->
    is_varying e = false ->
    eval vary_val fuel t1 rho1 e = eval vary_val fuel t2 rho2 e.
Proof.
  intros e fuel t1 t2 rho1 rho2 Hf Hv.
  apply (not_varying_uniform vary_val fuel [] e t1 t2 rho1 rho2 Hf).
  - simpl. reflexivity.
  - exact (var_free_is_strongly_uniform_empty e Hf Hv).
Qed.

End Evaluator.
