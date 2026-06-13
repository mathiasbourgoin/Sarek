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

From Stdlib Require Import List Arith Lia Bool.
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


(* ===== 7. T3-S2 — Uniformity soundness of is_varying_in_env ===== *)

(* -----------------------------------------------------------------------
 * T3-S2 design note: env_agrees + not_varying_uniform
 *
 * PLAN intended:
 *   not_varying_uniform: forall fuel env e t1 t2 rho1 rho2,
 *     env_agrees env rho1 rho2 -> is_varying_in_env env e = false ->
 *     eval vary_val fuel t1 rho1 e = eval vary_val fuel t2 rho2 e
 *
 * This statement is FALSE.  Counterexample:
 *   ELet n (EIf EVary EBarrier ELit) ELit
 *   is_varying_in_env [] (ELet n (EIf EVary EBarrier ELit) ELit)
 *   = is_varying_in_env [n→true] ELit = false  (PLAN's hypothesis holds)
 *   but eval t1 rho _ produces trace [EvBarrier] when vary_val t1 ≠ 0,
 *   while eval t2 rho _ produces trace [] when vary_val t2 = 0.
 *
 * Root cause: is_varying_in_env for ELet checks ONLY the body after extending
 * the env, ignoring whether the BINDER itself is varying.  This is correct for
 * the checker (a varying binder that is never used in the body is harmless for
 * barrier analysis) but wrong for an evaluator-uniformity claim.
 *
 * Fix: define is_strongly_uniform env e, which for ELet x v b additionally
 * requires v to be uniform.  Prove not_varying_uniform with this predicate.
 * Prove a bridge: is_strongly_uniform env e = false implies
 *                  is_varying_in_env env e = false.
 * The closed_uniform corollary (binding-blind, using is_varying) remains
 * exactly as the PLAN intended.
 * ----------------------------------------------------------------------- *)

(* ----------------------------------------------------------------------- *)
(* 7.1  env_agrees — uniform variables agree across two value environments  *)
(* ----------------------------------------------------------------------- *)

(** env_agrees env rho1 rho2:
    Every variable that env marks as NOT varying (env_lookup env x = false)
    has the same runtime value in rho1 and rho2. *)
Definition env_agrees (env : Env) (rho1 rho2 : venv) : Prop :=
  forall x, env_lookup env x = false ->
            venv_lookup rho1 x = venv_lookup rho2 x.

(* ----------------------------------------------------------------------- *)
(* 7.2  venv support lemmas                                                  *)
(* ----------------------------------------------------------------------- *)

Lemma venv_lookup_extend_same : forall rho x v,
  venv_lookup (venv_extend rho x v) x = v.
Proof.
  intros rho x v.
  unfold venv_lookup, venv_extend. simpl.
  rewrite Nat.eqb_refl. reflexivity.
Qed.

Lemma venv_lookup_extend_diff : forall rho x y v,
  x <> y -> venv_lookup (venv_extend rho x v) y = venv_lookup rho y.
Proof.
  intros rho x y v Hne.
  unfold venv_lookup, venv_extend. simpl.
  destruct (Nat.eqb x y) eqn:Heq.
  - apply Nat.eqb_eq in Heq. exfalso. exact (Hne Heq).
  - reflexivity.
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.3  env_lookup support lemmas                                            *)
(* ----------------------------------------------------------------------- *)

Lemma env_lookup_extend_same_key : forall env x v,
  env_lookup (env_extend env x v) x = v.
Proof.
  intros env x v.
  unfold env_lookup, env_extend. simpl.
  rewrite Nat.eqb_refl. reflexivity.
Qed.

Lemma env_lookup_extend_diff_key : forall env x y v,
  x <> y -> env_lookup (env_extend env x v) y = env_lookup env y.
Proof.
  intros env x y v Hne.
  unfold env_lookup, env_extend. simpl.
  destruct (Nat.eqb x y) eqn:Heq.
  - apply Nat.eqb_eq in Heq. exfalso. exact (Hne Heq).
  - reflexivity.
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.4  env_agrees_extend                                                    *)
(* ----------------------------------------------------------------------- *)

(** Extending both venvs with the SAME value v for variable x preserves
    env_agrees regardless of the variability flag vv. *)
Lemma env_agrees_extend :
  forall env rho1 rho2 x v vv,
    env_agrees env rho1 rho2 ->
    env_agrees (env_extend env x vv) (venv_extend rho1 x v) (venv_extend rho2 x v).
Proof.
  intros env rho1 rho2 x v vv Hagr y Henv.
  (* Case split: y = x or y ≠ x *)
  destruct (Nat.eq_dec x y) as [Heq | Hne].
  - (* y = x: both sides return v *)
    subst. rewrite !venv_lookup_extend_same. reflexivity.
  - (* y ≠ x: reduce to original rhos *)
    rewrite (venv_lookup_extend_diff rho1 x y v Hne).
    rewrite (venv_lookup_extend_diff rho2 x y v Hne).
    apply Hagr.
    (* env_lookup (env_extend env x vv) y = env_lookup env y since y ≠ x *)
    rewrite env_lookup_extend_diff_key in Henv.
    + exact Henv.
    + exact Hne.
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
(* 7.6  not_varying_uniform — main uniformity theorem                        *)
(* ----------------------------------------------------------------------- *)

(** Uniformity soundness of is_strongly_uniform.
    If is_strongly_uniform g e = true then eval is independent of tid and of
    values of g-varying variables, for any two g-agreeing value environments.
    Proof by induction on fuel only; destruct e in the S case. *)

(** Helper: eval_args equality for strongly-uniform arg lists *)
Lemma not_varying_uniform_args :
  forall vary_val fuel g (es : list expr) t1 t2 rho1 rho2,
    env_agrees g rho1 rho2 ->
    forallb (is_strongly_uniform g) es = true ->
    (forall e0, In e0 es ->
       is_strongly_uniform g e0 = true ->
       eval vary_val fuel t1 rho1 e0 = eval vary_val fuel t2 rho2 e0) ->
    forall acc last_v,
    (fix eval_args (xs : list expr) (acc_tr : trace) (lv : value)
        : option (outcome * trace) :=
      match xs with
      | []      => Some (ONorm lv, acc_tr)
      | hd :: tl0 =>
          match eval vary_val fuel t1 rho1 hd with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args tl0 (acc_tr ++ tr) v
          | None               => None
          end
      end) es acc last_v
    =
    (fix eval_args (xs : list expr) (acc_tr : trace) (lv : value)
        : option (outcome * trace) :=
      match xs with
      | []      => Some (ONorm lv, acc_tr)
      | hd :: tl0 =>
          match eval vary_val fuel t2 rho2 hd with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args tl0 (acc_tr ++ tr) v
          | None               => None
          end
      end) es acc last_v.
Proof.
  intros vary_val fuel g es t1 t2 rho1 rho2 Hagr Hfa IH.
  induction es as [| h tl IHtl].
  - intros acc lv. reflexivity.
  - intros acc lv.
    simpl in Hfa. apply andb_true_iff in Hfa as [Hh Htl].
    simpl.
    rewrite (IH h (in_eq h tl) Hh).
    destruct (eval vary_val fuel t2 rho2 h) as [[[v|v] tr]|]; try reflexivity.
    apply IHtl.
    + exact Htl.
    + intros e0 Hin Hu0. apply IH.
      * apply in_cons. exact Hin.
      * exact Hu0.
Qed.

Theorem not_varying_uniform :
  forall vary_val fuel g e t1 t2 rho1 rho2,
    env_agrees g rho1 rho2 ->
    is_strongly_uniform g e = true ->
    eval vary_val fuel t1 rho1 e = eval vary_val fuel t2 rho2 e.
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - intros g e t1 t2 rho1 rho2 _ _. reflexivity.
  - intros g e t1 t2 rho1 rho2 Hagr Hu.
    destruct e as [ (* ELit *)
                  | (* EVary *)
                  | (* EBarrier *)
                  | (* EWarpPoint *)
                  | n (* EVar n *)
                  | ec1 ec2 (* EBinop *)
                  | eu (* EUnop *)
                  | econd ethen eelse (* EIf *)
                  | ec eb (* EWhile *)
                  | elo ehi ebody (* EFor *)
                  | ess (* ESeq *)
                  | x eval0 ebody (* ELet *)
                  | edv ebody0 econt (* ESuperstep *)
                  | eargs (* EApp *)
                  | er (* EReturn *) ];
    simpl in Hu; simpl.
    (* ELit *) + reflexivity.
    (* EVary *) + discriminate.
    (* EBarrier *) + reflexivity.
    (* EWarpPoint *) + reflexivity.
    (* EVar n *)
    + apply negb_true_iff in Hu. rewrite (Hagr n Hu). reflexivity.
    (* EBinop ec1 ec2 *)
    + apply andb_true_iff in Hu as [Hu1 Hu2].
      rewrite (IHfuel g ec1 t1 t2 rho1 rho2 Hagr Hu1).
      destruct (eval vary_val fuel' t2 rho2 ec1) as [[[v|v] tr]|]; try reflexivity.
      rewrite (IHfuel g ec2 t1 t2 rho1 rho2 Hagr Hu2). reflexivity.
    (* EUnop eu *)
    + rewrite (IHfuel g eu t1 t2 rho1 rho2 Hagr Hu). reflexivity.
    (* EIf econd ethen eelse *)
    + apply andb_true_iff in Hu as [Huet Huel].
      apply andb_true_iff in Huet as [Huc Hut].
      rewrite (IHfuel g econd t1 t2 rho1 rho2 Hagr Huc).
      destruct (eval vary_val fuel' t2 rho2 econd) as [[[cv|cv] tr_c]|]; try reflexivity.
      destruct (Nat.eqb cv 0).
      * rewrite (IHfuel g eelse t1 t2 rho1 rho2 Hagr Huel). reflexivity.
      * rewrite (IHfuel g ethen t1 t2 rho1 rho2 Hagr Hut). reflexivity.
    (* EWhile ec eb *)
    + apply andb_true_iff in Hu as [Huc Hub].
      rewrite (IHfuel g ec t1 t2 rho1 rho2 Hagr Huc).
      destruct (eval vary_val fuel' t2 rho2 ec) as [[[cv|cv] tr_c]|]; try reflexivity.
      destruct (Nat.eqb cv 0); try reflexivity.
      rewrite (IHfuel g eb t1 t2 rho1 rho2 Hagr Hub).
      destruct (eval vary_val fuel' t2 rho2 eb) as [[[bv|bv] tr_b]|]; try reflexivity.
      assert (Hwu : is_strongly_uniform g (EWhile ec eb) = true).
      { simpl. apply andb_true_iff. exact (conj Huc Hub). }
      rewrite (IHfuel g (EWhile ec eb) t1 t2 rho1 rho2 Hagr Hwu). reflexivity.
    (* EFor elo ehi ebody *)
    + apply andb_true_iff in Hu as [Hlohi Hb].
      apply andb_true_iff in Hlohi as [Hlo Hhi].
      rewrite (IHfuel g elo t1 t2 rho1 rho2 Hagr Hlo).
      destruct (eval vary_val fuel' t2 rho2 elo) as [[[lv|lv] tr_lo]|]; try reflexivity.
      rewrite (IHfuel g ehi t1 t2 rho1 rho2 Hagr Hhi).
      destruct (eval vary_val fuel' t2 rho2 ehi) as [[[hv0|hv0] tr_hi]|]; try reflexivity.
      destruct (Nat.leb hv0 lv); try reflexivity.
      generalize (hv0 - lv) as steps. intro steps.
      induction steps as [| s' IHs].
      * simpl. reflexivity.
      * simpl.
        rewrite (IHfuel g ebody t1 t2 rho1 rho2 Hagr Hb).
        destruct (eval vary_val fuel' t2 rho2 ebody) as [[[bv|bv] tr_b]|]; reflexivity.
    (* ESeq ess *)
    + rewrite forallb_forall in Hu.
      assert (Hseq : forall acc,
          (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
            match xs with
            | []      => Some (ONorm 0, acc_tr)
            | hd0 :: rest0 =>
                match eval vary_val fuel' t1 rho1 hd0 with
                | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                | Some (ONorm _, tr) => eval_seq rest0 (acc_tr ++ tr)
                | None               => None
                end
            end) ess acc
          =
          (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
            match xs with
            | []      => Some (ONorm 0, acc_tr)
            | hd0 :: rest0 =>
                match eval vary_val fuel' t2 rho2 hd0 with
                | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                | Some (ONorm _, tr) => eval_seq rest0 (acc_tr ++ tr)
                | None               => None
                end
            end) ess acc).
      { induction ess as [| h0 tl0 IHtl].
        - intro acc. reflexivity.
        - intro acc.
          assert (Hh0 : is_strongly_uniform g h0 = true) by (apply Hu; apply in_eq).
          simpl.
          rewrite (IHfuel g h0 t1 t2 rho1 rho2 Hagr Hh0).
          destruct (eval vary_val fuel' t2 rho2 h0) as [[[hv|hv] tr_h]|]; try reflexivity.
          apply IHtl. intros e0 Hin. apply Hu. apply in_cons. exact Hin. }
      exact (Hseq []).
    (* ELet x eval0 ebody *)
    + apply andb_true_iff in Hu as [Huv Hub].
      assert (Hvv : negb (is_strongly_uniform g eval0) = false) by (rewrite Huv; reflexivity).
      rewrite Hvv in Hub.
      rewrite (IHfuel g eval0 t1 t2 rho1 rho2 Hagr Huv).
      destruct (eval vary_val fuel' t2 rho2 eval0) as [[[val0|val0] tr_v]|]; try reflexivity.
      assert (Hagr' : env_agrees (env_extend g x false)
                                   (venv_extend rho1 x val0)
                                   (venv_extend rho2 x val0))
        by exact (env_agrees_extend g rho1 rho2 x val0 false Hagr).
      rewrite (IHfuel (env_extend g x false) ebody t1 t2
                      (venv_extend rho1 x val0) (venv_extend rho2 x val0)
                      Hagr' Hub).
      reflexivity.
    (* ESuperstep edv ebody0 econt *)
    + apply andb_true_iff in Hu as [Hub Huc].
      rewrite (IHfuel g ebody0 t1 t2 rho1 rho2 Hagr Hub).
      destruct (eval vary_val fuel' t2 rho2 ebody0) as [[[bv|bv] tr_b]|]; try reflexivity.
      rewrite (IHfuel g econt t1 t2 rho1 rho2 Hagr Huc). reflexivity.
    (* EApp eargs *)
    + rewrite (not_varying_uniform_args vary_val fuel' g eargs t1 t2 rho1 rho2 Hagr Hu
                 (fun e0 Hin Hu0 => IHfuel g e0 t1 t2 rho1 rho2 Hagr Hu0)).
      reflexivity.
    (* EReturn er *)
    + rewrite (IHfuel g er t1 t2 rho1 rho2 Hagr Hu). reflexivity.
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.7  Bridge: is_strongly_uniform implies ¬is_varying_in_env              *)
(* ----------------------------------------------------------------------- *)

(** Monotonicity bridge: if e is strongly uniform then is_varying_in_env
    cannot see any varying component either.
    (The converse does NOT hold — see the design note above.) *)
Lemma is_strongly_uniform_impl_is_not_varying :
  forall e env,
    is_strongly_uniform env e = true ->
    is_varying_in_env env e = false.
Proof.
  apply (expr_list_rect
    (fun e => forall env, is_strongly_uniform env e = true -> is_varying_in_env env e = false)
    (fun es => forall env, forallb (is_strongly_uniform env) es = true -> existsb (is_varying_in_env env) es = false)).
  (* ELit *) - intros env _. reflexivity.
  (* EVary *) - intros env Hu. simpl in Hu. discriminate.
  (* EBarrier *) - intros env _. reflexivity.
  (* EWarpPoint *) - intros env _. reflexivity.
  (* EVar n *) - intros n env Hu. simpl in Hu. simpl. apply negb_true_iff in Hu. exact Hu.
  (* EBinop a b *)
  - intros a b IHa IHb env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Hua Hub].
    simpl. rewrite (IHa env Hua). rewrite (IHb env Hub). reflexivity.
  (* EUnop e0 *)
  - intros e0 IHe0 env Hu. simpl in Hu. simpl. exact (IHe0 env Hu).
  (* EIf c t0 el *)
  - intros c t0 el IHc IHt IHf env Hu.
    simpl in Hu.
    apply andb_true_iff in Hu as [Huef Huf].
    apply andb_true_iff in Huef as [Huc Hut].
    simpl. rewrite (IHc env Huc). rewrite (IHt env Hut). rewrite (IHf env Huf). reflexivity.
  (* EWhile c b *)
  - intros c b IHc IHb env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Huc Hub].
    simpl. rewrite (IHc env Huc). rewrite (IHb env Hub). reflexivity.
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb env Hu.
    simpl in Hu.
    apply andb_true_iff in Hu as [Hubod Hb].
    apply andb_true_iff in Hubod as [Hulo Huhi].
    simpl. rewrite (IHlo env Hulo). rewrite (IHhi env Huhi). rewrite (IHb env Hb). reflexivity.
  (* ESeq es *)
  - intros es IHes env Hu. simpl in Hu. simpl. exact (IHes env Hu).
  (* ELet x0 v b *)
  - intros x0 v b IHv IHb env Hu.
    simpl in Hu.
    apply andb_true_iff in Hu as [Huv Hub].
    assert (Hvv : negb (is_strongly_uniform env v) = false).
    { rewrite Huv. reflexivity. }
    rewrite Hvv in Hub.
    simpl. rewrite (IHv env Huv).
    exact (IHb (env_extend env x0 false) Hub).
  (* ESuperstep dv bd ct *)
  - intros dv bd ct IHbd IHct env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Hub Huc].
    simpl. rewrite (IHbd env Hub). rewrite (IHct env Huc). reflexivity.
  (* EApp args *)
  - intros args IHargs env Hu. simpl in Hu. simpl. exact (IHargs env Hu).
  (* EReturn e0 *)
  - intros e0 IHe0 env Hu. simpl in Hu. simpl. exact (IHe0 env Hu).
  (* Plist [] *) - intros env _. reflexivity.
  (* Plist (h :: tl) *)
  - intros h tl IHh IHtl env Hu.
    simpl in Hu. apply andb_true_iff in Hu as [Huh Hutl].
    simpl. rewrite (IHh env Huh). exact (IHtl env Hutl).
Qed.

(* ----------------------------------------------------------------------- *)
(* 7.8  Closed-expression corollary                                          *)
(* ----------------------------------------------------------------------- *)

(** is_var_free e — syntactic check that e contains no EVar constructor.
    Closed expressions are independent of any venv. *)
Fixpoint is_var_free (e : expr) : bool :=
  match e with
  | ELit | EVary | EBarrier | EWarpPoint => true
  | EVar _        => false
  | EBinop a b    => is_var_free a && is_var_free b
  | EUnop e1      => is_var_free e1
  | EIf c et ef   => is_var_free c && is_var_free et && is_var_free ef
  | EWhile c b    => is_var_free c && is_var_free b
  | EFor lo hi b  => is_var_free lo && is_var_free hi && is_var_free b
  | ESeq es       => forallb is_var_free es
  | ELet _ v b    => is_var_free v && is_var_free b
  | ESuperstep _ body cont => is_var_free body && is_var_free cont
  | EApp args     => forallb is_var_free args
  | EReturn e1    => is_var_free e1
  end.

(** var_free_env_irrelevant: a closed expression evaluates the same in any
    two venvs (for the same tid and fuel).
    Proof: by induction on expr; EVar case is vacuously excluded. *)
Lemma var_free_env_irrelevant :
  forall vary_val fuel t rho1 rho2 e,
    is_var_free e = true ->
    eval vary_val fuel t rho1 e = eval vary_val fuel t rho2 e.
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - intros. reflexivity.
  - (* fuel = S fuel' *)
    intros t rho1 rho2 e Hf.
    revert Hf.
    apply (expr_list_rect
      (fun e0 => is_var_free e0 = true ->
                 eval vary_val (S fuel') t rho1 e0 = eval vary_val (S fuel') t rho2 e0)
      (fun es => forallb is_var_free es = true ->
                 forall acc lv,
                 (fix eval_args (xs : list expr) (acc_tr : trace) (last_v : value)
                     : option (outcome * trace) :=
                   match xs with
                   | []      => Some (ONorm last_v, acc_tr)
                   | hd :: tl0 =>
                       match eval vary_val fuel' t rho1 hd with
                       | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                       | Some (ONorm v, tr) => eval_args tl0 (acc_tr ++ tr) v
                       | None               => None
                       end
                   end) es acc lv
                 =
                 (fix eval_args (xs : list expr) (acc_tr : trace) (last_v : value)
                     : option (outcome * trace) :=
                   match xs with
                   | []      => Some (ONorm last_v, acc_tr)
                   | hd :: tl0 =>
                       match eval vary_val fuel' t rho2 hd with
                       | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                       | Some (ONorm v, tr) => eval_args tl0 (acc_tr ++ tr) v
                       | None               => None
                       end
                   end) es acc lv)).
    (* ELit *) + intros _. simpl. reflexivity.
    (* EVary *) + intros _. simpl. reflexivity.
    (* EBarrier *) + intros _. simpl. reflexivity.
    (* EWarpPoint *) + intros _. simpl. reflexivity.
    (* EVar n0 — is_var_free = false, contradiction *)
    + intros n0 Hf0. simpl in Hf0. discriminate.
    (* EBinop a b *)
    + intros a b _ _ Hf0.
      simpl in Hf0. apply andb_true_iff in Hf0 as [Hfa Hfb].
      simpl. rewrite (IHfuel t rho1 rho2 a Hfa).
      destruct (eval vary_val fuel' t rho2 a) as [[[v|v] tr]|]; try reflexivity.
      rewrite (IHfuel t rho1 rho2 b Hfb). reflexivity.
    (* EUnop e0 *)
    + intros e0 _ Hf0. simpl in Hf0. simpl.
      rewrite (IHfuel t rho1 rho2 e0 Hf0). reflexivity.
    (* EIf c t0 el *)
    + intros c t0 el _ _ _ Hf0.
      simpl in Hf0.
      apply andb_true_iff in Hf0 as [Hfef Hff].
      apply andb_true_iff in Hfef as [Hfc Hft0].
      simpl. rewrite (IHfuel t rho1 rho2 c Hfc).
      destruct (eval vary_val fuel' t rho2 c) as [[[cv|cv] tr_c]|]; try reflexivity.
      destruct (Nat.eqb cv 0).
      * rewrite (IHfuel t rho1 rho2 el Hff). reflexivity.
      * rewrite (IHfuel t rho1 rho2 t0 Hft0). reflexivity.
    (* EWhile c b *)
    + intros c b _ _ Hf0.
      simpl in Hf0. apply andb_true_iff in Hf0 as [Hfc Hfb].
      simpl. rewrite (IHfuel t rho1 rho2 c Hfc).
      destruct (eval vary_val fuel' t rho2 c) as [[[cv|cv] tr_c]|]; try reflexivity.
      destruct (Nat.eqb cv 0); try reflexivity.
      rewrite (IHfuel t rho1 rho2 b Hfb).
      destruct (eval vary_val fuel' t rho2 b) as [[[bv|bv] tr_b]|]; try reflexivity.
      assert (Hwf : is_var_free (EWhile c b) = true).
      { simpl. apply andb_true_iff. exact (conj Hfc Hfb). }
      rewrite (IHfuel t rho1 rho2 (EWhile c b) Hwf). reflexivity.
    (* EFor lo hi b *)
    + intros lo hi b _ _ _ Hf0.
      simpl in Hf0.
      apply andb_true_iff in Hf0 as [Hfbod Hfb].
      apply andb_true_iff in Hfbod as [Hflo Hfhi].
      simpl. rewrite (IHfuel t rho1 rho2 lo Hflo).
      destruct (eval vary_val fuel' t rho2 lo) as [[[lv|lv] tr_lo]|]; try reflexivity.
      rewrite (IHfuel t rho1 rho2 hi Hfhi).
      destruct (eval vary_val fuel' t rho2 hi) as [[[hv|hv] tr_hi]|]; try reflexivity.
      destruct (Nat.leb hv lv); try reflexivity.
      generalize (hv - lv) as steps. intro steps.
      induction steps as [| s' _].
      * simpl. reflexivity.
      * simpl.
        rewrite (IHfuel t rho1 rho2 b Hfb).
        destruct (eval vary_val fuel' t rho2 b) as [[[bv|bv] tr_b]|]; reflexivity.
    (* ESeq es *)
    + intros es _ Hf0.
      simpl.
      assert (Hseq : forall acc,
          (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
            match xs with
            | []      => Some (ONorm 0, acc_tr)
            | x :: rest =>
                match eval vary_val fuel' t rho1 x with
                | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
                | None               => None
                end
            end) es acc
          =
          (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
            match xs with
            | []      => Some (ONorm 0, acc_tr)
            | x :: rest =>
                match eval vary_val fuel' t rho2 x with
                | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
                | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
                | None               => None
                end
            end) es acc).
      { induction es as [| h tl IHtl].
        - intro acc. simpl. reflexivity.
        - intro acc. simpl in Hf0. apply andb_true_iff in Hf0 as [Hfh Hftl].
          simpl.
          rewrite (IHfuel t rho1 rho2 h Hfh).
          destruct (eval vary_val fuel' t rho2 h) as [[[hv|hv] tr_h]|]; try reflexivity.
          apply IHtl. exact Hftl. }
      exact (Hseq []).
    (* ELet x0 v b *)
    + intros x0 v b _ _ Hf0.
      simpl in Hf0. apply andb_true_iff in Hf0 as [Hfv Hfb].
      simpl.
      rewrite (IHfuel t rho1 rho2 v Hfv).
      destruct (eval vary_val fuel' t rho2 v) as [[[val0|val0] tr_v]|]; try reflexivity.
      rewrite (IHfuel t (venv_extend rho1 x0 val0) (venv_extend rho2 x0 val0) b Hfb). reflexivity.
    (* ESuperstep dv bd ct *)
    + intros dv bd ct _ _ Hf0.
      simpl in Hf0. apply andb_true_iff in Hf0 as [Hfb Hfc].
      simpl.
      rewrite (IHfuel t rho1 rho2 bd Hfb).
      destruct (eval vary_val fuel' t rho2 bd) as [[[bv|bv] tr_b]|]; try reflexivity.
      rewrite (IHfuel t rho1 rho2 ct Hfc). reflexivity.
    (* EApp args *)
    + intros args IHargs Hf0.
      simpl in Hf0. simpl.
      exact (IHargs Hf0 [] 0).
    (* EReturn e0 *)
    + intros e0 _ Hf0. simpl in Hf0. simpl.
      rewrite (IHfuel t rho1 rho2 e0 Hf0). reflexivity.
    (* Plist [] *)
    + intros _ acc lv. simpl. reflexivity.
    (* Plist (h :: tl) *)
    + intros h tl _ IHtl Hf0 acc lv.
      simpl in Hf0. apply andb_true_iff in Hf0 as [Hfh Hftl].
      simpl.
      rewrite (IHfuel t rho1 rho2 h Hfh).
      destruct (eval vary_val fuel' t rho2 h) as [[[hv|hv] tr_h]|]; try reflexivity.
      exact (IHtl Hftl (acc ++ tr_h) hv).
Qed.

(** is_strongly_uniform_env_irrelevant: for var-free expressions,
    is_strongly_uniform does not depend on the env (no EVar to look up). *)
Lemma is_strongly_uniform_env_irrelevant :
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
  (* EVar n0 — is_var_free = false *)
  - intros n0 Hf. simpl in Hf. discriminate.
  (* EBinop a b *)
  - intros a b IHa IHb Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfa Hfb].
    simpl in Hv. apply orb_false_iff in Hv as [Hva Hvb].
    simpl. apply andb_true_iff. exact (conj (IHa Hfa Hva) (IHb Hfb Hvb)).
  (* EUnop eu *)
  - intros eu IHeu Hf Hv. simpl in Hf. simpl in Hv. simpl. exact (IHeu Hf Hv).
  (* EIf c t0 el *)
  - intros c t0 el IHc IHt0 IHel Hf Hv.
    simpl in Hf.
    apply andb_true_iff in Hf as [Hfet Hfel].
    apply andb_true_iff in Hfet as [Hfc Hft0].
    simpl in Hv.
    apply orb_false_iff in Hv as [Hvcet Hvel].
    apply orb_false_iff in Hvcet as [Hvc Hvt0].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHc Hfc Hvc) (IHt0 Hft0 Hvt0)).
    + exact (IHel Hfel Hvel).
  (* EWhile c b *)
  - intros c b IHc IHb Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfc Hfb].
    simpl in Hv. apply orb_false_iff in Hv as [Hvc Hvb].
    simpl. apply andb_true_iff. exact (conj (IHc Hfc Hvc) (IHb Hfb Hvb)).
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb Hf Hv.
    simpl in Hf.
    apply andb_true_iff in Hf as [Hfbod Hfb].
    apply andb_true_iff in Hfbod as [Hflo Hfhi].
    simpl in Hv.
    apply orb_false_iff in Hv as [Hvbod Hvb].
    apply orb_false_iff in Hvbod as [Hvlo Hvhi].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHlo Hflo Hvlo) (IHhi Hfhi Hvhi)).
    + exact (IHb Hfb Hvb).
  (* ESeq es *)
  - intros es IHes Hf Hv. simpl in Hf. simpl in Hv. simpl. exact (IHes Hf Hv).
  (* ELet x0 v b *)
  - intros x0 v b IHv IHb Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfv Hfb].
    simpl in Hv. apply orb_false_iff in Hv as [Hvv Hvb].
    simpl.
    assert (Huv : is_strongly_uniform [] v = true). { exact (IHv Hfv Hvv). }
    assert (Hvvflag : negb (is_strongly_uniform [] v) = false). { rewrite Huv. reflexivity. }
    rewrite Hvvflag. apply andb_true_iff. split.
    + exact Huv.
    + (* is_strongly_uniform (env_extend [] x0 false) b = true *)
      rewrite <- (is_strongly_uniform_env_irrelevant b [] (env_extend [] x0 false) Hfb).
      exact (IHb Hfb Hvb).
  (* ESuperstep dv bd ct *)
  - intros dv bd ct IHbd IHct Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfb Hfc].
    simpl in Hv. apply orb_false_iff in Hv as [Hvb Hvc].
    simpl. apply andb_true_iff. exact (conj (IHbd Hfb Hvb) (IHct Hfc Hvc)).
  (* EApp args *)
  - intros args IHargs Hf Hv. simpl in *. exact (IHargs Hf Hv).
  (* EReturn er *)
  - intros er IHer Hf Hv. simpl in *. exact (IHer Hf Hv).
  (* Plist [] *) - intros _ _. reflexivity.
  (* Plist (h :: tl) *)
  - intros h tl IHh IHtl Hf Hv.
    simpl in Hf. apply andb_true_iff in Hf as [Hfh Hftl].
    simpl in Hv. apply orb_false_iff in Hv as [Hvh Hvtl].
    simpl. rewrite (IHh Hfh Hvh). rewrite (IHtl Hftl Hvtl). reflexivity.
Qed.

(** closed_uniform: For closed (EVar-free), EVary-free expressions,
    evaluation is completely independent of tid and venv. *)
Theorem closed_uniform :
  forall vary_val fuel e t1 t2 rho1 rho2,
    is_var_free e = true ->
    is_varying e = false ->
    eval vary_val fuel t1 rho1 e = eval vary_val fuel t2 rho2 e.
Proof.
  intros vary_val fuel e t1 t2 rho1 rho2 Hf Hv.
  assert (H_self_agrees : forall rho, env_agrees [] rho rho).
  { intros rho x0 _. reflexivity. }
  rewrite (var_free_env_irrelevant vary_val fuel t1 rho1 rho2 e Hf).
  exact (not_varying_uniform vary_val fuel [] e t1 t2 rho2 rho2
           (H_self_agrees rho2) (var_free_is_strongly_uniform_empty e Hf Hv)).
Qed.


(* ===== 8. T3-S3 — Trace silence of barrier-free, superstep-free expressions ===== *)

(*
 * T3-S3 design note:
 *
 * The PLAN stated: barrier_free_silent → tr = [].
 * This is too strong: EWarpPoint is barrier_free but emits [EvWarp], and
 * ESuperstep true with barrier_free body emits [EvBarrier].
 *
 * Correct statement: barrier_free + superstep_free → no EvBarrier in tr.
 * The superstep_free side-condition excludes ESuperstep (which emits EvBarrier
 * for all dv values). EWarpPoint may emit [EvWarp]; this is correct and
 * intentional. For T3-S4 barrier_safe purposes, EWarpPoint is handled by
 * T3-S2 (not_varying_uniform covers it since EWarpPoint is not varying).
 *)

(** no_barrier_event: the trace contains no EvBarrier events. *)
Definition no_barrier_event (tr : trace) : bool :=
  forallb (fun ev => match ev with EvBarrier => false | _ => true end) tr.

(** superstep_free: e contains no ESuperstep node at any depth. *)
Fixpoint superstep_free (e : expr) : bool :=
  match e with
  | ELit | EVary | EBarrier | EWarpPoint | EVar _ => true
  | EBinop a b       => superstep_free a && superstep_free b
  | EUnop e0         => superstep_free e0
  | EIf c t0 el      => superstep_free c && superstep_free t0 && superstep_free el
  | EWhile c b       => superstep_free c && superstep_free b
  | EFor lo hi b     => superstep_free lo && superstep_free hi && superstep_free b
  | ESeq es          => forallb superstep_free es
  | ELet _ v b       => superstep_free v && superstep_free b
  | ESuperstep _ _ _ => false
  | EApp args        => forallb superstep_free args
  | EReturn e0       => superstep_free e0
  end.

(** no_barrier_app: no_barrier_event distributes over list append. *)
Lemma no_barrier_app : forall tr1 tr2,
  no_barrier_event (tr1 ++ tr2) = no_barrier_event tr1 && no_barrier_event tr2.
Proof.
  intros tr1 tr2. unfold no_barrier_event. apply forallb_app.
Qed.

(** for_loop_fixed_no_barrier: the EFor body loop preserves no_barrier_event. *)
Lemma for_loop_fixed_no_barrier :
  forall body k acc o tr,
    no_barrier_event acc = true ->
    (forall o' tr', body = Some (o', tr') -> no_barrier_event tr' = true) ->
    for_loop_fixed body k acc = Some (o, tr) ->
    no_barrier_event tr = true.
Proof.
  intros body k.
  induction k as [| k' IHk].
  - intros acc o tr Hacc _ H. simpl in H. inversion H. subst. exact Hacc.
  - intros acc o tr Hacc Hbody H.
    simpl in H.
    destruct body as [[[bv | bv] tr_b] |] eqn:Ebody.
    + (* ONorm bv — recursive case *)
      apply IHk with (acc := acc ++ tr_b) (o := o) (tr := tr).
      * rewrite no_barrier_app. apply andb_true_iff; split.
        { exact Hacc. } { exact (Hbody (ONorm bv) tr_b eq_refl). }
      * exact Hbody.
      * exact H.
    + (* ORet bv — terminal: tr = acc ++ tr_b *)
      inversion H. subst.
      rewrite no_barrier_app. apply andb_true_iff; split.
      * exact Hacc. * exact (Hbody (ORet bv) tr_b eq_refl).
    + discriminate.
Qed.

(** eval_seq_no_barrier: the ESeq inner accumulator loop preserves no_barrier_event. *)
Lemma eval_seq_no_barrier :
  forall vary_val n t rho,
    (forall e o tr,
      superstep_free e = true ->
      barrier_free e = true ->
      eval vary_val n t rho e = Some (o, tr) ->
      no_barrier_event tr = true) ->
  forall xs acc o tr,
    no_barrier_event acc = true ->
    forallb superstep_free xs = true ->
    forallb barrier_free xs = true ->
    (fix eval_seq (xs0 : list expr) (acc_tr : trace) : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm 0, acc_tr)
      | x :: rest =>
          match eval vary_val n t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
          | None               => None
          end
      end) xs acc = Some (o, tr) ->
    no_barrier_event tr = true.
Proof.
  intros vary_val n t rho IHn.
  induction xs as [| x xs' IHxs].
  - intros acc o tr Hacc _ _ H. simpl in H. inversion H. subst. exact Hacc.
  - intros acc o tr Hacc Hsfree Hbfree H.
    simpl in Hsfree. apply andb_true_iff in Hsfree as [Hsfx Hsfxs].
    simpl in Hbfree. apply andb_true_iff in Hbfree as [Hbfx Hbfxs].
    simpl in H.
    destruct (eval vary_val n t rho x) as [[[xv | xv] xtr] |] eqn:Hx.
    + (* ONorm — recurse *)
      apply IHxs with (acc := acc ++ xtr) (o := o) (tr := tr).
      * rewrite no_barrier_app. apply andb_true_iff; split.
        { exact Hacc. } { exact (IHn x (ONorm xv) xtr Hsfx Hbfx Hx). }
      * exact Hsfxs. * exact Hbfxs. * exact H.
    + (* ORet — terminal: tr = acc ++ xtr *)
      inversion H. subst.
      rewrite no_barrier_app. apply andb_true_iff; split.
      * exact Hacc. * exact (IHn x (ORet xv) xtr Hsfx Hbfx Hx).
    + discriminate.
Qed.

(** eval_args_no_barrier: the EApp inner accumulator loop preserves no_barrier_event.
    Analogous to eval_seq_no_barrier but with a last_v value accumulator. *)
Lemma eval_args_no_barrier :
  forall vary_val n t rho,
    (forall e o tr,
      superstep_free e = true ->
      barrier_free e = true ->
      eval vary_val n t rho e = Some (o, tr) ->
      no_barrier_event tr = true) ->
  forall xs acc last_v o tr,
    no_barrier_event acc = true ->
    forallb superstep_free xs = true ->
    forallb barrier_free xs = true ->
    (fix eval_args (xs0 : list expr) (acc_tr : trace) (lv : value)
        : option (outcome * trace) :=
      match xs0 with
      | []      => Some (ONorm lv, acc_tr)
      | x :: rest =>
          match eval vary_val n t rho x with
          | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
          | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
          | None               => None
          end
      end) xs acc last_v = Some (o, tr) ->
    no_barrier_event tr = true.
Proof.
  intros vary_val n t rho IHn.
  induction xs as [| x xs' IHxs].
  - intros acc last_v o tr Hacc _ _ H. simpl in H. inversion H. subst. exact Hacc.
  - intros acc last_v o tr Hacc Hsfree Hbfree H.
    simpl in Hsfree. apply andb_true_iff in Hsfree as [Hsfx Hsfxs].
    simpl in Hbfree. apply andb_true_iff in Hbfree as [Hbfx Hbfxs].
    simpl in H.
    destruct (eval vary_val n t rho x) as [[[xv | xv] xtr] |] eqn:Hx.
    + (* ONorm xv — recurse with new last_v = xv *)
      apply IHxs with (acc := acc ++ xtr) (last_v := xv) (o := o) (tr := tr).
      * rewrite no_barrier_app. apply andb_true_iff; split.
        { exact Hacc. } { exact (IHn x (ONorm xv) xtr Hsfx Hbfx Hx). }
      * exact Hsfxs. * exact Hbfxs. * exact H.
    + (* ORet xv — terminal: tr = acc ++ xtr *)
      inversion H. subst.
      rewrite no_barrier_app. apply andb_true_iff; split.
      * exact Hacc. * exact (IHn x (ORet xv) xtr Hsfx Hbfx Hx).
    + discriminate.
Qed.

(** barrier_free_no_barriers (T3-S3 main): if e is barrier_free and
    superstep_free, any completed evaluation emits no EvBarrier events.
    EWarpPoint may emit [EvWarp] — this is intentional (see design note above). *)
Theorem barrier_free_no_barriers :
  forall vary_val fuel t rho e o tr,
    superstep_free e = true ->
    barrier_free e = true ->
    eval vary_val fuel t rho e = Some (o, tr) ->
    no_barrier_event tr = true.
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - intros t rho e o tr _ _ H. simpl in H. discriminate.
  - intros t rho e o tr Hsf Hbf H.
    destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec ebody
                  | elo ehi ebod | ess | xn eval0 ebody | dv ebod econt | eargs | er ];
    simpl in Hsf; simpl in Hbf; simpl in H.
    (* ELit *)     + inversion H. subst. reflexivity.
    (* EVary *)    + inversion H. subst. reflexivity.
    (* EBarrier — barrier_free = false *) + discriminate.
    (* EWarpPoint — emits [EvWarp]; match ev with EvBarrier => false | _ => true end EvWarp = true *)
    + inversion H. subst. reflexivity.
    (* EVar n0 *)  + inversion H. subst. reflexivity.
    (* EBinop ea eb *)
    + apply andb_true_iff in Hsf as [Hsfa Hsfb].
      apply andb_true_iff in Hbf as [Hbfa Hbfb].
      destruct (eval vary_val fuel' t rho ea) as [[[va | va] tra] |] eqn:Hea.
      * destruct (eval vary_val fuel' t rho eb) as [[[vb | vb] trb] |] eqn:Heb.
        -- inversion H. subst. rewrite no_barrier_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho ea _ tra Hsfa Hbfa Hea).
           ++ exact (IHfuel t rho eb _ trb Hsfb Hbfb Heb).
        -- inversion H. subst. rewrite no_barrier_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho ea _ tra Hsfa Hbfa Hea).
           ++ exact (IHfuel t rho eb _ trb Hsfb Hbfb Heb).
        -- discriminate.
      * inversion H. subst. exact (IHfuel t rho ea (ORet va) tr Hsfa Hbfa Hea).
      * discriminate.
    (* EUnop eu *)
    + destruct (eval vary_val fuel' t rho eu) as [[[v | v] tr0] |] eqn:Heu.
      * inversion H. subst. exact (IHfuel t rho eu (ONorm v) tr Hsf Hbf Heu).
      * inversion H. subst. exact (IHfuel t rho eu (ORet v) tr Hsf Hbf Heu).
      * discriminate.
    (* EIf econd ethen eelse *)
    + apply andb_true_iff in Hsf as [Hsfce Hsfe].
      apply andb_true_iff in Hsfce as [Hsfc Hsft].
      apply andb_true_iff in Hbf as [Hbfce Hbfe].
      apply andb_true_iff in Hbfce as [Hbfc Hbft].
      destruct (eval vary_val fuel' t rho econd) as [[[cv | cv] tr_c] |] eqn:Hcond.
      * (* ONorm cv: evaluate selected branch *)
        destruct (eval vary_val fuel' t rho (if Nat.eqb cv 0 then eelse else ethen))
              as [[ob tr_b] |] eqn:Hbranch.
        -- inversion H. subst. rewrite no_barrier_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho econd (ONorm cv) tr_c Hsfc Hbfc Hcond).
           ++ destruct (Nat.eqb cv 0); simpl in Hbranch.
              ** exact (IHfuel t rho eelse _ tr_b Hsfe Hbfe Hbranch).
              ** exact (IHfuel t rho ethen _ tr_b Hsft Hbft Hbranch).
        -- discriminate.
      * (* ORet cv: short-circuit *)
        inversion H. subst. exact (IHfuel t rho econd (ORet cv) tr Hsfc Hbfc Hcond).
      * discriminate.
    (* EWhile ec ebody *)
    + apply andb_true_iff in Hsf as [Hsfc Hsfb].
      apply andb_true_iff in Hbf as [Hbfc Hbfb].
      destruct (eval vary_val fuel' t rho ec) as [[[cv | cv] tr_c] |] eqn:Hcond.
      * (* ONorm cv: check loop condition *)
        destruct (Nat.eqb cv 0).
        -- (* cv = 0: loop done *)
           inversion H. subst. exact (IHfuel t rho ec (ONorm cv) tr Hsfc Hbfc Hcond).
        -- (* cv ≠ 0: loop body *)
           destruct (eval vary_val fuel' t rho ebody) as [[[bv | bv] tr_b] |] eqn:Hbod.
           ++ (* ONorm bv — recurse on EWhile *)
              destruct (eval vary_val fuel' t rho (EWhile ec ebody))
                  as [[ol tr_l] |] eqn:Hloop.
              ** inversion H. subst.
                 rewrite no_barrier_app, no_barrier_app.
                 apply andb_true_iff; split.
                 { exact (IHfuel t rho ec (ONorm cv) tr_c Hsfc Hbfc Hcond). }
                 apply andb_true_iff; split.
                 { exact (IHfuel t rho ebody (ONorm bv) tr_b Hsfb Hbfb Hbod). }
                 assert (Hsfw : superstep_free (EWhile ec ebody) = true).
                 { simpl. rewrite Hsfc, Hsfb. reflexivity. }
                 assert (Hbfw : barrier_free (EWhile ec ebody) = true).
                 { simpl. rewrite Hbfc, Hbfb. reflexivity. }
                 exact (IHfuel t rho (EWhile ec ebody) _ tr_l Hsfw Hbfw Hloop).
              ** discriminate.
           ++ (* ORet bv — early return from body *)
              inversion H. subst. rewrite no_barrier_app. apply andb_true_iff; split.
              ** exact (IHfuel t rho ec (ONorm cv) tr_c Hsfc Hbfc Hcond).
              ** exact (IHfuel t rho ebody (ORet bv) tr_b Hsfb Hbfb Hbod).
           ++ discriminate.
      * (* ORet cv: short-circuit from condition *)
        inversion H. subst. exact (IHfuel t rho ec (ORet cv) tr Hsfc Hbfc Hcond).
      * discriminate.
    (* EFor elo ehi ebod *)
    + apply andb_true_iff in Hsf as [Hsflh Hsfb].
      apply andb_true_iff in Hsflh as [Hsfl Hsfh].
      apply andb_true_iff in Hbf as [Hbflh Hbfb].
      apply andb_true_iff in Hbflh as [Hbfl Hbfh].
      destruct (eval vary_val fuel' t rho elo) as [[[lo_v | lo_v] tr_lo] |] eqn:Hlo.
      * (* ONorm lo_v: evaluate upper bound *)
        destruct (eval vary_val fuel' t rho ehi) as [[[hi_v | hi_v] tr_hi] |] eqn:Hhi.
        -- (* ONorm hi_v: run the loop *)
           destruct (Nat.leb hi_v lo_v) eqn:Hle.
           ++ (* empty loop: tr = tr_lo ++ tr_hi *)
              inversion H. subst. rewrite no_barrier_app. apply andb_true_iff; split.
              ** exact (IHfuel t rho elo (ONorm lo_v) tr_lo Hsfl Hbfl Hlo).
              ** exact (IHfuel t rho ehi (ONorm hi_v) tr_hi Hsfh Hbfh Hhi).
           ++ (* for-loop: rewrite as for_loop_fixed then apply helper *)
              rewrite (for_loop_eq vary_val fuel' t rho ebod (hi_v - lo_v) (tr_lo ++ tr_hi)) in H.
              apply for_loop_fixed_no_barrier
                with (body := eval vary_val fuel' t rho ebod)
                     (k    := hi_v - lo_v)
                     (acc  := tr_lo ++ tr_hi)
                     (o    := o)
                     (tr   := tr).
              ** rewrite no_barrier_app. apply andb_true_iff; split.
                 { exact (IHfuel t rho elo (ONorm lo_v) tr_lo Hsfl Hbfl Hlo). }
                 { exact (IHfuel t rho ehi (ONorm hi_v) tr_hi Hsfh Hbfh Hhi). }
              ** intros o' tr' Hbod. exact (IHfuel t rho ebod o' tr' Hsfb Hbfb Hbod).
              ** exact H.
        -- (* ORet hi_v: short-circuit from upper bound; tr = tr_lo ++ tr_hi *)
           inversion H. subst. rewrite no_barrier_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho elo (ONorm lo_v) tr_lo Hsfl Hbfl Hlo).
           ++ exact (IHfuel t rho ehi (ORet hi_v) tr_hi Hsfh Hbfh Hhi).
        -- discriminate.
      * (* ORet lo_v: short-circuit from lower bound *)
        inversion H. subst. exact (IHfuel t rho elo (ORet lo_v) tr Hsfl Hbfl Hlo).
      * discriminate.
    (* ESeq ess *)
    + apply eval_seq_no_barrier
        with (vary_val := vary_val) (n := fuel') (t := t) (rho := rho)
             (xs := ess) (acc := []) (o := o) (tr := tr).
      * intros e0 o' tr' Hsfe Hbfe He.
        exact (IHfuel t rho e0 o' tr' Hsfe Hbfe He).
      * reflexivity.
      * exact Hsf.
      * exact Hbf.
      * exact H.
    (* ELet xn eval0 ebody *)
    + apply andb_true_iff in Hsf as [Hsfv Hsfb].
      apply andb_true_iff in Hbf as [Hbfv Hbfb].
      destruct (eval vary_val fuel' t rho eval0) as [[[vv | vv] tr_v] |] eqn:Hval.
      * (* ONorm vv — eval body with extended env *)
        destruct (eval vary_val fuel' t (venv_extend rho xn vv) ebody)
            as [[ob tr_b] |] eqn:Hbod.
        -- inversion H. subst. rewrite no_barrier_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho eval0 _ tr_v Hsfv Hbfv Hval).
           ++ exact (IHfuel t (venv_extend rho xn vv) ebody _ tr_b Hsfb Hbfb Hbod).
        -- discriminate.
      * inversion H. subst. exact (IHfuel t rho eval0 (ORet vv) tr Hsfv Hbfv Hval).
      * discriminate.
    (* ESuperstep — superstep_free = false *) + discriminate.
    (* EApp eargs *)
    + apply eval_args_no_barrier
        with (vary_val := vary_val) (n := fuel') (t := t) (rho := rho)
             (xs := eargs) (acc := []) (last_v := 0) (o := o) (tr := tr).
      * intros e0 o' tr' Hsfe Hbfe He.
        exact (IHfuel t rho e0 o' tr' Hsfe Hbfe He).
      * reflexivity.
      * exact Hsf.
      * exact Hbf.
      * exact H.
    (* EReturn er *)
    + destruct (eval vary_val fuel' t rho er) as [[[v | v] tr0] |] eqn:Her.
      * inversion H. subst. exact (IHfuel t rho er (ONorm v) tr Hsf Hbf Her).
      * inversion H. subst. exact (IHfuel t rho er (ORet v) tr Hsf Hbf Her).
      * discriminate.
Qed.

(** diverged_clean_no_barriers (T3-S3 corollary): if check Diverged e = [] and e
    is superstep_free, any completed evaluation emits no EvBarrier events.
    Follows directly from barrier_free_no_barriers via diverged_clean_iff_barrier_free. *)
Corollary diverged_clean_no_barriers :
  forall vary_val fuel t rho e o tr,
    superstep_free e = true ->
    check Diverged e = [] ->
    eval vary_val fuel t rho e = Some (o, tr) ->
    no_barrier_event tr = true.
Proof.
  intros vary_val fuel t rho e o tr Hsf Hclean Heval.
  apply barrier_free_no_barriers with (vary_val := vary_val) (e := e) (o := o) (fuel := fuel) (t := t) (rho := rho).
  - exact Hsf.
  - apply diverged_clean_iff_barrier_free. exact Hclean.
  - exact Heval.
Qed.
