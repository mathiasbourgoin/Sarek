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

(* ===== 9. T3-S4 — Core semantic soundness of check_env ===== *)

(*
 * T3-S4 design notes:
 *
 * 1. core_frag excludes ESuperstep and EReturn from the fragment. This makes
 *    superstep_free automatic and avoids the early-exit complication.
 *
 * 2. barrier_safe is stated using erase_warp trace equality rather than full
 *    trace equality. The T3-S3 design note explicitly states that EWarpPoint
 *    may emit [EvWarp] and is intentionally not forced to silence. In the
 *    varying-condition EIf case, two threads may take different branches and
 *    emit different EvWarp events (e.g., one branch contains EWarpPoint, the
 *    other does not). Barrier safety only requires that barrier events are
 *    uniform across threads; warp events are independent per-thread signals.
 *    erase_warp filters traces to EvBarrier-only, giving the correct predicate.
 *
 * 3. The bridge lemma check_env_diverged_clean_barrier_free is the env-threaded
 *    analogue of diverged_clean_iff_barrier_free. It is proved by expr_list_rect.
 *    Key: in Diverged mode, check_env propagates Diverged into all branches
 *    (diverged_absorbing applies via boolean), so the EIf/EWhile/EFor cases
 *    reduce to IH on all sub-expressions.
 *
 * 4. check_env_nonvarying_uniform is the bridge for the non-varying-condition
 *    EIf/EWhile/EFor cases. It proves that if is_varying_in_env env e = false
 *    and check_env Converged env e = [] then eval is uniform across threads.
 *    This requires induction on fuel (same structure as not_varying_uniform)
 *    with careful env_agrees propagation for the ELet case.
 *
 * 5. The main theorem check_env_sound_core is proved by induction on fuel.
 *    The simultaneous fuel+expr technique for EWhile (cited in PLAN as "new")
 *    is handled by the IH on EWhile itself at fuel' (the recursive eval call
 *    always uses fuel', so the fuel IH covers it without a secondary induction).
 *)

(* ----------------------------------------------------------------------- *)
(* 9.1  core_frag — fragment excluding ESuperstep and EReturn               *)
(* ----------------------------------------------------------------------- *)

Fixpoint core_frag (e : expr) : bool :=
  match e with
  | ELit | EVary | EBarrier | EWarpPoint | EVar _ => true
  | EBinop a b        => core_frag a && core_frag b
  | EUnop e0          => core_frag e0
  | EIf c t el        => core_frag c && core_frag t && core_frag el
  | EWhile c b        => core_frag c && core_frag b
  | EFor lo hi b      => core_frag lo && core_frag hi && core_frag b
  | ESeq es           => forallb core_frag es
  | ELet _ v b        => core_frag v && core_frag b
  | ESuperstep _ _ _  => false   (* implicit barrier at boundary; excluded *)
  | EApp args         => forallb core_frag args
  | EReturn _         => false   (* early-exit bypasses later barriers; excluded *)
  end.

(* ----------------------------------------------------------------------- *)
(* 9.2  erase_warp — project trace to barrier events only                  *)
(* ----------------------------------------------------------------------- *)

(** erase_warp tr: keep only EvBarrier events, discard EvWarp.
    barrier_safe uses erase_warp equality: all threads cross barriers in the
    same order, but warp-collective events may differ per thread. *)
Definition erase_warp (tr : trace) : trace :=
  filter (fun ev => match ev with EvBarrier => true | EvWarp => false end) tr.

Lemma erase_warp_app : forall tr1 tr2,
  erase_warp (tr1 ++ tr2) = erase_warp tr1 ++ erase_warp tr2.
Proof.
  intros tr1 tr2. unfold erase_warp. apply filter_app.
Qed.

Lemma erase_warp_no_barrier : forall tr,
  no_barrier_event tr = true -> erase_warp tr = [].
Proof.
  intros tr H.
  unfold erase_warp, no_barrier_event in *.
  induction tr as [| ev tr' IHtr].
  - reflexivity.
  - simpl in H. apply andb_true_iff in H as [Hev Htr'].
    simpl. destruct ev.
    + (* EvBarrier *) simpl in Hev. discriminate.
    + (* EvWarp *) exact (IHtr Htr').
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.3  barrier_safe — barrier-trace uniformity property                    *)
(* ----------------------------------------------------------------------- *)

(** barrier_safe vary_val env e: any two env-agreeing threads that complete
    evaluation of e produce traces with the same barrier-event sequence.
    (Warp events may differ — see design note 2 above.) *)
(** barrier_safe vary_val env e: any two env-agreeing threads that complete
    evaluation of e produce traces with the same barrier-event sequence. *)
Definition barrier_safe (vary_val : tid -> value) (env : Env) (e : expr) : Prop :=
  forall fuel t1 t2 rho1 rho2 o1 o2 tr1 tr2,
    env_agrees env rho1 rho2 ->
    eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
    eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
    erase_warp tr1 = erase_warp tr2.

(* ----------------------------------------------------------------------- *)
(* 9.4  core_frag implies superstep_free                                    *)
(* ----------------------------------------------------------------------- *)

Lemma core_frag_impl_superstep_free :
  forall e, core_frag e = true -> superstep_free e = true.
Proof.
  apply (expr_list_rect
    (fun e  => core_frag e = true -> superstep_free e = true)
    (fun es => forallb core_frag es = true -> forallb superstep_free es = true)).
  (* ELit *)       - intros _. reflexivity.
  (* EVary *)      - intros _. reflexivity.
  (* EBarrier *)   - intros _. reflexivity.
  (* EWarpPoint *) - intros _. reflexivity.
  (* EVar x *)     - intros x _. reflexivity.
  (* EBinop a b *)
  - intros a b IHa IHb Hcf.
    simpl in Hcf. apply andb_true_iff in Hcf as [Ha Hb].
    simpl. apply andb_true_iff. exact (conj (IHa Ha) (IHb Hb)).
  (* EUnop e0 *)
  - intros e0 IHe0 Hcf. simpl in Hcf. simpl. exact (IHe0 Hcf).
  (* EIf c t el *)
  - intros c t el IHc IHt IHel Hcf.
    simpl in Hcf.
    apply andb_true_iff in Hcf as [Hct Hel].
    apply andb_true_iff in Hct as [Hc Ht].
    simpl.
    apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHc Hc) (IHt Ht)).
    + exact (IHel Hel).
  (* EWhile c b *)
  - intros c b IHc IHb Hcf.
    simpl in Hcf. apply andb_true_iff in Hcf as [Hc Hb].
    simpl. apply andb_true_iff. exact (conj (IHc Hc) (IHb Hb)).
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb Hcf.
    simpl in Hcf.
    apply andb_true_iff in Hcf as [Hlohib Hb].
    apply andb_true_iff in Hlohib as [Hlo Hhi].
    simpl.
    apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHlo Hlo) (IHhi Hhi)).
    + exact (IHb Hb).
  (* ESeq es *)
  - intros es IHes Hcf. simpl in Hcf. simpl. exact (IHes Hcf).
  (* ELet x v b *)
  - intros x v b IHv IHb Hcf.
    simpl in Hcf. apply andb_true_iff in Hcf as [Hv Hb].
    simpl. apply andb_true_iff. exact (conj (IHv Hv) (IHb Hb)).
  (* ESuperstep: core_frag = false *)
  - intros dv body cont _ _ Hcf. simpl in Hcf. discriminate.
  (* EApp args *)
  - intros args IHargs Hcf. simpl in Hcf. simpl. exact (IHargs Hcf).
  (* EReturn: core_frag = false *)
  - intros e0 _ Hcf. simpl in Hcf. discriminate.
  (* Plist [] *)  - intros _. reflexivity.
  (* Plist (h :: tl) *)
  - intros h tl IHh IHtl Hcf.
    simpl in Hcf. apply andb_true_iff in Hcf as [Hh Htl].
    simpl. apply andb_true_iff. exact (conj (IHh Hh) (IHtl Htl)).
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.5  Bridge: check_env Diverged implies barrier_free                     *)
(* ----------------------------------------------------------------------- *)

(** check_env_diverged_clean_barrier_free: env-threaded analogue of
    diverged_clean_iff_barrier_free (which works for `check Diverged`).
    In Diverged mode, check_env flags EBarrier; ESuperstep false also flags.
    Because Diverged is absorbing for the inner mode (diverged_absorbing),
    all sub-expressions are also checked in Diverged mode.
    Proved by expr_list_rect over a universally quantified env so that the ELet
    and other env-extending cases can change the environment in the IH. *)
Lemma check_env_diverged_clean_barrier_free :
  forall e env, check_env Diverged env e = [] -> barrier_free e = true.
Proof.
  apply (expr_list_rect
    (fun e  => forall env, check_env Diverged env e = [] -> barrier_free e = true)
    (fun es => forall env, concat (map (check_env Diverged env) es) = [] ->
               forallb barrier_free es = true)).
  (* ELit *)       - intros env _. reflexivity.
  (* EVary *)      - intros env _. reflexivity.
  (* EBarrier: check_env Diverged _ EBarrier = [BarrierError] ≠ [] *)
  - intros env H. simpl in H. discriminate.
  (* EWarpPoint *)
  - intros env _. reflexivity.
  (* EVar x *)
  - intros x env _. reflexivity.
  (* EBinop a b *)
  - intros a b IHa IHb env H.
    simpl in H. apply app_eq_nil in H as [Ha Hb].
    simpl. apply andb_true_iff. exact (conj (IHa env Ha) (IHb env Hb)).
  (* EUnop e0 *)
  - intros e0 IHe0 env H. simpl in H. simpl. exact (IHe0 env H).
  (* EIf cond t el: Diverged mode is absorbing *)
  - intros cond t el IHcond IHt IHel env H.
    simpl in H.
    rewrite diverged_absorbing in H.
    apply app_eq_nil in H as [Hcond H'].
    apply app_eq_nil in H' as [Ht Hel_clean].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHcond env Hcond) (IHt env Ht)).
    + exact (IHel env Hel_clean).
  (* EWhile cond b *)
  - intros cond b IHcond IHb env H.
    simpl in H.
    rewrite diverged_absorbing in H.
    apply app_eq_nil in H as [Hcond Hb].
    simpl. apply andb_true_iff. exact (conj (IHcond env Hcond) (IHb env Hb)).
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb env H.
    simpl in H.
    rewrite diverged_absorbing in H.
    apply app_eq_nil in H as [Hlo H'].
    apply app_eq_nil in H' as [Hhi Hb].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHlo env Hlo) (IHhi env Hhi)).
    + exact (IHb env Hb).
  (* ESeq es: Plist IH *)
  - intros es IHes env H. simpl in H. simpl. exact (IHes env H).
  (* ELet x v b: body checked with env_extend env x vv *)
  - intros x v b IHv IHb env H.
    simpl in H. apply app_eq_nil in H as [Hv Hb].
    simpl. apply andb_true_iff.
    split.
    + exact (IHv env Hv).
    + (* IHb : forall env', check_env Diverged env' b = [] -> barrier_free b = true *)
      exact (IHb (env_extend env x (is_varying_in_env env v)) Hb).
  (* ESuperstep *)
  - intros dv body cont IHbody IHcont env H.
    simpl in H.
    destruct dv.
    + simpl in H. apply app_eq_nil in H as [Hbody Hcont].
      simpl.
      apply andb_true_iff. exact (conj (IHbody env Hbody) (IHcont env Hcont)).
    + simpl in H. discriminate.
  (* EApp args: Plist IH *)
  - intros args IHargs env H. simpl in H. simpl. exact (IHargs env H).
  (* EReturn e0 *)
  - intros e0 IHe0 env H. simpl in H. simpl. exact (IHe0 env H).
  (* Plist [] *)
  - intros env _. reflexivity.
  (* Plist (h :: tl) *)
  - intros h tl IHh IHtl env H.
    simpl in H. apply app_eq_nil in H as [Hh Htl].
    simpl. apply andb_true_iff. exact (conj (IHh env Hh) (IHtl env Htl)).
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.6  Diverged-mode barrier silence for check_env                        *)
(* ----------------------------------------------------------------------- *)

(** check_env_diverged_no_barriers: if core_frag e = true and
    check_env Diverged env e = [] then any completed evaluation emits no
    EvBarrier events.
    Follows from the bridge lemma (→ barrier_free) and T3-S3 (barrier_free +
    superstep_free → no EvBarrier), with superstep_free from core_frag. *)
Lemma check_env_diverged_no_barriers :
  forall vary_val env e,
    core_frag e = true ->
    check_env Diverged env e = [] ->
    forall fuel t rho o tr,
      eval vary_val fuel t rho e = Some (o, tr) ->
      no_barrier_event tr = true.
Proof.
  intros vary_val env e Hcf Hclean fuel t rho o tr Heval.
  apply barrier_free_no_barriers with
    (vary_val := vary_val) (fuel := fuel) (t := t) (rho := rho) (e := e) (o := o).
  - exact (core_frag_impl_superstep_free e Hcf).
  - exact (check_env_diverged_clean_barrier_free e env Hclean).
  - exact Heval.
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.7  Helper: check_env_nonvarying_uniform                                *)
(* ----------------------------------------------------------------------- *)

(** Auxiliary: when check_env Converged env es = [] and forallb
    (is_varying_in_env env) es = false, the eval_seq loop is uniform. *)
Lemma check_env_nonvarying_uniform_seq :
  forall vary_val (env : Env) (es : list expr) fuel t1 t2 rho1 rho2,
    (forall e0, In e0 es ->
       core_frag e0 = true ->
       is_varying_in_env env e0 = false ->
       check_env Converged env e0 = [] ->
       env_agrees env rho1 rho2 ->
       eval vary_val fuel t1 rho1 e0 = eval vary_val fuel t2 rho2 e0) ->
    forallb core_frag es = true ->
    existsb (is_varying_in_env env) es = false ->
    concat (map (check_env Converged env) es) = [] ->
    env_agrees env rho1 rho2 ->
    forall acc,
      (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
        match xs with
        | []      => Some (ONorm 0, acc_tr)
        | x :: rest =>
            match eval vary_val fuel t1 rho1 x with
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
            match eval vary_val fuel t2 rho2 x with
            | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
            | Some (ONorm _, tr) => eval_seq rest (acc_tr ++ tr)
            | None               => None
            end
        end) es acc.
Proof.
  intros vary_val env es fuel t1 t2 rho1 rho2 IHes Hcf Hvar Hclean Hagr.
  induction es as [| h tl IHtl].
  - intros acc. reflexivity.
  - intros acc.
    simpl in Hcf. apply andb_true_iff in Hcf as [Hcfh Hcftl].
    simpl in Hvar. apply Bool.orb_false_iff in Hvar as [Hvh Hvtl].
    simpl in Hclean. apply app_eq_nil in Hclean as [Hch Hctl].
    simpl.
    rewrite (IHes h (in_eq h tl) Hcfh Hvh Hch Hagr).
    destruct (eval vary_val fuel t2 rho2 h) as [[[v|v] tr]|]; try reflexivity.
    apply IHtl.
    + intros e0 Hin Hcfe0 Hve Hce Hagr'. apply IHes.
      * apply in_cons. exact Hin.
      * exact Hcfe0.
      * exact Hve.
      * exact Hce.
      * exact Hagr'.
    + exact Hcftl.
    + exact Hvtl.
    + exact Hctl.
Qed.

Lemma check_env_nonvarying_uniform_args :
  forall vary_val (env : Env) (es : list expr) fuel t1 t2 rho1 rho2,
    (forall e0, In e0 es ->
       core_frag e0 = true ->
       is_varying_in_env env e0 = false ->
       check_env Converged env e0 = [] ->
       env_agrees env rho1 rho2 ->
       eval vary_val fuel t1 rho1 e0 = eval vary_val fuel t2 rho2 e0) ->
    forallb core_frag es = true ->
    existsb (is_varying_in_env env) es = false ->
    concat (map (check_env Converged env) es) = [] ->
    env_agrees env rho1 rho2 ->
    forall acc last_v,
      (fix eval_args (xs : list expr) (acc_tr : trace) (lv : value) : option (outcome * trace) :=
        match xs with
        | []      => Some (ONorm lv, acc_tr)
        | x :: rest =>
            match eval vary_val fuel t1 rho1 x with
            | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
            | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
            | None               => None
            end
        end) es acc last_v
      =
      (fix eval_args (xs : list expr) (acc_tr : trace) (lv : value) : option (outcome * trace) :=
        match xs with
        | []      => Some (ONorm lv, acc_tr)
        | x :: rest =>
            match eval vary_val fuel t2 rho2 x with
            | Some (ORet v, tr)  => Some (ORet v, acc_tr ++ tr)
            | Some (ONorm v, tr) => eval_args rest (acc_tr ++ tr) v
            | None               => None
            end
        end) es acc last_v.
Proof.
  intros vary_val env es fuel t1 t2 rho1 rho2 IHes Hcf Hvar Hclean Hagr.
  induction es as [| h tl IHtl].
  - intros acc last_v. reflexivity.
  - intros acc last_v.
    simpl in Hcf. apply andb_true_iff in Hcf as [Hcfh Hcftl].
    simpl in Hvar. apply Bool.orb_false_iff in Hvar as [Hvh Hvtl].
    simpl in Hclean. apply app_eq_nil in Hclean as [Hch Hctl].
    simpl.
    rewrite (IHes h (in_eq h tl) Hcfh Hvh Hch Hagr).
    destruct (eval vary_val fuel t2 rho2 h) as [[[v|v] tr]|]; try reflexivity.
    apply IHtl.
    + intros e0 Hin Hcfe0 Hve Hce Hagr'. apply IHes.
      * apply in_cons. exact Hin.
      * exact Hcfe0.
      * exact Hve.
      * exact Hce.
      * exact Hagr'.
    + exact Hcftl.
    + exact Hvtl.
    + exact Hctl.
Qed.

(** core_frag_no_ret: if core_frag e = true, eval never returns ORet.
    EReturn is the only constructor that produces ORet in eval, and core_frag excludes it.
    Proof by induction on fuel (with all e universally quantified so sub-expression IH works). *)
Lemma core_frag_no_ret :
  forall vary_val fuel,
  forall e t rho v tr,
    core_frag e = true ->
    eval vary_val fuel t rho e = Some (ORet v, tr) -> False.
Proof.
  intros vary_val.
  induction fuel as [| f' IHf].
  - intros e t rho v tr _ H. simpl in H. discriminate.
  - intros e t rho v tr Hcf H.
    destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                  | elo ehi ebod | ess | xn ev eb | dv ebod econt | eargs | er ];
    simpl in Hcf; simpl in H; try discriminate.
    (* EBinop ea eb *)
    + apply andb_true_iff in Hcf as [Hcfa Hcfb].
      set (rea := eval vary_val f' t rho ea) in *.
      destruct rea as [[[va|va] tra_a]|] eqn:Hea.
      { (* ONorm va: check eb *)
        set (reb := eval vary_val f' t rho eb) in *.
        destruct reb as [[[vb|vb] trb]|] eqn:Heb.
        - (* ONorm vb: result is ONorm, H says ORet, contradiction *)
          discriminate.
        - (* ORet vb: result is ORet vb *)
          inversion H. subst.
          unfold reb in Heb.
          exact (IHf eb t rho v trb Hcfb Heb).
        - (* None: H = None = Some ORet, contradiction *)
          discriminate. }
      { (* ORet va: result of EBinop is Some (ORet va, tra_a) *)
        inversion H. subst.
        unfold rea in Hea.
        exact (IHf ea t rho v tr Hcfa Hea). }
      { (* None: H = None = Some ORet, contradiction *)
        discriminate. }
    (* EUnop eu *)
    + set (reu := eval vary_val f' t rho eu) in *.
      destruct reu as [[[vu|vu] tru]|] eqn:Heu; try discriminate.
      inversion H. subst. unfold reu in Heu. exact (IHf eu t rho v tr Hcf Heu).
    (* EIf econd ethen eelse *)
    + apply andb_true_iff in Hcf as [Hcfct Hcfel].
      apply andb_true_iff in Hcfct as [Hcfc Hcft].
      set (rcond := eval vary_val f' t rho econd) in *.
      destruct rcond as [[[vc|vc] trc]|] eqn:Hcond; try discriminate.
      { (* ONorm vc: evaluate branch *)
        set (rbr := eval vary_val f' t rho (if Nat.eqb vc 0 then eelse else ethen)) in *.
        destruct rbr as [[ob trb]|] eqn:Hbr; try discriminate.
        inversion H. subst.
        unfold rbr in Hbr.
        case_eq (Nat.eqb vc 0); intro Hb; rewrite Hb in Hbr.
        - exact (IHf eelse t rho v trb Hcfel Hbr).
        - exact (IHf ethen t rho v trb Hcft Hbr). }
      { (* ORet vc: propagates *)
        inversion H. subst. unfold rcond in Hcond. exact (IHf econd t rho v tr Hcfc Hcond). }
    (* EWhile ec eb *)
    + apply andb_true_iff in Hcf as [Hcfc Hcfb].
      set (rcond_w := eval vary_val f' t rho ec) in *.
      destruct rcond_w as [[[vc|vc] trc]|] eqn:Hcond; try discriminate.
      { destruct (Nat.eqb vc 0); try discriminate.
        set (rbod_w := eval vary_val f' t rho eb) in *.
        destruct rbod_w as [[[vb|vb] trb]|] eqn:Hbod; try discriminate.
        { (* ONorm vb: recursive EWhile call at fuel f' *)
          set (rloop := eval vary_val f' t rho (EWhile ec eb)) in *.
          destruct rloop as [[oloop trloop]|] eqn:Hloop; try discriminate.
          inversion H. subst.
          assert (HcfW : core_frag (EWhile ec eb) = true).
          { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
          unfold rloop in Hloop.
          exact (IHf (EWhile ec eb) t rho v trloop HcfW Hloop). }
        { (* ORet vb: propagates from body *)
          inversion H. subst. unfold rbod_w in Hbod. exact (IHf eb t rho v trb Hcfb Hbod). } }
      { inversion H. subst. unfold rcond_w in Hcond. exact (IHf ec t rho v tr Hcfc Hcond). }
    (* EFor elo ehi ebod *)
    + apply andb_true_iff in Hcf as [Hcflohi Hcfb].
      apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
      set (rlo := eval vary_val f' t rho elo) in *.
      destruct rlo as [[[vlo|vlo] trlo]|] eqn:Hlo; try discriminate.
      { set (rhi := eval vary_val f' t rho ehi) in *.
        destruct rhi as [[[vhi|vhi] trhi]|] eqn:Hhi; try discriminate.
        { destruct (Nat.leb vhi vlo); try discriminate.
          rewrite (for_loop_eq vary_val f' t rho ebod (vhi - vlo) (trlo ++ trhi)) in H.
          (* Induct on steps, generalize acc *)
          revert H.
          generalize (trlo ++ trhi) as acc0.
          induction (vhi - vlo) as [| s IHs]; intros acc0 H.
          - simpl in H. discriminate.
          - simpl in H.
            set (rbod := eval vary_val f' t rho ebod) in *.
            destruct rbod as [[[vb|vb] trb]|] eqn:Hbod; try discriminate.
            + exact (IHs _ H).
            + inversion H. subst. unfold rbod in Hbod. exact (IHf ebod t rho v trb Hcfb Hbod). }
        { inversion H. subst. unfold rhi in Hhi. exact (IHf ehi t rho v trhi Hcfhi Hhi). } }
      { inversion H. subst. unfold rlo in Hlo. exact (IHf elo t rho v tr Hcflo Hlo). }
    (* ESeq ess *)
    + rewrite forallb_forall in Hcf.
      rename H into Hseq_outer.
      assert (Hseq : forall acc,
        (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
          match xs with
          | []      => Some (ONorm 0, acc_tr)
          | x :: rest =>
              match eval vary_val f' t rho x with
              | Some (ORet v0, tr0)  => Some (ORet v0, acc_tr ++ tr0)
              | Some (ONorm _, tr0) => eval_seq rest (acc_tr ++ tr0)
              | None               => None
              end
          end) ess acc = Some (ORet v, tr) -> False).
      { clear Hseq_outer.
        induction ess as [| h0 tl0 IHtl0].
        - intros acc H0. simpl in H0. discriminate.
        - intros acc H0. simpl in H0.
          assert (Hcfh0 : core_frag h0 = true) by (apply Hcf; apply in_eq).
          destruct (eval vary_val f' t rho h0) as [[[vh|vh] trh]|] eqn:Hh0; try discriminate.
          + exact (IHtl0 (fun x Hin => Hcf x (in_cons _ _ _ Hin)) (acc ++ trh) H0).
          + inversion H0. subst. exact (IHf h0 t rho v trh Hcfh0 Hh0). }
      exact (Hseq [] Hseq_outer).
    (* ELet xn ev eb *)
    + apply andb_true_iff in Hcf as [Hcfv Hcfb].
      set (rev_let := eval vary_val f' t rho ev) in *.
      destruct rev_let as [[[vv|vv] trv]|] eqn:Hev; try discriminate.
      { (* ONorm vv: evaluate body with extended env *)
        set (rbod_let := eval vary_val f' t (venv_extend rho xn vv) eb) in *.
        destruct rbod_let as [[ob'' trb'']|] eqn:Hbod; try discriminate.
        inversion H. subst.
        unfold rbod_let in Hbod.
        exact (IHf eb t (venv_extend rho xn vv) v trb'' Hcfb Hbod). }
      { (* ORet vv: propagates from binder *)
        inversion H. subst. unfold rev_let in Hev. exact (IHf ev t rho v tr Hcfv Hev). }
    (* ESuperstep: excluded by core_frag = false (try discriminate closed it) *)
    (* EApp eargs *)
    + rewrite forallb_forall in Hcf.
      rename H into Hargs_outer.
      assert (Hargs : forall acc lv,
        (fix eval_args (xs : list expr) (acc_tr : trace) (last_v : value) : option (outcome * trace) :=
          match xs with
          | []      => Some (ONorm last_v, acc_tr)
          | x :: rest =>
              match eval vary_val f' t rho x with
              | Some (ORet v0, tr0)  => Some (ORet v0, acc_tr ++ tr0)
              | Some (ONorm v0, tr0) => eval_args rest (acc_tr ++ tr0) v0
              | None               => None
              end
          end) eargs acc lv = Some (ORet v, tr) -> False).
      { clear Hargs_outer.
        induction eargs as [| h0 tl0 IHtl0].
        - intros acc lv H0. simpl in H0. discriminate.
        - intros acc lv H0. simpl in H0.
          assert (Hcfh0 : core_frag h0 = true) by (apply Hcf; apply in_eq).
          destruct (eval vary_val f' t rho h0) as [[[vh|vh] trh]|] eqn:Hh0; try discriminate.
          + exact (IHtl0 (fun x Hin => Hcf x (in_cons _ _ _ Hin)) (acc ++ trh) vh H0).
          + inversion H0. subst. exact (IHf h0 t rho v trh Hcfh0 Hh0). }
      exact (Hargs [] 0 Hargs_outer).
    (* EReturn: excluded by core_frag = false (try discriminate closed it) *)
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.7  eval_while_exits_immediately                                        *)
(* ----------------------------------------------------------------------- *)

(** eval_while_exits_immediately: if a core_frag EWhile terminates, the
    condition must have evaluated to 0 on the first iteration.  Because
    eval uses the same fixed rho at every recursive call, a non-zero
    condition value would force the loop to diverge at any finite fuel. *)
Lemma eval_while_exits_immediately :
  forall vary_val fuel ec eb t rho o tr,
    core_frag (EWhile ec eb) = true ->
    eval vary_val fuel t rho (EWhile ec eb) = Some (o, tr) ->
    eval vary_val (Nat.pred fuel) t rho ec = Some (ONorm 0, tr) /\ o = ONorm 0.
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - intros ec eb t rho o tr _ H. simpl in H. discriminate.
  - intros ec eb t rho o tr Hcf H.
    apply andb_true_iff in Hcf as [Hcfc Hcfb].
    simpl in H.
    (* outcome = ONorm | ORet; first branch = ONorm, second = ORet *)
    destruct (eval vary_val fuel' t rho ec) as [[[cv|cv] tr_c]|] eqn:Hec.
    + (* ec → ONorm cv: check if cv = 0 *)
      destruct (Nat.eqb cv 0) eqn:Hcv0.
      * apply Nat.eqb_eq in Hcv0. subst cv.
        inversion H. subst. simpl. exact (conj Hec eq_refl).
      * apply Nat.eqb_neq in Hcv0.
        (* cv ≠ 0: body must return Some for the whole thing to be Some *)
        destruct (eval vary_val fuel' t rho eb) as [[[bv|bv] tr_b]|] eqn:Heb.
        -- (* body → ONorm bv: recursive EWhile call *)
           destruct (eval vary_val fuel' t rho (EWhile ec eb)) as [[oloop trloop]|] eqn:Hloop.
           ++ (* recursive call → Some: derive contradiction via IH *)
              assert (Hwcf : core_frag (EWhile ec eb) = true).
              { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
              destruct (IHfuel ec eb t rho oloop trloop Hwcf Hloop) as [Hec0 _].
              (* Hec0 : eval (pred fuel') t rho ec = Some(ONorm 0, trloop) *)
              destruct fuel' as [|fuel''].
              { simpl in Hloop. discriminate. }
              (* fuel' = S fuel'': pred (S fuel'') = fuel'' *)
              simpl in Hec0.
              apply eval_fuel_monotone in Hec0.
              rewrite Hec in Hec0.
              exfalso. apply Hcv0. congruence.
           ++ discriminate.
        -- (* body → ORet: excluded by core_frag *)
           exfalso. exact (core_frag_no_ret vary_val fuel' eb t rho bv tr_b Hcfb Heb).
        -- discriminate.
    + (* ec → ORet: excluded by core_frag *)
      exfalso. exact (core_frag_no_ret vary_val fuel' ec t rho cv tr_c Hcfc Hec).
    + discriminate.
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.8  for_loop_fixed erase_warp helpers                                   *)
(* ----------------------------------------------------------------------- *)

(** for_loop_erase_body_nil: if the body trace has no barrier events, the
    for_loop_fixed trace has the same erase_warp as the initial accumulator.
    The accumulator is universally quantified so the inductive step can
    shift it from acc to acc ++ trb. *)
Lemma for_loop_erase_body_nil :
  forall (bv : value) (trb : trace) (steps : nat),
    erase_warp trb = [] ->
    forall (acc : trace),
    match for_loop_fixed (Some (ONorm bv, trb)) steps acc with
    | Some (_, tr) => erase_warp tr = erase_warp acc
    | None => True
    end.
Proof.
  intros bv trb steps Htrb.
  induction steps as [| k' IHk]; intro acc.
  - simpl. reflexivity.
  - simpl. (* goal: match for_loop_fixed body k' (acc ++ trb) with ... end *)
    specialize (IHk (acc ++ trb)).
    destruct (for_loop_fixed (Some (ONorm bv, trb)) k' (acc ++ trb))
      as [[o' tr']|] eqn:Hfl.
    + rewrite IHk. rewrite erase_warp_app. rewrite Htrb. apply app_nil_r.
    + exact I.
Qed.

(** for_loop_erase_body_eq: when two ONorm bodies have equal erase_warp traces
    and the initial accumulators have equal erase_warp, the resulting
    for_loop_fixed outputs have equal erase_warp traces (for the same steps).
    Both accumulators are universally quantified so the inductive step can
    shift them simultaneously. *)
Lemma for_loop_erase_body_eq :
  forall (bv1 bv2 : value) (trb1 trb2 : trace) (steps : nat),
    erase_warp trb1 = erase_warp trb2 ->
    forall (acc1 acc2 : trace),
    erase_warp acc1 = erase_warp acc2 ->
    match for_loop_fixed (Some (ONorm bv1, trb1)) steps acc1,
          for_loop_fixed (Some (ONorm bv2, trb2)) steps acc2 with
    | Some (_, tr1), Some (_, tr2) => erase_warp tr1 = erase_warp tr2
    | None, None => True
    | _, _ => True
    end.
Proof.
  intros bv1 bv2 trb1 trb2 steps Htrb.
  induction steps as [| k' IHk]; intros acc1 acc2 Hacc.
  - simpl. exact Hacc.
  - simpl.
    apply IHk.
    rewrite !erase_warp_app. congruence.
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.9  eval_check_uniform — combined barrier and outcome uniformity        *)
(* ----------------------------------------------------------------------- *)

(** eval_check_uniform: combined induction on fuel establishing:
    Part A — for any core_frag expression with a clean Converged check,
      any two env-agreeing threads that complete evaluation produce the same
      barrier-event sequence (erase_warp tr1 = erase_warp tr2).
    Part B — additionally, if the expression is non-varying, the outcomes
      also agree (o1 = o2).
    These two parts are proved simultaneously by induction on fuel, since
    the ELet-varying case of Part A needs Part A on the binder (possibly
    varying) and Part A on the body; the EWhile/EIf non-varying cases of
    Part A need Part B on the condition to establish that both threads take
    the same branch. *)
Lemma eval_check_uniform : forall vary_val fuel,
  (* Part A: barrier-trace equality for all clean core_frag expressions *)
  (forall env e t1 t2 rho1 rho2 o1 o2 tr1 tr2,
     core_frag e = true ->
     env_agrees env rho1 rho2 ->
     check_env Converged env e = [] ->
     eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
     eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
     erase_warp tr1 = erase_warp tr2) /\
  (* Part B: outcome equality for non-varying clean core_frag expressions *)
  (forall env e t1 t2 rho1 rho2 o1 o2 tr1 tr2,
     core_frag e = true ->
     env_agrees env rho1 rho2 ->
     is_varying_in_env env e = false ->
     check_env Converged env e = [] ->
     eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
     eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
     o1 = o2).
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - split.
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 _ _ _ H _. simpl in H. discriminate.
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 _ _ _ _ H _. simpl in H. discriminate.
  - destruct IHfuel as [IH_A IH_B].
    split.

(* ------------------------------------------------------------------ *)
(* Part A for S fuel'                                                   *)
(* ------------------------------------------------------------------ *)
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hcf Hagr Hclean Heval1 Heval2.
      destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                    | elo ehi ebod | ess | xn ev ebody | dv ebod econt | eargs | er ];
      simpl in Hcf; simpl in Heval1; simpl in Heval2; try discriminate.

      (* ELit *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVary *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EBarrier *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EWarpPoint *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVar n0 *)
      * inversion Heval1; inversion Heval2; reflexivity.

      (* EBinop ea eb *)
      * apply andb_true_iff in Hcf as [Hcfa Hcfb].
        simpl in Hclean. apply app_eq_nil in Hclean as [Hca Hcb].
        destruct (eval vary_val fuel' t1 rho1 ea) as [[[va1|va1] tra1]|] eqn:Hea1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t1 rho1 va1 tra1 Hcfa Hea1). }
        destruct (eval vary_val fuel' t1 rho1 eb) as [[[vb1|vb1] trb1]|] eqn:Heb1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 vb1 trb1 Hcfb Heb1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ea) as [[[va2|va2] tra2]|] eqn:Hea2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t2 rho2 va2 tra2 Hcfa Hea2). }
        destruct (eval vary_val fuel' t2 rho2 eb) as [[[vb2|vb2] trb2]|] eqn:Heb2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 vb2 trb2 Hcfb Heb2). }
        inversion Heval2; subst.
        rewrite !erase_warp_app. f_equal.
        { exact (IH_A env ea t1 t2 rho1 rho2 (ONorm va1) (ONorm va2) tra1 tra2
                   Hcfa Hagr Hca Hea1 Hea2). }
        { exact (IH_A env eb t1 t2 rho1 rho2 (ONorm vb1) (ONorm vb2) trb1 trb2
                   Hcfb Hagr Hcb Heb1 Heb2). }

      (* EUnop eu *)
      * destruct (eval vary_val fuel' t1 rho1 eu) as [[[vu1|vu1] tru1]|] eqn:Heu1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t1 rho1 vu1 tru1 Hcf Heu1). }
        destruct (eval vary_val fuel' t2 rho2 eu) as [[[vu2|vu2] tru2]|] eqn:Heu2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t2 rho2 vu2 tru2 Hcf Heu2). }
        (* Heval1 : Some (ONorm (unop_eval vu1), tru1) = Some (o1, tr1) *)
        (* Heval2 : Some (ONorm (unop_eval vu2), tru2) = Some (o2, tr2) *)
        injection Heval1 as Hout1 Htr1. injection Heval2 as Hout2 Htr2.
        simpl in Hclean. rewrite <- Htr1. rewrite <- Htr2.
        exact (IH_A env eu t1 t2 rho1 rho2 (ONorm vu1) (ONorm vu2) tru1 tru2
                 Hcf Hagr Hclean Heu1 Heu2).

      (* EIf econd ethen eelse *)
      * apply andb_true_iff in Hcf as [Hcfct Hcfel].
        apply andb_true_iff in Hcfct as [Hcfc Hcft].
        simpl in Hclean.
        set (inner_if := if is_varying_in_env env econd then Diverged else Converged).
        (* Split check_env *)
        assert (Hclean_c_br : check_env Converged env econd = [] /\
                               check_env inner_if env ethen = [] /\
                               check_env inner_if env eelse = []).
        { unfold inner_if.
          destruct (is_varying_in_env env econd);
          simpl in Hclean;
          apply app_eq_nil in Hclean as [HcC HtE];
          apply app_eq_nil in HtE as [HcT HcEl];
          exact (conj HcC (conj HcT HcEl)). }
        destruct Hclean_c_br as [Hcc [Hct Hcel]].
        (* Destruct eval of condition *)
        destruct (eval vary_val fuel' t1 rho1 econd) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 econd) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t2 rho2 cv2 trc2 Hcfc Hcond2). }
        (* Branch evaluation *)
        destruct (eval vary_val fuel' t1 rho1
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr1 trbr1]|] eqn:Hbr1;
          try discriminate.
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2
                   (if Nat.eqb cv2 0 then eelse else ethen)) as [[obr2 trbr2]|] eqn:Hbr2;
          try discriminate.
        inversion Heval2; subst.
        rewrite !erase_warp_app.
        f_equal.
        { exact (IH_A env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hcc Hcond1 Hcond2). }
        { (* Show branch traces have equal erase_warp *)
          destruct (is_varying_in_env env econd) eqn:Hcvar.
          - (* Condition varying: both branches in Diverged mode, no barriers *)
            unfold inner_if in Hct, Hcel.
            (* Thread 1's branch has no barriers *)
            assert (Hnb1 : no_barrier_event trbr1 = true).
            { destruct (Nat.eqb cv1 0) eqn:Hcv1z.
              - exact (check_env_diverged_no_barriers vary_val env eelse
                         Hcfel Hcel fuel' t1 rho1 o1 trbr1 Hbr1).
              - exact (check_env_diverged_no_barriers vary_val env ethen
                         Hcft Hct fuel' t1 rho1 o1 trbr1 Hbr1). }
            assert (Hnb2 : no_barrier_event trbr2 = true).
            { destruct (Nat.eqb cv2 0) eqn:Hcv2z.
              - exact (check_env_diverged_no_barriers vary_val env eelse
                         Hcfel Hcel fuel' t2 rho2 o2 trbr2 Hbr2).
              - exact (check_env_diverged_no_barriers vary_val env ethen
                         Hcft Hct fuel' t2 rho2 o2 trbr2 Hbr2). }
            rewrite (erase_warp_no_barrier _ Hnb1).
            rewrite (erase_warp_no_barrier _ Hnb2).
            reflexivity.
          - (* Condition non-varying: same branch taken *)
            unfold inner_if in Hct, Hcel.
            (* cv1 = cv2 from Part B *)
            assert (Hcveq : ONorm cv1 = ONorm cv2).
            { exact (IH_B env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                       Hcfc Hagr Hcvar Hcc Hcond1 Hcond2). }
            injection Hcveq as Hcveq'. subst cv2.
            (* Both take same branch *)
            destruct (Nat.eqb cv1 0) eqn:Hcv1z.
            + exact (IH_A env eelse t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                       Hcfel Hagr Hcel Hbr1 Hbr2).
            + exact (IH_A env ethen t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                       Hcft Hagr Hct Hbr1 Hbr2). }

      (* EWhile ec eb *)
      * apply andb_true_iff in Hcf as [Hcfc Hcfb].
        simpl in Hclean.
        set (inner_w := if is_varying_in_env env ec then Diverged else Converged).
        assert (Hclean_w : check_env Converged env ec = [] /\
                            check_env inner_w env eb = []).
        { unfold inner_w. destruct (is_varying_in_env env ec);
          simpl in Hclean; apply app_eq_nil in Hclean as [? ?]; auto. }
        destruct Hclean_w as [Hcc_w Hcb_w].
        destruct (is_varying_in_env env ec) eqn:Hcvar_w.
        -- (* Condition varying: use eval_while_exits_immediately *)
           assert (HwcfW : core_frag (EWhile ec eb) = true).
           { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
           destruct (eval_while_exits_immediately vary_val (S fuel') ec eb
                       t1 rho1 o1 tr1 HwcfW Heval1) as [Hec1 Ho1].
           destruct (eval_while_exits_immediately vary_val (S fuel') ec eb
                       t2 rho2 o2 tr2 HwcfW Heval2) as [Hec2 Ho2].
           (* Hec1 : eval fuel' t1 rho1 ec = Some(ONorm 0, tr1)
              Hec2 : eval fuel' t2 rho2 ec = Some(ONorm 0, tr2) *)
           simpl in Hec1, Hec2.
           exact (IH_A env ec t1 t2 rho1 rho2 (ONorm 0) (ONorm 0) tr1 tr2
                    Hcfc Hagr Hcc_w Hec1 Hec2).
        -- (* Condition non-varying: IH_B for condition *)
           unfold inner_w in Hcb_w.
           destruct (eval vary_val fuel' t1 rho1 ec) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t1 rho1 cv1 trc1 Hcfc Hcond1). }
           destruct (eval vary_val fuel' t2 rho2 ec) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t2 rho2 cv2 trc2 Hcfc Hcond2). }
           assert (Hcveq_w : ONorm cv1 = ONorm cv2).
           { exact (IH_B env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                      Hcfc Hagr Hcvar_w Hcc_w Hcond1 Hcond2). }
           injection Hcveq_w as Hcveq_w'. subst cv2.
           destruct (Nat.eqb cv1 0) eqn:Hcv1z.
           ++ (* cv = 0: exits immediately *)
              apply Nat.eqb_eq in Hcv1z. subst cv1.
              inversion Heval1; subst. inversion Heval2; subst.
              exact (IH_A env ec t1 t2 rho1 rho2 (ONorm 0) (ONorm 0) tr1 tr2
                       Hcfc Hagr Hcc_w Hcond1 Hcond2).
           ++ (* cv ≠ 0: loop continues *)
              destruct (eval vary_val fuel' t1 rho1 eb) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
                try discriminate.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 bv1 trb1 Hcfb Hbod1). }
              destruct (eval vary_val fuel' t1 rho1 (EWhile ec eb))
                as [[oloop1 trloop1]|] eqn:Hloop1; try discriminate.
              inversion Heval1; subst.
              destruct (eval vary_val fuel' t2 rho2 eb) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
                try discriminate.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 bv2 trb2 Hcfb Hbod2). }
              destruct (eval vary_val fuel' t2 rho2 (EWhile ec eb))
                as [[oloop2 trloop2]|] eqn:Hloop2; try discriminate.
              inversion Heval2; subst.
              assert (HwcfW : core_frag (EWhile ec eb) = true).
              { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
              assert (Hwclean : check_env Converged env (EWhile ec eb) = []).
              { simpl. rewrite Hcvar_w. simpl.
                rewrite Hcc_w. rewrite Hcb_w. reflexivity. }
              rewrite !erase_warp_app. f_equal.
              { exact (IH_A env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv1) trc1 trc2
                         Hcfc Hagr Hcc_w Hcond1 Hcond2). }
              f_equal.
              { exact (IH_A env eb t1 t2 rho1 rho2 (ONorm bv1) (ONorm bv2) trb1 trb2
                         Hcfb Hagr Hcb_w Hbod1 Hbod2). }
              { exact (IH_A env (EWhile ec eb) t1 t2 rho1 rho2 o1 o2
                         trloop1 trloop2 HwcfW Hagr Hwclean Hloop1 Hloop2). }

      (* EFor elo ehi ebod *)
      * apply andb_true_iff in Hcf as [Hcflohi Hcfb].
        apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
        simpl in Hclean.
        set (inner_f := if is_varying_in_env env elo || is_varying_in_env env ehi
                        then Diverged else Converged).
        assert (Hclean_f : check_env Converged env elo = [] /\
                            check_env Converged env ehi = [] /\
                            check_env inner_f env ebod = []).
        { unfold inner_f.
          destruct (is_varying_in_env env elo || is_varying_in_env env ehi);
          simpl in Hclean;
          apply app_eq_nil in Hclean as [HcL HtE];
          apply app_eq_nil in HtE as [HcH HcB];
          exact (conj HcL (conj HcH HcB)). }
        destruct Hclean_f as [Hclo [Hchi Hcbod]].
        (* Destruct lo and hi for both threads *)
        destruct (eval vary_val fuel' t1 rho1 elo) as [[[lv1|lv1] trlo1]|] eqn:Hlo1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t1 rho1 lv1 trlo1 Hcflo Hlo1). }
        destruct (eval vary_val fuel' t1 rho1 ehi) as [[[hv1|hv1] trhi1]|] eqn:Hhi1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t1 rho1 hv1 trhi1 Hcfhi Hhi1). }
        destruct (eval vary_val fuel' t2 rho2 elo) as [[[lv2|lv2] trlo2]|] eqn:Hlo2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t2 rho2 lv2 trlo2 Hcflo Hlo2). }
        destruct (eval vary_val fuel' t2 rho2 ehi) as [[[hv2|hv2] trhi2]|] eqn:Hhi2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t2 rho2 hv2 trhi2 Hcfhi Hhi2). }
        (* Rewrite using for_loop_eq *)
        destruct (Nat.leb hv1 lv1) eqn:Hlb1.
        { inversion Heval1; subst.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst.
            rewrite !erase_warp_app. f_equal.
            { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                       Hcflo Hagr Hclo Hlo1 Hlo2). }
            { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                       Hcfhi Hagr Hchi Hhi1 Hhi2). } }
          (* t1 exits, t2 loops: need to show erase_warp equal *)
          { rewrite (for_loop_eq vary_val fuel' t2 rho2 ebod (hv2 - lv2)
                       (trlo2 ++ trhi2)) in Heval2.
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - (* Varying: body in Diverged mode, no barriers *)
              unfold inner_f in Hcbod.
              (* Destruct body eval for t2 to apply for_loop_erase_body_nil *)
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv2 - lv2) as [|steps_pos] eqn:Hsteps2.
                   { apply Nat.leb_nle in Hlb2. lia. }
                   simpl in Heval2. discriminate. }
              (* ONorm bv2: body has no barriers *)
              assert (Htrb2 : erase_warp trb2 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers vary_val env ebod
                         Hcfb Hcbod fuel' t2 rho2 (ONorm bv2) trb2 Hbod2). }
              assert (Hnb2 := for_loop_erase_body_nil bv2 trb2 (hv2 - lv2) Htrb2 (trlo2 ++ trhi2)).
              rewrite Heval2 in Hnb2.
              (* Hnb2 : erase_warp tr2 = erase_warp (trlo2 ++ trhi2) *)
              inversion Heval1; subst.
              rewrite Hnb2. rewrite !erase_warp_app.
              f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - (* Non-varying lo/hi: same lv, hv from Part B *)
              assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              rewrite Hlb1 in Hlb2. discriminate. } }
        { (* t1 loops *)
          rewrite (for_loop_eq vary_val fuel' t1 rho1 ebod (hv1 - lv1)
                     (trlo1 ++ trhi1)) in Heval1.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { (* t2 exits: symmetric to above *)
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - unfold inner_f in Hcbod.
              (* Destruct body eval for t1 to apply for_loop_erase_body_nil *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              (* ONorm bv1: body has no barriers *)
              assert (Htrb1 : erase_warp trb1 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers vary_val env ebod
                         Hcfb Hcbod fuel' t1 rho1 (ONorm bv1) trb1 Hbod1). }
              assert (Hnb1 := for_loop_erase_body_nil bv1 trb1 (hv1 - lv1) Htrb1 (trlo1 ++ trhi1)).
              rewrite Heval1 in Hnb1.
              (* Hnb1 : erase_warp tr1 = erase_warp (trlo1 ++ trhi1) *)
              inversion Heval2; subst.
              rewrite Hnb1. rewrite !erase_warp_app.
              f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              rewrite Hlb2 in Hlb1. discriminate. }
          { (* Both loop *)
            rewrite (for_loop_eq vary_val fuel' t2 rho2 ebod (hv2 - lv2)
                       (trlo2 ++ trhi2)) in Heval2.
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - (* Varying: body in Diverged mode *)
              unfold inner_f in Hcbod.
              (* Destruct body eval for t1 *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              assert (Htrb1 : erase_warp trb1 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers vary_val env ebod
                         Hcfb Hcbod fuel' t1 rho1 (ONorm bv1) trb1 Hbod1). }
              assert (Hnb1 := for_loop_erase_body_nil bv1 trb1 (hv1 - lv1) Htrb1 (trlo1 ++ trhi1)).
              rewrite Heval1 in Hnb1.
              (* Hnb1 : erase_warp tr1 = erase_warp (trlo1 ++ trhi1) *)
              (* Destruct body eval for t2 *)
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv2 - lv2) as [|steps_pos] eqn:Hsteps2.
                   { apply Nat.leb_nle in Hlb2. lia. }
                   simpl in Heval2. discriminate. }
              assert (Htrb2 : erase_warp trb2 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers vary_val env ebod
                         Hcfb Hcbod fuel' t2 rho2 (ONorm bv2) trb2 Hbod2). }
              assert (Hnb2 := for_loop_erase_body_nil bv2 trb2 (hv2 - lv2) Htrb2 (trlo2 ++ trhi2)).
              rewrite Heval2 in Hnb2.
              (* Hnb2 : erase_warp tr2 = erase_warp (trlo2 ++ trhi2) *)
              rewrite Hnb1. rewrite Hnb2.
              rewrite !erase_warp_app. f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - (* Non-varying lo/hi: same steps, use for_loop_erase_eq *)
              apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
              unfold inner_f in Hcbod.
              assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              (* Same steps: hv1 - lv1 *)
              assert (Herase_acc : erase_warp (trlo1 ++ trhi1) = erase_warp (trlo2 ++ trhi2)).
              { rewrite !erase_warp_app. f_equal.
                { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv1) trlo1 trlo2
                           Hcflo Hagr Hclo Hlo1 Hlo2). }
                { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv1) trhi1 trhi2
                           Hcfhi Hagr Hchi Hhi1 Hhi2). } }
              (* Destruct body evaluations to apply for_loop_erase_body_eq *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval2. discriminate. }
              (* ONorm bv1, ONorm bv2: apply IH_A for body erase, then for_loop_erase_body_eq *)
              assert (Hbody_erase : erase_warp trb1 = erase_warp trb2).
              { exact (IH_A env ebod t1 t2 rho1 rho2 (ONorm bv1) (ONorm bv2) trb1 trb2
                           Hcfb Hagr Hcbod Hbod1 Hbod2). }
              assert (Hfl := for_loop_erase_body_eq bv1 bv2 trb1 trb2 (hv1 - lv1)
                               Hbody_erase (trlo1 ++ trhi1) (trlo2 ++ trhi2) Herase_acc).
              rewrite Heval1 in Hfl. rewrite Heval2 in Hfl. exact Hfl. } }

      (* ESeq ess *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag ess = true) by exact Hcf.
        clear Hcf.
        (* We prove erase_warp equality by induction on ess,
           generalizing over the accumulator. *)
        assert (Hgen : forall xs,
          forallb core_frag xs = true ->
          concat (map (check_env Converged env) xs) = [] ->
          forall acc1 acc2 o1' o2' tr1' tr2',
          erase_warp acc1 = erase_warp acc2 ->
          (fix eval_seq xs0 acc : option _ :=
            match xs0 with [] => Some (ONorm 0, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
              | None => None end end) xs acc1 = Some (o1', tr1') ->
          (fix eval_seq xs0 acc : option _ :=
            match xs0 with [] => Some (ONorm 0, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
              | None => None end end) xs acc2 = Some (o2', tr2') ->
          erase_warp tr1' = erase_warp tr2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ acc1 acc2 o1' o2' tr1' tr2' Hacc Hs1 Hs2.
            inversion Hs1; inversion Hs2; subst. exact Hacc.
          - intros Hcfal Hcln acc1 acc2 o1' o2' tr1' tr2' Hacc Hs1 Hs2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Hs1, Hs2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            apply IHtl with (acc1 := acc1 ++ trh1) (acc2 := acc2 ++ trh2)
              (o1' := o1') (o2' := o2').
            + exact Hcftl.
            + exact Hctl.
            + rewrite !erase_warp_app. f_equal.
              * exact Hacc.
              * exact (IH_A env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                         Hcfh Hagr Hch Hh1 Hh2).
            + exact Hs1.
            + exact Hs2. }
        exact (Hgen ess Hcfall Hclean [] [] o1 o2 tr1 tr2 eq_refl Heval1 Heval2).

      (* ELet xn ev ebody *)
      * apply andb_true_iff in Hcf as [Hcfv Hcfb].
        simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcv Hcb].
        (* Destruct eval of ev for both threads *)
        destruct (eval vary_val fuel' t1 rho1 ev) as [[[vv1|vv1] trv1]|] eqn:Hev1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t1 rho1 vv1 trv1 Hcfv Hev1). }
        destruct (eval vary_val fuel' t2 rho2 ev) as [[[vv2|vv2] trv2]|] eqn:Hev2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t2 rho2 vv2 trv2 Hcfv Hev2). }
        (* Destruct eval of body for both threads *)
        destruct (eval vary_val fuel' t1 (venv_extend rho1 xn vv1) ebody)
          as [[ob1 trb1]|] eqn:Hbod1; try discriminate.
        destruct (eval vary_val fuel' t2 (venv_extend rho2 xn vv2) ebody)
          as [[ob2 trb2]|] eqn:Hbod2; try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        rewrite !erase_warp_app. f_equal.
        { exact (IH_A env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                   Hcfv Hagr Hcv Hev1 Hev2). }
        { (* Build env_agrees for extended env *)
          set (vv := is_varying_in_env env ev).
          assert (Hagr' : env_agrees (env_extend env xn vv)
                            (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)).
          { unfold vv.
            destruct (is_varying_in_env env ev) eqn:Hvv.
            - (* ev varying: xn is varying, no constraint on vv1 vs vv2 *)
              intros y Hlookup.
              assert (Hxy : xn <> y).
              { intro Heq. subst. rewrite env_lookup_extend_same in Hlookup. discriminate. }
              rewrite (venv_lookup_extend_diff rho1 xn y vv1 Hxy).
              rewrite (venv_lookup_extend_diff rho2 xn y vv2 Hxy).
              apply Hagr.
              unfold env_lookup, env_extend in Hlookup |- *.
              simpl find in *.
              destruct (Nat.eqb xn y) eqn:Heqn.
              { apply Nat.eqb_eq in Heqn. subst. exfalso. exact (Hxy eq_refl). }
              { exact Hlookup. }
            - (* ev non-varying: vv1 = vv2 from IH_B *)
              assert (Hveq : ONorm vv1 = ONorm vv2).
              { exact (IH_B env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                         Hcfv Hagr Hvv Hcv Hev1 Hev2). }
              injection Hveq as Hveq'. subst vv2.
              exact (env_agrees_extend env rho1 rho2 xn vv1 false Hagr). }
          exact (IH_A (env_extend env xn vv) ebody t1 t2
                   (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)
                   o1 o2 trb1 trb2 Hcfb Hagr' Hcb Hbod1 Hbod2). }

      (* ESuperstep: core_frag = false, already handled by try discriminate *)

      (* EApp eargs *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag eargs = true) by exact Hcf.
        clear Hcf.
        assert (Hgen : forall xs,
          forallb core_frag xs = true ->
          concat (map (check_env Converged env) xs) = [] ->
          forall acc1 acc2 lv1 lv2 o1' o2' tr1' tr2',
          erase_warp acc1 = erase_warp acc2 ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc1 lv1 = Some (o1', tr1') ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc2 lv2 = Some (o2', tr2') ->
          erase_warp tr1' = erase_warp tr2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hacc Ha1 Ha2.
            inversion Ha1; inversion Ha2; subst. exact Hacc.
          - intros Hcfal Hcln acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hacc Ha1 Ha2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Ha1, Ha2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            apply IHtl with (acc1 := acc1 ++ trh1) (acc2 := acc2 ++ trh2)
              (lv1 := vh1) (lv2 := vh2) (o1' := o1') (o2' := o2').
            + exact Hcftl.
            + exact Hctl.
            + rewrite !erase_warp_app. f_equal.
              * exact Hacc.
              * exact (IH_A env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                         Hcfh Hagr Hch Hh1 Hh2).
            + exact Ha1.
            + exact Ha2. }
        exact (Hgen eargs Hcfall Hclean [] [] 0 0 o1 o2 tr1 tr2 eq_refl Heval1 Heval2).

(* ------------------------------------------------------------------ *)
(* Part B for S fuel'                                                   *)
(* ------------------------------------------------------------------ *)
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hcf Hagr Hvar Hclean Heval1 Heval2.
      destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                    | elo ehi ebod | ess | xn ev ebody | dv ebod econt | eargs | er ];
      simpl in Hcf; simpl in Hvar; simpl in Heval1; simpl in Heval2; try discriminate.

      (* ELit: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVary: closed by try discriminate above (Hvar : true = false) *)
      (* EBarrier: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EWarpPoint: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVar n0: env_agrees gives same value *)
      * inversion Heval1; inversion Heval2. subst. f_equal. apply Hagr. exact Hvar.
      (* EBinop ea eb *)
      * apply andb_true_iff in Hcf as [Hcfa Hcfb].
        apply Bool.orb_false_iff in Hvar as [Hva Hvb].
        simpl in Hclean. apply app_eq_nil in Hclean as [Hca Hcb].
        destruct (eval vary_val fuel' t1 rho1 ea) as [[[va1|va1] tra1]|] eqn:Hea1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t1 rho1 va1 tra1 Hcfa Hea1). }
        destruct (eval vary_val fuel' t1 rho1 eb) as [[[vb1|vb1] trb1]|] eqn:Heb1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 vb1 trb1 Hcfb Heb1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ea) as [[[va2|va2] tra2]|] eqn:Hea2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t2 rho2 va2 tra2 Hcfa Hea2). }
        destruct (eval vary_val fuel' t2 rho2 eb) as [[[vb2|vb2] trb2]|] eqn:Heb2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 vb2 trb2 Hcfb Heb2). }
        inversion Heval2; subst.
        assert (Hva_eq : ONorm va1 = ONorm va2).
        { exact (IH_B env ea t1 t2 rho1 rho2 (ONorm va1) (ONorm va2) tra1 tra2
                   Hcfa Hagr Hva Hca Hea1 Hea2). }
        assert (Hvb_eq : ONorm vb1 = ONorm vb2).
        { exact (IH_B env eb t1 t2 rho1 rho2 (ONorm vb1) (ONorm vb2) trb1 trb2
                   Hcfb Hagr Hvb Hcb Heb1 Heb2). }
        injection Hva_eq as Hva_eq. injection Hvb_eq as Hvb_eq.
        subst va2 vb2. reflexivity.
      (* EUnop eu *)
      * destruct (eval vary_val fuel' t1 rho1 eu) as [[[vu1|vu1] tru1]|] eqn:Heu1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t1 rho1 vu1 tru1 Hcf Heu1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 eu) as [[[vu2|vu2] tru2]|] eqn:Heu2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t2 rho2 vu2 tru2 Hcf Heu2). }
        inversion Heval2; subst.
        simpl in Hclean.
        assert (Hvu_eq : ONorm vu1 = ONorm vu2).
        { exact (IH_B env eu t1 t2 rho1 rho2 (ONorm vu1) (ONorm vu2) tr1 tr2
                   Hcf Hagr Hvar Hclean Heu1 Heu2). }
        injection Hvu_eq as Hvu_eq. subst vu2. reflexivity.
      (* EIf econd ethen eelse *)
      * apply andb_true_iff in Hcf as [Hcfct Hcfel].
        apply andb_true_iff in Hcfct as [Hcfc Hcft].
        apply Bool.orb_false_iff in Hvar as [Hvctel Hvel].
        apply Bool.orb_false_iff in Hvctel as [Hvc Hvt].
        (* cond non-varying: inner = Converged *)
        simpl in Hclean. rewrite Hvc in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcc H'].
        apply app_eq_nil in H' as [Hct Hcel].
        destruct (eval vary_val fuel' t1 rho1 econd) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 econd) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t2 rho2 cv2 trc2 Hcfc Hcond2). }
        assert (Hcveq : ONorm cv1 = ONorm cv2).
        { exact (IH_B env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hvc Hcc Hcond1 Hcond2). }
        injection Hcveq as Hcveq'. subst cv2.
        destruct (eval vary_val fuel' t1 rho1
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr1 trbr1]|] eqn:Hbr1;
          try discriminate.
        destruct (eval vary_val fuel' t2 rho2
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr2 trbr2]|] eqn:Hbr2;
          try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        destruct (Nat.eqb cv1 0) eqn:Hcv1z.
        -- exact (IH_B env eelse t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                   Hcfel Hagr Hvel Hcel Hbr1 Hbr2).
        -- exact (IH_B env ethen t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                   Hcft Hagr Hvt Hct Hbr1 Hbr2).
      (* EWhile ec eb *)
      * apply andb_true_iff in Hcf as [Hcfc Hcfb].
        apply Bool.orb_false_iff in Hvar as [Hvc_w Hvb_w].
        (* cond non-varying: inner = Converged *)
        simpl in Hclean. rewrite Hvc_w in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcc_w Hcb_w].
        destruct (eval vary_val fuel' t1 rho1 ec) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 ec) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t2 rho2 cv2 trc2 Hcfc Hcond2). }
        assert (Hcveq_w : ONorm cv1 = ONorm cv2).
        { exact (IH_B env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hvc_w Hcc_w Hcond1 Hcond2). }
        injection Hcveq_w as Hcveq_w'. subst cv2.
        destruct (Nat.eqb cv1 0) eqn:Hcv1z.
        -- (* cv = 0: both return ONorm 0 *)
           inversion Heval1; subst. inversion Heval2; subst. reflexivity.
        -- (* cv ≠ 0: recurse *)
           destruct (eval vary_val fuel' t1 rho1 eb) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 bv1 trb1 Hcfb Hbod1). }
           destruct (eval vary_val fuel' t1 rho1 (EWhile ec eb)) as [[oloop1 trloop1]|]
             eqn:Hloop1; try discriminate.
           inversion Heval1; subst.
           destruct (eval vary_val fuel' t2 rho2 eb) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 bv2 trb2 Hcfb Hbod2). }
           destruct (eval vary_val fuel' t2 rho2 (EWhile ec eb)) as [[oloop2 trloop2]|]
             eqn:Hloop2; try discriminate.
           inversion Heval2; subst.
           assert (HwcfW : core_frag (EWhile ec eb) = true).
           { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
           assert (HwvarW : is_varying_in_env env (EWhile ec eb) = false).
           { simpl. apply Bool.orb_false_iff. exact (conj Hvc_w Hvb_w). }
           assert (HwcleanW : check_env Converged env (EWhile ec eb) = []).
           { simpl. rewrite Hvc_w. simpl. rewrite Hcc_w. rewrite Hcb_w. reflexivity. }
           exact (IH_B env (EWhile ec eb) t1 t2 rho1 rho2 o1 o2 trloop1 trloop2
                    HwcfW Hagr HwvarW HwcleanW Hloop1 Hloop2).
      (* EFor elo ehi ebod: outcome is always ONorm 0 for core_frag *)
      * apply andb_true_iff in Hcf as [Hcflohi Hcfb].
        apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
        apply Bool.orb_false_iff in Hvar as [Hvlohi Hvb].
        apply Bool.orb_false_iff in Hvlohi as [Hvlo Hvhi].
        simpl in Hclean. rewrite Hvlo, Hvhi in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hclo H'].
        apply app_eq_nil in H' as [Hchi Hcb].
        destruct (eval vary_val fuel' t1 rho1 elo) as [[[lv1|lv1] trlo1]|] eqn:Hlo1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t1 rho1 lv1 trlo1 Hcflo Hlo1). }
        destruct (eval vary_val fuel' t1 rho1 ehi) as [[[hv1|hv1] trhi1]|] eqn:Hhi1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t1 rho1 hv1 trhi1 Hcfhi Hhi1). }
        destruct (eval vary_val fuel' t2 rho2 elo) as [[[lv2|lv2] trlo2]|] eqn:Hlo2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t2 rho2 lv2 trlo2 Hcflo Hlo2). }
        destruct (eval vary_val fuel' t2 rho2 ehi) as [[[hv2|hv2] trhi2]|] eqn:Hhi2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t2 rho2 hv2 trhi2 Hcfhi Hhi2). }
        (* Both outcomes are ONorm 0 regardless of loop body, for core_frag ebod *)
        (* Use for_loop_fixed approach with inline loop outcome lemma *)
        assert (Hfl1 : forall k acc r,
          (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
            match k0 with
            | O => Some (ONorm 0, acc_tr)
            | S k' =>
                match eval vary_val fuel' t1 rho1 ebod with
                | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                | None => None end end) k acc = Some r ->
          fst r = ONorm 0).
        { intros k. induction k as [|k' IHk'].
          - intros acc r H. simpl in H. inversion H. reflexivity.
          - intros acc r H. simpl in H.
            destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv|bv] trb]|] eqn:Hbod.
            + exact (IHk' _ _ H).
            + exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv trb Hcfb Hbod).
            + discriminate. }
        assert (Hfl2 : forall k acc r,
          (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
            match k0 with
            | O => Some (ONorm 0, acc_tr)
            | S k' =>
                match eval vary_val fuel' t2 rho2 ebod with
                | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                | None => None end end) k acc = Some r ->
          fst r = ONorm 0).
        { intros k. induction k as [|k' IHk'].
          - intros acc r H. simpl in H. inversion H. reflexivity.
          - intros acc r H. simpl in H.
            destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv|bv] trb]|] eqn:Hbod.
            + exact (IHk' _ _ H).
            + exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv trb Hcfb Hbod).
            + discriminate. }
        (* Destruct leb in Heval1/Heval2 to distinguish base vs loop case *)
        destruct (Nat.leb hv1 lv1) eqn:Hlb1.
        { inversion Heval1; subst.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst. reflexivity. }
          { apply Hfl2 in Heval2. simpl in Heval2. destruct o2; congruence. } }
        { destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst.
            apply Hfl1 in Heval1. simpl in Heval1. destruct o1; congruence. }
          { apply Hfl1 in Heval1. apply Hfl2 in Heval2.
            destruct o1, o2; simpl in *; congruence. } }
      (* ESeq ess: outcome is always ONorm 0 for core_frag *)
      * simpl in Hclean.
        (* Both return ONorm 0: seq never returns ORet with core_frag *)
        (* All core_frag seqs return ONorm 0 *)
        assert (Hgen1 : forall xs acc o tr,
          forallb core_frag xs = true ->
          (fix eval_seq xs0 acc_tr : option (outcome * trace) :=
            match xs0 with [] => Some (ONorm 0, acc_tr) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr') => Some (ORet v, acc_tr ++ tr')
              | Some (ONorm _, tr') => eval_seq rest (acc_tr ++ tr')
              | None => None end end) xs acc = Some (o, tr) ->
          o = ONorm 0).
        { intros xs. induction xs as [|h tl IHtl]; intros acc o tr Hcfs He.
          - inversion He. reflexivity.
          - simpl in He. simpl in Hcfs. apply andb_true_iff in Hcfs as [Hcfh Hcftl].
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh|vh] trh]|] eqn:Hh;
              try discriminate.
            + exact (IHtl _ _ _ Hcftl He).
            + exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh trh Hcfh Hh). }
        assert (Hgen2 : forall xs acc o tr,
          forallb core_frag xs = true ->
          (fix eval_seq xs0 acc_tr : option (outcome * trace) :=
            match xs0 with [] => Some (ONorm 0, acc_tr) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr') => Some (ORet v, acc_tr ++ tr')
              | Some (ONorm _, tr') => eval_seq rest (acc_tr ++ tr')
              | None => None end end) xs acc = Some (o, tr) ->
          o = ONorm 0).
        { intros xs. induction xs as [|h tl IHtl]; intros acc o tr Hcfs He.
          - inversion He. reflexivity.
          - simpl in He. simpl in Hcfs. apply andb_true_iff in Hcfs as [Hcfh Hcftl].
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh|vh] trh]|] eqn:Hh;
              try discriminate.
            + exact (IHtl _ _ _ Hcftl He).
            + exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh trh Hcfh Hh). }
        assert (Ho1 : o1 = ONorm 0) by exact (Hgen1 ess [] o1 tr1 Hcf Heval1).
        assert (Ho2 : o2 = ONorm 0) by exact (Hgen2 ess [] o2 tr2 Hcf Heval2).
        subst o1 o2. reflexivity.
      (* ELet xn ev ebody *)
      * apply andb_true_iff in Hcf as [Hcfv Hcfb].
        simpl in Hvar. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcv Hcb].
        destruct (eval vary_val fuel' t1 rho1 ev) as [[[vv1|vv1] trv1]|] eqn:Hev1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t1 rho1 vv1 trv1 Hcfv Hev1). }
        destruct (eval vary_val fuel' t2 rho2 ev) as [[[vv2|vv2] trv2]|] eqn:Hev2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t2 rho2 vv2 trv2 Hcfv Hev2). }
        destruct (eval vary_val fuel' t1 (venv_extend rho1 xn vv1) ebody)
          as [[ob1 trb1]|] eqn:Hbod1; try discriminate.
        destruct (eval vary_val fuel' t2 (venv_extend rho2 xn vv2) ebody)
          as [[ob2 trb2]|] eqn:Hbod2; try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        (* Build env_agrees for extended env *)
        set (vvflag := is_varying_in_env env ev).
        assert (Hagr' : env_agrees (env_extend env xn vvflag)
                          (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)).
        { unfold vvflag. destruct (is_varying_in_env env ev) eqn:Hvv.
          - intros y Hlookup.
            assert (Hxy : xn <> y).
            { intro Heq. subst. rewrite env_lookup_extend_same in Hlookup. discriminate. }
            rewrite (venv_lookup_extend_diff rho1 xn y vv1 Hxy).
            rewrite (venv_lookup_extend_diff rho2 xn y vv2 Hxy).
            apply Hagr.
            unfold env_lookup, env_extend in Hlookup |- *.
            simpl find in *.
            destruct (Nat.eqb xn y) eqn:Heqn.
            { apply Nat.eqb_eq in Heqn. subst. exfalso. exact (Hxy eq_refl). }
            { exact Hlookup. }
          - assert (Hveq : ONorm vv1 = ONorm vv2).
            { exact (IH_B env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                       Hcfv Hagr Hvv Hcv Hev1 Hev2). }
            injection Hveq as Hveq'. subst vv2.
            exact (env_agrees_extend env rho1 rho2 xn vv1 false Hagr). }
        exact (IH_B (env_extend env xn vvflag) ebody t1 t2
                 (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)
                 o1 o2 trb1 trb2 Hcfb Hagr' Hvar Hcb Hbod1 Hbod2).
      (* ESuperstep: core_frag = false *)
      (* EApp eargs: last value equality *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag eargs = true) by exact Hcf.
        clear Hcf.
        assert (Hgen : forall xs,
          forallb core_frag xs = true ->
          concat (map (check_env Converged env) xs) = [] ->
          existsb (is_varying_in_env env) xs = false ->
          forall acc1 acc2 lv1 lv2 o1' o2' tr1' tr2',
          lv1 = lv2 ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc1 lv1 = Some (o1', tr1') ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc2 lv2 = Some (o2', tr2') ->
          o1' = o2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ _ acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hlv Ha1 Ha2.
            inversion Ha1; inversion Ha2; subst. congruence.
          - intros Hcfal Hcln Hv acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hlv Ha1 Ha2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Hv. apply Bool.orb_false_iff in Hv as [Hvh Hvtl].
            simpl in Ha1, Ha2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            assert (Hvheq : ONorm vh1 = ONorm vh2).
            { exact (IH_B env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                       Hcfh Hagr Hvh Hch Hh1 Hh2). }
            injection Hvheq as Hvheq'. subst vh2.
            exact (IHtl Hcftl Hctl Hvtl (acc1 ++ trh1) (acc2 ++ trh2) vh1 vh1
                     o1' o2' tr1' tr2' eq_refl Ha1 Ha2). }
        exact (Hgen eargs Hcfall Hclean Hvar [] [] 0 0 o1 o2 tr1 tr2 eq_refl Heval1 Heval2).
      (* EReturn: core_frag = false *)
Qed.

(* ----------------------------------------------------------------------- *)
(* 9.10  check_env_sound_core — main barrier safety theorem                 *)
(* ----------------------------------------------------------------------- *)

(** check_env_sound_core: if core_frag e = true and check_env Converged env e = [],
    then e is barrier_safe: any two env-agreeing threads that complete evaluation
    produce the same barrier-event sequence. *)
Theorem check_env_sound_core :
  forall vary_val env e,
    core_frag e = true ->
    check_env Converged env e = [] ->
    barrier_safe vary_val env e.
Proof.
  intros vary_val env e Hcf Hclean.
  unfold barrier_safe.
  intros fuel t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hagr Heval1 Heval2.
  exact (proj1 (eval_check_uniform vary_val fuel) env e t1 t2 rho1 rho2 o1 o2 tr1 tr2
           Hcf Hagr Hclean Heval1 Heval2).
Qed.

(* ----------------------------------------------------------------------- *)
(* 10.  T3-S5 — EReturn residual-divergence verdict (F-04)                  *)
(* ----------------------------------------------------------------------- *)

(** F-04: EReturn transparency is a kernel-granularity false negative.

    The checker treats EReturn as a transparent wrapper
    (check_env m env (EReturn e) = check_env m env e), mirroring the real
    Sarek_convergence.ml TEReturn handling. As a consequence, a thread-varying
    early return inside an EIf branch — followed by a barrier — passes the
    static checker with an empty error list, yet is NOT barrier_safe: some
    threads take the early return (never reaching the barrier) while others
    fall through to it. The barrier traces diverge.

    hazard models exactly this:
      ESeq [ EIf EVary (EReturn ELit) ELit ; EBarrier ]
    - EIf on a thread-varying condition (EVary): branch is e_else (=ELit) when
      the condition value is 0, e_then (=EReturn ELit) when nonzero.
    - the early-return branch short-circuits the ESeq (ORet), so EBarrier is
      never evaluated for those threads (trace []).
    - the fall-through branch (ONorm) lets ESeq reach EBarrier (trace
      [EvBarrier]).
    Because the EReturn sits in a branch under a varying condition, and
    Converged-mode EBarrier is clean, the checker reports []. *)

Definition hazard : expr :=
  ESeq [EIf EVary (EReturn ELit) ELit; EBarrier].

(** hazard_checker_blind: the static checker is blind to the hazard.
    Converged-mode check_env returns no errors on the empty environment. *)
Lemma hazard_checker_blind :
  check_env Converged [] hazard = [].
Proof. reflexivity. Qed.

(** Concrete thread-varying witness: thread 0 takes the early return
    (vary_val 0 = 1, nonzero -> e_then = EReturn ELit), thread 1 falls
    through to the barrier (vary_val 1 = 0 -> e_else = ELit). *)
Definition hazard_vary (n : tid) : value :=
  match n with O => 1 | S _ => 0 end.

(** Thread 0 completes hazard with the early-return outcome and an empty
    barrier trace: the EBarrier is never reached. *)
Lemma hazard_eval_thread0 :
  eval hazard_vary 6 0 [] hazard = Some (ORet 0, []).
Proof. reflexivity. Qed.

(** Thread 1 completes hazard by falling through to the barrier, emitting
    exactly one EvBarrier event. *)
Lemma hazard_eval_thread1 :
  eval hazard_vary 6 1 [] hazard = Some (ONorm 0, [EvBarrier]).
Proof. reflexivity. Qed.

(** hazard_not_barrier_safe: the hazard is NOT barrier_safe, witnessed by the
    concrete vary_val = hazard_vary and the two threads above. Thread 0 and
    thread 1 trivially env-agree on the empty environment, yet their barrier
    traces ([] vs [EvBarrier]) differ. This is a formal counterexample to
    barrier safety that the checker accepts (hazard_checker_blind). *)
Theorem hazard_not_barrier_safe :
  ~ barrier_safe hazard_vary [] hazard.
Proof.
  unfold barrier_safe. intro Hsafe.
  (* Instantiate at the two concrete threads. env_agrees [] holds vacuously. *)
  assert (Hagr : env_agrees [] [] []).
  { intros y _. reflexivity. }
  pose proof (Hsafe 6 0 1 [] [] (ORet 0) (ONorm 0) [] [EvBarrier]
                Hagr hazard_eval_thread0 hazard_eval_thread1) as Heq.
  (* Heq : erase_warp [] = erase_warp [EvBarrier], i.e. [] = [EvBarrier] *)
  unfold erase_warp in Heq. simpl in Heq. discriminate.
Qed.

(* ======================================================================= *)
(* 11.  T3-S6 — ESuperstep semantic grounding (semantic F-01)              *)
(* ======================================================================= *)

(** core_frag_ss: the core fragment ENLARGED to include ESuperstep nodes.
    Differs from core_frag in exactly one constructor: ESuperstep false body cont
    is now in the fragment provided its body and cont are (recursively) in
    core_frag_ss. The clause is literally [negb dv && core_frag_ss b && core_frag_ss c],
    so ONLY uniform-reachability (dv=false) supersteps enter the
    verified fragment. EReturn remains excluded — an early return bypasses later
    barriers and breaks barrier uniformity (cf. the T3-S5 hazard / F-04).

    TRUST BOUNDARY (dv=true): the operational semantics emits the implicit
    boundary [EvBarrier] uniformly on every thread regardless of dv (the runtime
    barrier always fires; dv affects only the static check_env verdict). But a
    dv=true superstep is NOT flagged by check_env in Diverged mode, so admitting
    it into core_frag_ss would make check_env_diverged_no_barriers_ss false. dv=true
    is therefore OUT of the verified fragment and relies on the front-end's
    divergence-uniformity annotation being correct (ASSUMPTIONS.md, §T3-S6). *)
Fixpoint core_frag_ss (e : expr) : bool :=
  match e with
  | ELit | EVary | EBarrier | EWarpPoint | EVar _ => true
  | EBinop a b        => core_frag_ss a && core_frag_ss b
  | EUnop e0          => core_frag_ss e0
  | EIf c t el        => core_frag_ss c && core_frag_ss t && core_frag_ss el
  | EWhile c b        => core_frag_ss c && core_frag_ss b
  | EFor lo hi b      => core_frag_ss lo && core_frag_ss hi && core_frag_ss b
  | ESeq es           => forallb core_frag_ss es
  | ELet _ v b        => core_frag_ss v && core_frag_ss b
  | ESuperstep dv b c =>
      (* only uniform-reachability (dv=false) supersteps are in the verified
         fragment; dv=true is a documented trust boundary (ASSUMPTIONS.md) *)
      negb dv && core_frag_ss b && core_frag_ss c
  | EApp args         => forallb core_frag_ss args
  | EReturn _         => false   (* early-exit bypasses later barriers; excluded *)
  end.

(** core_frag implies core_frag_ss: the enlarged fragment subsumes the core. *)
Lemma core_frag_impl_ss : forall e, core_frag e = true -> core_frag_ss e = true.
Proof.
  apply (expr_list_rect
    (fun e  => core_frag e = true -> core_frag_ss e = true)
    (fun es => forallb core_frag es = true -> forallb core_frag_ss es = true)).
  - intros _. reflexivity.
  - intros _. reflexivity.
  - intros _. reflexivity.
  - intros _. reflexivity.
  - intros x _. reflexivity.
  - intros a b IHa IHb H. simpl in H. apply andb_true_iff in H as [Ha Hb].
    simpl. apply andb_true_iff. exact (conj (IHa Ha) (IHb Hb)).
  - intros e0 IH H. simpl in H. simpl. exact (IH H).
  - intros c t el IHc IHt IHel H. simpl in H.
    apply andb_true_iff in H as [Hct Hel]. apply andb_true_iff in Hct as [Hc Ht].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHc Hc) (IHt Ht)).
    + exact (IHel Hel).
  - intros c b IHc IHb H. simpl in H. apply andb_true_iff in H as [Hc Hb].
    simpl. apply andb_true_iff. exact (conj (IHc Hc) (IHb Hb)).
  - intros lo hi b IHlo IHhi IHb H. simpl in H.
    apply andb_true_iff in H as [Hlohi Hb]. apply andb_true_iff in Hlohi as [Hlo Hhi].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHlo Hlo) (IHhi Hhi)).
    + exact (IHb Hb).
  - intros es IHes H. simpl in H. simpl. exact (IHes H).
  - intros x v b IHv IHb H. simpl in H. apply andb_true_iff in H as [Hv Hb].
    simpl. apply andb_true_iff. exact (conj (IHv Hv) (IHb Hb)).
  - intros dv body cont _ _ H. simpl in H. discriminate.  (* core_frag ESuperstep = false *)
  - intros args IHargs H. simpl in H. simpl. exact (IHargs H).
  - intros e0 _ H. simpl in H. discriminate.  (* core_frag EReturn = false *)
  - intros _. reflexivity.
  - intros h tl IHh IHtl H. simpl in H. apply andb_true_iff in H as [Hh Htl].
    simpl. apply andb_true_iff. exact (conj (IHh Hh) (IHtl Htl)).
Qed.

(** core_frag_ss_no_ret: like core_frag_ss_no_ret but for the enlarged fragment.
    Even with ESuperstep in the fragment, no core_frag_ss expression returns ORet:
    a superstep's body cannot return (core_frag_ss body) and its overall
    outcome is the cont outcome, which is itself ORet-free by induction. *)
Lemma core_frag_ss_no_ret :
  forall vary_val fuel,
  forall e t rho v tr,
    core_frag_ss e = true ->
    eval vary_val fuel t rho e = Some (ORet v, tr) -> False.
Proof.
  intros vary_val.
  induction fuel as [| f' IHf].
  - intros e t rho v tr _ H. simpl in H. discriminate.
  - intros e t rho v tr Hcf H.
    destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                  | elo ehi ebod | ess | xn ev eb | dv ebod econt | eargs | er ];
    simpl in Hcf; simpl in H; try discriminate.
    (* EBinop ea eb *)
    + apply andb_true_iff in Hcf as [Hcfa Hcfb].
      set (rea := eval vary_val f' t rho ea) in *.
      destruct rea as [[[va|va] tra_a]|] eqn:Hea.
      { (* ONorm va: check eb *)
        set (reb := eval vary_val f' t rho eb) in *.
        destruct reb as [[[vb|vb] trb]|] eqn:Heb.
        - (* ONorm vb: result is ONorm, H says ORet, contradiction *)
          discriminate.
        - (* ORet vb: result is ORet vb *)
          inversion H. subst.
          unfold reb in Heb.
          exact (IHf eb t rho v trb Hcfb Heb).
        - (* None: H = None = Some ORet, contradiction *)
          discriminate. }
      { (* ORet va: result of EBinop is Some (ORet va, tra_a) *)
        inversion H. subst.
        unfold rea in Hea.
        exact (IHf ea t rho v tr Hcfa Hea). }
      { (* None: H = None = Some ORet, contradiction *)
        discriminate. }
    (* EUnop eu *)
    + set (reu := eval vary_val f' t rho eu) in *.
      destruct reu as [[[vu|vu] tru]|] eqn:Heu; try discriminate.
      inversion H. subst. unfold reu in Heu. exact (IHf eu t rho v tr Hcf Heu).
    (* EIf econd ethen eelse *)
    + apply andb_true_iff in Hcf as [Hcfct Hcfel].
      apply andb_true_iff in Hcfct as [Hcfc Hcft].
      set (rcond := eval vary_val f' t rho econd) in *.
      destruct rcond as [[[vc|vc] trc]|] eqn:Hcond; try discriminate.
      { (* ONorm vc: evaluate branch *)
        set (rbr := eval vary_val f' t rho (if Nat.eqb vc 0 then eelse else ethen)) in *.
        destruct rbr as [[ob trb]|] eqn:Hbr; try discriminate.
        inversion H. subst.
        unfold rbr in Hbr.
        case_eq (Nat.eqb vc 0); intro Hb; rewrite Hb in Hbr.
        - exact (IHf eelse t rho v trb Hcfel Hbr).
        - exact (IHf ethen t rho v trb Hcft Hbr). }
      { (* ORet vc: propagates *)
        inversion H. subst. unfold rcond in Hcond. exact (IHf econd t rho v tr Hcfc Hcond). }
    (* EWhile ec eb *)
    + apply andb_true_iff in Hcf as [Hcfc Hcfb].
      set (rcond_w := eval vary_val f' t rho ec) in *.
      destruct rcond_w as [[[vc|vc] trc]|] eqn:Hcond; try discriminate.
      { destruct (Nat.eqb vc 0); try discriminate.
        set (rbod_w := eval vary_val f' t rho eb) in *.
        destruct rbod_w as [[[vb|vb] trb]|] eqn:Hbod; try discriminate.
        { (* ONorm vb: recursive EWhile call at fuel f' *)
          set (rloop := eval vary_val f' t rho (EWhile ec eb)) in *.
          destruct rloop as [[oloop trloop]|] eqn:Hloop; try discriminate.
          inversion H. subst.
          assert (HcfW : core_frag_ss (EWhile ec eb) = true).
          { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
          unfold rloop in Hloop.
          exact (IHf (EWhile ec eb) t rho v trloop HcfW Hloop). }
        { (* ORet vb: propagates from body *)
          inversion H. subst. unfold rbod_w in Hbod. exact (IHf eb t rho v trb Hcfb Hbod). } }
      { inversion H. subst. unfold rcond_w in Hcond. exact (IHf ec t rho v tr Hcfc Hcond). }
    (* EFor elo ehi ebod *)
    + apply andb_true_iff in Hcf as [Hcflohi Hcfb].
      apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
      set (rlo := eval vary_val f' t rho elo) in *.
      destruct rlo as [[[vlo|vlo] trlo]|] eqn:Hlo; try discriminate.
      { set (rhi := eval vary_val f' t rho ehi) in *.
        destruct rhi as [[[vhi|vhi] trhi]|] eqn:Hhi; try discriminate.
        { destruct (Nat.leb vhi vlo); try discriminate.
          rewrite (for_loop_eq vary_val f' t rho ebod (vhi - vlo) (trlo ++ trhi)) in H.
          (* Induct on steps, generalize acc *)
          revert H.
          generalize (trlo ++ trhi) as acc0.
          induction (vhi - vlo) as [| s IHs]; intros acc0 H.
          - simpl in H. discriminate.
          - simpl in H.
            set (rbod := eval vary_val f' t rho ebod) in *.
            destruct rbod as [[[vb|vb] trb]|] eqn:Hbod; try discriminate.
            + exact (IHs _ H).
            + inversion H. subst. unfold rbod in Hbod. exact (IHf ebod t rho v trb Hcfb Hbod). }
        { inversion H. subst. unfold rhi in Hhi. exact (IHf ehi t rho v trhi Hcfhi Hhi). } }
      { inversion H. subst. unfold rlo in Hlo. exact (IHf elo t rho v tr Hcflo Hlo). }
    (* ESeq ess *)
    + rewrite forallb_forall in Hcf.
      rename H into Hseq_outer.
      assert (Hseq : forall acc,
        (fix eval_seq (xs : list expr) (acc_tr : trace) : option (outcome * trace) :=
          match xs with
          | []      => Some (ONorm 0, acc_tr)
          | x :: rest =>
              match eval vary_val f' t rho x with
              | Some (ORet v0, tr0)  => Some (ORet v0, acc_tr ++ tr0)
              | Some (ONorm _, tr0) => eval_seq rest (acc_tr ++ tr0)
              | None               => None
              end
          end) ess acc = Some (ORet v, tr) -> False).
      { clear Hseq_outer.
        induction ess as [| h0 tl0 IHtl0].
        - intros acc H0. simpl in H0. discriminate.
        - intros acc H0. simpl in H0.
          assert (Hcfh0 : core_frag_ss h0 = true) by (apply Hcf; apply in_eq).
          destruct (eval vary_val f' t rho h0) as [[[vh|vh] trh]|] eqn:Hh0; try discriminate.
          + exact (IHtl0 (fun x Hin => Hcf x (in_cons _ _ _ Hin)) (acc ++ trh) H0).
          + inversion H0. subst. exact (IHf h0 t rho v trh Hcfh0 Hh0). }
      exact (Hseq [] Hseq_outer).
    (* ELet xn ev eb *)
    + apply andb_true_iff in Hcf as [Hcfv Hcfb].
      set (rev_let := eval vary_val f' t rho ev) in *.
      destruct rev_let as [[[vv|vv] trv]|] eqn:Hev; try discriminate.
      { (* ONorm vv: evaluate body with extended env *)
        set (rbod_let := eval vary_val f' t (venv_extend rho xn vv) eb) in *.
        destruct rbod_let as [[ob'' trb'']|] eqn:Hbod; try discriminate.
        inversion H. subst.
        unfold rbod_let in Hbod.
        exact (IHf eb t (venv_extend rho xn vv) v trb'' Hcfb Hbod). }
      { (* ORet vv: propagates from binder *)
        inversion H. subst. unfold rev_let in Hev. exact (IHf ev t rho v tr Hcfv Hev). }
    (* ESuperstep dv ebod econt: body never returns ORet; cont's ORet excluded *)
    + apply andb_true_iff in Hcf as [Hcfdvb Hcfc].
      apply andb_true_iff in Hcfdvb as [_ Hcfb].
      set (rbody := eval vary_val f' t rho ebod) in *.
      destruct rbody as [[[vb|vb] trb]|] eqn:Hb; try discriminate.
      { set (rcont := eval vary_val f' t rho econt) in *.
        destruct rcont as [[oc trc]|] eqn:Hc; try discriminate.
        inversion H. subst.
        unfold rcont in Hc.
        exact (IHf econt t rho v trc Hcfc Hc). }
      { exfalso. unfold rbody in Hb. exact (IHf ebod t rho vb trb Hcfb Hb). }
    (* EApp eargs *)
    + rewrite forallb_forall in Hcf.
      rename H into Hargs_outer.
      assert (Hargs : forall acc lv,
        (fix eval_args (xs : list expr) (acc_tr : trace) (last_v : value) : option (outcome * trace) :=
          match xs with
          | []      => Some (ONorm last_v, acc_tr)
          | x :: rest =>
              match eval vary_val f' t rho x with
              | Some (ORet v0, tr0)  => Some (ORet v0, acc_tr ++ tr0)
              | Some (ONorm v0, tr0) => eval_args rest (acc_tr ++ tr0) v0
              | None               => None
              end
          end) eargs acc lv = Some (ORet v, tr) -> False).
      { clear Hargs_outer.
        induction eargs as [| h0 tl0 IHtl0].
        - intros acc lv H0. simpl in H0. discriminate.
        - intros acc lv H0. simpl in H0.
          assert (Hcfh0 : core_frag_ss h0 = true) by (apply Hcf; apply in_eq).
          destruct (eval vary_val f' t rho h0) as [[[vh|vh] trh]|] eqn:Hh0; try discriminate.
          + exact (IHtl0 (fun x Hin => Hcf x (in_cons _ _ _ Hin)) (acc ++ trh) vh H0).
          + inversion H0. subst. exact (IHf h0 t rho v trh Hcfh0 Hh0). }
      exact (Hargs [] 0 Hargs_outer).
    (* EReturn: excluded by core_frag_ss = false (try discriminate closed it) *)
Qed.

(** eval_while_exits_immediately_ss: the core_frag_ss analogue of
    eval_while_exits_immediately. EWhile contains no ESuperstep at the head,
    so the proof is identical modulo the fragment predicate. *)
Lemma eval_while_exits_immediately_ss :
  forall vary_val fuel ec eb t rho o tr,
    core_frag_ss (EWhile ec eb) = true ->
    eval vary_val fuel t rho (EWhile ec eb) = Some (o, tr) ->
    eval vary_val (Nat.pred fuel) t rho ec = Some (ONorm 0, tr) /\ o = ONorm 0.
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - intros ec eb t rho o tr _ H. simpl in H. discriminate.
  - intros ec eb t rho o tr Hcf H.
    apply andb_true_iff in Hcf as [Hcfc Hcfb].
    simpl in H.
    (* outcome = ONorm | ORet; first branch = ONorm, second = ORet *)
    destruct (eval vary_val fuel' t rho ec) as [[[cv|cv] tr_c]|] eqn:Hec.
    + (* ec → ONorm cv: check if cv = 0 *)
      destruct (Nat.eqb cv 0) eqn:Hcv0.
      * apply Nat.eqb_eq in Hcv0. subst cv.
        inversion H. subst. simpl. exact (conj Hec eq_refl).
      * apply Nat.eqb_neq in Hcv0.
        (* cv ≠ 0: body must return Some for the whole thing to be Some *)
        destruct (eval vary_val fuel' t rho eb) as [[[bv|bv] tr_b]|] eqn:Heb.
        -- (* body → ONorm bv: recursive EWhile call *)
           destruct (eval vary_val fuel' t rho (EWhile ec eb)) as [[oloop trloop]|] eqn:Hloop.
           ++ (* recursive call → Some: derive contradiction via IH *)
              assert (Hwcf : core_frag_ss (EWhile ec eb) = true).
              { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
              destruct (IHfuel ec eb t rho oloop trloop Hwcf Hloop) as [Hec0 _].
              (* Hec0 : eval (pred fuel') t rho ec = Some(ONorm 0, trloop) *)
              destruct fuel' as [|fuel''].
              { simpl in Hloop. discriminate. }
              (* fuel' = S fuel'': pred (S fuel'') = fuel'' *)
              simpl in Hec0.
              apply eval_fuel_monotone in Hec0.
              rewrite Hec in Hec0.
              exfalso. apply Hcv0. congruence.
           ++ discriminate.
        -- (* body → ORet: excluded by core_frag_ss *)
           exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t rho bv tr_b Hcfb Heb).
        -- discriminate.
    + (* ec → ORet: excluded by core_frag_ss *)
      exfalso. exact (core_frag_ss_no_ret vary_val fuel' ec t rho cv tr_c Hcfc Hec).
    + discriminate.
Qed.

(** core_frag_ss_barrier_free_superstep_free: a core_frag_ss expression that is
    also barrier_free contains no superstep at all (hence is superstep_free).
    Key fact: core_frag_ss admits only dv=false supersteps, and barrier_free of
    a dv=false superstep is false. So barrier_free rules supersteps out entirely
    on this fragment. *)
Lemma core_frag_ss_barrier_free_superstep_free :
  forall e, core_frag_ss e = true -> barrier_free e = true -> superstep_free e = true.
Proof.
  apply (expr_list_rect
    (fun e  => core_frag_ss e = true -> barrier_free e = true -> superstep_free e = true)
    (fun es => forallb core_frag_ss es = true -> forallb barrier_free es = true ->
               forallb superstep_free es = true)).
  - intros _ _. reflexivity.
  - intros _ _. reflexivity.
  - intros _ _. reflexivity.
  - intros _ _. reflexivity.
  - intros x _ _. reflexivity.
  - intros a b IHa IHb Hcf Hbf. simpl in Hcf, Hbf.
    apply andb_true_iff in Hcf as [Hca Hcb]. apply andb_true_iff in Hbf as [Hba Hbb].
    simpl. apply andb_true_iff. exact (conj (IHa Hca Hba) (IHb Hcb Hbb)).
  - intros e0 IH Hcf Hbf. simpl in Hcf, Hbf. simpl. exact (IH Hcf Hbf).
  - intros c t el IHc IHt IHel Hcf Hbf. simpl in Hcf, Hbf.
    apply andb_true_iff in Hcf as [Hct Hcel]. apply andb_true_iff in Hct as [Hcc Hct'].
    apply andb_true_iff in Hbf as [Hbt Hbel]. apply andb_true_iff in Hbt as [Hbc Hbt'].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHc Hcc Hbc) (IHt Hct' Hbt')).
    + exact (IHel Hcel Hbel).
  - intros c b IHc IHb Hcf Hbf. simpl in Hcf, Hbf.
    apply andb_true_iff in Hcf as [Hcc Hcb]. apply andb_true_iff in Hbf as [Hbc Hbb].
    simpl. apply andb_true_iff. exact (conj (IHc Hcc Hbc) (IHb Hcb Hbb)).
  - intros lo hi b IHlo IHhi IHb Hcf Hbf. simpl in Hcf, Hbf.
    apply andb_true_iff in Hcf as [Hclohi Hcb]. apply andb_true_iff in Hclohi as [Hclo Hchi].
    apply andb_true_iff in Hbf as [Hblohi Hbb]. apply andb_true_iff in Hblohi as [Hblo Hbhi].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHlo Hclo Hblo) (IHhi Hchi Hbhi)).
    + exact (IHb Hcb Hbb).
  - intros es IHes Hcf Hbf. simpl in Hcf, Hbf. simpl. exact (IHes Hcf Hbf).
  - intros x v b IHv IHb Hcf Hbf. simpl in Hcf, Hbf.
    apply andb_true_iff in Hcf as [Hcv Hcb]. apply andb_true_iff in Hbf as [Hbv Hbb].
    simpl. apply andb_true_iff. exact (conj (IHv Hcv Hbv) (IHb Hcb Hbb)).
  - (* ESuperstep dv body cont: barrier_free forces dv=true, but core_frag_ss
       admits only dv=false — contradiction. *)
    intros dv body cont _ _ Hcf Hbf.
    destruct dv; simpl in Hcf, Hbf.
    + (* dv=true: barrier_free has (if true then true else false)=true so ok,
         but core_frag_ss has negb true = false -> Hcf false *)
      simpl in Hcf. discriminate.
    + (* dv=false: barrier_free has (if false..)=false -> Hbf false *)
      simpl in Hbf. discriminate.
  - intros args IHargs Hcf Hbf. simpl in Hcf, Hbf. simpl. exact (IHargs Hcf Hbf).
  - intros e0 _ Hcf _. simpl in Hcf. discriminate.  (* EReturn excluded *)
  - intros _ _. reflexivity.
  - intros h tl IHh IHtl Hcf Hbf. simpl in Hcf, Hbf.
    apply andb_true_iff in Hcf as [Hch Hctl]. apply andb_true_iff in Hbf as [Hbh Hbtl].
    simpl. apply andb_true_iff. exact (conj (IHh Hch Hbh) (IHtl Hctl Hbtl)).
Qed.

(** check_env_diverged_no_barriers_ss: the core_frag_ss analogue of
    check_env_diverged_no_barriers. On the enlarged fragment, a Diverged-clean
    expression is still barrier-silent: the Diverged-mode checker would flag any
    reachable dv=false superstep with [BarrierError], so a clean check forces
    superstep-freedom, and barrier-freedom follows from the existing bridge. *)
Lemma check_env_diverged_no_barriers_ss :
  forall vary_val env e,
    core_frag_ss e = true ->
    check_env Diverged env e = [] ->
    forall fuel t rho o tr,
      eval vary_val fuel t rho e = Some (o, tr) ->
      no_barrier_event tr = true.
Proof.
  intros vary_val env e Hcf Hclean fuel t rho o tr Heval.
  assert (Hbf : barrier_free e = true)
    by exact (check_env_diverged_clean_barrier_free e env Hclean).
  apply barrier_free_no_barriers with
    (vary_val := vary_val) (fuel := fuel) (t := t) (rho := rho) (e := e) (o := o).
  - exact (core_frag_ss_barrier_free_superstep_free e Hcf Hbf).
  - exact Hbf.
  - exact Heval.
Qed.

(** eval_check_uniform_ss: the combined barrier/outcome uniformity lemma over
    the ENLARGED fragment core_frag_ss. Identical structure to
    eval_check_uniform; the one genuinely new case is ESuperstep, where the
    implicit boundary [EvBarrier] is emitted uniformly on every thread, so the
    barrier traces of two env-agreeing threads stay equal (Part A) and the
    superstep outcome (= cont outcome) agrees when non-varying (Part B). *)
Lemma eval_check_uniform_ss : forall vary_val fuel,
  (* Part A: barrier-trace equality for all clean core_frag_ss expressions *)
  (forall env e t1 t2 rho1 rho2 o1 o2 tr1 tr2,
     core_frag_ss e = true ->
     env_agrees env rho1 rho2 ->
     check_env Converged env e = [] ->
     eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
     eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
     erase_warp tr1 = erase_warp tr2) /\
  (* Part B: outcome equality for non-varying clean core_frag_ss expressions *)
  (forall env e t1 t2 rho1 rho2 o1 o2 tr1 tr2,
     core_frag_ss e = true ->
     env_agrees env rho1 rho2 ->
     is_varying_in_env env e = false ->
     check_env Converged env e = [] ->
     eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
     eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
     o1 = o2).
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - split.
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 _ _ _ H _. simpl in H. discriminate.
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 _ _ _ _ H _. simpl in H. discriminate.
  - destruct IHfuel as [IH_A IH_B].
    split.

(* ------------------------------------------------------------------ *)
(* Part A for S fuel'                                                   *)
(* ------------------------------------------------------------------ *)
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hcf Hagr Hclean Heval1 Heval2.
      destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                    | elo ehi ebod | ess | xn ev ebody | dv ebod econt | eargs | er ];
      simpl in Hcf; simpl in Heval1; simpl in Heval2; try discriminate.

      (* ELit *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVary *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EBarrier *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EWarpPoint *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVar n0 *)
      * inversion Heval1; inversion Heval2; reflexivity.

      (* EBinop ea eb *)
      * apply andb_true_iff in Hcf as [Hcfa Hcfb].
        simpl in Hclean. apply app_eq_nil in Hclean as [Hca Hcb].
        destruct (eval vary_val fuel' t1 rho1 ea) as [[[va1|va1] tra1]|] eqn:Hea1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ea t1 rho1 va1 tra1 Hcfa Hea1). }
        destruct (eval vary_val fuel' t1 rho1 eb) as [[[vb1|vb1] trb1]|] eqn:Heb1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t1 rho1 vb1 trb1 Hcfb Heb1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ea) as [[[va2|va2] tra2]|] eqn:Hea2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ea t2 rho2 va2 tra2 Hcfa Hea2). }
        destruct (eval vary_val fuel' t2 rho2 eb) as [[[vb2|vb2] trb2]|] eqn:Heb2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t2 rho2 vb2 trb2 Hcfb Heb2). }
        inversion Heval2; subst.
        rewrite !erase_warp_app. f_equal.
        { exact (IH_A env ea t1 t2 rho1 rho2 (ONorm va1) (ONorm va2) tra1 tra2
                   Hcfa Hagr Hca Hea1 Hea2). }
        { exact (IH_A env eb t1 t2 rho1 rho2 (ONorm vb1) (ONorm vb2) trb1 trb2
                   Hcfb Hagr Hcb Heb1 Heb2). }

      (* EUnop eu *)
      * destruct (eval vary_val fuel' t1 rho1 eu) as [[[vu1|vu1] tru1]|] eqn:Heu1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eu t1 rho1 vu1 tru1 Hcf Heu1). }
        destruct (eval vary_val fuel' t2 rho2 eu) as [[[vu2|vu2] tru2]|] eqn:Heu2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eu t2 rho2 vu2 tru2 Hcf Heu2). }
        (* Heval1 : Some (ONorm (unop_eval vu1), tru1) = Some (o1, tr1) *)
        (* Heval2 : Some (ONorm (unop_eval vu2), tru2) = Some (o2, tr2) *)
        injection Heval1 as Hout1 Htr1. injection Heval2 as Hout2 Htr2.
        simpl in Hclean. rewrite <- Htr1. rewrite <- Htr2.
        exact (IH_A env eu t1 t2 rho1 rho2 (ONorm vu1) (ONorm vu2) tru1 tru2
                 Hcf Hagr Hclean Heu1 Heu2).

      (* EIf econd ethen eelse *)
      * apply andb_true_iff in Hcf as [Hcfct Hcfel].
        apply andb_true_iff in Hcfct as [Hcfc Hcft].
        simpl in Hclean.
        set (inner_if := if is_varying_in_env env econd then Diverged else Converged).
        (* Split check_env *)
        assert (Hclean_c_br : check_env Converged env econd = [] /\
                               check_env inner_if env ethen = [] /\
                               check_env inner_if env eelse = []).
        { unfold inner_if.
          destruct (is_varying_in_env env econd);
          simpl in Hclean;
          apply app_eq_nil in Hclean as [HcC HtE];
          apply app_eq_nil in HtE as [HcT HcEl];
          exact (conj HcC (conj HcT HcEl)). }
        destruct Hclean_c_br as [Hcc [Hct Hcel]].
        (* Destruct eval of condition *)
        destruct (eval vary_val fuel' t1 rho1 econd) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' econd t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 econd) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' econd t2 rho2 cv2 trc2 Hcfc Hcond2). }
        (* Branch evaluation *)
        destruct (eval vary_val fuel' t1 rho1
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr1 trbr1]|] eqn:Hbr1;
          try discriminate.
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2
                   (if Nat.eqb cv2 0 then eelse else ethen)) as [[obr2 trbr2]|] eqn:Hbr2;
          try discriminate.
        inversion Heval2; subst.
        rewrite !erase_warp_app.
        f_equal.
        { exact (IH_A env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hcc Hcond1 Hcond2). }
        { (* Show branch traces have equal erase_warp *)
          destruct (is_varying_in_env env econd) eqn:Hcvar.
          - (* Condition varying: both branches in Diverged mode, no barriers *)
            unfold inner_if in Hct, Hcel.
            (* Thread 1's branch has no barriers *)
            assert (Hnb1 : no_barrier_event trbr1 = true).
            { destruct (Nat.eqb cv1 0) eqn:Hcv1z.
              - exact (check_env_diverged_no_barriers_ss vary_val env eelse
                         Hcfel Hcel fuel' t1 rho1 o1 trbr1 Hbr1).
              - exact (check_env_diverged_no_barriers_ss vary_val env ethen
                         Hcft Hct fuel' t1 rho1 o1 trbr1 Hbr1). }
            assert (Hnb2 : no_barrier_event trbr2 = true).
            { destruct (Nat.eqb cv2 0) eqn:Hcv2z.
              - exact (check_env_diverged_no_barriers_ss vary_val env eelse
                         Hcfel Hcel fuel' t2 rho2 o2 trbr2 Hbr2).
              - exact (check_env_diverged_no_barriers_ss vary_val env ethen
                         Hcft Hct fuel' t2 rho2 o2 trbr2 Hbr2). }
            rewrite (erase_warp_no_barrier _ Hnb1).
            rewrite (erase_warp_no_barrier _ Hnb2).
            reflexivity.
          - (* Condition non-varying: same branch taken *)
            unfold inner_if in Hct, Hcel.
            (* cv1 = cv2 from Part B *)
            assert (Hcveq : ONorm cv1 = ONorm cv2).
            { exact (IH_B env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                       Hcfc Hagr Hcvar Hcc Hcond1 Hcond2). }
            injection Hcveq as Hcveq'. subst cv2.
            (* Both take same branch *)
            destruct (Nat.eqb cv1 0) eqn:Hcv1z.
            + exact (IH_A env eelse t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                       Hcfel Hagr Hcel Hbr1 Hbr2).
            + exact (IH_A env ethen t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                       Hcft Hagr Hct Hbr1 Hbr2). }

      (* EWhile ec eb *)
      * apply andb_true_iff in Hcf as [Hcfc Hcfb].
        simpl in Hclean.
        set (inner_w := if is_varying_in_env env ec then Diverged else Converged).
        assert (Hclean_w : check_env Converged env ec = [] /\
                            check_env inner_w env eb = []).
        { unfold inner_w. destruct (is_varying_in_env env ec);
          simpl in Hclean; apply app_eq_nil in Hclean as [? ?]; auto. }
        destruct Hclean_w as [Hcc_w Hcb_w].
        destruct (is_varying_in_env env ec) eqn:Hcvar_w.
        -- (* Condition varying: use eval_while_exits_immediately_ss *)
           assert (HwcfW : core_frag_ss (EWhile ec eb) = true).
           { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
           destruct (eval_while_exits_immediately_ss vary_val (S fuel') ec eb
                       t1 rho1 o1 tr1 HwcfW Heval1) as [Hec1 Ho1].
           destruct (eval_while_exits_immediately_ss vary_val (S fuel') ec eb
                       t2 rho2 o2 tr2 HwcfW Heval2) as [Hec2 Ho2].
           (* Hec1 : eval fuel' t1 rho1 ec = Some(ONorm 0, tr1)
              Hec2 : eval fuel' t2 rho2 ec = Some(ONorm 0, tr2) *)
           simpl in Hec1, Hec2.
           exact (IH_A env ec t1 t2 rho1 rho2 (ONorm 0) (ONorm 0) tr1 tr2
                    Hcfc Hagr Hcc_w Hec1 Hec2).
        -- (* Condition non-varying: IH_B for condition *)
           unfold inner_w in Hcb_w.
           destruct (eval vary_val fuel' t1 rho1 ec) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
             try discriminate.
           2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ec t1 rho1 cv1 trc1 Hcfc Hcond1). }
           destruct (eval vary_val fuel' t2 rho2 ec) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
             try discriminate.
           2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ec t2 rho2 cv2 trc2 Hcfc Hcond2). }
           assert (Hcveq_w : ONorm cv1 = ONorm cv2).
           { exact (IH_B env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                      Hcfc Hagr Hcvar_w Hcc_w Hcond1 Hcond2). }
           injection Hcveq_w as Hcveq_w'. subst cv2.
           destruct (Nat.eqb cv1 0) eqn:Hcv1z.
           ++ (* cv = 0: exits immediately *)
              apply Nat.eqb_eq in Hcv1z. subst cv1.
              inversion Heval1; subst. inversion Heval2; subst.
              exact (IH_A env ec t1 t2 rho1 rho2 (ONorm 0) (ONorm 0) tr1 tr2
                       Hcfc Hagr Hcc_w Hcond1 Hcond2).
           ++ (* cv ≠ 0: loop continues *)
              destruct (eval vary_val fuel' t1 rho1 eb) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
                try discriminate.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t1 rho1 bv1 trb1 Hcfb Hbod1). }
              destruct (eval vary_val fuel' t1 rho1 (EWhile ec eb))
                as [[oloop1 trloop1]|] eqn:Hloop1; try discriminate.
              inversion Heval1; subst.
              destruct (eval vary_val fuel' t2 rho2 eb) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
                try discriminate.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t2 rho2 bv2 trb2 Hcfb Hbod2). }
              destruct (eval vary_val fuel' t2 rho2 (EWhile ec eb))
                as [[oloop2 trloop2]|] eqn:Hloop2; try discriminate.
              inversion Heval2; subst.
              assert (HwcfW : core_frag_ss (EWhile ec eb) = true).
              { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
              assert (Hwclean : check_env Converged env (EWhile ec eb) = []).
              { simpl. rewrite Hcvar_w. simpl.
                rewrite Hcc_w. rewrite Hcb_w. reflexivity. }
              rewrite !erase_warp_app. f_equal.
              { exact (IH_A env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv1) trc1 trc2
                         Hcfc Hagr Hcc_w Hcond1 Hcond2). }
              f_equal.
              { exact (IH_A env eb t1 t2 rho1 rho2 (ONorm bv1) (ONorm bv2) trb1 trb2
                         Hcfb Hagr Hcb_w Hbod1 Hbod2). }
              { exact (IH_A env (EWhile ec eb) t1 t2 rho1 rho2 o1 o2
                         trloop1 trloop2 HwcfW Hagr Hwclean Hloop1 Hloop2). }

      (* EFor elo ehi ebod *)
      * apply andb_true_iff in Hcf as [Hcflohi Hcfb].
        apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
        simpl in Hclean.
        set (inner_f := if is_varying_in_env env elo || is_varying_in_env env ehi
                        then Diverged else Converged).
        assert (Hclean_f : check_env Converged env elo = [] /\
                            check_env Converged env ehi = [] /\
                            check_env inner_f env ebod = []).
        { unfold inner_f.
          destruct (is_varying_in_env env elo || is_varying_in_env env ehi);
          simpl in Hclean;
          apply app_eq_nil in Hclean as [HcL HtE];
          apply app_eq_nil in HtE as [HcH HcB];
          exact (conj HcL (conj HcH HcB)). }
        destruct Hclean_f as [Hclo [Hchi Hcbod]].
        (* Destruct lo and hi for both threads *)
        destruct (eval vary_val fuel' t1 rho1 elo) as [[[lv1|lv1] trlo1]|] eqn:Hlo1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' elo t1 rho1 lv1 trlo1 Hcflo Hlo1). }
        destruct (eval vary_val fuel' t1 rho1 ehi) as [[[hv1|hv1] trhi1]|] eqn:Hhi1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ehi t1 rho1 hv1 trhi1 Hcfhi Hhi1). }
        destruct (eval vary_val fuel' t2 rho2 elo) as [[[lv2|lv2] trlo2]|] eqn:Hlo2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' elo t2 rho2 lv2 trlo2 Hcflo Hlo2). }
        destruct (eval vary_val fuel' t2 rho2 ehi) as [[[hv2|hv2] trhi2]|] eqn:Hhi2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ehi t2 rho2 hv2 trhi2 Hcfhi Hhi2). }
        (* Rewrite using for_loop_eq *)
        destruct (Nat.leb hv1 lv1) eqn:Hlb1.
        { inversion Heval1; subst.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst.
            rewrite !erase_warp_app. f_equal.
            { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                       Hcflo Hagr Hclo Hlo1 Hlo2). }
            { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                       Hcfhi Hagr Hchi Hhi1 Hhi2). } }
          (* t1 exits, t2 loops: need to show erase_warp equal *)
          { rewrite (for_loop_eq vary_val fuel' t2 rho2 ebod (hv2 - lv2)
                       (trlo2 ++ trhi2)) in Heval2.
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - (* Varying: body in Diverged mode, no barriers *)
              unfold inner_f in Hcbod.
              (* Destruct body eval for t2 to apply for_loop_erase_body_nil *)
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv2 - lv2) as [|steps_pos] eqn:Hsteps2.
                   { apply Nat.leb_nle in Hlb2. lia. }
                   simpl in Heval2. discriminate. }
              (* ONorm bv2: body has no barriers *)
              assert (Htrb2 : erase_warp trb2 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers_ss vary_val env ebod
                         Hcfb Hcbod fuel' t2 rho2 (ONorm bv2) trb2 Hbod2). }
              assert (Hnb2 := for_loop_erase_body_nil bv2 trb2 (hv2 - lv2) Htrb2 (trlo2 ++ trhi2)).
              rewrite Heval2 in Hnb2.
              (* Hnb2 : erase_warp tr2 = erase_warp (trlo2 ++ trhi2) *)
              inversion Heval1; subst.
              rewrite Hnb2. rewrite !erase_warp_app.
              f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - (* Non-varying lo/hi: same lv, hv from Part B *)
              assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              rewrite Hlb1 in Hlb2. discriminate. } }
        { (* t1 loops *)
          rewrite (for_loop_eq vary_val fuel' t1 rho1 ebod (hv1 - lv1)
                     (trlo1 ++ trhi1)) in Heval1.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { (* t2 exits: symmetric to above *)
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - unfold inner_f in Hcbod.
              (* Destruct body eval for t1 to apply for_loop_erase_body_nil *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              (* ONorm bv1: body has no barriers *)
              assert (Htrb1 : erase_warp trb1 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers_ss vary_val env ebod
                         Hcfb Hcbod fuel' t1 rho1 (ONorm bv1) trb1 Hbod1). }
              assert (Hnb1 := for_loop_erase_body_nil bv1 trb1 (hv1 - lv1) Htrb1 (trlo1 ++ trhi1)).
              rewrite Heval1 in Hnb1.
              (* Hnb1 : erase_warp tr1 = erase_warp (trlo1 ++ trhi1) *)
              inversion Heval2; subst.
              rewrite Hnb1. rewrite !erase_warp_app.
              f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              rewrite Hlb2 in Hlb1. discriminate. }
          { (* Both loop *)
            rewrite (for_loop_eq vary_val fuel' t2 rho2 ebod (hv2 - lv2)
                       (trlo2 ++ trhi2)) in Heval2.
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - (* Varying: body in Diverged mode *)
              unfold inner_f in Hcbod.
              (* Destruct body eval for t1 *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              assert (Htrb1 : erase_warp trb1 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers_ss vary_val env ebod
                         Hcfb Hcbod fuel' t1 rho1 (ONorm bv1) trb1 Hbod1). }
              assert (Hnb1 := for_loop_erase_body_nil bv1 trb1 (hv1 - lv1) Htrb1 (trlo1 ++ trhi1)).
              rewrite Heval1 in Hnb1.
              (* Hnb1 : erase_warp tr1 = erase_warp (trlo1 ++ trhi1) *)
              (* Destruct body eval for t2 *)
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv2 - lv2) as [|steps_pos] eqn:Hsteps2.
                   { apply Nat.leb_nle in Hlb2. lia. }
                   simpl in Heval2. discriminate. }
              assert (Htrb2 : erase_warp trb2 = []).
              { apply erase_warp_no_barrier.
                exact (check_env_diverged_no_barriers_ss vary_val env ebod
                         Hcfb Hcbod fuel' t2 rho2 (ONorm bv2) trb2 Hbod2). }
              assert (Hnb2 := for_loop_erase_body_nil bv2 trb2 (hv2 - lv2) Htrb2 (trlo2 ++ trhi2)).
              rewrite Heval2 in Hnb2.
              (* Hnb2 : erase_warp tr2 = erase_warp (trlo2 ++ trhi2) *)
              rewrite Hnb1. rewrite Hnb2.
              rewrite !erase_warp_app. f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - (* Non-varying lo/hi: same steps, use for_loop_erase_eq *)
              apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
              unfold inner_f in Hcbod.
              assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              (* Same steps: hv1 - lv1 *)
              assert (Herase_acc : erase_warp (trlo1 ++ trhi1) = erase_warp (trlo2 ++ trhi2)).
              { rewrite !erase_warp_app. f_equal.
                { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv1) trlo1 trlo2
                           Hcflo Hagr Hclo Hlo1 Hlo2). }
                { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv1) trhi1 trhi2
                           Hcfhi Hagr Hchi Hhi1 Hhi2). } }
              (* Destruct body evaluations to apply for_loop_erase_body_eq *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval2. discriminate. }
              (* ONorm bv1, ONorm bv2: apply IH_A for body erase, then for_loop_erase_body_eq *)
              assert (Hbody_erase : erase_warp trb1 = erase_warp trb2).
              { exact (IH_A env ebod t1 t2 rho1 rho2 (ONorm bv1) (ONorm bv2) trb1 trb2
                           Hcfb Hagr Hcbod Hbod1 Hbod2). }
              assert (Hfl := for_loop_erase_body_eq bv1 bv2 trb1 trb2 (hv1 - lv1)
                               Hbody_erase (trlo1 ++ trhi1) (trlo2 ++ trhi2) Herase_acc).
              rewrite Heval1 in Hfl. rewrite Heval2 in Hfl. exact Hfl. } }

      (* ESeq ess *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag_ss ess = true) by exact Hcf.
        clear Hcf.
        (* We prove erase_warp equality by induction on ess,
           generalizing over the accumulator. *)
        assert (Hgen : forall xs,
          forallb core_frag_ss xs = true ->
          concat (map (check_env Converged env) xs) = [] ->
          forall acc1 acc2 o1' o2' tr1' tr2',
          erase_warp acc1 = erase_warp acc2 ->
          (fix eval_seq xs0 acc : option _ :=
            match xs0 with [] => Some (ONorm 0, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
              | None => None end end) xs acc1 = Some (o1', tr1') ->
          (fix eval_seq xs0 acc : option _ :=
            match xs0 with [] => Some (ONorm 0, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
              | None => None end end) xs acc2 = Some (o2', tr2') ->
          erase_warp tr1' = erase_warp tr2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ acc1 acc2 o1' o2' tr1' tr2' Hacc Hs1 Hs2.
            inversion Hs1; inversion Hs2; subst. exact Hacc.
          - intros Hcfal Hcln acc1 acc2 o1' o2' tr1' tr2' Hacc Hs1 Hs2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Hs1, Hs2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            apply IHtl with (acc1 := acc1 ++ trh1) (acc2 := acc2 ++ trh2)
              (o1' := o1') (o2' := o2').
            + exact Hcftl.
            + exact Hctl.
            + rewrite !erase_warp_app. f_equal.
              * exact Hacc.
              * exact (IH_A env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                         Hcfh Hagr Hch Hh1 Hh2).
            + exact Hs1.
            + exact Hs2. }
        exact (Hgen ess Hcfall Hclean [] [] o1 o2 tr1 tr2 eq_refl Heval1 Heval2).

      (* ELet xn ev ebody *)
      * apply andb_true_iff in Hcf as [Hcfv Hcfb].
        simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcv Hcb].
        (* Destruct eval of ev for both threads *)
        destruct (eval vary_val fuel' t1 rho1 ev) as [[[vv1|vv1] trv1]|] eqn:Hev1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ev t1 rho1 vv1 trv1 Hcfv Hev1). }
        destruct (eval vary_val fuel' t2 rho2 ev) as [[[vv2|vv2] trv2]|] eqn:Hev2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ev t2 rho2 vv2 trv2 Hcfv Hev2). }
        (* Destruct eval of body for both threads *)
        destruct (eval vary_val fuel' t1 (venv_extend rho1 xn vv1) ebody)
          as [[ob1 trb1]|] eqn:Hbod1; try discriminate.
        destruct (eval vary_val fuel' t2 (venv_extend rho2 xn vv2) ebody)
          as [[ob2 trb2]|] eqn:Hbod2; try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        rewrite !erase_warp_app. f_equal.
        { exact (IH_A env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                   Hcfv Hagr Hcv Hev1 Hev2). }
        { (* Build env_agrees for extended env *)
          set (vv := is_varying_in_env env ev).
          assert (Hagr' : env_agrees (env_extend env xn vv)
                            (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)).
          { unfold vv.
            destruct (is_varying_in_env env ev) eqn:Hvv.
            - (* ev varying: xn is varying, no constraint on vv1 vs vv2 *)
              intros y Hlookup.
              assert (Hxy : xn <> y).
              { intro Heq. subst. rewrite env_lookup_extend_same in Hlookup. discriminate. }
              rewrite (venv_lookup_extend_diff rho1 xn y vv1 Hxy).
              rewrite (venv_lookup_extend_diff rho2 xn y vv2 Hxy).
              apply Hagr.
              unfold env_lookup, env_extend in Hlookup |- *.
              simpl find in *.
              destruct (Nat.eqb xn y) eqn:Heqn.
              { apply Nat.eqb_eq in Heqn. subst. exfalso. exact (Hxy eq_refl). }
              { exact Hlookup. }
            - (* ev non-varying: vv1 = vv2 from IH_B *)
              assert (Hveq : ONorm vv1 = ONorm vv2).
              { exact (IH_B env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                         Hcfv Hagr Hvv Hcv Hev1 Hev2). }
              injection Hveq as Hveq'. subst vv2.
              exact (env_agrees_extend env rho1 rho2 xn vv1 false Hagr). }
          exact (IH_A (env_extend env xn vv) ebody t1 t2
                   (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)
                   o1 o2 trb1 trb2 Hcfb Hagr' Hcb Hbod1 Hbod2). }

      (* ESuperstep dv ebod econt: implicit boundary barrier on every thread *)
      * apply andb_true_iff in Hcf as [Hcfdvb Hcfc].
        apply andb_true_iff in Hcfdvb as [_ Hcfb].
        simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcb Hcc].
        destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1 Hcfb Hbod1). }
        destruct (eval vary_val fuel' t1 rho1 econt) as [[oc1 trc1]|] eqn:Hcon1;
          try discriminate.
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2 Hcfb Hbod2). }
        destruct (eval vary_val fuel' t2 rho2 econt) as [[oc2 trc2]|] eqn:Hcon2;
          try discriminate.
        inversion Heval2; subst.
        rewrite !erase_warp_app. simpl. f_equal.
        { exact (IH_A env ebod t1 t2 rho1 rho2 (ONorm bv1) (ONorm bv2) trb1 trb2
                   Hcfb Hagr Hcb Hbod1 Hbod2). }
        f_equal.
        { exact (IH_A env econt t1 t2 rho1 rho2 o1 o2 trc1 trc2
                   Hcfc Hagr Hcc Hcon1 Hcon2). }

      (* EApp eargs *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag_ss eargs = true) by exact Hcf.
        clear Hcf.
        assert (Hgen : forall xs,
          forallb core_frag_ss xs = true ->
          concat (map (check_env Converged env) xs) = [] ->
          forall acc1 acc2 lv1 lv2 o1' o2' tr1' tr2',
          erase_warp acc1 = erase_warp acc2 ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc1 lv1 = Some (o1', tr1') ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc2 lv2 = Some (o2', tr2') ->
          erase_warp tr1' = erase_warp tr2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hacc Ha1 Ha2.
            inversion Ha1; inversion Ha2; subst. exact Hacc.
          - intros Hcfal Hcln acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hacc Ha1 Ha2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Ha1, Ha2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            apply IHtl with (acc1 := acc1 ++ trh1) (acc2 := acc2 ++ trh2)
              (lv1 := vh1) (lv2 := vh2) (o1' := o1') (o2' := o2').
            + exact Hcftl.
            + exact Hctl.
            + rewrite !erase_warp_app. f_equal.
              * exact Hacc.
              * exact (IH_A env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                         Hcfh Hagr Hch Hh1 Hh2).
            + exact Ha1.
            + exact Ha2. }
        exact (Hgen eargs Hcfall Hclean [] [] 0 0 o1 o2 tr1 tr2 eq_refl Heval1 Heval2).

(* ------------------------------------------------------------------ *)
(* Part B for S fuel'                                                   *)
(* ------------------------------------------------------------------ *)
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hcf Hagr Hvar Hclean Heval1 Heval2.
      destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                    | elo ehi ebod | ess | xn ev ebody | dv ebod econt | eargs | er ];
      simpl in Hcf; simpl in Hvar; simpl in Heval1; simpl in Heval2; try discriminate.

      (* ELit: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVary: closed by try discriminate above (Hvar : true = false) *)
      (* EBarrier: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EWarpPoint: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVar n0: env_agrees gives same value *)
      * inversion Heval1; inversion Heval2. subst. f_equal. apply Hagr. exact Hvar.
      (* EBinop ea eb *)
      * apply andb_true_iff in Hcf as [Hcfa Hcfb].
        apply Bool.orb_false_iff in Hvar as [Hva Hvb].
        simpl in Hclean. apply app_eq_nil in Hclean as [Hca Hcb].
        destruct (eval vary_val fuel' t1 rho1 ea) as [[[va1|va1] tra1]|] eqn:Hea1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ea t1 rho1 va1 tra1 Hcfa Hea1). }
        destruct (eval vary_val fuel' t1 rho1 eb) as [[[vb1|vb1] trb1]|] eqn:Heb1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t1 rho1 vb1 trb1 Hcfb Heb1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ea) as [[[va2|va2] tra2]|] eqn:Hea2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ea t2 rho2 va2 tra2 Hcfa Hea2). }
        destruct (eval vary_val fuel' t2 rho2 eb) as [[[vb2|vb2] trb2]|] eqn:Heb2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t2 rho2 vb2 trb2 Hcfb Heb2). }
        inversion Heval2; subst.
        assert (Hva_eq : ONorm va1 = ONorm va2).
        { exact (IH_B env ea t1 t2 rho1 rho2 (ONorm va1) (ONorm va2) tra1 tra2
                   Hcfa Hagr Hva Hca Hea1 Hea2). }
        assert (Hvb_eq : ONorm vb1 = ONorm vb2).
        { exact (IH_B env eb t1 t2 rho1 rho2 (ONorm vb1) (ONorm vb2) trb1 trb2
                   Hcfb Hagr Hvb Hcb Heb1 Heb2). }
        injection Hva_eq as Hva_eq. injection Hvb_eq as Hvb_eq.
        subst va2 vb2. reflexivity.
      (* EUnop eu *)
      * destruct (eval vary_val fuel' t1 rho1 eu) as [[[vu1|vu1] tru1]|] eqn:Heu1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eu t1 rho1 vu1 tru1 Hcf Heu1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 eu) as [[[vu2|vu2] tru2]|] eqn:Heu2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eu t2 rho2 vu2 tru2 Hcf Heu2). }
        inversion Heval2; subst.
        simpl in Hclean.
        assert (Hvu_eq : ONorm vu1 = ONorm vu2).
        { exact (IH_B env eu t1 t2 rho1 rho2 (ONorm vu1) (ONorm vu2) tr1 tr2
                   Hcf Hagr Hvar Hclean Heu1 Heu2). }
        injection Hvu_eq as Hvu_eq. subst vu2. reflexivity.
      (* EIf econd ethen eelse *)
      * apply andb_true_iff in Hcf as [Hcfct Hcfel].
        apply andb_true_iff in Hcfct as [Hcfc Hcft].
        apply Bool.orb_false_iff in Hvar as [Hvctel Hvel].
        apply Bool.orb_false_iff in Hvctel as [Hvc Hvt].
        (* cond non-varying: inner = Converged *)
        simpl in Hclean. rewrite Hvc in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcc H'].
        apply app_eq_nil in H' as [Hct Hcel].
        destruct (eval vary_val fuel' t1 rho1 econd) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' econd t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 econd) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' econd t2 rho2 cv2 trc2 Hcfc Hcond2). }
        assert (Hcveq : ONorm cv1 = ONorm cv2).
        { exact (IH_B env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hvc Hcc Hcond1 Hcond2). }
        injection Hcveq as Hcveq'. subst cv2.
        destruct (eval vary_val fuel' t1 rho1
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr1 trbr1]|] eqn:Hbr1;
          try discriminate.
        destruct (eval vary_val fuel' t2 rho2
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr2 trbr2]|] eqn:Hbr2;
          try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        destruct (Nat.eqb cv1 0) eqn:Hcv1z.
        -- exact (IH_B env eelse t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                   Hcfel Hagr Hvel Hcel Hbr1 Hbr2).
        -- exact (IH_B env ethen t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                   Hcft Hagr Hvt Hct Hbr1 Hbr2).
      (* EWhile ec eb *)
      * apply andb_true_iff in Hcf as [Hcfc Hcfb].
        apply Bool.orb_false_iff in Hvar as [Hvc_w Hvb_w].
        (* cond non-varying: inner = Converged *)
        simpl in Hclean. rewrite Hvc_w in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcc_w Hcb_w].
        destruct (eval vary_val fuel' t1 rho1 ec) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ec t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 ec) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ec t2 rho2 cv2 trc2 Hcfc Hcond2). }
        assert (Hcveq_w : ONorm cv1 = ONorm cv2).
        { exact (IH_B env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hvc_w Hcc_w Hcond1 Hcond2). }
        injection Hcveq_w as Hcveq_w'. subst cv2.
        destruct (Nat.eqb cv1 0) eqn:Hcv1z.
        -- (* cv = 0: both return ONorm 0 *)
           inversion Heval1; subst. inversion Heval2; subst. reflexivity.
        -- (* cv ≠ 0: recurse *)
           destruct (eval vary_val fuel' t1 rho1 eb) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
             try discriminate.
           2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t1 rho1 bv1 trb1 Hcfb Hbod1). }
           destruct (eval vary_val fuel' t1 rho1 (EWhile ec eb)) as [[oloop1 trloop1]|]
             eqn:Hloop1; try discriminate.
           inversion Heval1; subst.
           destruct (eval vary_val fuel' t2 rho2 eb) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
             try discriminate.
           2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' eb t2 rho2 bv2 trb2 Hcfb Hbod2). }
           destruct (eval vary_val fuel' t2 rho2 (EWhile ec eb)) as [[oloop2 trloop2]|]
             eqn:Hloop2; try discriminate.
           inversion Heval2; subst.
           assert (HwcfW : core_frag_ss (EWhile ec eb) = true).
           { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
           assert (HwvarW : is_varying_in_env env (EWhile ec eb) = false).
           { simpl. apply Bool.orb_false_iff. exact (conj Hvc_w Hvb_w). }
           assert (HwcleanW : check_env Converged env (EWhile ec eb) = []).
           { simpl. rewrite Hvc_w. simpl. rewrite Hcc_w. rewrite Hcb_w. reflexivity. }
           exact (IH_B env (EWhile ec eb) t1 t2 rho1 rho2 o1 o2 trloop1 trloop2
                    HwcfW Hagr HwvarW HwcleanW Hloop1 Hloop2).
      (* EFor elo ehi ebod: outcome is always ONorm 0 for core_frag_ss *)
      * apply andb_true_iff in Hcf as [Hcflohi Hcfb].
        apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
        apply Bool.orb_false_iff in Hvar as [Hvlohi Hvb].
        apply Bool.orb_false_iff in Hvlohi as [Hvlo Hvhi].
        simpl in Hclean. rewrite Hvlo, Hvhi in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hclo H'].
        apply app_eq_nil in H' as [Hchi Hcb].
        destruct (eval vary_val fuel' t1 rho1 elo) as [[[lv1|lv1] trlo1]|] eqn:Hlo1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' elo t1 rho1 lv1 trlo1 Hcflo Hlo1). }
        destruct (eval vary_val fuel' t1 rho1 ehi) as [[[hv1|hv1] trhi1]|] eqn:Hhi1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ehi t1 rho1 hv1 trhi1 Hcfhi Hhi1). }
        destruct (eval vary_val fuel' t2 rho2 elo) as [[[lv2|lv2] trlo2]|] eqn:Hlo2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' elo t2 rho2 lv2 trlo2 Hcflo Hlo2). }
        destruct (eval vary_val fuel' t2 rho2 ehi) as [[[hv2|hv2] trhi2]|] eqn:Hhi2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ehi t2 rho2 hv2 trhi2 Hcfhi Hhi2). }
        (* Both outcomes are ONorm 0 regardless of loop body, for core_frag_ss ebod *)
        (* Use for_loop_fixed approach with inline loop outcome lemma *)
        assert (Hfl1 : forall k acc r,
          (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
            match k0 with
            | O => Some (ONorm 0, acc_tr)
            | S k' =>
                match eval vary_val fuel' t1 rho1 ebod with
                | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                | None => None end end) k acc = Some r ->
          fst r = ONorm 0).
        { intros k. induction k as [|k' IHk'].
          - intros acc r H. simpl in H. inversion H. reflexivity.
          - intros acc r H. simpl in H.
            destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv|bv] trb]|] eqn:Hbod.
            + exact (IHk' _ _ H).
            + exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t1 rho1 bv trb Hcfb Hbod).
            + discriminate. }
        assert (Hfl2 : forall k acc r,
          (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
            match k0 with
            | O => Some (ONorm 0, acc_tr)
            | S k' =>
                match eval vary_val fuel' t2 rho2 ebod with
                | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                | None => None end end) k acc = Some r ->
          fst r = ONorm 0).
        { intros k. induction k as [|k' IHk'].
          - intros acc r H. simpl in H. inversion H. reflexivity.
          - intros acc r H. simpl in H.
            destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv|bv] trb]|] eqn:Hbod.
            + exact (IHk' _ _ H).
            + exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t2 rho2 bv trb Hcfb Hbod).
            + discriminate. }
        (* Destruct leb in Heval1/Heval2 to distinguish base vs loop case *)
        destruct (Nat.leb hv1 lv1) eqn:Hlb1.
        { inversion Heval1; subst.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst. reflexivity. }
          { apply Hfl2 in Heval2. simpl in Heval2. destruct o2; congruence. } }
        { destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst.
            apply Hfl1 in Heval1. simpl in Heval1. destruct o1; congruence. }
          { apply Hfl1 in Heval1. apply Hfl2 in Heval2.
            destruct o1, o2; simpl in *; congruence. } }
      (* ESeq ess: outcome is always ONorm 0 for core_frag_ss *)
      * simpl in Hclean.
        (* Both return ONorm 0: seq never returns ORet with core_frag_ss *)
        (* All core_frag_ss seqs return ONorm 0 *)
        assert (Hgen1 : forall xs acc o tr,
          forallb core_frag_ss xs = true ->
          (fix eval_seq xs0 acc_tr : option (outcome * trace) :=
            match xs0 with [] => Some (ONorm 0, acc_tr) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr') => Some (ORet v, acc_tr ++ tr')
              | Some (ONorm _, tr') => eval_seq rest (acc_tr ++ tr')
              | None => None end end) xs acc = Some (o, tr) ->
          o = ONorm 0).
        { intros xs. induction xs as [|h tl IHtl]; intros acc o tr Hcfs He.
          - inversion He. reflexivity.
          - simpl in He. simpl in Hcfs. apply andb_true_iff in Hcfs as [Hcfh Hcftl].
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh|vh] trh]|] eqn:Hh;
              try discriminate.
            + exact (IHtl _ _ _ Hcftl He).
            + exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t1 rho1 vh trh Hcfh Hh). }
        assert (Hgen2 : forall xs acc o tr,
          forallb core_frag_ss xs = true ->
          (fix eval_seq xs0 acc_tr : option (outcome * trace) :=
            match xs0 with [] => Some (ONorm 0, acc_tr) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr') => Some (ORet v, acc_tr ++ tr')
              | Some (ONorm _, tr') => eval_seq rest (acc_tr ++ tr')
              | None => None end end) xs acc = Some (o, tr) ->
          o = ONorm 0).
        { intros xs. induction xs as [|h tl IHtl]; intros acc o tr Hcfs He.
          - inversion He. reflexivity.
          - simpl in He. simpl in Hcfs. apply andb_true_iff in Hcfs as [Hcfh Hcftl].
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh|vh] trh]|] eqn:Hh;
              try discriminate.
            + exact (IHtl _ _ _ Hcftl He).
            + exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t2 rho2 vh trh Hcfh Hh). }
        assert (Ho1 : o1 = ONorm 0) by exact (Hgen1 ess [] o1 tr1 Hcf Heval1).
        assert (Ho2 : o2 = ONorm 0) by exact (Hgen2 ess [] o2 tr2 Hcf Heval2).
        subst o1 o2. reflexivity.
      (* ELet xn ev ebody *)
      * apply andb_true_iff in Hcf as [Hcfv Hcfb].
        simpl in Hvar. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcv Hcb].
        destruct (eval vary_val fuel' t1 rho1 ev) as [[[vv1|vv1] trv1]|] eqn:Hev1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ev t1 rho1 vv1 trv1 Hcfv Hev1). }
        destruct (eval vary_val fuel' t2 rho2 ev) as [[[vv2|vv2] trv2]|] eqn:Hev2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ev t2 rho2 vv2 trv2 Hcfv Hev2). }
        destruct (eval vary_val fuel' t1 (venv_extend rho1 xn vv1) ebody)
          as [[ob1 trb1]|] eqn:Hbod1; try discriminate.
        destruct (eval vary_val fuel' t2 (venv_extend rho2 xn vv2) ebody)
          as [[ob2 trb2]|] eqn:Hbod2; try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        (* Build env_agrees for extended env *)
        set (vvflag := is_varying_in_env env ev).
        assert (Hagr' : env_agrees (env_extend env xn vvflag)
                          (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)).
        { unfold vvflag. destruct (is_varying_in_env env ev) eqn:Hvv.
          - intros y Hlookup.
            assert (Hxy : xn <> y).
            { intro Heq. subst. rewrite env_lookup_extend_same in Hlookup. discriminate. }
            rewrite (venv_lookup_extend_diff rho1 xn y vv1 Hxy).
            rewrite (venv_lookup_extend_diff rho2 xn y vv2 Hxy).
            apply Hagr.
            unfold env_lookup, env_extend in Hlookup |- *.
            simpl find in *.
            destruct (Nat.eqb xn y) eqn:Heqn.
            { apply Nat.eqb_eq in Heqn. subst. exfalso. exact (Hxy eq_refl). }
            { exact Hlookup. }
          - assert (Hveq : ONorm vv1 = ONorm vv2).
            { exact (IH_B env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                       Hcfv Hagr Hvv Hcv Hev1 Hev2). }
            injection Hveq as Hveq'. subst vv2.
            exact (env_agrees_extend env rho1 rho2 xn vv1 false Hagr). }
        exact (IH_B (env_extend env xn vvflag) ebody t1 t2
                 (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)
                 o1 o2 trb1 trb2 Hcfb Hagr' Hvar Hcb Hbod1 Hbod2).
      (* ESuperstep dv ebod econt: non-varying => cont outcome agrees *)
      * apply andb_true_iff in Hcf as [Hcfdvb Hcfc].
        apply andb_true_iff in Hcfdvb as [_ Hcfb].
        apply Bool.orb_false_iff in Hvar as [Hvb Hvc].
        simpl in Hclean. apply app_eq_nil in Hclean as [Hcb Hcc].
        destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1 Hcfb Hbod1). }
        destruct (eval vary_val fuel' t1 rho1 econt) as [[oc1 trc1]|] eqn:Hcon1;
          try discriminate.
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
          try discriminate.
        2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2 Hcfb Hbod2). }
        destruct (eval vary_val fuel' t2 rho2 econt) as [[oc2 trc2]|] eqn:Hcon2;
          try discriminate.
        inversion Heval2; subst.
        exact (IH_B env econt t1 t2 rho1 rho2 o1 o2 trc1 trc2
                 Hcfc Hagr Hvc Hcc Hcon1 Hcon2).
      (* EApp eargs: last value equality *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag_ss eargs = true) by exact Hcf.
        clear Hcf.
        assert (Hgen : forall xs,
          forallb core_frag_ss xs = true ->
          concat (map (check_env Converged env) xs) = [] ->
          existsb (is_varying_in_env env) xs = false ->
          forall acc1 acc2 lv1 lv2 o1' o2' tr1' tr2',
          lv1 = lv2 ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc1 lv1 = Some (o1', tr1') ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc2 lv2 = Some (o2', tr2') ->
          o1' = o2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ _ acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hlv Ha1 Ha2.
            inversion Ha1; inversion Ha2; subst. congruence.
          - intros Hcfal Hcln Hv acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hlv Ha1 Ha2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Hv. apply Bool.orb_false_iff in Hv as [Hvh Hvtl].
            simpl in Ha1, Ha2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_ss_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            assert (Hvheq : ONorm vh1 = ONorm vh2).
            { exact (IH_B env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                       Hcfh Hagr Hvh Hch Hh1 Hh2). }
            injection Hvheq as Hvheq'. subst vh2.
            exact (IHtl Hcftl Hctl Hvtl (acc1 ++ trh1) (acc2 ++ trh2) vh1 vh1
                     o1' o2' tr1' tr2' eq_refl Ha1 Ha2). }
        exact (Hgen eargs Hcfall Hclean Hvar [] [] 0 0 o1 o2 tr1 tr2 eq_refl Heval1 Heval2).
      (* EReturn: core_frag_ss = false *)
Qed.

(* ----------------------------------------------------------------------- *)
(* 11.5  check_env_sound_superstep — soundness over core_frag_ss            *)
(* ----------------------------------------------------------------------- *)

(** check_env_sound_superstep: if core_frag_ss e = true (the enlarged fragment
    including ESuperstep) and check_env Converged env e = [], then e is
    barrier_safe. Uniform-reachability supersteps emit [EvBarrier] on all
    threads (traces agree); thread-varying entry into a dv=false superstep is
    excluded because the Diverged-mode checker raises [BarrierError] for it.
    This is the runtime grounding of the static F-01 verdict
    (superstep_outer_diverged_error). *)
Theorem check_env_sound_superstep :
  forall vary_val env e,
    core_frag_ss e = true ->
    check_env Converged env e = [] ->
    barrier_safe vary_val env e.
Proof.
  intros vary_val env e Hcf Hclean.
  unfold barrier_safe.
  intros fuel t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hagr Heval1 Heval2.
  exact (proj1 (eval_check_uniform_ss vary_val fuel) env e t1 t2 rho1 rho2 o1 o2 tr1 tr2
           Hcf Hagr Hclean Heval1 Heval2).
Qed.

(* ----------------------------------------------------------------------- *)
(* 11.6  semantic_f01_corollary — runtime meaning of the static F-01 verdict *)
(* ----------------------------------------------------------------------- *)

(** The static checker (ConvergenceSpec.superstep_outer_diverged_error) proves
    that a dv=false superstep entered under a Diverged mode flags a BarrierError.
    The env-threaded check_env realizes this for a thread-varying entry: when
    an EIf condition is varying (EVary), both branches are checked in Diverged
    mode, and a dv=false ESuperstep in a branch then yields [BarrierError].

    The witness expression couples the static verdict to a concrete runtime
    counterexample:
        susp_hazard := EIf EVary (ESuperstep false EBarrier ELit) ELit
    - EVary is thread-varying, so check_env enters the then-branch in Diverged
      mode and the dv=false ESuperstep raises [BarrierError]: the checker flags
      it (semantic_f01_flagged).
    - Operationally the branch is taken per-thread on the EVary value; threads
      taking the then-branch evaluate the ESuperstep (body EBarrier then the
      implicit boundary barrier => trace [EvBarrier; EvBarrier]), while threads
      taking the else-branch (ELit) emit no barrier (trace []). The barrier
      traces diverge, so susp_hazard is NOT barrier_safe
      (semantic_f01_not_barrier_safe). *)

Definition susp_hazard : expr :=
  EIf EVary (ESuperstep false EBarrier ELit) ELit.

(** semantic_f01_flagged: the env-threaded checker flags susp_hazard. *)
Lemma semantic_f01_flagged :
  check_env Converged [] susp_hazard <> [].
Proof.
  unfold susp_hazard. simpl. intro H. discriminate.
Qed.

(** Concrete thread-varying witness: thread 0 takes the then-branch (the
    superstep), thread 1 takes the else-branch (ELit). *)
Definition susp_vary (n : tid) : value :=
  match n with O => 1 | S _ => 0 end.

(** Thread 0 (EVary => susp_vary 0 = 1, nonzero => then-branch) evaluates the
    ESuperstep: its body EBarrier emits [EvBarrier], then the implicit boundary
    emits another, giving trace [EvBarrier; EvBarrier]. *)
Lemma susp_eval_thread0 :
  eval susp_vary 6 0 [] susp_hazard = Some (ONorm 0, [EvBarrier; EvBarrier]).
Proof. reflexivity. Qed.

(** Thread 1 (EVary => susp_vary 1 = 0 => else-branch ELit) emits no barrier. *)
Lemma susp_eval_thread1 :
  eval susp_vary 6 1 [] susp_hazard = Some (ONorm 0, []).
Proof. reflexivity. Qed.

(** semantic_f01_not_barrier_safe: susp_hazard is NOT barrier_safe, witnessed
    by the two threads above whose barrier traces ([EvBarrier; EvBarrier] vs [])
    differ even though both env-agree on the empty environment. *)
Theorem semantic_f01_not_barrier_safe :
  ~ barrier_safe susp_vary [] susp_hazard.
Proof.
  unfold barrier_safe. intro Hsafe.
  assert (Hagr : env_agrees [] [] []).
  { intros y _. reflexivity. }
  pose proof (Hsafe 6 0 1 [] [] (ONorm 0) (ONorm 0)
                [EvBarrier; EvBarrier] []
                Hagr susp_eval_thread0 susp_eval_thread1) as Heq.
  unfold erase_warp in Heq. simpl in Heq. discriminate.
Qed.

(** semantic_f01_corollary: the F-01 verdict has runtime force. The witness
    susp_hazard is simultaneously (a) flagged by the env-threaded checker and
    (b) genuinely not barrier_safe. The checker's BarrierError is therefore a
    SOUND rejection: it catches a real barrier-divergence hazard. This grounds
    ConvergenceSpec.superstep_outer_diverged_error operationally. *)
Theorem semantic_f01_corollary :
  check_env Converged [] susp_hazard <> [] /\ ~ barrier_safe susp_vary [] susp_hazard.
Proof.
  exact (conj semantic_f01_flagged semantic_f01_not_barrier_safe).
Qed.

(* ======================================================================= *)
(* ===== 12. T3-S7 — Warp-collective semantic soundness ================== *)
(* ======================================================================= *)

(*
 * T3-S7 design notes:
 *
 * 1. PARAMETRIZATION ATTEMPT (flagged per task).  The plan asked whether the
 *    T3-S4 eval_check_uniform induction could be made parametric over
 *    (event class, agreement domain, checker) so that T3-S7 becomes an
 *    instantiation.  This was attempted and found INFEASIBLE within budget:
 *    every leaf/structural case of eval_check_uniform reduces the CONCRETE
 *    checker Fixpoint via `simpl; apply app_eq_nil` (e.g. check_env Converged
 *    EBarrier = []), and the leaf inversions (inversion Heval; reflexivity)
 *    depend on the concrete event constructors.  A genuinely parametric
 *    induction would require threading ~15 checker-algebra lemmas as Section
 *    hypotheses (mode propagation, per-node app-split, leaf silence) — itself a
 *    ~900-line abstraction with no working template to debug against.  We
 *    therefore DUPLICATE the induction (eval_check_warp_uniform below) with a
 *    mechanical EBarrier<->EWarpPoint / erase_warp<->erase_barrier /
 *    check_env<->check_warp_env substitution.  The two for-loop accumulator
 *    helpers and the trace-silence theorem ARE shared in spirit but restated
 *    for erase_barrier (15-line proofs each).
 *
 * 2. erase_barrier is the DUAL projection to erase_warp: it keeps EvWarp and
 *    discards EvBarrier.  warp_safe is stated with erase_barrier equality,
 *    restricted (via warp_of) to thread pairs in the same warp.  A varying-EIf
 *    where one branch contains EWarpPoint and the other does not produces
 *    diverging warp traces; check_warp_env catches exactly this by checking
 *    both branches in Diverged mode, where EWarpPoint flags WarpError.
 *
 * 3. The warp restriction warp_of t1 = warp_of t2 is a WEAKENING of the
 *    conclusion: eval does not depend on warp_of, so the core uniformity holds
 *    for ALL thread pairs; warp_safe (same-warp pairs only) follows a fortiori.
 *)

(* ----------------------------------------------------------------------- *)
(* 12.1  erase_barrier — project trace to warp events only                  *)
(* ----------------------------------------------------------------------- *)

(** erase_barrier tr: keep only EvWarp events, discard EvBarrier.
    Dual of erase_warp.  warp_safe uses erase_barrier equality: same-warp
    threads issue warp-collective ops in the same order. *)
Definition erase_barrier (tr : trace) : trace :=
  filter (fun ev => match ev with EvWarp => true | EvBarrier => false end) tr.

Lemma erase_barrier_app : forall tr1 tr2,
  erase_barrier (tr1 ++ tr2) = erase_barrier tr1 ++ erase_barrier tr2.
Proof.
  intros tr1 tr2. unfold erase_barrier. apply filter_app.
Qed.

(** no_warp_event: the trace contains no EvWarp events. *)
Definition no_warp_event (tr : trace) : bool :=
  forallb (fun ev => match ev with EvWarp => false | _ => true end) tr.

Lemma no_warp_app : forall tr1 tr2,
  no_warp_event (tr1 ++ tr2) = no_warp_event tr1 && no_warp_event tr2.
Proof.
  intros tr1 tr2. unfold no_warp_event. apply forallb_app.
Qed.

Lemma erase_barrier_no_warp : forall tr,
  no_warp_event tr = true -> erase_barrier tr = [].
Proof.
  intros tr H.
  unfold erase_barrier, no_warp_event in *.
  induction tr as [| ev tr' IHtr].
  - reflexivity.
  - simpl in H. apply andb_true_iff in H as [Hev Htr'].
    simpl. destruct ev.
    + (* EvBarrier *) exact (IHtr Htr').
    + (* EvWarp *) simpl in Hev. discriminate.
Qed.

(* ----------------------------------------------------------------------- *)
(* 12.2  warp_free — e contains no EWarpPoint                               *)
(* ----------------------------------------------------------------------- *)

(** warp_free: dual of barrier_free.  EWarpPoint is the only warp-collective
    leaf, so it is the single false case.  ESuperstep emits only EvBarrier
    (never EvWarp), so a superstep is warp_free iff both sub-expressions are. *)
Fixpoint warp_free (e : expr) : bool :=
  match e with
  | EWarpPoint     => false
  | ELit | EVary | EBarrier | EVar _ => true
  | EBinop a b     => warp_free a && warp_free b
  | EUnop e        => warp_free e
  | EIf c t el     => warp_free c && warp_free t && warp_free el
  | EWhile c b     => warp_free c && warp_free b
  | EFor lo hi b   => warp_free lo && warp_free hi && warp_free b
  | ESeq es        => forallb warp_free es
  | ELet _ v b     => warp_free v && warp_free b
  | ESuperstep _ body cont => warp_free body && warp_free cont
  | EApp args      => forallb warp_free args
  | EReturn e      => warp_free e
  end.


(* ----------------------------------------------------------------------- *)
(* 12.3  warp-trace silence helpers (mirror of the no_barrier helpers)      *)
(* ----------------------------------------------------------------------- *)

(** for_loop_fixed_no_warp: the EFor body loop preserves no_warp_event. *)
Lemma for_loop_fixed_no_warp :
  forall body k acc o tr,
    no_warp_event acc = true ->
    (forall o' tr', body = Some (o', tr') -> no_warp_event tr' = true) ->
    for_loop_fixed body k acc = Some (o, tr) ->
    no_warp_event tr = true.
Proof.
  intros body k.
  induction k as [| k' IHk].
  - intros acc o tr Hacc _ H. simpl in H. inversion H. subst. exact Hacc.
  - intros acc o tr Hacc Hbody H.
    simpl in H.
    destruct body as [[[bv | bv] tr_b] |] eqn:Ebody.
    + apply IHk with (acc := acc ++ tr_b) (o := o) (tr := tr).
      * rewrite no_warp_app. apply andb_true_iff; split.
        { exact Hacc. } { exact (Hbody (ONorm bv) tr_b eq_refl). }
      * exact Hbody.
      * exact H.
    + inversion H. subst.
      rewrite no_warp_app. apply andb_true_iff; split.
      * exact Hacc. * exact (Hbody (ORet bv) tr_b eq_refl).
    + discriminate.
Qed.

(** eval_seq_no_warp: the ESeq inner accumulator loop preserves no_warp_event. *)
Lemma eval_seq_no_warp :
  forall vary_val n t rho,
    (forall e o tr,
      superstep_free e = true ->
      warp_free e = true ->
      eval vary_val n t rho e = Some (o, tr) ->
      no_warp_event tr = true) ->
  forall xs acc o tr,
    no_warp_event acc = true ->
    forallb superstep_free xs = true ->
    forallb warp_free xs = true ->
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
    no_warp_event tr = true.
Proof.
  intros vary_val n t rho IHn.
  induction xs as [| x xs' IHxs].
  - intros acc o tr Hacc _ _ H. simpl in H. inversion H. subst. exact Hacc.
  - intros acc o tr Hacc Hsfree Hwfree H.
    simpl in Hsfree. apply andb_true_iff in Hsfree as [Hsfx Hsfxs].
    simpl in Hwfree. apply andb_true_iff in Hwfree as [Hwfx Hwfxs].
    simpl in H.
    destruct (eval vary_val n t rho x) as [[[xv | xv] xtr] |] eqn:Hx.
    + apply IHxs with (acc := acc ++ xtr) (o := o) (tr := tr).
      * rewrite no_warp_app. apply andb_true_iff; split.
        { exact Hacc. } { exact (IHn x (ONorm xv) xtr Hsfx Hwfx Hx). }
      * exact Hsfxs. * exact Hwfxs. * exact H.
    + inversion H. subst.
      rewrite no_warp_app. apply andb_true_iff; split.
      * exact Hacc. * exact (IHn x (ORet xv) xtr Hsfx Hwfx Hx).
    + discriminate.
Qed.

(** eval_args_no_warp: the EApp inner accumulator loop preserves no_warp_event. *)
Lemma eval_args_no_warp :
  forall vary_val n t rho,
    (forall e o tr,
      superstep_free e = true ->
      warp_free e = true ->
      eval vary_val n t rho e = Some (o, tr) ->
      no_warp_event tr = true) ->
  forall xs acc last_v o tr,
    no_warp_event acc = true ->
    forallb superstep_free xs = true ->
    forallb warp_free xs = true ->
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
    no_warp_event tr = true.
Proof.
  intros vary_val n t rho IHn.
  induction xs as [| x xs' IHxs].
  - intros acc last_v o tr Hacc _ _ H. simpl in H. inversion H. subst. exact Hacc.
  - intros acc last_v o tr Hacc Hsfree Hwfree H.
    simpl in Hsfree. apply andb_true_iff in Hsfree as [Hsfx Hsfxs].
    simpl in Hwfree. apply andb_true_iff in Hwfree as [Hwfx Hwfxs].
    simpl in H.
    destruct (eval vary_val n t rho x) as [[[xv | xv] xtr] |] eqn:Hx.
    + apply IHxs with (acc := acc ++ xtr) (last_v := xv) (o := o) (tr := tr).
      * rewrite no_warp_app. apply andb_true_iff; split.
        { exact Hacc. } { exact (IHn x (ONorm xv) xtr Hsfx Hwfx Hx). }
      * exact Hsfxs. * exact Hwfxs. * exact H.
    + inversion H. subst.
      rewrite no_warp_app. apply andb_true_iff; split.
      * exact Hacc. * exact (IHn x (ORet xv) xtr Hsfx Hwfx Hx).
    + discriminate.
Qed.

(* ----------------------------------------------------------------------- *)
(* 12.4  warp_free_no_warps — trace silence (mirror of barrier_free_no_barriers) *)
(* ----------------------------------------------------------------------- *)

(** warp_free_no_warps (T3-S7 trace-silence): if e is warp_free and
    superstep_free, any completed evaluation emits no EvWarp events.
    EBarrier may emit [EvBarrier] — this is intentional (dual of T3-S3). *)
Theorem warp_free_no_warps :
  forall vary_val fuel t rho e o tr,
    superstep_free e = true ->
    warp_free e = true ->
    eval vary_val fuel t rho e = Some (o, tr) ->
    no_warp_event tr = true.
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - intros t rho e o tr _ _ H. simpl in H. discriminate.
  - intros t rho e o tr Hsf Hwf H.
    destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec ebody
                  | elo ehi ebod | ess | xn eval0 ebody | dv ebod econt | eargs | er ];
    simpl in Hsf; simpl in Hwf; simpl in H.
    (* ELit *)     + inversion H. subst. reflexivity.
    (* EVary *)    + inversion H. subst. reflexivity.
    (* EBarrier — emits [EvBarrier]; no_warp_event [EvBarrier] = true *)
    + inversion H. subst. reflexivity.
    (* EWarpPoint — warp_free = false *) + discriminate.
    (* EVar n0 *)  + inversion H. subst. reflexivity.
    (* EBinop ea eb *)
    + apply andb_true_iff in Hsf as [Hsfa Hsfb].
      apply andb_true_iff in Hwf as [Hwfa Hwfb].
      destruct (eval vary_val fuel' t rho ea) as [[[va | va] tra] |] eqn:Hea.
      * destruct (eval vary_val fuel' t rho eb) as [[[vb | vb] trb] |] eqn:Heb.
        -- inversion H. subst. rewrite no_warp_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho ea _ tra Hsfa Hwfa Hea).
           ++ exact (IHfuel t rho eb _ trb Hsfb Hwfb Heb).
        -- inversion H. subst. rewrite no_warp_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho ea _ tra Hsfa Hwfa Hea).
           ++ exact (IHfuel t rho eb _ trb Hsfb Hwfb Heb).
        -- discriminate.
      * inversion H. subst. exact (IHfuel t rho ea (ORet va) tr Hsfa Hwfa Hea).
      * discriminate.
    (* EUnop eu *)
    + destruct (eval vary_val fuel' t rho eu) as [[[v | v] tr0] |] eqn:Heu.
      * inversion H. subst. exact (IHfuel t rho eu (ONorm v) tr Hsf Hwf Heu).
      * inversion H. subst. exact (IHfuel t rho eu (ORet v) tr Hsf Hwf Heu).
      * discriminate.
    (* EIf econd ethen eelse *)
    + apply andb_true_iff in Hsf as [Hsfce Hsfe].
      apply andb_true_iff in Hsfce as [Hsfc Hsft].
      apply andb_true_iff in Hwf as [Hwfce Hwfe].
      apply andb_true_iff in Hwfce as [Hwfc Hwft].
      destruct (eval vary_val fuel' t rho econd) as [[[cv | cv] tr_c] |] eqn:Hcond.
      * destruct (eval vary_val fuel' t rho (if Nat.eqb cv 0 then eelse else ethen))
              as [[ob tr_b] |] eqn:Hbranch.
        -- inversion H. subst. rewrite no_warp_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho econd (ONorm cv) tr_c Hsfc Hwfc Hcond).
           ++ destruct (Nat.eqb cv 0); simpl in Hbranch.
              ** exact (IHfuel t rho eelse _ tr_b Hsfe Hwfe Hbranch).
              ** exact (IHfuel t rho ethen _ tr_b Hsft Hwft Hbranch).
        -- discriminate.
      * inversion H. subst. exact (IHfuel t rho econd (ORet cv) tr Hsfc Hwfc Hcond).
      * discriminate.
    (* EWhile ec ebody *)
    + apply andb_true_iff in Hsf as [Hsfc Hsfb].
      apply andb_true_iff in Hwf as [Hwfc Hwfb].
      destruct (eval vary_val fuel' t rho ec) as [[[cv | cv] tr_c] |] eqn:Hcond.
      * destruct (Nat.eqb cv 0).
        -- inversion H. subst. exact (IHfuel t rho ec (ONorm cv) tr Hsfc Hwfc Hcond).
        -- destruct (eval vary_val fuel' t rho ebody) as [[[bv | bv] tr_b] |] eqn:Hbod.
           ++ destruct (eval vary_val fuel' t rho (EWhile ec ebody))
                  as [[ol tr_l] |] eqn:Hloop.
              ** inversion H. subst.
                 rewrite no_warp_app, no_warp_app.
                 apply andb_true_iff; split.
                 { exact (IHfuel t rho ec (ONorm cv) tr_c Hsfc Hwfc Hcond). }
                 apply andb_true_iff; split.
                 { exact (IHfuel t rho ebody (ONorm bv) tr_b Hsfb Hwfb Hbod). }
                 assert (Hsfw : superstep_free (EWhile ec ebody) = true).
                 { simpl. rewrite Hsfc, Hsfb. reflexivity. }
                 assert (Hwfw : warp_free (EWhile ec ebody) = true).
                 { simpl. rewrite Hwfc, Hwfb. reflexivity. }
                 exact (IHfuel t rho (EWhile ec ebody) _ tr_l Hsfw Hwfw Hloop).
              ** discriminate.
           ++ inversion H. subst. rewrite no_warp_app. apply andb_true_iff; split.
              ** exact (IHfuel t rho ec (ONorm cv) tr_c Hsfc Hwfc Hcond).
              ** exact (IHfuel t rho ebody (ORet bv) tr_b Hsfb Hwfb Hbod).
           ++ discriminate.
      * inversion H. subst. exact (IHfuel t rho ec (ORet cv) tr Hsfc Hwfc Hcond).
      * discriminate.
    (* EFor elo ehi ebod *)
    + apply andb_true_iff in Hsf as [Hsflh Hsfb].
      apply andb_true_iff in Hsflh as [Hsfl Hsfh].
      apply andb_true_iff in Hwf as [Hwflh Hwfb].
      apply andb_true_iff in Hwflh as [Hwfl Hwfh].
      destruct (eval vary_val fuel' t rho elo) as [[[lo_v | lo_v] tr_lo] |] eqn:Hlo.
      * destruct (eval vary_val fuel' t rho ehi) as [[[hi_v | hi_v] tr_hi] |] eqn:Hhi.
        -- destruct (Nat.leb hi_v lo_v) eqn:Hle.
           ++ inversion H. subst. rewrite no_warp_app. apply andb_true_iff; split.
              ** exact (IHfuel t rho elo (ONorm lo_v) tr_lo Hsfl Hwfl Hlo).
              ** exact (IHfuel t rho ehi (ONorm hi_v) tr_hi Hsfh Hwfh Hhi).
           ++ rewrite (for_loop_eq vary_val fuel' t rho ebod (hi_v - lo_v) (tr_lo ++ tr_hi)) in H.
              apply for_loop_fixed_no_warp
                with (body := eval vary_val fuel' t rho ebod)
                     (k    := hi_v - lo_v)
                     (acc  := tr_lo ++ tr_hi)
                     (o    := o)
                     (tr   := tr).
              ** rewrite no_warp_app. apply andb_true_iff; split.
                 { exact (IHfuel t rho elo (ONorm lo_v) tr_lo Hsfl Hwfl Hlo). }
                 { exact (IHfuel t rho ehi (ONorm hi_v) tr_hi Hsfh Hwfh Hhi). }
              ** intros o' tr' Hbod. exact (IHfuel t rho ebod o' tr' Hsfb Hwfb Hbod).
              ** exact H.
        -- inversion H. subst. rewrite no_warp_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho elo (ONorm lo_v) tr_lo Hsfl Hwfl Hlo).
           ++ exact (IHfuel t rho ehi (ORet hi_v) tr_hi Hsfh Hwfh Hhi).
        -- discriminate.
      * inversion H. subst. exact (IHfuel t rho elo (ORet lo_v) tr Hsfl Hwfl Hlo).
      * discriminate.
    (* ESeq ess *)
    + apply eval_seq_no_warp
        with (vary_val := vary_val) (n := fuel') (t := t) (rho := rho)
             (xs := ess) (acc := []) (o := o) (tr := tr).
      * intros e0 o' tr' Hsfe Hwfe He.
        exact (IHfuel t rho e0 o' tr' Hsfe Hwfe He).
      * reflexivity.
      * exact Hsf.
      * exact Hwf.
      * exact H.
    (* ELet xn eval0 ebody *)
    + apply andb_true_iff in Hsf as [Hsfv Hsfb].
      apply andb_true_iff in Hwf as [Hwfv Hwfb].
      destruct (eval vary_val fuel' t rho eval0) as [[[vv | vv] tr_v] |] eqn:Hval.
      * destruct (eval vary_val fuel' t (venv_extend rho xn vv) ebody)
            as [[ob tr_b] |] eqn:Hbod.
        -- inversion H. subst. rewrite no_warp_app. apply andb_true_iff; split.
           ++ exact (IHfuel t rho eval0 _ tr_v Hsfv Hwfv Hval).
           ++ exact (IHfuel t (venv_extend rho xn vv) ebody _ tr_b Hsfb Hwfb Hbod).
        -- discriminate.
      * inversion H. subst. exact (IHfuel t rho eval0 (ORet vv) tr Hsfv Hwfv Hval).
      * discriminate.
    (* ESuperstep — superstep_free = false *) + discriminate.
    (* EApp eargs *)
    + apply eval_args_no_warp
        with (vary_val := vary_val) (n := fuel') (t := t) (rho := rho)
             (xs := eargs) (acc := []) (last_v := 0) (o := o) (tr := tr).
      * intros e0 o' tr' Hsfe Hwfe He.
        exact (IHfuel t rho e0 o' tr' Hsfe Hwfe He).
      * reflexivity.
      * exact Hsf.
      * exact Hwf.
      * exact H.
    (* EReturn er *)
    + destruct (eval vary_val fuel' t rho er) as [[[v | v] tr0] |] eqn:Her.
      * inversion H. subst. exact (IHfuel t rho er (ONorm v) tr Hsf Hwf Her).
      * inversion H. subst. exact (IHfuel t rho er (ORet v) tr Hsf Hwf Her).
      * discriminate.
Qed.

(* ----------------------------------------------------------------------- *)
(* 12.5  check_warp_env — env-threaded warp checker                         *)
(* ----------------------------------------------------------------------- *)

(** check_warp_env: the env-threaded analogue of ConvergenceSpec.check_warp,
    mirroring check_env's environment threading (EVar/ELet) but flagging
    EWarpPoint (rather than EBarrier) under Diverged mode.  EBarrier carries no
    warp-collective hazard, so check_warp_env never flags it. *)
Fixpoint check_warp_env (m : exec_mode) (env : Env) (e : expr) : list error :=
  match e with
  | EWarpPoint =>
      match m with Diverged => [WarpError] | Converged => [] end
  | ELit | EVary | EBarrier => []
  | EVar x => []
  | EBinop a b     => check_warp_env m env a ++ check_warp_env m env b
  | EUnop e        => check_warp_env m env e
  | EIf cond t el  =>
      let inner := if is_varying_in_env env cond then Diverged else m in
      check_warp_env m env cond ++ check_warp_env inner env t ++ check_warp_env inner env el
  | EWhile cond b  =>
      let inner := if is_varying_in_env env cond then Diverged else m in
      check_warp_env m env cond ++ check_warp_env inner env b
  | EFor lo hi b   =>
      let inner :=
        if is_varying_in_env env lo || is_varying_in_env env hi
        then Diverged else m
      in
      check_warp_env m env lo ++ check_warp_env m env hi ++ check_warp_env inner env b
  | ESeq es        => concat (map (check_warp_env m env) es)
  | ELet x v b    =>
      let vv  := is_varying_in_env env v in
      let env' := env_extend env x vv in
      check_warp_env m env v ++ check_warp_env m env' b
  | ESuperstep divergent body cont =>
      check_warp_env m env body ++ check_warp_env m env cont
  | EApp args      => concat (map (check_warp_env m env) args)
  | EReturn e      => check_warp_env m env e
  end.

(* ----------------------------------------------------------------------- *)
(* 12.6  Bridge: check_warp_env Diverged implies warp_free                 *)
(* ----------------------------------------------------------------------- *)

(** check_warp_env_diverged_clean_warp_free: env-threaded analogue of the
    warp version of diverged_clean_iff_barrier_free.  In Diverged mode,
    check_warp_env flags EWarpPoint; because Diverged is absorbing for inner
    modes, all sub-expressions are checked in Diverged mode too.  Note: unlike
    the barrier bridge, ESuperstep does NOT flag here (a superstep carries no
    warp hazard at its boundary), so its case reduces to the IHs directly. *)
Lemma check_warp_env_diverged_clean_warp_free :
  forall e env, check_warp_env Diverged env e = [] -> warp_free e = true.
Proof.
  apply (expr_list_rect
    (fun e  => forall env, check_warp_env Diverged env e = [] -> warp_free e = true)
    (fun es => forall env, concat (map (check_warp_env Diverged env) es) = [] ->
               forallb warp_free es = true)).
  (* ELit *)       - intros env _. reflexivity.
  (* EVary *)      - intros env _. reflexivity.
  (* EBarrier *)   - intros env _. reflexivity.
  (* EWarpPoint: check_warp_env Diverged _ EWarpPoint = [WarpError] != [] *)
  - intros env H. simpl in H. discriminate.
  (* EVar x *)
  - intros x env _. reflexivity.
  (* EBinop a b *)
  - intros a b IHa IHb env H.
    simpl in H. apply app_eq_nil in H as [Ha Hb].
    simpl. apply andb_true_iff. exact (conj (IHa env Ha) (IHb env Hb)).
  (* EUnop e0 *)
  - intros e0 IHe0 env H. simpl in H. simpl. exact (IHe0 env H).
  (* EIf cond t el: Diverged mode is absorbing *)
  - intros cond t el IHcond IHt IHel env H.
    simpl in H.
    rewrite diverged_absorbing in H.
    apply app_eq_nil in H as [Hcond H'].
    apply app_eq_nil in H' as [Ht Hel_clean].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHcond env Hcond) (IHt env Ht)).
    + exact (IHel env Hel_clean).
  (* EWhile cond b *)
  - intros cond b IHcond IHb env H.
    simpl in H.
    rewrite diverged_absorbing in H.
    apply app_eq_nil in H as [Hcond Hb].
    simpl. apply andb_true_iff. exact (conj (IHcond env Hcond) (IHb env Hb)).
  (* EFor lo hi b *)
  - intros lo hi b IHlo IHhi IHb env H.
    simpl in H.
    rewrite diverged_absorbing in H.
    apply app_eq_nil in H as [Hlo H'].
    apply app_eq_nil in H' as [Hhi Hb].
    simpl. apply andb_true_iff. split.
    + apply andb_true_iff. exact (conj (IHlo env Hlo) (IHhi env Hhi)).
    + exact (IHb env Hb).
  (* ESeq es *)
  - intros es IHes env H. simpl in H. simpl. exact (IHes env H).
  (* ELet x v b *)
  - intros x v b IHv IHb env H.
    simpl in H. apply app_eq_nil in H as [Hv Hb].
    simpl. apply andb_true_iff.
    split.
    + exact (IHv env Hv).
    + exact (IHb (env_extend env x (is_varying_in_env env v)) Hb).
  (* ESuperstep: no entry-flag in check_warp_env; reduce to IHs *)
  - intros dv body cont IHbody IHcont env H.
    simpl in H. apply app_eq_nil in H as [Hbody Hcont].
    simpl. apply andb_true_iff. exact (conj (IHbody env Hbody) (IHcont env Hcont)).
  (* EApp args *)
  - intros args IHargs env H. simpl in H. simpl. exact (IHargs env H).
  (* EReturn e0 *)
  - intros e0 IHe0 env H. simpl in H. simpl. exact (IHe0 env H).
  (* Plist [] *)
  - intros env _. reflexivity.
  (* Plist (h :: tl) *)
  - intros h tl IHh IHtl env H.
    simpl in H. apply app_eq_nil in H as [Hh Htl].
    simpl. apply andb_true_iff. exact (conj (IHh env Hh) (IHtl env Htl)).
Qed.

(** check_warp_env_diverged_no_warps: if core_frag e = true and
    check_warp_env Diverged env e = [] then any completed evaluation emits no
    EvWarp events.  Dual of check_env_diverged_no_barriers. *)
Lemma check_warp_env_diverged_no_warps :
  forall vary_val env e,
    core_frag e = true ->
    check_warp_env Diverged env e = [] ->
    forall fuel t rho o tr,
      eval vary_val fuel t rho e = Some (o, tr) ->
      no_warp_event tr = true.
Proof.
  intros vary_val env e Hcf Hclean fuel t rho o tr Heval.
  apply warp_free_no_warps with
    (vary_val := vary_val) (fuel := fuel) (t := t) (rho := rho) (e := e) (o := o).
  - exact (core_frag_impl_superstep_free e Hcf).
  - exact (check_warp_env_diverged_clean_warp_free e env Hclean).
  - exact Heval.
Qed.

(* ----------------------------------------------------------------------- *)
(* 12.7  for_loop_fixed erase_barrier helpers (mirror of erase_warp ones)   *)
(* ----------------------------------------------------------------------- *)

Lemma for_loop_erase_barrier_body_nil :
  forall (bv : value) (trb : trace) (steps : nat),
    erase_barrier trb = [] ->
    forall (acc : trace),
    match for_loop_fixed (Some (ONorm bv, trb)) steps acc with
    | Some (_, tr) => erase_barrier tr = erase_barrier acc
    | None => True
    end.
Proof.
  intros bv trb steps Htrb.
  induction steps as [| k' IHk]; intro acc.
  - simpl. reflexivity.
  - simpl.
    specialize (IHk (acc ++ trb)).
    destruct (for_loop_fixed (Some (ONorm bv, trb)) k' (acc ++ trb))
      as [[o' tr']|] eqn:Hfl.
    + rewrite IHk. rewrite erase_barrier_app. rewrite Htrb. apply app_nil_r.
    + exact I.
Qed.

Lemma for_loop_erase_barrier_body_eq :
  forall (bv1 bv2 : value) (trb1 trb2 : trace) (steps : nat),
    erase_barrier trb1 = erase_barrier trb2 ->
    forall (acc1 acc2 : trace),
    erase_barrier acc1 = erase_barrier acc2 ->
    match for_loop_fixed (Some (ONorm bv1, trb1)) steps acc1,
          for_loop_fixed (Some (ONorm bv2, trb2)) steps acc2 with
    | Some (_, tr1), Some (_, tr2) => erase_barrier tr1 = erase_barrier tr2
    | None, None => True
    | _, _ => True
    end.
Proof.
  intros bv1 bv2 trb1 trb2 steps Htrb.
  induction steps as [| k' IHk]; intros acc1 acc2 Hacc.
  - simpl. exact Hacc.
  - simpl.
    apply IHk.
    rewrite !erase_barrier_app. congruence.
Qed.

(* ----------------------------------------------------------------------- *)
(* 12.8  eval_check_warp_uniform — combined warp-trace + outcome uniformity *)
(* ----------------------------------------------------------------------- *)

(** eval_check_warp_uniform: the warp dual of eval_check_uniform (T3-S4),
    obtained by the mechanical EBarrier<->EWarpPoint / erase_warp<->erase_barrier
    / check_env<->check_warp_env substitution discussed in design note 1.
    Part A establishes erase_barrier (warp-event) trace equality for clean
    core_frag expressions; Part B establishes outcome equality for non-varying
    ones (outcome equality is projection-independent, so Part B is structurally
    identical to the barrier version). *)
Lemma eval_check_warp_uniform : forall vary_val fuel,
  (* Part A: barrier-trace equality for all clean core_frag expressions *)
  (forall env e t1 t2 rho1 rho2 o1 o2 tr1 tr2,
     core_frag e = true ->
     env_agrees env rho1 rho2 ->
     check_warp_env Converged env e = [] ->
     eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
     eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
     erase_barrier tr1 = erase_barrier tr2) /\
  (* Part B: outcome equality for non-varying clean core_frag expressions *)
  (forall env e t1 t2 rho1 rho2 o1 o2 tr1 tr2,
     core_frag e = true ->
     env_agrees env rho1 rho2 ->
     is_varying_in_env env e = false ->
     check_warp_env Converged env e = [] ->
     eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
     eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
     o1 = o2).
Proof.
  intros vary_val.
  induction fuel as [| fuel' IHfuel].
  - split.
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 _ _ _ H _. simpl in H. discriminate.
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 _ _ _ _ H _. simpl in H. discriminate.
  - destruct IHfuel as [IH_A IH_B].
    split.

(* ------------------------------------------------------------------ *)
(* Part A for S fuel'                                                   *)
(* ------------------------------------------------------------------ *)
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hcf Hagr Hclean Heval1 Heval2.
      destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                    | elo ehi ebod | ess | xn ev ebody | dv ebod econt | eargs | er ];
      simpl in Hcf; simpl in Heval1; simpl in Heval2; try discriminate.

      (* ELit *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVary *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EBarrier *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EWarpPoint *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVar n0 *)
      * inversion Heval1; inversion Heval2; reflexivity.

      (* EBinop ea eb *)
      * apply andb_true_iff in Hcf as [Hcfa Hcfb].
        simpl in Hclean. apply app_eq_nil in Hclean as [Hca Hcb].
        destruct (eval vary_val fuel' t1 rho1 ea) as [[[va1|va1] tra1]|] eqn:Hea1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t1 rho1 va1 tra1 Hcfa Hea1). }
        destruct (eval vary_val fuel' t1 rho1 eb) as [[[vb1|vb1] trb1]|] eqn:Heb1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 vb1 trb1 Hcfb Heb1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ea) as [[[va2|va2] tra2]|] eqn:Hea2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t2 rho2 va2 tra2 Hcfa Hea2). }
        destruct (eval vary_val fuel' t2 rho2 eb) as [[[vb2|vb2] trb2]|] eqn:Heb2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 vb2 trb2 Hcfb Heb2). }
        inversion Heval2; subst.
        rewrite !erase_barrier_app. f_equal.
        { exact (IH_A env ea t1 t2 rho1 rho2 (ONorm va1) (ONorm va2) tra1 tra2
                   Hcfa Hagr Hca Hea1 Hea2). }
        { exact (IH_A env eb t1 t2 rho1 rho2 (ONorm vb1) (ONorm vb2) trb1 trb2
                   Hcfb Hagr Hcb Heb1 Heb2). }

      (* EUnop eu *)
      * destruct (eval vary_val fuel' t1 rho1 eu) as [[[vu1|vu1] tru1]|] eqn:Heu1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t1 rho1 vu1 tru1 Hcf Heu1). }
        destruct (eval vary_val fuel' t2 rho2 eu) as [[[vu2|vu2] tru2]|] eqn:Heu2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t2 rho2 vu2 tru2 Hcf Heu2). }
        (* Heval1 : Some (ONorm (unop_eval vu1), tru1) = Some (o1, tr1) *)
        (* Heval2 : Some (ONorm (unop_eval vu2), tru2) = Some (o2, tr2) *)
        injection Heval1 as Hout1 Htr1. injection Heval2 as Hout2 Htr2.
        simpl in Hclean. rewrite <- Htr1. rewrite <- Htr2.
        exact (IH_A env eu t1 t2 rho1 rho2 (ONorm vu1) (ONorm vu2) tru1 tru2
                 Hcf Hagr Hclean Heu1 Heu2).

      (* EIf econd ethen eelse *)
      * apply andb_true_iff in Hcf as [Hcfct Hcfel].
        apply andb_true_iff in Hcfct as [Hcfc Hcft].
        simpl in Hclean.
        set (inner_if := if is_varying_in_env env econd then Diverged else Converged).
        (* Split check_warp_env *)
        assert (Hclean_c_br : check_warp_env Converged env econd = [] /\
                               check_warp_env inner_if env ethen = [] /\
                               check_warp_env inner_if env eelse = []).
        { unfold inner_if.
          destruct (is_varying_in_env env econd);
          simpl in Hclean;
          apply app_eq_nil in Hclean as [HcC HtE];
          apply app_eq_nil in HtE as [HcT HcEl];
          exact (conj HcC (conj HcT HcEl)). }
        destruct Hclean_c_br as [Hcc [Hct Hcel]].
        (* Destruct eval of condition *)
        destruct (eval vary_val fuel' t1 rho1 econd) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 econd) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t2 rho2 cv2 trc2 Hcfc Hcond2). }
        (* Branch evaluation *)
        destruct (eval vary_val fuel' t1 rho1
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr1 trbr1]|] eqn:Hbr1;
          try discriminate.
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2
                   (if Nat.eqb cv2 0 then eelse else ethen)) as [[obr2 trbr2]|] eqn:Hbr2;
          try discriminate.
        inversion Heval2; subst.
        rewrite !erase_barrier_app.
        f_equal.
        { exact (IH_A env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hcc Hcond1 Hcond2). }
        { (* Show branch traces have equal erase_barrier *)
          destruct (is_varying_in_env env econd) eqn:Hcvar.
          - (* Condition varying: both branches in Diverged mode, no barriers *)
            unfold inner_if in Hct, Hcel.
            (* Thread 1's branch has no barriers *)
            assert (Hnb1 : no_warp_event trbr1 = true).
            { destruct (Nat.eqb cv1 0) eqn:Hcv1z.
              - exact (check_warp_env_diverged_no_warps vary_val env eelse
                         Hcfel Hcel fuel' t1 rho1 o1 trbr1 Hbr1).
              - exact (check_warp_env_diverged_no_warps vary_val env ethen
                         Hcft Hct fuel' t1 rho1 o1 trbr1 Hbr1). }
            assert (Hnb2 : no_warp_event trbr2 = true).
            { destruct (Nat.eqb cv2 0) eqn:Hcv2z.
              - exact (check_warp_env_diverged_no_warps vary_val env eelse
                         Hcfel Hcel fuel' t2 rho2 o2 trbr2 Hbr2).
              - exact (check_warp_env_diverged_no_warps vary_val env ethen
                         Hcft Hct fuel' t2 rho2 o2 trbr2 Hbr2). }
            rewrite (erase_barrier_no_warp _ Hnb1).
            rewrite (erase_barrier_no_warp _ Hnb2).
            reflexivity.
          - (* Condition non-varying: same branch taken *)
            unfold inner_if in Hct, Hcel.
            (* cv1 = cv2 from Part B *)
            assert (Hcveq : ONorm cv1 = ONorm cv2).
            { exact (IH_B env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                       Hcfc Hagr Hcvar Hcc Hcond1 Hcond2). }
            injection Hcveq as Hcveq'. subst cv2.
            (* Both take same branch *)
            destruct (Nat.eqb cv1 0) eqn:Hcv1z.
            + exact (IH_A env eelse t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                       Hcfel Hagr Hcel Hbr1 Hbr2).
            + exact (IH_A env ethen t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                       Hcft Hagr Hct Hbr1 Hbr2). }

      (* EWhile ec eb *)
      * apply andb_true_iff in Hcf as [Hcfc Hcfb].
        simpl in Hclean.
        set (inner_w := if is_varying_in_env env ec then Diverged else Converged).
        assert (Hclean_w : check_warp_env Converged env ec = [] /\
                            check_warp_env inner_w env eb = []).
        { unfold inner_w. destruct (is_varying_in_env env ec);
          simpl in Hclean; apply app_eq_nil in Hclean as [? ?]; auto. }
        destruct Hclean_w as [Hcc_w Hcb_w].
        destruct (is_varying_in_env env ec) eqn:Hcvar_w.
        -- (* Condition varying: use eval_while_exits_immediately *)
           assert (HwcfW : core_frag (EWhile ec eb) = true).
           { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
           destruct (eval_while_exits_immediately vary_val (S fuel') ec eb
                       t1 rho1 o1 tr1 HwcfW Heval1) as [Hec1 Ho1].
           destruct (eval_while_exits_immediately vary_val (S fuel') ec eb
                       t2 rho2 o2 tr2 HwcfW Heval2) as [Hec2 Ho2].
           (* Hec1 : eval fuel' t1 rho1 ec = Some(ONorm 0, tr1)
              Hec2 : eval fuel' t2 rho2 ec = Some(ONorm 0, tr2) *)
           simpl in Hec1, Hec2.
           exact (IH_A env ec t1 t2 rho1 rho2 (ONorm 0) (ONorm 0) tr1 tr2
                    Hcfc Hagr Hcc_w Hec1 Hec2).
        -- (* Condition non-varying: IH_B for condition *)
           unfold inner_w in Hcb_w.
           destruct (eval vary_val fuel' t1 rho1 ec) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t1 rho1 cv1 trc1 Hcfc Hcond1). }
           destruct (eval vary_val fuel' t2 rho2 ec) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t2 rho2 cv2 trc2 Hcfc Hcond2). }
           assert (Hcveq_w : ONorm cv1 = ONorm cv2).
           { exact (IH_B env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                      Hcfc Hagr Hcvar_w Hcc_w Hcond1 Hcond2). }
           injection Hcveq_w as Hcveq_w'. subst cv2.
           destruct (Nat.eqb cv1 0) eqn:Hcv1z.
           ++ (* cv = 0: exits immediately *)
              apply Nat.eqb_eq in Hcv1z. subst cv1.
              inversion Heval1; subst. inversion Heval2; subst.
              exact (IH_A env ec t1 t2 rho1 rho2 (ONorm 0) (ONorm 0) tr1 tr2
                       Hcfc Hagr Hcc_w Hcond1 Hcond2).
           ++ (* cv ≠ 0: loop continues *)
              destruct (eval vary_val fuel' t1 rho1 eb) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
                try discriminate.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 bv1 trb1 Hcfb Hbod1). }
              destruct (eval vary_val fuel' t1 rho1 (EWhile ec eb))
                as [[oloop1 trloop1]|] eqn:Hloop1; try discriminate.
              inversion Heval1; subst.
              destruct (eval vary_val fuel' t2 rho2 eb) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
                try discriminate.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 bv2 trb2 Hcfb Hbod2). }
              destruct (eval vary_val fuel' t2 rho2 (EWhile ec eb))
                as [[oloop2 trloop2]|] eqn:Hloop2; try discriminate.
              inversion Heval2; subst.
              assert (HwcfW : core_frag (EWhile ec eb) = true).
              { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
              assert (Hwclean : check_warp_env Converged env (EWhile ec eb) = []).
              { simpl. rewrite Hcvar_w. simpl.
                rewrite Hcc_w. rewrite Hcb_w. reflexivity. }
              rewrite !erase_barrier_app. f_equal.
              { exact (IH_A env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv1) trc1 trc2
                         Hcfc Hagr Hcc_w Hcond1 Hcond2). }
              f_equal.
              { exact (IH_A env eb t1 t2 rho1 rho2 (ONorm bv1) (ONorm bv2) trb1 trb2
                         Hcfb Hagr Hcb_w Hbod1 Hbod2). }
              { exact (IH_A env (EWhile ec eb) t1 t2 rho1 rho2 o1 o2
                         trloop1 trloop2 HwcfW Hagr Hwclean Hloop1 Hloop2). }

      (* EFor elo ehi ebod *)
      * apply andb_true_iff in Hcf as [Hcflohi Hcfb].
        apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
        simpl in Hclean.
        set (inner_f := if is_varying_in_env env elo || is_varying_in_env env ehi
                        then Diverged else Converged).
        assert (Hclean_f : check_warp_env Converged env elo = [] /\
                            check_warp_env Converged env ehi = [] /\
                            check_warp_env inner_f env ebod = []).
        { unfold inner_f.
          destruct (is_varying_in_env env elo || is_varying_in_env env ehi);
          simpl in Hclean;
          apply app_eq_nil in Hclean as [HcL HtE];
          apply app_eq_nil in HtE as [HcH HcB];
          exact (conj HcL (conj HcH HcB)). }
        destruct Hclean_f as [Hclo [Hchi Hcbod]].
        (* Destruct lo and hi for both threads *)
        destruct (eval vary_val fuel' t1 rho1 elo) as [[[lv1|lv1] trlo1]|] eqn:Hlo1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t1 rho1 lv1 trlo1 Hcflo Hlo1). }
        destruct (eval vary_val fuel' t1 rho1 ehi) as [[[hv1|hv1] trhi1]|] eqn:Hhi1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t1 rho1 hv1 trhi1 Hcfhi Hhi1). }
        destruct (eval vary_val fuel' t2 rho2 elo) as [[[lv2|lv2] trlo2]|] eqn:Hlo2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t2 rho2 lv2 trlo2 Hcflo Hlo2). }
        destruct (eval vary_val fuel' t2 rho2 ehi) as [[[hv2|hv2] trhi2]|] eqn:Hhi2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t2 rho2 hv2 trhi2 Hcfhi Hhi2). }
        (* Rewrite using for_loop_eq *)
        destruct (Nat.leb hv1 lv1) eqn:Hlb1.
        { inversion Heval1; subst.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst.
            rewrite !erase_barrier_app. f_equal.
            { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                       Hcflo Hagr Hclo Hlo1 Hlo2). }
            { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                       Hcfhi Hagr Hchi Hhi1 Hhi2). } }
          (* t1 exits, t2 loops: need to show erase_barrier equal *)
          { rewrite (for_loop_eq vary_val fuel' t2 rho2 ebod (hv2 - lv2)
                       (trlo2 ++ trhi2)) in Heval2.
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - (* Varying: body in Diverged mode, no barriers *)
              unfold inner_f in Hcbod.
              (* Destruct body eval for t2 to apply for_loop_erase_barrier_body_nil *)
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv2 - lv2) as [|steps_pos] eqn:Hsteps2.
                   { apply Nat.leb_nle in Hlb2. lia. }
                   simpl in Heval2. discriminate. }
              (* ONorm bv2: body has no barriers *)
              assert (Htrb2 : erase_barrier trb2 = []).
              { apply erase_barrier_no_warp.
                exact (check_warp_env_diverged_no_warps vary_val env ebod
                         Hcfb Hcbod fuel' t2 rho2 (ONorm bv2) trb2 Hbod2). }
              assert (Hnb2 := for_loop_erase_barrier_body_nil bv2 trb2 (hv2 - lv2) Htrb2 (trlo2 ++ trhi2)).
              rewrite Heval2 in Hnb2.
              (* Hnb2 : erase_barrier tr2 = erase_barrier (trlo2 ++ trhi2) *)
              inversion Heval1; subst.
              rewrite Hnb2. rewrite !erase_barrier_app.
              f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - (* Non-varying lo/hi: same lv, hv from Part B *)
              assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              rewrite Hlb1 in Hlb2. discriminate. } }
        { (* t1 loops *)
          rewrite (for_loop_eq vary_val fuel' t1 rho1 ebod (hv1 - lv1)
                     (trlo1 ++ trhi1)) in Heval1.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { (* t2 exits: symmetric to above *)
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - unfold inner_f in Hcbod.
              (* Destruct body eval for t1 to apply for_loop_erase_barrier_body_nil *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              (* ONorm bv1: body has no barriers *)
              assert (Htrb1 : erase_barrier trb1 = []).
              { apply erase_barrier_no_warp.
                exact (check_warp_env_diverged_no_warps vary_val env ebod
                         Hcfb Hcbod fuel' t1 rho1 (ONorm bv1) trb1 Hbod1). }
              assert (Hnb1 := for_loop_erase_barrier_body_nil bv1 trb1 (hv1 - lv1) Htrb1 (trlo1 ++ trhi1)).
              rewrite Heval1 in Hnb1.
              (* Hnb1 : erase_barrier tr1 = erase_barrier (trlo1 ++ trhi1) *)
              inversion Heval2; subst.
              rewrite Hnb1. rewrite !erase_barrier_app.
              f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
                exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              rewrite Hlb2 in Hlb1. discriminate. }
          { (* Both loop *)
            rewrite (for_loop_eq vary_val fuel' t2 rho2 ebod (hv2 - lv2)
                       (trlo2 ++ trhi2)) in Heval2.
            destruct (is_varying_in_env env elo || is_varying_in_env env ehi) eqn:Hvar_lohi.
            - (* Varying: body in Diverged mode *)
              unfold inner_f in Hcbod.
              (* Destruct body eval for t1 *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              assert (Htrb1 : erase_barrier trb1 = []).
              { apply erase_barrier_no_warp.
                exact (check_warp_env_diverged_no_warps vary_val env ebod
                         Hcfb Hcbod fuel' t1 rho1 (ONorm bv1) trb1 Hbod1). }
              assert (Hnb1 := for_loop_erase_barrier_body_nil bv1 trb1 (hv1 - lv1) Htrb1 (trlo1 ++ trhi1)).
              rewrite Heval1 in Hnb1.
              (* Hnb1 : erase_barrier tr1 = erase_barrier (trlo1 ++ trhi1) *)
              (* Destruct body eval for t2 *)
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv2 - lv2) as [|steps_pos] eqn:Hsteps2.
                   { apply Nat.leb_nle in Hlb2. lia. }
                   simpl in Heval2. discriminate. }
              assert (Htrb2 : erase_barrier trb2 = []).
              { apply erase_barrier_no_warp.
                exact (check_warp_env_diverged_no_warps vary_val env ebod
                         Hcfb Hcbod fuel' t2 rho2 (ONorm bv2) trb2 Hbod2). }
              assert (Hnb2 := for_loop_erase_barrier_body_nil bv2 trb2 (hv2 - lv2) Htrb2 (trlo2 ++ trhi2)).
              rewrite Heval2 in Hnb2.
              (* Hnb2 : erase_barrier tr2 = erase_barrier (trlo2 ++ trhi2) *)
              rewrite Hnb1. rewrite Hnb2.
              rewrite !erase_barrier_app. f_equal.
              { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hclo Hlo1 Hlo2). }
              { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hchi Hhi1 Hhi2). }
            - (* Non-varying lo/hi: same steps, use for_loop_erase_eq *)
              apply Bool.orb_false_iff in Hvar_lohi as [Hvlo Hvhi].
              unfold inner_f in Hcbod.
              assert (Hlv_eq : ONorm lv1 = ONorm lv2).
              { exact (IH_B env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv2) trlo1 trlo2
                         Hcflo Hagr Hvlo Hclo Hlo1 Hlo2). }
              assert (Hhv_eq : ONorm hv1 = ONorm hv2).
              { exact (IH_B env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv2) trhi1 trhi2
                         Hcfhi Hagr Hvhi Hchi Hhi1 Hhi2). }
              injection Hlv_eq as Hlv_eq. injection Hhv_eq as Hhv_eq.
              subst lv2 hv2.
              (* Same steps: hv1 - lv1 *)
              assert (Herase_acc : erase_barrier (trlo1 ++ trhi1) = erase_barrier (trlo2 ++ trhi2)).
              { rewrite !erase_barrier_app. f_equal.
                { exact (IH_A env elo t1 t2 rho1 rho2 (ONorm lv1) (ONorm lv1) trlo1 trlo2
                           Hcflo Hagr Hclo Hlo1 Hlo2). }
                { exact (IH_A env ehi t1 t2 rho1 rho2 (ONorm hv1) (ONorm hv1) trhi1 trhi2
                           Hcfhi Hagr Hchi Hhi1 Hhi2). } }
              (* Destruct body evaluations to apply for_loop_erase_barrier_body_eq *)
              destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv1|bv1] trb1]|] eqn:Hbod1.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv1 trb1
                                     Hcfb Hbod1). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval1 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval1. discriminate. }
              destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv2|bv2] trb2]|] eqn:Hbod2.
              2: { exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv2 trb2
                                     Hcfb Hbod2). }
              2: { (* None body: for_loop_fixed None k acc = None contradicts Heval2 *)
                   destruct (hv1 - lv1) as [|steps_pos] eqn:Hsteps1.
                   { apply Nat.leb_nle in Hlb1. lia. }
                   simpl in Heval2. discriminate. }
              (* ONorm bv1, ONorm bv2: apply IH_A for body erase, then for_loop_erase_barrier_body_eq *)
              assert (Hbody_erase : erase_barrier trb1 = erase_barrier trb2).
              { exact (IH_A env ebod t1 t2 rho1 rho2 (ONorm bv1) (ONorm bv2) trb1 trb2
                           Hcfb Hagr Hcbod Hbod1 Hbod2). }
              assert (Hfl := for_loop_erase_barrier_body_eq bv1 bv2 trb1 trb2 (hv1 - lv1)
                               Hbody_erase (trlo1 ++ trhi1) (trlo2 ++ trhi2) Herase_acc).
              rewrite Heval1 in Hfl. rewrite Heval2 in Hfl. exact Hfl. } }

      (* ESeq ess *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag ess = true) by exact Hcf.
        clear Hcf.
        (* We prove erase_barrier equality by induction on ess,
           generalizing over the accumulator. *)
        assert (Hgen : forall xs,
          forallb core_frag xs = true ->
          concat (map (check_warp_env Converged env) xs) = [] ->
          forall acc1 acc2 o1' o2' tr1' tr2',
          erase_barrier acc1 = erase_barrier acc2 ->
          (fix eval_seq xs0 acc : option _ :=
            match xs0 with [] => Some (ONorm 0, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
              | None => None end end) xs acc1 = Some (o1', tr1') ->
          (fix eval_seq xs0 acc : option _ :=
            match xs0 with [] => Some (ONorm 0, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm _, tr) => eval_seq rest (acc ++ tr)
              | None => None end end) xs acc2 = Some (o2', tr2') ->
          erase_barrier tr1' = erase_barrier tr2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ acc1 acc2 o1' o2' tr1' tr2' Hacc Hs1 Hs2.
            inversion Hs1; inversion Hs2; subst. exact Hacc.
          - intros Hcfal Hcln acc1 acc2 o1' o2' tr1' tr2' Hacc Hs1 Hs2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Hs1, Hs2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            apply IHtl with (acc1 := acc1 ++ trh1) (acc2 := acc2 ++ trh2)
              (o1' := o1') (o2' := o2').
            + exact Hcftl.
            + exact Hctl.
            + rewrite !erase_barrier_app. f_equal.
              * exact Hacc.
              * exact (IH_A env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                         Hcfh Hagr Hch Hh1 Hh2).
            + exact Hs1.
            + exact Hs2. }
        exact (Hgen ess Hcfall Hclean [] [] o1 o2 tr1 tr2 eq_refl Heval1 Heval2).

      (* ELet xn ev ebody *)
      * apply andb_true_iff in Hcf as [Hcfv Hcfb].
        simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcv Hcb].
        (* Destruct eval of ev for both threads *)
        destruct (eval vary_val fuel' t1 rho1 ev) as [[[vv1|vv1] trv1]|] eqn:Hev1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t1 rho1 vv1 trv1 Hcfv Hev1). }
        destruct (eval vary_val fuel' t2 rho2 ev) as [[[vv2|vv2] trv2]|] eqn:Hev2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t2 rho2 vv2 trv2 Hcfv Hev2). }
        (* Destruct eval of body for both threads *)
        destruct (eval vary_val fuel' t1 (venv_extend rho1 xn vv1) ebody)
          as [[ob1 trb1]|] eqn:Hbod1; try discriminate.
        destruct (eval vary_val fuel' t2 (venv_extend rho2 xn vv2) ebody)
          as [[ob2 trb2]|] eqn:Hbod2; try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        rewrite !erase_barrier_app. f_equal.
        { exact (IH_A env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                   Hcfv Hagr Hcv Hev1 Hev2). }
        { (* Build env_agrees for extended env *)
          set (vv := is_varying_in_env env ev).
          assert (Hagr' : env_agrees (env_extend env xn vv)
                            (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)).
          { unfold vv.
            destruct (is_varying_in_env env ev) eqn:Hvv.
            - (* ev varying: xn is varying, no constraint on vv1 vs vv2 *)
              intros y Hlookup.
              assert (Hxy : xn <> y).
              { intro Heq. subst. rewrite env_lookup_extend_same in Hlookup. discriminate. }
              rewrite (venv_lookup_extend_diff rho1 xn y vv1 Hxy).
              rewrite (venv_lookup_extend_diff rho2 xn y vv2 Hxy).
              apply Hagr.
              unfold env_lookup, env_extend in Hlookup |- *.
              simpl find in *.
              destruct (Nat.eqb xn y) eqn:Heqn.
              { apply Nat.eqb_eq in Heqn. subst. exfalso. exact (Hxy eq_refl). }
              { exact Hlookup. }
            - (* ev non-varying: vv1 = vv2 from IH_B *)
              assert (Hveq : ONorm vv1 = ONorm vv2).
              { exact (IH_B env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                         Hcfv Hagr Hvv Hcv Hev1 Hev2). }
              injection Hveq as Hveq'. subst vv2.
              exact (env_agrees_extend env rho1 rho2 xn vv1 false Hagr). }
          exact (IH_A (env_extend env xn vv) ebody t1 t2
                   (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)
                   o1 o2 trb1 trb2 Hcfb Hagr' Hcb Hbod1 Hbod2). }

      (* ESuperstep: core_frag = false, already handled by try discriminate *)

      (* EApp eargs *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag eargs = true) by exact Hcf.
        clear Hcf.
        assert (Hgen : forall xs,
          forallb core_frag xs = true ->
          concat (map (check_warp_env Converged env) xs) = [] ->
          forall acc1 acc2 lv1 lv2 o1' o2' tr1' tr2',
          erase_barrier acc1 = erase_barrier acc2 ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc1 lv1 = Some (o1', tr1') ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc2 lv2 = Some (o2', tr2') ->
          erase_barrier tr1' = erase_barrier tr2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hacc Ha1 Ha2.
            inversion Ha1; inversion Ha2; subst. exact Hacc.
          - intros Hcfal Hcln acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hacc Ha1 Ha2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Ha1, Ha2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            apply IHtl with (acc1 := acc1 ++ trh1) (acc2 := acc2 ++ trh2)
              (lv1 := vh1) (lv2 := vh2) (o1' := o1') (o2' := o2').
            + exact Hcftl.
            + exact Hctl.
            + rewrite !erase_barrier_app. f_equal.
              * exact Hacc.
              * exact (IH_A env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                         Hcfh Hagr Hch Hh1 Hh2).
            + exact Ha1.
            + exact Ha2. }
        exact (Hgen eargs Hcfall Hclean [] [] 0 0 o1 o2 tr1 tr2 eq_refl Heval1 Heval2).

(* ------------------------------------------------------------------ *)
(* Part B for S fuel'                                                   *)
(* ------------------------------------------------------------------ *)
    + intros env e t1 t2 rho1 rho2 o1 o2 tr1 tr2 Hcf Hagr Hvar Hclean Heval1 Heval2.
      destruct e as [ | | | | n0 | ea eb | eu | econd ethen eelse | ec eb
                    | elo ehi ebod | ess | xn ev ebody | dv ebod econt | eargs | er ];
      simpl in Hcf; simpl in Hvar; simpl in Heval1; simpl in Heval2; try discriminate.

      (* ELit: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVary: closed by try discriminate above (Hvar : true = false) *)
      (* EBarrier: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EWarpPoint: both return ONorm 0 *)
      * inversion Heval1; inversion Heval2; reflexivity.
      (* EVar n0: env_agrees gives same value *)
      * inversion Heval1; inversion Heval2. subst. f_equal. apply Hagr. exact Hvar.
      (* EBinop ea eb *)
      * apply andb_true_iff in Hcf as [Hcfa Hcfb].
        apply Bool.orb_false_iff in Hvar as [Hva Hvb].
        simpl in Hclean. apply app_eq_nil in Hclean as [Hca Hcb].
        destruct (eval vary_val fuel' t1 rho1 ea) as [[[va1|va1] tra1]|] eqn:Hea1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t1 rho1 va1 tra1 Hcfa Hea1). }
        destruct (eval vary_val fuel' t1 rho1 eb) as [[[vb1|vb1] trb1]|] eqn:Heb1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 vb1 trb1 Hcfb Heb1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 ea) as [[[va2|va2] tra2]|] eqn:Hea2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ea t2 rho2 va2 tra2 Hcfa Hea2). }
        destruct (eval vary_val fuel' t2 rho2 eb) as [[[vb2|vb2] trb2]|] eqn:Heb2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 vb2 trb2 Hcfb Heb2). }
        inversion Heval2; subst.
        assert (Hva_eq : ONorm va1 = ONorm va2).
        { exact (IH_B env ea t1 t2 rho1 rho2 (ONorm va1) (ONorm va2) tra1 tra2
                   Hcfa Hagr Hva Hca Hea1 Hea2). }
        assert (Hvb_eq : ONorm vb1 = ONorm vb2).
        { exact (IH_B env eb t1 t2 rho1 rho2 (ONorm vb1) (ONorm vb2) trb1 trb2
                   Hcfb Hagr Hvb Hcb Heb1 Heb2). }
        injection Hva_eq as Hva_eq. injection Hvb_eq as Hvb_eq.
        subst va2 vb2. reflexivity.
      (* EUnop eu *)
      * destruct (eval vary_val fuel' t1 rho1 eu) as [[[vu1|vu1] tru1]|] eqn:Heu1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t1 rho1 vu1 tru1 Hcf Heu1). }
        inversion Heval1; subst.
        destruct (eval vary_val fuel' t2 rho2 eu) as [[[vu2|vu2] tru2]|] eqn:Heu2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' eu t2 rho2 vu2 tru2 Hcf Heu2). }
        inversion Heval2; subst.
        simpl in Hclean.
        assert (Hvu_eq : ONorm vu1 = ONorm vu2).
        { exact (IH_B env eu t1 t2 rho1 rho2 (ONorm vu1) (ONorm vu2) tr1 tr2
                   Hcf Hagr Hvar Hclean Heu1 Heu2). }
        injection Hvu_eq as Hvu_eq. subst vu2. reflexivity.
      (* EIf econd ethen eelse *)
      * apply andb_true_iff in Hcf as [Hcfct Hcfel].
        apply andb_true_iff in Hcfct as [Hcfc Hcft].
        apply Bool.orb_false_iff in Hvar as [Hvctel Hvel].
        apply Bool.orb_false_iff in Hvctel as [Hvc Hvt].
        (* cond non-varying: inner = Converged *)
        simpl in Hclean. rewrite Hvc in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcc H'].
        apply app_eq_nil in H' as [Hct Hcel].
        destruct (eval vary_val fuel' t1 rho1 econd) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 econd) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' econd t2 rho2 cv2 trc2 Hcfc Hcond2). }
        assert (Hcveq : ONorm cv1 = ONorm cv2).
        { exact (IH_B env econd t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hvc Hcc Hcond1 Hcond2). }
        injection Hcveq as Hcveq'. subst cv2.
        destruct (eval vary_val fuel' t1 rho1
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr1 trbr1]|] eqn:Hbr1;
          try discriminate.
        destruct (eval vary_val fuel' t2 rho2
                   (if Nat.eqb cv1 0 then eelse else ethen)) as [[obr2 trbr2]|] eqn:Hbr2;
          try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        destruct (Nat.eqb cv1 0) eqn:Hcv1z.
        -- exact (IH_B env eelse t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                   Hcfel Hagr Hvel Hcel Hbr1 Hbr2).
        -- exact (IH_B env ethen t1 t2 rho1 rho2 o1 o2 trbr1 trbr2
                   Hcft Hagr Hvt Hct Hbr1 Hbr2).
      (* EWhile ec eb *)
      * apply andb_true_iff in Hcf as [Hcfc Hcfb].
        apply Bool.orb_false_iff in Hvar as [Hvc_w Hvb_w].
        (* cond non-varying: inner = Converged *)
        simpl in Hclean. rewrite Hvc_w in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcc_w Hcb_w].
        destruct (eval vary_val fuel' t1 rho1 ec) as [[[cv1|cv1] trc1]|] eqn:Hcond1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t1 rho1 cv1 trc1 Hcfc Hcond1). }
        destruct (eval vary_val fuel' t2 rho2 ec) as [[[cv2|cv2] trc2]|] eqn:Hcond2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ec t2 rho2 cv2 trc2 Hcfc Hcond2). }
        assert (Hcveq_w : ONorm cv1 = ONorm cv2).
        { exact (IH_B env ec t1 t2 rho1 rho2 (ONorm cv1) (ONorm cv2) trc1 trc2
                   Hcfc Hagr Hvc_w Hcc_w Hcond1 Hcond2). }
        injection Hcveq_w as Hcveq_w'. subst cv2.
        destruct (Nat.eqb cv1 0) eqn:Hcv1z.
        -- (* cv = 0: both return ONorm 0 *)
           inversion Heval1; subst. inversion Heval2; subst. reflexivity.
        -- (* cv ≠ 0: recurse *)
           destruct (eval vary_val fuel' t1 rho1 eb) as [[[bv1|bv1] trb1]|] eqn:Hbod1;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t1 rho1 bv1 trb1 Hcfb Hbod1). }
           destruct (eval vary_val fuel' t1 rho1 (EWhile ec eb)) as [[oloop1 trloop1]|]
             eqn:Hloop1; try discriminate.
           inversion Heval1; subst.
           destruct (eval vary_val fuel' t2 rho2 eb) as [[[bv2|bv2] trb2]|] eqn:Hbod2;
             try discriminate.
           2: { exfalso. exact (core_frag_no_ret vary_val fuel' eb t2 rho2 bv2 trb2 Hcfb Hbod2). }
           destruct (eval vary_val fuel' t2 rho2 (EWhile ec eb)) as [[oloop2 trloop2]|]
             eqn:Hloop2; try discriminate.
           inversion Heval2; subst.
           assert (HwcfW : core_frag (EWhile ec eb) = true).
           { simpl. apply andb_true_iff. exact (conj Hcfc Hcfb). }
           assert (HwvarW : is_varying_in_env env (EWhile ec eb) = false).
           { simpl. apply Bool.orb_false_iff. exact (conj Hvc_w Hvb_w). }
           assert (HwcleanW : check_warp_env Converged env (EWhile ec eb) = []).
           { simpl. rewrite Hvc_w. simpl. rewrite Hcc_w. rewrite Hcb_w. reflexivity. }
           exact (IH_B env (EWhile ec eb) t1 t2 rho1 rho2 o1 o2 trloop1 trloop2
                    HwcfW Hagr HwvarW HwcleanW Hloop1 Hloop2).
      (* EFor elo ehi ebod: outcome is always ONorm 0 for core_frag *)
      * apply andb_true_iff in Hcf as [Hcflohi Hcfb].
        apply andb_true_iff in Hcflohi as [Hcflo Hcfhi].
        apply Bool.orb_false_iff in Hvar as [Hvlohi Hvb].
        apply Bool.orb_false_iff in Hvlohi as [Hvlo Hvhi].
        simpl in Hclean. rewrite Hvlo, Hvhi in Hclean. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hclo H'].
        apply app_eq_nil in H' as [Hchi Hcb].
        destruct (eval vary_val fuel' t1 rho1 elo) as [[[lv1|lv1] trlo1]|] eqn:Hlo1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t1 rho1 lv1 trlo1 Hcflo Hlo1). }
        destruct (eval vary_val fuel' t1 rho1 ehi) as [[[hv1|hv1] trhi1]|] eqn:Hhi1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t1 rho1 hv1 trhi1 Hcfhi Hhi1). }
        destruct (eval vary_val fuel' t2 rho2 elo) as [[[lv2|lv2] trlo2]|] eqn:Hlo2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' elo t2 rho2 lv2 trlo2 Hcflo Hlo2). }
        destruct (eval vary_val fuel' t2 rho2 ehi) as [[[hv2|hv2] trhi2]|] eqn:Hhi2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ehi t2 rho2 hv2 trhi2 Hcfhi Hhi2). }
        (* Both outcomes are ONorm 0 regardless of loop body, for core_frag ebod *)
        (* Use for_loop_fixed approach with inline loop outcome lemma *)
        assert (Hfl1 : forall k acc r,
          (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
            match k0 with
            | O => Some (ONorm 0, acc_tr)
            | S k' =>
                match eval vary_val fuel' t1 rho1 ebod with
                | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                | None => None end end) k acc = Some r ->
          fst r = ONorm 0).
        { intros k. induction k as [|k' IHk'].
          - intros acc r H. simpl in H. inversion H. reflexivity.
          - intros acc r H. simpl in H.
            destruct (eval vary_val fuel' t1 rho1 ebod) as [[[bv|bv] trb]|] eqn:Hbod.
            + exact (IHk' _ _ H).
            + exfalso. exact (core_frag_no_ret vary_val fuel' ebod t1 rho1 bv trb Hcfb Hbod).
            + discriminate. }
        assert (Hfl2 : forall k acc r,
          (fix loop (k0 : nat) (acc_tr : trace) : option (outcome * trace) :=
            match k0 with
            | O => Some (ONorm 0, acc_tr)
            | S k' =>
                match eval vary_val fuel' t2 rho2 ebod with
                | Some (ORet v, tr_b) => Some (ORet v, acc_tr ++ tr_b)
                | Some (ONorm _, tr_b) => loop k' (acc_tr ++ tr_b)
                | None => None end end) k acc = Some r ->
          fst r = ONorm 0).
        { intros k. induction k as [|k' IHk'].
          - intros acc r H. simpl in H. inversion H. reflexivity.
          - intros acc r H. simpl in H.
            destruct (eval vary_val fuel' t2 rho2 ebod) as [[[bv|bv] trb]|] eqn:Hbod.
            + exact (IHk' _ _ H).
            + exfalso. exact (core_frag_no_ret vary_val fuel' ebod t2 rho2 bv trb Hcfb Hbod).
            + discriminate. }
        (* Destruct leb in Heval1/Heval2 to distinguish base vs loop case *)
        destruct (Nat.leb hv1 lv1) eqn:Hlb1.
        { inversion Heval1; subst.
          destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst. reflexivity. }
          { apply Hfl2 in Heval2. simpl in Heval2. destruct o2; congruence. } }
        { destruct (Nat.leb hv2 lv2) eqn:Hlb2.
          { inversion Heval2; subst.
            apply Hfl1 in Heval1. simpl in Heval1. destruct o1; congruence. }
          { apply Hfl1 in Heval1. apply Hfl2 in Heval2.
            destruct o1, o2; simpl in *; congruence. } }
      (* ESeq ess: outcome is always ONorm 0 for core_frag *)
      * simpl in Hclean.
        (* Both return ONorm 0: seq never returns ORet with core_frag *)
        (* All core_frag seqs return ONorm 0 *)
        assert (Hgen1 : forall xs acc o tr,
          forallb core_frag xs = true ->
          (fix eval_seq xs0 acc_tr : option (outcome * trace) :=
            match xs0 with [] => Some (ONorm 0, acc_tr) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr') => Some (ORet v, acc_tr ++ tr')
              | Some (ONorm _, tr') => eval_seq rest (acc_tr ++ tr')
              | None => None end end) xs acc = Some (o, tr) ->
          o = ONorm 0).
        { intros xs. induction xs as [|h tl IHtl]; intros acc o tr Hcfs He.
          - inversion He. reflexivity.
          - simpl in He. simpl in Hcfs. apply andb_true_iff in Hcfs as [Hcfh Hcftl].
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh|vh] trh]|] eqn:Hh;
              try discriminate.
            + exact (IHtl _ _ _ Hcftl He).
            + exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh trh Hcfh Hh). }
        assert (Hgen2 : forall xs acc o tr,
          forallb core_frag xs = true ->
          (fix eval_seq xs0 acc_tr : option (outcome * trace) :=
            match xs0 with [] => Some (ONorm 0, acc_tr) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr') => Some (ORet v, acc_tr ++ tr')
              | Some (ONorm _, tr') => eval_seq rest (acc_tr ++ tr')
              | None => None end end) xs acc = Some (o, tr) ->
          o = ONorm 0).
        { intros xs. induction xs as [|h tl IHtl]; intros acc o tr Hcfs He.
          - inversion He. reflexivity.
          - simpl in He. simpl in Hcfs. apply andb_true_iff in Hcfs as [Hcfh Hcftl].
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh|vh] trh]|] eqn:Hh;
              try discriminate.
            + exact (IHtl _ _ _ Hcftl He).
            + exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh trh Hcfh Hh). }
        assert (Ho1 : o1 = ONorm 0) by exact (Hgen1 ess [] o1 tr1 Hcf Heval1).
        assert (Ho2 : o2 = ONorm 0) by exact (Hgen2 ess [] o2 tr2 Hcf Heval2).
        subst o1 o2. reflexivity.
      (* ELet xn ev ebody *)
      * apply andb_true_iff in Hcf as [Hcfv Hcfb].
        simpl in Hvar. simpl in Hclean.
        apply app_eq_nil in Hclean as [Hcv Hcb].
        destruct (eval vary_val fuel' t1 rho1 ev) as [[[vv1|vv1] trv1]|] eqn:Hev1;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t1 rho1 vv1 trv1 Hcfv Hev1). }
        destruct (eval vary_val fuel' t2 rho2 ev) as [[[vv2|vv2] trv2]|] eqn:Hev2;
          try discriminate.
        2: { exfalso. exact (core_frag_no_ret vary_val fuel' ev t2 rho2 vv2 trv2 Hcfv Hev2). }
        destruct (eval vary_val fuel' t1 (venv_extend rho1 xn vv1) ebody)
          as [[ob1 trb1]|] eqn:Hbod1; try discriminate.
        destruct (eval vary_val fuel' t2 (venv_extend rho2 xn vv2) ebody)
          as [[ob2 trb2]|] eqn:Hbod2; try discriminate.
        inversion Heval1; subst. inversion Heval2; subst.
        (* Build env_agrees for extended env *)
        set (vvflag := is_varying_in_env env ev).
        assert (Hagr' : env_agrees (env_extend env xn vvflag)
                          (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)).
        { unfold vvflag. destruct (is_varying_in_env env ev) eqn:Hvv.
          - intros y Hlookup.
            assert (Hxy : xn <> y).
            { intro Heq. subst. rewrite env_lookup_extend_same in Hlookup. discriminate. }
            rewrite (venv_lookup_extend_diff rho1 xn y vv1 Hxy).
            rewrite (venv_lookup_extend_diff rho2 xn y vv2 Hxy).
            apply Hagr.
            unfold env_lookup, env_extend in Hlookup |- *.
            simpl find in *.
            destruct (Nat.eqb xn y) eqn:Heqn.
            { apply Nat.eqb_eq in Heqn. subst. exfalso. exact (Hxy eq_refl). }
            { exact Hlookup. }
          - assert (Hveq : ONorm vv1 = ONorm vv2).
            { exact (IH_B env ev t1 t2 rho1 rho2 (ONorm vv1) (ONorm vv2) trv1 trv2
                       Hcfv Hagr Hvv Hcv Hev1 Hev2). }
            injection Hveq as Hveq'. subst vv2.
            exact (env_agrees_extend env rho1 rho2 xn vv1 false Hagr). }
        exact (IH_B (env_extend env xn vvflag) ebody t1 t2
                 (venv_extend rho1 xn vv1) (venv_extend rho2 xn vv2)
                 o1 o2 trb1 trb2 Hcfb Hagr' Hvar Hcb Hbod1 Hbod2).
      (* ESuperstep: core_frag = false *)
      (* EApp eargs: last value equality *)
      * simpl in Hclean.
        assert (Hcfall : forallb core_frag eargs = true) by exact Hcf.
        clear Hcf.
        assert (Hgen : forall xs,
          forallb core_frag xs = true ->
          concat (map (check_warp_env Converged env) xs) = [] ->
          existsb (is_varying_in_env env) xs = false ->
          forall acc1 acc2 lv1 lv2 o1' o2' tr1' tr2',
          lv1 = lv2 ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t1 rho1 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc1 lv1 = Some (o1', tr1') ->
          (fix eval_args xs0 acc lv : option _ :=
            match xs0 with [] => Some (ONorm lv, acc) | x::rest =>
              match eval vary_val fuel' t2 rho2 x with
              | Some (ORet v, tr) => Some (ORet v, acc ++ tr)
              | Some (ONorm v, tr) => eval_args rest (acc ++ tr) v
              | None => None end end) xs acc2 lv2 = Some (o2', tr2') ->
          o1' = o2').
        { intros xs. induction xs as [|h tl IHtl].
          - intros _ _ _ acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hlv Ha1 Ha2.
            inversion Ha1; inversion Ha2; subst. congruence.
          - intros Hcfal Hcln Hv acc1 acc2 lv1 lv2 o1' o2' tr1' tr2' Hlv Ha1 Ha2.
            simpl in Hcfal. apply andb_true_iff in Hcfal as [Hcfh Hcftl].
            simpl in Hcln. apply app_eq_nil in Hcln as [Hch Hctl].
            simpl in Hv. apply Bool.orb_false_iff in Hv as [Hvh Hvtl].
            simpl in Ha1, Ha2.
            destruct (eval vary_val fuel' t1 rho1 h) as [[[vh1|vh1] trh1]|] eqn:Hh1;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t1 rho1 vh1 trh1 Hcfh Hh1). }
            destruct (eval vary_val fuel' t2 rho2 h) as [[[vh2|vh2] trh2]|] eqn:Hh2;
              try discriminate.
            2: { exfalso. exact (core_frag_no_ret vary_val fuel' h t2 rho2 vh2 trh2 Hcfh Hh2). }
            assert (Hvheq : ONorm vh1 = ONorm vh2).
            { exact (IH_B env h t1 t2 rho1 rho2 (ONorm vh1) (ONorm vh2) trh1 trh2
                       Hcfh Hagr Hvh Hch Hh1 Hh2). }
            injection Hvheq as Hvheq'. subst vh2.
            exact (IHtl Hcftl Hctl Hvtl (acc1 ++ trh1) (acc2 ++ trh2) vh1 vh1
                     o1' o2' tr1' tr2' eq_refl Ha1 Ha2). }
        exact (Hgen eargs Hcfall Hclean Hvar [] [] 0 0 o1 o2 tr1 tr2 eq_refl Heval1 Heval2).
      (* EReturn: core_frag = false *)
Qed.

(* ----------------------------------------------------------------------- *)
(* 12.9  warp_safe and check_warp_sound_core                                *)
(* ----------------------------------------------------------------------- *)

Section WarpModel.

(** warp_of partitions threads into warps.  Two threads are warp-collective
    peers iff they share a warp.  This is left abstract (a Section Variable):
    the soundness result is parametric in the warp partition, which closes the
    ASSUMPTIONS.md "warp-size parameterization" item — no fixed warp size (32,
    64, ...) is baked into the proof. *)
Variable warp_of : tid -> nat.

(** warp_safe vary_val env e: any two env-agreeing threads in the SAME warp
    that complete evaluation of e issue warp-collective operations (EvWarp
    events) in the same order.  Barrier events (EvBarrier) may differ and are
    projected out by erase_barrier.  The same-warp restriction
    (warp_of t1 = warp_of t2) is the defining feature of warp-collective
    safety, dual to barrier_safe's all-threads scope. *)
Definition warp_safe (vary_val : tid -> value) (env : Env) (e : expr) : Prop :=
  forall fuel t1 t2 rho1 rho2 o1 o2 tr1 tr2,
    warp_of t1 = warp_of t2 ->
    env_agrees env rho1 rho2 ->
    eval vary_val fuel t1 rho1 e = Some (o1, tr1) ->
    eval vary_val fuel t2 rho2 e = Some (o2, tr2) ->
    erase_barrier tr1 = erase_barrier tr2.

(** check_warp_sound_core (T3-S7 main): if e is in the core fragment and the
    env-threaded warp checker reports no error under Converged mode, then e is
    warp_safe.  The same-warp hypothesis is not needed for the trace equality
    (eval is independent of warp_of), so warp_safe holds a fortiori — the
    checker's WarpError is therefore a SOUND rejection of warp-divergence
    hazards.  This is the warp dual of check_env_sound_core (T3-S4). *)
Theorem check_warp_sound_core :
  forall vary_val env e,
    core_frag e = true ->
    check_warp_env Converged env e = [] ->
    warp_safe vary_val env e.
Proof.
  intros vary_val env e Hcf Hclean.
  unfold warp_safe.
  intros fuel t1 t2 rho1 rho2 o1 o2 tr1 tr2 _Hwarp Hagr Heval1 Heval2.
  exact (proj1 (eval_check_warp_uniform vary_val fuel) env e t1 t2 rho1 rho2
           o1 o2 tr1 tr2 Hcf Hagr Hclean Heval1 Heval2).
Qed.

End WarpModel.

(* ======================================================================= *)
(* 13.  T3-S8 — Concrete evaluator for extraction (CMBT closure)           *)
(* ======================================================================= *)

(** eval_concrete: the operational evaluator with the abstract per-thread
    varying value vary_val instantiated as the identity [fun t => t]. This is
    the concrete witness extracted to OCaml (ConvergenceModel.eval_concrete)
    so the differential conformance suite can exercise the *operational*
    semantics — not only the static checkers — against an inline reference.

    Identity is a sound, maximally-discriminating instantiation: distinct
    threads receive distinct EVary values (vary_val t1 = t1 <> t2 = vary_val t2
    for t1 <> t2), so any thread-dependent control-flow divergence (the F-04
    class) becomes observable in the extracted traces. All section lemmas
    (eval_fuel_monotone, barrier_free_no_barriers, the differential
    check_env_sound_core, hazard_not_barrier_safe) hold for *every* vary_val,
    hence a fortiori for this identity instantiation. *)
Definition eval_concrete (fuel : nat) (t : tid) (rho : venv) (e : expr)
    : option (outcome * trace) :=
  eval (fun th => th) fuel t rho e.

(** eval_concrete_fuel_monotone: the headline fuel-monotonicity sanity
    property, specialized to the extracted instantiation. This is the Rocq
    statement the QCheck property [eval_fuel_monotone] mirrors over random
    inputs (CMBT link 4). *)
Lemma eval_concrete_fuel_monotone :
  forall n t rho e r,
    eval_concrete n t rho e = Some r ->
    eval_concrete (S n) t rho e = Some r.
Proof. intros n t rho e r H. apply eval_fuel_monotone. exact H. Qed.

(** eval_concrete_barrier_free_silent: the barrier-silence property for the
    extracted instantiation. A barrier_free + superstep_free expression emits
    no EvBarrier event in any completed concrete run. Mirrored by the QCheck
    property [barrier_free_silent]. *)
Lemma eval_concrete_barrier_free_silent :
  forall fuel t rho e o tr,
    superstep_free e = true ->
    barrier_free e = true ->
    eval_concrete fuel t rho e = Some (o, tr) ->
    no_barrier_event tr = true.
Proof.
  intros fuel t rho e o tr Hsf Hbf H.
  exact (barrier_free_no_barriers (fun th => th) fuel t rho e o tr Hsf Hbf H).
Qed.
