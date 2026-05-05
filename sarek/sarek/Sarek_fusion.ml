(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_fusion - Kernel fusion on Sarek_ir

    Implements vertical fusion: when kernel A writes to an intermediate array
    that kernel B reads, fuse them by inlining A's computation into B,
    eliminating the intermediate array. *)

open Sarek_ir_types

(** {1 Access pattern analysis} *)

(** Access pattern for arrays *)
type access_pattern =
  | OneToOne of expr  (** arr[idx] where idx is thread-uniform *)
  | Stencil of int list  (** arr[idx + offset] for each offset *)
  | Reduction of binop  (** Reduction with associative op (Add, Mul, etc.) *)
  | Gather  (** arr[f(idx)] - arbitrary index *)
  | Unknown

(** Fusion analysis result for a kernel *)
type fusion_info = {
  reads : (string * access_pattern) list;  (** arrays read *)
  writes : (string * access_pattern) list;  (** arrays written *)
  has_barriers : bool;  (** contains barriers *)
  has_atomics : bool;  (** contains atomic ops *)
}

(** {1 Expression utilities} *)

(** Check if two expressions are structurally equal *)
let rec expr_equal e1 e2 =
  match (e1, e2) with
  | EConst c1, EConst c2 -> c1 = c2
  | EVar v1, EVar v2 -> v1.var_id = v2.var_id
  | EBinop (op1, a1, b1), EBinop (op2, a2, b2) ->
      op1 = op2 && expr_equal a1 a2 && expr_equal b1 b2
  | EUnop (op1, e1), EUnop (op2, e2) -> op1 = op2 && expr_equal e1 e2
  | EArrayRead (arr1, idx1), EArrayRead (arr2, idx2) ->
      arr1 = arr2 && expr_equal idx1 idx2
  | EIntrinsic (p1, n1, args1), EIntrinsic (p2, n2, args2) ->
      p1 = p2 && n1 = n2
      && List.length args1 = List.length args2
      && List.for_all2 expr_equal args1 args2
  | _ -> false

(** Check if expression references an array *)
let rec expr_uses_array arr expr =
  match expr with
  | EArrayRead (a, idx) -> a = arr || expr_uses_array arr idx
  | EBinop (_, e1, e2) -> expr_uses_array arr e1 || expr_uses_array arr e2
  | EUnop (_, e) -> expr_uses_array arr e
  | ERecordField (e, _) -> expr_uses_array arr e
  | EIntrinsic (_, _, args) -> List.exists (expr_uses_array arr) args
  | ECast (_, e) -> expr_uses_array arr e
  | ETuple es -> List.exists (expr_uses_array arr) es
  | EApp (fn, args) ->
      expr_uses_array arr fn || List.exists (expr_uses_array arr) args
  | ERecord (_, fields) ->
      List.exists (fun (_, e) -> expr_uses_array arr e) fields
  | EVariant (_, _, args) -> List.exists (expr_uses_array arr) args
  | EArrayLen _ | EArrayCreate _ -> false
  | EConst _ | EVar _ -> false
  | EArrayReadExpr (base, idx) ->
      expr_uses_array arr base || expr_uses_array arr idx
  | EIf (cond, then_, else_) ->
      expr_uses_array arr cond || expr_uses_array arr then_
      || expr_uses_array arr else_
  | EMatch (e, cases) ->
      expr_uses_array arr e
      || List.exists (fun (_, body) -> expr_uses_array arr body) cases

(** Substitute array reads in an expression *)
let rec subst_array_read arr idx_var replacement expr =
  match expr with
  | EArrayRead (a, idx) when a = arr && expr_equal idx idx_var -> replacement
  | EArrayRead (a, idx) ->
      EArrayRead (a, subst_array_read arr idx_var replacement idx)
  | EBinop (op, e1, e2) ->
      EBinop
        ( op,
          subst_array_read arr idx_var replacement e1,
          subst_array_read arr idx_var replacement e2 )
  | EUnop (op, e) -> EUnop (op, subst_array_read arr idx_var replacement e)
  | ERecordField (e, f) ->
      ERecordField (subst_array_read arr idx_var replacement e, f)
  | EIntrinsic (path, name, args) ->
      EIntrinsic
        (path, name, List.map (subst_array_read arr idx_var replacement) args)
  | ECast (ty, e) -> ECast (ty, subst_array_read arr idx_var replacement e)
  | ETuple es -> ETuple (List.map (subst_array_read arr idx_var replacement) es)
  | EApp (fn, args) ->
      EApp
        ( subst_array_read arr idx_var replacement fn,
          List.map (subst_array_read arr idx_var replacement) args )
  | ERecord (name, fields) ->
      ERecord
        ( name,
          List.map
            (fun (f, e) -> (f, subst_array_read arr idx_var replacement e))
            fields )
  | EVariant (ty, tag, args) ->
      EVariant
        (ty, tag, List.map (subst_array_read arr idx_var replacement) args)
  | EArrayLen _ | EArrayCreate _ | EConst _ | EVar _ -> expr
  | EArrayReadExpr (base, idx) ->
      EArrayReadExpr
        ( subst_array_read arr idx_var replacement base,
          subst_array_read arr idx_var replacement idx )
  | EIf (cond, then_, else_) ->
      EIf
        ( subst_array_read arr idx_var replacement cond,
          subst_array_read arr idx_var replacement then_,
          subst_array_read arr idx_var replacement else_ )
  | EMatch (e, cases) ->
      EMatch
        ( subst_array_read arr idx_var replacement e,
          List.map
            (fun (p, body) ->
              (p, subst_array_read arr idx_var replacement body))
            cases )

(** {1 Statement utilities} *)

(** Substitute array reads in a statement *)
let rec subst_array_read_stmt arr idx_var replacement stmt =
  let subst_e = subst_array_read arr idx_var replacement in
  match stmt with
  | SAssign (lv, e) ->
      let lv' =
        match lv with
        | LArrayElem (a, idx) -> LArrayElem (a, subst_e idx)
        | lv -> lv
      in
      SAssign (lv', subst_e e)
  | SSeq stmts ->
      SSeq (List.map (subst_array_read_stmt arr idx_var replacement) stmts)
  | SIf (cond, s1, s2) ->
      SIf
        ( subst_e cond,
          subst_array_read_stmt arr idx_var replacement s1,
          Option.map (subst_array_read_stmt arr idx_var replacement) s2 )
  | SWhile (cond, body) ->
      SWhile (subst_e cond, subst_array_read_stmt arr idx_var replacement body)
  | SFor (v, start, stop, dir, body) ->
      SFor
        ( v,
          subst_e start,
          subst_e stop,
          dir,
          subst_array_read_stmt arr idx_var replacement body )
  | SMatch (e, cases) ->
      SMatch
        ( subst_e e,
          List.map
            (fun (p, s) -> (p, subst_array_read_stmt arr idx_var replacement s))
            cases )
  | SReturn e -> SReturn (subst_e e)
  | SExpr e -> SExpr (subst_e e)
  | SLet (v, e, body) ->
      SLet (v, subst_e e, subst_array_read_stmt arr idx_var replacement body)
  | SLetMut (v, e, body) ->
      SLetMut (v, subst_e e, subst_array_read_stmt arr idx_var replacement body)
  | SPragma (hints, body) ->
      SPragma (hints, subst_array_read_stmt arr idx_var replacement body)
  | SBlock body -> SBlock (subst_array_read_stmt arr idx_var replacement body)
  | SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _ -> stmt

(** Check if statement uses an array *)
let rec stmt_uses_array arr stmt =
  match stmt with
  | SAssign (LArrayElem (a, idx), e) ->
      a = arr || expr_uses_array arr idx || expr_uses_array arr e
  | SAssign (_, e) -> expr_uses_array arr e
  | SSeq stmts -> List.exists (stmt_uses_array arr) stmts
  | SIf (cond, s1, s2) ->
      expr_uses_array arr cond || stmt_uses_array arr s1
      || Option.fold ~none:false ~some:(stmt_uses_array arr) s2
  | SWhile (cond, body) -> expr_uses_array arr cond || stmt_uses_array arr body
  | SFor (_, start, stop, _, body) ->
      expr_uses_array arr start || expr_uses_array arr stop
      || stmt_uses_array arr body
  | SMatch (e, cases) ->
      expr_uses_array arr e
      || List.exists (fun (_, s) -> stmt_uses_array arr s) cases
  | SReturn e | SExpr e -> expr_uses_array arr e
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      expr_uses_array arr e || stmt_uses_array arr body
  | SPragma (_, body) | SBlock body -> stmt_uses_array arr body
  | SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _ -> false

(** {1 Analysis} *)

(** Collect all array reads from an expression *)
let rec collect_reads_expr acc expr =
  match expr with
  | EArrayRead (arr, idx) -> (arr, idx) :: collect_reads_expr acc idx
  | EBinop (_, e1, e2) -> collect_reads_expr (collect_reads_expr acc e1) e2
  | EUnop (_, e) -> collect_reads_expr acc e
  | ERecordField (e, _) -> collect_reads_expr acc e
  | EIntrinsic (_, _, args) -> List.fold_left collect_reads_expr acc args
  | ECast (_, e) -> collect_reads_expr acc e
  | ETuple es -> List.fold_left collect_reads_expr acc es
  | EApp (fn, args) ->
      List.fold_left collect_reads_expr (collect_reads_expr acc fn) args
  | ERecord (_, fields) ->
      List.fold_left (fun acc (_, e) -> collect_reads_expr acc e) acc fields
  | EVariant (_, _, args) -> List.fold_left collect_reads_expr acc args
  | EArrayLen _ | EArrayCreate _ | EConst _ | EVar _ -> acc
  | EArrayReadExpr (base, idx) ->
      collect_reads_expr (collect_reads_expr acc base) idx
  | EIf (cond, then_, else_) ->
      collect_reads_expr
        (collect_reads_expr (collect_reads_expr acc cond) then_)
        else_
  | EMatch (e, cases) ->
      List.fold_left
        (fun acc (_, body) -> collect_reads_expr acc body)
        (collect_reads_expr acc e)
        cases

(** Collect all array writes from a statement *)
let rec collect_writes_stmt acc stmt =
  match stmt with
  | SAssign (LArrayElem (arr, idx), _) -> (arr, idx) :: acc
  | SAssign _ -> acc
  | SSeq stmts -> List.fold_left collect_writes_stmt acc stmts
  | SIf (_, s1, s2) ->
      let acc' = collect_writes_stmt acc s1 in
      Option.fold ~none:acc' ~some:(collect_writes_stmt acc') s2
  | SWhile (_, body) | SFor (_, _, _, _, body) -> collect_writes_stmt acc body
  | SMatch (_, cases) ->
      List.fold_left (fun acc (_, s) -> collect_writes_stmt acc s) acc cases
  | SLet (_, _, body) | SLetMut (_, _, body) | SPragma (_, body) | SBlock body
    ->
      collect_writes_stmt acc body
  | SReturn _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence | SEmpty
  | SNative _ ->
      acc

(** Collect all array reads from a statement *)
let rec collect_reads_stmt acc stmt =
  match stmt with
  | SAssign (LArrayElem (_, idx), e) ->
      collect_reads_expr (collect_reads_expr acc idx) e
  | SAssign (_, e) -> collect_reads_expr acc e
  | SSeq stmts -> List.fold_left collect_reads_stmt acc stmts
  | SIf (cond, s1, s2) ->
      let acc' = collect_reads_expr acc cond in
      let acc'' = collect_reads_stmt acc' s1 in
      Option.fold ~none:acc'' ~some:(collect_reads_stmt acc'') s2
  | SWhile (cond, body) -> collect_reads_stmt (collect_reads_expr acc cond) body
  | SFor (_, start, stop, _, body) ->
      let acc' = collect_reads_expr (collect_reads_expr acc start) stop in
      collect_reads_stmt acc' body
  | SMatch (e, cases) ->
      let acc' = collect_reads_expr acc e in
      List.fold_left (fun acc (_, s) -> collect_reads_stmt acc s) acc' cases
  | SReturn e | SExpr e -> collect_reads_expr acc e
  | SLet (_, e, body) | SLetMut (_, e, body) ->
      collect_reads_stmt (collect_reads_expr acc e) body
  | SPragma (_, body) | SBlock body -> collect_reads_stmt acc body
  | SBarrier | SWarpBarrier | SMemFence | SEmpty | SNative _ -> acc

(** Check if statement contains barriers *)
let rec has_barrier stmt =
  match stmt with
  | SBarrier | SWarpBarrier -> true
  | SSeq stmts -> List.exists has_barrier stmts
  | SIf (_, s1, s2) ->
      has_barrier s1 || Option.fold ~none:false ~some:has_barrier s2
  | SWhile (_, body) | SFor (_, _, _, _, body) -> has_barrier body
  | SMatch (_, cases) -> List.exists (fun (_, s) -> has_barrier s) cases
  | SLet (_, _, body) | SLetMut (_, _, body) | SPragma (_, body) | SBlock body
    ->
      has_barrier body
  | SAssign _ | SReturn _ | SExpr _ | SMemFence | SEmpty | SNative _ -> false

(** Extract constant offset from index expression. Returns Some (base, offset)
    if idx = base + const or base - const. *)
let extract_offset idx =
  match idx with
  | EBinop (Add, base, EConst (CInt32 n)) -> Some (base, Int32.to_int n)
  | EBinop (Sub, base, EConst (CInt32 n)) -> Some (base, -Int32.to_int n)
  | EBinop (Add, EConst (CInt32 n), base) -> Some (base, Int32.to_int n)
  | _ -> None

(** Check if index is a base expression (no offset) *)
let is_base_index idx =
  match idx with EVar _ | EIntrinsic _ -> true | _ -> false

(** Analyze access pattern for an array *)
let analyze_pattern reads arr =
  let arr_reads = List.filter (fun (a, _) -> a = arr) reads in
  match arr_reads with
  | [] -> Unknown
  | [(_, idx)] -> OneToOne idx
  | _ ->
      (* Check if all reads use same base with constant offsets -> stencil *)
      let offsets_and_bases =
        List.filter_map
          (fun (_, idx) ->
            match extract_offset idx with
            | Some (base, off) -> Some (base, off)
            | None when is_base_index idx -> Some (idx, 0)
            | None -> None)
          arr_reads
      in
      if List.length offsets_and_bases = List.length arr_reads then
        (* All accesses have extractable offsets *)
        match offsets_and_bases with
        | [] -> Unknown
        | (base, _) :: rest ->
            (* Check all have same base *)
            if List.for_all (fun (b, _) -> expr_equal b base) rest then
              let offsets = List.map snd offsets_and_bases in
              Stencil (List.sort_uniq compare offsets)
            else Gather
      else Gather

(** Analyze a kernel to extract fusion information.

    Performs static analysis to determine:
    - Which arrays are read and with what access patterns
    - Which arrays are written and with what access patterns
    - Whether barriers are present (prevents fusion)
    - Whether atomic operations are present (TODO: not yet detected)

    Access patterns determine fusability:
    - OneToOne: Element-wise access (arr[tid]) - always fusable
    - Stencil: Neighbor access (arr[tid±k]) - fusable with care
    - Reduction: Associative reduction - special fusion rules
    - Gather: Irregular access (arr[f(tid)]) - generally not fusable

    @param k Kernel to analyze
    @return Fusion information record *)
let analyze (k : kernel) : fusion_info =
  let reads = collect_reads_stmt [] k.kern_body in
  let writes = collect_writes_stmt [] k.kern_body in

  (* Group by array name *)
  let read_arrs = List.sort_uniq compare (List.map fst reads) in
  let write_arrs = List.sort_uniq compare (List.map fst writes) in

  {
    reads = List.map (fun arr -> (arr, analyze_pattern reads arr)) read_arrs;
    writes = List.map (fun arr -> (arr, analyze_pattern writes arr)) write_arrs;
    has_barriers = has_barrier k.kern_body;
    has_atomics = false;
    (* TODO: detect atomics *)
  }

(** {1 Fusion} *)

(** Find the expression written to arr[idx] in a statement. Returns None if not
    found or if multiple/conditional writes. *)
let rec find_write_expr stmt arr idx =
  match stmt with
  | SAssign (LArrayElem (a, i), e) when a = arr && expr_equal i idx -> Some e
  | SSeq stmts ->
      (* Find first write *)
      List.fold_left
        (fun acc s ->
          match acc with Some _ -> acc | None -> find_write_expr s arr idx)
        None
        stmts
  | SIf _ | SWhile _ | SFor _ | SMatch _ ->
      (* Can't safely extract from conditional *)
      None
  | SLet (_, _, body) | SLetMut (_, _, body) | SPragma (_, body) | SBlock body
    ->
      find_write_expr body arr idx
  | SAssign _ | SReturn _ | SExpr _ | SBarrier | SWarpBarrier | SMemFence
  | SEmpty | SNative _ ->
      None

(** Check if two kernels can be fused via an intermediate array.

    Vertical fusion eliminates intermediate arrays by inlining the producer's
    computation into the consumer. Requirements:

    1. Producer writes to intermediate with OneToOne pattern 2. Consumer reads
    from intermediate with OneToOne pattern 3. Both use same index expression
    (typically thread ID) 4. No barriers in either kernel (would break
    synchronization) 5. Intermediate not used elsewhere in consumer

    Example fusable pattern:
    {[
      (* Producer: temp[tid] = a[tid] * 2 *)
      (* Consumer: b[tid] = temp[tid] + 1 *)
      (* Fused: b[tid] = (a[tid] * 2) + 1 *)
    ]}

    @param producer First kernel (writes to intermediate)
    @param consumer Second kernel (reads from intermediate)
    @param intermediate Name of array to eliminate
    @return true if fusion is safe and beneficial *)
let can_fuse (producer : kernel) (consumer : kernel) (intermediate : string) :
    bool =
  let prod_info = analyze producer in
  let cons_info = analyze consumer in

  (* Check producer writes to intermediate *)
  let prod_writes_inter =
    List.exists (fun (arr, _) -> arr = intermediate) prod_info.writes
  in

  (* Check consumer reads from intermediate *)
  let cons_reads_inter =
    List.exists (fun (arr, _) -> arr = intermediate) cons_info.reads
  in

  (* Check patterns are compatible *)
  let patterns_ok =
    match
      ( List.assoc_opt intermediate prod_info.writes,
        List.assoc_opt intermediate cons_info.reads )
    with
    | Some (OneToOne prod_idx), Some (OneToOne cons_idx) ->
        expr_equal prod_idx cons_idx
    | _ -> false
  in

  (* No barriers in either *)
  let no_barriers =
    (not prod_info.has_barriers) && not cons_info.has_barriers
  in

  prod_writes_inter && cons_reads_inter && patterns_ok && no_barriers

(** Fuse producer into consumer, eliminating intermediate array.

    Creates a new kernel that combines both computations: 1. Takes consumer's
    structure (parameters, signature) 2. Replaces reads of intermediate[idx]
    with producer's computation 3. Removes intermediate array from parameters 4.
    Preserves all other operations from both kernels

    The transformation is safe because can_fuse verified:
    - Same index expression used by both
    - No barriers (so no ordering requirements)
    - OneToOne access patterns (no data races)

    Example:
    {[
      (* Before: *)
      let prod = temp[tid] <- a[tid] * 2 in
      let cons = b[tid] <- temp[tid] + 1 in

      (* After fusion: *)
      let fused = b[tid] <- (a[tid] * 2) + 1 in
    ]}

    @param producer First kernel (computation to inline)
    @param consumer Second kernel (receives inlined code)
    @param intermediate Array to eliminate
    @return New fused kernel *)
let fuse (producer : kernel) (consumer : kernel) (intermediate : string) :
    kernel =
  (* Find the index variable used to access intermediate in consumer *)
  let cons_reads = collect_reads_stmt [] consumer.kern_body in
  let inter_idx =
    List.find_map
      (fun (arr, idx) -> if arr = intermediate then Some idx else None)
      cons_reads
  in

  match inter_idx with
  | None ->
      (* Consumer doesn't read intermediate, just return consumer *)
      consumer
  | Some idx -> (
      (* Find what producer writes to intermediate[idx] *)
      let prod_expr = find_write_expr producer.kern_body intermediate idx in
      match prod_expr with
      | None ->
          (* Can't find simple write, return consumer unchanged *)
          consumer
      | Some replacement ->
          (* Substitute in consumer *)
          let fused_body =
            subst_array_read_stmt
              intermediate
              idx
              replacement
              consumer.kern_body
          in

          (* Remove intermediate from params if it was a param *)
          let fused_params =
            List.filter
              (fun d ->
                match d with
                | DParam (v, _) -> v.var_name <> intermediate
                | DShared (name, _, _) -> name <> intermediate
                | _ -> true)
              consumer.kern_params
          in

          (* Add producer's params that aren't already in consumer *)
          let prod_param_names =
            List.filter_map
              (fun d ->
                match d with
                | DParam (v, _) -> Some v.var_name
                | DShared (name, _, _) -> Some name
                | _ -> None)
              producer.kern_params
          in

          let cons_param_names =
            List.filter_map
              (fun d ->
                match d with
                | DParam (v, _) -> Some v.var_name
                | DShared (name, _, _) -> Some name
                | _ -> None)
              fused_params
          in

          let _ = prod_param_names in
          (* suppress warning *)

          let new_params =
            List.filter
              (fun d ->
                match d with
                | DParam (v, _) ->
                    v.var_name <> intermediate
                    && not (List.mem v.var_name cons_param_names)
                | DShared (name, _, _) ->
                    name <> intermediate && not (List.mem name cons_param_names)
                | _ -> true)
              producer.kern_params
          in

          {
            kern_name = consumer.kern_name ^ "_fused";
            kern_params = fused_params @ new_params;
            kern_locals = consumer.kern_locals @ producer.kern_locals;
            kern_body = fused_body;
            kern_types = consumer.kern_types @ producer.kern_types;
            kern_variants = consumer.kern_variants @ producer.kern_variants;
            kern_funcs = consumer.kern_funcs @ producer.kern_funcs;
            kern_native_fn = None;
          })

(** {1 High-level interface} *)

(** Find arrays written by [producer] and read by [consumer]. *)
let candidate_intermediates (producer : kernel) (consumer : kernel) :
    string list =
  let prod_info = analyze producer in
  let cons_info = analyze consumer in
  let prod_writes = List.map fst prod_info.writes in
  let cons_reads = List.map fst cons_info.reads in
  List.filter (fun arr -> List.mem arr cons_reads) prod_writes

(** Fuse a pipeline of kernels sequentially, preserving unfused stages.

    Attempts to fuse each consecutive kernel pair, eliminating intermediate
    arrays. Processes left-to-right, greedily fusing when possible.

    Algorithm: 1. Start with the first kernel as the current stage. 2. For each
    subsequent kernel:
    - Find arrays that current writes and next reads (candidates).
    - If fusable, replace the current stage and next stage with their fused
      kernel.
    - Otherwise, append the current stage to the result pipeline and continue
      from the next kernel. 3. Append the final current stage.

    Every input kernel is preserved unless it is replaced by a proven fused
    kernel. The returned kernel list is therefore the complete optimized
    pipeline in execution order.

    Note: This is greedy and may not find optimal fusion. For example, fusing
    A→B might prevent better fusion of B→C. Use auto_fuse_pipeline_list for
    heuristic-based fusion that considers these tradeoffs.

    @param kernels List of kernels to fuse (must be non-empty)
    @return (optimized_pipeline, eliminated_intermediates)
    @raise Empty_pipeline if kernel list is empty *)
let fuse_pipeline_list (kernels : kernel list) : kernel list * string list =
  match kernels with
  | [] ->
      Fusion_error.raise_error
        (Fusion_error.Empty_pipeline {function_name = "fuse_pipeline_list"})
  | [k] -> ([k], [])
  | k1 :: rest ->
      let rec loop acc current eliminated = function
        | [] -> (List.rev (current :: acc), List.rev eliminated)
        | k :: ks -> (
            let intermediates = candidate_intermediates current k in
            match intermediates with
            | inter :: _ when can_fuse current k inter ->
                let fused = fuse current k inter in
                loop acc fused (inter :: eliminated) ks
            | _ ->
                (* Can't fuse this pair; keep the current stage. *)
                loop (current :: acc) k eliminated ks)
      in
      loop [] k1 [] rest

(** Fuse a pipeline that is expected to collapse to one kernel.

    Prefer {!fuse_pipeline_list} for general pipeline optimization. This legacy
    wrapper returns a single kernel only when every preserved stage has fused
    into one result. If any unfused kernel remains, it raises [Invalid_fusion]
    instead of silently dropping stages.

    @param kernels List of kernels to fuse (must be non-empty)
    @return (fused_kernel, eliminated_intermediates)
    @raise Empty_pipeline if kernel list is empty
    @raise Invalid_fusion if the optimized pipeline still has multiple kernels
*)
let fuse_pipeline (kernels : kernel list) : kernel * string list =
  let pipeline, eliminated = fuse_pipeline_list kernels in
  match pipeline with
  | [kernel] -> (kernel, eliminated)
  | _ ->
      Fusion_error.raise_error
        (Fusion_error.Invalid_fusion
           {
             kernel = "fuse_pipeline";
             reason =
               Printf.sprintf
                 "optimized pipeline contains %d kernels; use \
                  fuse_pipeline_list to preserve unfused stages"
                 (List.length pipeline);
           })

(** {1 Reduction Fusion}

    Fuses a map kernel with a subsequent reduction, eliminating the intermediate
    array. Pattern:
    - Map: temp[i] = f(input[i])
    - Reduce: result = fold(op, temp)
    - Fused: result = fold(op, f(input[i])) *)

(** Detect if a statement is a reduction pattern. Returns the accumulator
    variable, the reduction operator, the array being reduced, and the loop
    body. *)
let detect_reduction_pattern stmt =
  (* Look for pattern: for i = 0 to n: acc = acc op arr[i] *)
  match stmt with
  | SFor (loop_var, _start, _stop, Upto, body) -> (
      match body with
      | SAssign (LVar acc, EBinop (op, EVar acc', EArrayRead (arr, EVar idx)))
        when acc.var_id = acc'.var_id && idx.var_id = loop_var.var_id ->
          Some (acc, op, arr, loop_var)
      | SAssign (LVar acc, EBinop (op, EArrayRead (arr, EVar idx), EVar acc'))
        when acc.var_id = acc'.var_id && idx.var_id = loop_var.var_id ->
          Some (acc, op, arr, loop_var)
      | _ -> None)
  | _ -> None

(** Check if a kernel is a reduction over an array *)
let is_reduction_kernel (k : kernel) (arr : string) : binop option =
  let rec find_reduction = function
    | SSeq stmts -> List.find_map find_reduction stmts
    | stmt -> (
        match detect_reduction_pattern stmt with
        | Some (_acc, op, reduced_arr, _) when reduced_arr = arr -> Some op
        | _ -> None)
  in
  find_reduction k.kern_body

(** Check if a map kernel can be fused with a reduction kernel.

    Requirements: 1. Map writes to intermediate with OneToOne pattern 2.
    Reduction reads from intermediate in a reduction pattern 3. No barriers in
    either kernel *)
let can_fuse_reduction (map_kernel : kernel) (reduce_kernel : kernel)
    (intermediate : string) : bool =
  let map_info = analyze map_kernel in
  let reduce_info = analyze reduce_kernel in

  (* Map must write to intermediate with OneToOne *)
  let map_writes_inter =
    match List.assoc_opt intermediate map_info.writes with
    | Some (OneToOne _) -> true
    | _ -> false
  in

  (* Reduction must read from intermediate *)
  let reduce_reads_inter = is_reduction_kernel reduce_kernel intermediate in

  (* No barriers *)
  let no_barriers =
    (not map_info.has_barriers) && not reduce_info.has_barriers
  in

  map_writes_inter && Option.is_some reduce_reads_inter && no_barriers

(** Fuse a map kernel into a reduction kernel, eliminating intermediate array.

    The resulting kernel: 1. Has reduction structure 2. Inlines map computation
    at each reduction step 3. Does not reference the intermediate array *)
let fuse_reduction (map_kernel : kernel) (reduce_kernel : kernel)
    (intermediate : string) : kernel =
  (* Find what map writes to intermediate[idx] *)
  let map_idx =
    (* Get the index expression from map's write *)
    let writes = collect_writes_stmt [] map_kernel.kern_body in
    List.find_map
      (fun (arr, idx) -> if arr = intermediate then Some idx else None)
      writes
  in

  match map_idx with
  | None -> reduce_kernel (* Can't find map write, return unchanged *)
  | Some _idx -> (
      (* Find the map expression: what is written to intermediate *)
      let map_expr = find_write_expr map_kernel.kern_body intermediate _idx in
      match map_expr with
      | None -> reduce_kernel
      | Some replacement ->
          (* In the reduction, substitute arr[loop_var] with map_expr *)
          let rec transform_stmt stmt =
            match stmt with
            | SFor (loop_var, start, stop, dir, body) -> (
                match detect_reduction_pattern stmt with
                | Some (_acc, _op, arr, _) when arr = intermediate ->
                    (* This is the reduction loop - substitute array reads *)
                    let new_body =
                      subst_array_read_stmt
                        intermediate
                        (EVar loop_var)
                        replacement
                        body
                    in
                    SFor (loop_var, start, stop, dir, new_body)
                | _ -> SFor (loop_var, start, stop, dir, transform_stmt body))
            | SSeq stmts -> SSeq (List.map transform_stmt stmts)
            | SIf (cond, s1, s2) ->
                SIf (cond, transform_stmt s1, Option.map transform_stmt s2)
            | SWhile (cond, body) -> SWhile (cond, transform_stmt body)
            | SMatch (e, cases) ->
                SMatch (e, List.map (fun (p, s) -> (p, transform_stmt s)) cases)
            | other -> other
          in

          let fused_body = transform_stmt reduce_kernel.kern_body in

          (* Remove intermediate from params, add map's input params *)
          let fused_params =
            List.filter
              (fun d ->
                match d with
                | DParam (v, _) -> v.var_name <> intermediate
                | DShared (name, _, _) -> name <> intermediate
                | _ -> true)
              reduce_kernel.kern_params
          in

          let cons_param_names =
            List.filter_map
              (fun d ->
                match d with
                | DParam (v, _) -> Some v.var_name
                | DShared (name, _, _) -> Some name
                | _ -> None)
              fused_params
          in

          let new_params =
            List.filter
              (fun d ->
                match d with
                | DParam (v, _) ->
                    v.var_name <> intermediate
                    && not (List.mem v.var_name cons_param_names)
                | DShared (name, _, _) ->
                    name <> intermediate && not (List.mem name cons_param_names)
                | _ -> true)
              map_kernel.kern_params
          in

          {
            kern_name = reduce_kernel.kern_name ^ "_fused";
            kern_params = fused_params @ new_params;
            kern_locals = reduce_kernel.kern_locals @ map_kernel.kern_locals;
            kern_body = fused_body;
            kern_types = reduce_kernel.kern_types @ map_kernel.kern_types;
            kern_variants =
              reduce_kernel.kern_variants @ map_kernel.kern_variants;
            kern_funcs = reduce_kernel.kern_funcs @ map_kernel.kern_funcs;
            kern_native_fn = None;
          })

(** Try to fuse map+reduce, falling back to regular fusion if not applicable *)
let try_fuse (producer : kernel) (consumer : kernel) (intermediate : string) :
    kernel option =
  if can_fuse_reduction producer consumer intermediate then
    Some (fuse_reduction producer consumer intermediate)
  else if can_fuse producer consumer intermediate then
    Some (fuse producer consumer intermediate)
  else None

(** {1 Stencil Fusion}

    Fuses stencil operations with tiling. When producer uses stencil pattern and
    consumer also uses stencil, the combined radius determines halo size.

    Example:
    - Producer: temp[i] = (input[i-1] + input[i] + input[i+1]) / 3 (radius 1)
    - Consumer: out[i] = (temp[i-1] + temp[i] + temp[i+1]) / 3 (radius 1)
    - Fused: out[i] computed from input[i-2..i+2] (combined radius 2) *)

(** Compute stencil radius from offset list *)
let stencil_radius offsets =
  List.fold_left (fun acc o -> max acc (abs o)) 0 offsets

(** Get stencil info for array access in a kernel *)
let get_stencil_info (k : kernel) (arr : string) : int list option =
  let info = analyze k in
  match List.assoc_opt arr info.reads with
  | Some (Stencil offsets) -> Some offsets
  | Some (OneToOne _) -> Some [0] (* OneToOne is stencil with radius 0 *)
  | _ -> None

(** Check if two stencil kernels can be fused.

    Requirements: 1. Producer writes to intermediate 2. Consumer reads from
    intermediate with stencil pattern 3. No barriers in either kernel 4.
    Producer's output stencil is compatible *)
let can_fuse_stencil (producer : kernel) (consumer : kernel)
    (intermediate : string) : bool =
  let prod_info = analyze producer in
  let cons_info = analyze consumer in

  (* Producer must write to intermediate *)
  let prod_writes_inter =
    List.exists (fun (arr, _) -> arr = intermediate) prod_info.writes
  in

  (* Consumer must read from intermediate with stencil or one-to-one *)
  let cons_stencil =
    match List.assoc_opt intermediate cons_info.reads with
    | Some (Stencil _) -> true
    | Some (OneToOne _) -> true
    | _ -> false
  in

  (* No barriers *)
  let no_barriers =
    (not prod_info.has_barriers) && not cons_info.has_barriers
  in

  prod_writes_inter && cons_stencil && no_barriers

(** Information about fused stencil *)
type stencil_fusion_info = {
  combined_radius : int;  (** Total radius after fusion *)
  producer_radius : int;  (** Radius of producer's input stencil *)
  consumer_radius : int;  (** Radius of consumer's intermediate stencil *)
  input_arrays : string list;  (** Arrays read by producer *)
}

(** Analyze stencil fusion parameters *)
let analyze_stencil_fusion (producer : kernel) (consumer : kernel)
    (intermediate : string) : stencil_fusion_info option =
  let prod_info = analyze producer in
  let cons_info = analyze consumer in

  (* Get producer's input stencil radius *)
  let prod_input_radius =
    List.fold_left
      (fun acc (arr, pattern) ->
        if arr <> intermediate then
          match pattern with
          | Stencil offsets -> max acc (stencil_radius offsets)
          | OneToOne _ -> acc
          | _ -> acc
        else acc)
      0
      prod_info.reads
  in

  (* Get consumer's intermediate stencil radius *)
  let cons_inter_radius =
    match List.assoc_opt intermediate cons_info.reads with
    | Some (Stencil offsets) -> Some (stencil_radius offsets)
    | Some (OneToOne _) -> Some 0
    | _ -> None
  in
  match cons_inter_radius with
  | None -> None
  | Some cons_inter_radius ->
      (* Input arrays from producer *)
      let input_arrays =
        List.filter_map
          (fun (arr, _) -> if arr <> intermediate then Some arr else None)
          prod_info.reads
      in

      Some
        {
          combined_radius = prod_input_radius + cons_inter_radius;
          producer_radius = prod_input_radius;
          consumer_radius = cons_inter_radius;
          input_arrays;
        }

(** Substitute all stencil reads of an array in an expression. For each read
    arr[base + offset], substitute with the producer's computation shifted by
    that offset. *)
let rec subst_stencil_reads arr base_idx producer_expr offset_shift expr =
  match expr with
  | EArrayRead (a, idx) when a = arr -> (
      match extract_offset idx with
      | Some (base, off) when expr_equal base base_idx ->
          (* Shift the producer expression by the offset *)
          shift_expr producer_expr (off + offset_shift)
      | None when expr_equal idx base_idx ->
          shift_expr producer_expr offset_shift
      | _ -> expr)
  | EBinop (op, e1, e2) ->
      EBinop
        ( op,
          subst_stencil_reads arr base_idx producer_expr offset_shift e1,
          subst_stencil_reads arr base_idx producer_expr offset_shift e2 )
  | EUnop (op, e) ->
      EUnop (op, subst_stencil_reads arr base_idx producer_expr offset_shift e)
  | EIntrinsic (path, name, args) ->
      EIntrinsic
        ( path,
          name,
          List.map
            (subst_stencil_reads arr base_idx producer_expr offset_shift)
            args )
  | ECast (ty, e) ->
      ECast (ty, subst_stencil_reads arr base_idx producer_expr offset_shift e)
  | ETuple es ->
      ETuple
        (List.map
           (subst_stencil_reads arr base_idx producer_expr offset_shift)
           es)
  | _ -> expr

(** Shift all array indices in an expression by a constant offset *)
and shift_expr expr offset =
  if offset = 0 then expr
  else
    let shift_idx idx =
      if offset > 0 then EBinop (Add, idx, EConst (CInt32 (Int32.of_int offset)))
      else EBinop (Sub, idx, EConst (CInt32 (Int32.of_int (-offset))))
    in
    match expr with
    | EArrayRead (arr, idx) -> EArrayRead (arr, shift_idx idx)
    | EBinop (op, e1, e2) ->
        EBinop (op, shift_expr e1 offset, shift_expr e2 offset)
    | EUnop (op, e) -> EUnop (op, shift_expr e offset)
    | EIntrinsic (path, name, args) ->
        EIntrinsic (path, name, List.map (fun e -> shift_expr e offset) args)
    | ECast (ty, e) -> ECast (ty, shift_expr e offset)
    | ETuple es -> ETuple (List.map (fun e -> shift_expr e offset) es)
    | _ -> expr

(** Fuse stencil kernels, substituting intermediate reads with producer
    computation *)
let fuse_stencil (producer : kernel) (consumer : kernel) (intermediate : string)
    : kernel =
  (* Find the index and expression producer writes to intermediate *)
  let writes = collect_writes_stmt [] producer.kern_body in
  let write_info = List.find_opt (fun (arr, _) -> arr = intermediate) writes in

  match write_info with
  | None -> consumer
  | Some (_, write_idx) -> (
      let prod_expr =
        find_write_expr producer.kern_body intermediate write_idx
      in
      match prod_expr with
      | None -> consumer
      | Some replacement ->
          (* Substitute each read of intermediate in consumer *)
          let rec transform_stmt stmt =
            match stmt with
            | SAssign (lv, e) ->
                let e' =
                  subst_stencil_reads intermediate write_idx replacement 0 e
                in
                SAssign (lv, e')
            | SSeq stmts -> SSeq (List.map transform_stmt stmts)
            | SIf (cond, s1, s2) ->
                let cond' =
                  subst_stencil_reads intermediate write_idx replacement 0 cond
                in
                SIf (cond', transform_stmt s1, Option.map transform_stmt s2)
            | SWhile (cond, body) ->
                let cond' =
                  subst_stencil_reads intermediate write_idx replacement 0 cond
                in
                SWhile (cond', transform_stmt body)
            | SFor (v, start, stop, dir, body) ->
                SFor (v, start, stop, dir, transform_stmt body)
            | SMatch (e, cases) ->
                let e' =
                  subst_stencil_reads intermediate write_idx replacement 0 e
                in
                SMatch (e', List.map (fun (p, s) -> (p, transform_stmt s)) cases)
            | other -> other
          in

          let fused_body = transform_stmt consumer.kern_body in

          (* Merge params, removing intermediate *)
          let fused_params =
            List.filter
              (fun d ->
                match d with
                | DParam (v, _) -> v.var_name <> intermediate
                | DShared (name, _, _) -> name <> intermediate
                | _ -> true)
              consumer.kern_params
          in

          let cons_param_names =
            List.filter_map
              (fun d ->
                match d with
                | DParam (v, _) -> Some v.var_name
                | DShared (name, _, _) -> Some name
                | _ -> None)
              fused_params
          in

          let new_params =
            List.filter
              (fun d ->
                match d with
                | DParam (v, _) ->
                    v.var_name <> intermediate
                    && not (List.mem v.var_name cons_param_names)
                | DShared (name, _, _) ->
                    name <> intermediate && not (List.mem name cons_param_names)
                | _ -> true)
              producer.kern_params
          in

          {
            kern_name = consumer.kern_name ^ "_stencil_fused";
            kern_params = fused_params @ new_params;
            kern_locals = consumer.kern_locals @ producer.kern_locals;
            kern_body = fused_body;
            kern_types = consumer.kern_types @ producer.kern_types;
            kern_variants = consumer.kern_variants @ producer.kern_variants;
            kern_funcs = consumer.kern_funcs @ producer.kern_funcs;
            kern_native_fn = None;
          })

(** Enhanced try_fuse that includes stencil fusion *)
let try_fuse_all (producer : kernel) (consumer : kernel) (intermediate : string)
    : kernel option =
  if can_fuse_reduction producer consumer intermediate then
    Some (fuse_reduction producer consumer intermediate)
  else if can_fuse_stencil producer consumer intermediate then
    Some (fuse_stencil producer consumer intermediate)
  else if can_fuse producer consumer intermediate then
    Some (fuse producer consumer intermediate)
  else None

(** {1 Phase 5: Auto-Fusion Heuristics}

    Simple pattern-based heuristics for deciding when fusion is beneficial.
    Conservative - never recommends fusion when it would hurt. *)

(** Fusion decision *)
type fusion_decision =
  | Fuse  (** Definitely fuse *)
  | DontFuse  (** Definitely don't fuse *)
  | MaybeFuse  (** User should decide / profile *)

(** Fusion hint with decision and reason *)
type fusion_hint = {decision : fusion_decision; reason : string}

(** Heuristic: decide if fusion should be applied.

    Not all legal fusions are beneficial. This function applies heuristics to
    decide when fusion improves performance:

    **Fuse (always beneficial):**
    - Element-wise producer → element-wise consumer (eliminates memory)
    - Map → Reduction (eliminates temporary, enables tree reduction)

    **Don't fuse (likely harmful):**
    - Any kernel with barriers (breaks synchronization semantics)
    - Producer → Gather consumer (unpredictable memory access)
    - Large stencil radius (>3 neighbors, shared memory better)

    **Maybe fuse (profile to decide):**
    - Small stencil (≤3 neighbors, computation cost vs memory)
    - Unknown patterns (conservative - don't auto-fuse)

    @param producer First kernel
    @param consumer Second kernel
    @param intermediate Array between them
    @return Decision (Fuse/DontFuse/MaybeFuse) with human-readable reason *)
let should_fuse (producer : kernel) (consumer : kernel) (intermediate : string)
    : fusion_hint =
  let prod_info = analyze producer in
  let cons_info = analyze consumer in

  (* Rule 1: Can't fuse if barriers present *)
  if prod_info.has_barriers || cons_info.has_barriers then
    {decision = DontFuse; reason = "Barrier prevents fusion"}
  else
    (* Rule 2: Always fuse simple element-wise (OneToOne -> OneToOne) *)
    match
      ( List.assoc_opt intermediate prod_info.writes,
        List.assoc_opt intermediate cons_info.reads )
    with
    | Some (OneToOne prod_idx), Some (OneToOne cons_idx)
      when expr_equal prod_idx cons_idx ->
        {decision = Fuse; reason = "Element-wise producer/consumer"}
    | Some (OneToOne _), Some (OneToOne _) ->
        {
          decision = DontFuse;
          reason = "Producer and consumer use different element indices";
        }
    (* Rule 3: Map -> Reduction is always good *)
    | Some (OneToOne _), Some (Reduction _) ->
        {decision = Fuse; reason = "Map-reduce pattern"}
    (* Rule 4: Stencil needs shared memory - maybe beneficial *)
    | Some (OneToOne _), Some (Stencil offsets) ->
        if List.length offsets <= 3 then
          {decision = MaybeFuse; reason = "Small stencil - profile to decide"}
        else
          {decision = DontFuse; reason = "Large stencil radius - keep separate"}
    (* Rule 5: Gather patterns - don't fuse (unpredictable access) *)
    | Some (OneToOne _), Some Gather ->
        {decision = DontFuse; reason = "Gather pattern - keep separate"}
    (* Rule 6: Unknown - be conservative *)
    | _ ->
        {decision = MaybeFuse; reason = "Unknown pattern - profile to decide"}

(** Auto-fuse a pipeline using heuristics, preserving unfused stages.

    Similar to fuse_pipeline_list, but uses should_fuse heuristics to decide
    which fusions to apply. This prevents harmful fusions while still
    eliminating obvious intermediates.

    Decision process for each candidate fusion:
    - **Fuse**: Apply fusion, eliminate intermediate
    - **DontFuse**: Skip this pair and preserve both kernels
    - **MaybeFuse**: Conservative - skip (user can manually fuse if profiling
      shows benefit), preserving both kernels

    This is the recommended entry point for automatic optimization. Manual
    fusion (via fuse_pipeline_list) is available for expert users who have
    profiled specific fusion decisions.

    Example usage:
    {[
      (* Define pipeline *)
      let kernels = [map_kernel; filter_kernel; reduce_kernel] in

      (* Auto-fuse with heuristics *)
      let pipeline, eliminated, skipped =
        Sarek_fusion.auto_fuse_pipeline_list kernels
      in

      (* Execute each kernel in optimized pipeline order. *)
      List.iter
        (fun ir -> Execute.run_vectors ~device ~ir ~args ~block ~grid ())
        pipeline
    ]}

    @param kernels Pipeline to optimize (must be non-empty)
    @return (optimized_pipeline, eliminated_arrays, skipped_reasons)
    @raise Empty_pipeline if kernel list is empty *)
let auto_fuse_pipeline_list (kernels : kernel list) :
    kernel list * string list * string list =
  match kernels with
  | [] ->
      Fusion_error.raise_error
        (Fusion_error.Empty_pipeline {function_name = "auto_fuse_pipeline_list"})
  | [k] -> ([k], [], [])
  | k1 :: rest ->
      let rec loop acc current eliminated skipped = function
        | [] ->
            (List.rev (current :: acc), List.rev eliminated, List.rev skipped)
        | k :: ks -> (
            let candidates = candidate_intermediates current k in
            match candidates with
            | [] -> loop (current :: acc) k eliminated skipped ks
            | inter :: _ -> (
                let hint = should_fuse current k inter in
                match hint.decision with
                | Fuse -> (
                    match try_fuse current k inter with
                    | Some fused ->
                        loop acc fused (inter :: eliminated) skipped ks
                    | None ->
                        loop
                          (current :: acc)
                          k
                          eliminated
                          (hint.reason :: skipped)
                          ks)
                | DontFuse ->
                    loop
                      (current :: acc)
                      k
                      eliminated
                      (hint.reason :: skipped)
                      ks
                | MaybeFuse ->
                    (* Conservative: don't auto-fuse uncertain cases *)
                    loop
                      (current :: acc)
                      k
                      eliminated
                      (hint.reason :: skipped)
                      ks))
      in
      loop [] k1 [] [] rest

(** Auto-fuse a pipeline that is expected to collapse to one kernel.

    Prefer {!auto_fuse_pipeline_list} for general pipeline optimization. This
    legacy wrapper returns a single kernel only when every preserved stage has
    fused into one result. If any unfused kernel remains, it raises
    [Invalid_fusion] instead of silently dropping stages.

    @param kernels Pipeline to optimize (must be non-empty)
    @return (fused_kernel, eliminated_arrays, skipped_reasons)
    @raise Empty_pipeline if kernel list is empty
    @raise Invalid_fusion if the optimized pipeline still has multiple kernels
*)
let auto_fuse_pipeline (kernels : kernel list) :
    kernel * string list * string list =
  let pipeline, eliminated, skipped = auto_fuse_pipeline_list kernels in
  match pipeline with
  | [kernel] -> (kernel, eliminated, skipped)
  | _ ->
      Fusion_error.raise_error
        (Fusion_error.Invalid_fusion
           {
             kernel = "auto_fuse_pipeline";
             reason =
               Printf.sprintf
                 "optimized pipeline contains %d kernels; use \
                  auto_fuse_pipeline_list to preserve unfused stages"
                 (List.length pipeline);
           })

(** Print fusion analysis for a pipeline.

    Diagnostic function that analyzes each consecutive kernel pair and prints
    detailed information about fusability:
    - Access patterns (OneToOne, Stencil, Reduction, etc.)
    - Candidate intermediate arrays
    - Fusion decision and reasoning
    - Potential benefits or risks

    Useful for understanding why auto_fuse_pipeline_list made specific decisions
    or for identifying manual fusion opportunities.

    @param kernels Pipeline to analyze *)
let analyze_pipeline (kernels : kernel list) : unit =
  Printf.printf "=== Fusion Analysis ===\n" ;
  let pairs =
    match kernels with
    | [] | [_] -> []
    | k1 :: rest ->
        let rec pairs acc prev = function
          | [] -> List.rev acc
          | k :: ks -> pairs ((prev, k) :: acc) k ks
        in
        pairs [] k1 rest
  in
  List.iteri
    (fun i (prod, cons) ->
      let prod_info = analyze prod in
      let cons_info = analyze cons in
      let candidates =
        List.filter
          (fun arr -> List.mem_assoc arr cons_info.reads)
          (List.map fst prod_info.writes)
      in
      Printf.printf "\n[%d] %s -> %s\n" i prod.kern_name cons.kern_name ;
      List.iter
        (fun inter ->
          let hint = should_fuse prod cons inter in
          let symbol =
            match hint.decision with
            | Fuse -> "[FUSE]"
            | DontFuse -> "[SKIP]"
            | MaybeFuse -> "[MAYBE]"
          in
          Printf.printf "  %s via '%s': %s\n" symbol inter hint.reason)
        candidates)
    pairs ;
  Printf.printf "\n"
