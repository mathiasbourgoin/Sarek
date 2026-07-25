(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Codegen guard for the Sarek_df64 contraction barrier. CPU-only: generates
 * PTX from the IR and inspects the string, no CUDA device required.
 *
 * WHY THIS TEST EXISTS
 *
 * df64's error-free transformations are only exact if the backend compiler
 * does not fuse a multiply into a neighbouring add or subtract. ptxas does
 * exactly that by default, and on sm_61 (GTX 1070, CUDA 12.9) it rebuilt the
 * exact product a.hi*b.hi inside the quick_two_sum closing df64_mul, undoing
 * the TwoProd split and collapsing mul/div/sqrt from ~2^-47 to ~2^-24 - plain
 * float32. Nothing failed loudly; the numbers were simply single precision.
 *
 * Sarek_df64.two_prod therefore forms its product with [fma _ _ 0.0], which is
 * already fused and so cannot be fused again. This test asserts the property
 * that fix delivers, at the only place it is observable without the hardware:
 * the emitted PTX for a df64 kernel must contain no unqualified [mul.f32],
 * because every such instruction is a licence for ptxas to contract.
 *
 * It is deliberately a PTX-string assertion rather than a precision check:
 * the precision check (test_df64.ml) passes on every device this project can
 * reach locally, which is precisely how the sm_61 regression went unnoticed.
 ******************************************************************************)

module Vector = Spoc_core.Vector

type float32 = float

type ('a, 'b) vector = ('a, 'b) Vector.t

let%sarek_include _ = "../../Sarek_df64/Sarek_df64.ml"

let mul_kernel =
  [%kernel
    fun (a : Sarek_df64.df64 vector)
        (b : Sarek_df64.df64 vector)
        (out : Sarek_df64.df64 vector)
        (n : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then out.(tid) <- df64_mul a.(tid) b.(tid)]

let sqrt_kernel =
  [%kernel
    fun (a : Sarek_df64.df64 vector)
        (out : Sarek_df64.df64 vector)
        (n : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then out.(tid) <- df64_sqrt a.(tid)]

let div_kernel =
  [%kernel
    fun (a : Sarek_df64.df64 vector)
        (b : Sarek_df64.df64 vector)
        (out : Sarek_df64.df64 vector)
        (n : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then out.(tid) <- df64_div a.(tid) b.(tid)]

(* df64_of_int32 is the fourth caller of two_prod, and the only one outside
   mul/div/sqrt. It is covered so that the guard tracks two_prod's call sites
   rather than a hand-picked three. *)
let of_int32_kernel =
  [%kernel
    fun (out : Sarek_df64.df64 vector) (n : int32) ->
      let open Sarek_df64 in
      let tid = thread_idx_x + (block_idx_x * block_dim_x) in
      if tid < n then out.(tid) <- df64_of_int32 tid]

let ptx_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir ->
      Sarek_codegen.Sarek_ir_ptx.generate_with_types
        ~types:ir.Sarek_ir_types.kern_types
        ir
  | None -> failwith "kernel has no IR"

(* Count occurrences of [needle] in [hay] (no Str dependency here). *)
let count needle hay =
  let nl = String.length needle and hl = String.length hay in
  let rec go i acc =
    if i + nl > hl then acc
    else if String.sub hay i nl = needle then go (i + 1) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

(* First line of [ptx] containing [needle], for a self-diagnosing failure. *)
let first_line_with needle ptx =
  let rec go = function
    | [] -> None
    | l :: tl -> if count needle l > 0 then Some (String.trim l) else go tl
  in
  go (String.split_on_char '\n' ptx)

let failures = ref 0

let check name ptx =
  (* An unqualified [mul.f32] is contractable; a rounding-qualified multiply
     ([mul.rn.f32]) and [fma.rn.f32] are not, since PTX forbids fusing an
     operation that carries an explicit rounding modifier. df64 must emit none
     of the former. NB the emitter never currently produces [mul.rn.f32] - the
     f32 [Mul] arm emits bare [mul.f32] (Sarek_ir_ptx_expr.ml) - so in practice
     zero here means every float product went through the fma barrier. *)
  let contractable = count "mul.f32 " ptx in
  let fused = count "fma.rn.f32 " ptx in
  let ok = contractable = 0 && fused > 0 in
  Printf.printf
    "  %-10s contractable mul.f32 = %d (want 0), fma.rn.f32 = %d (want > 0) %s\n\
     %!"
    name
    contractable
    fused
    (if ok then "PASS" else "FAIL") ;
  if not ok then begin
    incr failures ;
    if contractable > 0 then begin
      Printf.printf
        "    -> df64 %s emits a contractable multiply; ptxas may fuse it into \
         the\n\
        \       following add/sub and silently degrade df64 to float32.\n\
        \       See \"Contraction barrier\" in sarek/Sarek_df64/Sarek_df64.ml.\n\
         %!"
        name ;
      match first_line_with "mul.f32 " ptx with
      | Some l -> Printf.printf "       first offending line: %s\n%!" l
      | None -> ()
    end ;
    if fused = 0 then
      Printf.printf
        "    -> df64 %s emits no fma.rn.f32 at all; the TwoProd error extraction\n\
        \       is missing entirely, which is a worse failure than contraction.\n\
         %!"
        name
  end

(* The Newton seed of df64_sqrt. [sqrt.approx.f32] is ~1 ulp rather than
   correctly rounded, and it was the ONLY non-correctly-rounded instruction in
   the whole emitted df64_sqrt body -- every other op is fma.rn / div.rn / an
   exact add-sub. Same bug class as [div.approx.f32] (audit finding M2), one
   operator over.

   SCOPE OF THIS ASSERTION: it proves which instruction is EMITTED, nothing
   more. It does not measure precision, and the precision this buys on NVIDIA
   hardware has NOT been remeasured since the lowering changed -- see the
   KNOWN RESIDUAL block in sarek/Sarek_df64/Sarek_df64.ml for what is and is
   not established.

   Both needles are anchored with a leading space, because "sqrt.approx.f32"
   is a SUBSTRING of "rsqrt.approx.f32": [rsqrt] deliberately keeps the
   approximate form, and rsqrt is the textbook Newton seed for a square root,
   so an unanchored count here would fail with a message asserting the exact
   opposite of what happened the day someone reseeds df64_sqrt with it. The
   [check] function above anchors for the same reason (trailing space on
   "mul.f32 "). *)
let check_sqrt_seed ptx =
  let approx = count " sqrt.approx.f32 " ptx in
  let exact = count " sqrt.rn.f32 " ptx in
  let ok = approx = 0 && exact > 0 in
  Printf.printf
    "  %-10s sqrt.approx.f32 = %d (want 0), sqrt.rn.f32 = %d (want > 0) %s\n%!"
    "df64_seed"
    approx
    exact
    (if ok then "PASS" else "FAIL") ;
  if not ok then begin
    incr failures ;
    Printf.printf
      "    -> df64_sqrt's Newton seed is not correctly rounded; the Karp\n\
      \       correction cannot recover an error already in its own seed.\n\
      \       See \"KNOWN RESIDUAL\" in sarek/Sarek_df64/Sarek_df64.ml.\n\
       %!"
  end

let () =
  print_endline "Sarek_df64 PTX contraction guard:" ;
  check "df64_mul" (ptx_of mul_kernel) ;
  check "df64_div" (ptx_of div_kernel) ;
  check "df64_sqrt" (ptx_of sqrt_kernel) ;
  check_sqrt_seed (ptx_of sqrt_kernel) ;
  check "df64_of_i32" (ptx_of of_int32_kernel) ;
  if !failures = 0 then print_endline "test_df64_no_contraction PASSED"
  else begin
    Printf.printf "test_df64_no_contraction FAILED (%d failures)\n" !failures ;
    exit 1
  end
