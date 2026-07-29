(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Tier 1b device-side SoA emitter equivalence.
 *
 * Tier 1a proved the host SoA transpose + scalar transfer path. This test
 * proves the Tier 1b EMITTER: a single custom (record) vector kernel parameter
 * lowered as Structure-of-Arrays (Sarek_ir_ptx.generate ~soa_params) — N
 * per-leaf base pointers + coalesced per-leaf scalar loads — computes the same
 * result on CUDA/PTX as the default AoS lowering of the very same kernel IR,
 * and as a pure-OCaml reference.
 *
 * Mechanics: the same [%kernel] IR is compiled twice.
 *   - AoS: run via Execute.run_vectors with the single custom vector argument
 *     (backend generate_source, default packed layout).
 *   - SoA: Sarek_ir_ptx.generate ~soa_params:[<the custom vector param>] emits
 *     N pointer params + one length; the AoS host buffer is transposed into N
 *     contiguous leaf vectors (Spoc_core.Soa.scatter) and fed positionally via
 *     Execute.run_source ~inject_lengths:false — exactly the N-base-pointer ABI
 *     the emitter now produces. (The user-facing Vector.create ~layout:SoA +
 *     automatic launch expansion is Tier 1c; this drives the emitter directly.)
 *
 * SoA is PTX-only in this deliverable, so the SoA leg runs on CUDA/PTX devices
 * only; the AoS leg + reference run everywhere and are always checked. f32 and
 * f64 leaves are exercised end-to-end here; i32/i64 leaf codegen is covered at
 * the PTX-instruction + ptxas-assembly level in tests/unit/test_ptx_snapshot.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Soa = Spoc_core.Soa
module Soa_vector = Spoc_core.Soa_vector
module Soa_launch = Sarek.Soa_launch
module Benchmarks = Test_helpers.Benchmarks
open Sarek_codegen

type ('a, 'b) vector = ('a, 'b) Vector.t

type float32 = float

type float64 = float

(* Fields mutable so the round-trip kernel can write a leaf in place
   (pts.(i).y <- ...); the read legs are unaffected. *)
type point3d = {mutable x : float32; mutable y : float32; mutable z : float32}
[@@sarek.type]

type dpair = {u : float64; v : float64} [@@sarek.type]

(* f32 headline case: reads three fields of a custom vector and sums them. *)
let p3_kernel =
  snd
    [%kernel
      fun (pts : point3d vector) (out : float32 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          let p = pts.(tid) in
          out.(tid) <- p.x +. p.y +. p.z]

(* f64 case: two 8-byte leaves. *)
let dpair_kernel =
  snd
    [%kernel
      fun (pv : dpair vector) (out : float64 vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then out.(tid) <- pv.(tid).u +. pv.(tid).v]

(* Write case: scales the y leaf in place. Exercises the SoA field STORE path
   (D2H leaf readback + gather round-trip on the host side). *)
let p3_scale_y_kernel =
  snd
    [%kernel
      fun (pts : point3d vector) (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then pts.(tid).y <- pts.(tid).y *. 2.0]

let ir_of kirc =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

(* Name of the first (custom vector) kernel parameter — what we lower as SoA. *)
let first_param_name (ir : Sarek_ir_types.kernel) =
  match ir.Sarek_ir_types.kern_params with
  | Sarek_ir_types.DParam (v, _) :: _ -> v.Sarek_ir_types.var_name
  | _ -> failwith "kernel has no parameters"

let is_ptx (dev : Device.t) = dev.Device.framework = "CUDA/PTX"

let dims threads = Sarek.Execute.dims1d threads

(* Launch the SoA compilation of [ir] whose first param (a flat 2/3-field record
   vector) is lowered SoA. [leaves] are the per-leaf scalar vectors (declaration
   order); [out] the scalar output. Arg order mirrors the emitted param block:
   leaf pointers, the shared length, then (out ptr, out length), then n — all
   with inject_lengths:false so we control every slot. *)
let run_soa dev ir ~leaves ~out ~n ~block ~grid =
  let ptx = Sarek_ir_ptx.generate ~soa_params:[first_param_name ir] ir in
  let leaf_args = List.map (fun v -> Sarek.Execute.Vec v) leaves in
  let len = Sarek.Execute.Int32 (Int32.of_int n) in
  let args =
    leaf_args @ [len; Sarek.Execute.Vec out; len; Sarek.Execute.Int n]
  in
  Sarek.Execute.run_source
    ~device:dev
    ~source:ptx
    ~lang:Sarek.Execute.PTX
    ~kernel_name:ir.Sarek_ir_types.kern_name
    ~block
    ~grid
    ~inject_lengths:false
    args ;
  Transfer.flush dev

(* Tier 1c: the SAME kernel driven through the real user-facing API —
   Soa_vector storage + Soa_launch.run_soa. Unlike run_soa above (which pokes
   the emitter directly), this exercises the whole host path: SoA storage
   allocation, host AoS->leaf scatter, per-leaf H2D transfer, N-base-pointer
   launch expansion, and the CUDA/PTX gate. [sv] is the SoA input vector (kernel
   param 0), [out] the scalar output, [n] the length. *)
let run_soa_via_api dev ir ~sv ~out ~n ~block ~grid =
  Soa_launch.run_soa
    ~device:dev
    ~ir
    ~args:
      [
        Soa_launch.SA_Soa sv;
        Soa_launch.SA_Reg (Sarek.Execute.Vec out);
        Soa_launch.SA_Reg (Sarek.Execute.Int n);
      ]
    ~block
    ~grid
    () ;
  Transfer.flush dev

(* ---- point3d (f32) ---- *)

let run_p3 dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let src = Vector.create_custom point3d_custom n in
  for i = 0 to n - 1 do
    Vector.set
      src
      i
      {
        x = float_of_int i;
        y = (float_of_int i *. 0.5) +. 1.0;
        z = float_of_int (n - i);
      }
  done ;
  let ir = ir_of p3_kernel in
  (* AoS *)
  let out_aos = Vector.create Vector.float32 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec src; Vec out_aos; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  (* SoA (PTX only) *)
  let out_soa =
    if not (is_ptx dev) then None
    else begin
      let plan =
        Soa.plan
          ~name:"point3d"
          Sarek_ir_types.[("x", TFloat32); ("y", TFloat32); ("z", TFloat32)]
      in
      let xs = Vector.create Vector.float32 n in
      let ys = Vector.create Vector.float32 n in
      let zs = Vector.create Vector.float32 n in
      Soa.scatter
        plan
        ~aos:(Vector.to_ctypes_ptr src)
        ~length:n
        ~leaves:
          [|
            Vector.to_ctypes_ptr xs;
            Vector.to_ctypes_ptr ys;
            Vector.to_ctypes_ptr zs;
          |] ;
      let out = Vector.create Vector.float32 n in
      run_soa dev ir ~leaves:[xs; ys; zs] ~out ~n ~block ~grid ;
      Some out
    end
  in
  (* SoA via the real user-facing API (Soa_vector + Soa_launch.run_soa). *)
  let out_api =
    if not (is_ptx dev) then None
    else begin
      let sv = Soa_vector.create point3d_custom n in
      for i = 0 to n - 1 do
        Soa_vector.set
          sv
          i
          {
            x = float_of_int i;
            y = (float_of_int i *. 0.5) +. 1.0;
            z = float_of_int (n - i);
          }
      done ;
      let out = Vector.create Vector.float32 n in
      run_soa_via_api dev ir ~sv ~out ~n ~block ~grid ;
      Some out
    end
  in
  let reference i =
    let p = Vector.get src i in
    p.x +. p.y +. p.z
  in
  (out_aos, out_soa, out_api, reference)

(* ---- dpair (f64) ---- *)

let run_dpair dev n =
  let threads = min 128 n in
  let block = dims threads and grid = dims ((n + threads - 1) / threads) in
  let src = Vector.create_custom dpair_custom n in
  for i = 0 to n - 1 do
    Vector.set
      src
      i
      {u = float_of_int i *. 1.5; v = float_of_int (n - i) -. 0.25}
  done ;
  let ir = ir_of dpair_kernel in
  let out_aos = Vector.create Vector.float64 n in
  Sarek.Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Vec src; Vec out_aos; Int n]
    ~block
    ~grid
    () ;
  Transfer.flush dev ;
  let out_soa =
    if not (is_ptx dev) then None
    else begin
      let plan =
        Soa.plan ~name:"dpair" Sarek_ir_types.[("u", TFloat64); ("v", TFloat64)]
      in
      let us = Vector.create Vector.float64 n in
      let vs = Vector.create Vector.float64 n in
      Soa.scatter
        plan
        ~aos:(Vector.to_ctypes_ptr src)
        ~length:n
        ~leaves:[|Vector.to_ctypes_ptr us; Vector.to_ctypes_ptr vs|] ;
      let out = Vector.create Vector.float64 n in
      run_soa dev ir ~leaves:[us; vs] ~out ~n ~block ~grid ;
      Some out
    end
  in
  let out_api =
    if not (is_ptx dev) then None
    else begin
      let sv = Soa_vector.create dpair_custom n in
      for i = 0 to n - 1 do
        Soa_vector.set
          sv
          i
          {u = float_of_int i *. 1.5; v = float_of_int (n - i) -. 0.25}
      done ;
      let out = Vector.create Vector.float64 n in
      run_soa_via_api dev ir ~sv ~out ~n ~block ~grid ;
      Some out
    end
  in
  let reference i =
    let p = Vector.get src i in
    p.u +. p.v
  in
  (out_aos, out_soa, out_api, reference)

let check name dev n runner =
  Printf.printf
    "SoA-emitter %s [%s] %s: %!"
    name
    dev.Device.framework
    dev.Device.name ;
  try
    let out_aos, out_soa, out_api, reference = runner dev n in
    let ok = ref true in
    let check_leg label o a r i =
      match o with
      | None -> ()
      | Some o ->
          let s = Vector.get o i in
          if abs_float (s -. r) > 1e-3 || abs_float (s -. a) > 1e-4 then begin
            ok := false ;
            if i < 5 then
              Printf.printf
                "\n  %s mismatch @%d: %s=%f aos=%f ref=%f%!"
                label
                i
                label
                s
                a
                r
          end
    in
    for i = 0 to n - 1 do
      let r = reference i in
      let a = Vector.get out_aos i in
      if abs_float (a -. r) > 1e-3 then begin
        ok := false ;
        if i < 5 then
          Printf.printf "\n  AoS mismatch @%d: aos=%f ref=%f%!" i a r
      end ;
      (* Direct-emitter SoA leg. *)
      check_leg "SoA" out_soa a r i ;
      (* Real user-facing API leg (Soa_vector + Soa_launch.run_soa). *)
      check_leg "SoA-API" out_api a r i
    done ;
    let soa_note =
      match out_soa with None -> " (SoA skipped: non-PTX)" | Some _ -> ""
    in
    if !ok then (
      Printf.printf "PASSED%s\n%!" soa_note ;
      true)
    else (
      Printf.printf "FAILED\n%!" ;
      false)
  with e ->
    Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e) ;
    false

(* Item 3 gate: run_soa on a non-PTX device MUST raise a located error rather
   than binding the SoA N-pointer ABI to an AoS kernel signature (which would
   read wrong data). This is the "never wrong data" guarantee, checked
   concretely on whatever non-PTX backends are present. *)
let check_gate dev =
  if is_ptx dev then true
  else begin
    Printf.printf "SoA-gate [%s] %s: %!" dev.Device.framework dev.Device.name ;
    let ir = ir_of p3_kernel in
    let sv = Soa_vector.create point3d_custom 16 in
    let out = Vector.create Vector.float32 16 in
    match
      run_soa_via_api dev ir ~sv ~out ~n:16 ~block:(dims 16) ~grid:(dims 1)
    with
    | () | (exception Not_found) ->
        Printf.printf "FAILED (run_soa did not reject a non-PTX device)\n%!" ;
        false
    | exception Sarek.Execute_error.Execution_error _ ->
        Printf.printf "rejected (located error) OK\n%!" ;
        true
  end

(* Round-trip: a kernel WRITES the y leaf on the device; then we transfer each
   leaf back (D2H) and gather into the AoS vector, and check the AoS y values.
   Exercises the leaf-writeback + Soa_vector.gather path (no other shipped test
   writes an SoA leaf). CUDA/PTX only. *)
let check_roundtrip dev n =
  if not (is_ptx dev) then true
  else begin
    Printf.printf
      "SoA-roundtrip [%s] %s: %!"
      dev.Device.framework
      dev.Device.name ;
    try
      let threads = min 128 n in
      let block = dims threads and grid = dims ((n + threads - 1) / threads) in
      let sv = Soa_vector.create point3d_custom n in
      let orig i =
        {
          x = float_of_int i;
          y = (float_of_int i *. 0.5) +. 1.0;
          z = float_of_int (n - i);
        }
      in
      for i = 0 to n - 1 do
        Soa_vector.set sv i (orig i)
      done ;
      let ir = ir_of p3_scale_y_kernel in
      Soa_launch.run_soa
        ~device:dev
        ~ir
        ~args:[Soa_launch.SA_Soa sv; Soa_launch.SA_Reg (Sarek.Execute.Int n)]
        ~block
        ~grid
        () ;
      (* Device wrote the y leaf; round-trip explicitly per the run_soa
         contract: D2H every leaf, then gather back into the AoS vector. *)
      Array.iter
        (fun (Soa_vector.Leaf v) -> Transfer.to_cpu ~force:true v)
        (Soa_vector.leaves sv) ;
      Soa_vector.gather sv ;
      let ok = ref true in
      for i = 0 to n - 1 do
        let got = Soa_vector.get sv i in
        let o = orig i in
        (* y doubled; x and z untouched. *)
        if
          abs_float (got.y -. (o.y *. 2.0)) > 1e-3
          || abs_float (got.x -. o.x) > 1e-3
          || abs_float (got.z -. o.z) > 1e-3
        then begin
          ok := false ;
          if i < 5 then
            Printf.printf
              "\n\
              \  roundtrip mismatch @%d: got {x=%f;y=%f;z=%f} expected \
               {x=%f;y=%f;z=%f}%!"
              i
              got.x
              got.y
              got.z
              o.x
              (o.y *. 2.0)
              o.z
        end
      done ;
      if !ok then (
        Printf.printf "PASSED\n%!" ;
        true)
      else (
        Printf.printf "FAILED\n%!" ;
        false)
    with e ->
      Printf.printf "FAIL (%s)\n%!" (Printexc.to_string e) ;
      false
  end

(* ── the launch still checks the layout, on a DIFFERENT axis than create ─────
   History, because it changes what these cases are for. [Soa_vector.create] used
   to take the field layout as a [~fields] argument, and a wrong list transposed
   against the wrong byte offsets — silently corrupted data, not an error. The
   launch check below existed to catch that at the last moment before any data
   moved. [create] now DERIVES the layout from [custom_type.ir_fields], so a
   caller can no longer describe it wrongly and that particular hazard is gone at
   the source rather than intercepted here.

   These cases are still real, and still guard something [create] cannot see:
   [create] knows only the VECTOR's element type, while the launch also holds the
   KERNEL's [DParam] [TRecord]. Those are two independent declarations, and
   binding a vector of one record type to a kernel parameter of another is still
   expressible — a mismatch that no amount of deriving inside [create] can
   detect. That is the axis these cases pin, which is why they build their
   declared plans by hand instead of going through [create]: the point is
   precisely to present the check with a plan that disagrees with the kernel.

   Device-independent by construction: the check is a pure function of (param
   name, kernel element type, declared plan), so it runs on this machine with no
   NVIDIA device. It is also ordered BEFORE run_soa's PTX gate precisely so the
   refusal is reachable off a CUDA host — behind the gate it could only ever fire
   where the gate passes, which is what would have made it untestable here. *)

let xyz_ty =
  Sarek_ir_types.TRecord
    ("point3d", [("x", TFloat32); ("y", TFloat32); ("z", TFloat32)])

let mixed_ty = Sarek_ir_types.TRecord ("mixed", [("a", TInt32); ("b", TFloat64)])

(* [check_soa_layout] raises via Execute_error.raise_error; a refusal is any
   Execution_error whose rendering mentions the parameter. Asserting on the
   MESSAGE as well as the exception, because "raised something" would also be
   satisfied by an unrelated failure inside the plan builders. *)
let contains hay needle =
  let nh = String.length hay and nn = String.length needle in
  let rec go i =
    if i + nn > nh then false
    else if String.sub hay i nn = needle then true
    else go (i + 1)
  in
  nn = 0 || go 0

let refuses ~label ~param ~kernel_ty ~declared ~expect_substr =
  match Sarek.Soa_launch.check_soa_layout ~param ~kernel_ty ~declared with
  | () ->
      Printf.printf "  %-56s FAIL (accepted a mismatch)\n%!" label ;
      false
  | exception Sarek.Execute_error.Execution_error e ->
      let msg = Sarek.Execute_error.error_to_string e in
      let has_param = contains msg param in
      let has_expect = contains msg expect_substr in
      if has_param && has_expect then (
        Printf.printf "  %-56s OK (refused)\n%!" label ;
        true)
      else (
        Printf.printf
          "  %-56s FAIL (refused, but message names neither %S nor %S: %s)\n%!"
          label
          param
          expect_substr
          msg ;
        false)

(* The DERIVATION, which replaced the [~fields] argument (backlog-54 slice 1).
   [Soa_vector.create] now builds its plan from [custom_type.ir_fields], so this
   asserts the derived plan is the RIGHT one — the leaf list and stride these
   records used to be given by hand. Without it, deriving from a wrong source
   (say a reversed or truncated [ir_fields]) would still typecheck, still refuse
   nothing, and silently transpose at the wrong offsets: exactly the failure the
   argument's removal was meant to make unreachable.

   Device-independent — [create] allocates host buffers and touches no device, so
   this runs on a machine with no GPU at all. Stride is asserted alongside the
   leaves because the two are what scatter/gather index with, and a plan can have
   correct leaves with a wrong stride (padding), which would corrupt every
   element after the first. *)
let check_field_derivation () =
  let ok = ref true in
  let check_plan label (plan : Soa.plan) ~expect_leaves ~expect_stride =
    let got =
      List.map
        (fun (l : Soa.leaf) -> (l.Soa.path, l.Soa.aos_offset, l.Soa.size))
        plan.Soa.leaves
    in
    if got <> expect_leaves then (
      let show l =
        String.concat
          ", "
          (List.map (fun (p, o, s) -> Printf.sprintf "%s@%d:%d" p o s) l)
      in
      Printf.printf
        "  %-56s FAIL (leaves [%s], expected [%s])\n%!"
        label
        (show got)
        (show expect_leaves) ;
      ok := false)
    else if plan.Soa.aos_stride <> expect_stride then (
      Printf.printf
        "  %-56s FAIL (stride %d, expected %d)\n%!"
        label
        plan.Soa.aos_stride
        expect_stride ;
      ok := false)
    else Printf.printf "  %-56s OK\n%!" label
  in
  (* point3d: three 4-byte f32 leaves, packed, stride 12. *)
  check_plan
    "derived plan for point3d (3 x f32)"
    (Soa_vector.plan (Soa_vector.create point3d_custom 4))
    ~expect_leaves:[("x", 0, 4); ("y", 4, 4); ("z", 8, 4)]
    ~expect_stride:12 ;
  (* dpair: two 8-byte f64 leaves, stride 16. A different width AND a different
     leaf count, so a derivation hard-wired to point3d cannot pass both. *)
  check_plan
    "derived plan for dpair (2 x f64)"
    (Soa_vector.plan (Soa_vector.create dpair_custom 4))
    ~expect_leaves:[("u", 0, 8); ("v", 8, 8)]
    ~expect_stride:16 ;
  !ok

let check_layout_validation () =
  let authoritative = Soa.plan_of_elttype xyz_ty in
  let ok = ref true in
  print_endline "  --- precondition enforced at launch ---" ;
  (* POSITIVE CONTROL first. Without it, "refuses a wrong list" and "refuses
     every list" are the same observation, and the second would make SoA
     unusable rather than safe. *)
  if
    match
      Sarek.Soa_launch.check_soa_layout
        ~param:"pts"
        ~kernel_ty:(Some xyz_ty)
        ~declared:authoritative
    with
    | () -> true
    | exception e ->
        Printf.printf
          "  %-56s FAIL (rejected the CORRECT layout: %s)\n%!"
          "matching is accepted"
          (Printexc.to_string e) ;
        false
  then Printf.printf "  %-56s OK\n%!" "matching is accepted"
  else ok := false ;
  (* Wrong ORDER: same fields, same widths, permuted. The offsets move, so the
     transpose would read every field from the wrong column. *)
  if
    not
      (refuses
         ~label:"permuted is refused"
         ~param:"pts"
         ~kernel_ty:(Some xyz_ty)
         ~declared:
           (Soa.plan
              ~name:"point3d"
              [("y", Sarek_ir_types.TFloat32); ("x", TFloat32); ("z", TFloat32)])
         ~expect_substr:"wrong byte offsets")
  then ok := false ;
  (* Wrong WIDTH at the same position: f32 declared where the record has f64.
     This is the case a name-and-order-only comparison would accept. *)
  if
    not
      (refuses
         ~label:"wrong leaf WIDTH is refused"
         ~param:"m"
         ~kernel_ty:(Some mixed_ty)
         ~declared:
           (Soa.plan
              ~name:"mixed"
              [("a", Sarek_ir_types.TInt32); ("b", TFloat32)])
         ~expect_substr:"wrong byte offsets")
  then ok := false ;
  (* MISSING field: fewer leaves than the record has. *)
  if
    not
      (refuses
         ~label:"missing field is refused"
         ~param:"pts"
         ~kernel_ty:(Some xyz_ty)
         ~declared:
           (Soa.plan
              ~name:"point3d"
              [("x", Sarek_ir_types.TFloat32); ("y", TFloat32)])
         ~expect_substr:"wrong byte offsets")
  then ok := false ;
  (* A SoA argument bound to a SCALAR parameter: there is no record to compare
     against, and N leaf pointers cannot bind to it. *)
  if
    not
      (refuses
         ~label:"SoA bound to a non-array param is refused"
         ~param:"scalar"
         ~kernel_ty:None
         ~declared:authoritative
         ~expect_substr:"non-array")
  then ok := false ;
  !ok

(* The WIRING, which the direct calls above cannot establish: run_soa must
   actually consult the check. Driven on ANY device, including non-PTX, and that
   is the point — the mismatch must surface the LAYOUT error rather than the
   CUDA/PTX device gate. If the check sat behind the gate (where it originally
   was) this case would report the gate message instead, so it pins the ordering
   as well as the call.

   The mismatch is now built by binding a SoA vector of the WRONG RECORD TYPE to
   the parameter, not by handing [create] a wrong field list. That is a
   deliberate change of mechanism, forced by [create] deriving its layout from
   [ir_fields]: a permuted list is no longer expressible, so the case that used
   one could no longer fail for its stated reason. What IS still expressible is
   this: [SA_Soa] is existential ([SA_Soa : 'a Soa_vector.t -> soa_arg]), so the
   type system does not relate the vector's element type to the parameter's, and
   a [dpair] vector (2 x f64, stride 16) binds to a [point3d] parameter (3 x f32,
   stride 12) without complaint. Both the leaf list and the stride disagree, so
   the launch check is what stands between that and a kernel reading garbage. *)
let check_layout_wired dev =
  let ir = ir_of p3_kernel in
  let sv = Soa_vector.create dpair_custom 8 in
  let out = Vector.create Vector.float32 8 in
  match
    Soa_launch.run_soa
      ~device:dev
      ~ir
      ~args:
        [
          Soa_launch.SA_Soa sv;
          Soa_launch.SA_Reg (Sarek.Execute.Vec out);
          Soa_launch.SA_Reg (Sarek.Execute.Int 8);
        ]
      ~block:(dims 8)
      ~grid:(dims 1)
      ()
  with
  | () ->
      Printf.printf
        "  %-56s FAIL (ran with a mismatched record type)\n%!"
        "run_soa consults the layout check"
      |> fun () -> false
  | exception Sarek.Execute_error.Execution_error e ->
      let msg = Sarek.Execute_error.error_to_string e in
      if contains msg "wrong byte offsets" then (
        Printf.printf "  %-56s OK\n%!" "run_soa consults the layout check" ;
        true)
      else (
        Printf.printf
          "  %-56s FAIL (raised, but not the layout error: %s)\n%!"
          "run_soa consults the layout check"
          msg ;
        false)

let () =
  Benchmarks.init () ;
  let n = 1024 in
  (* Device-independent, so it runs BEFORE the no-device early exit — otherwise
     a machine with no device would report SKIPPED while silently not checking a
     property that needs no device at all. *)
  let derive_ok = check_field_derivation () in
  let layout_ok = check_layout_validation () && derive_ok in
  let devs = Device.all () in
  if Array.length devs = 0 then (
    print_endline
      "test_soa_emitter_equiv: no device - SKIPPED (layout validation still \
       checked above)" ;
    exit (if layout_ok then 0 else 1)) ;
  let any_ptx = Array.exists is_ptx devs in
  if not any_ptx then
    print_endline
      "test_soa_emitter_equiv: no CUDA/PTX device - SoA leg skipped (AoS + \
       reference still checked)" ;
  let ok = ref true in
  Array.iter
    (fun dev ->
      (* point3d (f32) runs everywhere: cross-backend AoS + reference, plus the
         PTX SoA leg. *)
      if not (check "point3d(f32)" dev n run_p3) then ok := false ;
      (* dpair (f64) exists to prove the f64 SoA leaf on PTX; run it on CUDA/PTX
         only. (Some non-PTX backends — e.g. OpenCL/radeonsi — have an unrelated
         f64 custom-vector gap that is out of scope for this emitter test and is
         exercised elsewhere.) *)
      if is_ptx dev && not (check "dpair(f64)" dev n run_dpair) then ok := false ;
      (* Item 3: SoA launch must be rejected (never wrong data) on non-PTX. *)
      if not (check_gate dev) then ok := false ;
      (* Wiring + ordering: a mismatched must surface the LAYOUT error,
         not the device gate, on this very non-PTX device. *)
      if not (check_layout_wired dev) then ok := false ;
      (* Leaf-write round-trip (D2H + gather) on CUDA/PTX. *)
      if is_ptx dev && not (check_roundtrip dev n) then ok := false)
    devs ;
  if not (!ok && layout_ok) then exit 1
