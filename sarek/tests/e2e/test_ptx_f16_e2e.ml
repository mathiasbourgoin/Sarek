(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Scalar f16 on the CUDA/PTX backend, executed from SAREK-GENERATED PTX.
 *
 * f16 is a storage type: the PTX emitter loads and stores 16-bit values and
 * converts with [cvt.f32.f16] / [cvt.rn.f16.f32]; all arithmetic is f32. This
 * test runs the kernels Sarek generates over EVERY finite binary16 value
 * (63488) and compares bit for bit with (a) the Interpreter and (b) an
 * independent OCaml model (round to binary32 after every operation, narrow
 * with the interpreter's binary16 rounding). Shapes:
 *
 *   mul        f16(x * 1.1)                 the narrowing fed by an inexact op
 *   chain      f16(f16(x * 1.1) + 1000)     two narrowings in a row
 *   roundtrip  out = inp                    raw 16-bit load and store
 *   trunc      f16(float(int(x)))           f16 -> int32 (truncating) -> f16
 *   fromint    f16(float(i))                int32 -> f16, i = 0 .. 63487
 *
 * Positive controls: each shape has at least one reference that is
 * deliberately wrong and is required to DIFFER from the Interpreter (host
 * only, any machine) and from the device output, so the comparison can see a
 * wrongly rounded or mis-converted result:
 *   demoted    arithmetic done in binary16 with 1.1 narrowed to binary16 (what
 *              a compiler fusing the narrowing into the operation can produce);
 *   const      the constant one binary32 ulp above 1.1;
 *   flushed    binary16 subnormals flushed to zero;
 *   rounded    round-to-nearest instead of truncation for f16 -> int;
 *   shifted    the index off by one.
 *
 * The device part runs on every CUDA/PTX device and self-skips (exit 0) when
 * there is none; the host checks (interpreter = model, controls red, the
 * generated PTX contains the conversions) always run. On a machine whose CUDA
 * driver is ZLUDA, put its directory first on LD_LIBRARY_PATH; the device name
 * printed below says which implementation ran.
 *
 * On a machine with an NVIDIA GPU and its driver (the line
 * "=== f16 on CUDA/PTX: <device> (framework CUDA/PTX) ===" must appear):
 *   dune build sarek/tests/e2e/test_ptx_f16_e2e.exe
 *   ./_build/default/sarek/tests/e2e/test_ptx_f16_e2e.exe
 * On NVIDIA every "CUDA/PTX = ..." line is expected to be OK. On ZLUDA/AMD the
 * `mul` and `chain` lines differ (2913 and 620 inputs): the AMDGPU lowering
 * fuses the multiply into the narrowing, see docs/fp-contraction-policy.md.
 ******************************************************************************)

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Gpu = Sarek_stdlib.Gpu

let () = Test_helpers.Benchmarks.init_backends ()

let k_mul =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <- float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1)]

let k_chain =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (float32_of_float16
               (float16_of_float32 (float32_of_float16 inp.(tid) *. 1.1))
            +. 1000.0)]

let k_roundtrip =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then out.(tid) <- inp.(tid)]

let k_trunc =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then
        out.(tid) <-
          float16_of_float32
            (float
               (int_of_float
                  (Float64.of_float32 (float32_of_float16 inp.(tid)))))]

let k_fromint =
  [%kernel
    fun (out : float16 vector) (inp : float16 vector) (n : int32) ->
      let open Sarek_stdlib.Std in
      let tid = global_thread_id in
      if tid < n then out.(tid) <- float16_of_float32 (float tid)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* ---- the domain and the independent model ---- *)

let round16 = Sarek_interp.Sarek_float16.to_float16

(* Plain IEEE round-to-nearest-even binary32 (not the interpreter's
   subnormal-flushing [Sarek_float32.to_float32]). *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let c11 = f32 1.1

let c11_16 = round16 1.1

let c11_up = Int32.float_of_bits (Int32.succ (Int32.bits_of_float c11))

let finite_f16 =
  let value bits =
    let sign = if bits land 0x8000 <> 0 then -1.0 else 1.0 in
    let exp = (bits lsr 10) land 0x1f and man = bits land 0x3ff in
    if exp = 31 then None
    else if exp = 0 then Some (sign *. float_of_int man *. ldexp 1.0 (-24))
    else Some (sign *. float_of_int (1024 + man) *. ldexp 1.0 (exp - 25))
  in
  List.init 0x10000 Fun.id |> List.filter_map value |> Array.of_list

let count = Array.length finite_f16

(* The reference functions take the input INDEX as well: the [fromint] shapes
   ignore the input value. *)
type shape = {
  name : string;
  ir : Sarek_ir_types.kernel;
  model : int -> float -> float;
  controls : (string * (int -> float -> float)) list;
  ptx_marker : string;  (** an instruction the generated PTX must contain *)
  fused : (int -> float -> float) option;
      (** The result of a compiler that fuses the f32 operation into the
          narrowing (one rounding). Not what NVIDIA does; known to be what
          ZLUDA's AMDGPU lowering does. Accepted only on a ZLUDA device. *)
  with_n : bool;  (** hand-built IR kernels take no [n] argument *)
}

let flush_subnormal x = if Float.abs x < ldexp 1.0 (-14) then 0.0 else x

(* Single-rounding f64 -> binary16, round to nearest even. (The interpreter's
   [round16] double-rounds through binary32, which is exactly the difference the
   fused models are about.) *)
let round16_single v =
  if v = 0.0 || Float.is_nan v then v
  else
    let a = Float.abs v in
    let e = max (snd (Float.frexp a) - 1) (-14) in
    let ulp = ldexp 1.0 (e - 10) in
    let q = a /. ulp in
    let fl = Float.floor q in
    let d = q -. fl in
    let r =
      if d > 0.5 then fl +. 1.0
      else if d < 0.5 then fl
      else if Float.rem fl 2.0 = 0.0 then fl
      else fl +. 1.0
    in
    let res = r *. ulp in
    Float.copy_sign (if res >= 65520.0 then infinity else res) v

(* f16(x*1.1) with the product absorbed into the narrowing: the exact product
   (binary16 x binary32 fits a double) rounded once; [+. 0.0] is the [a*b + 0]
   form that turns -0 into +0. *)
let fused_mul x = round16_single ((x *. c11) +. 0.0)

(* The chain with only its inner narrowing fused; the outer add and narrowing
   stay f32 + plain. *)
let fused_chain x = round16 (f32 (fused_mul x +. 1000.0))

let shapes =
  [
    {
      name = "mul       f16(x*1.1)";
      ir = ir_of k_mul;
      ptx_marker = "cvt.rn.f16.f32";
      model = (fun _ x -> round16 (f32 (x *. c11)));
      controls =
        [
          ("demoted", fun _ x -> round16 (x *. c11_16));
          ("const", fun _ x -> round16 (f32 (x *. c11_up)));
        ];
      fused = Some (fun _ x -> fused_mul x);
      with_n = true;
    };
    {
      name = "chain     f16(f16(x*1.1)+1000)";
      ir = ir_of k_chain;
      ptx_marker = "cvt.f32.f16";
      model = (fun _ x -> round16 (f32 (round16 (f32 (x *. c11)) +. 1000.0)));
      controls =
        [
          ("demoted", fun _ x -> round16 (round16 (x *. c11_16) +. 1000.0));
          ( "const",
            fun _ x -> round16 (f32 (round16 (f32 (x *. c11_up)) +. 1000.0)) );
        ];
      fused = Some (fun _ x -> fused_chain x);
      with_n = true;
    };
    {
      name = "roundtrip out = inp";
      ir = ir_of k_roundtrip;
      ptx_marker = "ld.global.b16";
      model = (fun _ x -> x);
      controls = [("flushed", fun _ x -> flush_subnormal x)];
      fused = None;
      with_n = true;
    };
    {
      (* Composite path through intrinsics: f16 -> f32 -> f64 -> int32 (cvt.rzi)
         -> f32 -> f16. The direct casts are the two shapes after it. *)
      name = "trunc     f16(float(int(f64(x))))  [composite, intrinsics]";
      ir = ir_of k_trunc;
      ptx_marker = "cvt.rzi.s32.f64";
      (* through int32: -0.3 truncates to 0, not to -0 *)
      model = (fun _ x -> round16 (Float.of_int (Float.to_int x)));
      controls = [("rounded", fun _ x -> round16 (Float.round x))];
      fused = None;
      with_n = true;
    };
    {
      name = "fromint   f16(float(i))  [composite, intrinsic]";
      ir = ir_of k_fromint;
      ptx_marker = "cvt.rn.f32.s32";
      model = (fun i _ -> round16 (float_of_int i));
      controls = [("shifted", fun i _ -> round16 (float_of_int (i + 1)))];
      fused = None;
      with_n = true;
    };
    {
      (* hand IR: ECast (TFloat16, ECast (TInt32, f16 read)) *)
      name = "direct    ECast(f16, ECast(int32, f16 x))  [hand IR]";
      ir = Ptx_f16_kernels.direct_trunc ();
      ptx_marker = "cvt.rzi.s32.f32";
      model = (fun _ x -> round16 (Float.of_int (Float.to_int x)));
      controls = [("rounded", fun _ x -> round16 (Float.round x))];
      fused = None;
      with_n = false;
    };
    {
      (* hand IR: ECast (TFloat16, int32 index) *)
      name = "direct    ECast(f16, int32 i)  [hand IR]";
      ir = Ptx_f16_kernels.direct_fromint ();
      ptx_marker = "cvt.rn.f32.s32";
      model = (fun i _ -> round16 (float_of_int i));
      controls = [("shifted", fun i _ -> round16 (float_of_int (i + 1)))];
      fused = None;
      with_n = false;
    };
  ]

(* ---- runs ---- *)

let inputs () =
  let inp = Vector.create Vector.float16 count in
  Array.iteri (Vector.set inp) finite_f16 ;
  (Vector.create Vector.float16 count, inp)

let block = 256

let grid = (count + block - 1) / block

let args_of sh out inp =
  if sh.with_n then Execute.[Vec out; Vec inp; Int count]
  else Execute.[Vec out; Vec inp]

let run_device dev sh =
  let out, inp = inputs () in
  Execute.run_vectors
    ~device:dev
    ~ir:sh.ir
    ~args:(args_of sh out inp)
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d grid)
    () ;
  Transfer.flush dev ;
  Vector.to_array out

let run_interp sh =
  let out, inp = inputs () in
  Execute.run_interpreter_vectors
    ~ir:sh.ir
    ~args:(args_of sh out inp)
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d grid)
    ~parallel:false ;
  Vector.to_array out

(* ---- comparing ---- *)

let same_bits g e =
  Int64.bits_of_float g = Int64.bits_of_float e || (g <> g && e <> e)

let failures = ref 0

(* [check label ~expect_equal got reference]: prints how many inputs disagree
   and the first one. *)
let check label ~expect_equal got reference =
  let n = ref 0 and first = ref None in
  Array.iteri
    (fun i g ->
      let e = reference i finite_f16.(i) in
      if not (same_bits g e) then begin
        incr n ;
        if !first = None then first := Some (finite_f16.(i), g, e)
      end)
    got ;
  let good = if expect_equal then !n = 0 else !n > 0 in
  if not good then incr failures ;
  Printf.printf
    "  %-4s %-44s %5d/%d differ%s\n%!"
    (if good then "OK" else "FAIL")
    label
    !n
    count
    (match !first with
    | Some (x, g, e) when !n > 0 ->
        Printf.sprintf "  (first: x=%h got %h reference %h)" x g e
    | _ -> "")

let contains ~needle s =
  let n = String.length needle and l = String.length s in
  let rec go i = i + n <= l && (String.sub s i n = needle || go (i + 1)) in
  go 0

(* The generated PTX must carry the conversions this test is about; the device
   check then cannot pass through some other lowering by accident. *)
let check_ptx_text sh =
  let ptx = Sarek_codegen.Sarek_ir_ptx.generate sh.ir in
  let must = [sh.ptx_marker; "st.global.b16"; ".reg .b16 %h"] in
  let missing = List.filter (fun n -> not (contains ~needle:n ptx)) must in
  if missing <> [] then begin
    incr failures ;
    Printf.printf "  FAIL PTX lacks %s\n%!" (String.concat ", " missing)
  end

let host_part sh =
  Printf.printf "%s\n%!" sh.name ;
  check_ptx_text sh ;
  let interp = run_interp sh in
  check "interpreter = model" ~expect_equal:true interp sh.model ;
  List.iter
    (fun (cname, c) ->
      check
        (Printf.sprintf "control %s vs interpreter (red)" cname)
        ~expect_equal:false
        interp
        c)
    sh.controls ;
  (* The fused model must be distinguishable from the interpreter, or accepting
     it on ZLUDA would accept anything. *)
  Option.iter
    (fun f ->
      check "fused model vs interpreter (red)" ~expect_equal:false interp f)
    sh.fused ;
  interp

(* ZLUDA runs PTX on an AMD GPU through LLVM's AMDGPU backend, which may absorb
   the f32 multiply into the f32 -> f16 narrowing (one rounding where the DSL
   mandates two). That is a property of the consumer, not of the PTX, and it is
   not what NVIDIA does; it is tolerated here only on a device that names itself
   ZLUDA, and only as the explicit [fused] model, element by element. *)
let is_zluda dev = contains ~needle:"ZLUDA" dev.Device.name

let device_part_fused dev sh f got interp =
  let n_model = ref 0
  and n_fused = ref 0
  and n_neither = ref 0
  and n_int = ref 0 in
  Array.iteri
    (fun i g ->
      let x = finite_f16.(i) in
      if not (same_bits g interp.(i)) then incr n_int ;
      let m = same_bits g (sh.model i x) and fu = same_bits g (f i x) in
      if m then incr n_model ;
      if fu then incr n_fused ;
      if not (m || fu) then incr n_neither)
    got ;
  let good = !n_neither = 0 in
  if not good then incr failures ;
  Printf.printf
    "  %-4s %s: every element equals the interpreter model or the fused model \
     (%d equal the model, %d the fused model, %d neither)\n\
    \       known non-NVIDIA deviation: %d/%d differ from the interpreter\n\
     %!"
    (if good then "OK" else "FAIL")
    dev.Device.name
    !n_model
    !n_fused
    !n_neither
    !n_int
    count

let device_part dev sh interp =
  Printf.printf "%s\n%!" sh.name ;
  let got = run_device dev sh in
  match sh.fused with
  | Some f when is_zluda dev -> device_part_fused dev sh f got interp
  | _ ->
      check "CUDA/PTX = model" ~expect_equal:true got sh.model ;
      check "CUDA/PTX = interpreter" ~expect_equal:true got (fun i _ ->
          interp.(i)) ;
      List.iter
        (fun (cname, c) ->
          check
            (Printf.sprintf "control %s vs device (red)" cname)
            ~expect_equal:false
            got
            c)
        sh.controls

let () =
  let ptx =
    Device.init () |> Array.to_list
    |> List.filter (fun d -> d.Device.framework = "CUDA/PTX")
  in
  let interps = List.map host_part shapes in
  (match ptx with
  | [] ->
      print_endline "[SKIP] no CUDA/PTX device: the device checks did not run"
  | devs ->
      List.iter
        (fun dev ->
          Printf.printf
            "=== f16 on CUDA/PTX: %s (framework %s) ===\n%!"
            dev.Device.name
            dev.Device.framework ;
          List.iter2 (device_part dev) shapes interps)
        devs) ;
  if !failures > 0 then begin
    Printf.printf "[FAIL] %d check(s)\n" !failures ;
    exit 1
  end
  else print_endline "[PASS]"
