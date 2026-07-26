(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * E2E measurement: does the Vulkan driver HONOUR SPIR-V [NoContraction]?
 *
 * Issue #126. [Sarek_ir_glsl.gen_var_decl] prefixes every float local with
 * GLSL [precise], which glslang lowers to a SPIR-V [NoContraction] decoration
 * on each contributing arithmetic instruction. That the decoration is *emitted*
 * is settled (compiler-output tier, glslc 2026.2 / SPIRV-Tools 1.4.350.1).
 * Whether a driver *obeys* it is a separate question, and the two things
 * written down in this repository disagreed about the answer for Mesa RADV:
 * [Sarek_ir_glsl.ml] said [precise] was added because RADV needed it, campaign
 * notes said RADV ignores it. Neither was backed by a measurement.
 *
 * THE DISCRIMINATOR
 *
 * With a = b = 1 + 2^-12 and c = -(1 + 2^-11), all three exactly representable
 * in binary32:
 *
 *   exact a*b        = 1 + 2^-11 + 2^-24
 *   fl32(a*b)        = 1 + 2^-11        (2^-24 is exactly half an ulp at 1.0;
 *                                        ties-to-even rounds the trailing 1
 *                                        away)
 *   fl32(fl32(a*b)+c) = 0                <- separate  multiply then add
 *   fl32(fma(a,b,c))  = 2^-24 = 5.96e-08 <- contracted multiply-add
 *
 * So a contracted evaluation is not a one-ulp wobble here: it is the difference
 * between exactly zero and 5.96e-08. There is no way to read the result
 * ambiguously, and no tolerance to choose.
 *
 * THE EXPERIMENT
 *
 * The same shader source is compiled twice, differing ONLY by the [precise]
 * qualifier on the float locals (a [#define], so the two strings are otherwise
 * character-identical), and both are run on the same device, same driver, same
 * process. Reading the two columns together is what makes the result
 * conclusive:
 *
 *   plain contracts, precise does not  -> NoContraction is HONOURED
 *   both contract                      -> NoContraction is IGNORED
 *   neither contracts                  -> the driver does not contract this
 *                                         shape anyway; [precise] is untested
 *                                         on it (no hazard, no evidence)
 *
 * Eight expression shapes are probed, because "honoured" is not necessarily a
 * single yes/no: a driver may contract a mul-add but not a mul-sub, and
 * [precise] forbids reassociation as well as contraction. Shape S4 is an
 * explicit [fma()] call, which is a positive control in the other direction:
 * it MUST come back fused, on both columns, or the harness is not measuring
 * what it thinks it is.
 *
 * This test is deliberately NOT part of [runtest]: it is a measurement of a
 * third-party driver, not a property of this repository, and a driver that
 * starts contracting is a finding to record rather than a build to break. It
 * exits 0 on any outcome except a broken harness (see [main]).
 *
 * Run:
 *   dune exec sarek/tests/e2e/test_vulkan_no_contraction.exe
 * Dump the exact two shader strings for offline glslc/spirv-dis inspection:
 *   SAREK_NOCONTRACT_DUMP=/tmp/x dune exec \
 *     sarek/tests/e2e/test_vulkan_no_contraction.exe
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let () = Sarek_vulkan.Vulkan_plugin.init ()

(* ------------------------------------------------------------------ *)
(* binary32 reference arithmetic                                       *)
(* ------------------------------------------------------------------ *)

(* Round an OCaml binary64 to binary32, the same Int32 bit round-trip the
   interpreter's oracle uses (Sarek_float32.to_float32). *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Correctly-rounded fused multiply-add on binary32 operands. [Float.fma]
   correctly rounds the exact a*b+c to binary64; rounding that on to binary32 is
   benign double rounding, because binary64 carries 53 >= 2*24+2 bits of a
   binary32 operand. *)
let fma32 a b c = f32 (Float.fma a b c)

(* ------------------------------------------------------------------ *)
(* The shader                                                          *)
(* ------------------------------------------------------------------ *)

(* Buffer/push-constant layout matches Sarek's own GLSL calling convention
   (vector lengths first, then user scalars) so this goes through the ordinary
   [run_source] path with no special casing. *)
let shader_body =
  {|#version 450
@PRECISE@

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set=0, binding = 0) readonly  buffer BufA { float a[]; };
layout(std430, set=0, binding = 1) readonly  buffer BufB { float b[]; };
layout(std430, set=0, binding = 2) readonly  buffer BufC { float c[]; };
layout(std430, set=0, binding = 3) writeonly buffer BufO { float o[]; };

layout(push_constant) uniform PushConstants {
    int a_len; int b_len; int c_len; int o_len; int n;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(pc.n)) {
        P float av = a[i];
        P float bv = b[i];
        P float cv = c[i];

        /* S0: multiply into a named local, then add */
        P float q0 = av * bv;
        P float r0 = q0 + cv;

        /* S1: multiply then subtract, product on the LEFT  (q - c) */
        P float q1 = av * bv;
        P float r1 = q1 - cv;

        /* S2: multiply then subtract, product on the RIGHT (c - q) */
        P float q2 = av * bv;
        P float r2 = cv - q2;

        /* S3: multiply and add in ONE expression, no named product */
        P float r3 = (av * bv) + cv;

        /* S4: explicit fma() - positive control, MUST be fused */
        P float r4 = fma(av, bv, cv);

        /* S5: TwoProd error term - the df64 pattern that contraction kills */
        P float q5 = av * bv;
        P float r5 = fma(av, bv, -q5);

        /* S6: left-associated sum  ((a+b)+c) */
        P float s6 = av + bv;
        P float r6 = s6 + cv;

        /* S7: right-associated sum (a+(b+c)) */
        P float s7 = bv + cv;
        P float r7 = av + s7;

        /* S8: loop-carried accumulation `acc += a*b` - the matrix_mul inner
           loop, i.e. the shape `precise` was actually put in the codegen for.
           The trip count is a push constant, so the loop cannot be unrolled
           into a compile-time-constant straight line. */
        P float acc = cv;
        for (int k = 0; k < pc.n; ++k) {
            acc = acc + av * bv;
        }
        P float r8 = acc;

        /* S9, S10, S11: the MEASURED contracted value for S1, S2 and S8.
           docs/fp-contraction-policy.md establishes that RADV's fma is not
           correctly rounded, so predicting "what a contracted evaluation would
           return" from an exact-fma model can be wrong on exactly this driver
           -- and wrong in the dangerous direction, because a mispredicted
           target makes a genuinely contracted result fail to match and the
           shape reads as clean. Every contraction shape is therefore compared
           against a sibling shape that asks the DEVICE for the fused value of
           the same operands, not against an IEEE model. S4 plays that role for
           S0 and S3. */
        P float r9  = fma(av, bv, -cv);
        P float r10 = fma(-av, bv, cv);

        P float facc = cv;
        for (int k = 0; k < pc.n; ++k) {
            facc = fma(av, bv, facc);
        }
        P float r11 = facc;

        o[i * 12u +  0u] = r0;
        o[i * 12u +  1u] = r1;
        o[i * 12u +  2u] = r2;
        o[i * 12u +  3u] = r3;
        o[i * 12u +  4u] = r4;
        o[i * 12u +  5u] = r5;
        o[i * 12u +  6u] = r6;
        o[i * 12u +  7u] = r7;
        o[i * 12u +  8u] = r8;
        o[i * 12u +  9u] = r9;
        o[i * 12u + 10u] = r10;
        o[i * 12u + 11u] = r11;
    }
}
|}

(* The ONLY difference between the two compiled shaders: the expansion of [P].
   [#version] must be the first token in a GLSL translation unit, so the
   [#define] is substituted into the line after it rather than prepended. *)
let source ~precise =
  let marker = "@PRECISE@" in
  let def = if precise then "#define P precise" else "#define P" in
  let i =
    let n = String.length marker and h = String.length shader_body in
    let rec go k =
      if k + n > h then failwith "shader marker missing"
      else if String.sub shader_body k n = marker then k
      else go (k + 1)
    in
    go 0
  in
  String.sub shader_body 0 i ^ def
  ^ String.sub
      shader_body
      (i + String.length marker)
      (String.length shader_body - i - String.length marker)

(* ------------------------------------------------------------------ *)
(* Inputs                                                              *)
(* ------------------------------------------------------------------ *)

let p2 k = ldexp 1.0 k

(* Each triple is exactly representable in binary32. *)
let triples =
  [|
    (* T0: contraction probe. fl(fl(a*b)+c) = 0, fma(a,b,c) = 2^-24. *)
    ( "T0 contraction",
      f32 (1.0 +. p2 (-12)),
      f32 (1.0 +. p2 (-12)),
      f32 (-.(1.0 +. p2 (-11))) );
    (* T1: reassociation probe. ((a+b)+c) = 1, (a+(b+c)) = 1+2^-23. *)
    ("T1 reassociation", 1.0, p2 (-24), p2 (-24));
    (* T2: contraction probe, opposite sign of c, so the mul-sub shapes S1/S2
       are discriminating too. *)
    ( "T2 contraction'",
      f32 (1.0 +. p2 (-12)),
      f32 (1.0 +. p2 (-12)),
      f32 (1.0 +. p2 (-11)) );
  |]

let n_triples = Array.length triples

let n_shapes = 12

(* ------------------------------------------------------------------ *)
(* Per-shape reference values                                          *)
(* ------------------------------------------------------------------ *)

(* [sep] is what the interpreter/IEEE-as-written semantics give: every
   operation rounded. [alt] is what the shape degrades to if the compiler takes
   the liberty [precise] is supposed to forbid - contraction for S0..S5,
   reassociation for S6/S7. *)
type shape = {
  name : string;
  liberty : string;
  (* The value every operation-rounded-as-written evaluation must produce. *)
  sep : float -> float -> float -> float;
  (* What the shape degrades to if the compiler takes the liberty [precise]
     forbids, computed from an IEEE model. Used for PRINTING and as the
     fallback when no measured control exists. *)
  alt : float -> float -> float -> float;
  (* Index of the sibling shape that asks the DEVICE for the fused value of the
     same operands. When present this, not [alt], is the target a contracted
     result is recognised by -- see the S9/S10/S11 comment in the shader. *)
  control : int option;
}

let shapes =
  [|
    {
      name = "S0 q=a*b; q+c";
      control = Some 4;
      liberty = "contract";
      sep = (fun a b c -> f32 (f32 (a *. b) +. c));
      alt = (fun a b c -> fma32 a b c);
    };
    {
      name = "S1 q=a*b; q-c";
      control = Some 9;
      liberty = "contract";
      sep = (fun a b c -> f32 (f32 (a *. b) -. c));
      alt = (fun a b c -> fma32 a b (-.c));
    };
    {
      name = "S2 q=a*b; c-q";
      control = Some 10;
      liberty = "contract";
      sep = (fun a b c -> f32 (c -. f32 (a *. b)));
      alt = (fun a b c -> fma32 (-.a) b c);
    };
    {
      name = "S3 (a*b)+c inline";
      control = Some 4;
      liberty = "contract";
      sep = (fun a b c -> f32 (f32 (a *. b) +. c));
      alt = (fun a b c -> fma32 a b c);
    };
    (* Positive control, stated the other way round: [sep] here is the FUSED
       value, because an explicit fma() must fuse. If the device reports [alt]
       the harness is broken (or the driver's fma is not an fma at all). *)
    {
      name = "S4 fma(a,b,c) CTL";
      control = None;
      liberty = "must fuse";
      sep = (fun a b c -> fma32 a b c);
      alt = (fun a b c -> f32 (f32 (a *. b) +. c));
    };
    {
      name = "S5 fma(a,b,-a*b)";
      control = None;
      liberty = "contract";
      sep = (fun a b _ -> fma32 a b (-.f32 (a *. b)));
      alt = (fun _ _ _ -> 0.0);
    };
    {
      name = "S6 s=a+b; s+c";
      control = None;
      liberty = "reassoc";
      sep = (fun a b c -> f32 (f32 (a +. b) +. c));
      alt = (fun a b c -> f32 (a +. f32 (b +. c)));
    };
    {
      name = "S7 s=b+c; a+s";
      control = None;
      liberty = "reassoc";
      sep = (fun a b c -> f32 (a +. f32 (b +. c)));
      alt = (fun a b c -> f32 (f32 (a +. b) +. c));
    };
    (* The loop runs [n_triples] times, matching the shader's `k < pc.n`. *)
    {
      name = "S8 loop acc+=a*b";
      control = Some 11;
      liberty = "contract";
      sep =
        (fun a b c ->
          let acc = ref c in
          for _ = 1 to n_triples do
            acc := f32 (!acc +. f32 (a *. b))
          done ;
          !acc);
      alt =
        (fun a b c ->
          let acc = ref c in
          for _ = 1 to n_triples do
            acc := fma32 a b !acc
          done ;
          !acc);
    };
    (* S9/S10/S11 are the measured contracted targets for S1/S2/S8. Like S4
       they are explicit fma() calls, so [sep] is the FUSED value and they are
       integrity-checked: an explicit fma that does not fuse means the device
       is not giving us a fused value to compare against at all. *)
    {
      name = "S9 fma(a,b,-c) CTL";
      control = None;
      liberty = "must fuse";
      sep = (fun a b c -> fma32 a b (-.c));
      alt = (fun a b c -> f32 (f32 (a *. b) -. c));
    };
    {
      name = "S10 fma(-a,b,c) CTL";
      control = None;
      liberty = "must fuse";
      sep = (fun a b c -> fma32 (-.a) b c);
      alt = (fun a b c -> f32 (c -. f32 (a *. b)));
    };
    {
      name = "S11 loop fma CTL";
      control = None;
      liberty = "must fuse";
      sep =
        (fun a b c ->
          let acc = ref c in
          for _ = 1 to n_triples do
            acc := fma32 a b !acc
          done ;
          !acc);
      alt =
        (fun a b c ->
          let acc = ref c in
          for _ = 1 to n_triples do
            acc := f32 (!acc +. f32 (a *. b))
          done ;
          !acc);
    };
  |]

(* ------------------------------------------------------------------ *)
(* Execution                                                           *)
(* ------------------------------------------------------------------ *)

let run_on (dev : Device.t) ~precise =
  let a = Vector.create Vector.float32 n_triples in
  let b = Vector.create Vector.float32 n_triples in
  let c = Vector.create Vector.float32 n_triples in
  let o = Vector.create Vector.float32 (n_triples * n_shapes) in
  Array.iteri
    (fun i (_, av, bv, cv) ->
      Vector.set a i av ;
      Vector.set b i bv ;
      Vector.set c i cv)
    triples ;
  for i = 0 to (n_triples * n_shapes) - 1 do
    Vector.set o i nan
  done ;
  Sarek.Execute.run_source
    ~device:dev
    ~source:(source ~precise)
    ~lang:Sarek.Execute.GLSL_Source
    ~kernel_name:"main"
    ~block:(Sarek.Execute.dims1d 64)
    ~grid:(Sarek.Execute.dims1d 1)
    [
      Sarek.Execute.Vec a;
      Sarek.Execute.Vec b;
      Sarek.Execute.Vec c;
      Sarek.Execute.Vec o;
      Sarek.Execute.Int32 (Int32.of_int n_triples);
    ] ;
  Transfer.flush dev ;
  Vector.to_array o

(* ------------------------------------------------------------------ *)
(* Reporting                                                           *)
(* ------------------------------------------------------------------ *)

(* Position of a shape in [shapes], which is also its slot in the output
   buffer -- the shader writes r0..r11 to o[i*12 + 0..11] in this order. *)
let index_of sh =
  let r = ref (-1) in
  Array.iteri (fun i s -> if s.name = sh.name then r := i) shapes ;
  if !r < 0 then failwith ("unknown shape " ^ sh.name) ;
  !r

let () =
  if Array.length shapes <> n_shapes then
    failwith "shapes array and n_shapes disagree with the shader layout"

type verdict = Sep | Alt | Other | Undiscriminating

let eq_bits x y = Int32.bits_of_float x = Int32.bits_of_float y

(* The value a contracted (or reassociated) evaluation is recognised by. For
   every contraction shape this is a value the DEVICE produced for the same
   operands via an explicit fma(), never an IEEE model -- see the S9/S10/S11
   comment in the shader. Only the reassociation shapes, which involve no
   multiply and so no fma, fall back to the model. *)
let target_of sh (_, a, b, c) col ti =
  match sh.control with
  | Some ci -> col.((ti * n_shapes) + ci)
  | None -> sh.alt a b c

let classify sh tr col ti =
  let _, ta, tb, tc = tr in
  let v = col.((ti * n_shapes) + index_of sh) in
  let s = sh.sep ta tb tc in
  let t = target_of sh tr col ti in
  if eq_bits s t then Undiscriminating
  else if eq_bits v s then Sep
  else if eq_bits v t then Alt
  else Other

let verdict_str sh = function
  | Sep -> if sh.liberty = "must fuse" then "fused(ok)" else "as-written"
  | Alt -> (
      match sh.liberty with
      | "contract" -> "CONTRACTED"
      | "reassoc" -> "REASSOCIATED"
      | _ -> "NOT-FUSED")
  | Other -> "other"
  | Undiscriminating -> "-"

let broken = ref false

let report_device (dev : Device.t) =
  let name = dev.Device.name in
  Printf.printf "\n=== %s (framework %s) ===\n%!" name dev.Device.framework ;
  let plain = run_on dev ~precise:false in
  let prec = run_on dev ~precise:true in
  (* Anchor check. The classification never uses the IEEE fma model for a
     contraction shape, but whether the device AGREES with that model at these
     operands is worth stating, because it is the assumption the original
     design of this experiment would have rested on. *)
  Array.iteri
    (fun ti (tname, a, b, c) ->
      let dev_fma = plain.((ti * n_shapes) + 4) in
      let ieee_fma = fma32 a b c in
      Printf.printf
        "  fma anchor %-16s device %.17g  IEEE %.17g  %s\n"
        tname
        dev_fma
        ieee_fma
        (if eq_bits dev_fma ieee_fma then "agree"
         else "DISAGREE - model-based targets would be wrong here"))
    triples ;
  Printf.printf "\n" ;
  Printf.printf
    "  %-20s %-16s | %-12s %-12s | %-24s %s\n"
    "shape"
    "triple"
    "no-precise"
    "precise"
    "value(no-precise)"
    "value(precise)" ;
  Printf.printf "  %s\n" (String.make 108 '-') ;
  let plain_contracted = ref 0 and prec_contracted = ref 0 and total = ref 0 in
  Array.iter
    (fun sh ->
      Array.iteri
        (fun ti tr ->
          let tname, _, _, _ = tr in
          let idx = (ti * n_shapes) + index_of sh in
          let vp = plain.(idx) and vq = prec.(idx) in
          let cp = classify sh tr plain ti and cq = classify sh tr prec ti in
          if cp <> Undiscriminating || cq <> Undiscriminating then begin
            Printf.printf
              "  %-20s %-16s | %-12s %-12s | %-24.17g %.17g\n"
              sh.name
              tname
              (verdict_str sh cp)
              (verdict_str sh cq)
              vp
              vq ;
            if sh.liberty = "contract" then begin
              incr total ;
              if cp = Alt then incr plain_contracted ;
              if cq = Alt then incr prec_contracted
            end ;
            (* Harness integrity: an explicit fma() must fuse in both columns.
               If it does not, it is not supplying a fused value and every
               shape it is the control for is being compared to nothing. *)
            if sh.liberty = "must fuse" && (cp <> Sep || cq <> Sep) then begin
              Printf.printf
                "    !! CONTROL BROKEN: %s did not fuse on this device; the \
                 shapes it controls are not being measured\n"
                sh.name ;
              broken := true
            end ;
            if cp = Other || cq = Other then begin
              let _, a, b, c = tr in
              Printf.printf
                "    !! neither candidate matched (as-written %.17g, \
                 liberty-taken %.17g)\n"
                (sh.sep a b c)
                (target_of sh tr plain ti) ;
              broken := true
            end
          end)
        triples)
    shapes ;
  Printf.printf
    "\n\
    \  contraction summary on %s: no-precise %d/%d contracted, precise %d/%d \
     contracted\n"
    name
    !plain_contracted
    !total
    !prec_contracted
    !total ;
  Printf.printf
    "  -> %s\n%!"
    (match (!plain_contracted > 0, !prec_contracted > 0) with
    | true, false ->
        "NoContraction is HONOURED (precise suppressed a real, observed \
         contraction)"
    | true, true ->
        "NoContraction is IGNORED (the driver contracted despite the \
         decoration)"
    | false, false ->
        "INCONCLUSIVE for contraction: this driver did not contract these \
         shapes even when free to, so `precise` had nothing to suppress"
    | false, true ->
        "ANOMALY: `precise` introduced contraction that was absent without it")

let () =
  (match Sys.getenv_opt "SAREK_NOCONTRACT_DUMP" with
  | None -> ()
  | Some prefix ->
      List.iter
        (fun (suffix, precise) ->
          let path = prefix ^ suffix ^ ".comp" in
          let oc = open_out path in
          output_string oc (source ~precise) ;
          close_out oc ;
          Printf.printf "wrote %s\n%!" path)
        [("_plain", false); ("_precise", true)]) ;
  let devs = Device.by_framework "Vulkan" in
  if Array.length devs = 0 then
    print_endline "[SKIP] no Vulkan device available"
  else begin
    print_endline
      "=== SPIR-V NoContraction: is it honoured at execution? (issue #126) ===" ;
    Printf.printf
      "discriminator: a=b=1+2^-12, c=-(1+2^-11) -> as-written 0, contracted \
       2^-24 = %.17g\n\
       %!"
      (ldexp 1.0 (-24)) ;
    Array.iter report_device devs ;
    if !broken then begin
      print_endline
        "\n[FAIL] harness integrity check failed - see the !! lines above" ;
      exit 1
    end ;
    print_endline "\n[DONE] measurement complete"
  end
