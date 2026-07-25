(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek Real64 - portable "~double precision on every device"
 *
 * [real64] is a user-facing abstraction meaning "give me roughly double
 * precision, whatever the device can do". It selects, PER DEVICE at launch
 * time, between two concrete substrates that already ship in Sarek:
 *
 *   - Native float64 (sarek.float64) on devices that report fp64 support
 *     (Device.allows_fp64 = true): the real IEEE-754 binary64 path - native
 *     `double` locals, native `+. -. *. /.`, `Float64.sqrt`, etc.
 *
 *   - df64 double-float fallback (sarek.df64) on devices WITHOUT native fp64:
 *     a pair of float32 carrying ~2x float32 precision, implemented in pure
 *     Sarek and therefore runnable on every backend.
 *
 * The selection is done here, on the host, from the device capability flag;
 * the two substrates are materialised as two separately-lowered kernels and
 * the matching one is picked at launch (see {!substrate_for} / {!select}).
 *
 * HOST REPRESENTATION
 *   On the host a real64 value is just an OCaml [float] (binary64). Vectors
 *   are created through {!create_vector}, which picks the device-appropriate
 *   storage (a plain float64 vector for the native path, a df64 struct vector
 *   for the fallback) and exposes a uniform double-precision get/set so caller
 *   code never sees the representation.
 *
 * PRECISION CONTRACT (honest, per substrate)
 *   - Native f64 path  : full IEEE-754 binary64 (~2^-52 relative), i.e.
 *     at-least-f64-comparable precision. This is the existing, fully-working
 *     f64 kernel path (writable f64 literals via the `G` suffix since PR #240).
 *   - df64 fallback    : ~2^-47..2^-46 relative on a backend AND DEVICE whose
 *     float32 ops are IEEE-exact, whose fma is correctly rounded, and whose
 *     compiler does not contract a multiply into a neighbouring add -
 *     "significantly better than f32, near-f64". Exponent range is that of
 *     float32, NOT binary64; results are NOT bit-exact with binary64; and
 *     overflow yields NaN rather than a signed infinity.
 *
 *     DO NOT name backends here. Precision is a property of the backend
 *     compiler AND the device, so "OpenCL" or "CUDA/PTX" alone is not a
 *     meaningful qualifier - an earlier version of this line listed
 *     "(OpenCL, CUDA/PTX, Interpreter)" on the strength of measurements
 *     taken only on an AMD box, and a total collapse to ~2^-24 on real
 *     NVIDIA hardware went unnoticed for four years as a result.
 *     Sarek_df64's PER-BACKEND STATUS block is the single source of truth
 *     and names the hardware and toolchain behind every figure; consult it
 *     rather than restating it, and note it also records a still-unresolved
 *     sqrt residual on NVIDIA. On Native the error-free transformations
 *     collapse to plain f32 storage precision (~2^-24). These are the df64
 *     contract's known deviations, inherited verbatim.
 *
 * EXPOSED OPERATION SET
 *   The op set is the INTERSECTION of what both substrates provide:
 *   of_float / to_float, add / sub / mul / div, and sqrt. df64 has no
 *   transcendentals (sin/cos/exp/log/...), so real64 does NOT expose them -
 *   promising them would be a lie on the fallback path.
 *
 * AUTHORING KERNELS (palier A - see the deviation note below)
 *   A single [%kernel] produces ONE lowered IR with ONE concrete element
 *   type, so a real64 kernel is authored as a PAIR: a native-f64 body (over
 *   `float64 vector`, using Float64 ops) and a df64 body (over
 *   `Sarek_df64.df64 vector`, using the df64 ops pulled in with
 *   %sarek_include). Both are built once; {!select} picks the IR to launch
 *   from the device's substrate. The host plumbing here (vector
 *   materialisation, dispatch, readback) is what makes the two bodies feel
 *   like one abstraction to the caller. See sarek/tests/e2e/test_real64.ml
 *   for the canonical usage.
 *
 * AUTHORING KERNELS (palier B - single source)
 *   [%kernel.real64 ...] removes the hand-written pair: author the compute
 *   ONCE over an abstract `real64 vector` element type with the intersection
 *   op set (+. -. *. /. and sqrt), and the PPX expands the SAME AST twice into
 *   the (native, fallback) pair above. {!kernel_ir} picks the IR matching a
 *   device's substrate. Transcendentals are rejected at expansion (df64 has
 *   none). See sarek/tests/e2e/test_real64_single_source.ml.
 ******************************************************************************)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

(** A real64 value on the host is a binary64 float. *)
type real64 = float

(** Identity host conversions - the host always carries full binary64. *)
let of_float (x : float) : real64 = x

let to_float (x : real64) : float = x

(** {1 Host reference arithmetic}

    Plain binary64 operations on real64 values, for building references and
    doing host-side combination of results. These are NOT the device ops (those
    live in the two substrates); they are the "what a correct answer looks like"
    oracle. *)

let add (a : real64) (b : real64) : real64 = a +. b

let sub (a : real64) (b : real64) : real64 = a -. b

let mul (a : real64) (b : real64) : real64 = a *. b

let div (a : real64) (b : real64) : real64 = a /. b

let sqrt (a : real64) : real64 = Stdlib.sqrt a

(** {1 Substrate selection} *)

(** The two concrete lowerings [real64] can take on a device. *)
type substrate =
  | Native_f64  (** native IEEE-754 binary64 (device has fp64) *)
  | Fallback_df64  (** double-float emulation (device lacks fp64) *)

let string_of_substrate = function
  | Native_f64 -> "native-f64"
  | Fallback_df64 -> "fallback-df64"

(** Substrate a device would use by default: native f64 iff it reports fp64
    support, df64 otherwise. Pass [~force] to override (used by tests to run the
    df64 fallback even on fp64-capable hardware, so both lowering paths are
    exercised everywhere). *)
let substrate_for ?force (dev : Device.t) : substrate =
  match force with
  | Some s -> s
  | None -> if Device.allows_fp64 dev then Native_f64 else Fallback_df64

(** Pick one of two per-substrate values (typically the two lowered kernel IRs,
    but works for anything). *)
let select (s : substrate) ~(native : 'a) ~(fallback : 'a) : 'a =
  match s with Native_f64 -> native | Fallback_df64 -> fallback

(** {1 Single-source kernels (palier B)}

    A [%kernel.real64 ...] kernel is authored ONCE over an abstract
    [real64 vector] element type and expands to the pair [(native, fallback)] -
    the same two lowered [%kernel] values palier A authored by hand. Each
    element is a [(closure, kirc)] pair; {!kernel_ir} picks the one matching a
    device's substrate, ready to hand to [Sarek.Execute.run_vectors]. *)

(** Extract the lowered IR from one lowered kernel value. *)
let ir_of_kernel (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "Sarek_real64: kernel has no lowered IR"

(** Pick the IR variant matching [substrate] from the [(native, fallback)] pair
    produced by [%kernel.real64]. *)
let kernel_ir (substrate : substrate) (native_k, fallback_k) =
  select
    substrate
    ~native:(ir_of_kernel native_k)
    ~fallback:(ir_of_kernel fallback_k)

(** {1 Uniform host vectors}

    A [real64_vector] hides whether the underlying storage is a float64 vector
    (native path) or a df64 struct vector (fallback). Callers always read/write
    plain doubles; encoding/decoding to the df64 pair is done here. [arg] is the
    ready-to-pass kernel argument (the same physical vector run_vectors
    transfers to/from the device). *)
type real64_vector = {
  arg : Sarek.Execute.vector_arg;
  set : int -> float -> unit;
  get : int -> float;
  length : int;
}

(** Create an [n]-element real64 vector in the storage matching [substrate]. *)
let create_vector (substrate : substrate) (n : int) : real64_vector =
  match substrate with
  | Native_f64 ->
      let v = Vector.create Vector.float64 n in
      {
        arg = Sarek.Execute.Vec v;
        set = (fun i x -> Vector.set v i x);
        get = (fun i -> Vector.get v i);
        length = n;
      }
  | Fallback_df64 ->
      let v = Vector.create_custom Sarek_df64.df64_custom n in
      {
        arg = Sarek.Execute.Vec v;
        set = (fun i x -> Vector.set v i (Sarek_df64.Host.encode x));
        get = (fun i -> Sarek_df64.Host.decode (Vector.get v i));
        length = n;
      }

let vset (rv : real64_vector) (i : int) (x : float) : unit = rv.set i x

let vget (rv : real64_vector) (i : int) : float = rv.get i

let vlength (rv : real64_vector) : int = rv.length

let arg_of (rv : real64_vector) : Sarek.Execute.vector_arg = rv.arg

(** {1 Substrate re-exports for kernel authors}

    The fallback kernel body pulls df64 ops in with
    [let%sarek_include _ = ".../Sarek_df64.ml"] and [let open Sarek_df64 in];
    the native body uses [Float64]. Re-exported here for discoverability. *)

module Df64 = Sarek_df64
module Float64 = Sarek_float64.Float64
