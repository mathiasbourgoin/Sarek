(******************************************************************************)
(* coverage_probe.ml — eq-class probe instrument
 *
 * Records a (mode, is_varying_cond, has_barriers, depth) eq-class tuple
 * for each conformance test input, allowing domain-coverage saturation
 * tracking.
 *
 * Usage: call probe_record on every generated expr before checking it.
 * Call probe_report at the end of a test run.
 ******************************************************************************)

type exec_mode = Converged | Diverged

type eq_class = {
  mode : exec_mode;
  varying_root : bool; (* is_varying on the root expression *)
  has_barrier : bool; (* expression contains EBarrier *)
  has_diverging : bool; (* has_diverging_cf on the root *)
  depth : int; (* estimated AST depth bucket: 0=leaf, 1=shallow, 2=deep *)
}

let table : (eq_class, int) Hashtbl.t = Hashtbl.create 64

let probe_record cls =
  let n = try Hashtbl.find table cls with Not_found -> 0 in
  Hashtbl.replace table cls (n + 1)

let probe_report () =
  Printf.printf
    "\n=== coverage_probe: %d eq-classes observed ===\n"
    (Hashtbl.length table) ;
  Hashtbl.iter
    (fun cls count ->
      Printf.printf
        "  mode=%-10s var=%-5b bar=%-5b dcf=%-5b depth=%d  count=%d\n"
        (match cls.mode with
        | Converged -> "Converged"
        | Diverged -> "Diverged")
        cls.varying_root
        cls.has_barrier
        cls.has_diverging
        cls.depth
        count)
    table ;
  Printf.printf "=== end coverage_probe ===\n%!"
