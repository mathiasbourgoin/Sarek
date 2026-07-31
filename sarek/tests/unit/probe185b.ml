(* throwaway probe - multi-domain codegen race - not committed *)
open Sarek_ir_types

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

(* Two kernels with DIFFERENT variant tables, so a leak between two concurrent
   generations shows up as a wrong constructor payload type. *)
let kernel_with_variant name variants =
  let out = make_var "out" (TVec TInt32) in
  let i = make_var "i" TInt32 in
  let body =
    SLet
      ( i,
        EIntrinsic ([], "global_thread_id", []),
        SMatch
          ( EVariant ("t", "A", [EConst (CInt32 1l)]),
            [
              (PConstr ("A", ["x"]), SAssign (LArrayElem ("out", EVar i), EVar (make_var "x" TInt32)));
              (PWild, SEmpty);
            ] ) )
  in
  {
    default_kernel with
    kern_name = name;
    kern_params = [DParam (out, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_variants = variants;
    kern_body = body;
  }

let k1 = kernel_with_variant "k1" [("t", [("A", [TInt32]); ("B", [])])]

let k2 = kernel_with_variant "k2" [("t", [("A", [TFloat32]); ("B", [])])]

let gen k = Sarek_codegen.Sarek_ir_opencl.generate k

let () =
  let ref1 = gen k1 and ref2 = gen k2 in
  let iters = 2000 in
  let bad = Atomic.make 0 in
  let run k r () =
    for _ = 1 to iters do
      if not (String.equal (gen k) r) then ignore (Atomic.fetch_and_add bad 1)
    done
  in
  let d1 = Domain.spawn (run k1 ref1) in
  let d2 = Domain.spawn (run k2 ref2) in
  Domain.join d1 ; Domain.join d2 ;
  Printf.printf "MISMATCHES=%d / %d\n" (Atomic.get bad) (2 * iters)
