(* throwaway probe - not committed *)
open Sarek_ir_types

let () = Sarek_stdlib_meta.force_init ()

let make_var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let k =
  let a = make_var "a" (TVec TFloat32) in
  let b = make_var "b" (TVec TFloat32) in
  let i = make_var "i" TInt32 in
  let body =
    SLet
      ( i,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("b", EVar i),
            EIntrinsic (["Float32"], "sin", [EArrayRead ("a", EVar i)]) ) )
  in
  {
    default_kernel with
    kern_name = "probe";
    kern_params =
      [
        DParam (a, Some {arr_elttype = TFloat32; arr_memspace = Global});
        DParam (b, Some {arr_elttype = TFloat32; arr_memspace = Global});
      ];
    kern_body = body;
  }

let src =
  "fun (a : float32 vector) (b : float32 vector) ->\n\
  \  let i = global_thread_id in\n\
  \  b.(i) <- Float32.sin a.(i)"

let () =
  let before = Sarek_codegen.Sarek_ir_cuda.generate k in
  Printf.printf "=== CUDA before (fresh process) ===\n%s\n" before ;
  (match Sarek_transpile.of_source Sarek_transpile.OpenCL src with
  | Ok _ -> print_endline "[transpile OpenCL: Ok]"
  | Error e ->
      print_endline ("[transpile OpenCL: " ^ Sarek_transpile.string_of_error e ^ "]")) ;
  let after = Sarek_codegen.Sarek_ir_cuda.generate k in
  Printf.printf "=== CUDA after transpiling an unrelated kernel to OpenCL ===\n%s\n" after ;
  Printf.printf "IDENTICAL=%b\n" (String.equal before after)
