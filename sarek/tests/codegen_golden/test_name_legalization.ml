(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Names the target language cannot take as written (Sarek_ir_names).
 *
 * Two defects of valid OCaml kernels, found by a downstream user and recorded in the
 * 2026-10-17 review of the SPOC updates:
 *
 * 1. Name shadowing. [let s = s +. 1.0 in ...] (a parameter rebound) and two
 *    [let t] in one block were emitted as two C declarations of one name: a
 *    redefinition on CUDA/HIP and OpenCL, and on Metal, GLSL and WGSL.
 * 2. Reserved words. A parameter named [half] broke Metal ([half] is a type);
 *    [local], [kernel], [uint], [float4], [max], [x'] break or hijack the other
 *    C-family targets in the same way.
 *
 * Every case below runs on every source backend (CUDA, which HIP reuses,
 * OpenCL, Metal, GLSL, WGSL), without a device. Where a real compiler is
 * available (clang -x cl, the Metal runtime compiler) the generated source is
 * also compiled, and the pre-fix spelling is shown to be REJECTED by the same
 * compiler, so a pass is not vacuous.
 ******************************************************************************)

open Sarek_ir_types
module Codegen = Sarek_codegen

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  m = 0 || go 0

let count s sub =
  let n = String.length s and m = String.length sub in
  let rec go i acc =
    if i + m > n then acc
    else if String.sub s i m = sub then go (i + m) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

(* name, id, type *)
let var ?(mutable_ = false) name id ty =
  {var_name = name; var_id = id; var_type = ty; var_mutable = mutable_}

let vec_param name id =
  DParam
    ( var name id (TVec TFloat32),
      Some {arr_elttype = TFloat32; arr_memspace = Global} )

let scalar_param name id = DParam (var name id TFloat32, None)

let n_param = DParam (var "n" 99 TInt32, None)

let f x = EConst (CFloat32 x)

let backends : (string * (kernel -> string)) list =
  [
    ("CUDA", Codegen.Sarek_ir_cuda.generate);
    ("OpenCL", Codegen.Sarek_ir_opencl.generate);
    ("Metal", Codegen.Sarek_ir_metal.generate);
    ("GLSL", fun k -> Codegen.Sarek_ir_glsl.generate k);
    ("WGSL", fun k -> Codegen.Sarek_ir_wgsl.generate k);
  ]

let c_family = ["CUDA"; "OpenCL"; "Metal"]

(* tid = global id; <body tid>, guarded by tid < n *)
let guarded body_of_tid =
  let tid = var "tid" 1 TInt32 in
  SLet
    ( tid,
      EIntrinsic ([], "global_thread_id", []),
      SIf
        (EBinop (Lt, EVar tid, EVar (var "n" 99 TInt32)), body_of_tid tid, None)
    )

(******************************************************************************
 * 1. Shadowing
 ******************************************************************************)

(* a.(tid) <- (let s = s + 1 in let t = s in let t = t + 2 in t), where the
   outer s is a parameter. *)
let shadow_kernel =
  let s_in = var "s" 2 TFloat32 and s_out = var "s" 10 TFloat32 in
  let t1 = var "t" 11 TFloat32 and t2 = var "t" 12 TFloat32 in
  {
    default_kernel with
    kern_name = "shadow";
    kern_params = [vec_param "a" 0; scalar_param "s" 2; n_param];
    kern_body =
      SLet
        ( s_out,
          EBinop (Add, EVar s_in, f 1.0),
          guarded (fun tid ->
              SLet
                ( t1,
                  EVar s_out,
                  SLet
                    ( t2,
                      EBinop (Add, EVar t1, f 2.0),
                      SAssign (LArrayElem ("a", EVar tid), EVar t2) ) )) );
  }

(* The body-level redeclaration is the part GLSL and WGSL used to leave alone
   (their own pass only renames a local that hits a scalar PARAMETER). *)
let test_same_block_redeclaration () =
  List.iter
    (fun (name, gen) ->
      let src = gen shadow_kernel in
      let declarations =
        count src "float t " + count src "float t=" + count src "let t :"
      in
      Alcotest.(check bool)
        (Printf.sprintf "%s declares t at most once" name)
        true
        (declarations <= 1) ;
      Alcotest.(check bool)
        (Printf.sprintf "%s renames the second t" name)
        true
        (contains src "t_1"))
    backends

(* C: [float s = (s + 1.0f)] redeclares the parameter and reads itself. *)
let test_parameter_rebound () =
  List.iter
    (fun name ->
      let src = (List.assoc name backends) shadow_kernel in
      Alcotest.(check bool)
        (name ^ " does not redeclare the parameter s")
        false
        (contains src "float s = (s") ;
      Alcotest.(check bool)
        (name ^ " reads the parameter in the initialiser")
        true
        (contains src "float s_1 = (s + 1.0f)"))
    c_family

(* Siblings of a statement list share one target block: [SLet] has no braces, so
   the first [t] below is still in scope when the second is declared. *)
let test_siblings_share_a_block () =
  let tid = var "tid" 1 TInt32 in
  let assign t = SAssign (LArrayElem ("a", EVar tid), EVar t) in
  let t1 = var "t" 11 TFloat32 and t2 = var "t" 12 TFloat32 in
  let k =
    {
      default_kernel with
      kern_name = "siblings_in_block";
      kern_params = [vec_param "a" 0; n_param];
      kern_body =
        SLet
          ( tid,
            EIntrinsic ([], "global_thread_id", []),
            SSeq [SLet (t1, f 1.0, assign t1); SLet (t2, f 2.0, assign t2)] );
    }
  in
  List.iter
    (fun (name, gen) ->
      let src = gen k in
      Alcotest.(check bool)
        (name ^ " renames the second t")
        true
        (contains src "t_1"))
    backends

(* Disjoint scopes are not a shadowing: nothing is renamed. *)
let test_sibling_scopes_keep_names () =
  let tid_of = var "tid" 1 TInt32 in
  let branch id =
    SLet
      ( var "t" id TFloat32,
        f 1.0,
        SAssign (LArrayElem ("a", EVar tid_of), EVar (var "t" id TFloat32)) )
  in
  let k =
    {
      default_kernel with
      kern_name = "siblings";
      kern_params = [vec_param "a" 0; n_param];
      kern_body =
        SLet
          ( tid_of,
            EIntrinsic ([], "global_thread_id", []),
            SIf (EBinop (Lt, EVar tid_of, f 0.0), branch 5, Some (branch 6)) );
    }
  in
  Alcotest.(check bool)
    "returned physically unchanged"
    true
    (Sarek_ir_names.legalize
       ~reserved:Sarek_reserved_words.cuda
       ~rename_params:true
       k
    == k)

(******************************************************************************
 * 2. Reserved words, and names that are not identifiers
 ******************************************************************************)

(* Words an OCaml kernel can use as a variable that a target rejects or that
   hide one of its functions; each is tried as a scalar parameter and as a
   local. *)
let words_per_target =
  [
    ( "CUDA",
      [
        "half";
        "int";
        "float";
        "double";
        "register";
        "auto";
        "new";
        "template";
        "this";
        "max";
        "min";
        "sqrt";
        "exp";
        "uint";
        "float4";
        "threadIdx";
        "blockDim";
        "dim3";
        "x'";
        "2x";
      ] );
    ( "OpenCL",
      [
        "half";
        "kernel";
        "global";
        "local";
        "constant";
        "private";
        "generic";
        "uint";
        "float4";
        "uchar";
        "image2d_t";
        "clamp";
        "select";
        "get_global_id";
        "convert_int";
        "as_float";
        "half_sqrt";
        "max";
        "x'";
      ] );
    ( "Metal",
      [
        "half";
        "kernel";
        "device";
        "thread";
        "threadgroup";
        "constant";
        "uint";
        "float4";
        "half2x2";
        "simd_sum";
        "simdgroup_matrix";
        "metal";
        "sampler";
        "max";
        "x'";
      ] );
  ]

let reserved_kernel word =
  let w_in = var word 2 TFloat32 in
  {
    default_kernel with
    kern_name = "reserved";
    kern_params = [vec_param "a" 0; scalar_param word 2; n_param];
    kern_body =
      guarded (fun tid ->
          SLet
            ( var (word ^ "_loc") 19 TFloat32,
              EVar w_in,
              SAssign
                ( LArrayElem ("a", EVar tid),
                  EBinop (Add, EVar (var (word ^ "_loc") 19 TFloat32), f 0.0) )
            ));
  }

(* The renamed IR binds no reserved name and still binds the parameter. *)
let test_no_reserved_binder () =
  List.iter
    (fun (target, words) ->
      let reserved =
        match target with
        | "CUDA" -> Sarek_reserved_words.cuda
        | "OpenCL" -> Sarek_reserved_words.opencl
        | _ -> Sarek_reserved_words.metal
      in
      List.iter
        (fun word ->
          Alcotest.(check bool)
            (Printf.sprintf "%s: %S is reserved" target word)
            true
            (reserved word) ;
          let k =
            Sarek_ir_names.legalize
              ~reserved
              ~rename_params:true
              (reserved_kernel word)
          in
          List.iter
            (function
              | DParam (v, _) ->
                  Alcotest.(check bool)
                    (Printf.sprintf "%s: parameter %S renamed" target word)
                    false
                    (reserved v.var_name)
              | _ -> ())
            k.kern_params)
        words)
    words_per_target

let single_param_kernel word =
  {
    default_kernel with
    kern_name = "reserved";
    kern_params = [vec_param "a" 0; scalar_param word 2; n_param];
    kern_body =
      guarded (fun tid ->
          SAssign (LArrayElem ("a", EVar tid), EVar (var word 2 TFloat32)));
  }

(* The user's complaint: [half] as a kernel parameter. Per backend, the source
   no longer declares it. *)
let test_half_parameter () =
  let k = single_param_kernel "half" in
  List.iter
    (fun (name, gen) ->
      let src = gen k in
      let declared =
        contains src "float half," || contains src "float half)"
        || contains src "&half " || contains src "half :"
        || contains src "float half "
      in
      match name with
      | "GLSL" | "WGSL" ->
          (* These keep their own escapers: GLSL [halfv]; WGSL accepts [half]. *)
          Alcotest.(check bool) (name ^ " generates") true (src <> "")
      | _ ->
          Alcotest.(check bool) (name ^ " does not declare half") false declared)
    backends

(* [x'] is not an identifier anywhere. *)
let test_prime_everywhere () =
  let k =
    {
      default_kernel with
      kern_name = "prime";
      kern_params = [vec_param "a" 0; n_param];
      kern_body =
        guarded (fun tid ->
            let x' = var "x'" 7 TFloat32 in
            SLet (x', f 1.0, SAssign (LArrayElem ("a", EVar tid), EVar x')));
    }
  in
  List.iter
    (fun (name, gen) ->
      Alcotest.(check bool)
        (name ^ " emits no quote in an identifier")
        false
        (contains (gen k) "x'"))
    backends

(* An array parameter named like a reserved word: references follow. *)
let test_array_references_follow () =
  let k =
    {
      default_kernel with
      kern_name = "arr";
      kern_params = [vec_param "local" 0; n_param];
      kern_body =
        guarded (fun tid ->
            SAssign
              (LArrayElem ("local", EVar tid), EArrayRead ("local", EVar tid)));
    }
  in
  let src = Codegen.Sarek_ir_opencl.generate k in
  Alcotest.(check bool)
    "OpenCL: no [local] identifier survives"
    false
    (contains src "restrict local," || contains src " local[tid]") ;
  Alcotest.(check bool)
    "OpenCL: the access uses the renamed array"
    true
    (contains src "local_1[tid] = local_1[tid]")

(* A name that does not clash is left alone, and the pass is the identity. *)
let test_unreserved_untouched () =
  let k = single_param_kernel "scale" in
  List.iter
    (fun (target, reserved) ->
      Alcotest.(check bool)
        (target ^ ": physically unchanged")
        true
        (Sarek_ir_names.legalize ~reserved ~rename_params:true k == k))
    [
      ("CUDA", Sarek_reserved_words.cuda);
      ("OpenCL", Sarek_reserved_words.opencl);
      ("Metal", Sarek_reserved_words.metal);
    ] ;
  Alcotest.(check bool)
    "shader pass: physically unchanged"
    true
    (Sarek_ir_names.legalize_shader k == k)

(* A helper function named like a reserved word is renamed with its calls. *)
let test_helper_named_reserved () =
  let x = var "x" 30 TFloat32 in
  let helper =
    {
      hf_name = "half";
      hf_params = [x];
      hf_ret_type = TFloat32;
      hf_body = SReturn (EBinop (Add, EVar x, f 1.0));
    }
  in
  let k =
    {
      default_kernel with
      kern_name = "helper";
      kern_params = [vec_param "a" 0; n_param];
      kern_funcs = [helper];
      kern_body =
        guarded (fun tid ->
            SAssign
              ( LArrayElem ("a", EVar tid),
                EApp (EVar (var "half" 0 TFloat32), [f 1.0]) ));
    }
  in
  let src = Codegen.Sarek_ir_opencl.generate k in
  Alcotest.(check bool) "OpenCL: helper renamed" true (contains src "half_1(") ;
  Alcotest.(check bool) "OpenCL: no call to half" false (contains src " half(")

(* A kernel with opaque native source cannot be renamed soundly: untouched. *)
let test_native_kernel_untouched () =
  let k =
    {
      (single_param_kernel "half") with
      kern_body =
        SSeq
          [
            SNative
              {
                gpu = (fun ~framework:_ -> "");
                ocaml = {run = (fun ~block:_ ~grid:_ _ -> ())};
              };
          ];
    }
  in
  Alcotest.(check bool)
    "unchanged"
    true
    (Sarek_ir_names.legalize
       ~reserved:Sarek_reserved_words.metal
       ~rename_params:true
       k
    == k)

(******************************************************************************
 * 3. Real compilers, when present
 ******************************************************************************)

let skip_with reason =
  Printf.printf "[SKIP] %s\n%!" reason ;
  Alcotest.skip ()

(* The pre-fix source of the [half] parameter, as the generators emitted it. *)
let legacy_opencl_half =
  "__kernel void k(__global float* restrict a, float half, int n) { a[0] = \
   half; }\n"

let test_opencl_clang () =
  if not (Opencl_gate.Opencl_clang.available ()) then
    skip_with (Opencl_gate.Opencl_clang.why_unavailable ())
  else begin
    (match Opencl_gate.Opencl_clang.run_clang legacy_opencl_half with
    | Ok () -> Alcotest.fail "clang accepted the pre-fix [float half] parameter"
    | Error _ -> ()) ;
    List.iter
      (fun (label, k) ->
        match
          Opencl_gate.Opencl_clang.run_clang
            (Codegen.Sarek_ir_opencl.generate k)
        with
        | Ok () -> ()
        | Error e -> Alcotest.failf "%s rejected by clang:\n%s" label e)
      ([("shadow", shadow_kernel)]
      @ List.map
          (fun w -> ("reserved " ^ w, single_param_kernel w))
          (List.filter
             (fun w -> w <> "image2d_t" && w <> "2x")
             (List.assoc "OpenCL" words_per_target)))
  end

let legacy_metal_half =
  "#include <metal_stdlib>\n\
   using namespace metal;\n\
   kernel void k(device float* a [[buffer(0)]], constant float &half \
   [[buffer(1)]], uint3 gid [[thread_position_in_grid]]) { a[gid.x] = half; }\n"

let test_metal_compiler () =
  if not (Metal_gate.Metal_compile.available ()) then
    skip_with (Metal_gate.Metal_compile.why_unavailable ())
  else begin
    (match Metal_gate.Metal_compile.run_metal legacy_metal_half with
    | Ok () -> Alcotest.fail "Metal accepted the pre-fix [half] parameter"
    | Error _ -> ()) ;
    List.iter
      (fun (label, k) ->
        match
          Metal_gate.Metal_compile.run_metal (Codegen.Sarek_ir_metal.generate k)
        with
        | Ok () -> ()
        | Error e -> Alcotest.failf "%s rejected by Metal:\n%s" label e)
      ([("shadow", shadow_kernel)]
      @ List.map
          (fun w -> ("reserved " ^ w, single_param_kernel w))
          (List.assoc "Metal" words_per_target))
  end

let () =
  Alcotest.run
    "name legalization"
    [
      ( "shadowing",
        [
          Alcotest.test_case
            "same-block redeclaration, every backend"
            `Quick
            test_same_block_redeclaration;
          Alcotest.test_case
            "parameter rebound, C family"
            `Quick
            test_parameter_rebound;
          Alcotest.test_case
            "siblings of a statement list share a block"
            `Quick
            test_siblings_share_a_block;
          Alcotest.test_case
            "sibling scopes keep their names"
            `Quick
            test_sibling_scopes_keep_names;
        ] );
      ( "reserved",
        [
          Alcotest.test_case
            "no reserved binder survives"
            `Quick
            test_no_reserved_binder;
          Alcotest.test_case "half parameter" `Quick test_half_parameter;
          Alcotest.test_case "prime, every backend" `Quick test_prime_everywhere;
          Alcotest.test_case
            "array references follow"
            `Quick
            test_array_references_follow;
          Alcotest.test_case
            "helper named like a reserved word"
            `Quick
            test_helper_named_reserved;
        ] );
      ( "identity",
        [
          Alcotest.test_case
            "unreserved names untouched"
            `Quick
            test_unreserved_untouched;
          Alcotest.test_case
            "native kernels untouched"
            `Quick
            test_native_kernel_untouched;
        ] );
      ( "compilers",
        [
          Alcotest.test_case "clang -x cl" `Quick test_opencl_clang;
          Alcotest.test_case "Metal runtime compiler" `Quick test_metal_compiler;
        ] );
    ]
