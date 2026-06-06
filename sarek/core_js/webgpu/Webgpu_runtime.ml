(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** OCaml/js_of_ocaml WebGPU compute runtime. See {!Webgpu_runtime.mli} for full
    API documentation. *)

open Js_of_ocaml

(* jsoo 6.x: Js.to_float/Js.float exist; Js.to_int/Js.of_int/Js.of_float do not. *)
let js_to_int x = int_of_float (Js.to_float x)

let js_of_int n = Js.float (float_of_int n)

let js_of_float v = Js.float v

let then_ p f =
  ignore
    (Js.Unsafe.meth_call p "then" [|Js.Unsafe.inject (Js.wrap_callback f)|])

let catch_ p f =
  ignore
    (Js.Unsafe.meth_call p "catch" [|Js.Unsafe.inject (Js.wrap_callback f)|])

let js_global name = Js.Unsafe.get Js.Unsafe.global (Js.string name)

let gpu =
  js_global "navigator" |> fun nav -> Js.Unsafe.get nav (Js.string "gpu")

let json_parse s =
  Js.Unsafe.fun_call
    (Js.Unsafe.get (js_global "JSON") (Js.string "parse"))
    [|Js.Unsafe.inject (Js.string s)|]

let js_get obj key = Js.Unsafe.get obj (Js.string key)

let js_int obj key = js_to_int (js_get obj key)

let js_str obj key = Js.to_string (js_get obj key)

let js_arr_len arr = js_to_int (Js.Unsafe.get arr (Js.string "length"))

let js_arr_get arr i = Js.Unsafe.get arr (js_of_int i)

let js_is_null_or_undef v =
  Js.Optdef.case
    (Js.Optdef.return v)
    (fun () -> true)
    (fun v' -> Js.Opt.case (Js.Opt.return v') (fun () -> true) (fun _ -> false))

let make_ta ctor (data : float array) =
  let arr =
    Js.Unsafe.new_obj ctor [|Js.Unsafe.inject (js_of_int (Array.length data))|]
  in
  Array.iteri
    (fun i v ->
      Js.Unsafe.set arr (js_of_int i) (Js.Unsafe.inject (js_of_float v)))
    data ;
  arr

let typed_array et data =
  match et with
  | "f32" -> make_ta (js_global "Float32Array") data
  | "i32" -> make_ta (js_global "Int32Array") data
  | "u32" -> make_ta (js_global "Uint32Array") data
  | _ -> failwith ("SpocRT: unknown elementType: " ^ et)

let bpe = function
  | "f32" | "i32" | "u32" -> 4
  | et -> failwith ("SpocRT: unknown elementType: " ^ et)

let usage s =
  js_to_int (Js.Unsafe.get (js_global "GPUBufferUsage") (Js.string s))

let map_read () =
  js_to_int (Js.Unsafe.get (js_global "GPUMapMode") (Js.string "READ"))

let shader_compute () =
  js_to_int (Js.Unsafe.get (js_global "GPUShaderStage") (Js.string "COMPUTE"))

let js_err err =
  Js.Optdef.case
    (Js.Optdef.return (js_get err "message"))
    (fun () -> "unknown error")
    Js.to_string

(** Shared context threaded through the async promise chain. *)
type ctx = {
  device : Js.Unsafe.any;
  buffers_js : Js.Unsafe.any;
  n_bufs : int;
  params_js : Js.Unsafe.any;
  has_params : bool;
  workgroup_x : int;
  wgsl : string;
  inputs : (string * float array) list;
  scalars : (string * float) list;
  outputs_wanted : string list;
  on_done : (string * float array) list -> unit;
  on_error : string -> unit;
}

let create_buf ctx size usage =
  Js.Unsafe.meth_call
    ctx.device
    "createBuffer"
    [|
      Js.Unsafe.inject
        (Js.Unsafe.obj
           [|
             ("size", Js.Unsafe.inject (js_of_int size));
             ("usage", Js.Unsafe.inject (js_of_int usage));
           |]);
    |]

let queue_write ctx buf data =
  let q = js_get ctx.device "queue" in
  Js.Unsafe.meth_call
    q
    "writeBuffer"
    [|
      Js.Unsafe.inject buf;
      Js.Unsafe.inject (js_of_int 0);
      Js.Unsafe.inject data;
    |]
  |> ignore

let destroy buf = Js.Unsafe.meth_call buf "destroy" [||] |> ignore

let cleanup gbs rbs ub =
  Hashtbl.iter (fun _ b -> destroy b) gbs ;
  Hashtbl.iter (fun _ (b, _, _) -> destroy b) rbs ;
  Option.iter destroy ub

let build_bgl ctx =
  let vis = shader_compute () in
  let mk_entry binding typ =
    Js.Unsafe.obj
      [|
        ("binding", Js.Unsafe.inject (js_of_int binding));
        ("visibility", Js.Unsafe.inject (js_of_int vis));
        ( "buffer",
          Js.Unsafe.inject
            (Js.Unsafe.obj [|("type", Js.Unsafe.inject (Js.string typ))|]) );
      |]
  in
  let bents =
    Array.init ctx.n_bufs (fun i ->
        let d = js_arr_get ctx.buffers_js i in
        let typ =
          if String.equal (js_str d "access") "read" then "read-only-storage"
          else "storage"
        in
        mk_entry (js_int d "binding") typ)
  in
  let ents =
    if ctx.has_params then
      Js.array
        (Array.append
           bents
           [|mk_entry (js_int ctx.params_js "binding") "uniform"|])
    else Js.array bents
  in
  Js.Unsafe.meth_call
    ctx.device
    "createBindGroupLayout"
    [|Js.Unsafe.inject (Js.Unsafe.obj [|("entries", Js.Unsafe.inject ents)|])|]

let alloc_storage ctx =
  let gbs = Hashtbl.create 8 and rbs = Hashtbl.create 8 and maxn = ref 0 in
  let stor = usage "STORAGE" lor usage "COPY_SRC" lor usage "COPY_DST" in
  for i = 0 to ctx.n_bufs - 1 do
    let d = js_arr_get ctx.buffers_js i in
    let name = js_str d "name" and et = js_str d "elementType" in
    let data =
      match List.assoc_opt name ctx.inputs with
      | None -> failwith ("SpocRT: missing input for buffer \"" ^ name ^ "\"")
      | Some d -> d
    in
    let blen = Array.length data * bpe et in
    let gb = create_buf ctx blen stor in
    queue_write ctx gb (typed_array et data) ;
    Hashtbl.replace gbs name gb ;
    Hashtbl.replace
      rbs
      name
      (create_buf ctx blen (usage "MAP_READ" lor usage "COPY_DST"), blen, et) ;
    if Array.length data > !maxn then maxn := Array.length data
  done ;
  (gbs, rbs, !maxn)

let pack_uniform ctx =
  let bsz = js_int ctx.params_js "byteSize" in
  let ab =
    Js.Unsafe.new_obj
      (js_global "ArrayBuffer")
      [|Js.Unsafe.inject (js_of_int bsz)|]
  in
  let fields = js_get ctx.params_js "fields" in
  for fi = 0 to js_arr_len fields - 1 do
    let f = js_arr_get fields fi in
    let kind = js_str f "kind"
    and off = js_int f "offset"
    and ftype = js_str f "type" in
    let ctor, v =
      if String.equal kind "length" then
        let vn = js_str f (Js.to_string (Js.string "of")) in
        let d =
          match List.assoc_opt vn ctx.inputs with
          | None ->
              failwith
                ("SpocRT: missing input for length field, vec \"" ^ vn ^ "\"")
          | Some d -> d
        in
        ("Int32Array", float_of_int (Array.length d))
      else
        let fn = js_str f "name" in
        let v =
          match List.assoc_opt fn ctx.scalars with Some v -> v | None -> 0.0
        in
        ( (match ftype with
          | "f32" -> "Float32Array"
          | "u32" -> "Uint32Array"
          | _ -> "Int32Array"),
          v )
    in
    let view =
      Js.Unsafe.new_obj
        (js_global ctor)
        [|
          Js.Unsafe.inject ab;
          Js.Unsafe.inject (js_of_int off);
          Js.Unsafe.inject (js_of_int 1);
        |]
    in
    Js.Unsafe.set view (js_of_int 0) (Js.Unsafe.inject (js_of_float v))
  done ;
  let ub = create_buf ctx bsz (usage "UNIFORM" lor usage "COPY_DST") in
  queue_write ctx ub ab ;
  ub

let build_bg ctx bgl gbs ub =
  let mk b r =
    Js.Unsafe.obj
      [|
        ("binding", Js.Unsafe.inject (js_of_int b));
        ("resource", Js.Unsafe.inject r);
      |]
  in
  let mk_buf_res name =
    Js.Unsafe.obj [|("buffer", Js.Unsafe.inject (Hashtbl.find gbs name))|]
  in
  let bents =
    Array.init ctx.n_bufs (fun i ->
        let d = js_arr_get ctx.buffers_js i in
        mk (js_int d "binding") (mk_buf_res (js_str d "name")))
  in
  let ents =
    match ub with
    | None -> Js.array bents
    | Some u ->
        let ur = Js.Unsafe.obj [|("buffer", Js.Unsafe.inject u)|] in
        Js.array (Array.append bents [|mk (js_int ctx.params_js "binding") ur|])
  in
  Js.Unsafe.meth_call
    ctx.device
    "createBindGroup"
    [|
      Js.Unsafe.inject
        (Js.Unsafe.obj
           [|
             ("layout", Js.Unsafe.inject bgl); ("entries", Js.Unsafe.inject ents);
           |]);
    |]

let encode_submit ctx pipeline bg gbs rbs dispatch =
  let enc = Js.Unsafe.meth_call ctx.device "createCommandEncoder" [||] in
  let pass = Js.Unsafe.meth_call enc "beginComputePass" [||] in
  Js.Unsafe.meth_call pass "setPipeline" [|Js.Unsafe.inject pipeline|] |> ignore ;
  Js.Unsafe.meth_call
    pass
    "setBindGroup"
    [|Js.Unsafe.inject (js_of_int 0); Js.Unsafe.inject bg|]
  |> ignore ;
  Js.Unsafe.meth_call
    pass
    "dispatchWorkgroups"
    [|Js.Unsafe.inject (js_of_int dispatch)|]
  |> ignore ;
  Js.Unsafe.meth_call pass "end" [||] |> ignore ;
  for i = 0 to ctx.n_bufs - 1 do
    let name = js_str (js_arr_get ctx.buffers_js i) "name" in
    let rb, blen, _ = Hashtbl.find rbs name in
    Js.Unsafe.meth_call
      enc
      "copyBufferToBuffer"
      [|
        Js.Unsafe.inject (Hashtbl.find gbs name);
        Js.Unsafe.inject (js_of_int 0);
        Js.Unsafe.inject rb;
        Js.Unsafe.inject (js_of_int 0);
        Js.Unsafe.inject (js_of_int blen);
      |]
    |> ignore
  done ;
  let q = js_get ctx.device "queue" in
  Js.Unsafe.meth_call
    q
    "submit"
    [|Js.Unsafe.inject (Js.array [|Js.Unsafe.meth_call enc "finish" [||]|])|]
  |> ignore

let on_mapped gbs rbs ub results remaining oname rb blen et ctx () =
  let mapped = Js.Unsafe.meth_call rb "getMappedRange" [||] in
  let ctor =
    match et with
    | "f32" -> "Float32Array"
    | "i32" -> "Int32Array"
    | _ -> "Uint32Array"
  in
  let view = Js.Unsafe.new_obj (js_global ctor) [|Js.Unsafe.inject mapped|] in
  let arr =
    Array.init
      (blen / bpe et)
      (fun i -> Js.to_float (Js.Unsafe.get view (js_of_int i)))
  in
  Js.Unsafe.meth_call rb "unmap" [||] |> ignore ;
  results := (oname, arr) :: !results ;
  decr remaining ;
  if !remaining = 0 then (
    cleanup gbs rbs ub ;
    ctx.on_done (List.rev !results))

let on_compile_info ctx bgl pipeline gbs rbs ub maxn ci =
  let msgs = js_get ci "messages" in
  let errors = ref [] in
  for i = 0 to js_arr_len msgs - 1 do
    let m = js_arr_get msgs i in
    if String.equal (js_str m "type") "error" then
      let line = js_to_int (js_get m "lineNum") in
      errors := (js_str m "message" ^ " @line" ^ string_of_int line) :: !errors
  done ;
  if !errors <> [] then
    ctx.on_error
      ("SpocRT: WGSL compile errors: " ^ String.concat " | " (List.rev !errors))
  else
    let dispatch = (maxn + ctx.workgroup_x - 1) / ctx.workgroup_x in
    encode_submit ctx pipeline (build_bg ctx bgl gbs ub) gbs rbs dispatch ;
    let results = ref [] and remaining = ref (List.length ctx.outputs_wanted) in
    if !remaining = 0 then (
      cleanup gbs rbs ub ;
      ctx.on_done [])
    else
      List.iter
        (fun oname ->
          match Hashtbl.find_opt rbs oname with
          | None ->
              decr remaining ;
              if !remaining = 0 then (
                cleanup gbs rbs ub ;
                ctx.on_done (List.rev !results))
          | Some (rb, blen, et) ->
              let mp =
                Js.Unsafe.meth_call
                  rb
                  "mapAsync"
                  [|Js.Unsafe.inject (js_of_int (map_read ()))|]
              in
              catch_ mp (fun err ->
                  cleanup gbs rbs ub ;
                  ctx.on_error ("SpocRT: mapAsync failed: " ^ js_err err)) ;
              then_
                mp
                (on_mapped gbs rbs ub results remaining oname rb blen et ctx))
        ctx.outputs_wanted

let on_device ctx device =
  let ctx = {ctx with device} in
  let shader_module =
    Js.Unsafe.meth_call
      ctx.device
      "createShaderModule"
      [|
        Js.Unsafe.inject
          (Js.Unsafe.obj [|("code", Js.Unsafe.inject (Js.string ctx.wgsl))|]);
      |]
  in
  match try Ok (alloc_storage ctx) with Failure m -> Error m with
  | Error m -> ctx.on_error m
  | Ok (gbs, rbs, maxn) -> (
      let ub_result =
        if ctx.has_params then
          try Ok (Some (pack_uniform ctx)) with Failure m -> Error m
        else Ok None
      in
      match ub_result with
      | Error m ->
          cleanup gbs rbs None ;
          ctx.on_error m
      | Ok ub ->
          let bgl = build_bgl ctx in
          let pl =
            Js.Unsafe.meth_call
              ctx.device
              "createPipelineLayout"
              [|
                Js.Unsafe.inject
                  (Js.Unsafe.obj
                     [|
                       ("bindGroupLayouts", Js.Unsafe.inject (Js.array [|bgl|]));
                     |]);
              |]
          in
          let pipeline =
            Js.Unsafe.meth_call
              ctx.device
              "createComputePipeline"
              [|
                Js.Unsafe.inject
                  (Js.Unsafe.obj
                     [|
                       ("layout", Js.Unsafe.inject pl);
                       ( "compute",
                         Js.Unsafe.inject
                           (Js.Unsafe.obj
                              [|
                                ("module", Js.Unsafe.inject shader_module);
                                ( "entryPoint",
                                  Js.Unsafe.inject (Js.string "main") );
                              |]) );
                     |]);
              |]
          in
          let ci_p =
            Js.Unsafe.meth_call shader_module "getCompilationInfo" [||]
          in
          catch_ ci_p (fun _ ->
              cleanup gbs rbs ub ;
              ctx.on_error "SpocRT: getCompilationInfo failed") ;
          then_ ci_p (on_compile_info ctx bgl pipeline gbs rbs ub maxn))

let request_adapter_high_perf () =
  Js.Unsafe.fun_call
    (Js.Unsafe.js_expr "navigator.gpu.requestAdapter.bind(navigator.gpu)")
    [|
      Js.Unsafe.inject
        (Js.Unsafe.obj
           [|
             ("powerPreference", Js.Unsafe.inject (Js.string "high-performance"));
           |]);
    |]

let on_adapter ctx adapter =
  if js_is_null_or_undef adapter then
    ctx.on_error "SpocRT: no WebGPU adapter available"
  else
    let dp = Js.Unsafe.meth_call adapter "requestDevice" [||] in
    catch_ dp (fun err ->
        ctx.on_error ("SpocRT: requestDevice failed: " ^ js_err err)) ;
    then_ dp (on_device ctx)

let run ~wgsl ~abi_json ~inputs ~scalars ~outputs_wanted ~on_done ~on_error () =
  if js_is_null_or_undef gpu then
    on_error "SpocRT: WebGPU not available (navigator.gpu is undefined)"
  else
    let abi = json_parse abi_json in
    let wg = js_get abi "workgroupSize" in
    let wg1 = js_to_int (js_arr_get wg 1)
    and wg2 = js_to_int (js_arr_get wg 2) in
    if wg1 <> 1 || wg2 <> 1 then
      on_error
        (Printf.sprintf
           "SpocRT: only 1D workgroups supported (got workgroupSize[1]=%d, \
            [2]=%d)"
           wg1
           wg2)
    else
      let ctx =
        {
          device = Js.Unsafe.inject Js.null;
          buffers_js = js_get abi "buffers";
          n_bufs = js_arr_len (js_get abi "buffers");
          params_js = js_get abi "params";
          has_params = not (js_is_null_or_undef (js_get abi "params"));
          workgroup_x = js_to_int (js_arr_get wg 0);
          wgsl;
          inputs;
          scalars;
          outputs_wanted;
          on_done;
          on_error;
        }
      in
      let ap = request_adapter_high_perf () in
      catch_ ap (fun err ->
          on_error ("SpocRT: requestAdapter failed: " ^ js_err err)) ;
      then_ ap (on_adapter ctx)
