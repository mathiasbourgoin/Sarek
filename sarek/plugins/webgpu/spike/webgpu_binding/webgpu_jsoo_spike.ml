(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** SPIKE -- WebGPU js_of_ocaml end-to-end de-risking proof. NOT production
    code. Self-contained; depends only on js_of_ocaml.

    Exports [globalThis.SpocWebGpuSpike.runVectorAdd(aArray, bArray, callback)]
    which drives a full WebGPU compute dispatch (vector add) entirely from OCaml
    via [Js_of_ocaml.Js.Unsafe] and returns the result to OCaml via an async
    callback after [mapAsync] resolves.

    Promise chaining strategy: every [then_] call wraps an OCaml function in a
    JS callback so control returns to OCaml at each async step; no JS blobs are
    delegated to [eval] or external JS helpers. *)

open Js_of_ocaml

(* -------------------------------------------------------------------------
   Tiny promise helpers -- the core of the async seam.
   [then_ p f] chains [f] onto promise [p].  The returned promise is ignored
   intentionally: this is a fire-and-forget compute chain; errors surface as
   unhandled rejections, which is acceptable in a spike.
   ------------------------------------------------------------------------- *)

let then_ p f =
  ignore
    (Js.Unsafe.meth_call p "then" [|Js.Unsafe.inject (Js.wrap_callback f)|])

let catch_ p f =
  ignore
    (Js.Unsafe.meth_call p "catch" [|Js.Unsafe.inject (Js.wrap_callback f)|])

(* -------------------------------------------------------------------------
   WGSL shader -- hardcoded; layout:'auto' so no explicit GPUBindGroupLayout.
   ------------------------------------------------------------------------- *)

let wgsl_source =
  "@group(0) @binding(0) var<storage, read>       a : array<f32>;\n\
   @group(0) @binding(1) var<storage, read>       b : array<f32>;\n\
   @group(0) @binding(2) var<storage, read_write> c : array<f32>;\n\n\
   @compute @workgroup_size(64)\n\
   fn main(@builtin(global_invocation_id) gid : vec3<u32>) {\n\
   \\  let i = gid.x;\n\
   \\  if (i < arrayLength(&c)) { c[i] = a[i] + b[i]; }\n\
   }\n"

(* -------------------------------------------------------------------------
   GPUBufferUsage flags (WebGPU spec section 24.4).
   ------------------------------------------------------------------------- *)

let usage_storage = 0x0080

let usage_copy_src = 0x0004

let usage_copy_dst = 0x0008

let usage_map_read = 0x0001

let create_buffer device byte_length usage =
  let desc = Js.Unsafe.obj [||] in
  Js.Unsafe.set
    desc
    "size"
    (Js.Unsafe.inject (Js.number_of_float (float_of_int byte_length))) ;
  Js.Unsafe.set
    desc
    "usage"
    (Js.Unsafe.inject (Js.number_of_float (float_of_int usage))) ;
  Js.Unsafe.meth_call device "createBuffer" [|Js.Unsafe.inject desc|]

(* -------------------------------------------------------------------------
   Core implementation.

   Called from JS as SpocWebGpuSpike.runVectorAdd(a, b, callback) where
   [a] and [b] are JS Float32Arrays of equal length N, and [callback] is a JS
   function called as callback(Float32Array|null, errorString|null).

   The async read-back seam: mapAsync returns a Promise; its resolve handler
   copies the mapped range into an OCaml float array, builds a JS Float32Array
   from it, and calls the OCaml-wrapped callback -- returning GPU results to
   OCaml via a callback after one async boundary.
   ------------------------------------------------------------------------- *)

let run_vector_add a_js b_js callback_js =
  let gpu = Js.Unsafe.get (Js.Unsafe.get Js.Unsafe.global "navigator") "gpu" in
  let fail msg =
    ignore
      (Js.Unsafe.fun_call
         callback_js
         [|Js.Unsafe.inject Js.null; Js.Unsafe.inject (Js.string msg)|])
  in
  let adapter_promise =
    Js.Unsafe.meth_call
      gpu
      "requestAdapter"
      [|Js.Unsafe.inject (Js.Unsafe.obj [||])|]
  in
  catch_ adapter_promise (fun err ->
      let s = Js.to_string (Js.Unsafe.meth_call err "toString" [||]) in
      fail ("requestAdapter rejected: " ^ s)) ;
  then_ adapter_promise (fun adapter ->
      if
        Js.Unsafe.equals adapter Js.null
        || Js.Unsafe.equals adapter Js.undefined
      then fail "no WebGPU adapter available"
      else
        let device_promise =
          Js.Unsafe.meth_call
            adapter
            "requestDevice"
            [|Js.Unsafe.inject (Js.Unsafe.obj [||])|]
        in
        catch_ device_promise (fun err ->
            let s = Js.to_string (Js.Unsafe.meth_call err "toString" [||]) in
            fail ("requestDevice rejected: " ^ s)) ;
        then_ device_promise (fun device ->
            let n =
              int_of_float (Js.float_of_number (Js.Unsafe.get a_js "length"))
            in
            let byte_len = n * 4 in
            (* Storage buffers for a, b (read) and c (read_write + copy_src). *)
            let buf_a =
              create_buffer device byte_len (usage_storage lor usage_copy_dst)
            in
            let buf_b =
              create_buffer device byte_len (usage_storage lor usage_copy_dst)
            in
            let buf_c =
              create_buffer device byte_len (usage_storage lor usage_copy_src)
            in
            (* Readback buffer: MAP_READ | COPY_DST. *)
            let buf_read =
              create_buffer device byte_len (usage_map_read lor usage_copy_dst)
            in
            let queue = Js.Unsafe.get device "queue" in
            (* Upload a and b. *)
            Js.Unsafe.meth_call
              queue
              "writeBuffer"
              [|
                Js.Unsafe.inject buf_a;
                Js.Unsafe.inject (Js.number_of_float 0.0);
                Js.Unsafe.inject a_js;
              |]
            |> ignore ;
            Js.Unsafe.meth_call
              queue
              "writeBuffer"
              [|
                Js.Unsafe.inject buf_b;
                Js.Unsafe.inject (Js.number_of_float 0.0);
                Js.Unsafe.inject b_js;
              |]
            |> ignore ;
            (* Shader module. *)
            let shader_desc = Js.Unsafe.obj [||] in
            Js.Unsafe.set shader_desc "code" (Js.string wgsl_source) ;
            let shader =
              Js.Unsafe.meth_call
                device
                "createShaderModule"
                [|Js.Unsafe.inject shader_desc|]
            in
            (* Compute pipeline with layout:'auto'. *)
            let compute_stage = Js.Unsafe.obj [||] in
            Js.Unsafe.set compute_stage "module" shader ;
            Js.Unsafe.set compute_stage "entryPoint" (Js.string "main") ;
            let pipeline_desc = Js.Unsafe.obj [||] in
            Js.Unsafe.set pipeline_desc "layout" (Js.string "auto") ;
            Js.Unsafe.set pipeline_desc "compute" compute_stage ;
            let pipeline =
              Js.Unsafe.meth_call
                device
                "createComputePipeline"
                [|Js.Unsafe.inject pipeline_desc|]
            in
            (* Bind group entries. *)
            let mk_entry binding buf =
              let e = Js.Unsafe.obj [||] in
              Js.Unsafe.set
                e
                "binding"
                (Js.Unsafe.inject (Js.number_of_float (float_of_int binding))) ;
              let res = Js.Unsafe.obj [||] in
              Js.Unsafe.set res "buffer" buf ;
              Js.Unsafe.set e "resource" res ;
              e
            in
            let bg_desc = Js.Unsafe.obj [||] in
            Js.Unsafe.set
              bg_desc
              "layout"
              (Js.Unsafe.meth_call
                 pipeline
                 "getBindGroupLayout"
                 [|Js.Unsafe.inject (Js.number_of_float 0.0)|]) ;
            Js.Unsafe.set
              bg_desc
              "entries"
              (Js.array
                 [|mk_entry 0 buf_a; mk_entry 1 buf_b; mk_entry 2 buf_c|]) ;
            let bind_group =
              Js.Unsafe.meth_call
                device
                "createBindGroup"
                [|Js.Unsafe.inject bg_desc|]
            in
            (* Command encoding. *)
            let encoder =
              Js.Unsafe.meth_call
                device
                "createCommandEncoder"
                [|Js.Unsafe.inject (Js.Unsafe.obj [||])|]
            in
            let pass =
              Js.Unsafe.meth_call
                encoder
                "beginComputePass"
                [|Js.Unsafe.inject (Js.Unsafe.obj [||])|]
            in
            Js.Unsafe.meth_call pass "setPipeline" [|Js.Unsafe.inject pipeline|]
            |> ignore ;
            Js.Unsafe.meth_call
              pass
              "setBindGroup"
              [|
                Js.Unsafe.inject (Js.number_of_float 0.0);
                Js.Unsafe.inject bind_group;
              |]
            |> ignore ;
            let workgroups = (n + 63) / 64 in
            Js.Unsafe.meth_call
              pass
              "dispatchWorkgroups"
              [|
                Js.Unsafe.inject (Js.number_of_float (float_of_int workgroups));
              |]
            |> ignore ;
            Js.Unsafe.meth_call pass "end" [||] |> ignore ;
            (* Copy c to readback buffer. *)
            Js.Unsafe.meth_call
              encoder
              "copyBufferToBuffer"
              [|
                Js.Unsafe.inject buf_c;
                Js.Unsafe.inject (Js.number_of_float 0.0);
                Js.Unsafe.inject buf_read;
                Js.Unsafe.inject (Js.number_of_float 0.0);
                Js.Unsafe.inject (Js.number_of_float (float_of_int byte_len));
              |]
            |> ignore ;
            let commands = Js.Unsafe.meth_call encoder "finish" [||] in
            Js.Unsafe.meth_call
              queue
              "submit"
              [|Js.Unsafe.inject (Js.array [|commands|])|]
            |> ignore ;
            (* --- ASYNC READ-BACK SEAM ---
               GPUMapMode.READ = 1.
               mapAsync resolves when the GPU result is ready to read from the
               CPU.  The resolve handler runs entirely in OCaml: it reads the
               mapped ArrayBuffer via a Float32Array view, copies values into
               an OCaml float array, then builds a new JS Float32Array to
               deliver to the callback.  This is the key correctness proof:
               GPU data crosses the async boundary back into OCaml. *)
            let map_promise =
              Js.Unsafe.meth_call
                buf_read
                "mapAsync"
                [|Js.Unsafe.inject (Js.number_of_float 1.0)|]
            in
            catch_ map_promise (fun err ->
                let s =
                  Js.to_string (Js.Unsafe.meth_call err "toString" [||])
                in
                fail ("mapAsync rejected: " ^ s)) ;
            then_ map_promise (fun _unit ->
                let ab = Js.Unsafe.meth_call buf_read "getMappedRange" [||] in
                let f32_ctor = Js.Unsafe.get Js.Unsafe.global "Float32Array" in
                let view = Js.Unsafe.new_obj f32_ctor [|Js.Unsafe.inject ab|] in
                (* Copy into OCaml-side float array. *)
                let result_arr =
                  Array.init n (fun i ->
                      Js.float_of_number
                        (Js.Unsafe.get
                           view
                           (Js.number_of_float (float_of_int i))))
                in
                Js.Unsafe.meth_call buf_read "unmap" [||] |> ignore ;
                Js.Unsafe.meth_call buf_a "destroy" [||] |> ignore ;
                Js.Unsafe.meth_call buf_b "destroy" [||] |> ignore ;
                Js.Unsafe.meth_call buf_c "destroy" [||] |> ignore ;
                Js.Unsafe.meth_call buf_read "destroy" [||] |> ignore ;
                (* Build a JS Float32Array from the OCaml result. *)
                let out =
                  Js.Unsafe.new_obj
                    f32_ctor
                    [|Js.Unsafe.inject (Js.number_of_float (float_of_int n))|]
                in
                Array.iteri
                  (fun i v ->
                    Js.Unsafe.set
                      out
                      (Js.number_of_float (float_of_int i))
                      (Js.number_of_float v))
                  result_arr ;
                ignore
                  (Js.Unsafe.fun_call
                     callback_js
                     [|Js.Unsafe.inject out; Js.Unsafe.inject Js.null|]))))

(* -------------------------------------------------------------------------
   Module registration.
   ------------------------------------------------------------------------- *)

let () =
  let m = Js.Unsafe.obj [||] in
  Js.Unsafe.set
    m
    "runVectorAdd"
    (Js.Unsafe.callback (fun a b cb -> run_vector_add a b cb)) ;
  Js.Unsafe.set Js.Unsafe.global "SpocWebGpuSpike" m
