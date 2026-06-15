(* Backend initialization - loads all available backends *)

let init () =
  (* Always-available backends *)
  Sarek_native.Native_plugin.init () ;
  Sarek_interpreter.Interpreter_plugin.init () ;
  (* GPU backends are conditionally included via select *)
  Backend_cuda.init () ;
  Backend_opencl.init () ;
  Backend_vulkan.init () ;
  Backend_metal.init ()
