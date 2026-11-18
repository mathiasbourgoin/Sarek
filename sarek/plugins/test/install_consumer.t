The Native and Interpreter CPU backends are installed under package sarek, so a
project outside this repository can link them as reference backends.

  $ cat > dune-project <<'DUNE'
  > (lang dune 3.15)
  > DUNE
  $ cat > dune <<'DUNE'
  > (executable
  >  (name consumer)
  >  (libraries sarek.native sarek.interpreter sarek.framework.registry))
  > DUNE
  $ cat > consumer.ml <<'ML'
  > let () =
  >   Sarek_native.Native_plugin.init () ;
  >   Sarek_interpreter.Interpreter_plugin.init () ;
  >   Spoc_framework_registry.Framework_registry.all_backend_names ()
  >   |> List.filter (fun n -> n = "Native" || n = "Interpreter")
  >   |> List.sort compare
  >   |> List.iter print_endline
  > ML

Dune puts the install tree of the packages this test declares with
(deps (package ...)) on OCAMLPATH (third-party deps still come from the
switch, where sarek is not installed), so this resolves the installed copies.

  $ DUNE_CACHE=disabled dune build --root . ./consumer.exe 2>&1
  $ ./_build/default/consumer.exe
  Interpreter
  Native
