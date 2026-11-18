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

The install tree is put on OCAMLPATH explicitly (third-party deps still come
from the switch, where sarek is not installed), so this resolves the installed
copies. Up to dune 3.24 every action inherited _build/install/<context>/lib on
OCAMLPATH; dune 3.25 (ocaml/dune#14373) exposes only the packages an action
declares with (deps (package ...)), and (package sarek) is not usable here (see
the dune file). INSIDE_DUNE is the absolute path of the build context
(_build/default), so its install tree is ../install/default/lib.

  $ SAREK_LIB="$INSIDE_DUNE/../install/default/lib"
  $ test -f "$SAREK_LIB/sarek/META"
  $ export OCAMLPATH="$SAREK_LIB${OCAMLPATH:+:$OCAMLPATH}"
  $ DUNE_CACHE=disabled dune build --root . ./consumer.exe 2>&1
  $ ./_build/default/consumer.exe
  Interpreter
  Native
