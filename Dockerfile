FROM ocaml/opam:ubuntu-22.04-ocaml-5.4

USER root
# Install critical system deps
RUN apt-get update && apt-get install -y \
    python3-pip libgmp-dev pkg-config libffi-dev m4 libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyterlab thebe

# Fix for Binder: repo2docker often passes NB_USER as 'jovyan'
# We want to ensure opam can still find its root in /home/opam/.opam
ENV OPAMROOT=/home/opam/.opam
# Ensure the opam user has ownership of the home directory
RUN chown -R opam:opam /home/opam

USER opam
WORKDIR /home/opam/Sarek

# 1. Update opam and install build deps
RUN opam update
# Build deps come from the project's own opam metadata rather than a hand-kept
# list. The hardcoded list silently drifted from reality — js_of_ocaml-ppx is
# needed by the public sarek.webgpu_runtime library and was absent, so this
# image build failed on `Library "js_of_ocaml-ppx" not found` while every local
# build passed. Deriving them from dune-project makes the declaration
# load-bearing instead of decorative.
COPY --chown=opam:opam *.opam dune-project ./
RUN opam install -y . --deps-only
RUN opam install -y ocamlfind ocaml-compiler-libs

# 2. Install Jupyter and ZeroMQ
RUN opam install -y conf-zmq zmq zmq-lwt jupyter

# 3. Copy and build the project
COPY --chown=opam:opam . .
RUN opam exec -- dune build @install && \
    opam exec -- dune install

# 4. Final configuration
RUN opam exec -- ocaml-jupyter-opam-genspec && \
    opam exec -- jupyter kernelspec install --user --name ocaml-jupyter $(opam var share)/jupyter

RUN cat > /home/opam/.ocaml-jupyter-init.ml <<'EOF'
#use "topfind";;
#directory "/home/opam/.opam/5.4/lib/ocaml";;
#load "Stdlib__Effect.cma";;
EOF

RUN python3 - <<'PY'
import json
import pathlib

path = pathlib.Path("/home/opam/.local/share/jupyter/kernels/ocaml-jupyter/kernel.json")
data = json.loads(path.read_text())
argv = data["argv"]
if "--init-file" not in argv:
    argv[1:1] = ["--init-file", "/home/opam/.ocaml-jupyter-init.ml"]
else:
    idx = argv.index("--init-file")
    if idx + 1 >= len(argv) or argv[idx + 1] != "/home/opam/.ocaml-jupyter-init.ml":
        argv[idx + 1:idx + 2] = ["/home/opam/.ocaml-jupyter-init.ml"]

path.write_text(json.dumps(data, indent=2))
PY

# Provide the Stdlib__Effect bytecode so the toplevel can load it
RUN opam exec -- sh -c "cd \$(opam var lib)/ocaml && \
    cp effect.mli stdlib__Effect.mli && \
    cp effect.ml stdlib__Effect.ml && \
    ocamlc.opt -c stdlib__Effect.mli && \
    ocamlc.opt -c stdlib__Effect.ml && \
    ocamlc.opt -a -o Stdlib__Effect.cma stdlib__Effect.cmo && \
    rm -f stdlib__Effect.mli stdlib__Effect.ml"

# Fix for OCaml 5 Effects in Toplevel
RUN echo '#use "topfind";;' > /home/opam/.ocamlinit && \
    echo '#directory "^";;' >> /home/opam/.ocamlinit && \
    echo '#load "Stdlib__Effect.cma";;' >> /home/opam/.ocamlinit

EXPOSE 8888
ENTRYPOINT ["opam", "exec", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
