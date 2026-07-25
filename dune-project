(lang dune 3.15)
(name spoc)
(version dev)
(cram enable)

(generate_opam_files true)

; ctypes >= 0.24.0: the f16 host path depends on ctypes INTERNALS that its
; public API does not cover — Ctypes_bigarray.unsafe_address (the kind GADT has
; no Float16 arm, so Ctypes.bigarray_start raises for it) and
; Ctypes_static.CPointer / Ctypes_ptr.Fat.make, used to rebuild a MANAGED fat
; pointer so an f16 bigarray stays GC-rooted across a transfer. See
; sarek/core/Memory.ml's bigarray_void_ptr and the GC-root canary in
; sarek/tests/unit/test_float16.ml, which fails loudly if those semantics change.
; 0.24.0 is the version those semantics were verified against.

(source (github mathiasbourgoin/Sarek))
(license Cecill-B)
(authors "Mathias Bourgoin")
(maintainers "mathias.bourgoin@gmail.com")

; SDK/Framework - minimal types and plugin interface, no runtime
(package
 (name spoc)
 (synopsis "GPGPU framework for OCaml - SDK types and plugin interface")
 (description
  "SPOC provides a unified interface for GPU computing in OCaml.
This package contains the SDK: framework types, device abstraction,
IR types, and plugin registration interface.
Install sarek for the full runtime and DSL.")
 (depends
  (ocaml (>= 5.4.0))
  (dune (>= 3.15))
  (ctypes (>= 0.24.0))
  (bisect_ppx :with-test)))

; Sarek: full runtime, DSL, PPX, stdlib, and CPU backends
(package
 (name sarek)
 (synopsis "GPU kernel DSL for OCaml with runtime and CPU backends")
 (description
  "Sarek is a DSL for writing GPU kernels in OCaml syntax.
Includes the full runtime (Device, Vector, Memory management),
PPX preprocessor, standard library (Float32, Int32, etc.),
and CPU backends (native, interpreter) for development and testing.
Install sarek-cuda, sarek-opencl, or sarek-vulkan for GPU execution.")
 (depends
  (ocaml (>= 5.4.0))
  (dune (>= 3.15))
  (spoc (= :version))
  (ppxlib (>= 0.22.0))
  (ctypes (>= 0.24.0))
  (ctypes-foreign (>= 0.24.0))
  ; sarek.webgpu_runtime / sarek.transpile_web are public libraries, so they are
  ; part of @install — js_of_ocaml is a hard runtime dep, not a test-only one.
  (js_of_ocaml (>= 6.0.0))
  (js_of_ocaml-ppx (>= 6.0.0))
  (yojson :with-test)
  (alcotest :with-test)
  (qcheck-core :with-test)
  (bisect_ppx :with-test)))

; CUDA backend plugin
(package
 (name sarek-cuda)
 (synopsis "CUDA backend for Sarek GPU kernels")
 (description "Enables execution of Sarek kernels on NVIDIA GPUs via CUDA.")
 (depends
  (ocaml (>= 5.4.0))
  (dune (>= 3.15))
  (sarek (= :version))
  (ctypes (>= 0.24.0))
  (ctypes-foreign (>= 0.24.0)))
 (depopts
  conf-cuda))

; HIP (ROCm/AMD) backend plugin - native, bypasses ZLUDA/PTX
(package
 (name sarek-hip)
 (synopsis "ROCm/HIP backend for Sarek GPU kernels")
 (description "Enables native execution of Sarek kernels on AMD GPUs via ROCm/HIP (hiprtc JIT); RDNA3 rocWMMA tensor-core GEMM path.")
 (depends
  (ocaml (>= 5.4.0))
  (dune (>= 3.15))
  (sarek (= :version))
  (ctypes (>= 0.24.0))
  (ctypes-foreign (>= 0.24.0))))

; OpenCL backend plugin
(package
 (name sarek-opencl)
 (synopsis "OpenCL backend for Sarek GPU kernels")
 (description "Enables execution of Sarek kernels on GPUs and CPUs via OpenCL.")
 (depends
  (ocaml (>= 5.4.0))
  (dune (>= 3.15))
  (sarek (= :version))
  (ctypes (>= 0.24.0))
  (ctypes-foreign (>= 0.24.0)))
 (depopts
  conf-opencl))

; Vulkan backend plugin
(package
 (name sarek-vulkan)
 (synopsis "Vulkan compute backend for Sarek GPU kernels")
 (description "Enables execution of Sarek kernels on GPUs via Vulkan compute shaders.")
 (depends
  (ocaml (>= 5.4.0))
  (dune (>= 3.15))
  (sarek (= :version))
  (ctypes (>= 0.24.0))
  (ctypes-foreign (>= 0.24.0)))
 (depopts
  conf-vulkan))

; Metal backend plugin
(package
 (name sarek-metal)
 (synopsis "Metal backend for Sarek GPU kernels")
 (description "Enables execution of Sarek kernels on Apple GPUs via Metal framework (macOS/iOS).")
 (depends
  (ocaml (>= 5.4.0))
  (dune (>= 3.15))
  (sarek (= :version))
  (ctypes (>= 0.24.0))
  (ctypes-foreign (>= 0.24.0))))
