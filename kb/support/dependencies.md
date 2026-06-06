# Vendored Dependencies and Static Assets

## Component Inventory

- `dependencies/CL/*.h`: vendored OpenCL/Khronos C headers and extensions.
- `dependencies/CL/LICENCE`: GPLv3 text file currently stored beside OpenCL headers.
- `dependencies/Cuda/*.h`: vendored CUDA/NVRTC/vector headers.
- `dependencies/Cuda/LICENCE`: GPLv3 text file currently stored beside CUDA headers.
- `gh-pages/static/**`: vendored browser assets for legacy Jupyter/IOCaml pages, including jQuery, jQuery UI, Bootstrap, RequireJS, CodeMirror, highlight.js, marked, notebook JS, CSS, images, and Font Awesome font.
- `gh-pages/pres_resources/**`, `gh-pages/docs/talks/**`, image/PDF/font/binary files: static archival assets.

## Per-File Purpose

- `dependencies/CL/opencl.h`: aggregate include for OpenCL headers.
- `dependencies/CL/cl.h`, `cl_platform.h`: core OpenCL API and platform type definitions.
- `dependencies/CL/cl_ext.h`, `cl_gl*.h`, `cl_d3d*.h`: OpenCL extension, GL sharing, and Direct3D sharing declarations.
- `dependencies/Cuda/nvrtc.h`: NVRTC runtime compilation API declarations.
- `dependencies/Cuda/cuComplex.h`: CUDA complex helper declarations.
- `dependencies/Cuda/host_defines.h`, `vector_types.h`: CUDA host/device annotation macros and vector type definitions.
- `gh-pages/static/components/*`: vendored JS/CSS libraries used by legacy notebook UI.
- `gh-pages/static/notebook/*`, `gh-pages/static/tree/*`, `gh-pages/static/base/*`: vendored/static notebook frontend code.
- `gh-pages/static/custom/*`, `gh-pages/static/dateformat/*`, `gh-pages/static/auth/js/loginwidget.js`: legacy custom/static helpers.
- Binary/static media under `gh-pages/pres_resources`, `gh-pages/docs/talks`, `gh-pages/docs/lena.png`, `gh-pages/benchmarks/descriptions/images`, and `gh-pages/static/**/images`: static assets; no semantic code review performed.

## Features and APIs

- OpenCL headers provide constants, typedefs, and function prototypes used by OpenCL bindings.
- CUDA headers provide NVRTC API shape, CUDA vector types, host/device macros, and complex-number helpers used by CUDA bindings.
- Static browser assets support legacy notebook browsing/editing/presentation pages and old site UI.

## Invariants

- Vendored headers should match the API version expected by the backend bindings.
- License files should match the vendored header licenses and preserve upstream notices.
- Static/vendor assets should be treated as third-party unless explicitly owned by this project.
- Runtime code should not modify vendored dependency headers.

## Potential Invariant Violations or Bugs

- `dependencies/CL/LICENCE` is GPLv3 text (`dependencies/CL/LICENCE:1-10`), but OpenCL headers contain Khronos permissive notices (`dependencies/CL/opencl.h:2-17`, `dependencies/CL/cl_d3d10.h:2-17`). This looks like a mismatched license file.
- `dependencies/Cuda/LICENCE` is also GPLv3 text (`dependencies/Cuda/LICENCE:1-10`), but CUDA headers contain NVIDIA ownership/restriction notices (`dependencies/Cuda/nvrtc.h:4-10`, `dependencies/Cuda/host_defines.h:2-17`, `dependencies/Cuda/vector_types.h:2-17`). This is a higher-risk provenance/compliance issue.
- No source URL, upstream version/date, or checksum metadata was found for `dependencies/**` or `gh-pages/static/**`.
- Some vendored static libraries are minified or old (`gh-pages/static/components/jquery/jquery.min.js`, `gh-pages/static/components/jquery-ui/ui/minified/jquery-ui.min.js`, `gh-pages/javascripts/screenfull.min.js`); they were inventoried as vendor/static, not line-by-line audited.

## Performance and Maintainability Risks

- Vendored C headers can drift from system SDK versions and backend FFI bindings.
- Old static JS/CSS libraries increase security-maintenance burden if active pages load them.
- Binary assets and one-line vendored/minified files make code review and provenance tracking difficult.
- Inaccurate license metadata can block package publication or redistribution.

## Related Tests and Checks

- No direct dependency provenance or license validation checks were found.
- Backend builds implicitly compile/use headers through backend code outside this support slice.
- Site build implicitly includes static assets through Jekyll.

## Missing Tests

- License/provenance inventory check for `dependencies/**` and `gh-pages/static/**`.
- Header version compatibility checks against backend FFI expectations.
- Static asset vulnerability/version scan or explicit archived-status policy.
- Checksum verification for vendored third-party files.

## Concrete Improvement Candidates

- Add `dependencies/THIRD_PARTY.md` with source URL, upstream version/date, license, and checksum per vendored header set.
- Replace mismatched `LICENCE` files with correct upstream license notices or add clear explanation if GPLv3 applies through another source.
- Prefer system SDK headers where possible, or pin exact upstream SDK header versions.
- Add a static/vendor manifest for `gh-pages/static/**` and mark inactive legacy notebook assets as archived.
- Consider removing unused legacy static libraries from active Jekyll output.
