all:
	dune build

clean:
	dune clean

opam:
	dune build @install
	echo 'available: [ os = "linux" ]'  >> sarek.opam
	echo 'available: [ os = "linux" ]'  >> sarek_ppx.opam


install: opam
	dune install

uninstall:
	dune uninstall

# Core PPX/unit/comparison/e2e builds (GPU execution not required)
test:
	dune build @sarek/tests/runtest

# Build and run selected GPU e2e tests (requires a working OpenCL/CUDA device)
test_e2e_gpu:
	dune build \
		sarek/tests/e2e/test_nbody_ppx.exe \
		sarek/tests/e2e/test_ray_ppx.exe
	dune exec sarek/tests/e2e/test_nbody_ppx.exe -- -d 0
	dune exec sarek/tests/e2e/test_ray_ppx.exe -- -d 0

# Force rebuild/re-run of PPX tests even if previously built
test-force:
	dune clean || true
	dune build --force @sarek/tests/runtest

# Optional CUDA samples (require CUDA toolchain and graphics libs)
test_samples_cuda:
	@if [ -z "$$CUDA_PATH" ] || [ ! -f "$$CUDA_PATH/lib64/libnvrtc.so" ]; then \
	  echo "Skipping CUDA samples (CUDA_PATH/libnvrtc.so not found)"; \
	else \
	  dune build --profile=cuda \
	    Samples/src/DeviceQuery/DeviceQuery.exe \
	    Samples/src/VecAdd/VecAdd.exe \
	    Samples/src/Mandelbrot/Mandelbrot.exe && \
	  dune exec --profile=cuda Samples/src/DeviceQuery/DeviceQuery.exe && \
	  dune exec --profile=cuda Samples/src/VecAdd/VecAdd.exe && \
	  dune exec --profile=cuda Samples/src/Mandelbrot/Mandelbrot.exe ; \
	fi

test_ppx:
	# Build unit/comparison tests and all Sarek PPX e2e binaries (execution may require GPU)
	SKIP_OCAMLFORMAT=1 dune build @sarek/tests/runtest

# Run interpreter tests (no GPU required)
test_interpreter:
	# @echo "=== Interpreter unit tests ==="
	# dune build sarek/tests/unit/test_interp.exe
	# @echo "Running interpreter unit tests..."
	# dune exec sarek/tests/unit/test_interp.exe
	@echo "=== Interpreter e2e tests ==="
	dune build sarek/tests/e2e/test_vector_add.exe
	@echo "Running vector_add on interpreter..."
	dune exec sarek/tests/e2e/test_vector_add.exe -- --interpreter -s 64 -b 16
	@echo "=== Interpreter tests passed ==="

# Negative tests - verify that expected compile errors are raised
# Uses --profile=negative to enable the negative test libraries
test_negative:
	@echo "=== Negative tests (expected compile errors) ==="
	@echo "Testing type mismatch detection..."
	@dune build --profile=negative sarek/tests/negative/neg_test_convention_kernel_fail2.cma 2>&1 | tee /tmp/neg1.out | grep -q "Cannot unify types" && echo "  PASS: type mismatch" || (cat /tmp/neg1.out; false)
	@echo "Testing barrier in diverged control flow..."
	@dune build --profile=negative sarek/tests/negative/neg_test_barrier_diverged.cma 2>&1 | tee /tmp/neg2.out | grep -q "Barrier called in diverged control flow" && echo "  PASS: barrier diverged" || (cat /tmp/neg2.out; false)
	@echo "Testing superstep in diverged control flow..."
	@dune build --profile=negative sarek/tests/negative/neg_test_superstep_diverged.cma 2>&1 | tee /tmp/neg3.out | grep -q "Barrier called in diverged control flow" && echo "  PASS: superstep diverged" || (cat /tmp/neg3.out; false)
	@echo "Testing unbound function detection..."
	@dune build --profile=negative sarek/tests/negative/neg_test_unbound_function.cma 2>&1 | tee /tmp/neg4.out | grep -q "Unbound" && echo "  PASS: unbound function" || (cat /tmp/neg4.out; false)
	@echo "Testing reserved keyword detection..."
	@dune build --profile=negative sarek/tests/negative/neg_test_reserved_keyword.cma 2>&1 | tee /tmp/neg5.out | grep -q "reserved C/CUDA/OpenCL keyword" && echo "  PASS: reserved keyword" || (cat /tmp/neg5.out; false)
	@echo "Testing inline node exhaustion..."
	@dune build --profile=negative sarek/tests/negative/neg_test_inline_node_exhaustion.cma 2>&1 | tee /tmp/neg6.out | grep -q "Inlining produced .* nodes (limit: 10000)" && echo "  PASS: inline node exhaustion" || (cat /tmp/neg6.out; false)
	@echo "All negative tests passed"


# TODO: make following test v2 only
# sarek/tests/e2e/test_histogram.exe \
# Run new comprehensive Sarek e2e tests (GPU required)
test_comprehensive:
	@echo "=== Comprehensive Sarek e2e tests ==="
	dune build \
		sarek/tests/e2e/test_stencil.exe \
		sarek/tests/e2e/test_matrix_mul.exe \
		sarek/tests/e2e/test_reduce.exe \
		sarek/tests/e2e/test_complex_types.exe \
		sarek/tests/e2e/test_math_intrinsics.exe \
		sarek/tests/e2e/test_bitwise_ops.exe \
		sarek/tests/e2e/test_scan.exe \
		sarek/tests/e2e/test_transpose.exe \
		sarek/tests/e2e/test_sort.exe \
		sarek/tests/e2e/test_convolution.exe \
		sarek/tests/e2e/test_mandelbrot.exe \
		sarek/tests/e2e/test_inline_pragma.exe
	@echo "Running comprehensive tests..."
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_stencil.exe -- -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_matrix_mul.exe -- -s 1024
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_reduce.exe -- -s 8192
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_histogram.exe -- -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_complex_types.exe -- -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_math_intrinsics.exe -- -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_bitwise_ops.exe -- -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_scan.exe -- -s 256
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_transpose.exe -- -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_sort.exe -- -s 512
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_convolution.exe -- -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_mandelbrot.exe -- -s 4096
	@echo "Running inline pragma tests on GPU..."
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_inline_pragma.exe
	@echo "=== Comprehensive e2e tests passed ==="

# Run spoc package unit tests (framework, ir, registry)
test_spoc:
	@echo "=== SPOC package unit tests ==="
	dune build @spoc/runtest
	@echo "=== SPOC package tests passed ==="

# Run sarek/core unit tests
test_sarek_core:
	@echo "=== Sarek core unit tests ==="
	dune build @sarek/core/test/runtest
	@echo "=== Sarek core tests passed ==="

# Run all tests: unit tests, e2e tests, negative tests, and spoc tests
test-all: test test_spoc test_sarek_core test_interpreter test_negative
	@echo "=== All tests passed ==="

# E2E tests - quick verification with small datasets comparing GPU vs native CPU
# removed test for v2 compatibilit: test_histogram
E2E_TESTS = test_stencil test_matrix_mul test_reduce  \
            test_complex_types test_math_intrinsics test_bitwise_ops \
            test_scan test_transpose test_sort test_convolution test_mandelbrot

test-e2e:
	@echo "=== E2E Tests (small datasets, verification enabled) ==="
	@dune build $(addprefix sarek/tests/e2e/,$(addsuffix .exe,$(E2E_TESTS)))
	@echo ""
	@echo "--- Stencil ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_stencil.exe -- -s 1024
	@echo ""
	@echo "--- Matrix Multiplication ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_matrix_mul.exe -- -s 256
	@echo ""
	@echo "--- Reduction ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_reduce.exe -- -s 2048
	@echo ""
	@echo "--- Histogram ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_histogram.exe -- -s 1024
	@echo ""
	@echo "--- Complex Types ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_complex_types.exe -- -s 1024
	@echo ""
	@echo "--- Math Intrinsics ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_math_intrinsics.exe -- -s 1024
	@echo ""
	@echo "--- Bitwise Operations ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_bitwise_ops.exe -- -s 1024
	@echo ""
	@echo "--- Scan ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_scan.exe -- -s 256
	@echo ""
	@echo "--- Transpose ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_transpose.exe -- -s 1024
	@echo ""
	@echo "--- Sort ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_sort.exe -- -s 256
	@echo ""
	@echo "--- Convolution ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_convolution.exe -- -s 1024
	@echo ""
	@echo "--- Mandelbrot ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_mandelbrot.exe -- -s 1024
	@echo ""
	@echo "=== All E2E tests passed ==="

# Benchmarks - run all tests with --benchmark to compare all devices
benchmarks:
	@echo "=============================================="
	@echo "       SAREK BENCHMARK SUITE"
	@echo "=============================================="
	@echo ""
	@dune build $(addprefix sarek/tests/e2e/,$(addsuffix .exe,$(E2E_TESTS)))
	@echo ""
	@echo "--- Stencil (1D/2D with shared memory) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_stencil.exe -- --benchmark -s 1048576
	@echo ""
	@echo "--- Matrix Multiplication (naive + tiled) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_matrix_mul.exe -- --benchmark -s 262144
	@echo ""
	@echo "--- Reduction (sum, max, dot product) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_reduce.exe -- --benchmark -s 4194304
	@echo ""
	@echo "--- Histogram (shared memory atomics) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_histogram.exe -- --benchmark -s 4194304
	@echo ""
	@echo "--- Complex Types (records, particles) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_complex_types.exe -- --benchmark -s 1048576
	@echo ""
	@echo "--- Math Intrinsics (sin, cos, exp, log, sqrt) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_math_intrinsics.exe -- --benchmark -s 4194304
	@echo ""
	@echo "--- Bitwise Operations ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_bitwise_ops.exe -- --benchmark -s 4194304
	@echo ""
	@echo "--- Scan (prefix sum) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_scan.exe -- --benchmark -s 256
	@echo ""
	@echo "--- Transpose (naive + coalesced) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_transpose.exe -- --benchmark -s 4194304
	@echo ""
	@echo "--- Sort (bitonic, odd-even) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_sort.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Convolution (1D, 2D, Sobel) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_convolution.exe -- --benchmark -s 4194304
	@echo ""
	@echo "--- Mandelbrot / Julia ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_mandelbrot.exe -- --benchmark -s 4194304
	@echo ""
	@echo "=============================================="
	@echo "       BENCHMARK COMPLETE"
	@echo "=============================================="

# Fast benchmarks for CI - small sizes, Native+OpenCL only
benchmarks-fast:
	@echo "=============================================="
	@echo "   SAREK FAST BENCHMARK (CI-friendly)"
	@echo "=============================================="
	@echo ""
	@dune build sarek/tests/e2e/test_vector_add.exe \
		sarek/tests/e2e/test_matrix_mul.exe \
		sarek/tests/e2e/test_reduce.exe \
		sarek/tests/e2e/test_transpose.exe \
		sarek/tests/e2e/test_math_intrinsics.exe
	@echo ""
	@echo "--- Vector Add ---"
	@dune exec sarek/tests/e2e/test_vector_add.exe -- -s 4096
	@echo ""
	@echo "--- Matrix Mul ---"
	@dune exec sarek/tests/e2e/test_matrix_mul.exe -- -s 1024
	@echo ""
	@echo "--- Reduction ---"
	@dune exec sarek/tests/e2e/test_reduce.exe -- -s 8192
	@echo ""
	@echo "--- Transpose ---"
	@dune exec sarek/tests/e2e/test_transpose.exe -- -s 4096
	@echo ""
	@echo "--- Math Intrinsics ---"
	@dune exec sarek/tests/e2e/test_math_intrinsics.exe -- -s 4096
	@echo ""
	@echo "=============================================="
	@echo "   FAST BENCHMARK COMPLETE"
	@echo "=============================================="

# Tiered test suite - tests organized by complexity
# Tier 1: Simple kernels (low complexity, good starting point)
TIER1_TESTS = test_vector_add test_bitwise_ops test_math_intrinsics test_transpose

# Tier 2: Medium complexity (2D indexing, neighbor access, atomics, shared memory)
TIER2_TESTS = test_matrix_mul test_stencil test_convolution test_reduce test_scan test_sort

# Tier 3: Complex types (custom types, type registration, variants)
TIER3_TESTS = test_ktype_record test_registered_type test_registered_variant test_complex_types test_nested_types

# Tier 4: Advanced features (real algorithms, physics, raytracing, polymorphism)
TIER4_TESTS = test_mandelbrot test_nbody_ppx test_ray_ppx test_polymorphism

# Run a single tier
test-tier1:
	@echo "=============================================="
	@echo "  TIER 1: Simple Kernels"
	@echo "=============================================="
	@dune build $(addprefix sarek/tests/e2e/,$(addsuffix .exe,$(TIER1_TESTS)))
	@for t in $(TIER1_TESTS); do \
		echo ""; \
		echo "--- $$t ---"; \
		LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/$$t.exe -- -s 1024 || exit 1; \
	done
	@echo ""
	@echo "=== Tier 1 PASSED ==="

test-tier2:
	@echo "=============================================="
	@echo "  TIER 2: Medium Complexity (Metal)"
	@echo "=============================================="
	@dune build $(addprefix sarek/tests/e2e/,$(addsuffix .exe,$(TIER2_TESTS)))
	@echo ""
	@echo "--- test_matrix_mul ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_matrix_mul.exe -- --metal -s 256
	@echo ""
	@echo "--- test_stencil ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_stencil.exe -- --metal -s 1024
	@echo ""
	@echo "--- test_convolution ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_convolution.exe -- --metal -s 1024
	@echo ""
	@echo "--- test_reduce ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_reduce.exe -- --metal -s 2048
	@echo ""
	@echo "--- test_scan ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_scan.exe -- --metal -s 256
	@echo ""
	@echo "--- test_sort ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_sort.exe -- --metal -s 256
	@echo ""
	@echo "=== Tier 2 PASSED ==="

test-tier3:
	@echo "=============================================="
	@echo "  TIER 3: Complex Types"
	@echo "=============================================="
	@dune build $(addprefix sarek/tests/e2e/,$(addsuffix .exe,$(TIER3_TESTS)))
	@for t in $(TIER3_TESTS); do \
		echo ""; \
		echo "--- $$t ---"; \
		LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/$$t.exe || exit 1; \
	done
	@echo ""
	@echo "=== Tier 3 PASSED ==="

test-tier4:
	@echo "=============================================="
	@echo "  TIER 4: Advanced Features"
	@echo "=============================================="
	@dune build $(addprefix sarek/tests/e2e/,$(addsuffix .exe,$(TIER4_TESTS)))
	@echo ""
	@echo "--- test_mandelbrot ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_mandelbrot.exe -- -s 1024
	@echo ""
	@echo "--- test_nbody_ppx ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_nbody_ppx.exe -- -d 0
	@echo ""
	@echo "--- test_ray_ppx ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_ray_ppx.exe -- -d 0
	@echo ""
	@echo "--- test_polymorphism ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec sarek/tests/e2e/test_polymorphism.exe
	@echo ""
	@echo "=== Tier 4 PASSED ==="

# Run all tiers sequentially
test-tiers: test-tier1 test-tier2 test-tier3 test-tier4
	@echo ""
	@echo "=============================================="
	@echo "  ALL TIERS PASSED"
	@echo "=============================================="

check: all install install_sarek samples test 

mr_proper: clean
	rm -rf _build

release:
	dune-release tag
	dune-release distrib
	dune-release publish
	dune-release opam pkg
	dune-release opam submit

# Benchmark targets
.PHONY: benchmarks bench-all bench-update

benchmarks: bench-all

bench-all:
@./benchmarks/run_all_benchmarks.sh

bench-update:
@echo "Running benchmarks and updating web data..."
@./benchmarks/run_all_benchmarks.sh results
@echo "Benchmark data updated. Review and commit changes."
