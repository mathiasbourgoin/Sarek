# Benchmark Infrastructure TODO

Focus on tooling and infrastructure before implementing more benchmarks.

## Current Status

✅ **Completed:**
- Common library (statistics, timing, system info)
- Backend loader with conditional GPU support
- JSON output format with full system metadata
- CSV conversion tool (to_csv.exe)
- Aggregation tool (aggregate.exe)
- Matrix multiplication benchmark (working example)

## Priority Infrastructure Tasks

### 1. Unified Benchmark Runner
**Status:** Not started  
**Priority:** High  
**Description:** Single executable that runs multiple benchmarks

**Requirements:**
- [ ] Create `bench_runner.ml` main executable
- [ ] Command-line interface for selecting benchmarks
- [ ] Support running all benchmarks with `--all`
- [ ] Support running specific benchmark: `--benchmark matrix_mul`
- [ ] Support device filtering: `--device 0` or `--device-filter "OpenCL"`
- [ ] Support size ranges: `--sizes 256,512,1024` or `--size-range 256:4096:2x`
- [ ] Progress reporting for long-running suites
- [ ] Summary report at the end

**Design:**
```ocaml
type benchmark_spec = {
  name: string;
  run: config -> Device.t -> int -> unit;
  description: string;
  default_sizes: int list;
}

let all_benchmarks = [
  matrix_mul_naive;
  matrix_mul_tiled;
  vector_add;
  reduction_sum;
  (* ... *)
]
```

### 2. Plotting Tools (OCaml-based)
**Status:** Not started  
**Priority:** High  
**Description:** Generate plots from benchmark results

**Options:**
- **Option A:** Use existing OCaml libraries
  - `archimedes` - 2D plotting library
  - `owl` - Scientific computing with plotting
  - `plotly` - Interactive plots
  
- **Option B:** Generate gnuplot scripts
  - Output `.gp` files that can be run with gnuplot
  - Portable and widely used
  - Easy to customize

**Tasks:**
- [ ] Choose plotting approach (recommend gnuplot for portability)
- [ ] Create `plot_throughput.ml` - Throughput vs size plots
- [ ] Create `plot_speedup.ml` - Speedup compared to baseline
- [ ] Create `plot_scaling.ml` - Weak/strong scaling
- [ ] Create `plot_backends.ml` - Compare CUDA vs OpenCL vs Vulkan
- [ ] Support multiple output formats (PNG, PDF, SVG, EPS)
- [ ] Publication-quality defaults (fonts, line widths, colors)

**Example gnuplot output:**
```gnuplot
set terminal pdfcairo enhanced color font 'Times,12' size 6,4
set output 'matrix_mul_throughput.pdf'
set xlabel 'Matrix Size'
set ylabel 'GFLOPS'
set logscale x 2
set grid
plot 'results.csv' using 1:2 with linespoints title 'GPU' lw 2 pt 7
```

### 3. LaTeX Table Generation
**Status:** Not started  
**Priority:** Medium  
**Description:** Generate LaTeX tables from results

**Tasks:**
- [ ] Create `to_latex.ml` executable
- [ ] Read CSV or JSON results
- [ ] Generate publication-ready LaTeX tables
- [ ] Support different table formats:
  - [ ] Performance comparison table
  - [ ] Speedup matrix (rows=benchmarks, cols=devices)
  - [ ] Scaling results table
- [ ] Automatic formatting (bold best results, etc.)
- [ ] Include system info as table caption/footnote

**Example output:**
```latex
\begin{table}[ht]
\centering
\caption{Matrix Multiplication Performance (GFLOPS)}
\begin{tabular}{lrrr}
\toprule
Device & 512×512 & 1024×1024 & 2048×2048 \\
\midrule
CUDA RTX 4090 & 245.3 & \textbf{512.7} & \textbf{892.4} \\
OpenCL Arc A770 & 123.4 & 234.5 & 445.6 \\
CPU Baseline & 12.3 & 23.4 & 45.6 \\
\bottomrule
\end{tabular}
\end{table}
```

### 4. Statistics and Analysis Tools
**Status:** Partial (basic stats in common.ml)  
**Priority:** Medium  
**Description:** More advanced statistical analysis

**Tasks:**
- [ ] Confidence intervals (95%, 99%)
- [ ] Outlier detection (IQR method)
- [ ] Regression analysis (performance vs size)
- [ ] Efficiency calculations (% of peak theoretical)
- [ ] Roofline model analysis
- [ ] Create `analyze.ml` tool for post-processing

**Features:**
```ocaml
(* Detect if results are statistically different *)
val t_test : float array -> float array -> bool

(* Calculate confidence interval *)
val confidence_interval : float array -> float -> float * float

(* Detect outliers *)
val detect_outliers : float array -> int list

(* Fit performance model: time = a + b*size + c*size^2 *)
val fit_model : (int * float) list -> float * float * float
```

### 5. Result Comparison and Diffing
**Status:** Not started  
**Priority:** Medium  
**Description:** Compare results across runs/machines

**Tasks:**
- [ ] Create `compare.ml` tool
- [ ] Compare two benchmark runs (before/after optimization)
- [ ] Highlight performance regressions
- [ ] Show speedup/slowdown percentages
- [ ] Support multiple result files
- [ ] Output in human-readable format or JSON

**Example usage:**
```bash
# Compare before and after optimization
dune exec benchmarks/compare.exe -- \
  results/before/*.json \
  results/after/*.json \
  --output comparison.txt

# Compare across machines
dune exec benchmarks/compare.exe -- \
  results/machine1/*.json \
  results/machine2/*.json \
  --show-speedup
```

### 6. Configuration Files
**Status:** Not started  
**Priority:** Low  
**Description:** YAML/JSON config for benchmark suites

**Tasks:**
- [ ] Define configuration file format
- [ ] Support benchmark selection
- [ ] Support parameter sweeps
- [ ] Device filtering rules
- [ ] Output preferences
- [ ] Create example configs

**Example config:**
```yaml
benchmarks:
  - name: matrix_mul_naive
    sizes: [256, 512, 1024, 2048, 4096]
    iterations: 20
    warmup: 5
  
  - name: vector_add
    sizes: [1024, 4096, 16384, 65536]
    iterations: 50
    warmup: 10

devices:
  filter: "GPU"  # Only GPU backends
  exclude: ["Interpreter"]

output:
  directory: results/
  format: json
  csv_summary: true
  plot: true
```

### 7. Web Dashboard (Optional/Future)
**Status:** Not planned  
**Priority:** Low  
**Description:** Web interface for viewing results

**Could use:**
- OCaml Dream framework for web server
- Store results in SQLite database
- Interactive plots with plotly.js
- Compare runs, filter by device/date
- Download results as CSV/JSON

### 8. CI Integration
**Status:** Not started  
**Priority:** Low (after more benchmarks exist)  
**Description:** Performance regression tracking in CI

**Tasks:**
- [ ] GitHub Actions workflow
- [ ] Run subset of benchmarks on each commit
- [ ] Compare against baseline
- [ ] Fail if performance regression > threshold
- [ ] Store historical data
- [ ] Generate performance trend plots

## Implementation Order

### Phase 1: Essential Tools
1. **Unified benchmark runner** - Makes it easy to run benchmarks
2. **Gnuplot script generator** - Basic plotting capability
3. **LaTeX table generator** - For paper writing

### Phase 2: Analysis Tools  
4. **Statistics enhancements** - Better analysis
5. **Comparison tool** - Before/after analysis

### Phase 3: Nice-to-Have
6. **Configuration files** - Easier to manage suites
7. **CI integration** - Performance tracking

## Questions to Decide

1. **Plotting library choice:**
   - Gnuplot (recommended: portable, powerful, widely used)
   - OCaml native (archimedes, owl)
   - Generate data + Python/matplotlib script

2. **Unified runner design:**
   - Register benchmarks in a list (simple)
   - Plugin system with dynamic loading (complex)
   - Compile-time selection (current approach)

3. **Configuration format:**
   - YAML (readable, need library)
   - JSON (already have yojson)
   - S-expressions (native to OCaml)
   - Command-line only (current)

4. **Output organization:**
   - Flat directory with hostname prefixes (current)
   - Hierarchical: results/hostname/benchmark/
   - Database (SQLite)

## Next Steps

Let's discuss which infrastructure tasks to tackle first. I recommend:

1. Start with **unified benchmark runner** - makes everything easier
2. Then **gnuplot script generator** - immediate visualization
3. Then **LaTeX table tool** - for documentation/papers

Sound good?
