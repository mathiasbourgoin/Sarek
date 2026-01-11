---
layout: page
title: Benchmark Results
---

<style>
.benchmark-header {
    margin: 40px 0 20px 0;
}

.benchmark-selector {
    margin: 20px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.benchmark-selector label {
    font-weight: 600;
    color: var(--text-color);
}

.benchmark-selector select {
    padding: 8px 12px;
    font-size: 1em;
    border: 2px solid var(--link-color);
    border-radius: 4px;
    background: var(--bg-color);
    color: var(--text-color);
    cursor: pointer;
    min-width: 250px;
}

.benchmark-selector select:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(var(--link-color-rgb), 0.2);
}

.filter-controls {
    display: flex;
    gap: 20px;
    margin: 20px 0;
    padding: 15px;
    background: var(--code-bg);
    border-radius: 4px;
    flex-wrap: wrap;
    align-items: center;
}

.filter-group {
    display: flex;
    gap: 10px;
    align-items: center;
}

.filter-group label {
    font-weight: 600;
    color: var(--text-color);
}

.metric-selector {
    padding: 6px 10px;
    font-size: 0.95em;
    border: 2px solid var(--link-color);
    border-radius: 4px;
    background: var(--bg-color);
    color: var(--text-color);
    cursor: pointer;
    min-width: 180px;
}

.metric-selector:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(var(--link-color-rgb), 0.2);
}

.filter-checkbox {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: var(--bg-color);
    border-radius: 3px;
    cursor: pointer;
    user-select: none;
}

.filter-checkbox input[type="checkbox"] {
    cursor: pointer;
}

.chart-container {
    position: relative;
    height: 400px;
    margin: 30px 0;
    padding: 20px;
    background: var(--bg-color);
    border: 1px solid var(--border-color);
    border-radius: 4px;
}

.benchmark-content {
    display: none;
}

.benchmark-content.active {
    display: block;
}

.info-box {
    padding: 15px;
    background: var(--highlight-bg, #fff3cd);
    color: var(--text-color, #333);
    border-left: 4px solid var(--link-color);
    margin: 20px 0;
    border-radius: 4px;
}

.system-info {
    margin: 30px 0;
    padding: 15px;
    background: var(--code-bg);
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.9em;
}

.contribute-box {
    margin: 40px 0;
    border-radius: 8px;
    border: 2px solid var(--link-color);
}

.contribute-box summary {
    padding: 15px 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 1.1em;
    user-select: none;
    list-style: none;
}

.contribute-box summary::-webkit-details-marker {
    display: none;
}

.contribute-box summary::before {
    content: '▶ ';
    display: inline-block;
    transition: transform 0.2s;
}

.contribute-box[open] summary::before {
    transform: rotate(90deg);
}

.contribute-box summary:hover {
    opacity: 0.9;
}

.contribute-content {
    padding: 20px;
    background: var(--code-bg);
    border-radius: 0 0 6px 6px;
}

.contribute-content a {
    color: var(--link-color);
    text-decoration: underline;
}

/* Multi-chart dashboard layouts */
.dashboard-grid-4 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin: 30px 0;
}

.dashboard-grid-1 {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    margin: 30px 0;
}

@media (max-width: 768px) {
    .dashboard-grid-4 {
        grid-template-columns: 1fr;
    }
}

.chart-card {
    background: var(--bg-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.chart-card.large {
    min-height: 500px;
}

.chart-card h4 {
    margin: 0 0 15px 0;
    color: var(--text-color);
    font-size: 1.1em;
    border-bottom: 2px solid var(--link-color);
    padding-bottom: 8px;
}

.chart-card canvas {
    max-height: 300px;
}

.chart-card.large canvas {
    max-height: 450px;
}

.perf-matrix {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

.perf-matrix th,
.perf-matrix td {
    padding: 12px;
    text-align: center;
    border: 1px solid var(--border-color);
}

.perf-matrix th {
    background: var(--link-color);
    color: white;
    font-weight: 600;
}

.perf-matrix .device-name {
    text-align: left;
    font-weight: 600;
    background: var(--code-bg);
}

.perf-matrix .perf-cell {
    font-family: monospace;
    font-size: 1.1em;
}

.perf-matrix .perf-cell.no-data {
    color: var(--text-color);
    opacity: 0.3;
}

.no-data {
    text-align: center;
    padding: 40px;
    color: var(--text-color);
    opacity: 0.6;
    font-style: italic;
}

</style>

Interactive performance results for Sarek across different GPUs and backends. 

**6 comprehensive benchmarks** test different aspects of GPU performance:
- 🔢 **Compute-bound**: Matrix multiplication, Mandelbrot fractals
- 💾 **Memory-bound**: Vector addition, parallel reduction
- ⚡ **Optimization showcase**: Transpose (naive vs tiled)

Switch between **4 view modes**: Single Chart, 4-Panel Comparison, System Ranking, or Device Matrix. Each benchmark includes detailed descriptions with Sarek kernel code.

<details class="contribute-box">
    <summary>📊 Contribute Your Benchmarks!</summary>
    <div class="contribute-content">
        <p>Help build a comprehensive performance database by running benchmarks on your hardware:</p>
        <ol>
            <li>Clone the repository: <code>git clone https://github.com/mathiasbourgoin/SPOC.git</code></li>
            <li>Install dependencies: <code>opam install --deps-only -y .</code></li>
            <li>Run all benchmarks: <code>eval $(opam env) && ./benchmarks/run_all_benchmarks.sh</code></li>
            <li>Submit results as a PR (takes ~5-15 minutes)</li>
        </ol>
        <p><strong>One script runs all 6 benchmarks:</strong> matrix multiplication, vector addition, reduction, transpose (naive + tiled), and Mandelbrot fractal generation</p>
        <p>We're especially interested in results from:</p>
        <ul>
            <li>NVIDIA GPUs (RTX 40xx, 30xx, Tesla, etc.)</li>
            <li>AMD GPUs (RX 7000, 6000 series, MI series)</li>
            <li>Intel Arc GPUs (A770, A750, A380)</li>
            <li>Apple Silicon (M1, M2, M3, M4 series)</li>
            <li>Mobile GPUs (Qualcomm, Mali, PowerVR)</li>
        </ul>
        <p>See <a href="https://github.com/mathiasbourgoin/SPOC/blob/main/benchmarks/CONTRIBUTING.md">CONTRIBUTING.md</a> for detailed instructions.</p>
    </div>
</details>

## Benchmark Suite

<div class="benchmark-selector">
    <label for="benchmark-select">Select Benchmark:</label>
    <select id="benchmark-select">
        <option value="matrix_mul">Matrix Multiplication (Naive)</option>
        <option value="vector_add">Vector Addition (Memory Bandwidth)</option>
        <option value="reduction">Parallel Reduction (Sum)</option>
        <option value="transpose">Matrix Transpose (Naive)</option>
        <option value="transpose_tiled">Matrix Transpose (Tiled - Optimized)</option>
        <option value="mandelbrot">Mandelbrot Set (Fractal Generation)</option>
    </select>
</div>

## Benchmark Details

<div id="benchmark-description" style="margin: 30px 0; padding: 20px; background: var(--code-bg); border-radius: 8px; border-left: 4px solid var(--link-color);">
    <!-- Dynamically filled by JavaScript -->
</div>

<div class="view-mode-selector" style="margin: 20px 0;">
    <label for="view-mode-select">View Mode:</label>
    <select id="view-mode-select">
        <option value="single" selected>Single Chart View</option>
        <option value="comparison">4-Panel Comparison</option>
        <option value="ranking">System Ranking</option>
        <option value="matrix">Device Matrix</option>
    </select>
</div>

<!-- Matrix Multiplication Benchmark -->
<div id="matrix_mul" class="benchmark-content active">
    <h3>Matrix Multiplication (Naive)</h3>
    <p>Dense matrix multiplication: <code>C = A × B</code> where A, B, C are N×N matrices.</p>
    <p><strong>Metric:</strong> GFLOPS (billions of floating-point operations per second)</p>
    
    <div class="filter-controls">
        <div class="filter-group">
            <label>System:</label>
            <select id="system-select" class="metric-selector">
                <option value="all">All Systems</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Metric:</label>
            <select id="metric-select" class="metric-selector">
                <option value="time">Time (ms)</option>
                <option value="throughput">Throughput (GFLOPS/GB/s)</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Backends:</label>
            <label class="filter-checkbox">
                <input type="checkbox" class="backend-filter" data-backend="CUDA" checked>
                CUDA
            </label>
            <label class="filter-checkbox">
                <input type="checkbox" class="backend-filter" data-backend="OpenCL" checked>
                OpenCL
            </label>
            <label class="filter-checkbox">
                <input type="checkbox" class="backend-filter" data-backend="Vulkan" checked>
                Vulkan
            </label>
            <label class="filter-checkbox">
                <input type="checkbox" class="backend-filter" data-backend="Metal" checked>
                Metal
            </label>
        </div>
        <div class="filter-group">
            <label>Show CPU Baseline:</label>
            <label class="filter-checkbox">
                <input type="checkbox" class="cpu-filter" id="show-cpu">
                Include
            </label>
        </div>
    </div>
    
    <!-- Dynamic chart area - changes based on view mode -->
    <div id="chart-area">
        <div class="chart-container">
            <canvas id="matrixMulChart"></canvas>
            <div id="no-data-message" style="display: none; text-align: center; padding: 100px 20px; color: var(--text-muted, #666);">
                <p style="font-size: 1.2em; margin-bottom: 10px;">📊 No benchmark data available yet</p>
                <p>Submit your benchmark results via PR to see them displayed here!</p>
            </div>
        </div>
    </div>
    
    <div class="system-info" id="system-info">
        Loading system information...
    </div>
</div>

<!-- Vector Add Benchmark (placeholder) -->
<div id="vector_add" class="benchmark-content">
    <h3>Vector Add</h3>
    <p>Element-wise vector addition: <code>C[i] = A[i] + B[i]</code></p>
    <p><strong>Coming Soon</strong> - This benchmark is being implemented.</p>
</div>

<!-- Reduction Benchmark (placeholder) -->
<div id="reduction" class="benchmark-content">
    <h3>Parallel Reduction (Sum)</h3>
    <p>Parallel sum reduction of array elements.</p>
    <p><strong>Coming Soon</strong> - This benchmark is being implemented.</p>
</div>

<!-- Transpose Benchmark (placeholder) -->
<div id="transpose" class="benchmark-content">
    <h3>Matrix Transpose</h3>
    <p>Out-of-place matrix transpose measuring memory access patterns.</p>
    <p><strong>Coming Soon</strong> - This benchmark is being implemented.</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-ocaml.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-c.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-glsl.min.js"></script>
<script src="{{ site.baseurl }}/javascripts/benchmark-viewer.js"></script>
<script>
    // Load benchmark data and initialize
    loadBenchmarkData('{{ site.baseurl }}/benchmarks/data/latest.json');
</script>
