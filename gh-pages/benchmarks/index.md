---
layout: default
title: Benchmark Results
---

<style>
.benchmark-header {
    margin: 40px 0 20px 0;
}

.benchmark-tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    border-bottom: 2px solid var(--border-color);
    flex-wrap: wrap;
}

.benchmark-tab {
    padding: 10px 20px;
    background: none;
    border: none;
    border-bottom: 3px solid transparent;
    cursor: pointer;
    font-size: 16px;
    font-weight: 500;
    color: var(--text-color);
    transition: all 0.2s;
}

.benchmark-tab:hover {
    background: var(--hover-bg);
}

.benchmark-tab.active {
    border-bottom-color: var(--link-color);
    color: var(--link-color);
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
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 8px;
}

.contribute-box h3 {
    margin-top: 0;
    color: white;
}

.contribute-box a {
    color: white;
    text-decoration: underline;
}
</style>

# Performance Benchmarks

<div class="info-box">
    <strong>🚧 Work in Progress:</strong> This page shows preliminary benchmark results. More benchmarks and devices are being added continuously.
</div>

Interactive performance results for Sarek across different GPUs and backends. You can filter by backend and device to compare performance.

<div class="contribute-box">
    <h3>📊 Contribute Your Benchmarks!</h3>
    <p>Help build a comprehensive performance database by running benchmarks on your hardware:</p>
    <ol>
        <li>Clone the repository: <code>git clone https://github.com/mathiasbourgoin/SPOC.git</code></li>
        <li>Build and run benchmarks: <code>dune exec benchmarks/bench_matrix_mul.exe -- --output results/</code></li>
        <li>Submit results as a PR with files in <code>benchmarks/results/</code></li>
    </ol>
    <p>We're especially interested in results from:</p>
    <ul>
        <li>NVIDIA GPUs (RTX 40xx, 30xx, Tesla, etc.)</li>
        <li>AMD GPUs (RX 7000, 6000 series, MI series)</li>
        <li>Intel Arc GPUs (A770, A750, A380)</li>
        <li>Apple Silicon (M1, M2, M3, M4 series)</li>
        <li>Mobile GPUs (Qualcomm, Mali, PowerVR)</li>
    </ul>
    <p>See <a href="https://github.com/mathiasbourgoin/SPOC/tree/main/benchmarks">benchmarks/README.md</a> for detailed instructions.</p>
</div>

## Benchmark Suite

<div class="benchmark-tabs">
    <button class="benchmark-tab active" data-benchmark="matrix_mul">Matrix Multiplication</button>
    <button class="benchmark-tab" data-benchmark="vector_add">Vector Add (Coming Soon)</button>
    <button class="benchmark-tab" data-benchmark="reduction">Reduction (Coming Soon)</button>
    <button class="benchmark-tab" data-benchmark="transpose">Transpose (Coming Soon)</button>
</div>

<!-- Matrix Multiplication Benchmark -->
<div id="matrix_mul" class="benchmark-content active">
    <h3>Matrix Multiplication (Naive)</h3>
    <p>Dense matrix multiplication: <code>C = A × B</code> where A, B, C are N×N matrices.</p>
    <p><strong>Metric:</strong> GFLOPS (billions of floating-point operations per second)</p>
    
    <div class="filter-controls">
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
    
    <div class="chart-container">
        <canvas id="matrixMulChart"></canvas>
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
<script src="{{ site.baseurl }}/javascripts/benchmark-viewer.js"></script>
<script>
    // Load benchmark data and initialize
    loadBenchmarkData('{{ site.baseurl }}/benchmarks/data/latest.json');
</script>
