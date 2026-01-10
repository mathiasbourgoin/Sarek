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

<p style="margin: 20px 0; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; text-align: center;">
    <strong>🎨 Try the new <a href="dashboard.html" style="color: white; text-decoration: underline; font-weight: bold;">Multi-Chart Dashboard</a>!</strong><br>
    View 4 charts simultaneously: scaling, comparison, backends, and speedup analysis.
</p>

<div class="contribute-box">
    <h3>📊 Contribute Your Benchmarks!</h3>
    <p>Help build a comprehensive performance database by running benchmarks on your hardware:</p>
    <ol>
        <li>Clone the repository: <code>git clone https://github.com/mathiasbourgoin/SPOC.git</code></li>
        <li>Install dependencies: <code>opam install --deps-only -y .</code></li>
        <li>Run all benchmarks: <code>eval $(opam env) && ./benchmarks/run_all_benchmarks.sh</code></li>
        <li>Submit results as a PR (takes ~5-15 minutes)</li>
    </ol>
    <p><strong>One script runs all 5 benchmarks:</strong> matrix multiplication, vector addition, reduction, transpose (naive + tiled)</p>
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

## Benchmark Suite

<div class="benchmark-selector">
    <label for="benchmark-select">Select Benchmark:</label>
    <select id="benchmark-select">
        <option value="matrix_mul">Matrix Multiplication (Naive)</option>
        <option value="vector_add">Vector Addition (Memory Bandwidth)</option>
        <option value="reduction">Parallel Reduction (Sum)</option>
        <option value="transpose">Matrix Transpose (Naive)</option>
        <option value="transpose_tiled">Matrix Transpose (Tiled - Optimized)</option>
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
    
    <div class="chart-container">
        <canvas id="matrixMulChart"></canvas>
        <div id="no-data-message" style="display: none; text-align: center; padding: 100px 20px; color: var(--text-muted, #666);">
            <p style="font-size: 1.2em; margin-bottom: 10px;">📊 No benchmark data available yet</p>
            <p>Submit your benchmark results via PR to see them displayed here!</p>
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
<script src="{{ site.baseurl }}/javascripts/benchmark-viewer.js"></script>
<script>
    // Load benchmark data and initialize
    loadBenchmarkData('{{ site.baseurl }}/benchmarks/data/latest.json');
</script>
