---
layout: page
title: Benchmark Dashboard
---

<style>
.dashboard-controls {
    display: flex;
    gap: 20px;
    margin: 20px 0;
    padding: 15px;
    background: var(--code-bg);
    border-radius: 8px;
    flex-wrap: wrap;
    align-items: center;
}

.control-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.control-group label {
    font-weight: 600;
    font-size: 0.9em;
    color: var(--text-color);
}

.control-group select {
    padding: 8px 12px;
    font-size: 1em;
    border: 2px solid var(--link-color);
    border-radius: 4px;
    background: var(--bg-color);
    color: var(--text-color);
    cursor: pointer;
    min-width: 200px;
}

.control-group select:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(var(--link-color-rgb), 0.2);
}

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

.no-data {
    text-align: center;
    padding: 40px;
    color: var(--text-color);
    opacity: 0.6;
    font-style: italic;
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

.info-box {
    padding: 15px;
    background: var(--highlight-bg, #fff3cd);
    color: var(--text-color, #333);
    border-left: 4px solid var(--link-color);
    margin: 20px 0;
    border-radius: 4px;
}

.back-link {
    display: inline-block;
    margin: 20px 0;
    padding: 10px 15px;
    background: var(--link-color);
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-weight: 600;
}

.back-link:hover {
    opacity: 0.9;
}
</style>

# Interactive Benchmark Dashboard

<div class="info-box">
    <strong>📊 Multi-Chart Comparison View:</strong> This dashboard displays multiple charts simultaneously to help you understand performance from different angles. Select a benchmark and view mode below.
</div>

<a href="index.html" class="back-link">← Back to Simple View</a>

## Dashboard Controls

<div class="dashboard-controls">
    <div class="control-group">
        <label for="benchmark-select">Benchmark:</label>
        <select id="benchmark-select">
            <option value="transpose" selected>Matrix Transpose (Naive vs Tiled)</option>
            <option value="matrix_mul">Matrix Multiplication</option>
            <option value="vector_add">Vector Addition</option>
            <option value="reduction">Parallel Reduction</option>
        </select>
    </div>
    
    <div class="control-group">
        <label for="view-select">View Mode:</label>
        <select id="view-select">
            <option value="comparison" selected>4-Chart Comparison</option>
            <option value="detailed">Detailed Scaling</option>
            <option value="device-matrix">Device Matrix</option>
        </select>
    </div>
</div>

## Performance Analysis

<div id="dashboard-grid" class="dashboard-grid-4">
    <div class="chart-card">
        <p class="no-data">Loading benchmark data...</p>
    </div>
</div>

## View Modes Explained

- **4-Chart Comparison**: See performance scaling, algorithm comparison, backend comparison, and speedup analysis all at once
- **Detailed Scaling**: Large chart showing all devices, backends, and algorithms together
- **Device Matrix**: Table view comparing all devices across algorithm variants

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
<script src="../javascripts/benchmark-dashboard.js"></script>
