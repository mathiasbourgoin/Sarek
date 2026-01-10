// Multi-chart benchmark dashboard
// Displays 4 charts simultaneously for comprehensive comparison

let benchmarkData = null;
let charts = {
    scaling: null,
    comparison: null,
    backends: null,
    speedup: null,
    ranking: null
};
let currentBenchmark = 'transpose';
let currentView = 'comparison';

// Color palette for different backends
const BACKEND_COLORS = {
    'CUDA': '#76b900',
    'OpenCL': '#0071c5',
    'Vulkan': '#ac162c',
    'Metal': '#555555',
    'Native': '#ff9500',
    'Interpreter': '#999999'
};

// Algorithm colors (for naive vs optimized comparisons)
const ALGO_COLORS = {
    'naive': '#dc3545',      // Red
    'tiled': '#28a745',      // Green
    'optimized': '#28a745',  // Green
    'baseline': '#6c757d'    // Gray
};

// Benchmark configurations
const BENCHMARK_CONFIGS = {
    'matrix_mul': {
        title: 'Matrix Multiplication',
        xLabel: 'Matrix Size (N×N)',
        throughputLabel: 'GFLOPS',
        variants: ['matrix_mul_naive']
    },
    'vector_add': {
        title: 'Vector Addition',
        xLabel: 'Vector Size (elements)',
        throughputLabel: 'GB/s',
        variants: ['vector_add']
    },
    'reduction': {
        title: 'Parallel Reduction',
        xLabel: 'Array Size (elements)',
        throughputLabel: 'GB/s',
        variants: ['reduction_sum']
    },
    'transpose': {
        title: 'Matrix Transpose',
        xLabel: 'Matrix Size (N×N)',
        throughputLabel: 'GB/s',
        variants: ['transpose_naive', 'transpose_tiled']
    }
};

// Load benchmark data
async function loadBenchmarkData(dataUrl) {
    try {
        const response = await fetch(dataUrl);
        benchmarkData = await response.json();
        
        initializeControls();
        updateDashboard();
    } catch (error) {
        console.error('Error loading benchmark data:', error);
        document.getElementById('dashboard-grid').innerHTML = 
            '<div class="error">Error loading benchmark data. Please check console.</div>';
    }
}

// Initialize UI controls
function initializeControls() {
    const benchmarkSelect = document.getElementById('benchmark-select');
    const viewSelect = document.getElementById('view-select');
    
    if (benchmarkSelect) {
        benchmarkSelect.addEventListener('change', (e) => {
            currentBenchmark = e.target.value;
            updateDashboard();
        });
    }
    
    if (viewSelect) {
        viewSelect.addEventListener('change', (e) => {
            currentView = e.target.value;
            updateDashboard();
        });
    }
}

// Update entire dashboard
function updateDashboard() {
    if (!benchmarkData) return;
    
    const config = BENCHMARK_CONFIGS[currentBenchmark];
    if (!config) return;
    
    // Destroy existing charts
    Object.keys(charts).forEach(key => {
        if (charts[key]) {
            charts[key].destroy();
            charts[key] = null;
        }
    });
    
    if (currentView === 'comparison') {
        showComparisonView(config);
    } else if (currentView === 'detailed') {
        showDetailedView(config);
    } else if (currentView === 'device-matrix') {
        showDeviceMatrix(config);
    } else if (currentView === 'system-ranking') {
        showSystemRanking(config);
    }
}

// Show 4-chart comparison view
function showComparisonView(config) {
    const grid = document.getElementById('dashboard-grid');
    grid.className = 'dashboard-grid-4';
    grid.innerHTML = `
        <div class="chart-card">
            <h4>Performance Scaling</h4>
            <canvas id="chart-scaling"></canvas>
        </div>
        <div class="chart-card">
            <h4>Algorithm Comparison</h4>
            <canvas id="chart-comparison"></canvas>
        </div>
        <div class="chart-card">
            <h4>Backend Comparison</h4>
            <canvas id="chart-backends"></canvas>
        </div>
        <div class="chart-card">
            <h4>Speedup Analysis</h4>
            <canvas id="chart-speedup"></canvas>
        </div>
    `;
    
    // Create all 4 charts
    createScalingChart(config);
    createComparisonChart(config);
    createBackendChart(config);
    createSpeedupChart(config);
}

// Show single large detailed view
function showDetailedView(config) {
    const grid = document.getElementById('dashboard-grid');
    grid.className = 'dashboard-grid-1';
    grid.innerHTML = `
        <div class="chart-card large">
            <h4>${config.title} - Detailed Scaling</h4>
            <canvas id="chart-detailed"></canvas>
        </div>
    `;
    
    createDetailedChart(config);
}

// Show device matrix heatmap
function showDeviceMatrix(config) {
    const grid = document.getElementById('dashboard-grid');
    grid.className = 'dashboard-grid-1';
    grid.innerHTML = `
        <div class="chart-card large">
            <h4>Device × Algorithm Performance Matrix</h4>
            <div id="matrix-container"></div>
        </div>
    `;
    
    createDeviceMatrix(config);
}

// Chart 1: Line chart showing performance scaling with size
function createScalingChart(config) {
    const ctx = document.getElementById('chart-scaling');
    if (!ctx) return;
    
    const datasets = [];
    const variants = config.variants;
    
    variants.forEach((variantName, idx) => {
        const variantResults = benchmarkData.results.filter(r => 
            r.benchmark && r.benchmark.name === variantName
        );
        
        if (variantResults.length === 0) return;
        
        // Group by device
        const deviceGroups = groupByDevice(variantResults);
        
        // Take first device for simplicity (or we could show multiple)
        const firstDevice = Object.keys(deviceGroups)[0];
        if (!firstDevice) return;
        
        const data = deviceGroups[firstDevice]
            .sort((a, b) => {
                const sizeA = getProblemSize(a.benchmark);
                const sizeB = getProblemSize(b.benchmark);
                return sizeA - sizeB;
            })
            .map(result => ({
                x: getProblemSize(result.benchmark),
                y: result.results[0]?.throughput_gflops || 0
            }));
        
        const algoType = variantName.includes('tiled') ? 'tiled' : 
                         variantName.includes('naive') ? 'naive' : 'baseline';
        
        datasets.push({
            label: formatAlgoName(variantName),
            data: data,
            borderColor: ALGO_COLORS[algoType],
            backgroundColor: ALGO_COLORS[algoType] + '20',
            borderWidth: 2,
            tension: 0.1,
            fill: false
        });
    });
    
    charts.scaling = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${config.throughputLabel}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: config.xLabel
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: config.throughputLabel
                    }
                }
            }
        }
    });
}

// Chart 2: Grouped bar chart comparing algorithms at specific size
function createComparisonChart(config) {
    const ctx = document.getElementById('chart-comparison');
    if (!ctx) return;
    
    const variants = config.variants;
    if (variants.length < 2) {
        // No comparison possible
        ctx.parentElement.innerHTML = '<p class="no-data">No comparison available (single algorithm)</p>';
        return;
    }
    
    // Find a common size to compare (prefer largest)
    const allSizes = new Set();
    variants.forEach(variantName => {
        const results = benchmarkData.results.filter(r => 
            r.benchmark && r.benchmark.name === variantName
        );
        results.forEach(r => {
            const size = getProblemSize(r.benchmark);
            if (size > 0) allSizes.add(size);
        });
    });
    
    if (allSizes.size === 0) {
        ctx.parentElement.innerHTML = '<p class="no-data">No data available for comparison</p>';
        return;
    }
    
    const sizes = Array.from(allSizes).sort((a, b) => a - b);
    const targetSize = sizes[Math.floor(sizes.length * 0.7)] || sizes[sizes.length - 1]; // Use ~70% of max size
    
    const labels = [];
    const datasets = {};
    
    variants.forEach(variantName => {
        const results = benchmarkData.results.filter(r => 
            r.benchmark && 
            r.benchmark.name === variantName &&
            getProblemSize(r.benchmark) === targetSize
        );
        
        results.forEach(result => {
            const deviceName = getDeviceName(result);
            if (!labels.includes(deviceName)) {
                labels.push(deviceName);
            }
            
            const algoName = formatAlgoName(variantName);
            if (!datasets[algoName]) {
                const algoType = variantName.includes('tiled') ? 'tiled' : 
                                 variantName.includes('naive') ? 'naive' : 'baseline';
                datasets[algoName] = {
                    label: algoName,
                    data: [],
                    backgroundColor: ALGO_COLORS[algoType],
                    borderWidth: 1
                };
            }
            
            datasets[algoName].data.push(result.results[0]?.throughput_gflops || 0);
        });
    });
    
    charts.comparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: Object.values(datasets)
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: `@ Size ${formatSize(targetSize)}`
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${config.throughputLabel}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: config.throughputLabel
                    }
                }
            }
        }
    });
}

// Chart 3: Backend comparison (grouped bars)
function createBackendChart(config) {
    const ctx = document.getElementById('chart-backends');
    if (!ctx) return;
    
    // Use first variant
    const variantName = config.variants[0];
    const results = benchmarkData.results.filter(r => 
        r.benchmark && r.benchmark.name === variantName
    );
    
    if (results.length === 0) {
        ctx.parentElement.innerHTML = '<p class="no-data">No data available</p>';
        return;
    }
    
    // Find common size
    const allSizes = new Set();
    results.forEach(r => {
        const size = getProblemSize(r.benchmark);
        if (size > 0) allSizes.add(size);
    });
    
    if (allSizes.size === 0) {
        ctx.parentElement.innerHTML = '<p class="no-data">No valid sizes found in data</p>';
        return;
    }
    
    const sizes = Array.from(allSizes).sort((a, b) => a - b);
    const targetSize = sizes[Math.floor(sizes.length * 0.7)] || sizes[sizes.length - 1];
    
    // Group by backend
    const backendData = {};
    const devices = new Set();
    
    results
        .filter(r => getProblemSize(r.benchmark) === targetSize)
        .forEach(result => {
            const backend = result.results[0]?.framework || 'Unknown';
            const device = getDeviceName(result);
            devices.add(device);
            
            if (!backendData[backend]) {
                backendData[backend] = {
                    label: backend,
                    data: [],
                    backgroundColor: BACKEND_COLORS[backend] || '#999999',
                    borderWidth: 1
                };
            }
        });
    
    const deviceLabels = Array.from(devices);
    
    // Fill data for each backend
    Object.keys(backendData).forEach(backend => {
        deviceLabels.forEach(device => {
            const result = results.find(r => 
                getProblemSize(r.benchmark) === targetSize &&
                r.results[0]?.framework === backend &&
                getDeviceName(r) === device
            );
            backendData[backend].data.push(result?.results[0]?.throughput_gflops || 0);
        });
    });
    
    charts.backends = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: deviceLabels,
            datasets: Object.values(backendData)
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: `@ Size ${formatSize(targetSize)}`
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${config.throughputLabel}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: config.throughputLabel
                    }
                }
            }
        }
    });
}

// Chart 4: Speedup line chart (tiled vs naive)
function createSpeedupChart(config) {
    const ctx = document.getElementById('chart-speedup');
    if (!ctx) return;
    
    const variants = config.variants;
    if (variants.length < 2) {
        ctx.parentElement.innerHTML = '<p class="no-data">No speedup comparison (single algorithm)</p>';
        return;
    }
    
    // Assume first is baseline, second is optimized
    const baselineVariant = variants.find(v => v.includes('naive')) || variants[0];
    const optimizedVariant = variants.find(v => v.includes('tiled') || v.includes('optimized')) || variants[1];
    
    const baselineResults = benchmarkData.results.filter(r => 
        r.benchmark && r.benchmark.name === baselineVariant
    );
    const optimizedResults = benchmarkData.results.filter(r => 
        r.benchmark && r.benchmark.name === optimizedVariant
    );
    
    if (baselineResults.length === 0 || optimizedResults.length === 0) {
        ctx.parentElement.innerHTML = '<p class="no-data">Insufficient data for speedup</p>';
        return;
    }
    
    // Calculate speedup by size
    const speedupData = [];
    const allSizes = new Set();
    baselineResults.forEach(r => allSizes.add(getProblemSize(r.benchmark)));
    
    Array.from(allSizes).sort((a, b) => a - b).forEach(size => {
        const baseline = baselineResults.find(r => getProblemSize(r.benchmark) === size);
        const optimized = optimizedResults.find(r => getProblemSize(r.benchmark) === size);
        
        if (baseline && optimized) {
            const baselinePerf = baseline.results[0]?.throughput_gflops || 0;
            const optimizedPerf = optimized.results[0]?.throughput_gflops || 0;
            
            if (baselinePerf > 0) {
                speedupData.push({
                    x: size,
                    y: optimizedPerf / baselinePerf
                });
            }
        }
    });
    
    charts.speedup = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Speedup (Optimized / Naive)',
                data: speedupData,
                borderColor: '#17a2b8',
                backgroundColor: '#17a2b820',
                borderWidth: 3,
                tension: 0.1,
                fill: true
            }, {
                label: 'Baseline (1×)',
                data: speedupData.map(d => ({ x: d.x, y: 1 })),
                borderColor: '#6c757d',
                borderWidth: 1,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            if (context.datasetIndex === 0) {
                                return `Speedup: ${context.parsed.y.toFixed(2)}×`;
                            }
                            return null;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: config.xLabel
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Speedup (×)'
                    }
                }
            }
        }
    });
}

// Create detailed single large chart
function createDetailedChart(config) {
    const ctx = document.getElementById('chart-detailed');
    if (!ctx) return;
    
    const datasets = [];
    const variants = config.variants;
    
    variants.forEach((variantName) => {
        const variantResults = benchmarkData.results.filter(r => 
            r.benchmark && r.benchmark.name === variantName
        );
        
        // Group by device and backend
        const deviceGroups = {};
        variantResults.forEach(result => {
            const key = `${getDeviceName(result)}_${result.results[0]?.framework || 'Unknown'}`;
            if (!deviceGroups[key]) {
                deviceGroups[key] = {
                    device: getDeviceName(result),
                    backend: result.results[0]?.framework || 'Unknown',
                    data: []
                };
            }
            deviceGroups[key].data.push(result);
        });
        
        Object.values(deviceGroups).forEach(group => {
            const data = group.data
                .sort((a, b) => {
                    const sizeA = getProblemSize(a.benchmark);
                    const sizeB = getProblemSize(b.benchmark);
                    return sizeA - sizeB;
                })
                .map(result => ({
                    x: getProblemSize(result.benchmark),
                    y: result.results[0]?.throughput_gflops || 0
                }));
            
            const algoType = variantName.includes('tiled') ? 'tiled' : 
                             variantName.includes('naive') ? 'naive' : 'baseline';
            const baseColor = ALGO_COLORS[algoType];
            
            datasets.push({
                label: `${formatAlgoName(variantName)} - ${group.device} (${group.backend})`,
                data: data,
                borderColor: baseColor,
                backgroundColor: baseColor + '20',
                borderWidth: 2,
                tension: 0.1,
                fill: false
            });
        });
    });
    
    charts.detailed = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${config.throughputLabel}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: config.xLabel
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: config.throughputLabel
                    }
                }
            }
        }
    });
}

// Create device matrix heatmap
function createDeviceMatrix(config) {
    const container = document.getElementById('matrix-container');
    if (!container) return;
    
    // Build a table showing device × algorithm performance
    const variants = config.variants;
    const deviceSet = new Set();
    
    benchmarkData.results
        .filter(r => r.benchmark && variants.includes(r.benchmark.name))
        .forEach(r => deviceSet.add(getDeviceName(r)));
    
    const devices = Array.from(deviceSet);
    
    let html = '<table class="perf-matrix"><thead><tr><th>Device</th>';
    variants.forEach(v => {
        html += `<th>${formatAlgoName(v)}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    devices.forEach(device => {
        html += `<tr><td class="device-name">${device}</td>`;
        variants.forEach(variantName => {
            const results = benchmarkData.results.filter(r => 
                r.benchmark && 
                r.benchmark.name === variantName &&
                getDeviceName(r) === device
            );
            
            if (results.length > 0) {
                // Get max performance across all sizes
                const maxPerf = Math.max(...results.map(r => r.results[0]?.throughput_gflops || 0));
                html += `<td class="perf-cell">${maxPerf.toFixed(2)}</td>`;
            } else {
                html += `<td class="perf-cell no-data">—</td>`;
            }
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

// Show system ranking view - sorts systems by peak performance
function showSystemRanking(config) {
    const grid = document.getElementById('dashboard-grid');
    grid.className = 'dashboard-grid-1';
    grid.innerHTML = `
        <div class="chart-card large">
            <h4>System Ranking: ${config.title}</h4>
            <p style="color: var(--text-color); opacity: 0.8; margin-bottom: 15px;">
                Systems ranked by peak performance across all problem sizes. 
                Shows best performance for each system/backend/algorithm combination.
            </p>
            <canvas id="chart-ranking"></canvas>
        </div>
    `;
    
    createRankingChart(config);
}

// Create ranking bar chart
function createRankingChart(config) {
    const ctx = document.getElementById('chart-ranking');
    if (!ctx) return;
    
    const variants = config.variants;
    
    // Collect all system/device/backend/algorithm combinations with their peak performance
    const systemPerformances = [];
    
    benchmarkData.results
        .filter(r => r.benchmark && variants.includes(r.benchmark.name))
        .forEach(result => {
            const systemName = result.system?.hostname || 'Unknown';
            const deviceName = getDeviceName(result);
            const backend = result.results[0]?.framework || 'Unknown';
            const algoName = formatAlgoName(result.benchmark.name);
            const throughput = result.results[0]?.throughput_gflops || 0;
            
            if (throughput === 0) return;
            
            // Create unique key for this combination
            const key = `${systemName}|${deviceName}|${backend}|${algoName}`;
            
            // Find existing entry or create new one
            let entry = systemPerformances.find(e => e.key === key);
            if (!entry) {
                entry = {
                    key: key,
                    systemName: systemName,
                    deviceName: deviceName,
                    backend: backend,
                    algoName: algoName,
                    peakPerformance: throughput,
                    label: `${deviceName} @ ${systemName} (${backend}, ${algoName})`
                };
                systemPerformances.push(entry);
            } else {
                // Update peak if this is better
                entry.peakPerformance = Math.max(entry.peakPerformance, throughput);
            }
        });
    
    // Sort by peak performance descending
    systemPerformances.sort((a, b) => b.peakPerformance - a.peakPerformance);
    
    // Take top 20 to avoid overcrowding
    const topSystems = systemPerformances.slice(0, 20);
    
    // Prepare data for horizontal bar chart
    const labels = topSystems.map(s => {
        // Truncate long labels
        const label = s.label;
        return label.length > 50 ? label.substring(0, 47) + '...' : label;
    });
    const data = topSystems.map(s => s.peakPerformance);
    const colors = topSystems.map(s => BACKEND_COLORS[s.backend] || '#666666');
    
    charts.ranking = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: config.throughputLabel,
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',  // Horizontal bars
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: (context) => {
                            const idx = context[0].dataIndex;
                            const entry = topSystems[idx];
                            return entry.label;
                        },
                        label: (context) => {
                            return `Peak: ${context.parsed.x.toFixed(2)} ${config.throughputLabel}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: `Peak ${config.throughputLabel}`
                    }
                },
                y: {
                    ticks: {
                        autoSkip: false,
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

// Helper functions
function groupByDevice(results) {
    const groups = {};
    results.forEach(result => {
        const device = getDeviceName(result);
        if (!groups[device]) groups[device] = [];
        groups[device].push(result);
    });
    return groups;
}

function getDeviceName(result) {
    // First, try to get the actual device that ran this benchmark from the results
    if (result.results && result.results.length > 0 && result.results[0].device_name) {
        return result.results[0].device_name;
    }
    
    // Fallback: use system devices list
    if (result.system && result.system.devices && result.system.devices.length > 0) {
        // Use first GPU device (not CPU)
        const gpuDevice = result.system.devices.find(d => d.framework !== 'Native' && d.framework !== 'Interpreter');
        if (gpuDevice) {
            return gpuDevice.name || 'Unknown GPU';
        }
        return result.system.devices[0].name || 'Unknown Device';
    }
    
    return 'Unknown Device';
}

function getProblemSize(benchmark) {
    if (!benchmark || !benchmark.parameters) return 0;
    const params = benchmark.parameters;
    if (params.n !== undefined) return params.n;
    if (params.size !== undefined) return params.size;
    if (params.width !== undefined) return params.width;
    return 0;
}

function formatAlgoName(name) {
    return name
        .replace('_naive', ' (Naive)')
        .replace('_tiled', ' (Tiled)')
        .replace('_', ' ')
        .split(' ')
        .map(w => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');
}

function formatSize(size) {
    if (size === undefined || size === null || size === 0) return 'N/A';
    if (size >= 1e9) return (size / 1e9).toFixed(1) + 'B';
    if (size >= 1e6) return (size / 1e6).toFixed(1) + 'M';
    if (size >= 1e3) return (size / 1e3).toFixed(1) + 'K';
    return size.toString();
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadBenchmarkData('data/latest.json');
});
