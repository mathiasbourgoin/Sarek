// Benchmark viewer with interactive filtering
// Handles tab switching, backend filtering, and chart updates

let benchmarkData = null;
let currentChart = null;
let currentBenchmark = 'matrix_mul';

// Color palette for different backends
const BACKEND_COLORS = {
    'CUDA': '#76b900',      // NVIDIA green
    'OpenCL': '#0071c5',    // Intel blue
    'Vulkan': '#ac162c',    // Vulkan red
    'Metal': '#555555',     // Apple gray
    'Native': '#ff9500',    // Orange
    'Interpreter': '#999999' // Gray
};

// Load benchmark data from JSON
async function loadBenchmarkData(dataUrl) {
    try {
        const response = await fetch(dataUrl);
        benchmarkData = await response.json();
        
        initializeBenchmarkSelector();
        initializeFilters();
        updateChart();
        updateSystemInfo();
    } catch (error) {
        console.error('Error loading benchmark data:', error);
        document.getElementById('system-info').textContent = 
            'Error loading benchmark data. Please check console for details.';
    }
}

// Initialize benchmark selector
function initializeBenchmarkSelector() {
    const selector = document.getElementById('benchmark-select');
    if (!selector) return;
    
    selector.addEventListener('change', (e) => {
        currentBenchmark = e.target.value;
        updateChart();
    });
}

// Initialize filter controls
function initializeFilters() {
    // Backend filters
    const backendFilters = document.querySelectorAll('.backend-filter');
    backendFilters.forEach(filter => {
        filter.addEventListener('change', updateChart);
    });
    
    // CPU baseline filter
    const cpuFilter = document.getElementById('show-cpu');
    if (cpuFilter) {
        cpuFilter.addEventListener('change', updateChart);
    }
}

// Update chart based on current filters
function updateChart() {
    if (!benchmarkData) return;
    
    const canvas = document.getElementById('matrixMulChart');
    if (!canvas) return;
    
    // Get selected backends
    const selectedBackends = Array.from(document.querySelectorAll('.backend-filter:checked'))
        .map(cb => cb.dataset.backend);
    
    const showCpu = document.getElementById('show-cpu')?.checked || false;
    
    // Prepare datasets
    const datasets = prepareChartData(currentBenchmark, selectedBackends, showCpu);
    
    // Destroy old chart
    if (currentChart) {
        currentChart.destroy();
    }
    
    // Get benchmark-specific labels
    const benchmarkConfig = {
        'matrix_mul': {
            title: 'Matrix Multiplication Performance (Naive Kernel)',
            xLabel: 'Matrix Size (elements)',
            yLabel: 'Throughput (GFLOPS)',
            yUnit: 'GFLOPS'
        },
        'vector_add': {
            title: 'Vector Addition Performance (Memory Bandwidth)',
            xLabel: 'Vector Size (elements)',
            yLabel: 'Memory Bandwidth (GB/s)',
            yUnit: 'GB/s'
        }
    };
    
    const config = benchmarkConfig[currentBenchmark] || benchmarkConfig['matrix_mul'];
    
    // Create new chart
    const ctx = canvas.getContext('2d');
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: config.title,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.y.toFixed(2) + ' ' + config.yUnit;
                            return label;
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
                    },
                    ticks: {
                        callback: function(value) {
                            return value;
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: config.yLabel
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Prepare chart data from benchmark results
function prepareChartData(benchmarkName, selectedBackends, showCpu) {
    if (!benchmarkData || !benchmarkData.results) {
        return [];
    }
    
    const datasets = [];
    const deviceData = new Map(); // device_name -> {framework, data: [{x, y}]}
    
    // Map benchmark tab names to JSON benchmark names
    const benchmarkNameMap = {
        'matrix_mul': 'matrix_mul_naive',
        'vector_add': 'vector_add'
    };
    
    const targetBenchmark = benchmarkNameMap[benchmarkName];
    if (!targetBenchmark) return [];
    
    // Process all results
    benchmarkData.results.forEach(result => {
        if (result.benchmark.name !== targetBenchmark) return;
        
        result.results.forEach(deviceResult => {
            const framework = deviceResult.framework;
            
            // Skip if framework not selected
            if (!selectedBackends.includes(framework)) return;
            
            // Skip CPU backends unless explicitly shown
            if (!showCpu && (framework === 'Native' || framework === 'Interpreter')) {
                return;
            }
            
            const deviceName = deviceResult.device_name;
            const key = `${deviceName} (${framework})`;
            
            if (!deviceData.has(key)) {
                deviceData.set(key, {
                    framework: framework,
                    data: []
                });
            }
            
            const size = result.benchmark.parameters.size;
            const throughput = deviceResult.throughput_gflops;
            
            deviceData.get(key).data.push({
                x: size,
                y: throughput
            });
        });
    });
    
    // Convert to Chart.js datasets
    deviceData.forEach((value, deviceName) => {
        // Sort data by size
        value.data.sort((a, b) => a.x - b.x);
        
        datasets.push({
            label: deviceName,
            data: value.data,
            borderColor: BACKEND_COLORS[value.framework] || '#000000',
            backgroundColor: BACKEND_COLORS[value.framework] || '#000000',
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.1
        });
    });
    
    return datasets;
}

// Update system information display
function updateSystemInfo() {
    if (!benchmarkData || !benchmarkData.results || benchmarkData.results.length === 0) {
        return;
    }
    
    const infoDiv = document.getElementById('system-info');
    if (!infoDiv) return;
    
    // Get system info from first result
    const firstResult = benchmarkData.results[0];
    const system = firstResult.system;
    const benchmark = firstResult.benchmark;
    
    let html = '<h4>Test Configuration</h4>';
    html += '<table style="width: 100%; font-family: monospace;">';
    html += `<tr><td><strong>Hostname:</strong></td><td>${system.hostname}</td></tr>`;
    html += `<tr><td><strong>OS:</strong></td><td>${system.os} ${system.kernel}</td></tr>`;
    html += `<tr><td><strong>CPU:</strong></td><td>${system.cpu.model} (${system.cpu.cores} cores)</td></tr>`;
    html += `<tr><td><strong>Memory:</strong></td><td>${system.memory_gb.toFixed(1)} GB</td></tr>`;
    html += `<tr><td><strong>Git Commit:</strong></td><td><code>${benchmark.git_commit.substring(0, 8)}</code></td></tr>`;
    html += `<tr><td><strong>Timestamp:</strong></td><td>${benchmark.timestamp}</td></tr>`;
    html += '</table>';
    
    html += '<h4>Detected Devices</h4>';
    html += '<ul>';
    system.devices.forEach(device => {
        html += `<li><strong>${device.name}</strong> (${device.framework}) - ${device.memory_gb.toFixed(1)} GB`;
        if (device.compute_capability) {
            html += ` - Compute ${device.compute_capability}`;
        }
        html += '</li>';
    });
    html += '</ul>';
    
    infoDiv.innerHTML = html;
}
