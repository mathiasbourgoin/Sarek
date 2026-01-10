// Benchmark viewer with interactive filtering
// Handles tab switching, backend filtering, and chart updates

let benchmarkData = null;
let currentChart = null;
let currentBenchmark = 'matrix_mul';
let currentMetric = 'throughput'; // 'time' or 'throughput'
let currentSystem = 'all'; // 'all' or specific hostname

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
        populateSystemFilter();
        initializeFilters();
        updateChart();
        updateSystemInfo();
    } catch (error) {
        console.error('Error loading benchmark data:', error);
        document.getElementById('system-info').textContent = 
            'Error loading benchmark data. Please check console for details.';
    }
}

// Populate system filter dropdown
function populateSystemFilter() {
    const systemSelect = document.getElementById('system-select');
    if (!systemSelect || !benchmarkData) return;
    
    // Collect unique systems with full identification
    const systemsMap = new Map();
    const systemCounts = new Map(); // Track duplicates
    
    benchmarkData.results.forEach(result => {
        if (result.system && result.system.hostname) {
            const hostname = result.system.hostname;
            
            // Create unique system ID based on hostname + all GPUs
            let systemId = hostname;
            let gpuNames = [];
            
            if (result.system.devices && result.system.devices.length > 0) {
                // Get all GPU devices
                const gpuDevices = result.system.devices.filter(d => 
                    d.framework !== 'Native' && d.framework !== 'Interpreter'
                );
                
                gpuNames = gpuDevices.map(d => {
                    let name = d.name || 'Unknown';
                    // Shorten GPU name
                    return name
                        .replace('Intel(R) ', '')
                        .replace('NVIDIA ', '')
                        .replace('AMD ', '')
                        .replace('(R)', '')
                        .replace('(TM)', '')
                        .replace('(tm)', '')
                        .replace(' Graphics', '')
                        .replace('GeForce ', '')
                        .trim();
                });
                
                // Create unique ID including GPU info
                if (gpuNames.length > 0) {
                    systemId = `${hostname}_${gpuNames.join('_')}`;
                }
            }
            
            if (!systemsMap.has(systemId)) {
                // Create descriptive label
                let label = hostname;
                
                if (gpuNames.length > 0) {
                    if (gpuNames.length === 1) {
                        label += ` (${gpuNames[0]})`;
                    } else {
                        // Multiple GPUs
                        label += ` (${gpuNames.join(' + ')})`;
                    }
                }
                
                // Track if this label already exists (same hostname + GPU combo)
                const baseLabel = label;
                const labelCount = systemCounts.get(baseLabel) || 0;
                systemCounts.set(baseLabel, labelCount + 1);
                
                if (labelCount > 0) {
                    // Add date suffix for duplicates
                    const timestamp = result.benchmark?.timestamp;
                    if (timestamp) {
                        const date = new Date(timestamp);
                        const dateStr = date.toISOString().split('T')[0]; // YYYY-MM-DD
                        label = `${baseLabel} [${dateStr}]`;
                    } else {
                        label = `${baseLabel} #${labelCount + 1}`;
                    }
                }
                
                systemsMap.set(systemId, {
                    hostname: hostname,
                    label: label
                });
            }
        }
    });
    
    // Clear existing options except "All Systems"
    systemSelect.innerHTML = '<option value="all">All Systems</option>';
    
    // Add each system with descriptive label
    Array.from(systemsMap.entries())
        .sort((a, b) => a[1].label.localeCompare(b[1].label))
        .forEach(([systemId, info]) => {
            const option = document.createElement('option');
            option.value = info.hostname; // Still use hostname as value for filtering
            option.textContent = info.label;
            option.dataset.systemId = systemId; // Store full ID for reference
            systemSelect.appendChild(option);
        });
    
    // Add event listener
    systemSelect.addEventListener('change', (e) => {
        currentSystem = e.target.value;
        updateChart();
    });
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
    
    // Metric selector
    const metricSelector = document.getElementById('metric-select');
    if (metricSelector) {
        metricSelector.addEventListener('change', (e) => {
            currentMetric = e.target.value;
            updateChart();
        });
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
    
    // Get benchmark-specific labels and units
    const benchmarkConfig = {
        'matrix_mul': {
            title: 'Matrix Multiplication Performance (Naive Kernel)',
            xLabel: 'Matrix Size (elements)',
            throughputLabel: 'Throughput (GFLOPS)',
            throughputUnit: 'GFLOPS',
            timeLabel: 'Execution Time (ms)',
            timeUnit: 'ms'
        },
        'vector_add': {
            title: 'Vector Addition Performance (Memory Bandwidth)',
            xLabel: 'Vector Size (elements)',
            throughputLabel: 'Memory Bandwidth (GB/s)',
            throughputUnit: 'GB/s',
            timeLabel: 'Execution Time (ms)',
            timeUnit: 'ms'
        },
        'reduction': {
            title: 'Parallel Reduction Performance (Sum)',
            xLabel: 'Array Size (elements)',
            throughputLabel: 'Memory Bandwidth (GB/s)',
            throughputUnit: 'GB/s',
            timeLabel: 'Execution Time (ms)',
            timeUnit: 'ms'
        },
        'transpose': {
            title: 'Matrix Transpose Performance (Naive Kernel)',
            xLabel: 'Matrix Size (NxN)',
            throughputLabel: 'Memory Bandwidth (GB/s)',
            throughputUnit: 'GB/s',
            timeLabel: 'Execution Time (ms)',
            timeUnit: 'ms'
        },
        'transpose_tiled': {
            title: 'Matrix Transpose Performance (Tiled with Shared Memory)',
            xLabel: 'Matrix Size (NxN)',
            throughputLabel: 'Memory Bandwidth (GB/s)',
            throughputUnit: 'GB/s',
            timeLabel: 'Execution Time (ms)',
            timeUnit: 'ms'
        }
    };
    
    const config = benchmarkConfig[currentBenchmark] || benchmarkConfig['matrix_mul'];
    
    // Determine labels based on current metric
    const yLabel = currentMetric === 'time' ? config.timeLabel : config.throughputLabel;
    const yUnit = currentMetric === 'time' ? config.timeUnit : config.throughputUnit;
    
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
                            label += context.parsed.y.toFixed(2) + ' ' + yUnit;
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
                        text: yLabel
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
        'vector_add': 'vector_add',
        'reduction': 'reduction_sum',
        'transpose': 'transpose_naive',
        'transpose_tiled': 'transpose_tiled'
    };
    
    const targetBenchmark = benchmarkNameMap[benchmarkName];
    if (!targetBenchmark) return [];
    
    // Process all results
    benchmarkData.results.forEach(result => {
        // Skip malformed results
        if (!result || !result.benchmark || !result.benchmark.name) {
            console.warn('Skipping malformed result:', result);
            return;
        }
        
        if (result.benchmark.name !== targetBenchmark) return;
        
        // Filter by system if not "all"
        if (currentSystem !== 'all') {
            if (!result.system || result.system.hostname !== currentSystem) {
                return;
            }
        }
        
        // Skip if no results array
        if (!result.results || !Array.isArray(result.results)) {
            console.warn('Result has no results array:', result);
            return;
        }
        
        result.results.forEach(deviceResult => {
            const framework = deviceResult.framework;
            
            // Skip if framework not selected
            if (!selectedBackends.includes(framework)) return;
            
            // Skip CPU backends unless explicitly shown
            if (!showCpu && (framework === 'Native' || framework === 'Interpreter')) {
                return;
            }
            
            const deviceName = deviceResult.device_name;
            
            // Create system suffix for "All Systems" view
            let systemSuffix = '';
            if (currentSystem === 'all' && result.system && result.system.hostname) {
                // Shorten GPU name for compact display
                let gpuName = deviceName
                    .replace('Intel(R) ', '')
                    .replace('(R)', '')
                    .replace('(TM)', '')
                    .replace('(tm)', '')
                    .replace(' Graphics', '')
                    .trim();
                systemSuffix = ` @ ${result.system.hostname}`;
            }
            
            const key = `${deviceName} (${framework})${systemSuffix}`;
            
            if (!deviceData.has(key)) {
                deviceData.set(key, {
                    framework: framework,
                    data: []
                });
            }
            
            const size = result.benchmark.parameters.size;
            
            // Get the appropriate metric value
            let yValue;
            if (currentMetric === 'time') {
                // Use minimum time for best performance representation
                yValue = deviceResult.min_ms;
            } else {
                // Use throughput (GFLOPS or GB/s depending on benchmark)
                yValue = deviceResult.throughput_gflops || deviceResult.throughput || 0;
            }
            
            deviceData.get(key).data.push({
                x: size,
                y: yValue
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
    
    // Collect all unique systems from results
    const systemsMap = new Map();
    benchmarkData.results.forEach(result => {
        if (result.system && result.system.hostname) {
            const hostname = result.system.hostname;
            if (!systemsMap.has(hostname)) {
                systemsMap.set(hostname, {
                    system: result.system,
                    benchmark: result.benchmark,
                    count: 0
                });
            }
            systemsMap.get(hostname).count++;
        }
    });
    
    if (systemsMap.size === 0) {
        infoDiv.innerHTML = '<p style="color: #888;">System information not available.</p>';
        return;
    }
    
    let html = '<h4>Test Systems</h4>';
    html += `<p style="color: #666; font-size: 0.9em;">Data collected from ${systemsMap.size} system${systemsMap.size > 1 ? 's' : ''}</p>`;
    
    // Show each system
    Array.from(systemsMap.entries()).forEach(([hostname, data], index) => {
        const system = data.system;
        const benchmark = data.benchmark;
        
        if (index > 0) html += '<hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">';
        
        html += '<table style="width: 100%; font-family: monospace; font-size: 0.9em;">';
        html += `<tr><td><strong>Hostname:</strong></td><td>${system.hostname || 'N/A'}</td></tr>`;
        html += `<tr><td><strong>OS:</strong></td><td>${system.os || 'N/A'} ${system.kernel || ''}</td></tr>`;
        html += `<tr><td><strong>CPU:</strong></td><td>${system.cpu?.model || 'N/A'} (${system.cpu?.cores || 'N/A'} cores)</td></tr>`;
        html += `<tr><td><strong>Memory:</strong></td><td>${system.memory_gb ? system.memory_gb.toFixed(1) : 'N/A'} GB</td></tr>`;
        if (benchmark) {
            html += `<tr><td><strong>Results:</strong></td><td>${data.count} benchmark${data.count > 1 ? 's' : ''}</td></tr>`;
        }
        html += '</table>';
        
        if (system.devices && system.devices.length > 0) {
            html += '<p style="margin: 10px 0 5px 0;"><strong>Devices:</strong></p>';
            html += '<ul style="margin: 5px 0;">';
            system.devices.forEach(device => {
                html += `<li><strong>${device.name || 'Unknown'}</strong> (${device.framework || 'N/A'})`;
                if (device.memory_gb) {
                    html += ` - ${device.memory_gb.toFixed(1)} GB`;
                }
                if (device.compute_capability) {
                    html += ` - Compute ${device.compute_capability}`;
                }
                html += '</li>';
            });
            html += '</ul>';
        }
    });
    
    infoDiv.innerHTML = html;
}
