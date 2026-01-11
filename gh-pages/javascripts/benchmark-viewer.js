// Unified benchmark viewer with multiple view modes
// Supports single chart, 4-panel comparison, ranking, and matrix views

let benchmarkData = null;
let currentChart = null;
let charts = {
    scaling: null,
    comparison: null,
    backends: null,
    speedup: null,
    ranking: null
};
let currentBenchmark = 'matrix_mul';
let currentMetric = 'throughput'; // 'time' or 'throughput'
let currentSystem = 'all'; // 'all' or specific hostname
let currentViewMode = 'single'; // 'single', 'comparison', 'ranking', 'matrix'

// Color palette for different backends
const BACKEND_COLORS = {
    'CUDA': '#76b900',      // NVIDIA green
    'OpenCL': '#0071c5',    // Intel blue
    'Vulkan': '#ac162c',    // Vulkan red
    'Metal': '#555555',     // Apple gray
    'Native': '#ff9500',    // Orange
    'Interpreter': '#999999' // Gray
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
        variants: ['matrix_mul_naive'],
        readme: 'descriptions/matrix_mul.md'
    },
    'vector_add': {
        title: 'Vector Addition',
        xLabel: 'Vector Size (elements)',
        throughputLabel: 'GB/s',
        variants: ['vector_add'],
        readme: 'descriptions/vector_add.md'
    },
    'reduction': {
        title: 'Parallel Reduction',
        xLabel: 'Array Size (elements)',
        throughputLabel: 'GB/s',
        variants: ['reduction_sum'],
        readme: 'descriptions/reduction.md'
    },
    'transpose': {
        title: 'Matrix Transpose',
        xLabel: 'Matrix Size (N×N)',
        throughputLabel: 'GB/s',
        variants: ['transpose_naive', 'transpose_tiled'],
        readme: 'descriptions/transpose.md'
    },
    'transpose_tiled': {
        title: 'Matrix Transpose (Tiled)',
        xLabel: 'Matrix Size (N×N)',
        throughputLabel: 'GB/s',
        variants: ['transpose_tiled'],
        readme: 'descriptions/transpose.md'
    },
    'mandelbrot': {
        title: 'Mandelbrot Set',
        xLabel: 'Resolution (pixels)',
        throughputLabel: 'Mpixels/s',
        variants: ['mandelbrot'],
        readme: 'descriptions/mandelbrot.md'
    }
};

// Load benchmark data from JSON
async function loadBenchmarkData(dataUrl) {
    try {
        const response = await fetch(dataUrl);
        benchmarkData = await response.json();
        
        initializeViewModeSelector();
        initializeBenchmarkSelector();
        populateSystemFilter();
        initializeFilters();
        updateView();
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
        updateView();
    });
}

// Initialize view mode selector
function initializeViewModeSelector() {
    const viewModeSelect = document.getElementById('view-mode-select');
    if (!viewModeSelect) return;
    
    viewModeSelect.addEventListener('change', (e) => {
        currentViewMode = e.target.value;
        updateView();
    });
}

// Initialize benchmark selector
function initializeBenchmarkSelector() {
    const selector = document.getElementById('benchmark-select');
    if (!selector) return;
    
    selector.addEventListener('change', (e) => {
        currentBenchmark = e.target.value;
        updateBenchmarkDescription();
        updateView();
    });
    
    // Show initial description
    updateBenchmarkDescription();
}

// Process generated code tabs markers in markdown
async function processGeneratedCodeTabs(markdown) {
    // Find all markers like <!-- GENERATED_CODE_TABS: benchmark_name -->
    const markerRegex = /<!--\s*GENERATED_CODE_TABS:\s*(\w+)\s*-->/g;
    const matches = [...markdown.matchAll(markerRegex)];
    
    for (const match of matches) {
        const benchmarkName = match[1];
        const marker = match[0];
        
        try {
            // Fetch the generated code file
            const generatedFile = `descriptions/generated/${benchmarkName}_generated.md`;
            const response = await fetch(generatedFile);
            
            if (response.ok) {
                const generatedMarkdown = await response.text();
                const tabsHtml = createGeneratedCodeTabs(generatedMarkdown);
                markdown = markdown.replace(marker, tabsHtml);
            } else {
                // Replace with error message if file not found
                markdown = markdown.replace(marker, '*Generated code not available yet.*');
            }
        } catch (error) {
            console.error(`Error loading generated code for ${benchmarkName}:`, error);
            markdown = markdown.replace(marker, '*Error loading generated code.*');
        }
    }
    
    return markdown;
}

// Create tabbed interface from generated code markdown
function createGeneratedCodeTabs(generatedMarkdown) {
    // Parse the generated markdown to extract code sections
    const backends = ['CUDA C', 'OpenCL C', 'Vulkan GLSL', 'Metal'];
    const codeBlocks = {};
    const languages = {
        'CUDA C': 'cuda',
        'OpenCL C': 'opencl',
        'Vulkan GLSL': 'glsl',
        'Metal': 'metal'
    };
    
    backends.forEach(backend => {
        const regex = new RegExp(`## ${backend}\\s*\\n\\s*\`\`\`\\w*\\n([\\s\\S]*?)\`\`\``, 'i');
        const match = generatedMarkdown.match(regex);
        if (match) {
            codeBlocks[backend] = match[1].trim();
        }
    });
    
    // Generate unique ID for this tabs instance
    const tabsId = `tabs-${Math.random().toString(36).substr(2, 9)}`;
    
    // Create tabs HTML
    let html = '<div class="code-tabs-container" style="margin: 20px 0;">';
    html += '<h3 style="color: var(--link-color); margin-bottom: 10px;">Generated Backend Code</h3>';
    html += `<div class="code-tabs" style="display: flex; gap: 5px; margin-bottom: 0; border-bottom: 2px solid var(--border-color);">`;
    
    backends.forEach((backend, index) => {
        const tabId = `${tabsId}-${index}`;
        const isActive = index === 0 ? ' active' : '';
        html += `<button class="code-tab${isActive}" data-tabs-id="${tabsId}" data-tab-index="${index}" 
                 style="padding: 10px 20px; border: none; background: ${index === 0 ? 'var(--link-color)' : 'transparent'}; 
                 color: ${index === 0 ? 'white' : 'var(--text-color)'}; cursor: pointer; font-weight: 600; 
                 border-radius: 4px 4px 0 0; transition: all 0.2s;"
                 onmouseover="if(!this.classList.contains('active')) this.style.background='var(--code-bg)';"
                 onmouseout="if(!this.classList.contains('active')) this.style.background='transparent';">
            ${backend}
        </button>`;
    });
    
    html += '</div>';
    html += '<div class="code-panels">';
    
    backends.forEach((backend, index) => {
        const isActive = index === 0 ? 'block' : 'none';
        const code = codeBlocks[backend] || '// Code not available';
        const lang = languages[backend];
        html += `<div class="code-panel" data-tabs-id="${tabsId}" data-panel-index="${index}" style="display: ${isActive};">`;
        html += `<pre style="margin: 0; padding: 15px; background: var(--bg-color); border: 1px solid var(--border-color); border-top: none; border-radius: 0 0 4px 4px; overflow-x: auto; font-size: 0.85em; line-height: 1.4;"><code class="language-${lang}">${escapeHtml(code)}</code></pre>`;
        html += '</div>';
    });
    
    html += '</div></div>';
    
    return html;
}

// Handle tab clicks (will be set up after DOM loads)
function setupTabClickHandlers() {
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('code-tab')) {
            const tabsId = e.target.dataset.tabsId;
            const tabIndex = e.target.dataset.tabIndex;
            
            // Update tabs
            const tabs = document.querySelectorAll(`[data-tabs-id="${tabsId}"].code-tab`);
            tabs.forEach((tab, index) => {
                if (index == tabIndex) {
                    tab.classList.add('active');
                    tab.style.background = 'var(--link-color)';
                    tab.style.color = 'white';
                } else {
                    tab.classList.remove('active');
                    tab.style.background = 'transparent';
                    tab.style.color = 'var(--text-color)';
                }
            });
            
            // Update panels
            const panels = document.querySelectorAll(`[data-tabs-id="${tabsId}"].code-panel`);
            panels.forEach((panel, index) => {
                panel.style.display = index == tabIndex ? 'block' : 'none';
            });
        }
    });
}

// Update benchmark description panel
async function updateBenchmarkDescription() {
    const descDiv = document.getElementById('benchmark-description');
    if (!descDiv) return;
    
    const config = BENCHMARK_CONFIGS[currentBenchmark];
    if (!config) return;
    
    // Show loading state
    descDiv.innerHTML = '<p style="color: #888;"><em>Loading description...</em></p>';
    
    try {
        // Fetch the markdown file
        const response = await fetch(config.readme);
        if (!response.ok) {
            throw new Error(`Failed to load: ${response.statusText}`);
        }
        let markdown = await response.text();
        
        // Process generated code tabs markers
        markdown = await processGeneratedCodeTabs(markdown);
        
        // Convert markdown to HTML (simple conversion)
        const html = markdownToHtml(markdown);
        descDiv.innerHTML = html;
        
        // Apply syntax highlighting if Prism is available
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(descDiv);
        }
    } catch (error) {
        console.error('Error loading benchmark description:', error);
        descDiv.innerHTML = `
            <h3 style="margin-top: 0; color: var(--link-color);">${config.title}</h3>
            <p style="color: #888;"><em>Description not available yet.</em></p>
        `;
    }
}

// Simple markdown to HTML converter
function markdownToHtml(markdown) {
    let html = markdown;
    
    // Headers
    html = html.replace(/^### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 style="color: var(--link-color);">$1</h2>');
    
    // Code blocks FIRST (before bold/italic to avoid issues with comments)
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
        const language = lang || 'ocaml';
        return `<pre style="margin: 15px 0; padding: 15px; background: var(--bg-color); border: 1px solid var(--border-color); border-radius: 4px; overflow-x: auto; font-size: 0.85em; line-height: 1.4;"><code class="language-${language}">${escapeHtml(code.trim())}</code></pre>`;
    });
    
    // Inline code (before bold/italic)
    html = html.replace(/`([^`]+)`/g, '<code style="background: var(--code-bg); padding: 2px 6px; border-radius: 3px; font-size: 0.9em;">$1</code>');
    
    // Bold and italic (after code) - more restrictive to avoid matching comment syntax
    html = html.replace(/\*\*\*([^\*\n]+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*([^\*\n]+?)\*\*/g, '<strong>$1</strong>');
    // Only match italic if not preceded by ( or followed by ) to avoid (* comments *)
    html = html.replace(/(?<!\()\*([^\*\n]+?)\*(?!\))/g, '<em>$1</em>');
    
    // Images
    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 100%; height: auto; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" style="color: var(--link-color); text-decoration: underline;">$1</a>');
    
    // Tables
    html = html.replace(/\n\|(.+)\|\n\|[-\s|]+\|\n((?:\|.+\|\n?)+)/g, function(match, header, rows) {
        let table = '<table style="width: 100%; border-collapse: collapse; margin: 15px 0;">';
        table += '<thead><tr>';
        header.split('|').slice(1, -1).forEach(cell => {
            table += `<th style="padding: 10px; text-align: left; border: 1px solid var(--border-color); background: var(--code-bg); font-weight: 600;">${cell.trim()}</th>`;
        });
        table += '</tr></thead><tbody>';
        rows.trim().split('\n').forEach(row => {
            table += '<tr>';
            row.split('|').slice(1, -1).forEach(cell => {
                table += `<td style="padding: 10px; border: 1px solid var(--border-color);">${cell.trim()}</td>`;
            });
            table += '</tr>';
        });
        table += '</tbody></table>';
        return table;
    });
    
    // Lists
    html = html.replace(/^\- (.+)$/gim, '<li>$1</li>');
    html = html.replace(/^\d+\. (.+)$/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul style="margin: 10px 0; padding-left: 30px;">$1</ul>');
    
    // Paragraphs
    html = html.split('\n\n').map(para => {
        para = para.trim();
        if (!para) return '';
        if (para.startsWith('<')) return para; // Already HTML
        return `<p style="margin: 12px 0; line-height: 1.6;">${para}</p>`;
    }).join('\n');
    
    return html;
}

// Helper to escape HTML in code
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize filter controls
function initializeFilters() {
    // Backend filters
    const backendFilters = document.querySelectorAll('.backend-filter');
    backendFilters.forEach(filter => {
        filter.addEventListener('change', updateView);
    });
    
    // CPU baseline filter
    const cpuFilter = document.getElementById('show-cpu');
    if (cpuFilter) {
        cpuFilter.addEventListener('change', updateView);
    }
    
    // Metric selector
    const metricSelector = document.getElementById('metric-select');
    if (metricSelector) {
        metricSelector.addEventListener('change', (e) => {
            currentMetric = e.target.value;
            updateView();
        });
    }
}

// Main update function - dispatches to correct view
function updateView() {
    if (!benchmarkData) return;
    
    // Destroy all existing charts
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
    Object.keys(charts).forEach(key => {
        if (charts[key]) {
            charts[key].destroy();
            charts[key] = null;
        }
    });
    
    if (currentViewMode === 'single') {
        updateChart();
    } else if (currentViewMode === 'comparison') {
        showComparisonView();
    } else if (currentViewMode === 'ranking') {
        showSystemRanking();
    } else if (currentViewMode === 'matrix') {
        showDeviceMatrix();
    }
}

// Update chart based on current filters
function updateChart() {
    if (!benchmarkData) return;
    
    // Restore single chart layout if needed
    restoreSingleChartView();
    
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
        } else {
            // When showing "all systems", filter out test systems
            if (result.system && result.system.hostname && 
                result.system.hostname.toLowerCase().includes('test')) {
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

// ============================================================================
// MULTI-VIEW FUNCTIONS (Comparison, Ranking, Matrix)
// ============================================================================

// Show 4-chart comparison view
function showComparisonView() {
    const config = BENCHMARK_CONFIGS[currentBenchmark] || BENCHMARK_CONFIGS['transpose'];
    const chartArea = document.getElementById('chart-area');
    if (!chartArea) return;
    
    // Hide filters for multi-view (they don't apply)
    const filterControls = document.querySelector('.filter-controls');
    if (filterControls) filterControls.style.display = 'none';
    
    chartArea.innerHTML = `
        <div class="dashboard-grid-4">
            <div class="chart-card">
                <h4>Performance Scaling</h4>
                <p style="font-size: 0.85em; color: #888; margin: 5px 0;">All sizes, all backends</p>
                <canvas id="chart-scaling"></canvas>
            </div>
            <div class="chart-card">
                <h4>Algorithm Comparison</h4>
                <p style="font-size: 0.85em; color: #888; margin: 5px 0;">At target size (shown in chart)</p>
                <canvas id="chart-comparison"></canvas>
            </div>
            <div class="chart-card">
                <h4>Backend Comparison</h4>
                <p style="font-size: 0.85em; color: #888; margin: 5px 0;">At target size (shown in chart)</p>
                <canvas id="chart-backends"></canvas>
            </div>
            <div class="chart-card">
                <h4>Speedup Analysis</h4>
                <p style="font-size: 0.85em; color: #888; margin: 5px 0;">Optimized vs baseline across sizes</p>
                <canvas id="chart-speedup"></canvas>
            </div>
        </div>
    `;
    
    // Create all 4 charts
    createScalingChart(config);
    createComparisonChart(config);
    createBackendChart(config);
    createSpeedupChart(config);
}

// Show system ranking view
function showSystemRanking() {
    const config = BENCHMARK_CONFIGS[currentBenchmark] || BENCHMARK_CONFIGS['transpose'];
    const chartArea = document.getElementById('chart-area');
    if (!chartArea) return;
    
    // Hide filters
    const filterControls = document.querySelector('.filter-controls');
    if (filterControls) filterControls.style.display = 'none';
    
    chartArea.innerHTML = `
        <div class="dashboard-grid-1">
            <div class="chart-card large">
                <h4>System Ranking: ${config.title}</h4>
                <p style="color: var(--text-color); opacity: 0.8; margin-bottom: 15px;">
                    Systems ranked by peak performance across all problem sizes.
                </p>
                <canvas id="chart-ranking"></canvas>
            </div>
        </div>
    `;
    
    createRankingChart(config);
}

// Show device matrix view
function showDeviceMatrix() {
    const config = BENCHMARK_CONFIGS[currentBenchmark] || BENCHMARK_CONFIGS['transpose'];
    const chartArea = document.getElementById('chart-area');
    if (!chartArea) return;
    
    // Hide filters
    const filterControls = document.querySelector('.filter-controls');
    if (filterControls) filterControls.style.display = 'none';
    
    chartArea.innerHTML = `
        <div class="dashboard-grid-1">
            <div class="chart-card large">
                <h4>Device × Algorithm Performance Matrix</h4>
                <div id="matrix-container"></div>
            </div>
        </div>
    `;
    
    createDeviceMatrix(config);
}

// Restore single chart view
function restoreSingleChartView() {
    const chartArea = document.getElementById('chart-area');
    if (!chartArea) return;
    
    // Show filters again
    const filterControls = document.querySelector('.filter-controls');
    if (filterControls) filterControls.style.display = 'flex';
    
    chartArea.innerHTML = `
        <div class="chart-container">
            <canvas id="matrixMulChart"></canvas>
        </div>
    `;
}

// Chart 1: Line chart showing performance scaling with size
function createScalingChart(config) {
    const ctx = document.getElementById('chart-scaling');
    if (!ctx) return;
    
    const datasets = [];
    const variants = config.variants;
    
    // Collect data for ALL device/backend combinations
    const deviceBackendMap = new Map(); // Key: "deviceName|backend|variantName"
    
    variants.forEach((variantName) => {
        const variantResults = benchmarkData.results.filter(r => 
            r.benchmark && r.benchmark.name === variantName && shouldIncludeResult(r)
        );
        
        if (variantResults.length === 0) return;
        
        // Process all device results for this variant
        variantResults.forEach(result => {
            if (!result.results || !Array.isArray(result.results)) return;
            
            result.results.forEach(deviceResult => {
                const deviceName = deviceResult.device_name || 'Unknown';
                const backend = deviceResult.framework || 'Unknown';
                const key = `${deviceName}|${backend}|${variantName}`;
                
                if (!deviceBackendMap.has(key)) {
                    deviceBackendMap.set(key, {
                        deviceName,
                        backend,
                        variantName,
                        data: []
                    });
                }
                
                const size = getProblemSize(result.benchmark);
                const throughput = deviceResult.throughput_gflops || 0;
                
                deviceBackendMap.get(key).data.push({ x: size, y: throughput });
            });
        });
    });
    
    // Create datasets from collected data
    deviceBackendMap.forEach(({ deviceName, backend, variantName, data }) => {
        // Sort by size
        data.sort((a, b) => a.x - b.x);
        
        const algoType = variantName.includes('tiled') ? 'tiled' : 
                         variantName.includes('naive') ? 'naive' : 'baseline';
        
        // Shorten device name
        const shortDevice = deviceName
            .replace('Intel(R) ', '')
            .replace('(R)', '')
            .replace('(TM)', '')
            .replace('(tm)', '')
            .replace(' Graphics', '')
            .trim();
        
        datasets.push({
            label: `${formatAlgoName(variantName)} - ${shortDevice} (${backend})`,
            data: data,
            borderColor: ALGO_COLORS[algoType],
            backgroundColor: ALGO_COLORS[algoType] + '20',
            borderWidth: 2,
            tension: 0.1,
            fill: false
        });
    });
    
    if (datasets.length === 0) {
        ctx.parentElement.innerHTML = '<p class="no-data">No scaling data available</p>';
        return;
    }
    
    charts.scaling = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: { size: 10 }
                    }
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
            r.benchmark && r.benchmark.name === variantName && shouldIncludeResult(r)
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
            getProblemSize(r.benchmark) === targetSize &&
            shouldIncludeResult(r)
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
        r.benchmark && r.benchmark.name === variantName && shouldIncludeResult(r)
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
    
    // Group by backend and device - FIXED: iterate through ALL device results
    const backendData = {};
    const deviceLabels = [];
    const deviceBackendMap = new Map(); // Track which devices have which backends
    
    results
        .filter(r => getProblemSize(r.benchmark) === targetSize)
        .forEach(result => {
            // Iterate through ALL device results, not just results[0]
            if (!result.results || !Array.isArray(result.results)) return;
            
            result.results.forEach(deviceResult => {
                const backend = deviceResult.framework || 'Unknown';
                const device = deviceResult.device_name || 'Unknown Device';
                
                // Track unique device-backend combinations
                const deviceKey = `${device}`;
                if (!deviceLabels.includes(deviceKey)) {
                    deviceLabels.push(deviceKey);
                    deviceBackendMap.set(deviceKey, new Map());
                }
                
                // Store performance for this device-backend combination
                const deviceMap = deviceBackendMap.get(deviceKey);
                deviceMap.set(backend, deviceResult.throughput_gflops || 0);
                
                // Initialize backend dataset if not exists
                if (!backendData[backend]) {
                    backendData[backend] = {
                        label: backend,
                        data: [],
                        backgroundColor: BACKEND_COLORS[backend] || '#999999',
                        borderWidth: 1
                    };
                }
            });
        });
    
    // Fill data arrays for each backend across all devices
    Object.keys(backendData).forEach(backend => {
        deviceLabels.forEach(device => {
            const deviceMap = deviceBackendMap.get(device);
            const value = deviceMap && deviceMap.has(backend) ? deviceMap.get(backend) : 0;
            backendData[backend].data.push(value);
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
        r.benchmark && r.benchmark.name === baselineVariant && shouldIncludeResult(r)
    );
    const optimizedResults = benchmarkData.results.filter(r => 
        r.benchmark && r.benchmark.name === optimizedVariant && shouldIncludeResult(r)
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
            r.benchmark && r.benchmark.name === variantName && shouldIncludeResult(r)
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
    const deviceBackendSet = new Map(); // Map device+backend to unique key
    
    // FIXED: Iterate through ALL device results, not just results[0]
    benchmarkData.results
        .filter(r => r.benchmark && variants.includes(r.benchmark.name) && shouldIncludeResult(r))
        .forEach(result => {
            if (!result.results || !Array.isArray(result.results)) return;
            
            result.results.forEach(deviceResult => {
                const deviceName = deviceResult.device_name || 'Unknown Device';
                const backend = deviceResult.framework || 'Unknown';
                const key = `${deviceName} (${backend})`;
                deviceBackendSet.set(key, { device: deviceName, backend: backend });
            });
        });
    
    const devices = Array.from(deviceBackendSet.keys());
    
    if (devices.length === 0) {
        container.innerHTML = '<p class="no-data">No device data available</p>';
        return;
    }
    
    let html = '<p style="font-size: 0.9em; color: #888; margin-bottom: 15px;">Peak performance across all problem sizes (in ${config.throughputLabel})</p>';
    html += '<table class="perf-matrix"><thead><tr><th>Device</th>';
    variants.forEach(v => {
        html += `<th>${formatAlgoName(v)}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    devices.forEach(deviceKey => {
        const { device, backend } = deviceBackendSet.get(deviceKey);
        html += `<tr><td class="device-name">${deviceKey}</td>`;
        
        variants.forEach(variantName => {
            // Find all results for this variant
            const matchingResults = benchmarkData.results.filter(r => 
                r.benchmark && 
                r.benchmark.name === variantName &&
                r.results && 
                Array.isArray(r.results) &&
                shouldIncludeResult(r)
            );
            
            // Find the specific device result
            let maxPerf = 0;
            matchingResults.forEach(r => {
                r.results.forEach(deviceResult => {
                    if (deviceResult.device_name === device && deviceResult.framework === backend) {
                        const perf = deviceResult.throughput_gflops || 0;
                        maxPerf = Math.max(maxPerf, perf);
                    }
                });
            });
            
            if (maxPerf > 0) {
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
    // FIXED: Iterate through ALL device results, not just results[0]
    const systemPerformances = [];
    
    benchmarkData.results
        .filter(r => r.benchmark && variants.includes(r.benchmark.name) && shouldIncludeResult(r))
        .forEach(result => {
            const systemName = result.system?.hostname || 'Unknown';
            
            // Iterate through ALL device results
            if (!result.results || !Array.isArray(result.results)) return;
            
            result.results.forEach(deviceResult => {
                const deviceName = deviceResult.device_name || 'Unknown Device';
                const backend = deviceResult.framework || 'Unknown';
                const algoName = formatAlgoName(result.benchmark.name);
                const throughput = deviceResult.throughput_gflops || 0;
                
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
        });
    
    // Check if we have data
    if (systemPerformances.length === 0) {
        console.warn('System ranking: No performances collected');
        console.log('Filtered results count:', benchmarkData.results.filter(r => r.benchmark && variants.includes(r.benchmark.name) && shouldIncludeResult(r)).length);
        ctx.parentElement.innerHTML = '<p class="no-data">No ranking data available for selected system. Try selecting "All Systems" in the filter.</p>';
        return;
    }
    
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
function shouldIncludeResult(result) {
    // Filter by currentSystem
    if (currentSystem !== 'all') {
        if (!result.system || result.system.hostname !== currentSystem) {
            return false;
        }
    } else {
        // When showing "all systems", filter out test systems
        if (result.system && result.system.hostname && 
            result.system.hostname.toLowerCase().includes('test')) {
            return false;
        }
    }
    return true;
}

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
    setupTabClickHandlers();
});
