# Benchmark Output Formats

Support two output formats for different audiences:

1. **Publication Output** (LaTeX + gnuplot) - For academic papers
2. **Web Output** (HTML + JavaScript) - For GitHub Pages documentation

## 1. Publication Output (LaTeX + Gnuplot)

### Gnuplot Script Generator

**Tool:** `plot_gnuplot.exe`

**Features:**
- Generate `.gp` scripts that can be run with `gnuplot`
- Publication-quality defaults (PDF/EPS output, proper fonts)
- Multiple plot types:
  - Throughput vs Size (line plots)
  - Speedup comparison (bar charts)
  - Backend comparison (grouped bars)
  - Scaling analysis (log-log plots)
  - Roofline plots (if applicable)

**Example usage:**
```bash
# Generate throughput plot
dune exec benchmarks/plot_gnuplot.exe -- \
  --input results.csv \
  --type throughput \
  --output matrix_mul.gp

# Run gnuplot to generate PDF
gnuplot matrix_mul.gp
# Creates: matrix_mul.pdf
```

**Example gnuplot output:**
```gnuplot
set terminal pdfcairo enhanced color font 'Times,12' size 6,4
set output 'matrix_mul_throughput.pdf'
set xlabel 'Matrix Size'
set ylabel 'Throughput (GFLOPS)'
set logscale x 2
set key top left
set grid
plot 'results.csv' using 1:2 with linespoints title 'CUDA RTX 4090' lw 2 pt 7, \
     'results.csv' using 1:3 with linespoints title 'OpenCL Arc A770' lw 2 pt 5, \
     'results.csv' using 1:4 with linespoints title 'Vulkan AMD' lw 2 pt 9
```

### LaTeX Table Generator

**Tool:** `to_latex.exe`

**Features:**
- Read CSV or JSON results
- Multiple table styles:
  - Performance comparison (device × benchmark)
  - Speedup matrix
  - System specifications table
- Auto-formatting (bold max values, highlight regressions)
- Include git commit and system info as footnotes

**Example usage:**
```bash
dune exec benchmarks/to_latex.exe -- \
  --input aggregated_results.json \
  --type performance \
  --output results_table.tex

# Or create all tables at once
dune exec benchmarks/to_latex.exe -- \
  --input aggregated_results.json \
  --all \
  --output-dir latex_tables/
```

**Example LaTeX output:**
```latex
\begin{table}[ht]
\centering
\caption{Matrix Multiplication Performance (GFLOPS) across different backends}
\label{tab:matmul_performance}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Backend & 512$^2$ & 1024$^2$ & 2048$^2$ & 4096$^2$ \\
\midrule
CUDA RTX 4090     & 245.3  & 512.7  & 892.4  & \textbf{1205.6} \\
OpenCL Arc A770   & 123.4  & 234.5  & 445.6  & 678.9 \\
Vulkan AMD RX7900 & 198.7  & 389.2  & 723.4  & 989.3 \\
Metal M3 Max      & 156.2  & 298.4  & 534.2  & 712.8 \\
\midrule
CPU Baseline      & 12.3   & 23.4   & 45.6   & 78.9 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\item System: cachyos, Intel Core Ultra 9 185H, 32GB RAM
\item Git commit: 0a964f2d, Date: 2026-01-10
\end{tablenotes}
\end{table}
```

## 2. Web Output (HTML + JavaScript)

### Interactive JavaScript Charts

**Tool:** `plot_web.exe`

**JavaScript Libraries to Use:**
- **Chart.js** - Simple, clean charts (recommended for start)
- **Plotly.js** - Interactive plots with zoom/pan
- **D3.js** - Maximum flexibility (more complex)
- **ApexCharts** - Modern, responsive charts

**Recommendation:** Start with **Chart.js** - it's lightweight, easy to use, and looks great.

### Output Format

Generate standalone HTML files or JSON data + HTML template:

#### Option A: Self-contained HTML
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }
        .chart-container { max-width: 900px; margin: 40px auto; }
    </style>
</head>
<body>
    <div class="chart-container">
        <canvas id="matrixMulChart"></canvas>
    </div>
    <script>
        const ctx = document.getElementById('matrixMulChart');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['256', '512', '1024', '2048', '4096'],
                datasets: [{
                    label: 'CUDA RTX 4090',
                    data: [245.3, 512.7, 892.4, 1205.6, 1456.2],
                    borderColor: 'rgb(76, 175, 80)',
                    tension: 0.1
                }, {
                    label: 'OpenCL Arc A770',
                    data: [123.4, 234.5, 445.6, 678.9, 823.4],
                    borderColor: 'rgb(33, 150, 243)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Matrix Multiplication Performance (GFLOPS)'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Matrix Size'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'GFLOPS'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    </script>
</body>
</html>
```

#### Option B: JSON Data + Template
Generate `benchmarks_data.json`:
```json
{
  "charts": [
    {
      "id": "matrix_mul_throughput",
      "type": "line",
      "title": "Matrix Multiplication Performance",
      "xlabel": "Matrix Size",
      "ylabel": "GFLOPS",
      "datasets": [
        {
          "label": "CUDA RTX 4090",
          "data": [
            {"x": 256, "y": 245.3},
            {"x": 512, "y": 512.7},
            {"x": 1024, "y": 892.4}
          ],
          "color": "#4CAF50"
        }
      ]
    }
  ]
}
```

Then include in gh-pages with a JavaScript loader.

### GitHub Pages Integration

Create a benchmarks page in gh-pages:

**File:** `gh-pages/benchmarks/index.md`
```markdown
---
layout: default
title: Benchmark Results
---

# Sarek Performance Benchmarks

Interactive performance results across different GPUs and backends.

<div id="benchmark-charts"></div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="/javascripts/benchmark-charts.js"></script>
<script>
    loadBenchmarkData('/benchmarks/data/latest.json');
</script>
```

**File:** `gh-pages/javascripts/benchmark-charts.js`
```javascript
async function loadBenchmarkData(dataUrl) {
    const response = await fetch(dataUrl);
    const data = await response.json();
    
    data.charts.forEach(chartSpec => {
        createChart(chartSpec);
    });
}

function createChart(spec) {
    const container = document.getElementById('benchmark-charts');
    const canvas = document.createElement('canvas');
    canvas.id = spec.id;
    container.appendChild(canvas);
    
    new Chart(canvas, {
        type: spec.type,
        data: {
            datasets: spec.datasets.map(ds => ({
                label: ds.label,
                data: ds.data,
                borderColor: ds.color
            }))
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: spec.title }
            },
            scales: {
                x: { title: { display: true, text: spec.xlabel } },
                y: { title: { display: true, text: spec.ylabel } }
            }
        }
    });
}
```

## Implementation Plan

### Phase 1: Basic Plotting
1. **Create `plot_gnuplot.ml`** - Generate gnuplot scripts
   - Read CSV/JSON results
   - Generate line plots (throughput vs size)
   - Output `.gp` files
   
2. **Create `to_latex.ml`** - Generate LaTeX tables
   - Read CSV/JSON results
   - Format as LaTeX tabular
   - Support multiple table types

### Phase 2: Web Integration
3. **Create `plot_web.ml`** - Generate Chart.js HTML
   - Read CSV/JSON results
   - Generate self-contained HTML with embedded data
   - Multiple chart types (line, bar, scatter)

4. **Create JSON data format** for gh-pages
   - Convert results to structured JSON
   - Store in `gh-pages/benchmarks/data/`
   - Update on each benchmark run

### Phase 3: Automation
5. **Create `publish.ml`** - Publish results to gh-pages
   - Generate both gnuplot and web outputs
   - Copy files to gh-pages directory
   - Update index page
   - Optionally commit and push

## Example Workflow

```bash
# Run benchmarks
dune exec benchmarks/bench_runner.exe -- --all --output results/

# Aggregate results
dune exec benchmarks/aggregate.exe -- \
  aggregated.json results/*.json

# Generate publication outputs
dune exec benchmarks/plot_gnuplot.exe -- \
  --input aggregated.json \
  --output-dir paper/figures/
  
dune exec benchmarks/to_latex.exe -- \
  --input aggregated.json \
  --output-dir paper/tables/

# Generate web outputs
dune exec benchmarks/plot_web.exe -- \
  --input aggregated.json \
  --output gh-pages/benchmarks/results.html

# Or use all-in-one publish command
dune exec benchmarks/publish.exe -- \
  --input aggregated.json \
  --gnuplot paper/figures/ \
  --latex paper/tables/ \
  --web gh-pages/benchmarks/
```

## Chart Types to Support

### For Publications (Gnuplot)
- Line plots (performance vs size)
- Bar charts (backend comparison)
- Grouped bar charts (multiple benchmarks)
- Log-log plots (scaling analysis)
- Error bars (stddev)
- Roofline plots (compute vs memory bound)

### For Web (Chart.js)
- Interactive line charts with hover
- Stacked area charts
- Radar charts (multi-dimensional comparison)
- Scatter plots (efficiency analysis)
- Box plots (statistical distribution)
- Filtering/toggling datasets

## Styling Guidelines

### Publication Style
- Grayscale friendly (for B&W printing)
- Distinct line styles (solid, dashed, dotted)
- Clear markers (circles, squares, triangles)
- 12pt Times or Computer Modern fonts
- 6" × 4" figure size (fits 2-column paper)

### Web Style
- Modern color palette
- Responsive design (mobile-friendly)
- Dark mode support
- Smooth animations
- Downloadable as PNG/SVG

## Next Steps

Which should we implement first?

1. **Gnuplot generator** (quickest, publication-ready)
2. **LaTeX table generator** (also quick, very useful)
3. **Web chart generator** (more complex, but great for docs)
4. **All-in-one publish tool** (after above three are done)

I recommend starting with **gnuplot + LaTeX**, then adding **web output** once we have more benchmarks to visualize.
