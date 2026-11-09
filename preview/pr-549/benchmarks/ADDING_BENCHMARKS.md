# Adding New Benchmarks to the Web Viewer

This guide explains how to add a new benchmark to the SPOC benchmark web viewer.

## Prerequisites

- Benchmark implemented in `benchmarks/bench_*.ml`
- Benchmark integrated with `Benchmark_runner` module
- Benchmark produces JSON results in standard format

## Required Changes

### 1. Add Benchmark Description

Create a markdown file in `gh-pages/benchmarks/descriptions/`:

```markdown
# Your Benchmark Name

Brief description of what the benchmark measures...

## Algorithm

Explanation of the algorithm...

## Performance Characteristics

- Computational complexity: O(...)
- Memory access pattern: ...
- Key performance factors: ...
```

**Important:** If your benchmark includes images, place them in `descriptions/images/` and reference them as `images/filename.png` (relative to the markdown file).

### 2. Update BENCHMARK_CONFIGS (JavaScript)

In `gh-pages/javascripts/benchmark-viewer.js`, add entry to `BENCHMARK_CONFIGS` (around line 37):

```javascript
const BENCHMARK_CONFIGS = {
    // ... existing benchmarks ...
    
    'your_benchmark': {
        title: 'Your Benchmark Display Name',
        xLabel: 'Input Size Label (unit)',
        throughputLabel: 'Throughput Metric',
        variants: ['your_benchmark'],  // JSON benchmark name(s)
        readme: 'descriptions/your_benchmark.md'
    },
```

**Key fields:**
- `title`: Display name shown in charts
- `xLabel`: Label for X-axis (e.g., "Array Size (elements)")
- `throughputLabel`: Y-axis label for throughput mode (e.g., "GB/s")
- `variants`: Array of JSON benchmark names (usually just `['your_benchmark']`)
- `readme`: Path to your markdown description file

### 3. Update benchmarkConfig (JavaScript)

In the same file, add entry to `benchmarkConfig` object in `updateChart()` function (around line 658):

```javascript
const benchmarkConfig = {
    // ... existing benchmarks ...
    
    'your_benchmark': {
        title: 'Your Benchmark Performance',
        xLabel: 'Input Size (unit)',
        throughputLabel: 'Throughput (metric)',
        throughputUnit: 'metric',
        timeLabel: 'Execution Time (ms)',
        timeUnit: 'ms'
    },
```

**Note:** This duplicates some info from `BENCHMARK_CONFIGS` for detailed chart rendering. Future refactoring could eliminate this duplication.

### 4. Add Selector Option (HTML)

In `gh-pages/benchmarks/index.md`, add an option to the benchmark selector (around line 340):

```html
<select id="benchmark-select" size="8">
    <optgroup label="Your Category">
        <option value="your_benchmark">Your Benchmark Display Name</option>
    </optgroup>
    <!-- ... -->
</select>
```

**Important:** The `value` attribute must exactly match the key in `BENCHMARK_CONFIGS`.

## Example: Adding an FFT Benchmark

Here's a complete example for adding an FFT (Fast Fourier Transform) benchmark:

### 1. Create description file

`gh-pages/benchmarks/descriptions/fft.md`:
```markdown
# Fast Fourier Transform (FFT)

Computes 1D FFT using Cooley-Tukey algorithm...
```

### 2. Add to BENCHMARK_CONFIGS

```javascript
'fft': {
    title: 'Fast Fourier Transform (1D)',
    xLabel: 'Array Size (elements)',
    throughputLabel: 'GFLOPS',
    variants: ['fft'],
    readme: 'descriptions/fft.md'
},
```

### 3. Add to benchmarkConfig

```javascript
'fft': {
    title: 'FFT Performance',
    xLabel: 'Array Size (elements)',
    throughputLabel: 'Throughput (GFLOPS)',
    throughputUnit: 'GFLOPS',
    timeLabel: 'Execution Time (ms)',
    timeUnit: 'ms'
},
```

### 4. Add selector option

```html
<optgroup label="Signal Processing">
    <option value="fft">Fast Fourier Transform (1D)</option>
</optgroup>
```

## Testing

1. Run benchmarks: `make benchmarks`
2. Check JSON output: `cat gh-pages/benchmarks/data/latest.json | jq '.results[] | select(.benchmark.name == "fft")'`
3. Test locally: Open `gh-pages/benchmarks/index.html` in browser
4. Verify:
   - Benchmark appears in dropdown
   - Single chart view displays correctly
   - Comparison view includes benchmark
   - Description loads correctly
   - Images render if present

## Common Issues

### Chart is empty
- Check that `variants` array in `BENCHMARK_CONFIGS` matches JSON `benchmark.name`
- Verify benchmark data exists in `latest.json`
- Check browser console for errors

### Description not loading
- Verify markdown file path in `readme` field
- Check file exists in `gh-pages/benchmarks/descriptions/`

### Images not rendering
- Images should be in `descriptions/images/`
- Reference as `images/filename.png` in markdown
- JavaScript automatically fixes paths

### Dropdown doesn't show benchmark
- Check `<option value>` matches `BENCHMARK_CONFIGS` key exactly
- Verify HTML is valid

## Future Improvements

Consider these improvements to reduce duplication:

1. **Single source of truth**: Merge `BENCHMARK_CONFIGS` and `benchmarkConfig` into one structure
2. **Auto-generate selector**: Build dropdown from `BENCHMARK_CONFIGS` dynamically
3. **Validation tool**: Script to verify all required pieces are in place
4. **JSON schema**: Validate benchmark JSON output format
