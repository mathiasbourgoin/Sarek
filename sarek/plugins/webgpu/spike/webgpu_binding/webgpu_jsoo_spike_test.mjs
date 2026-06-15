// SPIKE: WebGPU jsoo end-to-end acceptance test.
// Loads webgpu_jsoo_spike.bc.js, calls SpocWebGpuSpike.runVectorAdd on a
// 256-element vector (a[i]=i*0.5, b[i]=i*2.0), and checks every element of the
// result equals a[i]+b[i] within 1e-4.
//
// Mirrors the harness of sarek/transpile/web/test/webgpu_wgsl_test.mjs exactly
// for playwright resolution, chrome launch flags, and skip/fail semantics:
// - exit 0 on SKIP (no WebGPU adapter, playwright missing, bundle missing)
// - exit 0 on PASS
// - exit 1 only on a real wrong result
//
// Usage:
//   node sarek/plugins/webgpu/spike/webgpu_binding/webgpu_jsoo_spike_test.mjs \
//     [path/to/webgpu_jsoo_spike.bc.js]
//
// The LEAD runs this on the GPU. This file is authored and syntax-checked only
// (node --check) in CI.
import http from 'http';
import fs from 'fs';
import { createRequire } from 'module';
import { execSync } from 'child_process';

function resolvePlaywright() {
  const reqHere = createRequire(import.meta.url);
  const candidates = [];
  try { candidates.push(execSync('npm root -g').toString().trim()); } catch (_) {}
  try {
    candidates.push(
      ...execSync('ls -d /home/*/.npm/_npx/*/node_modules 2>/dev/null')
        .toString()
        .trim()
        .split('\n')
    );
  } catch (_) {}
  for (const base of candidates.filter(Boolean)) {
    try { return createRequire(base + '/x')('playwright'); } catch (_) {}
  }
  try { return reqHere('playwright'); } catch (_) {}
  return null;
}

function skip(msg) { console.log('SKIP: ' + msg); process.exit(0); }
function fail(msg) { console.log('FAIL: ' + msg); process.exit(1); }

const pw = resolvePlaywright();
if (!pw) skip('playwright not resolvable');
const { chromium } = pw;

const BUNDLE =
  process.argv[2] ||
  '_build/default/sarek/plugins/webgpu/spike/webgpu_binding/webgpu_jsoo_spike.bc.js';
if (!fs.existsSync(BUNDLE)) skip('bundle not built: ' + BUNDLE);
const bundleJs = fs.readFileSync(BUNDLE);

const html =
  '<!doctype html><meta charset=utf-8><title>spoc-webgpu-spike</title>' +
  '<script src="/b.js"></script>';

const server = http.createServer((req, res) => {
  if (req.url === '/b.js') {
    res.setHeader('content-type', 'text/javascript');
    res.end(bundleJs);
  } else {
    res.setHeader('content-type', 'text/html');
    res.end(html);
  }
});
await new Promise((r) => server.listen(0, r));
const port = server.address().port;

let browser;
try {
  browser = await chromium.launch({
    headless: true,
    channel: 'chrome',
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-angle=vulkan',
      '--use-gl=angle',
      '--ignore-gpu-blocklist',
      '--no-sandbox',
    ],
  });
} catch (e) {
  server.close();
  skip('chrome launch failed: ' + e.message);
}

const page = await browser.newPage();
await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });
await page.waitForFunction(
  () => typeof globalThis.SpocWebGpuSpike !== 'undefined',
  { timeout: 20000 }
);

// Guard: skip gracefully if no WebGPU adapter is available.
const hasAdapter = await page.evaluate(async () => {
  if (!navigator.gpu) return false;
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  return a !== null;
});
if (!hasAdapter) {
  await browser.close();
  server.close();
  skip('no WebGPU adapter in this Chrome');
}

// Run the vector-add spike.
const result = await page.evaluate(async () => {
  const N = 256;
  const a = new Float32Array(N);
  const b = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    a[i] = i * 0.5;
    b[i] = i * 2.0;
  }

  // Wrap the OCaml callback-style API in a Promise so we can await it.
  const output = await new Promise((resolve, reject) => {
    globalThis.SpocWebGpuSpike.runVectorAdd(a, b, (result, err) => {
      if (err !== null) reject(new Error(String(err)));
      else resolve(result);
    });
  });

  // Validate.
  let badCount = 0;
  const errors = [];
  for (let i = 0; i < N; i++) {
    const expected = a[i] + b[i];
    if (Math.abs(output[i] - expected) > 1e-4) {
      badCount++;
      if (errors.length < 5)
        errors.push({ i, got: output[i], expected });
    }
  }
  return { ok: badCount === 0, badCount, errors, n: N };
});

await browser.close();
server.close();

if (result.ok) {
  console.log('PASS: SpocWebGpuSpike.runVectorAdd correct for all ' + result.n + ' elements');
  process.exit(0);
} else {
  fail(
    'SpocWebGpuSpike.runVectorAdd: ' +
      result.badCount +
      ' wrong values; first errors: ' +
      JSON.stringify(result.errors)
  );
}
