// SpocRT WebGPU runtime acceptance test.
//
// Runs vector_add and sin kernels via SpocRT.run (the OCaml jsoo runtime driver)
// on a real GPU. Skips gracefully where playwright/chrome/WebGPU are unavailable.
// Only exits non-zero on a real wrong numerical result.
//
// Usage:
//   node sarek/core_js/webgpu/test/webgpu_runtime_test.mjs [runtime_driver.bc.js]

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
        .toString().trim().split('\n')
    );
  } catch (_) {}
  for (const base of candidates.filter(Boolean)) {
    try { return createRequire(base + '/x')('playwright'); } catch (_) {}
  }
  try { return reqHere('playwright'); } catch (_) {}
  return null;
}

function skip(msg) { console.log('SKIP: ' + msg); process.exit(0); }

const pw = resolvePlaywright();
if (!pw) skip('playwright not resolvable');
const { chromium } = pw;

const DRIVER = process.argv[2] ||
  '_build/default/sarek/core_js/webgpu/test/runtime_driver.bc.js';
if (!fs.existsSync(DRIVER)) skip('driver not built: ' + DRIVER);
const driverJs = fs.readFileSync(DRIVER);

const html = `<!doctype html><meta charset=utf-8><title>spocrt-test</title>
<script src="/d.js"></script>`;
const server = http.createServer((req, res) => {
  if (req.url === '/d.js') {
    res.setHeader('content-type', 'text/javascript');
    res.end(driverJs);
  } else {
    res.setHeader('content-type', 'text/html');
    res.end(html);
  }
});
await new Promise(r => server.listen(0, r));
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
  () => typeof globalThis.SpocRT !== 'undefined',
  { timeout: 20000 }
);

const hasAdapter = await page.evaluate(async () =>
  !!(navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }))
);
if (!hasAdapter) {
  await browser.close();
  server.close();
  skip('no WebGPU adapter in this Chrome');
}

const N = 256;
const result = await page.evaluate(async (n) => {
  const out = [];
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) { a[i] = i * 0.5; b[i] = i * 2.0; }

  function runKernel(src, inputs, scalars, outName) {
    return new Promise((resolve, reject) => {
      globalThis.SpocRT.run(src, inputs, scalars, outName, (err, arr) => {
        if (err) reject(new Error(String(err)));
        else resolve(arr);
      });
    });
  }

  const cases = [
    {
      k: 'vector_add',
      src: 'fun (a:float32 vector)(b:float32 vector)(c:float32 vector) -> ' +
           'let i = global_thread_id in c.(i) <- a.(i) +. b.(i)',
      inputs: { a: a.slice(), b: b.slice(), c: new Float32Array(n) },
      scalars: {},
      outName: 'c',
      ref: (i) => a[i] + b[i],
    },
    {
      k: 'sin',
      src: 'fun (a:float32 vector)(b:float32 vector) -> ' +
           'let i = global_thread_id in b.(i) <- Float32.sin a.(i)',
      inputs: { a: a.slice(), b: new Float32Array(n) },
      scalars: {},
      outName: 'b',
      ref: (i) => Math.sin(a[i]),
    },
  ];

  for (const c of cases) {
    let arr;
    try {
      arr = await runKernel(c.src, c.inputs, c.scalars, c.outName);
    } catch (e) {
      out.push({ k: c.k, ok: false, why: e.message });
      continue;
    }
    let bad = 0;
    for (let i = 0; i < n; i++) {
      if (Math.abs(arr[i] - c.ref(i)) > 1e-4) bad++;
    }
    out.push({ k: c.k, ok: bad === 0, bad });
  }
  return out;
}, N);

console.log(JSON.stringify(result));
await browser.close();
server.close();

const allok = Array.isArray(result) && result.length > 0 && result.every(r => r.ok);
console.log(allok ? 'SpocRT-GPU: ALL PASS' : 'SpocRT-GPU: FAILURE');
process.exit(allok ? 0 : 1);
