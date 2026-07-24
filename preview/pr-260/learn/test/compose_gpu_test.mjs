// Playwright acceptance test for the "Composing kernels" lesson driver.
//
// Tests SpocCompose.run with:
//   (a) correct kernel A + correct kernel B -> out == a²+b (within 1e-4)
//   (b) wrong kernel A (identity instead of square) -> mismatch detected
//
// Graceful skip (exit 0) when playwright / chrome / WebGPU are unavailable.
// Only fails on a real wrong GPU result or an unexpected error.
//
// Usage:
//   node gh-pages/learn/test/compose_gpu_test.mjs [compose_driver.bc.js]

import http from 'http';
import fs from 'fs';
import { createRequire } from 'module';
import { execSync } from 'child_process';

// ── Playwright resolution (mirrors lessons_gpu_test.mjs) ─────────────────
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

// ── Asset path ────────────────────────────────────────────────────────────
const BUNDLE = process.argv[2] ||
  '_build/default/sarek/core_js/webgpu/lessons/compose_driver.bc.js';
if (!fs.existsSync(BUNDLE)) skip('compose bundle not built: ' + BUNDLE);
const bundleJs = fs.readFileSync(BUNDLE);

// ── Minimal HTTP server ───────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  if (req.url === '/compose.js') {
    res.setHeader('content-type', 'text/javascript');
    res.end(bundleJs);
  } else {
    const body = `<!doctype html>
<meta charset=utf-8>
<title>compose-test</title>
<script src="/compose.js"></script>`;
    res.setHeader('content-type', 'text/html');
    res.end(body);
  }
});
await new Promise(r => server.listen(0, r));
const port = server.address().port;

// ── Browser launch ────────────────────────────────────────────────────────
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
  () => typeof globalThis.SpocCompose !== 'undefined',
  { timeout: 20000 }
);

const hasAdapter = await page.evaluate(
  async () =>
    !!(navigator.gpu &&
      (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })))
);
if (!hasAdapter) {
  await browser.close();
  server.close();
  skip('no WebGPU adapter in this Chrome');
}

// ── Kernel sources ────────────────────────────────────────────────────────
const KERNEL_A_CORRECT =
  'fun (a : float32 vector) (tmp : float32 vector) ->\n' +
  '  let i = global_thread_id in\n' +
  '  tmp.(i) <- a.(i) *. a.(i)';

const KERNEL_A_WRONG =
  'fun (a : float32 vector) (tmp : float32 vector) ->\n' +
  '  let i = global_thread_id in\n' +
  '  tmp.(i) <- a.(i)';

const KERNEL_B =
  'fun (tmp : float32 vector) (b : float32 vector) (out : float32 vector) ->\n' +
  '  let i = global_thread_id in\n' +
  '  out.(i) <- tmp.(i) +. b.(i)';

// ── Test helpers ──────────────────────────────────────────────────────────
function makeInputs(N) {
  const a = new Array(N), b = new Array(N);
  for (let i = 0; i < N; i++) { a[i] = i * 0.5; b[i] = i * 2.0; }
  return { a, b };
}

function referenceOut(inputs) {
  return inputs.a.map((ai, i) => ai * ai + inputs.b[i]);
}

// ── Run a single compose case on the page ─────────────────────────────────
async function runCompose(kA, kB, inputs) {
  return page.evaluate(
    async ({ kA, kB, inputs }) => {
      return new Promise((resolve) => {
        SpocCompose.run(kA, kB, { a: inputs.a, b: inputs.b },
          function (result, err) {
            if (err) { resolve({ ok: false, error: err }); return; }
            resolve({ ok: true, result: Array.from(result) });
          }
        );
      });
    },
    { kA, kB, inputs }
  );
}

// ── Test execution ────────────────────────────────────────────────────────
const N = 256;
const inputs = makeInputs(N);
const ref = referenceOut(inputs);
const results = [];

// Case 1: correct kernels -> out == a²+b within 1e-4
const correctRun = await runCompose(KERNEL_A_CORRECT, KERNEL_B, inputs);
if (!correctRun.ok) {
  results.push({ id: 'correct-kernels', ok: false, detail: correctRun.error });
} else {
  let bad = 0;
  for (let i = 0; i < N; i++) {
    if (Math.abs(correctRun.result[i] - ref[i]) > 1e-4) bad++;
  }
  results.push({ id: 'correct-kernels', ok: bad === 0, bad });
}

// Case 2: wrong kernel A -> result should not match a²+b
const wrongRun = await runCompose(KERNEL_A_WRONG, KERNEL_B, inputs);
if (!wrongRun.ok) {
  // A transpile/runtime error from wrong kernel still counts as "not PASS".
  results.push({ id: 'wrong-kernel-a', ok: true, detail: 'error as expected: ' + wrongRun.error });
} else {
  let bad = 0;
  for (let i = 0; i < N; i++) {
    if (Math.abs(wrongRun.result[i] - ref[i]) > 1e-4) bad++;
  }
  // We want mismatches (bad > 0) to confirm wrong kernel is detected.
  results.push({ id: 'wrong-kernel-a', ok: bad > 0, bad });
}

// ── Teardown and report ───────────────────────────────────────────────────
await browser.close();
server.close();

console.log(JSON.stringify(results, null, 2));
const allOk = results.length > 0 && results.every(r => r.ok);
console.log(allOk ? 'COMPOSE-GPU: ALL PASS' : 'COMPOSE-GPU: FAILURE');
process.exit(allOk ? 0 : 1);
