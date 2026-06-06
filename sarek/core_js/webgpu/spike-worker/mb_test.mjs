// SPIKE (throwaway) — Milestone B test: jsoo-OCaml-in-a-worker obtains a real
// WebGPU vector_add result SYNCHRONOUSLY via SharedArrayBuffer + Atomics.wait,
// with the GPU service (Webgpu_runtime via SpocRT.run) on the main thread, under
// cross-origin isolation. Skips (exit 0) if playwright/chrome/adapter absent.
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { execSync } from 'child_process';

function resolvePlaywright() {
  const reqHere = createRequire(import.meta.url);
  const candidates = [];
  try { candidates.push(execSync('npm root -g').toString().trim()); } catch {}
  try { candidates.push(...execSync('ls -d /home/*/.npm/_npx/*/node_modules 2>/dev/null').toString().trim().split('\n')); } catch {}
  for (const base of candidates.filter(Boolean)) { try { return createRequire(base + '/x')('playwright'); } catch {} }
  try { return reqHere('playwright'); } catch {}
  return null;
}
function skip(m) { console.log('SKIP: ' + m); process.exit(0); }

const pw = resolvePlaywright();
if (!pw) skip('playwright not resolvable');
const { chromium } = pw;

const root = '/home/mathias/dev/SPOC';
const workerBc = fs.readFileSync(path.join(root, '_build/default/sarek/core_js/webgpu/spike-worker/worker_spike.bc.js'));
const driverBc = fs.readFileSync(path.join(root, '_build/default/sarek/core_js/webgpu/test/runtime_driver.bc.js'));

const page_html = `<!doctype html><meta charset=utf-8><title>mb</title>
<script src="/runtime_driver.bc.js"></script>
<script>
const N = 256;
const VA = "fun (a:float32 vector)(b:float32 vector)(c:float32 vector) -> let i = global_thread_id in c.(i) <- a.(i) +. b.(i)";
window.__mb = null;
(function () {
  const sab = new SharedArrayBuffer(8 + 3 * N * 4);
  const ctrl = new Int32Array(sab, 0, 2);
  const data = new Float32Array(sab, 8, 3 * N);
  const w = new Worker('/worker_spike.bc.js');
  w.onmessage = (e) => {
    const m = e.data;
    if (m.t === 'ready') { w.postMessage({ t: 'run' }); }
    else if (m.t === 'go') {
      const a = Array.from(data.subarray(0, N));
      const b = Array.from(data.subarray(N, 2 * N));
      // GPU service on the MAIN thread (never blocks): real WebGPU via SpocRT.run.
      SpocRT.run(VA, { a, b, c: new Array(N).fill(0) }, {}, 'c', (err, result) => {
        if (err || !result) {
          Atomics.store(ctrl, 1, 1); Atomics.store(ctrl, 0, 3 /*ERROR*/); Atomics.notify(ctrl, 0);
          return;
        }
        for (let i = 0; i < N; i++) data[2 * N + i] = result[i];
        Atomics.store(ctrl, 0, 2 /*READY*/); Atomics.notify(ctrl, 0);
      });
    } else if (m.t === 'result') { window.__mb = m; w.terminate(); }
  };
  w.postMessage({ t: 'init', sab });
})();
</script>`;

function setHeaders(res, type) {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('content-type', type);
}
const server = http.createServer((req, res) => {
  if (req.url === '/worker_spike.bc.js') { setHeaders(res, 'text/javascript'); res.end(workerBc); }
  else if (req.url === '/runtime_driver.bc.js') { setHeaders(res, 'text/javascript'); res.end(driverBc); }
  else { setHeaders(res, 'text/html'); res.end(page_html); }
});
await new Promise(r => server.listen(0, r));
const port = server.address().port;

let browser;
try {
  browser = await chromium.launch({
    headless: true, channel: 'chrome',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan',
           '--use-gl=angle', '--ignore-gpu-blocklist', '--no-sandbox'],
  });
} catch (e) { server.close(); skip('chrome launch failed: ' + e.message); }

const page = await browser.newPage();
await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });

const iso = await page.evaluate(() => globalThis.crossOriginIsolated === true);
if (!iso) { await browser.close(); server.close(); console.log('ISOLATION: FAIL'); process.exit(1); }
const hasAdapter = await page.evaluate(async () => !!(navigator.gpu && await navigator.gpu.requestAdapter()));
if (!hasAdapter) { await browser.close(); server.close(); skip('no WebGPU adapter'); }

let res;
try {
  res = await page.waitForFunction(() => globalThis.__mb, { timeout: 20000 }).then(h => h.jsonValue());
} catch (e) { await browser.close(); server.close(); console.log('MILESTONE-B: FAIL ✗ (deadlock/timeout — page watchdog fired)'); process.exit(1); }

console.log('result:', JSON.stringify(res));
await browser.close();
server.close();

const ok = res && res.t === 'result' && res.pass === true;
console.log(ok ? 'MILESTONE-B: PASS ✓ (jsoo-in-worker synchronous WebGPU vector_add via SAB+Atomics)'
               : 'MILESTONE-B: FAIL ✗');
process.exit(ok ? 0 : 1);
