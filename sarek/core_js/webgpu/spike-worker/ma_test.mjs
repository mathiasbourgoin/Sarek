// SPIKE (throwaway) — Milestone A test: worker-under-COEP + pure-JS SAB/Atomics
// synchronous round-trip, plus a forced-timeout to prove the deadlock watchdog.
// No GPU, no OCaml. Skips (exit 0) if playwright/chrome absent.
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
  for (const base of candidates.filter(Boolean)) {
    try { return createRequire(base + '/x')('playwright'); } catch {}
  }
  try { return reqHere('playwright'); } catch {}
  return null;
}
function skip(m) { console.log('SKIP: ' + m); process.exit(0); }

const pw = resolvePlaywright();
if (!pw) skip('playwright not resolvable');
const { chromium } = pw;

const dir = path.dirname(fileURLToPath(import.meta.url));
const workerJs = fs.readFileSync(path.join(dir, 'ma_worker.js'));

const page_html = `<!doctype html><meta charset=utf-8><title>ma</title>
<script>
function makeRun(forceTimeout) {
  return new Promise((resolve) => {
    const N = 256;
    const sab = new SharedArrayBuffer(8 + 3 * N * 4);
    const ctrl = new Int32Array(sab, 0, 2);
    const data = new Float32Array(sab, 8, 3 * N);
    const w = new Worker('/ma_worker.js');
    w.onmessage = (e) => {
      const m = e.data;
      if (m.t === 'ready') { w.postMessage({ t: 'run' }); }
      else if (m.t === 'go') {
        if (forceTimeout) return;            // deliberately never notify -> worker times out
        for (let i = 0; i < N; i++) data[2 * N + i] = data[i] + data[N + i];  // c = a + b (main thread)
        Atomics.store(ctrl, 0, 2 /*READY*/);
        Atomics.notify(ctrl, 0);
      } else if (m.t === 'result') { resolve(m); w.terminate(); }
    };
    w.postMessage({ t: 'init', sab });
  });
}
window.__runNormal = () => makeRun(false);
window.__runTimeout = () => makeRun(true);
</script>`;

function setHeaders(res, type) {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('content-type', type);
}
const server = http.createServer((req, res) => {
  if (req.url === '/ma_worker.js') { setHeaders(res, 'text/javascript'); res.end(workerJs); }
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

// Normal: synchronous round-trip must succeed.
const normal = await page.evaluate(() => window.__runNormal());
// Forced-timeout: worker must time out (proves the watchdog turns a deadlock into a FAIL).
const timed = await page.evaluate(() => window.__runTimeout());

console.log('normal :', JSON.stringify(normal));
console.log('timeout:', JSON.stringify(timed));

await browser.close();
server.close();

const ok =
  normal.t === 'result' && normal.pass === true &&
  timed.t === 'result' && timed.pass === false && /timed out/.test(timed.error || '');
console.log(ok ? 'MILESTONE-A: PASS ✓ (sync SAB/Atomics round-trip + watchdog)'
               : 'MILESTONE-A: FAIL ✗');
process.exit(ok ? 0 : 1);
