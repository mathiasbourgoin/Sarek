// SPIKE (throwaway) — Milestone A Step 1: cross-origin isolation under flagged Chrome.
// The make-or-break environment check for the worker+SAB Execute-parity approach.
// Serves a page with COOP/COEP/CORP headers and asserts crossOriginIsolated===true
// AND SharedArrayBuffer is available, under the same flagged Chrome used for WebGPU.
// Skips (exit 0) only if playwright/chrome are unavailable; FAILS (exit 1) if isolation
// cannot be achieved.
import http from 'http';
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

// COOP/COEP/CORP on EVERY response — required for crossOriginIsolated + SharedArrayBuffer.
function setIsolationHeaders(res) {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
}
const html = `<!doctype html><meta charset=utf-8><title>iso</title>`;
const server = http.createServer((_req, res) => {
  setIsolationHeaders(res);
  res.setHeader('content-type', 'text/html');
  res.end(html);
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

const result = await page.evaluate(() => ({
  crossOriginIsolated: globalThis.crossOriginIsolated === true,
  hasSAB: typeof SharedArrayBuffer !== 'undefined',
  // a SAB can only be *constructed* when isolated; prove it end-to-end
  canAllocSAB: (() => { try { new SharedArrayBuffer(16); return true; } catch { return false; } })(),
  hasGPU: !!navigator.gpu,
}));
console.log(JSON.stringify(result));

await browser.close();
server.close();

const ok = result.crossOriginIsolated && result.hasSAB && result.canAllocSAB;
console.log(ok ? 'ISOLATION: PASS ✓ (crossOriginIsolated + SharedArrayBuffer usable)'
               : 'ISOLATION: FAIL ✗ — worker+SAB approach is environment-blocked here');
process.exit(ok ? 0 : 1);
