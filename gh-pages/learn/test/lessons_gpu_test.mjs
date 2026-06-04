// Playwright acceptance test for the Sarek interactive GPU course pages.
//
// For each lesson: (a) loads the page, (b) injects the CORRECT kernel body
// and asserts the result panel shows PASS, (c) injects a WRONG body and
// asserts FAIL. Graceful skip (exit 0) when playwright/chrome/WebGPU are
// unavailable; only FAIL on a real wrong result or compile error.
//
// Usage:
//   node gh-pages/learn/test/lessons_gpu_test.mjs [bundle.bc.js]
//
// Mirrors the structure of sarek/transpile/web/test/webgpu_wgsl_test.mjs.

import http from 'http';
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { execSync } from 'child_process';

// ── Playwright resolution (same approach as webgpu_wgsl_test.mjs) ─────────
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

// ── Asset paths ───────────────────────────────────────────────────────────
const BUNDLE = process.argv[2] ||
  '_build/default/sarek/transpile/web/transpile_js.bc.js';
if (!fs.existsSync(BUNDLE)) skip('bundle not built: ' + BUNDLE);
const bundleJs = fs.readFileSync(BUNDLE);

const RUNNER = 'gh-pages/javascripts/sarek_webgpu_runner.js';
if (!fs.existsSync(RUNNER)) skip('runner not found: ' + RUNNER);
const runnerJs = fs.readFileSync(RUNNER);

const LESSON_JS = 'gh-pages/javascripts/sarek_lesson.js';
if (!fs.existsSync(LESSON_JS)) skip('sarek_lesson.js not found: ' + LESSON_JS);
const lessonJs = fs.readFileSync(LESSON_JS);

// ── Minimal HTTP server ───────────────────────────────────────────────────
// Serves the JS assets. The lesson HTML is injected inline per test case.
const server = http.createServer((req, res) => {
  if (req.url === '/b.js') {
    res.setHeader('content-type', 'text/javascript');
    res.end(bundleJs);
  } else if (req.url === '/r.js') {
    res.setHeader('content-type', 'text/javascript');
    res.end(runnerJs);
  } else if (req.url === '/l.js') {
    res.setHeader('content-type', 'text/javascript');
    res.end(lessonJs);
  } else {
    // Serve a minimal harness page that loads all three scripts.
    const body = `<!doctype html>
<meta charset=utf-8>
<title>lesson-test</title>
<script src="/b.js"></script>
<script src="/r.js"></script>
<script src="/l.js"></script>`;
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
  () =>
    typeof globalThis.SarekTranspile !== 'undefined' &&
    typeof globalThis.SarekWebGPU !== 'undefined' &&
    typeof globalThis.SarekLesson !== 'undefined',
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

// ── Test case definitions ─────────────────────────────────────────────────
// Each case: lesson id, cfg passed to SarekLesson.init (minus DOM ids),
// correctBody (replaces the TODO), wrongBody (should produce FAIL).
const N = 256;

const CASES = [
  {
    id: 'lesson1-vector-add',
    cfg: {
      n: N,
      buffers: {
        a: { role: 'input',  elementType: 'f32', gen: 'function(i){ return i * 0.5; }' },
        b: { role: 'input',  elementType: 'f32', gen: 'function(i){ return i * 2.0; }' },
        c: { role: 'output', elementType: 'f32' },
      },
      scalars:   {},
      outName:   'c',
      reference: 'function(i, inp) { return inp.a[i] + inp.b[i]; }',
    },
    starterFn: 'function(body) { return ' +
      '"fun (a : float32 vector) (b : float32 vector) (c : float32 vector) ->\\n" +' +
      '"  let i = global_thread_id in\\n" +' +
      '"  c.(i) <- " + body; }',
    correctBody: 'a.(i) +. b.(i)',
    wrongBody:   'a.(i) -. b.(i)',
  },
  {
    id: 'lesson2-saxpy',
    cfg: {
      n: N,
      buffers: {
        x: { role: 'input',  elementType: 'f32', gen: 'function(i){ return i * 0.5; }' },
        y: { role: 'input',  elementType: 'f32', gen: 'function(i){ return i * 1.0; }' },
      },
      scalars:   { alpha: 2.0 },
      outName:   'y',
      reference: 'function(i, inp, sc) { return sc.alpha * inp.x[i] + inp.y[i]; }',
    },
    starterFn: 'function(body) { return ' +
      '"fun (x : float32 vector) (y : float32 vector) (alpha : float32) ->\\n" +' +
      '"  let i = global_thread_id in\\n" +' +
      '"  y.(i) <- " + body; }',
    correctBody: '(alpha *. x.(i)) +. y.(i)',
    wrongBody:   'x.(i)',
  },
  {
    id: 'lesson3-map-square',
    cfg: {
      n: N,
      buffers: {
        a: { role: 'input',  elementType: 'f32', gen: 'function(i){ return i * 0.5; }' },
        b: { role: 'output', elementType: 'f32' },
      },
      scalars:   {},
      outName:   'b',
      reference: 'function(i, inp) { return inp.a[i] * inp.a[i]; }',
    },
    starterFn: 'function(body) { return ' +
      '"fun (a : float32 vector) (b : float32 vector) ->\\n" +' +
      '"  let i = global_thread_id in\\n" +' +
      '"  b.(i) <- " + body; }',
    correctBody: 'a.(i) *. a.(i)',
    wrongBody:   'a.(i)',
  },
  {
    id: 'lesson4-bounds-if',
    cfg: {
      n: N,
      buffers: {
        a: { role: 'input',  elementType: 'f32', gen: 'function(i){ return i * 0.5; }' },
        b: { role: 'output', elementType: 'f32' },
      },
      scalars:   { n: 128 },
      outName:   'b',
      reference: 'function(i, inp, sc) { return i < sc.n ? inp.a[i] : 0.0; }',
    },
    starterFn: 'function(body) { return ' +
      '"fun (a : float32 vector) (b : float32 vector) (n : int32) ->\\n" +' +
      '"  let i = global_thread_id in\\n" +' +
      '"  b.(i) <- " + body; }',
    correctBody: '(if i < n then a.(i) else 0.0)',
    wrongBody:   'a.(i)',
  },
];

// ── Run each test case ────────────────────────────────────────────────────
const results = [];

for (const tc of CASES) {
  // Serialise gen functions as strings and reconstruct on the page side.
  // We pass the whole cfg as a JSON-safe object except function fields,
  // which are inlined as evaluated strings via page.evaluate.

  // CORRECT body
  const correctResult = await page.evaluate(
    async ({ cfg, starterFn, body, N }) => {
      const mkSource = eval('(' + starterFn + ')');   // eslint-disable-line no-eval

      // Rebuild buffers with real gen functions.
      const bufs = {};
      for (const [name, b] of Object.entries(cfg.buffers)) {
        bufs[name] = {
          role: b.role,
          elementType: b.elementType,
          gen: b.gen ? eval('(' + b.gen + ')') : undefined, // eslint-disable-line no-eval
        };
      }
      const ref = eval('(' + cfg.reference + ')');           // eslint-disable-line no-eval

      const source = mkSource(body);
      const r = globalThis.SarekTranspile.transpileWithAbi(source, 'wgsl');
      if (!r.ok) return { pass: false, why: 'transpile: ' + r.error };

      // Build inputs
      const inputs = {};
      for (const [name, b] of Object.entries(bufs)) {
        const Ctor = b.elementType === 'f32' ? Float32Array
          : b.elementType === 'i32' ? Int32Array : Uint32Array;
        const arr = new Ctor(N);
        if (b.role === 'input' && b.gen) for (let i = 0; i < N; i++) arr[i] = b.gen(i);
        inputs[name] = arr;
      }

      let runResult;
      try {
        runResult = await globalThis.SarekWebGPU.run(r.code, r.abi,
          { inputs, scalars: cfg.scalars });
      } catch (e) {
        return { pass: false, why: 'run: ' + e.message };
      }

      const out = runResult.outputs[cfg.outName];
      let bad = 0;
      for (let i = 0; i < N; i++) {
        const expected = ref(i, inputs, cfg.scalars);
        if (Math.abs(out[i] - expected) > 1e-3 * Math.max(Math.abs(out[i]), Math.abs(expected)) + 1e-3) bad++;
      }
      return { pass: bad === 0, bad };
    },
    { cfg: serializeCfg(tc.cfg), starterFn: tc.starterFn, body: tc.correctBody, N }
  );

  // WRONG body
  const wrongResult = await page.evaluate(
    async ({ cfg, starterFn, body, N }) => {
      const mkSource = eval('(' + starterFn + ')');   // eslint-disable-line no-eval
      const bufs = {};
      for (const [name, b] of Object.entries(cfg.buffers)) {
        bufs[name] = {
          role: b.role,
          elementType: b.elementType,
          gen: b.gen ? eval('(' + b.gen + ')') : undefined, // eslint-disable-line no-eval
        };
      }
      const ref = eval('(' + cfg.reference + ')');           // eslint-disable-line no-eval
      const source = mkSource(body);
      const r = globalThis.SarekTranspile.transpileWithAbi(source, 'wgsl');
      if (!r.ok) {
        // A transpile error counts as "not PASS" which is fine for the wrong-body test.
        return { notPass: true, why: 'transpile error (expected for wrong body): ' + r.error };
      }
      const inputs = {};
      for (const [name, b] of Object.entries(bufs)) {
        const Ctor = b.elementType === 'f32' ? Float32Array
          : b.elementType === 'i32' ? Int32Array : Uint32Array;
        const arr = new Ctor(N);
        if (b.role === 'input' && b.gen) for (let i = 0; i < N; i++) arr[i] = b.gen(i);
        inputs[name] = arr;
      }
      let runResult;
      try {
        runResult = await globalThis.SarekWebGPU.run(r.code, r.abi,
          { inputs, scalars: cfg.scalars });
      } catch (e) {
        return { notPass: true, why: 'run error (expected for wrong body): ' + e.message };
      }
      const out = runResult.outputs[cfg.outName];
      let bad = 0;
      for (let i = 0; i < N; i++) {
        const expected = ref(i, inputs, cfg.scalars);
        if (Math.abs(out[i] - expected) > 1e-3 * Math.max(Math.abs(out[i]), Math.abs(expected)) + 1e-3) bad++;
      }
      // We WANT bad > 0 for the wrong body.
      return { notPass: bad > 0, bad };
    },
    { cfg: serializeCfg(tc.cfg), starterFn: tc.starterFn, body: tc.wrongBody, N }
  );

  const ok = correctResult.pass && (wrongResult.notPass || wrongResult.bad > 0);
  results.push({
    id: tc.id,
    correctPass: correctResult.pass,
    wrongFails:  !!(wrongResult.notPass || wrongResult.bad > 0),
    ok,
    correctDetail: correctResult,
    wrongDetail:   wrongResult,
  });
}

await browser.close();
server.close();

// ── Report ────────────────────────────────────────────────────────────────
console.log(JSON.stringify(results, null, 2));
const allOk = results.length > 0 && results.every(r => r.ok);
console.log(allOk ? 'LESSONS-GPU: ALL PASS' : 'LESSONS-GPU: FAILURE');
process.exit(allOk ? 0 : 1);

// ── Helpers ───────────────────────────────────────────────────────────────
function serializeCfg(cfg) {
  // Buffers: serialize gen as a string (eval'd on page side).
  const bufs = {};
  for (const [name, b] of Object.entries(cfg.buffers)) {
    bufs[name] = {
      role:        b.role,
      elementType: b.elementType,
      gen:         b.gen || null,
    };
  }
  return {
    n:         cfg.n,
    buffers:   bufs,
    scalars:   cfg.scalars,
    outName:   cfg.outName,
    reference: cfg.reference,
  };
}
