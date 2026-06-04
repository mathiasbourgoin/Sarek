// WGSL backend acceptance test: transpile kernels to WGSL via transpileWithAbi
// and run them on a real GPU using the generic SarekWebGPU runner loaded
// alongside the bundle. Validates that the generated WGSL compiles AND computes
// correctly against a CPU reference. Skips gracefully (exit 0) where
// playwright/chrome/WebGPU are unavailable; only FAILS on a real wrong result
// or compile error.
//
// Usage: node sarek/transpile/web/test/webgpu_wgsl_test.mjs [bundle.bc.js]
import http from 'http';
import fs from 'fs';
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
function skip(msg){ console.log('SKIP: ' + msg); process.exit(0); }

const pw = resolvePlaywright();
if (!pw) skip('playwright not resolvable');
const { chromium } = pw;

const BUNDLE = process.argv[2] || '_build/default/sarek/transpile/web/transpile_js.bc.js';
if (!fs.existsSync(BUNDLE)) skip('bundle not built: ' + BUNDLE);
const bundleJs = fs.readFileSync(BUNDLE);

const RUNNER = 'gh-pages/javascripts/sarek_webgpu_runner.js';
if (!fs.existsSync(RUNNER)) skip('runner not found: ' + RUNNER);
const runnerJs = fs.readFileSync(RUNNER);

const html = `<!doctype html><meta charset=utf-8><title>wgsl-test</title><script src="/b.js"></script><script src="/r.js"></script>`;
const server = http.createServer((req,res)=>{
  if (req.url==='/b.js'){res.setHeader('content-type','text/javascript');res.end(bundleJs);}
  else if (req.url==='/r.js'){res.setHeader('content-type','text/javascript');res.end(runnerJs);}
  else {res.setHeader('content-type','text/html');res.end(html);}
});
await new Promise(r=>server.listen(0,r));
const port = server.address().port;

let browser;
try {
  browser = await chromium.launch({ headless:true, channel:'chrome',
    args:['--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=vulkan','--use-gl=angle','--ignore-gpu-blocklist','--no-sandbox'] });
} catch (e) { server.close(); skip('chrome launch failed: ' + e.message); }

const page = await browser.newPage();
await page.goto(`http://localhost:${port}/`, {waitUntil:'load'});
await page.waitForFunction(()=>typeof globalThis.SarekTranspile!=='undefined'&&typeof globalThis.SarekWebGPU!=='undefined', {timeout:20000});

const hasAdapter = await page.evaluate(async()=> !!(navigator.gpu && await navigator.gpu.requestAdapter({powerPreference:'high-performance'})));
if (!hasAdapter) { await browser.close(); server.close(); skip('no WebGPU adapter in this Chrome'); }

const result = await page.evaluate(async () => {
  const N=256;
  const BC_N = N/2; // half array in-bounds
  const T = globalThis.SarekTranspile.transpileWithAbi;
  const runner = globalThis.SarekWebGPU;
  const a=new Float32Array(N), b=new Float32Array(N);
  for(let i=0;i<N;i++){a[i]=i*0.5; b[i]=i*2.0;}
  const out=[];
  const cases=[
    {k:'vector_add',
     src:"fun (a:float32 vector)(b:float32 vector)(c:float32 vector) -> let i = global_thread_id in c.(i) <- a.(i) +. b.(i)",
     inputs:{a:a.slice(),b:b.slice(),c:new Float32Array(N)},
     scalars:{},
     ref:i=>a[i]+b[i], outName:'c'},
    {k:'sin',
     src:"fun (a:float32 vector)(b:float32 vector) -> let i = global_thread_id in b.(i) <- Float32.sin a.(i)",
     inputs:{a:a.slice(),b:new Float32Array(N)},
     scalars:{},
     ref:i=>Math.sin(a[i]), outName:'b'},
    // bounds_check: exercises EIf -> select(else,then,cond) fix.
    // Kernel: b.(i) <- if i < n then a.(i) else 0.0
    // Expected: b[i] = a[i] for i < BC_N, else 0.0
    // n is supplied via scalars (no hand-packed Int32Array).
    {k:'bounds_check',
     src:"fun (a:float32 vector)(b:float32 vector)(n:int32) -> let i = global_thread_id in b.(i) <- (if i < n then a.(i) else 0.0)",
     inputs:{a:a.slice(),b:new Float32Array(N)},
     scalars:{n:BC_N},
     ref:i=>(i<BC_N?a[i]:0.0), outName:'b'},
  ];
  for(const c of cases){
    const r=T(c.src,'wgsl');
    if(!r.ok){ out.push({k:c.k,ok:false,why:'transpile: '+r.error}); continue; }
    let runResult;
    try {
      runResult = await runner.run(r.code, r.abi, {inputs:c.inputs, scalars:c.scalars});
    } catch(e) { out.push({k:c.k,ok:false,why:'runner: '+e.message}); continue; }
    const o = runResult.outputs;
    let bad=0; for(let i=0;i<N;i++) if(Math.abs(o[c.outName][i]-c.ref(i))>1e-4) bad++;
    out.push({k:c.k,ok:bad===0,bad});
  }
  return out;
});
console.log(JSON.stringify(result));
await browser.close(); server.close();
const allok = Array.isArray(result) && result.length>0 && result.every(r=>r.ok);
console.log(allok ? 'WGSL-GPU: ALL PASS ✓' : 'WGSL-GPU: FAILURE ✗');
process.exit(allok?0:1);
