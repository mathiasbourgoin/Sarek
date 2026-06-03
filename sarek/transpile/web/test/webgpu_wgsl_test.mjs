// WGSL backend acceptance test: transpile kernels to WGSL and run them on a
// real GPU via WebGPU in a Playwright-driven Chrome (Dawn/Vulkan). Validates
// that the generated WGSL compiles AND computes correctly against a CPU
// reference. Skips gracefully (exit 0) where playwright/chrome/WebGPU are
// unavailable; only FAILS on a real wrong result or compile error.
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

const html = `<!doctype html><meta charset=utf-8><title>wgsl-test</title><script src="/b.js"></script>`;
const server = http.createServer((req,res)=>{
  if (req.url==='/b.js'){res.setHeader('content-type','text/javascript');res.end(bundleJs);}
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
await page.waitForFunction(()=>typeof globalThis.SarekTranspile!=='undefined', {timeout:20000});

const hasAdapter = await page.evaluate(async()=> !!(navigator.gpu && await navigator.gpu.requestAdapter({powerPreference:'high-performance'})));
if (!hasAdapter) { await browser.close(); server.close(); skip('no WebGPU adapter in this Chrome'); }

const result = await page.evaluate(async () => {
  const N=256;
  // Run a WGSL kernel. bufs entries: {name, binding, data (TypedArray), out?, uniform?}
  // uniform:true → GPUBufferUsage.UNIFORM (no STORAGE); used for Params struct.
  async function runWGSL(wgsl, bufs){
    const ad=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    const dev=await ad.requestDevice();
    const mod=dev.createShaderModule({code:wgsl});
    const ci=await mod.getCompilationInfo();
    const es=ci.messages.filter(m=>m.type==='error');
    if(es.length) return {err:'compile: '+es.map(e=>e.message+' @line'+e.lineNum).join(' | ')};
    const pipe=dev.createComputePipeline({layout:'auto',compute:{module:mod,entryPoint:'main'}});
    const gb={}, rb={};
    for(const b of bufs){
      const usage = b.uniform
        ? GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
      gb[b.binding]=dev.createBuffer({size:b.data.byteLength,usage});
      dev.queue.writeBuffer(gb[b.binding],0,b.data);
      if(b.out) rb[b.binding]=dev.createBuffer({size:b.data.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    }
    const bg=dev.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:bufs.map(b=>({binding:b.binding,resource:{buffer:gb[b.binding]}}))});
    const enc=dev.createCommandEncoder(); const p=enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0,bg); p.dispatchWorkgroups(1); p.end();
    for(const b of bufs) if(b.out) enc.copyBufferToBuffer(gb[b.binding],0,rb[b.binding],0,b.data.byteLength);
    dev.queue.submit([enc.finish()]);
    const o={}; for(const b of bufs) if(b.out){ await rb[b.binding].mapAsync(GPUMapMode.READ); o[b.name]=Array.from(new Float32Array(rb[b.binding].getMappedRange().slice(0))); }
    return {o};
  }
  const T=globalThis.SarekTranspile.transpile;
  const a=new Float32Array(N), b=new Float32Array(N);
  for(let i=0;i<N;i++){a[i]=i*0.5; b[i]=i*2.0;}
  const out=[];
  // Set up scalar uniform data for bounds_check (Params struct: 3 × i32)
  // Layout: sarek_a_length, sarek_b_length, n  (each i32 = 4 bytes = 12 bytes total)
  const BC_N = N/2; // half array in-bounds
  const paramsBC = new Int32Array([N, N, BC_N]); // a_len=N, b_len=N, n=BC_N
  const cases=[
    {k:'vector_add', src:"fun (a:float32 vector)(b:float32 vector)(c:float32 vector) -> let i = global_thread_id in c.(i) <- a.(i) +. b.(i)",
     bufs:[{name:'a',binding:0,data:a.slice()},{name:'b',binding:1,data:b.slice()},{name:'c',binding:2,data:new Float32Array(N),out:true}], ref:i=>a[i]+b[i], outName:'c'},
    {k:'sin', src:"fun (a:float32 vector)(b:float32 vector) -> let i = global_thread_id in b.(i) <- Float32.sin a.(i)",
     bufs:[{name:'a',binding:0,data:a.slice()},{name:'b',binding:1,data:new Float32Array(N),out:true}], ref:i=>Math.sin(a[i]), outName:'b'},
    // bounds_check: exercises EIf → select(else,then,cond) fix.
    // Kernel: b.(i) <- if i < n then a.(i) else 0.0
    // Expected: b[i] = a[i] for i < BC_N, else 0.0
    {k:'bounds_check',
     src:"fun (a:float32 vector)(b:float32 vector)(n:int32) -> let i = global_thread_id in b.(i) <- (if i < n then a.(i) else 0.0)",
     bufs:[
       {name:'a',binding:0,data:a.slice()},
       {name:'b',binding:1,data:new Float32Array(N),out:true},
       {name:'params',binding:2,data:paramsBC,uniform:true}
     ], ref:i=>(i<BC_N?a[i]:0.0), outName:'b'},
  ];
  for(const c of cases){
    const r=T(c.src,"wgsl");
    if(!r.ok){ out.push({k:c.k,ok:false,why:'transpile: '+r.error}); continue; }
    const run=await runWGSL(r.code,c.bufs);
    if(run.err){ out.push({k:c.k,ok:false,why:run.err}); continue; }
    let bad=0; for(let i=0;i<N;i++) if(Math.abs(run.o[c.outName][i]-c.ref(i))>1e-4) bad++;
    out.push({k:c.k,ok:bad===0,bad});
  }
  return out;
});
console.log(JSON.stringify(result));
await browser.close(); server.close();
const allok = Array.isArray(result) && result.length>0 && result.every(r=>r.ok);
console.log(allok ? 'WGSL-GPU: ALL PASS ✓' : 'WGSL-GPU: FAILURE ✗');
process.exit(allok?0:1);
