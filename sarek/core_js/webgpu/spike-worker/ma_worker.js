// SPIKE (throwaway) — Milestone A worker (pure JS, no GPU, no OCaml).
// Proves: (2) a classic worker loads under COEP:require-corp, and (3) a SYNCHRONOUS
// round-trip to the main thread via SharedArrayBuffer + Atomics.wait works — the worker
// thread BLOCKS while the non-blocked main thread does the "work" and notifies.
//
// SAB layout (frozen, reused in Milestone B):
//   Int32 control @ byte 0 : [0]=state (IDLE=0,REQUEST=1,READY=2,ERROR=3), [1]=errcode
//   Float32 data  @ byte 8 : a[0..N) | b[N..2N) | c[2N..3N)
'use strict';
const N = 256;
const STATE = 0, ERRCODE = 1;
const IDLE = 0, REQUEST = 1, READY = 2, ERROR = 3;
const TIMEOUT_MS = 10000;

let sab = null, ctrl = null, data = null;

// The whole point: a SYNCHRONOUS call that returns the main-thread-computed result.
function vectorAddSync(a, b) {
  data.set(a, 0);          // a -> [0..N)
  data.set(b, N);          // b -> [N..2N)
  // Store the sentinel BEFORE postMessage and use it as the wait compare value
  // (avoids the lost-wakeup race: if main notifies first, state != REQUEST and
  // Atomics.wait returns "not-equal" immediately).
  Atomics.store(ctrl, STATE, REQUEST);
  postMessage({ t: 'go' });
  const w = Atomics.wait(ctrl, STATE, REQUEST, TIMEOUT_MS);
  if (w === 'timed-out') throw new Error('vectorAddSync: Atomics.wait timed out (deadlock)');
  const state = Atomics.load(ctrl, STATE);
  if (state === ERROR) throw new Error('vectorAddSync: main reported ERROR code ' + Atomics.load(ctrl, ERRCODE));
  if (state !== READY) throw new Error('vectorAddSync: unexpected state ' + state);
  return data.slice(2 * N, 3 * N);   // c -> [2N..3N)
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.t === 'init') {
    sab = m.sab;
    ctrl = new Int32Array(sab, 0, 2);
    data = new Float32Array(sab, 8, 3 * N);
    postMessage({ t: 'ready' });
    return;
  }
  if (m.t === 'run') {
    // Build inputs, do the synchronous round-trip, check vs CPU reference.
    const a = new Float32Array(N), b = new Float32Array(N);
    for (let i = 0; i < N; i++) { a[i] = i * 0.5; b[i] = i * 2.0; }
    try {
      const c = vectorAddSync(a, b);
      let bad = 0;
      for (let i = 0; i < N; i++) if (Math.abs(c[i] - (a[i] + b[i])) > 1e-4) bad++;
      postMessage({ t: 'result', pass: bad === 0, bad });
    } catch (err) {
      postMessage({ t: 'result', pass: false, error: String(err && err.message || err) });
    }
  }
};
