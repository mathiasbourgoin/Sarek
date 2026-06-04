// SPDX-License-Identifier: CECILL-B
// SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
//
// sarek_lesson.js — reusable harness for interactive Sarek GPU course pages.
//
// Exposes globalThis.SarekLesson with:
//   SarekLesson.init(cfg) — wire one lesson page
//
// Per-lesson cfg shape (all fields required unless marked optional):
//
//   SarekLesson.init({
//     editorId,      // id of the <textarea> to upgrade to CodeMirror
//     runButtonId,   // id of the Run <button>
//     outputId,      // id of the result/output element  (class lesson-output)
//     statusId,      // id of the status-bar <span>      (class lesson-status)
//     wgslId,        // id of the <pre> that shows generated WGSL
//     wgslToggleId,  // id of the "Show WGSL" <button>
//     n: 256,        // dataset size (number of elements per buffer)
//     starter: "fun (a:float32 vector) ...",  // initial kernel source
//     buffers: {
//       a: { role: 'input',  elementType: 'f32', gen: i => i * 0.5 },
//       b: { role: 'input',  elementType: 'f32', gen: i => i * 2.0 },
//       c: { role: 'output', elementType: 'f32' },  // gen omitted → zeros
//     },
//     scalars: { /* name: value */ },   // optional, default {}
//     outName: 'c',                     // key in outputs to grade
//     reference: (i, inputs, scalars) => inputs.a[i] + inputs.b[i],
//     tolerance: { rel: 1e-3, absFloor: 1e-3 },  // optional
//   });
//
// Dependencies (loaded by the page before this script):
//   - sarek_transpile.js  → globalThis.SarekTranspile.transpileWithAbi
//   - sarek_webgpu_runner.js → globalThis.SarekWebGPU.getAdapter / .run
//   - CodeMirror 5 (cdnjs) — optional; degrades to plain <textarea> if absent
//   - learn.css — for .lesson-output / .pass-output / .fail-output / etc.

(function () {
  'use strict';

  // ── Typed-array constructor for ABI element types ──────────────────────
  function typedArrayCtor(elementType) {
    if (elementType === 'f32') return Float32Array;
    if (elementType === 'i32') return Int32Array;
    if (elementType === 'u32') return Uint32Array;
    throw new Error('SarekLesson: unknown elementType: ' + elementType);
  }

  // ── Relative-tolerance comparison ─────────────────────────────────────
  //   |got - ref| <= rel * max(|got|, |ref|) + absFloor
  function withinTolerance(got, ref, rel, absFloor) {
    var diff = Math.abs(got - ref);
    var scale = Math.max(Math.abs(got), Math.abs(ref));
    return diff <= rel * scale + absFloor;
  }

  // ── Main init ─────────────────────────────────────────────────────────
  async function init(cfg) {
    var n          = cfg.n || 256;
    var buffers    = cfg.buffers || {};
    var scalars    = cfg.scalars || {};
    var tol        = cfg.tolerance || {};
    var rel        = tol.rel      !== undefined ? tol.rel      : 1e-3;
    var absFloor   = tol.absFloor !== undefined ? tol.absFloor : 1e-3;

    var editorEl    = document.getElementById(cfg.editorId);
    var runBtn      = document.getElementById(cfg.runButtonId);
    var outputEl    = document.getElementById(cfg.outputId);
    var statusEl    = document.getElementById(cfg.statusId);
    var wgslPre     = document.getElementById(cfg.wgslId);
    var wgslToggle  = document.getElementById(cfg.wgslToggleId);

    // ── WebGPU availability check ──────────────────────────────────────
    var adapter = null;
    try {
      adapter = await globalThis.SarekWebGPU.getAdapter();
    } catch (_) {}

    if (!adapter) {
      if (outputEl) {
        outputEl.className = 'lesson-output';
        outputEl.textContent =
          'WebGPU not available in this browser — try Chrome or Edge with ' +
          'WebGPU enabled (chrome://flags/#enable-unsafe-webgpu). ' +
          'The lesson text above is still readable.';
      }
      if (runBtn) runBtn.disabled = true;
      if (statusEl) statusEl.textContent = 'WebGPU unavailable';
      // Still set up the editor so the starter code is visible.
      if (editorEl && cfg.starter) editorEl.value = cfg.starter;
      setupEditor(editorEl, cfg.starter);
      return;
    }

    // ── Editor setup ──────────────────────────────────────────────────
    var cm = setupEditor(editorEl, cfg.starter);

    function getSource() {
      return cm ? cm.getValue() : (editorEl ? editorEl.value : '');
    }

    // ── WGSL toggle ───────────────────────────────────────────────────
    var wgslVisible = false;
    if (wgslPre) wgslPre.style.display = 'none';
    if (wgslToggle) {
      wgslToggle.addEventListener('click', function () {
        wgslVisible = !wgslVisible;
        if (wgslPre) wgslPre.style.display = wgslVisible ? 'block' : 'none';
        wgslToggle.textContent = wgslVisible ? 'Hide WGSL' : 'Show generated WGSL';
      });
    }

    // ── Run handler ───────────────────────────────────────────────────
    if (runBtn) {
      runBtn.addEventListener('click', function () {
        runLesson(
          getSource(), n, buffers, scalars, cfg.outName, cfg.reference,
          rel, absFloor, outputEl, statusEl, wgslPre
        );
      });
    }
  }

  // ── Editor upgrade (CodeMirror 5 or textarea fallback) ────────────────
  function setupEditor(el, starter) {
    if (!el) return null;
    if (starter !== undefined) el.value = starter;
    if (typeof CodeMirror === 'undefined') return null;
    return CodeMirror.fromTextArea(el, {
      mode:          'text/x-ocaml',
      lineNumbers:   true,
      lineWrapping:  true,
      viewportMargin: Infinity,
      tabSize:       2,
    });
  }

  // ── Core run/grade logic ──────────────────────────────────────────────
  async function runLesson(
    source, n, buffersCfg, scalarsCfg, outName, reference,
    rel, absFloor, outputEl, statusEl, wgslPre
  ) {
    function setStatus(msg) { if (statusEl) statusEl.textContent = msg; }
    function setOutput(cls, msg) {
      if (!outputEl) return;
      outputEl.className = 'lesson-output' + (cls ? ' ' + cls : '');
      outputEl.textContent = msg;
    }

    // 1. Transpile
    if (typeof globalThis.SarekTranspile === 'undefined') {
      setOutput('error-output', 'Transpiler not loaded — please reload the page.');
      return;
    }
    setStatus('Transpiling...');
    var result = globalThis.SarekTranspile.transpileWithAbi(source, 'wgsl');
    if (!result.ok) {
      setOutput('error-output', 'Transpile error:\n' + (result.error || 'unknown error'));
      setStatus('Transpile error');
      return;
    }

    // Show generated WGSL if the pre element exists.
    if (wgslPre) wgslPre.textContent = result.code;

    // 2. Build inputs from cfg + ABI
    var abi    = result.abi;
    var code   = result.code;
    var inputs = {};

    for (var bufName in buffersCfg) {
      if (!Object.prototype.hasOwnProperty.call(buffersCfg, bufName)) continue;
      var bcfg = buffersCfg[bufName];

      // Determine elementType: prefer ABI (authoritative) then cfg fallback.
      var abiBuf = null;
      if (abi && abi.buffers) {
        for (var bi = 0; bi < abi.buffers.length; bi++) {
          if (abi.buffers[bi].name === bufName) { abiBuf = abi.buffers[bi]; break; }
        }
      }
      var elemType = (abiBuf && abiBuf.elementType) ? abiBuf.elementType : (bcfg.elementType || 'f32');
      var Ctor = typedArrayCtor(elemType);
      var arr  = new Ctor(n);

      if (bcfg.role === 'input' && typeof bcfg.gen === 'function') {
        for (var i = 0; i < n; i++) arr[i] = bcfg.gen(i);
      }
      // outputs start as zeros (already zero-initialised by TypedArray)
      inputs[bufName] = arr;
    }

    // 3. Run on GPU
    setStatus('Running on GPU...');
    var runResult;
    try {
      runResult = await globalThis.SarekWebGPU.run(code, abi, { inputs: inputs, scalars: scalarsCfg });
    } catch (e) {
      setOutput('error-output', 'GPU run error:\n' + e.message);
      setStatus('GPU error');
      return;
    }

    // 4. Grade
    var gpuOut = runResult.outputs[outName];
    if (!gpuOut) {
      setOutput('error-output', 'Output buffer "' + outName + '" not found in GPU results.');
      setStatus('Grading error');
      return;
    }

    var firstBad = -1;
    var badGot, badRef;
    for (var idx = 0; idx < n; idx++) {
      var ref = reference(idx, inputs, scalarsCfg);
      var got = gpuOut[idx];
      if (!withinTolerance(got, ref, rel, absFloor)) {
        firstBad = idx;
        badGot = got;
        badRef = ref;
        break;
      }
    }

    if (firstBad === -1) {
      setOutput('pass-output', 'PASS — all ' + n + ' elements match the CPU reference.');
      setStatus('PASS');
    } else {
      setOutput(
        'fail-output',
        'FAIL at index ' + firstBad +
          ': expected ' + badRef.toPrecision(6) +
          ', got ' + badGot.toPrecision(6)
      );
      setStatus('FAIL');
    }
  }

  globalThis.SarekLesson = { init: init };
})();
