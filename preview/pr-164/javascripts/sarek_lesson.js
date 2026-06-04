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
    var width      = cfg.width  || 0;
    var height     = cfg.height || 0;
    var n          = cfg.n || (width && height ? width * height : 256);
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
    var canvasEl      = cfg.canvasId      ? document.getElementById(cfg.canvasId)      : null;
    var inputCanvasEl = cfg.inputCanvasId ? document.getElementById(cfg.inputCanvasId) : null;

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
    var env = {
      n: n, width: width, height: height, buffers: buffers, scalars: scalars,
      outName: cfg.outName, reference: cfg.reference, rel: rel, absFloor: absFloor,
      maxBadFraction: cfg.maxBadFraction || 0,
      makeInputs: cfg.makeInputs, colormap: cfg.colormap, onResult: cfg.onResult,
      outputEl: outputEl, statusEl: statusEl, wgslPre: wgslPre,
      canvasEl: canvasEl, inputCanvasEl: inputCanvasEl,
    };
    if (runBtn) {
      runBtn.addEventListener('click', function () { runLesson(getSource(), env); });
    }
    // For visual lessons, render the input preview immediately (if provided).
    if (typeof cfg.renderInput === 'function') {
      try { cfg.renderInput({ n: n, width: width, height: height, canvas: inputCanvasEl }); } catch (_) {}
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

  // ── Build inputs: per-buffer gen, or a whole-image makeInputs override ──
  function buildInputs(env, abi) {
    if (typeof env.makeInputs === 'function') {
      // Visual/image lessons own input construction (e.g. RGB channels or an
      // uploaded photo). Must return { name: TypedArray } for every storage
      // buffer in the ABI (outputs included, zero-filled).
      return env.makeInputs({ n: env.n, width: env.width, height: env.height });
    }
    var inputs = {};
    var buffersCfg = env.buffers;
    for (var bufName in buffersCfg) {
      if (!Object.prototype.hasOwnProperty.call(buffersCfg, bufName)) continue;
      var bcfg = buffersCfg[bufName];
      var abiBuf = null;
      if (abi && abi.buffers) {
        for (var bi = 0; bi < abi.buffers.length; bi++) {
          if (abi.buffers[bi].name === bufName) { abiBuf = abi.buffers[bi]; break; }
        }
      }
      var elemType = (abiBuf && abiBuf.elementType) ? abiBuf.elementType : (bcfg.elementType || 'f32');
      var Ctor = typedArrayCtor(elemType);
      var arr  = new Ctor(env.n);
      if (bcfg.role === 'input' && typeof bcfg.gen === 'function') {
        for (var i = 0; i < env.n; i++) arr[i] = bcfg.gen(i);
      }
      inputs[bufName] = arr;  // outputs stay zero
    }
    return inputs;
  }

  // ── Render a single float output buffer to a canvas via a colormap ─────
  function renderColormap(canvas, data, width, height, colormap) {
    if (!canvas) return;
    canvas.width = width; canvas.height = height;
    var ctx = canvas.getContext('2d');
    var img = ctx.createImageData(width, height);
    for (var i = 0; i < width * height; i++) {
      var rgb = colormap(data[i], i);
      img.data[i * 4]     = rgb[0];
      img.data[i * 4 + 1] = rgb[1];
      img.data[i * 4 + 2] = rgb[2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // ── Core run logic: transpile → build inputs → run → grade + render ────
  async function runLesson(source, env) {
    var outputEl = env.outputEl, statusEl = env.statusEl, wgslPre = env.wgslPre;
    function setStatus(msg) { if (statusEl) statusEl.textContent = msg; }
    function setOutput(cls, msg) {
      if (!outputEl) return;
      outputEl.className = 'lesson-output' + (cls ? ' ' + cls : '');
      outputEl.textContent = msg;
    }

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
    if (wgslPre) wgslPre.textContent = result.code;

    var abi = result.abi, code = result.code;
    var inputs = buildInputs(env, abi);

    setStatus('Running on GPU...');
    var runResult;
    try {
      runResult = await globalThis.SarekWebGPU.run(code, abi, { inputs: inputs, scalars: env.scalars });
    } catch (e) {
      setOutput('error-output', 'GPU run error:\n' + e.message);
      setStatus('GPU error');
      return;
    }

    var gpuOut = runResult.outputs[env.outName];
    if (!gpuOut) {
      setOutput('error-output', 'Output buffer "' + env.outName + '" not found in GPU results.');
      setStatus('Grading error');
      return;
    }

    // Visual rendering (optional): colormap a single output, or a custom hook.
    if (typeof env.colormap === 'function' && env.canvasEl) {
      renderColormap(env.canvasEl, gpuOut, env.width, env.height, env.colormap);
    }
    if (typeof env.onResult === 'function') {
      try {
        env.onResult({
          outputs: runResult.outputs, inputs: inputs, scalars: env.scalars,
          n: env.n, width: env.width, height: env.height,
          canvas: env.canvasEl, inputCanvas: env.inputCanvasEl,
        });
      } catch (e) {
        setOutput('error-output', 'Render error:\n' + e.message);
        setStatus('Render error');
        return;
      }
    }

    // Grading (optional): only when a reference is provided.
    if (typeof env.reference !== 'function') {
      setOutput('', 'Rendered ' + env.n + ' elements on your GPU.');
      setStatus('Done');
      return;
    }

    // maxBadFraction allows a small fraction of mismatches (default 0 = strict).
    // Useful for chaotic kernels (e.g. Mandelbrot) where a few boundary pixels
    // diverge under float32 vs the float64 reference.
    var maxBadFraction = env.maxBadFraction || 0;
    var firstBad = -1, badGot, badRef, badCount = 0;
    for (var idx = 0; idx < env.n; idx++) {
      var ref = env.reference(idx, inputs, env.scalars);
      var got = gpuOut[idx];
      if (!withinTolerance(got, ref, env.rel, env.absFloor)) {
        badCount++;
        if (firstBad === -1) { firstBad = idx; badGot = got; badRef = ref; }
      }
    }
    if (badCount <= maxBadFraction * env.n) {
      var note = badCount === 0
        ? 'all ' + env.n + ' elements match the CPU reference.'
        : badCount + ' / ' + env.n + ' within the allowed tolerance.';
      setOutput('pass-output', 'PASS — ' + note);
      setStatus('PASS');
    } else {
      setOutput(
        'fail-output',
        'FAIL — ' + badCount + ' / ' + env.n + ' elements differ. ' +
          'First at index ' + firstBad +
          ': expected ' + badRef.toPrecision(6) +
          ', got ' + badGot.toPrecision(6)
      );
      setStatus('FAIL');
    }
  }

  globalThis.SarekLesson = { init: init };
})();
