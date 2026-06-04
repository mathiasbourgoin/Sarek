// SPDX-License-Identifier: CECILL-B
// SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
//
// sarek_webgpu_runner.js — Reusable WebGPU runner for Sarek-generated shaders.
//
// Exposes globalThis.SarekWebGPU with:
//   async getAdapter()
//   async run(wgsl, abi, { inputs, scalars }) -> { outputs }
//
// Load with a plain <script> tag (no build step, no ES module import needed).
// The abi argument is the parsed JS object returned by SarekTranspile.transpileWithAbi.

(function () {
  'use strict';

  /**
   * Request a high-performance WebGPU adapter.
   * Returns the adapter or null if WebGPU is unavailable.
   */
  async function getAdapter() {
    if (!navigator.gpu) return null;
    return navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  }

  /**
   * Run a Sarek-generated WGSL compute kernel on the GPU.
   *
   * @param {string} wgsl  - Generated WGSL source code.
   * @param {object} abi   - ABI descriptor (parsed JSON from transpileWithAbi).
   * @param {object} opts
   * @param {object} opts.inputs  - { [bufferName]: TypedArray } initial data per storage buffer.
   * @param {object} opts.scalars - { [name]: number } values for scalar params fields.
   * @returns {{ outputs: { [bufferName]: TypedArray } }}
   */
  async function run(wgsl, abi, { inputs, scalars }) {
    const adapter = await getAdapter();
    if (!adapter) throw new Error('SarekWebGPU: no WebGPU adapter available');
    const device = await adapter.requestDevice();

    // Compile and check for errors.
    const shaderModule = device.createShaderModule({ code: wgsl });
    const compilationInfo = await shaderModule.getCompilationInfo();
    const errors = compilationInfo.messages.filter(m => m.type === 'error');
    if (errors.length > 0) {
      throw new Error(
        'SarekWebGPU: WGSL compile error: ' +
          errors.map(e => e.message + ' @line' + e.lineNum).join(' | ')
      );
    }

    // Build an EXPLICIT bind group / pipeline layout from the ABI rather than
    // layout:'auto'. Auto-layout omits any binding the shader does not
    // statically reference (e.g. the Params uniform for a kernel that uses no
    // lengths/scalars), which would make our bind-group entry for that binding
    // invalid and silently drop the dispatch. An explicit layout keeps every
    // declared binding, so the bind group always matches the shader.
    const layoutEntries = abi.buffers.map(buf => ({
      binding: buf.binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: buf.access === 'read' ? 'read-only-storage' : 'storage' },
    }));
    if (abi.params !== null && abi.params !== undefined) {
      layoutEntries.push({
        binding: abi.params.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      });
    }
    const bindGroupLayout = device.createBindGroupLayout({ entries: layoutEntries });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    // Create typed-array constructors for each element type.
    function typedArrayCtor(elementType) {
      if (elementType === 'f32') return Float32Array;
      if (elementType === 'i32') return Int32Array;
      if (elementType === 'u32') return Uint32Array;
      throw new Error('SarekWebGPU: unknown elementType: ' + elementType);
    }

    // Allocate and upload storage buffers.
    const gpuBuffers = {};
    const readbackBuffers = {};
    for (const buf of abi.buffers) {
      const inputData = inputs[buf.name];
      if (!inputData) {
        throw new Error('SarekWebGPU: missing input for buffer "' + buf.name + '"');
      }
      const gpuBuf = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(gpuBuf, 0, inputData);
      gpuBuffers[buf.name] = gpuBuf;

      // Readback buffer for each storage buffer.
      const rb = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      readbackBuffers[buf.name] = { buffer: rb, byteLength: inputData.byteLength, elementType: buf.elementType };
    }

    // Compute dispatch count from the maximum element count across all inputs.
    let maxElements = 0;
    for (const buf of abi.buffers) {
      const inputData = inputs[buf.name];
      const count = inputData.byteLength / typedArrayCtor(buf.elementType).BYTES_PER_ELEMENT;
      if (count > maxElements) maxElements = count;
    }
    const workgroupX = abi.workgroupSize[0];
    const dispatchCount = Math.ceil(maxElements / workgroupX);

    // Build and upload the uniform params buffer if present.
    const bindGroupEntries = abi.buffers.map(buf => ({
      binding: buf.binding,
      resource: { buffer: gpuBuffers[buf.name] },
    }));

    if (abi.params !== null && abi.params !== undefined) {
      const params = abi.params;
      const uniformData = new ArrayBuffer(params.byteSize);
      for (const field of params.fields) {
        if (field.kind === 'length') {
          // Length of the named vector buffer (element count).
          const vecName = field.of;
          const inputData = inputs[vecName];
          if (!inputData) {
            throw new Error('SarekWebGPU: missing input for length field "' + field.name + '"');
          }
          const ctor = typedArrayCtor(
            abi.buffers.find(b => b.name === vecName).elementType
          );
          const count = inputData.byteLength / ctor.BYTES_PER_ELEMENT;
          new Int32Array(uniformData, field.offset, 1)[0] = count;
        } else {
          // Scalar field — read from scalars map.
          const value = scalars && scalars[field.name] !== undefined
            ? scalars[field.name]
            : 0;
          if (field.type === 'f32') {
            new Float32Array(uniformData, field.offset, 1)[0] = value;
          } else if (field.type === 'u32') {
            new Uint32Array(uniformData, field.offset, 1)[0] = value;
          } else {
            new Int32Array(uniformData, field.offset, 1)[0] = value;
          }
        }
      }
      const uniformBuf = device.createBuffer({
        size: params.byteSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(uniformBuf, 0, uniformData);
      bindGroupEntries.push({ binding: params.binding, resource: { buffer: uniformBuf } });
    }

    // Create bind group and dispatch.
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: bindGroupEntries,
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchCount);
    pass.end();

    // Copy storage buffers to readback buffers.
    for (const buf of abi.buffers) {
      const rb = readbackBuffers[buf.name];
      encoder.copyBufferToBuffer(gpuBuffers[buf.name], 0, rb.buffer, 0, rb.byteLength);
    }

    device.queue.submit([encoder.finish()]);

    // Read back results.
    const outputs = {};
    for (const buf of abi.buffers) {
      const rb = readbackBuffers[buf.name];
      await rb.buffer.mapAsync(GPUMapMode.READ);
      const Ctor = typedArrayCtor(rb.elementType);
      outputs[buf.name] = new Ctor(rb.buffer.getMappedRange().slice(0));
    }

    return { outputs };
  }

  globalThis.SarekWebGPU = { getAdapter, run };
})();
