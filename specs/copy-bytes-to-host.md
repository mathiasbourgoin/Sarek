---
name: roster-spec
type: spec
status: live
feature: Byte-level ranged device-to-host copy (Transfer.copy_bytes_to_host)
brief: requested by a downstream user
date: 2026-10-11
version: 1.0.0
---

# Spec — copy-bytes-to-host

Driven by a downstream LLM inference engine's exclusive RAM tier: a VRAM victim (an `int32`
device-only vector, read by kernels as `int32`) is written back into a slot of
the host pool, a `char` Bigarray. `copy_to_host` needs the destination's
element kind to equal the vector's, and OCaml has no kind-changing blit, so
the copy was not expressible without Obj, Ctypes or a second copy. Decisions
not dictated by the brief are marked *(agent decision)*.

## Clarifications

| Q | A |
|---|---|
| Signature? | `copy_bytes_to_host ?stream ~src ~src_byte_offset dst`: the argument order of `copy_to_host` with a byte offset, the mirror of `copy_bytes_to_device ?stream src ~dst ~dst_byte_offset`. |
| How many bytes? | `Bigarray.Array1.size_in_bytes dst`; slice the destination with `Bigarray.Array1.sub` for a shorter copy. |
| Shared path? | `d2h_range ~op ?stream ~src ~src_byte_offset dst`, the body of today's `copy_to_host`. `copy_to_host` keeps its element-range check (and message) in front of it, as `copy_to_device` does in front of `h2d_range` *(agent decision)*. |
| Obj? | No new use: the ledger root `Obj.repr (dst, src)` moves from `copy_to_host` into `d2h_range`, which takes the Bigarray itself rather than an address and an `Obj.t` *(agent decision)*. |
| Alignment? | None imposed; the backend copies bytes. The downstream engine's offsets are multiples of 4 and 4096. |
| Host destination during a stream copy? | Undefined until the copy has completed (stream synchronised, an event recorded after it synchronised or queried complete, `flush`): reading it may observe old, new or mixed bytes; writing it races with the DMA. SPOC roots `dst` and `src` meanwhile and records `dst`'s byte range as in flight, so `free_pinned` / `unregister_host` of an overlapping pinned buffer are refused until then. HIP/CUDA may stage a stream copy into pageable memory through a driver bounce buffer and block the host call; use pinned memory when overlap matters. |
| Location of `src`? | Unchanged (a read). |
| Backends? | Those with ranged copies (HIP, CUDA, Metal, Vulkan, Native, Interpreter; Vulkan since specs/vulkan-async-api.md). OpenCL, WebGPU refuse with `Feature_not_supported` through `Async.device_to_host`, as for `copy_to_host`. |

## User Story

### US-1: Write back a device range into host bytes (P0)
As the downstream engine's exclusive tier, I want to copy a byte range of an `int32`
device vector into a `char` host slot, synchronously or on a stream.

**Acceptance Scenarios**
1. **Given** an int32 device-only vector filled with a known ramp, **When** `copy_bytes_to_host ~src ~src_byte_offset:o dst` with a `char` dst for o in {0, 4, 4*odd, 4096, end-4}, **Then** dst holds the raw bytes stored in the vector from byte o, with no byte-order conversion (the test derives them with `Bytes.set_int32_ne`, the host's order, which is the order `copy_to_device` stored).
2. **Given** that `char` buffer, **When** copied back with `copy_bytes_to_device` at another offset, **Then** the vector's elements there equal the original ones (round trip), and the rest is unchanged.
3. **Given** a zero-length dst at offset = vector bytes, **Then** accepted, no effect; at offset + 1, refused.
4. **Given** a range past the end or a negative offset, **Then** `Invalid_argument` and dst untouched.
5. **Given** a stream, **When** the copy is enqueued after a launch on the same stream that writes `src`, **Then** after `synchronize_stream` dst holds the kernel's output; while pending, `free_pinned` of a pinned dst is refused.
6. **Given** a destroyed stream, **Then** `Invalid_argument`. (The stream-of-another-device refusal is the `copy_to_host` code moved unchanged into `d2h_range`; not re-tested.)
7. **Given** an OpenCL device, **Then** `Feature_not_supported`.
8. `copy_to_host` behaviour unchanged (existing suite passes; same element error).

## Functional requirements
- FR-001 `copy_bytes_to_host` copies exactly `size_in_bytes dst` bytes from byte `src_byte_offset` of `src` into `dst`.
- FR-002 Range check `0 <= off`, `off + size_in_bytes dst <= length * elem_size` before any device call.
- FR-003 Stream variant: ledger roots `(dst, src)`, records `src.id` and `dst`'s host range; stream liveness and device checks as `copy_to_host`.
- FR-004 `copy_to_host` and `copy_bytes_to_host` share one implementation.
- FR-005 Refusal through the capability model on backends without ranged copies.
