# Plugin Buffer Copies

## Original Risk

Native and interpreter plugin buffer copies used representation conversion to
copy between existential Bigarray values. This worked only if the existential
storage and destination kind were exactly what the surrounding code assumed.

## Implemented Alternative

Buffer copies now use typed `Bigarray.Array1.get` and `Bigarray.Array1.set`
inside the branch that exposes the concrete Bigarray storage. Native pointer
copies also handle Bigarray-backed buffers by byte-copying through Ctypes
Bigarray pointers.

## Why This Improves The Code

The common scalar transfer path is typed and readable. The pointer path still
exists for generic runtime APIs and custom buffers, but it copies bytes from
actual buffer pointers instead of rejecting native Bigarray storage.
