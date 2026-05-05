# Legacy Native Direct API

## Original Risk

`Native_plugin.run_kernel_direct` still accepted raw runtime object arrays even
after the main native registry used typed execution arguments.

## Implemented Alternative

The function now accepts `Framework_sig.exec_arg array`, matching the native
kernel registry and the runtime launch path.

## Why This Improves The Code

There is one native direct-call ABI instead of a safe path plus a legacy unsafe
path. This is a breaking change, but it reduces maintenance cost and removes the
last active raw-object API surface in the native plugin.
