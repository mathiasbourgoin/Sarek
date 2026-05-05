# PPX Custom Descriptor Qualification

## Original Risk

Removing raw casts made generated code rely on typed custom descriptors such as
`point_custom`. Same-module generated kernels initially qualified those
descriptors with the module currently being compiled, which is not in scope
inside its own implementation.

## Implemented Alternative

Native generation now strips the current module prefix when referring to a
custom descriptor defined in the same compilation unit. The qualified form is
kept for external custom types.

## Why This Improves The Code

The typed descriptor path works for both local and external custom types. This
keeps generated native accessors type-safe without regressing same-module PPX
use cases.
