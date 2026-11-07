(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Global vector parameters a kernel never writes, for [ld.global.nc].

    The CUDA-C generator declares every vector parameter [T* __restrict__]: the
    kernel promises that no two vector parameters overlap, and NVVM then selects
    the read-only data path ([ld.global.nc]) for the parameters the kernel never
    stores to. The PTX generator makes the same promise and no other: {!params}
    names the parameters for which that selection is valid under the
    [__restrict__] contract, i.e. those for which nothing in the kernel can
    store.

    The analysis is conservative. A vector parameter is {e not} read-only when
    it is:
    - the target of an assignment ([v.(i) <- e], [v.(i).f <- e]);
    - mentioned in the base expression of an indirect store;
    - an argument of an intrinsic (atomics store through their array argument,
      and an arbitrary intrinsic may) or of a helper call (the helper may store
      through it);
    - the destination of a cooperative-matrix store;
    - or when the kernel contains inline native code, which is opaque: then no
      parameter is read-only.

    Names are matched as written, in the kernel body and in every helper body,
    so a helper that stores through a parameter named like a kernel parameter
    only makes the analysis more conservative. *)

(** [params k] is the names of the global vector parameters of [k] that [k]
    never writes, in declaration order. Empty when [k] contains inline native
    code. *)
val params : Sarek_ir_types.kernel -> string list
