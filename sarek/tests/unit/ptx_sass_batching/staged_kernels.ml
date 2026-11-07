(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A quantised-weight matrix-vector product, one wave of 32 lanes per row, with
   the activations (and, in the first variant, the 8-bit-quant weights too)
   staged through shared memory in chunks of 32 blocks, behind barriers. The
   two variants run together as one horizontally fused kernel. Shared by the
   SASS probe and the address-arithmetic tests. *)

(* x and the weights staged. *)
let staged_xw =
  [%kernel
    fun (er : int32 vector)
        (slots : int32 vector)
        (rec_words : int32)
        (o_d : int32)
        (o_qs : int32)
        (x : float32 vector)
        (x_stride : int32)
        (y : float32 vector)
        (rows : int32)
        (nb : int32) ->
      let%shared (xs : float32) = 1056l in
      let%shared (ws : int32) = 2304l in
      let%shared (red : float32) = 256l in
      let tid = thread_idx_x in
      let lane = tid land 31l in
      let wave = tid / 32l in
      let row = (block_idx_x * 8l) + wave in
      let base = slots.(block_idx_y) * rec_words in
      let xo = block_idx_y * x_stride in
      let yo = block_idx_y * rows in
      let niter = (nb + 31l) / 32l in
      let acc = mut 0.0 in
      for it = 0l to niter - 1l do
        let b0 = it * 32l in
        let%superstep stage =
          for q = 0 to 3l do
            let f = tid + (q * 256l) in
            if b0 + (f / 32l) < nb then
              xs.(f + (f / 32l)) <- x.(xo + (b0 * 32l) + f)
          done ;
          if row < rows then
            for q = 0 to 7l do
              let w = lane + (q * 32l) in
              if b0 + (w / 8l) < nb then
                ws.((wave * 288l) + (w / 8l * 9l) + (w land 7l)) <-
                  er.(base + o_qs + (((row * nb) + b0) * 8l) + w)
            done
        in
        let%superstep compute =
          let b = b0 + lane in
          if row < rows && b < nb then begin
            let blk = (row * nb) + b in
            let xb = lane * 33l in
            let wb = (wave * 288l) + (lane * 9l) in
            let s = mut 0.0 in
            for wi = 0 to 7l do
              let word = ws.(wb + wi) in
              let xi = xb + (wi * 4l) in
              let q0 = float ((word lsl 24l) asr 24l) in
              let q1 = float ((word lsl 16l) asr 24l) in
              let q2 = float ((word lsl 8l) asr 24l) in
              let q3 = float (word asr 24l) in
              s :=
                s
                +. (q0 *. xs.(xi))
                +. (q1 *. xs.(xi + 1l))
                +. (q2 *. xs.(xi + 2l))
                +. (q3 *. xs.(xi + 3l))
            done ;
            acc := acc +. (Int32.float_of_bits er.(base + o_d + blk) *. s)
          end
        in
        ()
      done ;
      let%superstep partial = red.(tid) <- acc in
      let%superstep r16 =
        if lane < 16l then red.(tid) <- red.(tid) +. red.(tid + 16l)
      in
      let%superstep r8 =
        if lane < 8l then red.(tid) <- red.(tid) +. red.(tid + 8l)
      in
      let%superstep r4 =
        if lane < 4l then red.(tid) <- red.(tid) +. red.(tid + 4l)
      in
      let%superstep r2 =
        if lane < 2l then red.(tid) <- red.(tid) +. red.(tid + 2l)
      in
      if lane = 0l && row < rows then
        y.(yo + row) <- red.(tid) +. red.(tid + 1l)]

(* x staged, the weights read directly as in {!Matvec.q8_0_wave}. *)
let staged_x =
  [%kernel
    fun (er : int32 vector)
        (slots : int32 vector)
        (rec_words : int32)
        (o_d : int32)
        (o_qs : int32)
        (x : float32 vector)
        (x_stride : int32)
        (y : float32 vector)
        (rows : int32)
        (nb : int32) ->
      let%shared (xs : float32) = 1056l in
      let%shared (red : float32) = 256l in
      let tid = thread_idx_x in
      let lane = tid land 31l in
      let wave = tid / 32l in
      let row = (block_idx_x * 8l) + wave in
      let base = slots.(block_idx_y) * rec_words in
      let xo = block_idx_y * x_stride in
      let yo = block_idx_y * rows in
      let niter = (nb + 31l) / 32l in
      let acc = mut 0.0 in
      for it = 0l to niter - 1l do
        let b0 = it * 32l in
        let%superstep stage =
          for q = 0 to 3l do
            let f = tid + (q * 256l) in
            if b0 + (f / 32l) < nb then
              xs.(f + (f / 32l)) <- x.(xo + (b0 * 32l) + f)
          done
        in
        let%superstep compute =
          let b = b0 + lane in
          if row < rows && b < nb then begin
            let blk = (row * nb) + b in
            let xb = lane * 33l in
            let wb = base + o_qs + (blk * 8l) in
            let s = mut 0.0 in
            for wi = 0 to 7l do
              let word = er.(wb + wi) in
              let xi = xb + (wi * 4l) in
              let q0 = float ((word lsl 24l) asr 24l) in
              let q1 = float ((word lsl 16l) asr 24l) in
              let q2 = float ((word lsl 8l) asr 24l) in
              let q3 = float (word asr 24l) in
              s :=
                s
                +. (q0 *. xs.(xi))
                +. (q1 *. xs.(xi + 1l))
                +. (q2 *. xs.(xi + 2l))
                +. (q3 *. xs.(xi + 3l))
            done ;
            acc := acc +. (Int32.float_of_bits er.(base + o_d + blk) *. s)
          end
        in
        ()
      done ;
      let%superstep partial = red.(tid) <- acc in
      let%superstep r16 =
        if lane < 16l then red.(tid) <- red.(tid) +. red.(tid + 16l)
      in
      let%superstep r8 =
        if lane < 8l then red.(tid) <- red.(tid) +. red.(tid + 8l)
      in
      let%superstep r4 =
        if lane < 4l then red.(tid) <- red.(tid) +. red.(tid + 4l)
      in
      let%superstep r2 =
        if lane < 2l then red.(tid) <- red.(tid) +. red.(tid + 2l)
      in
      if lane = 0l && row < rows then
        y.(yo + row) <- red.(tid) +. red.(tid + 1l)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* The two variants fused: grid x is the sum of the members'. *)
let fused () =
  Sarek_execute.Fusion_horizontal.fuse
    ~name:"hfused_staged"
    [ir_of staged_xw; ir_of staged_x]
