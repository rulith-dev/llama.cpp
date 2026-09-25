#include "gated_delta_net.cuh"
#include "common.cuh"

#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
template <int mask>
static __device__ __forceinline__ float gdn_dpp_row_xmask(const float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
}
static __device__ __forceinline__ float gdn_permlanex16_swap(const float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false));
}
static __device__ __forceinline__ float gdn_warp_reduce_sum32(float x) {
    x += gdn_permlanex16_swap(x);
    x += gdn_dpp_row_xmask<8>(x);
    x += gdn_dpp_row_xmask<4>(x);
    x += gdn_dpp_row_xmask<2>(x);
    x += gdn_dpp_row_xmask<1>(x);
    return x;
}
#define GDN_DPP_REDUCE 1
#endif

template <int width>
static __device__ __forceinline__ float gdn_warp_reduce_sum(const float x) {
#if defined(GDN_DPP_REDUCE)
    if constexpr (width == 32) {
        return gdn_warp_reduce_sum32(x);
    }
#endif
    return warp_reduce_sum<width>(x);
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_rows) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // strixllama: with state_rows, curr_state is the whole state tensor and the sequence's row is looked up
    const int64_t state_in_offset      = (state_rows ? (int64_t) state_rows[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <int S_v, int NUM_WARPS, int COLS, int TOKEN_TILE, bool keep_rs_t>
__global__ void __launch_bounds__(32 * NUM_WARPS, 1)
gated_delta_net_tiled_cuda(const float * q,
                           const float * k,
                           const float * v,
                           const float * g,
                           const float * beta,
                           const float * curr_state,
                           float *       dst,
                           float *       state,
                           int64_t       H,
                           int64_t       n_tokens,
                           int64_t       sq1,
                           int64_t       sq2,
                           int64_t       sq3,
                           int64_t       sv1,
                           int64_t       sv2,
                           int64_t       sv3,
                           int64_t       sb1,
                           int64_t       sb2,
                           int64_t       sb3,
                           const uint3   neqk1_magic,
                           const uint3   rq3_magic,
                           float         scale,
                           int64_t       state_slot_stride,
                           int           K,
                           const int32_t * state_rows) {
    constexpr int warp_size     = 32;
    constexpr int rows_per_lane = S_v / warp_size;
    constexpr int block_cols    = NUM_WARPS * COLS;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of the warp size");
    static_assert(S_v % block_cols == 0, "block columns must divide S_v");

    __shared__ float q_shared[TOKEN_TILE][S_v];
    __shared__ float k_shared[TOKEN_TILE][S_v];
    __shared__ float v_shared[TOKEN_TILE][block_cols];
    __shared__ float g_shared[TOKEN_TILE];
    __shared__ float beta_shared[TOKEN_TILE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = blockIdx.z * block_cols;          // first column of the block
    const int      colw     = threadIdx.y * COLS;               // first column of this warp inside the block
    const int      thread   = threadIdx.y * warp_size + lane;
    constexpr int  nthreads = NUM_WARPS * warp_size;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = (state_rows ? (int64_t) state_rows[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + (col0 + colw) * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col0 + colw;

    float s_shard[COLS][rows_per_lane];

#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[c * S_v + r * warp_size + lane];
        }
    }

    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile_size = min((int64_t) TOKEN_TILE, n_tokens - t0);

        for (int idx = thread; idx < tile_size * S_v; idx += nthreads) {
            const int tt = idx / S_v;
            const int i  = idx % S_v;
            const int t  = t0 + tt;
            q_shared[tt][i] = q[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
            k_shared[tt][i] = k[iq3 * sq3 + t * sq2 + iq1 * sq1 + i];
        }
        for (int idx = thread; idx < tile_size * block_cols; idx += nthreads) {
            const int tt = idx / block_cols;
            const int c  = idx % block_cols;
            const int t  = t0 + tt;
            v_shared[tt][c] = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col0 + c];
        }
        if (thread < tile_size) {
            const int64_t gb_offset = sequence * sb3 + (t0 + thread) * sb2 + h_idx * sb1;
            g_shared[thread]    = g[gb_offset];
            beta_shared[thread] = beta[gb_offset];
        }
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_shared[tt][i];
                q_reg[r] = q_shared[tt][i];
            }

            const float g_val    = expf(g_shared[tt]);
            const float beta_val = beta_shared[tt];

            float attn_col[COLS];
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard = fmaf(s_shard[c][r], k_reg[r], kv_shard);
                }
                const float kv_col = gdn_warp_reduce_sum<warp_size>(kv_shard);

                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = fmaf(-g_val, kv_col, v_shared[tt][colw + c]) * beta_val;

                // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
                    attn_partial  = fmaf(s_shard[c][r], q_reg[r], attn_partial);
                }
                attn_col[c] = gdn_warp_reduce_sum<warp_size>(attn_partial);
            }

            if (lane < COLS) {
                float a = attn_col[0];
#pragma unroll
                for (int c = 1; c < COLS; c++) {
                    a = lane == c ? attn_col[c] : a;
                }
                attn_data[(int64_t) t * S_v * H + lane] = a * scale;
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * snapshot = state + target_slot * state_slot_stride + (col0 + colw) * S_v;
#pragma unroll
                    for (int c = 0; c < COLS; c++) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            snapshot[c * S_v + r * warp_size + lane] = s_shard[c][r];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < COLS; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + colw + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}

// strixllama: the prefill recurrence with 16 state rows and C_LANE columns a lane (8 lanes share a column group), so a
// dot product over the 128 rows is 16 FMAs and a 3-step DPP reduction instead of 4 FMAs and 5 steps. The output
// comes from the state before the update, o = scale * (g * S^T q + delta * (k . q)) - the same value as scale * S'^T q
// - so both reductions of a token (S^T k and S^T q, every column at once) run together and nothing waits on the update.
// The tiled kernel above spends ~3.3 ms a layer at 2K tokens on latency; this one does the same arithmetic in ~1/3.
template <int mask>
static __device__ __forceinline__ float gdn_xor8_step(const float x) {
#if defined(GDN_DPP_REDUCE)
    return gdn_dpp_row_xmask<mask>(x);
#else
    return __shfl_xor(x, mask, 32);
#endif
}

static __device__ __forceinline__ float gdn_sum8(float x) {   // over the 8 lanes that differ in bits 0-2
    x += gdn_xor8_step<1>(x);
    x += gdn_xor8_step<2>(x);
    x += gdn_xor8_step<4>(x);
    return x;
}

template <int C_LANE, int NWAVES, int TOKEN_TILE, bool keep_rs_t>
__global__ void __launch_bounds__(32 * NWAVES)
gated_delta_net_r16_cuda(const float * q,
                         const float * k,
                         const float * v,
                         const float * g,
                         const float * beta,
                         const float * curr_state,
                         float *       dst,
                         float *       state,
                         int64_t       H,
                         int64_t       n_tokens,
                         int64_t       sq1,
                         int64_t       sq2,
                         int64_t       sq3,
                         int64_t       sv1,
                         int64_t       sv2,
                         int64_t       sv3,
                         int64_t       sb1,
                         int64_t       sb2,
                         int64_t       sb3,
                         const uint3   neqk1_magic,
                         const uint3   rq3_magic,
                         float         scale,
                         int64_t       state_slot_stride,
                         int           K,
                         const int32_t * state_rows) {
    constexpr int S_v    = 128;
    constexpr int RL     = 16;                  // state rows a lane
    constexpr int WCOLS  = 4 * C_LANE;          // a wave: 4 column groups of C_LANE, each over 8 lanes x 16 rows
    constexpr int BCOLS  = NWAVES * WCOLS;
    static_assert(S_v % BCOLS == 0, "block columns must divide S_v");

    // one tile: a second one (loads in flight while this one is consumed) doubles the LDS a workgroup takes, and then
    // fewer workgroups share a CU - with this kernel's per-token latency the co-resident count is what sets the time
    __shared__ __align__(16) float q_shared[1][TOKEN_TILE][S_v];
    __shared__ __align__(16) float k_shared[1][TOKEN_TILE][S_v];
    __shared__ __align__(16) float v_shared[1][TOKEN_TILE][BCOLS];
    __shared__ float g_shared[1][TOKEN_TILE];
    __shared__ float beta_shared[1][TOKEN_TILE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x & 31;
    const int      wave     = threadIdx.x >> 5;
    const int      rg       = lane & 7;          // rows 16 rg .. 16 rg + 15
    const int      cg       = lane >> 3;
    const int      col0     = blockIdx.z * BCOLS;
    const int      colb     = wave * WCOLS + cg * C_LANE;   // this lane's first column inside the block
    const int      thread   = threadIdx.x;
    constexpr int  nthreads = NWAVES * 32;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = (state_rows ? (int64_t) state_rows[sequence] : (int64_t) sequence) * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col0 + colb;

    // s[c][r] = S[16 rg + r][column c] = M[column c][16 rg + r]: the stored (transposed) rows are contiguous. The
    // registers hold the state divided by gm, the decay since it was last folded in: S = gm * s
    float s[C_LANE][RL];
    float gm = 1.0f;
#pragma unroll
    for (int c = 0; c < C_LANE; c++) {
        const float4 * src = (const float4 *) (curr_state + (int64_t) (col0 + colb + c) * S_v + RL * rg);
#pragma unroll
        for (int r4 = 0; r4 < RL / 4; r4++) {
            const float4 x = src[r4];
            s[c][4*r4 + 0] = x.x; s[c][4*r4 + 1] = x.y; s[c][4*r4 + 2] = x.z; s[c][4*r4 + 3] = x.w;
        }
    }

    // the tile loader: float4s of q and k (whole rows) and of v (this block's columns), g and beta per token
    constexpr int QK4 = TOKEN_TILE * S_v / 4;
    constexpr int V4  = TOKEN_TILE * BCOLS / 4;
    constexpr int NQ  = (QK4 + nthreads - 1) / nthreads;
    constexpr int NV  = (V4 + nthreads - 1) / nthreads;
    float4 rq[NQ], rk[NQ], rv[NV];
    float  rgv = 0.0f, rbv = 0.0f;
    auto fetch = [&](const int t0) {
        const int tile = (int) min((int64_t) TOKEN_TILE, n_tokens - t0);
#pragma unroll
        for (int j = 0; j < NQ; j++) {
            const int idx = thread + j * nthreads, tt = idx / (S_v / 4), i4 = idx % (S_v / 4);
            if (idx < QK4 && tt < tile) {
                const int64_t off = iq3 * sq3 + (int64_t) (t0 + tt) * sq2 + iq1 * sq1 + 4 * i4;
                rq[j] = *(const float4 *) (q + off);
                rk[j] = *(const float4 *) (k + off);
            }
        }
#pragma unroll
        for (int j = 0; j < NV; j++) {
            const int idx = thread + j * nthreads, tt = idx / (BCOLS / 4), c4 = idx % (BCOLS / 4);
            if (idx < V4 && tt < tile) {
                rv[j] = *(const float4 *) (v + sequence * sv3 + (int64_t) (t0 + tt) * sv2 + h_idx * sv1 + col0 + 4 * c4);
            }
        }
        if (thread < tile) {
            const int64_t gb_offset = sequence * sb3 + (t0 + thread) * sb2 + h_idx * sb1;
            rgv = g[gb_offset];
            rbv = beta[gb_offset];
        }
    };
    auto stash = [&](const int buf, const int t0) {
        const int tile = (int) min((int64_t) TOKEN_TILE, n_tokens - t0);
#pragma unroll
        for (int j = 0; j < NQ; j++) {
            const int idx = thread + j * nthreads, tt = idx / (S_v / 4), i4 = idx % (S_v / 4);
            if (idx < QK4 && tt < tile) {
                *(float4 *) &q_shared[buf][tt][4 * i4] = rq[j];
                *(float4 *) &k_shared[buf][tt][4 * i4] = rk[j];
            }
        }
#pragma unroll
        for (int j = 0; j < NV; j++) {
            const int idx = thread + j * nthreads, tt = idx / (BCOLS / 4), c4 = idx % (BCOLS / 4);
            if (idx < V4 && tt < tile) {
                *(float4 *) &v_shared[buf][tt][4 * c4] = rv[j];
            }
        }
        if (thread < tile) {
            g_shared[buf][thread]    = rgv;
            beta_shared[buf][thread] = rbv;
        }
    };

    const int buf = 0;
    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile_size = min((int64_t) TOKEN_TILE, n_tokens - t0);
        fetch(t0);
        stash(0, t0);
        __syncthreads();

        for (int tt = 0; tt < tile_size; ++tt) {
            const int t = t0 + tt;

            float k_reg[RL], q_reg[RL];
#pragma unroll
            for (int r4 = 0; r4 < RL / 4; r4++) {
                const float4 kx = *(const float4 *) &k_shared[buf][tt][RL * rg + 4 * r4];
                const float4 qx = *(const float4 *) &q_shared[buf][tt][RL * rg + 4 * r4];
                k_reg[4*r4 + 0] = kx.x; k_reg[4*r4 + 1] = kx.y; k_reg[4*r4 + 2] = kx.z; k_reg[4*r4 + 3] = kx.w;
                q_reg[4*r4 + 0] = qx.x; q_reg[4*r4 + 1] = qx.y; q_reg[4*r4 + 2] = qx.z; q_reg[4*r4 + 3] = qx.w;
            }

            // S^T k and S^T q for every column, and k . q, over this lane's rows; then over the 8 row groups
            float kv[C_LANE], qs[C_LANE], kq = 0.0f;
#pragma unroll
            for (int c = 0; c < C_LANE; c++) {
                kv[c] = 0.0f;
                qs[c] = 0.0f;
#pragma unroll
                for (int r = 0; r < RL; r++) {
                    kv[c] = fmaf(s[c][r], k_reg[r], kv[c]);
                    qs[c] = fmaf(s[c][r], q_reg[r], qs[c]);
                }
            }
#pragma unroll
            for (int r = 0; r < RL; r++) {
                kq = fmaf(k_reg[r], q_reg[r], kq);
            }
#pragma unroll
            for (int c = 0; c < C_LANE; c++) {
                kv[c] = gdn_sum8(kv[c]);
                qs[c] = gdn_sum8(qs[c]);
            }
            kq = gdn_sum8(kq);

            const float g_val    = expf(g_shared[buf][tt]);
            const float beta_val = beta_shared[buf][tt];
            // kv and qs were taken from the scaled state: S^T k = gm * kv, so g * S^T k = gt * kv
            const float gt   = gm * g_val;
            const bool  fold = gt < 0x1p-20f;   // uniform: the decay is the head's

            float o[C_LANE], dl[C_LANE];
#pragma unroll
            for (int c = 0; c < C_LANE; c++) {
                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta = fmaf(-gt, kv[c], v_shared[buf][tt][colb + c]) * beta_val;
                o[c]  = fmaf(gt, qs[c], delta * kq) * scale;
                dl[c] = delta;
            }
            if (!fold) {
                // S' = g S + k delta^T = gt * (Sh + k (delta / gt)^T): one FMA an element
                const float inv = 1.0f / gt;
#pragma unroll
                for (int c = 0; c < C_LANE; c++) {
                    const float dc = dl[c] * inv;
#pragma unroll
                    for (int r = 0; r < RL; r++) {
                        s[c][r] = fmaf(k_reg[r], dc, s[c][r]);
                    }
                }
                gm = gt;
            } else {
                // the scale is about to leave the float range the updates keep: fold it into the state
#pragma unroll
                for (int c = 0; c < C_LANE; c++) {
#pragma unroll
                    for (int r = 0; r < RL; r++) {
                        s[c][r] = fmaf(gt, s[c][r], k_reg[r] * dl[c]);
                    }
                }
                gm = 1.0f;
            }

            if (rg == 0) {
                float * out = attn_data + (int64_t) t * S_v * H;
#pragma unroll
                for (int c = 0; c < C_LANE; c++) {
                    out[c] = o[c];
                }
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * snapshot = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int c = 0; c < C_LANE; c++) {
                        float4 * d4 = (float4 *) (snapshot + (int64_t) (col0 + colb + c) * S_v + RL * rg);
#pragma unroll
                        for (int r4 = 0; r4 < RL / 4; r4++) {
                            d4[r4] = make_float4(gm * s[c][4*r4 + 0], gm * s[c][4*r4 + 1], gm * s[c][4*r4 + 2], gm * s[c][4*r4 + 3]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < C_LANE; c++) {
            float4 * d4 = (float4 *) (state + (int64_t) (col0 + colb + c) * S_v + RL * rg);
#pragma unroll
            for (int r4 = 0; r4 < RL / 4; r4++) {
                d4[r4] = make_float4(gm * s[c][4*r4 + 0], gm * s[c][4*r4 + 1], gm * s[c][4*r4 + 2], gm * s[c][4*r4 + 3]);
            }
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * state_rows, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    if constexpr (!KDA) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        // STRIX_GDN_R16: waves a workgroup for the 16-row kernel (2, 4 or 8; 0 = the tiled kernel). At 2K tokens a layer
        // takes 2.7 ms with 4 (96 workgroups), 2.9 with 8 and 4.6 with 2 (each workgroup loads the whole q/k tile);
        // the tiled kernel 3.4
        static const int r16 = getenv("STRIX_GDN_R16") ? atoi(getenv("STRIX_GDN_R16")) : 4;
        // the tile loader reads q, k and v in float4s
        const bool vec4 = ((uintptr_t) q_d) % 16 == 0 && ((uintptr_t) k_d) % 16 == 0 && ((uintptr_t) v_d) % 16 == 0 &&
            sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0 && sv3 % 4 == 0;
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && warp_size == 32 && S_v == 128 && n_seqs == 1 && n_tokens >= 16 &&
                n_tokens <= 32768 && vec4 && (r16 == 2 || r16 == 4 || r16 == 8)) {
            static unsigned hits = 0;
            if (hits++ == 0) fprintf(stderr, "FORK_GDN_R16 tokens=%ld waves=%d snapshots=%d keep=%d\n", n_tokens, r16, K, int(keep_rs_t));
            const dim3 grid(H, n_seqs, 128 / (16 * r16)), block(32 * r16, 1, 1);
            const ggml_cuda_kernel_launch_params params(grid, block, 0, stream);
#define GDN_R16_LAUNCH(NW) ggml_cuda_kernel_launch(gated_delta_net_r16_cuda<4, NW, 16, keep_rs_t>, params,                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,                 neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows)
            if (r16 == 2) {
                GDN_R16_LAUNCH(2);
            } else if (r16 == 4) {
                GDN_R16_LAUNCH(4);
            } else {
                GDN_R16_LAUNCH(8);
            }
#undef GDN_R16_LAUNCH
            return;
        }
        if (GGML_CUDA_CC_IS_RDNA3_5(cc) && S_v == 128 && H == 48 && n_seqs == 1 && n_tokens >= 16 && n_tokens <= 32768) {
            const dim3 tiled_grid(H, n_seqs, 2);
            const dim3 tiled_block(warp_size, 8, 1);
            const ggml_cuda_kernel_launch_params tiled_params(tiled_grid, tiled_block, 0, stream);
            static unsigned hits = 0;
            if (hits++ == 0) fprintf(stderr, "FORK_GDN_TILE tokens=%ld snapshots=%d keep=%d\n", n_tokens, K, int(keep_rs_t));
            ggml_cuda_kernel_launch(gated_delta_net_tiled_cuda<128, 8, 8, 16, keep_rs_t>, tiled_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    // strixllama: the states read straight from the cache rows the skipped gather would have copied
    const int32_t * state_rows = cache ? cache->state_rows : nullptr;
    const float *   s_d        = state_rows ? cache->state_base : (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
