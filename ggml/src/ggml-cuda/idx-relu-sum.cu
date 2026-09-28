#include "idx-relu-sum.cuh"
#include <cstdlib>

// Lightning-indexer head reduction: relu each head's block score, then sum the heads in graph order.
// Reproduces unary op_relu (fmaxf(x, 0)) followed by (((r0 + r1) + r2) + ...) with the same rounding, so the result is
// bitwise identical to the RELU + CONT + ADD chain it replaces, while reading the scores once instead of four times.
static __global__ void idx_relu_sum_f32(const float * __restrict__ src, float * __restrict__ dst,
                                        const int n_blocks, const int heads) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int64_t tt = blockIdx.y;                         // token within stream, streams stacked
    const float * p  = src + tt * (int64_t) heads * n_blocks + b;
    float acc = fmaxf(p[0], 0.0f);
    for (int h = 1; h < heads; ++h) {
        acc = acc + fmaxf(p[(int64_t) h * n_blocks], 0.0f);
    }
    dst[tt * (int64_t) n_blocks + b] = acc;
}

bool ggml_cuda_idx_relu_sum_enabled() {
    static const int v = getenv("LLAMA_IDX_RELU_SUM") ? atoi(getenv("LLAMA_IDX_RELU_SUM")) : 0;
    return v != 0;
}

void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args) {
    const ggml_tensor * s = args.score;
    const int n_blocks = (int) s->ne[0];
    const int64_t rows = s->ne[2] * s->ne[3];
    constexpr int threads = 256;
    const dim3 grid((n_blocks + threads - 1) / threads, (unsigned) rows, 1);
    idx_relu_sum_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) s->data, (float *) args.dst->data, n_blocks, args.heads);
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0;
    if (hits++ < 2) fprintf(stderr, "IDX_RELU_SUM blocks=%d heads=%d rows=%lld\n", n_blocks, args.heads, (long long) rows);
}

// strixllama: the lightning indexer's score for one query strip, whole. out[t][b] = tails[t] > starts[b] ?
// ((relu(s0) + relu(s1)) + relu(s2)) + relu(s3) : -inf, s_h = q[t][h] . keys[b] over 128 dims - what the graph computes
// as MUL_MAT -> RELU -> head sum -> score + log(step(tail - start)), without its [blocks x heads x queries] F32
// intermediate (137 MB a strip at 64K tokens) and the four passes over it. The dot products are taken as the tile GEMM's
// two-product mode takes them (q split into F16 hi + lo, the keys - F16 values out of the block-key cache - as F16, F32
// accumulation over 16-deep steps in order), so the result is the unfused chain's.
// A block: 8 waves x 4 queries against 128 keys staged in LDS. A wave's 4 queries x 4 heads are the 16 rows of its WMMA
// tile (row = head * 4 + query) against 16 keys at a time; each lane then holds all four heads of two queries for one key.
typedef short idxs_v16s __attribute__((ext_vector_type(16)));
typedef float idxs_v8f  __attribute__((ext_vector_type(8)));
#define IDXS_KB 128
#define IDXS_LS (128 + 8)

static __device__ __forceinline__ uint32_t idxs_h2(const float a, const float b) {
    return (uint32_t) __builtin_bit_cast(uint16_t, (_Float16) a) | ((uint32_t) __builtin_bit_cast(uint16_t, (_Float16) b) << 16);
}

static __global__ void __launch_bounds__(256) k_idx_score(const float * __restrict__ keys, const int64_t ks, const int nb,
        const float * __restrict__ q, const int64_t qs, const int nq,
        const int32_t * __restrict__ starts, const int32_t * __restrict__ tails, float * __restrict__ out) {
    __shared__ __align__(16) uint16_t kt[IDXS_KB * IDXS_LS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int b0 = blockIdx.y * IDXS_KB;
    for (int i = tid; i < IDXS_KB * 32; i += 256) {                    // 32 float4 a key
        const int r = i >> 5, c4 = (i & 31) * 4;
        const float4 v = *(const float4 *) (keys + (int64_t) min(b0 + r, nb - 1) * ks + c4);
        *(uint2 *) (kt + r * IDXS_LS + c4) = make_uint2(idxs_h2(v.x, v.y), idxs_h2(v.z, v.w));
    }
    const int r = lane & 15, h = r >> 2, tq = r & 3;
    const int t0 = blockIdx.x * 32 + wave * 4;
    const float * qr = q + ((int64_t) min(t0 + tq, nq - 1) * 4 + h) * qs;
    idxs_v16s ah[8], al[8];
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
        uint32_t ph[8], pl[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = *(const float4 *) (qr + kc * 16 + j * 4);
            const float x[4] = {v.x, v.y, v.z, v.w};
            uint16_t hi[4], lo[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                hi[c] = __builtin_bit_cast(uint16_t, (_Float16) x[c]);
                lo[c] = __builtin_bit_cast(uint16_t, (_Float16) (x[c] - (float) __builtin_bit_cast(_Float16, hi[c])));
            }
            ph[2 * j] = hi[0] | ((uint32_t) hi[1] << 16); ph[2 * j + 1] = hi[2] | ((uint32_t) hi[3] << 16);
            pl[2 * j] = lo[0] | ((uint32_t) lo[1] << 16); pl[2 * j + 1] = lo[2] | ((uint32_t) lo[3] << 16);
        }
        ah[kc] = __builtin_bit_cast(idxs_v16s, (uint4[2]){make_uint4(ph[0], ph[1], ph[2], ph[3]), make_uint4(ph[4], ph[5], ph[6], ph[7])});
        al[kc] = __builtin_bit_cast(idxs_v16s, (uint4[2]){make_uint4(pl[0], pl[1], pl[2], pl[3]), make_uint4(pl[4], pl[5], pl[6], pl[7])});
    }
    __syncthreads();
    const int half = lane >> 4;
    const int ta = t0 + half, tb = t0 + 2 + half;
    const int tail_a = ta < nq ? tails[ta] : 0, tail_b = tb < nq ? tails[tb] : 0;
    for (int st = 0; st < IDXS_KB / 16; ++st) {
        idxs_v8f acc;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
        const uint16_t * kr = kt + (st * 16 + r) * IDXS_LS;
#pragma unroll
        for (int kc = 0; kc < 8; ++kc) {
            const idxs_v16s bk = __builtin_bit_cast(idxs_v16s, (uint4[2]){*(const uint4 *) (kr + kc * 16), *(const uint4 *) (kr + kc * 16 + 8)});
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[kc], bk, acc);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[kc], bk, acc);
        }
        const int b = b0 + st * 16 + r;
        if (b < nb) {
            const int sb = starts[b];
            // rows 2e + half: e = 2h' (+1) is head h', query half (2 + half)
            if (ta < nq) {
                float s = fmaxf(acc[0], 0.0f);
                s = s + fmaxf(acc[2], 0.0f); s = s + fmaxf(acc[4], 0.0f); s = s + fmaxf(acc[6], 0.0f);
                out[(int64_t) ta * nb + b] = tail_a > sb ? s : -INFINITY;
            }
            if (tb < nq) {
                float s = fmaxf(acc[1], 0.0f);
                s = s + fmaxf(acc[3], 0.0f); s = s + fmaxf(acc[5], 0.0f); s = s + fmaxf(acc[7], 0.0f);
                out[(int64_t) tb * nb + b] = tail_b > sb ? s : -INFINITY;
            }
        }
    }
}

void ggml_cuda_op_idx_score(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_score_args & a) {
    const int nb = (int) a.keys->ne[1], nq = (int) (a.q->ne[1] / 4);
    const dim3 grid((unsigned) ((nq + 31) / 32), (unsigned) ((nb + IDXS_KB - 1) / IDXS_KB));
    k_idx_score<<<grid, 256, 0, ctx.stream()>>>((const float *) a.keys->data, (int64_t) (a.keys->nb[1] / sizeof(float)), nb,
        (const float *) a.q->data, (int64_t) (a.q->nb[1] / sizeof(float)), nq, a.starts, a.tails, (float *) a.out->data);
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0;
    if (hits++ < 2) fprintf(stderr, "IDX_SCORE fused: blocks=%d queries=%d\n", nb, nq);
}

// strixllama: the indexer score of a decode-sized strip (at most 8 queries), whole: out[t][b] = s + log(step(tails[t] - starts[b]))
// with s = ((relu(s0) + relu(s1)) + relu(s2)) + relu(s3), s_h = q[t][h] . kb[rows[b]] - what the graph computes as GET_ROWS (the
// F16 block keys out of the cache as F32) -> MUL_MAT -> RELU -> the head sum -> the visibility, one pass over each key instead of
// an F32 copy of every key and seven passes over the scores. Each dot product is the one the vector kernel takes for K = 128
// (mul_mat_vec_f, 64 threads a row): thread l accumulates dims 2l and 2l + 1 with ggml_cuda_mad from 0, each wave of 32 sums its
// partials by butterfly (offsets 16, 8, 4, 2, 1: lane 0 ends with ((((p0 + p16) + (p8 + p24)) + ...)), and the two waves' sums
// are reduced again by butterfly with zeros in the other lanes, (w0 + 0) + (w1 + 0). Here one thread takes a key and spells that
// tree out, so the scores are the chain's, bit for bit. `zero` is 0.0f, passed in so those additions stay in the code.
#define IDXD_MAXQ 8

static __device__ __forceinline__ float idxd_partial(const half2 k, const float * __restrict__ q) {
    const float2 kx = __half22float2(k);
    float s = 0.0f;
    ggml_cuda_mad(s, kx.x, q[0]);
    ggml_cuda_mad(s, kx.y, q[1]);
    return s;
}

// one wave's butterfly sum of its 32 partials: k and q hold that wave's 64 dims
static __device__ __forceinline__ float idxd_wave(const half2 * __restrict__ k, const float * __restrict__ q) {
    float a[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = idxd_partial(k[i], q + 2 * i) + idxd_partial(k[i + 16], q + 2 * (i + 16));
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = a[i] + a[i + 8]; }
#pragma unroll
    for (int i = 0; i < 4; ++i) { a[i] = a[i] + a[i + 4]; }
#pragma unroll
    for (int i = 0; i < 2; ++i) { a[i] = a[i] + a[i + 2]; }
    return a[0] + a[1];
}

// thread pair (2b, 2b + 1) takes key b: thread w the dims [64 w, 64 w + 64), the vector kernel's wave w
static __global__ void __launch_bounds__(256) k_idx_score_dec(const half * __restrict__ kb, const int64_t kbs,
        const int32_t * __restrict__ rows, const int nb, const float * __restrict__ q, const int64_t qs, const int nq,
        const int32_t * __restrict__ starts, const int32_t * __restrict__ tails, float * __restrict__ out, const float zero,
        const float * __restrict__ seq_blk, const float * __restrict__ seq_tok, const int n_slots) {
    __shared__ float qsh[IDXD_MAXQ * 4 * 128];
    for (int i = threadIdx.x; i < nq * 4 * 128; i += blockDim.x) {
        qsh[i] = q[(int64_t) (i / 128) * qs + i % 128];
    }
    __syncthreads();
    const int w = threadIdx.x & 1;
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 1;
    const bool valid = b < nb;
    half2 k[32];
    if (valid) {
        const uint4 * kp = (const uint4 *) (kb + (int64_t) rows[b] * kbs + 64 * w);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint4 v = kp[i];
            k[4 * i + 0] = __builtin_bit_cast(half2, v.x);
            k[4 * i + 1] = __builtin_bit_cast(half2, v.y);
            k[4 * i + 2] = __builtin_bit_cast(half2, v.z);
            k[4 * i + 3] = __builtin_bit_cast(half2, v.w);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) { k[i] = __builtin_bit_cast(half2, 0u); }
    }
    const int st = valid ? starts[b] : 0;
    for (int t = 0; t < nq; ++t) {
        float acc = 0.0f;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float mine  = idxd_wave(k, qsh + (t * 4 + h) * 128 + 64 * w);
            const float other = __shfl_xor_sync(0xffffffff, mine, 1, 32);
            const float w0 = w == 0 ? mine : other, w1 = w == 0 ? other : mine;
            const float v = (w0 + zero) + (w1 + zero);
            const float r = fmaxf(v, 0.0f);
            acc = h == 0 ? r : acc + r;
        }
        if (valid && w == 0) {
            bool vis = tails[t] > st;
            if (seq_blk) {
                // the membership product's 0/1 terms: exact in any order
                float m = 0.0f;
                for (int sl = 0; sl < n_slots; ++sl) {
                    m += seq_blk[(int64_t) b * n_slots + sl] * seq_tok[(int64_t) t * n_slots + sl];
                }
                vis = vis && m > 0.5f;
            }
            out[(int64_t) t * nb + b] = vis ? acc + zero : -INFINITY;
        }
    }
}

void ggml_cuda_op_idx_score_dec(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_score_dec_args & a) {
    const int nb = (int) a.rows->ne[0], nq = (int) (a.q->ne[1] / 4);
    GGML_ASSERT(nq >= 1 && nq <= IDXD_MAXQ && a.kb->nb[1] % 16 == 0 && ((uintptr_t) a.kb->data) % 16 == 0);
    const dim3 grid((unsigned) ((2 * (int64_t) nb + 255) / 256));
    k_idx_score_dec<<<grid, 256, 0, ctx.stream()>>>((const half *) a.kb->data, (int64_t) (a.kb->nb[1] / sizeof(half)),
        (const int32_t *) a.rows->data, nb, (const float *) a.q->data, (int64_t) (a.q->nb[1] / sizeof(float)), nq,
        a.starts, a.tails, (float *) a.out->data, 0.0f, a.seq_blk, a.seq_tok, a.n_slots);
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0;
    if (hits++ < 2) fprintf(stderr, "IDX_SCORE_DEC fused: blocks=%d queries=%d\n", nb, nq);
}
