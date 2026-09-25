#include "common.cuh"

struct ggml_cuda_idx_relu_sum_args {
    const ggml_tensor * score = nullptr;
    ggml_tensor *       dst   = nullptr;   // [n_blocks, n_tps, n_stream]
    int                 heads = 0;
};
bool ggml_cuda_idx_relu_sum_enabled();
void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args);

// strixllama: the lightning indexer's whole score for one query strip in one kernel (see ggml_cuda_match_idx_score)
struct ggml_cuda_idx_score_args {
    const ggml_tensor * keys   = nullptr;   // [128, n_blocks] F32, the block keys (F16 values out of the block-key cache)
    const ggml_tensor * q      = nullptr;   // [128, 4 * n_query] F32 view, column t * 4 + h
    const int32_t *     starts = nullptr;   // [n_blocks] first position that sees each block
    const int32_t *     tails  = nullptr;   // [n_query]  each query's position bound
    ggml_tensor *       out    = nullptr;   // [n_blocks, n_query] F32
};
void ggml_cuda_op_idx_score(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_score_args & args);
