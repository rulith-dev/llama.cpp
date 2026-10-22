#include "common.cuh"

// strixllama: where SSM_SCAN puts its new states when the CPY after it is fused (ggml_cuda_ssm_scan_cache_fuse): the
// cache rows of the ubatch's sequences, n_seqs rows of src0->nb[3] bytes, the first of them row0 of src0 (ids count
// rows of src0)
struct ggml_cuda_ssm_scan_cache {
    char *  rows = nullptr;
    int64_t row0 = 0;
};

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always the state row size, set in-kernel)
struct ggml_cuda_ssm_scan_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots
};

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_ssm_scan_cache * cache = nullptr);

// same op, but writes the state snapshot(s) into the cache instead of dst (see ggml_cuda_try_ssm_scan_cache_fusion)
void ggml_cuda_op_ssm_scan_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                       ggml_cuda_ssm_scan_fused_cache cache);
