#include "common.cuh"

// strixllama: where SSM_SCAN puts its new states when the CPY after it is fused (ggml_cuda_ssm_scan_cache_fuse): the
// cache rows of the ubatch's sequences, n_seqs rows of src0->nb[3] bytes, the first of them row0 of src0 (ids count
// rows of src0)
struct ggml_cuda_ssm_scan_cache {
    char *  rows = nullptr;
    int64_t row0 = 0;
};

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_ssm_scan_cache * cache = nullptr);
