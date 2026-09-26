#include "llama-memory-hybrid-idx.h"

#include <cstdlib>

#include "prefix.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

// A qwen4exp MTP context is built with a recurrent filter that matches nothing (the nextn layer is not recurrent) so
// it can carry an indexer cache for sparse draft attention. Skip the recurrent child there: an empty recurrent cache
// still refuses partial seq_rm, which aborts the server on the first cache trim.
static bool hybrid_idx_no_recr(const llama_memory_recurrent * r) {
    if (!r) { return true; }
    for (ggml_tensor * t : r->r_l) { if (t) { return false; } }
    for (ggml_tensor * t : r->s_l) { if (t) { return false; } }
    return true;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        // strixllama: the indexer keys stay f16 whatever the attention cache is - they are pooled into the block keys
        // the selection is scored on, and they are ~1/8 of the attention cache
        return new llama_kv_cache(
            model, hparams_idx, GGML_TYPE_F16, GGML_TYPE_F16, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // strixllama: block-key cache, one F16 [idx_dim, kv_size + 1] tensor per indexer layer, allocated next to
    // that layer's indexer keys (LLAMA_QSA_BLOCK_KEY_CACHE=0 turns it off and restores the full rebuild)
    const char * kb_env = getenv("LLAMA_QSA_BLOCK_KEY_CACHE");
    if (mem_idx && (kb_env == nullptr || atoi(kb_env) != 0)) {
        const uint32_t kv_size = mem_idx->get_size();
        const int64_t  idx_dim = model.hparams.indexer_head_size;
        const uint32_t n_layer = model.hparams.n_layer_all;

        std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!filter_idx(il)) {
                continue;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (offload) {
                buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
            }
            ggml_context * ctx = nullptr;
            auto it = ctx_map.find(buft);
            if (it == ctx_map.end()) {
                ggml_init_params params = { size_t(2u*n_layer*ggml_tensor_overhead()), nullptr, true };
                ctx = ggml_init(params);
                if (ctx == nullptr) {
                    throw std::runtime_error("failed to create ggml context for the QSA block-key cache");
                }
                ctx_map[buft] = ctx;
                kb_ctxs.emplace_back(ctx);
            } else {
                ctx = it->second;
            }
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, idx_dim, (int64_t) kv_size + 1);
            ggml_format_name(t, "cache_idx_kb_l%d", il);
            kb_map[(int32_t) il] = t;
        }
        size_t bytes = 0;
        for (auto & [buft, ctx] : ctx_map) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error("failed to allocate the QSA block-key cache");
            }
            ggml_backend_buffer_clear(buf, 0);
            bytes += ggml_backend_buffer_get_size(buf);
            kb_bufs.emplace_back(buf);
        }
        LLAMA_LOG_INFO("%s: QSA block-key cache: %zu layers, %.1f MiB\n", __func__, kb_map.size(), bytes/1024.0/1024.0);
    }

    // strixllama: regions (llama_kv_cache::set_regions) - with several sequences in one unified pool, each
    // conversation keeps one run of cells and a batch's graph views only its own. The indexer mirrors the
    // attention cells, so both caches take the window, and the moves, together. LLAMA_KV_REGIONS=0 turns it off.
    {
        const char * env = getenv("LLAMA_KV_REGIONS");
        const bool on = unified && n_seq_max > 1 && (env == nullptr || atoi(env) != 0);
        get_mem_attn()->set_regions(on);
        if (mem_idx) {
            mem_idx->set_regions(on);
            if (get_mem_attn()->get_regions() != mem_idx->get_regions()) {
                get_mem_attn()->set_regions(false);
                mem_idx->set_regions(false);
            }
        }
        if (get_mem_attn()->get_regions()) {
            // every move of the attention cells - a rebalance for a batch or for a restore - moves the indexer's
            // cells and the block keys with them, in the same step
            get_mem_attn()->set_move_hook([this](const llama_kv_cache::cell_move_vec_t & moves, bool sync) {
                if (mem_idx) {
                    GGML_ASSERT(mem_idx->can_move());
                    mem_idx->move_cells(moves, sync);
                }
                kb_move_rows(moves);
            });
            fprintf(stderr, "kv regions: on, one run of cells per conversation, %u sequences, %u cells\n", n_seq_max, kv_size);
        }
    }
}

ggml_tensor * llama_memory_hybrid_idx::get_kb(int32_t il) const {
    const auto it = kb_map.find(il);
    return it == kb_map.end() ? nullptr : it->second;
}

uint32_t llama_memory_hybrid_idx::kb_scratch_row() const {
    return mem_idx ? mem_idx->get_size() : 0;
}

bool llama_memory_hybrid_idx::kb_needs_full(const llama_ubatch & ubatch) const {
    if (kb_stale.none() || !ubatch.seq_id || !ubatch.n_seq_id) {
        return false;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            if (s >= 0 && s < LLAMA_MAX_SEQ && kb_stale.test(s)) {
                return true;
            }
        }
    }
    return false;
}

void llama_memory_hybrid_idx::kb_mark_full(const llama_ubatch & ubatch) const {
    if (!ubatch.seq_id || !ubatch.n_seq_id) {
        return;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            if (s >= 0 && s < LLAMA_MAX_SEQ) {
                kb_stale.reset(s);
            }
        }
    }
}

void llama_memory_hybrid_idx::kb_mark_stale(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
        kb_stale.set();
    } else {
        kb_stale.set(seq_id);
    }
}

bool llama_memory_hybrid_idx::kb_pos_dup() const {
    return kb_dup;
}

// A block's key sits at the row of its first cell. The moves lay a sequence out in position order, so its keys
// can follow their cells only if its cells were in that order already (then every block keeps its first cell);
// a sequence whose cells were not has its keys rebuilt once instead.
void llama_memory_hybrid_idx::kb_move_rows(const llama_kv_cache::cell_move_vec_t & moves) {
    llama_kv_cache::cell_move_vec_t rows;
    for (const auto & m : moves) {
        if (m.ordered) {
            rows.push_back(m);
        } else {
            kb_mark_stale(m.seq);
        }
    }
    for (const auto & [il, t] : kb_map) {
        get_mem_attn()->copy_rows(t, rows);
    }
}

// strixllama: does this ubatch carry a position per axis, i.e. an image under M-RoPE? A text token has
// the same value on every axis (the whole batch is then "position scalar"), an image token does not,
// and the cells it writes repeat one position across the image. Mirrors qwen4exp_pos_scalar().
static bool hybrid_idx_ubatch_pos_dup(const llama_ubatch & ubatch) {
    if (!ubatch.pos || ubatch.n_tokens == 0) {
        return false;
    }
    for (uint32_t axis = 1; axis < ubatch.n_pos; ++axis) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i + axis*ubatch.n_tokens] != ubatch.pos[i]) {
                return true;
            }
        }
    }
    return false;
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // strixllama: once a conversation holds an image (or a position gap, kb_dup), its ubatches take the dense
    // sparse-attention inputs - a KQ mask and a per-block bias over n_kv x n_tokens, ~6.6 bytes a pair staged in
    // pinned host memory and again on the GPU. They grow with every image, and each time the pinned buffer grew
    // ROCm on Windows kept the old one: 27 images of 1920x1080 on a 58K conversation took 12 GiB of RAM, and a 14K
    // text append at batch 8192 then faulted (issue #2). Such ubatches are held to n_kv x n_tokens <= 2^27: about
    // 1024 tokens at 128K cells, 512 at 256K. The dense path is bound by attention over the whole cache, so this
    // costs it little: the same 27 images prefilled at the same rate, the append at 245 t/s instead of 261 at 2^28,
    // and the machine kept 11 GiB of RAM instead of 3. Text before any image keeps the full ubatch.
    // STRIX_DENSE_UBATCH_BUDGET sets the bound in cell-token pairs (0 = no bound).
    if (kb_dup) {
        static const uint64_t budget = [] {
            const char * e = getenv("STRIX_DENSE_UBATCH_BUDGET");
            return e ? (uint64_t) strtoull(e, nullptr, 10) : (uint64_t) 1 << 27;
        }();
        const uint64_t n_kv = (uint64_t) get_mem_attn()->get_cells(0).used_max_p1() + balloc.get_n_tokens();
        if (budget > 0 && n_kv > 0) {
            const uint32_t cap = (uint32_t) std::max<uint64_t>(256, budget / n_kv / 256 * 256);
            if (cap < n_ubatch) {
                static uint32_t logged = 0;
                if (cap != logged) {
                    logged = cap;
                    LLAMA_LOG_INFO("%s: image in the conversation - ubatch %u -> %u tokens at %llu cells (dense inputs bounded)\n",
                                   __func__, n_ubatch, cap, (unsigned long long) n_kv);
                }
                n_ubatch = cap;
            }
        }
    }
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // strixllama: decide here whether the block-key cache may be wired into the graph built from this
        // context (see kb_pos_dup), because set_input_qsa will then rank cells instead of using their
        // position, and the cache cannot track ranked cells. Two ways to get there, both sticky:
        //   - an image ubatch: M-RoPE gives it a position per axis and repeats one position across the
        //     image, so cells share positions (`dup`);
        //   - a gap: this ubatch does not continue the sequence, so cells and positions drift apart and
        //     a cell can land outside the block window (`oor`). The MTP draft is where this bites -
        //     common_speculative_impl_draft_mtp::process() skips embedding batches, so the draft never
        //     stores the image's tokens and every later position is shifted past its cell.
        {
            // strixllama: per token and per sequence. A ubatch that serves several slots at once
            // (unified cache, equal-length split) carries tokens of several sequences, so its first
            // and last tokens belong to different sequences; comparing them as one made the flag
            // trip on any multi-stream decode, and it is sticky, so the block-key cache then stayed
            // off for every sequence, single streams included (measured: ROPE and GET_ROWS over
            // every block of the pool back in each step, 8% slower single-stream decode).
            std::map<llama_seq_id, llama_pos> next_pos;
            for (const auto & ub : ubatches) {
                if (hybrid_idx_ubatch_pos_dup(ub)) {
                    kb_dup = true;
                    break;
                }
                if (ub.n_tokens == 0 || !ub.pos || !ub.seq_id) {
                    continue;
                }
                bool gap = false;
                for (uint32_t i = 0; i < ub.n_tokens && !gap; ++i) {
                    if (!ub.seq_id[i]) {
                        continue;
                    }
                    const llama_seq_id s  = ub.seq_id[i][0];
                    const auto         it = next_pos.find(s);
                    const llama_pos    expect = it != next_pos.end()
                        ? it->second
                        : get_mem_attn()->seq_pos_max(s) + 1;
                    gap = ub.pos[i] > expect;
                    next_pos[s] = ub.pos[i] + 1;
                }
                if (gap) {
                    kb_dup = true;
                    break;
                }
            }
        }

        // strixllama: regions - when the batch does not fit after its conversations' last cells, the pool is
        // rebalanced first; the hook set in the constructor moves the indexer and the block keys along
        if (get_mem_attn()->can_move() && (!mem_idx || mem_idx->can_move())) {
            get_mem_attn()->move_cells(get_mem_attn()->plan_layout(ubatches));
        }

        // prepare the recurrent batches first
        if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    kb_mark_stale(-1);   // strixllama: block keys depend on positions and cell contents; rebuild them once
    kb_dup = false;   // strixllama: no cells left, so no image cells either
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    // strixllama: only a request that drops every sequence is proof the image cells are gone. A whole-
    // sequence seq_rm is NOT: the server calls it when it reuses a slot by longest-common-prefix and
    // then keeps the cached prefix, image cells included, so clearing the flag there wires the cache
    // back in under ranked cells and aborts. Measured that failure directly (tmp/vis_stress.py).
    if (seq_id < 0 && p0 <= 0 && p1 < 0) {
        kb_dup = false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // strixllama: the copy shares the source's cells, and with them block keys that may still have to be rebuilt
    if (seq_id_src >= 0 && seq_id_src < LLAMA_MAX_SEQ && kb_stale.test(seq_id_src)) {
        kb_mark_stale(seq_id_dst);
    }

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kb_mark_stale(-1);   // strixllama: block keys depend on positions; a shifted cell can be shared, so all of them
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kb_mark_stale(-1);   // strixllama: block keys depend on positions; a shifted cell can be shared, so all of them
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        get_mem_attn()->state_write(io, seq_id, flags);
    }
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_write(io, seq_id, flags); }

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kb_mark_stale(seq_id);   // strixllama: the restored cells' block keys were never written; rebuild them once
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            // strixllama: the attention restore drops the sequence's old cells and may rebalance the pool before
            // it places the new ones (regions); the move hook repeats that rebalance on the indexer, which must
            // then hold the same cells - so the indexer drops them first too (its own restore would, later)
            if (mem_idx && seq_id >= 0 && get_mem_attn()->get_regions()) {
                mem_idx->seq_rm(seq_id, -1, -1);
            }
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_read(io, seq_id, flags); }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->seq_rm(seq_id, -1, -1); }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

size_t llama_memory_hybrid_idx::kv_row_size() const {
    const size_t attn = get_mem_attn()->row_size();
    const size_t idx  = mem_idx ? mem_idx->row_size() : 0;
    return attn == 0 || (mem_idx && idx == 0) ? 0 : attn + idx;
}

bool llama_memory_hybrid_idx::kv_rows_get(llama_seq_id seq_id, llama_pos p0, uint32_t n, uint8_t * dst) const {
    if (kv_row_size() == 0) {
        return false;
    }
    return get_mem_attn()->seq_rows_get(seq_id, p0, n, dst) &&
           (!mem_idx || mem_idx->seq_rows_get(seq_id, p0, n, dst + (size_t) n*get_mem_attn()->row_size()));
}

bool llama_memory_hybrid_idx::kv_rows_set(llama_seq_id seq_id, llama_pos p0, uint32_t n, const uint8_t * src, uint32_t src_rows) {
    if (kv_row_size() == 0) {
        return false;
    }
    return get_mem_attn()->seq_rows_set(seq_id, p0, n, src, src_rows) &&
           (!mem_idx || mem_idx->seq_rows_set(seq_id, p0, n, src + (size_t) src_rows*get_mem_attn()->row_size(), src_rows));
}

bool llama_memory_hybrid_idx::kv_alloc(llama_seq_id seq_id, const llama_token * tokens, uint32_t n) {
    if (kv_row_size() == 0) {
        return false;
    }

    // the recurrent state goes too: the caller restores the one for position n next
    seq_rm(seq_id, -1, -1);
    kb_mark_stale(seq_id);                          // no block key of these cells was ever computed here

    // the indexer mirrors the attention cache cell for cell, as a restore does (state_read)
    // both with the attention cache's position sections, as a decode gives both the one batch
    const uint32_t n_pos = get_mem_attn()->n_pos_per_embd();
    llama_kv_cache::slot_info sinfo;
    if (!get_mem_attn()->seq_alloc(seq_id, tokens, n, n_pos, nullptr, &sinfo)) {
        return false;
    }
    if (mem_idx && !mem_idx->seq_alloc(seq_id, tokens, n, n_pos, n > 0 ? &sinfo : nullptr, nullptr)) {
        get_mem_attn()->seq_rm(seq_id, -1, -1);
        return false;
    }
    return true;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias, const qsa_kb_inputs * kb,
        uint32_t kv_off) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, nullptr, ubatch, ratio, blk_bias, kb, nullptr, false, kv_off);
}

void llama_memory_hybrid_idx::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio, const qsa_kb_inputs * kb, const qsa_mixed_inputs * mixed,
        bool active_only, uint32_t kv_off) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, true, kb, mixed, active_only, kv_off);
}

// strixllama: the indexer cells as the graph's view sees them, from its first cell (llama_kv_cache::get_kv_window)
struct hybrid_idx_cells_view {
    const llama_kv_cells & cells;
    const uint32_t         off;

    bool                              is_empty   (uint32_t j)                 const { return cells.is_empty(off + j); }
    const llama_kv_cells::seq_set_t & seq_get_all(uint32_t j)                 const { return cells.seq_get_all(off + j); }
    bool                              seq_has    (uint32_t j, llama_seq_id s) const { return cells.seq_has(off + j, s); }
    llama_pos                         pos_get    (uint32_t j)                 const { return cells.pos_get(off + j); }
    const llama_kv_cell_ext &         ext_get    (uint32_t j)                 const { return cells.ext_get(off + j); }
    llama_pos                         seq_pos_min(llama_seq_id s)             const { return cells.seq_pos_min(s); }
};

void llama_memory_hybrid_idx::set_input_qsa_impl(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * tail_idxs,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        const qsa_kb_inputs * kb,
        const qsa_mixed_inputs * mixed,
        bool active_only,
        uint32_t kv_off) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(tail_idxs || ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    // strixllama: n_blocks is the length of the block list the graph was built for; n_pb is the window of
    // position buckets the cells are grouped by. They are one number unless the list covers only the
    // ubatch's own sequences (active_only, llama_memory_hybrid_idx_context::qsa_active_blocks): then the
    // list is shorter than the window, and every cell of another sequence is left out of the grouping -
    // its blocks were enumerated before only to be marked invisible to every query of the ubatch.
    GGML_ASSERT(!active_only || (tail_idxs && bias->type == GGML_TYPE_I32 && n_ns == 1));
    const int64_t n_pb = active_only ? std::max(n_blocks, (n_kv + r - 1)/r) : n_blocks;
    std::bitset<LLAMA_MAX_SEQ> active_seqs;
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) { active_seqs.set(ubatch->seq_id[i][k]); }
    }
    // strixllama: a cell of no sequence of this ubatch is invisible to every query in it, and its block key is
    // rebuilt by its own sequence's graph when it goes stale (kb_needs_full), so it is left out of the blocks.
    // That also keeps a window's foreign cells - the neighbours of the run a window views, whose positions have
    // nothing to do with it - out of the position buckets.

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = tail_idxs ? nullptr : (int32_t *) cell_blk->data;
    // strixllama: with the block-key cache an incremental graph builds its keys from the dirty list, so no
    // node reads blk_cells / blk_pos and ggml leaves them unallocated. They are still needed here (the
    // dirty-block scan reads them), so fall back to local scratch instead of writing through null.
    std::vector<int32_t> blk_cells_local, blk_pos_local;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    if (dst_blk_cells == nullptr) {
        blk_cells_local.assign((size_t) blk_cells->ne[0]*blk_cells->ne[1], 0);
        dst_blk_cells = blk_cells_local.data();
    }
    int32_t * dst_blk_pos = (int32_t *) blk_pos->data;
    if (dst_blk_pos == nullptr) {
        blk_pos_local.assign((size_t) blk_pos->ne[0], 0);
        dst_blk_pos = blk_pos_local.data();
    }
    const bool compact = bias->type == GGML_TYPE_I32;
    float * dst_bias = compact ? nullptr : (float *) bias->data;
    int32_t * limits = compact ? (int32_t *) bias->data : nullptr;
    // strixllama: several sequences in one compact ubatch - the membership inputs carry the sequence half
    const bool mixed_seqs = compact && mixed != nullptr && mixed->seq_blk != nullptr && mixed->seq_tok != nullptr;
    if (compact) {
        GGML_ASSERT(tail_idxs && blk_bias && n_ns == 1 && ggml_nelements(bias) == n_blocks+n_tps);
        if (mixed_seqs) {
            GGML_ASSERT(ubatch->seq_idx && ubatch->seq_id_unq && ubatch->n_seqs_unq >= 1);
            GGML_ASSERT(mixed->seq_blk->type == GGML_TYPE_F32 && mixed->seq_tok->type == GGML_TYPE_F32);
            GGML_ASSERT(mixed->seq_blk->ne[0] == ubatch->n_seqs_unq && mixed->seq_blk->ne[1] == n_blocks);
            GGML_ASSERT(mixed->seq_tok->ne[0] == ubatch->n_seqs_unq && mixed->seq_tok->ne[1] == n_tps);
            GGML_ASSERT(mixed->seq_blk->data && mixed->seq_tok->data);
        } else {
            for (int64_t i=0;i<n_tokens;++i) { GGML_ASSERT(ubatch->seq_id[i][0] == ubatch->seq_id[0][0]); }
        }
    }
    int32_t * dst_tail = tail_idxs ? (int32_t *) tail_idxs->data : nullptr;
    if (tail_idxs) {
        GGML_ASSERT(blk_bias && r > 1 && ggml_backend_buffer_is_host(tail_idxs->buffer));
        GGML_ASSERT(tail_idxs->ne[0] == r-1 && tail_idxs->ne[1] == n_tps && tail_idxs->ne[2] == n_ns);
        std::fill(dst_tail, dst_tail + ggml_nelements(tail_idxs), -1);
    }

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_pb);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        // strixllama: every index below is relative to the view's first cell
        GGML_ASSERT(kv_off == 0 || n_ns == 1);
        const hybrid_idx_cells_view cells = { get_mem_idx()->get_cells(seq_of_stream), kv_off };

        int32_t * cur_cell_blk  = dst_cell_blk ? dst_cell_blk + s*n_kv : nullptr;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }
                if ((cells.seq_get_all(j) & active_seqs).none()) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_pb) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position.
        // strixllama: `oor` needs the same treatment. M-RoPE also advances the position past an image by
        // its grid extent rather than by its token count, so positions run ahead of the cells holding
        // them and a cell can land outside the block window - the rank is dense by construction, so
        // ranking fixes that too. Without it the assert below aborts the server on the first long
        // enough conversation that contains an image.
        if ((dup || oor) && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_pb; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            if (cur_cell_blk) { cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j]; }
        }

        std::vector<int32_t> group_members;
        if (dst_tail) {
            group_members.assign(grp_first.size()*r, -1);
            for (int64_t j=0;j<n_kv;++j) {
                const int32_t g=cell_grp[j];
                if (g<0) { continue; }
                const int64_t idx=ranked ? rank[j] : cells.pos_get(j);
                const int64_t slot=g*r+idx%r;
                GGML_ASSERT(group_members[slot]<0 || group_members[slot]==j);
                group_members[slot]=(int32_t)j;
            }
        }

        if (compact) {
            const llama_seq_id seq = ubatch->seq_id[0][0];
            for (int64_t b=0;b<n_blocks;++b) {
                // with several sequences the position half stays per block and the ownership goes to seq_blk
                limits[b] = b<n_bid && (mixed_seqs || cells.seq_has((uint32_t)bid_cell[b],seq)) ? bid_idx[b] : INT32_MAX;
            }
            if (mixed_seqs) {
                const int64_t n_slots = mixed->seq_blk->ne[0];
                float * sb = (float *) mixed->seq_blk->data;
                float * st = (float *) mixed->seq_tok->data;
                std::fill(sb, sb + n_slots*n_blocks, 0.0f);
                std::fill(st, st + n_slots*n_tps,    0.0f);
                for (int32_t b = 0; b < n_bid; ++b) {
                    const auto & owners = cells.seq_get_all((uint32_t) bid_cell[b]);
                    for (int64_t sl = 0; sl < n_slots; ++sl) {
                        if (owners.test(ubatch->seq_id_unq[sl])) { sb[b*n_slots + sl] = 1.0f; }
                    }
                }
                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const int32_t sl = ubatch->seq_idx[ubatch->seq_id[ii][0]];
                    GGML_ASSERT(sl >= 0 && sl < n_slots);
                    st[ii*n_slots + sl] = 1.0f;
                }
            }
        }

        // strixllama: block-key cache inputs. A cached block key goes stale only when one of the block's cells
        // is written, and a write happens exactly when a ubatch token lands in the cell, so the blocks this
        // ubatch completes are the full groups holding a cell whose (seq, pos) is a ubatch token.
        if (kb) {
            GGML_ASSERT(n_ns == 1 && !ranked && "qsa block-key cache: single stream, 1-D positions");
            GGML_ASSERT(kb->bid_rows && kb->bid_rows->ne[0] == n_blocks && kb->bid_rows->data);
            const int32_t scratch = (int32_t) kb_scratch_row();
            int32_t * br = (int32_t *) kb->bid_rows->data;
            for (int64_t b = 0; b < n_blocks; ++b) {
                br[b] = b < n_bid ? (int32_t) kv_off + bid_cell[b] : scratch;
            }
        }
        // the dirty list exists only in incremental graphs (a full-rebuild graph reads no such input)
        if (kb && kb->dirty_dst && kb->dirty_dst->data) {
            const int64_t dirty_max = kb->dirty_dst->ne[0];
            GGML_ASSERT(kb->dirty_cells->ne[0] == r*dirty_max && kb->dirty_pos->ne[0] == 4*dirty_max);
            GGML_ASSERT(kb->dirty_cells->data && kb->dirty_pos->data);
            const int32_t scratch = (int32_t) kb_scratch_row();

            llama_pos pmin = ubatch->pos[0], pmax = ubatch->pos[0];
            for (int64_t i = 1; i < n_tokens; ++i) {
                pmin = std::min(pmin, ubatch->pos[i]);
                pmax = std::max(pmax, ubatch->pos[i]);
            }
            std::vector<uint8_t> grp_dirty(grp_first.size(), 0);
            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t g = cell_grp[j];
                if (g < 0 || grp_bid[g] < 0) { continue; }
                const llama_pos p = cells.pos_get(j);
                if (p < pmin || p > pmax) { continue; }
                for (int64_t i = 0; i < n_tokens; ++i) {
                    if (ubatch->pos[i] == p && cells.seq_has(j, ubatch->seq_id[i][0])) { grp_dirty[g] = 1; break; }
                }
            }

            int32_t * dc = (int32_t *) kb->dirty_cells->data;
            int32_t * dp = (int32_t *) kb->dirty_pos->data;
            int32_t * dd = (int32_t *) kb->dirty_dst->data;
            int64_t n_dirty = 0;
            int64_t n_dirty_total = 0;
            for (int32_t b = 0; b < n_bid; ++b) {
                bool dirty = false;
                for (int64_t slot = 0; slot < r; ++slot) {
                    const int32_t g = cell_grp[cur_blk_cells[b*r + slot]];
                    if (g >= 0 && grp_dirty[g]) { dirty = true; break; }
                }
                if (!dirty) { continue; }
                ++n_dirty_total;
                if (n_dirty >= dirty_max) { continue; }
                for (int64_t slot = 0; slot < r; ++slot) { dc[n_dirty*r + slot] = cur_blk_cells[b*r + slot]; }
                for (int64_t sec = 0; sec < 4; ++sec) { dp[sec*dirty_max + n_dirty] = dst_blk_pos[sec*n_blocks + b]; }
                dd[n_dirty] = (int32_t) kv_off + bid_cell[b];
                ++n_dirty;
            }
            GGML_ASSERT(n_dirty_total <= dirty_max && "qsa block-key cache: more blocks completed than the graph can refresh");
            for (int64_t d = n_dirty; d < dirty_max; ++d) {
                for (int64_t slot = 0; slot < r; ++slot) { dc[d*r + slot] = (int32_t) std::min<int64_t>(slot, n_kv - 1); }
                for (int64_t sec = 0; sec < 4; ++sec) { dp[sec*dirty_max + d] = 0; }
                dd[d] = scratch;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;
            if (dst_tail && q+1>tail_start) {
                const int64_t pb=tail_start/r;
                if (pb>=0 && pb<n_pb) {
                    int32_t * tail=dst_tail+(s*n_tps+ii)*(r-1);
                    for (int32_t g=grp_head[pb];g>=0;g=grp_next[g]) {
                        if (!cells.seq_has((uint32_t)grp_first[g],seq_id)) { continue; }
                        for (int64_t slot=0;slot<q+1-tail_start;++slot) {
                            const int32_t cell=group_members[g*r+slot];
                            if (cell<0 || !cells.seq_has((uint32_t)cell,seq_id)) { continue; }
                            GGML_ASSERT(tail[slot]<0 || tail[slot]==cell);
                            tail[slot]=cell;
                        }
                    }
                }
            }

            if (compact) {
                GGML_ASSERT(tail_start >= 0 && tail_start <= 16777216);
                limits[n_blocks+ii] = (int32_t)tail_start;
                continue;
            }

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? (dst_tail ? -INFINITY : 1e9f) : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = dst_tail ? -INFINITY : 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        const llama_memory_hybrid_idx::qsa_kb_inputs * kb) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, kb, get_idx()->get_kv_off());
}

ggml_tensor * llama_memory_hybrid_idx_context::get_kb(int32_t il) const {
    return mem ? mem->get_kb(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::kb_scratch_row() const {
    return mem ? mem->kb_scratch_row() : 0;
}

bool llama_memory_hybrid_idx_context::kb_needs_full(const llama_ubatch & ubatch) const {
    return mem ? mem->kb_needs_full(ubatch) : false;
}

bool llama_memory_hybrid_idx_context::kb_pos_dup() const {
    return mem ? mem->kb_pos_dup() : false;
}

void llama_memory_hybrid_idx_context::kb_mark_full(const llama_ubatch & ubatch) const {
    if (mem) { mem->kb_mark_full(ubatch); }
}

void llama_memory_hybrid_idx_context::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio,
        const llama_memory_hybrid_idx::qsa_kb_inputs * kb,
        const llama_memory_hybrid_idx::qsa_mixed_inputs * mixed,
        bool active_only) const {
    GGML_ASSERT(mem != nullptr);
    mem->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, kb, mixed, active_only,
            get_idx()->get_kv_off());
}

bool llama_memory_hybrid_idx_context::qsa_position_prefix(const llama_ubatch & ubatch, bool active_only) const {
    if (!qsa_scalar_visibility(ubatch)) { return false; }
    const llama_seq_id seq=ubatch.seq_id[0][0];
    // a list of the ubatch's own sequences is one sequence's position prefix only if the ubatch has one sequence
    if (active_only) {
        for (uint32_t i=0;i<ubatch.n_tokens;++i) {
            if (ubatch.n_seq_id[i]!=1 || ubatch.seq_id[i][0]!=seq) { return false; }
        }
    }
    // strixllama: set_input_qsa_impl leaves other sequences' cells out of the blocks, so they cannot break the
    // prefix; the window starts at get_kv_off()
    return qsa_single_sequence_prefix(mem->get_mem_idx()->get_cells(seq),get_idx()->get_n_kv(),seq,true,
            get_idx()->get_kv_off());
}

// strixllama: in a unified cache every slot's conversation sits in one pool of cells and the graph spans the
// pool (n_kv), so the block list of a compact ubatch used to hold every conversation's blocks, the idle ones
// marked invisible: a decoded token paid for the block keys, the scores and the enumeration of conversations
// that were not running. The list can hold the ubatch's own sequences alone: a block's cells share one
// sequence set, every block of another set was invisible to every query here, and the visible ones keep
// their order, so the same blocks are selected in the same order. A sequence with at least the selection's
// budget of visible blocks gets bitwise the same result; a shorter one fills the budget with invisible
// entries, which now sit after its own blocks rather than among them, and that can move the last bits of its
// attention (towards what a single-slot server computes, which has no other conversation to fill it with).
// The length is an upper bound: at most one cell per position per sequence while no image has been written
// (kb_pos_dup), so a sequence holds at most pos_max - pos_min + 1 cells; rounded up to a multiple of 64
// blocks (256 cells at ratio 4, the step of n_kv) so a graph is rebuilt as rarely as before, and never below
// the budget. The reserve graph (positions all equal) keeps the whole pool: it sizes the buffers for the widest
// graph. LLAMA_QSA_ACTIVE_BLOCKS=0 turns this off.
int64_t llama_memory_hybrid_idx_context::qsa_active_blocks(const llama_ubatch & ubatch, uint32_t ratio, int64_t min_blocks) const {
    static const bool on = !getenv("LLAMA_QSA_ACTIVE_BLOCKS") || atoi(getenv("LLAMA_QSA_ACTIVE_BLOCKS")) != 0;
    if (!on || mem == nullptr || get_idx() == nullptr || ratio == 0 || get_n_stream() != 1 ||
            mem->kb_pos_dup() || !qsa_scalar_visibility(ubatch)) {
        return 0;
    }
    bool same_pos = ubatch.n_tokens > 1;
    for (uint32_t i = 1; same_pos && i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] != ubatch.pos[0]) { same_pos = false; }
    }
    if (same_pos) {
        return 0;
    }
    std::bitset<LLAMA_MAX_SEQ> seqs;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) { seqs.set(ubatch.seq_id[i][k]); }
    }
    const auto & cells = mem->get_mem_idx()->get_cells(ubatch.seq_id[0][0]);
    int64_t n_cells = 0;
    for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (!seqs.test(s)) {
            continue;
        }
        const llama_pos p0 = cells.seq_pos_min(s);
        const llama_pos p1 = cells.seq_pos_max(s);
        if (p0 < 0 || p1 < p0) {
            return 0;
        }
        n_cells += (int64_t) p1 - p0 + 1;
    }
    const int64_t full   = ((int64_t) get_idx()->get_n_kv() + ratio - 1)/ratio;
    const int64_t blocks = std::max(min_blocks, ((n_cells + ratio - 1)/ratio + 63)/64*64);
    return blocks < full ? blocks : 0;
}

bool llama_memory_hybrid_idx_context::qsa_scalar_visibility(const llama_ubatch & ubatch) const {
    // see qwen4exp_qsa_embd_ok: a batch with tokens AND embeddings is the MTP draft head, not a vision batch
    const char * embd_env = getenv("LLAMA_QSA_TOKEN_EMBD");
    const bool embd_ok = !ubatch.embd || !embd_env || atoi(embd_env) != 0;
    if (get_n_stream()!=1 || !get_idx() || !ubatch.token || !embd_ok || !ubatch.pos || !ubatch.n_tokens ||
            !ubatch.n_pos || !ubatch.seq_id || !ubatch.n_seq_id) { return false; }
    if (ubatch.n_seq_id[0]<1 || !ubatch.seq_id[0]) { return false; }
    // strixllama: several sequences in one ubatch are fine (set_input_qsa_impl's membership inputs carry
    // which sequence owns a block) as long as every token names exactly one and no image or gap has
    // been written: those cells need the ranked enumeration, which only a single-sequence ubatch gets.
    const llama_seq_id seq=ubatch.seq_id[0][0];
    std::bitset<LLAMA_MAX_SEQ> seqs;
    for (uint32_t i=0;i<ubatch.n_tokens;++i) {
        if (ubatch.n_seq_id[i]<1 || !ubatch.seq_id[i] || ubatch.seq_id[i][0]<0 || ubatch.seq_id[i][0]>=LLAMA_MAX_SEQ ||
                ubatch.pos[i]<0 || ubatch.pos[i]>=16777216) { return false; }
        if (ubatch.seq_id[i][0]!=seq && ubatch.n_seq_id[i]!=1) { return false; }
        seqs.set(ubatch.seq_id[i][0]);
        for (uint32_t axis=1;axis<ubatch.n_pos;++axis) {
            if (ubatch.pos[i+axis*ubatch.n_tokens]!=ubatch.pos[i]) { return false; }
        }
    }
    if (seqs.count()>1) {
        if (mem->kb_pos_dup() || !ubatch.seq_idx || !ubatch.seq_id_unq || ubatch.n_seqs_unq!=seqs.count()) { return false; }
        for (uint32_t i=0;i<ubatch.n_tokens;++i) {
            if (ubatch.n_seq_id[i]!=1 || ubatch.seq_idx[ubatch.seq_id[i][0]]<0) { return false; }
        }
    }
    if (ubatch.is_pos_2d()) {
        const auto & cells=mem->get_mem_idx()->get_cells(seq);
        const uint32_t off=get_idx()->get_kv_off();   // strixllama: the window's first cell
        for (uint32_t j=off;j<off+get_idx()->get_n_kv();++j) {
            if (!cells.is_empty(j) && (cells.seq_get_all(j) & seqs).any() && cells.ext_get(j).is_2d_gt(cells.pos_get(j),cells.pos_get(j))) { return false; }
        }
    }
    return true;
}
