#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // strixllama: layers this memory actually keeps state for; the layer filter can leave none (the MTP
    // draft's hybrid-idx memory), and then the cell position bookkeeping has nothing to protect
    uint32_t n_layer_recr = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // strixllama: deferred rollback for the gated delta net (ggml_gated_delta_net_lazy; STRIX_GDN_LAZY=0: off). A verify batch
    // leaves the state row of its cell as it found it and records per token what a replay needs; the next batch replays
    // the records its rollback kept. Per recurrent layer a record buffer [rec_floats, rec_cap, 2 * size] (two sets per
    // cell), and per cell how many records are pending - 0 means its rows hold the state as the plain net leaves them
    // (slot rs_idx of the cell) - which set holds them, and how many of them its last batch wrote (what a rollback may
    // take back). While its set has room a short batch appends its records after the pending ones and leaves the row
    // as it is (rec_append, STRIX_GDN_ACC=W: W records a set, default 8; also without MTP, where decode batches then
    // record); else it writes the replayed state back and starts the other set. The row is written once per ~W tokens.
    uint32_t rec_floats = 0;
    uint32_t rec_cap    = 0;  // records a set holds
    bool     rec_append = false;
    int32_t  gdn_s  = 0;      // S_v = S_k
    int32_t  gdn_hv = 0;      // value heads
    int32_t  gdn_hk = 0;      // key heads
    std::vector<ggml_tensor *> rec_l;
    std::vector<uint32_t>      rec_n;
    std::vector<uint8_t>       rec_set;
    std::vector<uint32_t>      rec_last;

    bool lazy_on() const { return rec_floats > 0; }

    // the records a cell's sequence would replay: its last batch's, less the rollback
    uint32_t rec_pending(uint32_t cell) const;

    // the state of `cell` for layer il as the plain net would hold it, computed from its row and records on the host
    // (the same fmaf as the kernels): what state_write saves, and what seq_cp copies
    void rec_materialize(uint32_t cell, int32_t il, std::vector<float> & out) const;

    // the cell's state moved to its slot-0 rows (all recurrent tensors), its records and its sequences' rollback
    // dropped: the plain net's view of it is right from then on. For cells two sequences share (seq_cp), and for
    // the cells a checkpoint writes
    void rec_flatten(uint32_t cell);

    // the replay of a cell's n_rep pending records on the device, by the kernel the next batch would run (bitwise
    // the host replay): its rows then hold the plain net's state. Needs the context's backends (init_update) and
    // a backend with the entry; false otherwise, and the host replays. STRIX_GDN_DEV_REPLAY=0: off
    bool rec_replay_device(uint32_t cell, uint32_t n_rep) const;
    bool rec_replay_device_ok() const;
    llama_context * lctx_sync = nullptr;

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    int32_t s_copy(int i) const;

    ggml_tensor * get_rec_l(int32_t il) const;
    bool lazy_on() const;
    uint32_t get_n_rs_seq() const;
    int32_t gdn_s() const;
    int32_t gdn_hv() const;
    int32_t gdn_hk() const;

    // strixllama: the per-cell input of the deferred-rollback net (lazy = true, n = n_seqs) or of the replay that brings
    // the states up to date before the plain net (lazy = false, n = n_rs: the extras build_rs copies along too): row
    // read, row written, records replayed, set read, set written - see llama_memory_recurrent::rec_l. Reads the rollback
    // index without consuming it (s_copy does that), and moves the cells on: for the lazy net they hold n_seq_tokens
    // records in the other set afterwards, else none.
    void lazy_info(int32_t * info, int n, bool lazy, uint32_t n_seq_tokens) const;

    // strixllama: whether every i < n reads its state from a row of its own cell (head + i: the newest state, or a
    // rollback snapshot of it), the cell its new state is written to. Unlike s_copy it consumes no rollback index
    bool s_copy_own_cell(int n) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
