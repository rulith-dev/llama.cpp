#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <bitset>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <set>
#include <vector>

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // when tok = LLAMA_TOKEN_NULL when the cell is produced by embedding input (i.e. multimodal)
    // use case: n-gram embeddings hash
    llama_token tok = LLAMA_TOKEN_NULL;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        *this = llama_kv_cell_ext{};
    }
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells {
public:
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    void reset() {
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
            seq_cells[s].clear();
            run_dirty[s] = 0;
            run_ok[s]    = 1;
        }
    }

    // strixllama: whether the cells of seq_id are one run - consecutive cells holding consecutive positions, one cell a
    // position - and if so its first and last cell and its first position. A conversation in a unified cache is one while
    // it holds text only; an image repeats a position and breaks it. Checked in full the first time it is asked after a
    // change other than an append at the next cell with the next position or the removal of the last cell
    bool seq_run(llama_seq_id seq_id, uint32_t & c0, uint32_t & c1, llama_pos & p0) const {
        assert(seq_id >= 0 && seq_id < LLAMA_MAX_SEQ);
        const auto & sc = seq_cells[seq_id];
        if (sc.empty()) {
            return false;
        }
        if (run_dirty[seq_id]) {
            bool ok = true;
            bool first = true;
            uint32_t prev = 0;
            for (const uint32_t c : sc) {
                if (!first && (c != prev + 1 || pos[c] != pos[prev] + 1)) {
                    ok = false;
                    break;
                }
                prev  = c;
                first = false;
            }
            run_ok[seq_id]    = ok;
            run_dirty[seq_id] = 0;
        }
        if (!run_ok[seq_id]) {
            return false;
        }
        c0 = *sc.begin();
        c1 = *sc.rbegin();
        p0 = pos[c0];
        return true;
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq.resize(n);

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used.size();
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        return used.empty() ? 0 : *used.begin();
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        return used.empty() ? 0 : *used.rbegin() + 1;
    }

    // strixllama: the used cells in index order, for walking the runs of free cells between them
    const std::set<uint32_t> & used_set() const {
        return used;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // move cell isrc to idst (used during defrag)
    //void mv(uint32_t isrc, uint32_t idst) {
    //    assert(isrc < pos.size());
    //    assert(idst < pos.size());

    //    assert(pos[idst] == -1);
    //    assert(pos[isrc] != -1);

    //    pos  [idst] = pos  [isrc];
    //    shift[idst] = shift[isrc];
    //    seq  [idst] = seq  [isrc];

    //    pos  [isrc] = -1;
    //    shift[isrc] =  0;
    //    seq  [isrc].reset();

    //    used.erase (isrc);
    //    used.insert(idst);
    //}

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = i + j;

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(i + j);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(i + j);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(i + j);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = idxs[j];

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(idx);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(idx);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(idx);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);
        seq[i].reset();

        pos[i] = -1;
        ext[i].reset();
        shift[i] = 0;

        used.erase(i);
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, i);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, i);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // the full set of sequences this cell is visible to
    const seq_set_t & seq_get_all(uint32_t i) const {
        assert(i < pos.size());

        return seq[i];
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // the token of the cell of sequence seq_id at the largest position <= p
    // when several cells share that position, the one with the highest index wins
    // return LLAMA_TOKEN_NULL if the sequence has no cell at or before p
    // note: used by n-gram input embeddings to recover the tokens preceding a ubatch
    llama_token seq_pos_tok_le(llama_seq_id seq_id, llama_pos p) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const auto & sp = seq_pos[seq_id];

        auto it = sp.upper_bound({ p, std::numeric_limits<uint32_t>::max() });
        if (it == sp.begin()) {
            return LLAMA_TOKEN_NULL;
        }

        return ext[(--it)->second].tok;
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, i);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                return s;
            }
        }

        return -1;
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        return seq_pos[seq_id].begin()->first;
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        return seq_pos[seq_id].rbegin()->first;
    }

    // strixllama: the span of the cell array that holds sequence seq_id, [seq_cell_min, seq_cell_max]
    // return -1 if the sequence is not present
    int64_t seq_cell_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return seq_cells[seq_id].empty() ? -1 : (int64_t) *seq_cells[seq_id].begin();
    }

    int64_t seq_cell_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return seq_cells[seq_id].empty() ? -1 : (int64_t) *seq_cells[seq_id].rbegin();
    }

    // the number of cells that carry sequence seq_id
    uint32_t seq_cell_count(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return (uint32_t) seq_cells[seq_id].size();
    }

    // the (pos, cell) pairs of sequence seq_id, ordered by position
    const std::set<std::pair<llama_pos, uint32_t>> & seq_pos_cells(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return seq_pos[seq_id];
    }

    // strixllama: move the state of cell isrc to the empty cell idst (the KV data is moved by the caller)
    void mv(uint32_t isrc, uint32_t idst) {
        assert(isrc < pos.size());
        assert(idst < pos.size());
        assert(isrc != idst);
        assert(pos[idst] == -1 && seq[idst].none());
        assert(pos[isrc] != -1);
        assert(shift[isrc] == 0);

        seq_pos_rm(isrc);

        pos[idst] = pos[isrc];
        ext[idst] = ext[isrc];
        seq[idst] = seq[isrc];

        pos[isrc] = -1;
        ext[isrc].reset();
        seq[isrc].reset();

        used.erase(isrc);
        used.insert(idst);

        seq_pos_add(idst);
    }

    // strixllama: mv() for the n cells [isrc, isrc + n), all of sequence seq_id alone, to the empty cells
    // [idst, idst + n); the two ranges must not overlap. The index sets' nodes are moved rather than freed and
    // allocated again, and mv()'s scans over every sequence id are skipped: a rebalance moves tens of thousands
    // of cells, which took ~100 ms through mv().
    void mv_range(uint32_t isrc, uint32_t idst, uint32_t n, llama_seq_id seq_id) {
        assert(isrc + n <= idst || idst + n <= isrc);
        assert(isrc + n <= pos.size() && idst + n <= pos.size());
        assert(seq_id >= 0 && seq_id < LLAMA_MAX_SEQ);

        for (uint32_t j = 0; j < n; ++j) {
            if (pos[isrc + j] == -1 || seq[isrc + j].count() != 1 || !seq[isrc + j].test(seq_id) || shift[isrc + j] != 0) {
                for (uint32_t k = 0; k < n; ++k) {
                    mv(isrc + k, idst + k);
                }
                return;
            }
        }

        move_index(used,              isrc, idst, n);
        move_index(seq_cells[seq_id], isrc, idst, n);
        run_dirty[seq_id] = 1;

        auto & sp = seq_pos[seq_id];
        for (uint32_t j = 0; j < n; ++j) {
            auto it = sp.find({ pos[isrc + j], isrc + j });
            assert(it != sp.end());
            auto next = std::next(it);
            auto nh = sp.extract(it);
            nh.value().second = idst + j;
            sp.insert(next, std::move(nh));
        }

        for (uint32_t j = 0; j < n; ++j) {
            assert(pos[idst + j] == -1 && seq[idst + j].none());

            pos[idst + j] = pos[isrc + j];
            ext[idst + j] = ext[isrc + j];
            seq[idst + j] = seq[isrc + j];

            pos[isrc + j] = -1;
            ext[isrc + j].reset();
            seq[isrc + j].reset();
        }
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        pos[i] = p;

        used.insert(i);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

private:
    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // the set seq_pos[s] holds one (pos, cell) pair per cell that carries sequence s, ordered by position
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    // and upper_bound() on a position finds the nearest cell of the sequence in logarithmic time
    //
    // the cell index is part of the key because a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    std::set<std::pair<llama_pos, uint32_t>> seq_pos[LLAMA_MAX_SEQ];

    // strixllama: the cells of each sequence in index order, kept in step with seq_pos. A unified cache
    // keeps a conversation in one run of cells, and the graph of a batch views only the run of its own
    // conversations (llama_kv_cache::get_kv_window), which needs a sequence's first and last cell.
    std::set<uint32_t> seq_cells[LLAMA_MAX_SEQ];

    // strixllama: seq_run's knowledge per sequence - whether it has to look again, and what it found
    mutable std::vector<uint8_t> run_dirty = std::vector<uint8_t>(LLAMA_MAX_SEQ, 0);
    mutable std::vector<uint8_t> run_ok    = std::vector<uint8_t>(LLAMA_MAX_SEQ, 1);

    // strixllama: the entries [isrc, isrc + n) of an index of cells, renumbered to [idst, idst + n) in place
    static void move_index(std::set<uint32_t> & index, uint32_t isrc, uint32_t idst, uint32_t n) {
        std::vector<std::set<uint32_t>::node_type> nodes;
        nodes.reserve(n);
        for (auto it = index.lower_bound(isrc); it != index.end() && *it < isrc + n; ) {
            auto next = std::next(it);
            nodes.push_back(index.extract(it));
            it = next;
        }
        auto hint = index.lower_bound(idst);
        for (auto & nh : nodes) {
            nh.value() = nh.value() - isrc + idst;
            hint = std::next(index.insert(hint, std::move(nh)));
        }
    }

    // helper functions for updating `seq_pos`, once cell at a time:

    void seq_pos_dec(llama_seq_id s, uint32_t i) {
        // strixllama: the removal of the last cell keeps a run one (and a sequence that is none, none)
        if (!run_dirty[s] && i != *seq_cells[s].rbegin()) {
            run_dirty[s] = 1;
        }

        const auto n = seq_pos[s].erase({ pos[i], i });
        assert(n == 1);
        GGML_UNUSED(n);

        seq_cells[s].erase(i);

        if (seq_cells[s].empty()) {
            run_dirty[s] = 0;
            run_ok[s]    = 1;
        }
    }

    void seq_pos_inc(llama_seq_id s, uint32_t i) {
        // strixllama: an append at the next cell with the next position keeps a run one (and a sequence that is none, none)
        if (seq_cells[s].empty()) {
            run_dirty[s] = 0;
            run_ok[s]    = 1;
        } else if (!run_dirty[s]) {
            const uint32_t last = *seq_cells[s].rbegin();
            if (i != last + 1 || pos[i] != pos[last] + 1) {
                run_dirty[s] = 1;
            }
        }

        seq_pos[s].insert({ pos[i], i });

        seq_cells[s].insert(i);
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_dec(s, i);
            }
        }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_inc(s, i);
            }
        }
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
