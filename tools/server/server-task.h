#pragma once

#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>
#include <set>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_GET,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
    SERVER_TASK_TYPE_DISK_PERSIST,    // strixllama: POST /strix/persist, before the server is stopped
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params);

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    server_slot_stats stats;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    server_slot_stats stats;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

// used by /metrics API
struct server_task_result_metrics : server_task_result {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    int n_processing_slots = 0;
    int n_tasks_deferred = 0;

    server_metrics metrics;

    virtual json to_json() override;

    struct metric_item {
        std::string name;
        std::string description;
        double value; // prometheus values are always float64
    };
    std::string to_metrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
    int n_idle_slots = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
        };
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // strixllama: a disk tier under the RAM cache; the files and their formats are described in
    // server-task.cpp. Entries are content-addressed and written by a background thread, so saving a
    // conversation again writes only what changed, never the whole state in one burst.
    struct disk_ckpt_ref {
        uint64_t key;
        int64_t  n_tokens;
        int32_t  pos_min;
        int32_t  pos_max;
        uint64_t size_tgt;
        uint64_t size_dft;
        uint64_t size_spec;
    };
    // strixllama: version 3 keeps a conversation's attention rows by position (llama_strix_kv_*) in runs, written
    // once each as the conversation grows and never rewritten: a run holds up to disk_run positions, and one that
    // ends short (the conversation left its slot there) stays that way - the next run starts where it ends
    struct disk_run_ref {
        uint64_t hi;                    // XXH3-128 of the rows
        uint64_t lo;
        int32_t  pos0;                  // the first position in the file
        int32_t  n;                     // positions in the file
        int32_t  n_use;                 // positions of it the entry uses, from pos0 (an edit can end it early)
    };
    // a slot's checkpoints handed back to the store: (n_tokens, pos_min, pos_max) -> the ref it is stored under
    using ckpt_paged_map = std::map<std::tuple<int64_t, int32_t, int32_t>, disk_ckpt_ref>;

    // strixllama: `before_restore` is called with the token count of the entry about to replace `prompt`, before
    // its state goes into the KV cache - the caller makes room for it there, and lets go of what the old prompt held.
    // With `paged_out`, an entry restored from the disk tier leaves its checkpoints there: `prompt` gets them without
    // their bytes and `paged_out` says where they are (pinned), exactly as if the slot had paged them out itself - a
    // 79K-token conversation carries ~3.3 GB of checkpoints of which a rewind reads one.
    //
    // strixllama: a version 3 entry is restored straight into the contexts' cells, a run at a time, and never whole in
    // memory; then `runs_out` (when given) holds the runs that cover the restored positions, for the slot to write on
    // from without writing them again. A slot whose conversation was restored any other way gets them cleared.
    struct disk_runs_state {
        std::vector<disk_run_ref> tgt;
        std::vector<disk_run_ref> dft;
        llama_tokens              tokens;  // what the runs were written for: they hold only while the prompt starts so
    };
    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
              const std::function<void(size_t)> & before_restore = nullptr, ckpt_paged_map * paged_out = nullptr,
              disk_runs_state * runs_out = nullptr);

    void update();
    struct disk_chunk_ref {
        uint64_t hi;                    // XXH3-128 of the chunk's bytes
        uint64_t lo;
        uint32_t size;
    };
    struct disk_entry {
        std::string   path;
        server_tokens tokens;
        uint64_t      bytes     = 0;    // the entry's own file
        int64_t       order     = 0;    // larger = more recently written or used
        int32_t       version   = 0;    // 1: one self-contained file, 2: a manifest over chunks/ and ckpt/, 3: over runs/ and ckpt/
        uint64_t      main_size = 0;
        uint64_t      drft_size = 0;
        std::vector<disk_chunk_ref> main;
        std::vector<disk_chunk_ref> drft;
        std::vector<disk_ckpt_ref>  ckpts;
        // version 3: bytes per position of the target's and the draft's rows, their runs from position 0, and
        // how many tokens can be restored exactly - the latest checkpoint the runs reach
        uint64_t row_tgt = 0;
        uint64_t row_dft = 0;
        std::vector<disk_run_ref> runs_tgt;
        std::vector<disk_run_ref> runs_dft;
        int64_t  n_exact = 0;
    };
    struct disk_object {                // a chunk or checkpoint file, shared between entries
        uint64_t bytes = 0;
        int32_t  refs  = 0;
    };
    struct disk_job {                   // a gathered state on its way to disk
        server_tokens tokens;
        std::vector<uint8_t> main;
        std::vector<uint8_t> drft;
        std::vector<disk_ckpt_ref> ckpts;                                      // every checkpoint of the prompt
        std::vector<std::pair<uint64_t, common_prompt_checkpoint>> ckpt_data;  // copies of the ones not stored yet
        std::vector<uint64_t> pinned;                                          // stored ones it names: held until counted
        // version 3: every run of the entry, the rows of the ones this job writes, and the stored ones it names
        int32_t  version = 2;
        int32_t  id_slot = -1;
        uint64_t row_tgt = 0;
        uint64_t row_dft = 0;
        int64_t  n_exact = 0;
        std::vector<disk_run_ref> runs_tgt;
        std::vector<disk_run_ref> runs_dft;
        std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::vector<uint8_t>>> run_data;
        std::vector<std::pair<uint64_t, uint64_t>> run_pinned;
    };

    std::string disk_dir;
    uint64_t    disk_limit    = 0;      // bytes, 0 = no disk tier
    bool        disk_has_mtmd = false;
    int32_t     disk_version  = 2;      // the manifest format this server writes; the store holds no older one
    int32_t     disk_block    = 4096;   // tokens a conversation grows by before it is written again
    int32_t     disk_run      = 4096;   // version 3: positions in a run (STRIX_PROMPT_CACHE_RUN)
    int64_t     disk_seq      = 0;
    uint64_t    disk_bytes    = 0;      // entries, and every object once

    // guarded by disk_mu; only the writer thread deletes files
    std::mutex                                           disk_mu;
    std::condition_variable                              disk_cv;
    std::vector<disk_entry>                              disk_index;
    std::map<std::pair<uint64_t, uint64_t>, disk_object> disk_chunks;
    std::map<uint64_t, disk_object>                      disk_ckpts;
    std::map<std::pair<uint64_t, uint64_t>, disk_object> disk_runs;
    std::deque<std::unique_ptr<disk_job>>                disk_queue;
    std::vector<disk_entry>                              disk_doomed;    // unreadable, for the writer to remove
    std::vector<std::pair<uint64_t, uint64_t>>           disk_bad_chunks;   // found damaged or missing by a read
    std::vector<uint64_t>                                disk_bad_ckpts;
    std::vector<std::pair<uint64_t, uint64_t>>           disk_bad_runs;
    std::vector<uint64_t>                                disk_unpin;        // pins to release, by the writer
    std::vector<std::pair<uint64_t, uint64_t>>           disk_unpin_runs;
    std::function<void()>                                disk_on_written;   // called by the writer after each job
    disk_job *                                           disk_current = nullptr;
    bool                                                 disk_stop    = false;
    std::thread                                          disk_thread;

    ~server_prompt_cache();

    // entries live in <root>/<model_tag>: a state is only meaningful to the model that computed it. `version`
    // is the format this server writes (3 where its contexts serve rows by position, else 2): entries of an
    // earlier one are deleted as the store is opened
    void     set_disk(const std::string & root, size_t limit_mib, bool has_mtmd, const std::string & model_tag, int32_t version);
    bool     has_disk() const { return disk_limit > 0; }
    uint64_t disk_size();
    bool     disk_busy();                  // a job is waiting: a save that would not wait can skip its work
    bool     disk_drain(int64_t timeout_ms);   // wait until the writer has nothing queued or in hand; false on timeout
    // how much of `tokens` is on disk or on its way there: the longest stored prompt it starts with, or all
    // of it when a stored prompt starts with it
    size_t   disk_covered(const server_tokens & tokens, bool with_jobs = true);
    // never for a prompt with media in it: the disk tier does not store those
    bool     disk_wants(const server_prompt & prompt) {
        return prompt.tokens.get_text_tokens().size() == prompt.tokens.size() && disk_covered(prompt.tokens) < prompt.tokens.size();
    }
    // hand a gathered state to the writer; `wait` blocks while the queue is full instead of giving up. Checkpoints
    // with no bytes are ones the slot handed back, and `paged` says where they are
    bool     persist(const server_prompt & prompt, std::vector<uint8_t> && data_main, std::vector<uint8_t> && data_drft, bool wait,
                     const ckpt_paged_map * paged = nullptr);
    // version 3: hand a conversation's runs to the writer. `tokens` are its first n positions and `runs_tgt` /
    // `runs_dft` every run it has from position 0: those with rows in `run_data` are new, the rest must be in the
    // store. `end`, when not null, is the recurrent state at exactly n (a conversation leaving its slot); the other
    // checkpoints come from `ckpts` and `paged` as in persist(), those past n left out - but only those in
    // `writable` are written, the rest named when the store has them already. `wait` blocks while the queue is full
    // instead of giving up. A job whose runs the store lost meanwhile is dropped, and the slot told through
    // runs_lost() to start its runs over.
    bool persist_runs(int32_t id_slot, const server_tokens & tokens, const std::list<common_prompt_checkpoint> & ckpts,
                      const std::set<const common_prompt_checkpoint *> & writable,
                      const ckpt_paged_map * paged, const common_prompt_checkpoint * end, uint64_t row_tgt, uint64_t row_dft,
                      const std::vector<disk_run_ref> & runs_tgt, const std::vector<disk_run_ref> & runs_dft,
                      std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::vector<uint8_t>>> && run_data, bool wait);
    bool runs_lost(int32_t id_slot);           // and clears it
    // the ref a run of rows is stored under: named by their XXH3-128
    static disk_run_ref make_run_ref(const std::vector<uint8_t> & rows, int32_t pos0, int32_t n);
    // positions of `tokens` the store brings back exactly, counting jobs on their way: a slot that is leaving writes
    // only when this falls short of its conversation
    size_t   disk_exact(const server_tokens & tokens);

    // checkpoints are cold data - only a rewind reads one - so an idle slot drops the bytes of those the store
    // holds; returns the bytes freed
    uint64_t ckpt_page_out(const server_tokens & tokens, std::list<common_prompt_checkpoint> & ckpts, ckpt_paged_map & paged);
    // and reads one back when a rewind picks it. False when it is not this conversation's any more, or unreadable
    bool     ckpt_page_in(const server_tokens & tokens, const ckpt_paged_map & paged, common_prompt_checkpoint & c);
    // a paged-out checkpoint holds a reference in the store, so eviction cannot delete it while the slot may
    // still rewind to it; the slot lets go when it is cleared or takes another conversation
    void     ckpt_release(ckpt_paged_map & paged);
    // appends the best disk entry to `states` when it beats (f_keep_best, f_sim_best); returns it or nullptr. With
    // `paged`, a version 2 entry's checkpoints stay on disk: pinned and named in `paged`, bytes read on demand
    server_prompt_cache_state * load_from_disk(const server_tokens & tokens_new, float & f_keep_best, float & f_sim_best,
                                               ckpt_paged_map * paged = nullptr);

    // version 3's side of load(): the best such entry, if it beats (f_keep_best, f_sim_best), restored
    bool   load_runs(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft,
                     int32_t id_slot, float & f_keep_best, float & f_sim_best, const std::function<void(size_t)> & before_restore,
                     ckpt_paged_map * paged_out, disk_runs_state * runs_out, bool & restored);

    // the writer's side, and helpers that expect disk_mu held
    void   disk_writer();
    void   disk_write(disk_job & job);
    void   disk_write_runs(disk_job & job);
    std::set<int32_t> disk_runs_lost;          // slots whose last job named runs the store no longer had
    void   disk_drop(const disk_entry & e, bool remove_file = true);
    void   disk_evict(const std::string & keep);
    void   disk_retire_bad();
    void   disk_release_pins();
    size_t disk_covered_locked(const server_tokens & tokens, bool with_jobs);
    bool   disk_in_flight(const server_tokens & tokens, float f_sim_base);
    bool   disk_ckpt_in_flight(uint64_t key);
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
