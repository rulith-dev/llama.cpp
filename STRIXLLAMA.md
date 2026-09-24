# Strix Llama branch

This branch is `pwilkin/llama.cpp` at `f5daaa3` plus the Strix Llama patch set: 42 files (38 modified, 4 added) that make Qwen3.8-Flash-Next fast on one AMD Strix Halo machine (Ryzen AI Max+ 395, Radeon 8060S / gfx1151) on Windows, built with HIP against TheRock ROCm 10.1.

Each release adds one commit, so the delta can be read either whole or a step at a time. The patches themselves, the build (`bootstrap/bootstrap.py`), a replay that rebuilds this exact tree from clean upstream and checks every file by hash, the manager, the desktop app, and the measurements live in the main repository:

https://github.com/rulith-dev/strixllama

What the delta does, in one line each (details in that repository's `patches/MANIFEST.md`):

- sparse-attention (QSA) block-key cache (a rebuild scores with the keys as the cache stores them, as every later graph does), decode-time gather, compact metadata, an image guard for M-RoPE cells
- mixed-sequence batches stay on the sparse path, so several conversations at depth share a step instead of falling back to dense attention over the whole pool
- with several slots in one pool, a batch's sparse-attention block list covers only its own conversations, so idle ones cost a decoding conversation nothing per step; a slot's state is read from the device one run of cell ranges at a time rather than one range at a time (a conversation decoded alongside others had stalled the server 6-10 s at every save)
- with several slots, every conversation keeps one run of the pool and a batch's graph views only the runs of its own conversations, so a conversation beside idle ones computes, and costs, what it does on a single slot - the MTP draft's dense attention included; a batch that does not fit has the pool laid out again on the device first (conversations kept in order, the same room after each one of the batch)
- compute buffers get a little headroom, so a graph a few MiB larger no longer frees and reallocates one: ROCm on Windows keeps a freed buffer's commit, and four long conversations ran the machine into its commit limit ("bad allocation")
- a slot named by id that holds nothing reads its conversation back from the prompt cache: its f_keep was 0/0, a NaN that skipped the cache, so a 173K-token conversation sent back to its emptied slot was processed again (186 s) instead of read back from the disk tier (7 s)
- a slot that fails part way through its turn loses its conversation, and a prompt batch starts only on a cache whose positions match its tokens: near the commit limit an exception had left a slot's tokens and memory apart, and the next request fed the recurrent state positions it had seen or skipped (GitHub issue #1); a leaving conversation also keeps a checkpoint every 32K tokens in the disk tier
- the disk tier and that check with a vision projector loaded: every prompt counts as a media prompt to server_tokens::get_tokens() then, which asserts, so version 3 aborted the server when the disk tier and image input were both on; it reads the text tokens, and the check skips only prompts that hold media
- MTP speculative decoding: the draft context's ubatch capped separately, the draft's own sparse-attention prefill, shape-keyed HIP graphs
- per-layer-embedding table read with unbuffered, overlapped direct I/O instead of the pager (`--load-mode none --lazy-mode on-direct`), so a 93.7 GB model runs in a 96 GB carve with the table on disk; the server gathers the next prompt batch's rows while the current one computes (`llama_strix_prefetch`)
- a disk tier under the server's prompt cache, kept per model: a conversation's attention rows by position (`llama_strix_kv_*`), in runs of 4096 positions written once each as they are computed, named by their hash; its recurrent state only when it leaves memory (evicted, or on `POST /strix/persist` before a stop); a restore streams the runs back and puts back the latest checkpoint they reach, bitwise what the resident conversation computes. One background writer, reads and writes straight from the device rather than through the page cache, and the store keeps only the format the build writes
- a few fused HIP kernels (chain fusion, getrows cast, an IQ3_S vec-dot, RDNA 3.5 row tiles for mmvq, a vector path for few-row matmuls past 8 columns), and the base's hyper-connection gate and PLE conv fusions extended to Unsloth's weight types (Q8_0, F32) with the unfused path's numerics
- the routed IQ3_S expert gate/up + SwiGLU, prefill's largest kernel, rebuilt for gfx1151, where the VALU and the WMMA unit never overlap: its dequantization spread over the block and cut to fewer instructions per weight, its loads kept off the critical path - 1.8x, bitwise the same output
- smaller: a bitonic sort for the QSA3 selection rows, conv-state tails copied without a cont

Measured on the target machine: prefill 983 tok/s over 95.6K tokens of real text, decode 34.9 tok/s at 85K tokens of context (MTP acceptance 68%). Numbers, method and the ways we measured it wrong first: `docs/results.md` and `docs/measuring.md` in the main repository.

Licence: upstream's (MIT). The patch set is MIT as well.
