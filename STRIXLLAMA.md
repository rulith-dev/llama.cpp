# Strix Llama branch

This branch is `pwilkin/llama.cpp` at `f5daaa3` plus the Strix Llama patch set: 34 files (30 modified, 4 added) that make Qwen3.8-Flash-Next fast on one AMD Strix Halo machine (Ryzen AI Max+ 395, Radeon 8060S / gfx1151) on Windows, built with HIP against TheRock ROCm 10.1.

Each release adds one commit, so the delta can be read either whole or a step at a time. The patches themselves, the build (`bootstrap/bootstrap.py`), a replay that rebuilds this exact tree from clean upstream and checks every file by hash, the manager, the desktop app, and the measurements live in the main repository:

https://github.com/rulith-dev/strixllama

What the delta does, in one line each (details in that repository's `patches/MANIFEST.md`):

- sparse-attention (QSA) block-key cache, decode-time gather, compact metadata, an image guard for M-RoPE cells
- mixed-sequence batches stay on the sparse path, so several conversations at depth share a step instead of falling back to dense attention over the whole pool
- MTP speculative decoding: the draft context's ubatch capped separately, the draft's own sparse-attention prefill, shape-keyed HIP graphs
- per-layer-embedding table read with unbuffered, overlapped direct I/O instead of the pager (`--load-mode none --lazy-mode on-direct`), so a 93.7 GB model runs in a 96 GB carve with the table on disk; the server gathers the next prompt batch's rows while the current one computes (`llama_strix_prefetch`)
- a disk tier under the server's prompt cache: a content-addressed store kept per model, written a block at a time by a background writer and read straight from the device rather than through the page cache; idle conversations stay in their slots and page their checkpoints out to it, and a restore reads the chunks in parallel and leaves the checkpoints there; usable whether or not the server drafts
- a few fused HIP kernels (chain fusion, getrows cast, an IQ3_S vec-dot, RDNA 3.5 row tiles for mmvq, a vector path for few-row matmuls past 8 columns), and the base's hyper-connection gate and PLE conv fusions extended to Unsloth's weight types (Q8_0, F32) with the unfused path's numerics
- smaller: a bitonic sort for the QSA3 selection rows, conv-state tails copied without a cont

Measured on the target machine at 85K tokens of context: prefill 886 tok/s, decode 34.9 tok/s (MTP acceptance 68%). Numbers, method and the ways we measured it wrong first: `docs/results.md` and `docs/measuring.md` in the main repository.

Licence: upstream's (MIT). The patch set is MIT as well.
