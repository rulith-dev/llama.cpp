# Strix Llama branch

This branch is `pwilkin/llama.cpp` at `f5daaa3` plus the Strix Llama patch set: 24 files (20 modified, 4 added) that make Qwen3.8-Flash-Next fast on one AMD Strix Halo machine (Ryzen AI Max+ 395, Radeon 8060S / gfx1151) on Windows, built with HIP against TheRock ROCm 10.1.

It is published as one commit so the delta can be read as a diff. The patches themselves, the build (`bootstrap/bootstrap.py`), a replay that rebuilds this exact tree from clean upstream and checks every file by hash, the manager, the desktop app, and the measurements live in the main repository:

https://github.com/rulith-dev/strixllama

What the delta does, in one line each (details in that repository's `patches/MANIFEST.md`):

- sparse-attention (QSA) block-key cache, decode-time gather, compact metadata, an image guard for M-RoPE cells
- MTP speculative decoding: the draft context's ubatch capped separately, the draft's own sparse-attention prefill, shape-keyed HIP graphs
- per-layer-embedding table read with unbuffered, overlapped direct I/O instead of the pager (`--load-mode none --lazy-mode on-direct`), so a 93.7 GB model runs in a 96 GB carve with the table on disk
- a few fused HIP kernels (chain fusion, getrows cast, an IQ3_S vec-dot, RDNA 3.5 row tiles for mmvq)

Measured on the target machine at 85K tokens of context: prefill 886 tok/s, decode 34.9 tok/s (MTP acceptance 68%). Numbers, method and the ways we measured it wrong first: `docs/results.md` and `docs/measuring.md` in the main repository.

Licence: upstream's (MIT). The patch set is MIT as well.
