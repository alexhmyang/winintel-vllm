# winintel-vllm v0.1.0 — Qwen3.8-27B on V100

Author: 数萌AI服务器 (winintel)

This release adapts vLLM 0.17.0 for Qwen3.8-27B text inference on V100.
It provides SM70 attention, official ModelOpt mixed FP8/NVFP4 loading,
FP8 KV storage, MTP speculative decoding, and target/draft CUDA Graph.

The English README includes measured concurrency and full-input
long-context results, deployment commands, model sources, and reproduction
instructions. The Chinese README is retained as `README.zh-CN.md`.
Performance claims are scoped to the checkpoint, prompt, concurrency,
power limit, and runtime configuration recorded with each benchmark.

At the maximum 300W power limit per V100, MTP5 reached 592.92 tokens/s
at eight concurrent coding requests and 880.95 tokens/s at sixteen.
Compared with MTP11 under matching conditions, two-round mean throughput
increased by 14.6% and 9.6%, respectively. All 150 programs from the
MTP11/MTP9/MTP5 comparison passed independent behavior checks and their
generated tests. The fastest single-request configuration was MTP9.
Full long-input performance is documented separately from short prompts.
DFlash2 requires explicit development opt-in and does not pass draft
numerical validation on this V100 configuration. MTP is the supported
deployment method; no DFlash2 speedup is claimed.

Release assets:

- `winintel-vllm-0.1.0-qwen3.8-install.tar.gz`: installer including official base source.
- `winintel-vllm-0.1.0-qwen3.8-source.tar.gz`: adaptation source overlay.
- Corresponding `.sha256` files.

Dependencies and model weights are obtained separately.
See `NOTICE` and `THIRD_PARTY.md` for source and license attribution.
AI assistance was used for development and documentation.
The packaged source and existing benchmark records are provided for review
and reproduction. Fresh-machine installation timing and installer acceleration
are being validated separately.
