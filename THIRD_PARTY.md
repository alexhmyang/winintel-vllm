# Source attribution

Adaptation author: 数萌AI服务器 (winintel)

| Component | Source and use | License / distribution |
| --- | --- | --- |
| vLLM 0.17.0 | https://github.com/vllm-project/vllm; engine, Qwen models, scheduling, and tensor parallelism | Apache-2.0; upstream attribution retained |
| PyTorch | https://github.com/pytorch/pytorch; tensors, extensions, and optional CUTLASS prefill | BSD-style and third-party terms; runtime dependency |
| compressed-tensors | https://github.com/neuralmagic/compressed-tensors; NVFP4 loading interfaces | Apache-2.0; runtime dependency |
| DFlash / speculators | https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py; public speculators 0.9.0 layers | MIT; full notice retained in `dflash2_layers.py` |
| NVIDIA CUDA / cuBLAS | https://developer.nvidia.com/cuda-toolkit; SM70 compilation and matrix operations | NVIDIA terms; system dependency |
| Project adaptation | `csrc/winintel/`, `winintel_kernels/`, `vllm/winintel/`, launch and validation scripts | Per-file Apache-2.0 or MIT; maintained by winintel AI with AI assistance |

File hashes are listed in `release-manifest.json`.
Original component licenses and copyright notices are retained.

The installer archive includes official vLLM 0.17.0 base source under
Apache-2.0, with one unused test containing a synthetic credential URL omitted.
The original upstream archive hash and the bundled archive hash are recorded
separately in the release manifest. Runtime and build sources and their
license notices are retained. The smaller source archive is an adaptation overlay.
Runtime dependencies are installed with uv from their package indexes.

Model weights and tokenizers are obtained separately under their publishers'
model cards and licenses.
