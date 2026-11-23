<div align="center">

# winintel-vllm

### 🚀 winintel AI · Large-model inference on NVIDIA V100

**💎 More value from proven hardware.**

**⚡ 880.9 tokens/s at 16 requests · 🔥 592.9 tokens/s at 8 requests · 🧠 256K context tested**

[🌐 Website](https://winintel.com) · [📊 Performance](#measured-performance) ·
[🛠️ Installation](#installation) · [🇨🇳 中文](README.zh-CN.md)

[🎬 Bilibili](https://space.bilibili.com/429959998) ·
[▶️ YouTube · winintel AI](https://www.youtube.com/@winintelcom)

[⭐ Star this project](https://github.com/alexhmyang/winintel-vllm) ·
[🔔 Watch releases](https://github.com/alexhmyang/winintel-vllm/subscription) ·
[📦 Download v0.1.0](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0)

**v0.1.0 · Qwen3.8-27B · Volta / SM70**

</div>

🚀 **Turn your NVIDIA V100 servers into practical AI infrastructure.**
winintel-vllm by **winintel AI** brings Qwen3.8-27B coding assistants and
private text services to proven hardware, with **up to 880.95 tokens/s
aggregate decode** in measured 16-request coding tests and **full 256K
single-request inputs tested**. Built on vLLM 0.17.0 and existing open-source
components, it combines faster concurrent serving, long-context processing,
and straightforward deployment.

- ⚡ **More concurrent work:** best measured coding throughput of **592.92 tokens/s at 8 requests** and **880.95 tokens/s at 16**, using 8 × V100 32GB.
- 🧠 **Room for long documents:** a **262,144-token context limit**, with full single-request long-input code generation and exact clause retrieval tested.
- 🏎️ **Acceleration for V100:** MTP speculative decoding, target and draft CUDA Graph, and the `WININTEL_SM70` attention backend.
- 💾 **Flexible model loading:** NVFP4/FP8 checkpoint compatibility with FP16 execution on V100, plus FP8 KV storage.
- 🔌 **Easy application integration:** an OpenAI-compatible API with live SSE streaming for coding assistants and private text applications.
- 🛠️ **One-command installation:** bundled base source, uv environment setup, SM70 extension builds, and documented deployment profiles.
- ✅ **Results you can check:** reproducible benchmark scripts, raw timing records, and **150/150 validated programs** in the MTP configuration comparison.
- 🤝 **Built for collaboration:** transparent component attribution and an open invitation to improve V100 inference together.

## Capabilities

| Feature | Support |
| --- | --- |
| Tensor parallelism | TP2 / TP4 / TP8 configurations; TP4 and TP8 measured |
| Attention backend | `WININTEL_SM70` |
| Acceleration | MTP3/5/7/9/11; target and draft CUDA Graph |
| Checkpoints | ModelOpt mixed FP8/NVFP4 and compressed-tensors NVFP4 |
| V100 execution | Quantized weights expanded to FP16 at load time; FP16 compute |
| KV cache | FP16 or FP8 E4M3 |
| Context limit | 262,144 tokens per request |
| API | OpenAI-compatible HTTP API with SSE streaming |

V100 does not execute these checkpoints with native FP4 or FP8 tensor cores.
FP8 here describes KV storage; the model weights and activations run in FP16.
DFlash2 adapters are for development only: draft numerical validation does
not pass on the tested V100 configuration. There is no validated DFlash2
speedup. The measured performance below uses MTP.

## Measured performance

### Official checkpoint: English code generation

**Hardware:** 8 × V100-SXM2 32GB, TP8.  
**Checkpoint:** [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4),
revision `482ca0f3832238542f8f5295dde86b5f22711d80`;
every weight shard checked against its official SHA256.  
**Configuration:** FP16 compute and KV, MTP11, target and draft CUDA Graph,
16,384-token context, maximum 16 sequences.

| Concurrent requests | Rounds | Best aggregate decode | Mean aggregate decode | Maximum TTFT | Code validation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | **93.26 tokens/s** | 93.06 tokens/s | 188 ms | 3/3 |
| 2 | 3 | **191.64 tokens/s** | 191.38 tokens/s | 460 ms | 6/6 |
| 4 | 3 | **332.96 tokens/s** | 332.76 tokens/s | 410 ms | 12/12 |
| 8 | 3 | **518.46 tokens/s** | 518.19 tokens/s | 455 ms | 24/24 |
| 16 | 2 | **802.81 tokens/s** | 802.10 tokens/s | 599 ms | 32/32 |

The table combines the three-round 1–8-request measurement with the
two-round 16-request measurement. Configuration comparisons follow below.

Each request generates an English Python LRU implementation and at least
eight executable unit tests. All 77 outputs passed syntax, independent LRU
behavior checks, and their generated tests. Requests start at a barrier and
use independent streaming HTTP sessions. Coding runs use a four-token
warmup before the measured requests. Inputs contain 100 tokens;
outputs contain 1,677–1,746 tokens. Greedy decoding is used, thinking is disabled,
EOS is respected, and the output budget is 2,048 tokens.

Decode throughput excludes the first streamed block. These are short-prompt
results and do not describe decoding after a 256K input.
The separately measured initial best single-request result was **93.56 tokens/s**.

[Round results, GPU memory and checkpoint verification tables](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

### Configuration comparison

MTP11, FP16 KV, 16K service context, CUTLASS prefill enabled,
two rounds per concurrency:

| Concurrent requests | Best aggregate decode | Mean aggregate decode | Maximum TTFT |
| ---: | ---: | ---: | ---: |
| 1 | **93.83 tokens/s** | 93.80 tokens/s | 194 ms |
| 8 | **517.92 tokens/s** | 516.89 tokens/s | 500 ms |
| 16 | **802.81 tokens/s** | 802.10 tokens/s | 599 ms |

All 50 generated programs passed validation.
Peak per-GPU memory use was **29,723 MiB**.

Changing only the MTP draft length produced these two-round results:

| Concurrent requests | MTP11 mean decode | MTP9 mean decode | MTP5 mean decode | MTP5 vs MTP11 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 93.80 tokens/s | **94.03 tokens/s** | 86.78 tokens/s | −7.5% |
| 8 | 516.89 tokens/s | 547.06 tokens/s | **592.26 tokens/s** | +14.6% |
| 16 | 802.10 tokens/s | 831.58 tokens/s | **879.43 tokens/s** | +9.6% |

All **150 outputs** across the three configurations passed code validation.
MTP5 best rounds reached **592.92 tokens/s**
at eight requests and **880.95 tokens/s** at sixteen requests.
Use MTP9 for this single-request task and MTP5 for higher concurrency.
These comparisons use matching runs of the same code task.

[MTP11 / MTP9 / MTP5 comparison and round tables](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

### Full long inputs

Official checkpoint, TP8, FP8 KV, MTP11, PyTorch CUTLASS prefill,
target and draft CUDA Graph, one request.

| Workload | Actual input tokens | TTFT | Input / TTFT | Client decode | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| English code | 8,192 | **2.21 s** | 3,704.29 tokens/s | **82.37 tokens/s** | Passed |
| English code | 260,096 | **160.66 s** | 1,618.89 tokens/s | **10.83 tokens/s** | Passed |
| Exact clause retrieval | 8,192 | 2.48 s | 3,302.89 tokens/s | 98.80 tokens/s | 3/3 |
| Exact clause retrieval | 32,768 | 9.68 s | 3,385.23 tokens/s | 91.76 tokens/s | 3/3 |
| Exact clause retrieval | 262,016 | 163.64 s | 1,601.17 tokens/s | 18.73 tokens/s | 3/3 |

Each entry is one measured round. Code inputs use repeated archived English
reference code followed by the implementation task. Both generated programs
passed behavior checks and at least eight generated unit tests.
The 256K code request produced 1,668 tokens; the retrieval request produced 73.
Retrieval validates exact Chinese contract clauses at 5%, 50%, and 95%
of the input. These workloads measure different output behavior.

Input / TTFT includes HTTP transfer, prefill, and first-block generation;
it is not a pure GPU prefill measurement. Separate engine metrics, where
provided, measure vLLM's scheduler PREFILL phase.
The first-token deadline is **180 seconds**.
Four simultaneous full 256K inputs have not been validated.

### Additional long-input measurements

MTP5, FP8 KV, 262,144-token service context, maximum four sequences,
12,288 batched tokens, CUTLASS prefill and CUDA Graph:

| Workload | Input tokens per request | Concurrency | Maximum TTFT | Aggregate client decode | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| English code | 8,192 | 4 | 8.56 s | **191.72 tokens/s** | 4/4 programs |
| Exact clause retrieval | 32,768 | 1 | 9.48 s | 74.51 tokens/s | 3/3 clauses |
| Exact clause retrieval | 262,016 | 1 | 171.77 s | 18.95 tokens/s | 3/3 clauses |

The retrieval measurements have no warmup and expose a separate engine
PREFILL measurement:

| Input tokens | Engine PREFILL time | Input / engine PREFILL time |
| ---: | ---: | ---: |
| 32,768 | **9.37 s** | **3,498.94 tokens/s** |
| 262,016 | **171.25 s** | **1,530.03 tokens/s** |

These are vLLM scheduler phase times, rather than isolated CUDA kernel times.
Each long-input entry is one measured round.
The MTP5/12,288 configuration did not improve full-input TTFT over the
MTP11/6,144 baseline: 171.77 versus 163.64 seconds. Retain the latter
configuration when prioritizing measured single-request long-input latency.
Four full 256K inputs remain untested; their estimated prefill time exceeds
the 180-second first-token budget.

[4-request 8K, 32K / 256K and engine metrics tables](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

Four independent 8K retrieval sessions were also measured with MTP11,
6,144 batched tokens, FP8 KV and the 262,144-token service context:

| Input tokens per request | Concurrency | Mean engine PREFILL | Input / mean PREFILL per request | Maximum TTFT | Aggregate client decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4 | **3.98 s** | **2,059.69 tokens/s** | **9.24 s** | **39.48 tokens/s** |

All four outputs reproduced all three required clauses exactly. No warmup
was used. Engine PREFILL counters cover four completed requests, with
15.91 seconds of summed phase time. The per-request rate is not aggregate
GPU throughput: phase durations can overlap, and TTFT includes queueing.
Retrieval produces only 73 output tokens per request and should not be
compared directly with the longer code-generation outputs.

[4-request retrieval and engine PREFILL tables](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

[Long-context code and GPU memory tables](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

Historical TP4 and other checkpoint results remain in the
[Chinese documentation](README.zh-CN.md). They are not directly comparable
to the official-checkpoint table above.

## Models

| Resource | Link |
| --- | --- |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| NVIDIA quantized checkpoint | [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |
| Matching DFlash2 draft | [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) |
| Base model mirror | [HF Mirror](https://hf-mirror.com/Qwen/Qwen3.8-27B) |
| NVIDIA checkpoint mirror | [HF Mirror](https://hf-mirror.com/nvidia/Qwen3.8-27B-NVFP4) |

Use a complete local checkpoint with config, tokenizer, and all weight shards.
The official Qwen3.8 checkpoint declares the internal `qwen3_5` architecture;
the installer checks this architecture and the 5,120-dimensional model.
This release supports text inference.

## Installation

[Complete Ubuntu, driver, CUDA, deployment and startup guide](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0).

**Memory requirements:** this release expands 27B weights to FP16, requiring
about 50 GiB for weights alone. One V100 32GB or two V100 16GB cannot run
this 27B model; choose a 7B/8B model for that memory capacity.
Four and eight V100 32GB configurations have been tested.
Two V100 32GB remain pending hardware validation.

**Requirements:** Linux x86_64, NVIDIA V100, NVIDIA driver,
CUDA Toolkit 12.8, C/C++ build tools, Git, and curl.
The recommended tested configuration is 8 × V100 32GB.

Extract `winintel-vllm-0.1.0-qwen3.8-install.tar.gz`, then run:

```bash
bash install.sh /path/to/Qwen3.8-27B-NVFP4
```

The installer includes official vLLM 0.17.0 base source, creates a Python 3.12
environment with uv, installs dependencies and pre-commit hooks, builds
SM70 extensions, and starts the service. Model weights are downloaded
separately. The smaller source archive requires the matching base-source
archive under `vendor/` for installation.

Default package mirrors are Tsinghua PyPI and Nanjing University's
PyTorch CUDA 12.8 mirror. To use official package sources:

```bash
UV_INDEX_URL=https://pypi.org/simple \
WININTEL_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
bash install.sh /path/to/model
```

To install without starting a model:

```bash
bash install.sh
```

Download the verified checkpoint from the installed `runtime` directory:

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/hf download \
  nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir /data/models/Qwen3.8-27B-NVFP4
```

Use `HF_ENDPOINT=https://huggingface.co` for the original host.
Existing uv installations are reused. Bootstrap and Python mirrors can be
changed with `WININTEL_UV_INSTALL_URL` and `UV_PYTHON_INSTALL_MIRROR`.

## Serving

| Profile | Maximum context | Maximum sequences | KV cache | GPU memory budget |
| --- | ---: | ---: | --- | ---: |
| `standard` | 8,192 | 8 | FP16 | 40% |
| `throughput` | 8,192 | 8 | FP8 E4M3 | 80% |
| `long` | 262,144 | 8 | FP8 E4M3 | 80% |

```bash
WININTEL_PROFILE=throughput bash start.sh /path/to/model
```

Reproduce the official short-prompt MTP11 configuration:

```bash
sudo nvidia-smi -i 0,1,2,3,4,5,6,7 -ac 877,1530
SM70_SPECULATIVE_TOKENS=11 SM70_MAX_MODEL_LEN=16384 \
SM70_MAX_NUM_SEQS=16 SM70_MEMORY_UTILIZATION=0.80 \
SM70_KV_CACHE_DTYPE=auto WININTEL_SM70_TORCH_PREFILL=1 \
bash start.sh /path/to/model
```

For the measured high-concurrency improvement, use the same command with
`SM70_SPECULATIVE_TOKENS=5`. Use `9` for the fastest measured single-request
configuration of this task.

DFlash2 is available only for development and does not pass draft numerical
validation on this V100 configuration. Use MTP for deployment.
Experimental DFlash2 uses the matching checkpoint and seven speculative tokens.
The launcher selects synchronous scheduling and eager draft execution for
this adapter; the target model retains CUDA Graph.

```bash
HF_ENDPOINT=https://hf-mirror.com runtime/.venv/bin/hf download \
  incoai/Qwen3.8-27B-DFlash2 \
  --revision 015e795645c74b1a0eeef3b570031fb62e769bc5 \
  --local-dir /data/models/Qwen3.8-27B-DFlash2
WININTEL_ENABLE_EXPERIMENTAL_DFLASH2=1 SM70_SPECULATIVE_METHOD=dflash2 \
SM70_DRAFT_MODEL=/data/models/Qwen3.8-27B-DFlash2 \
SM70_SPECULATIVE_TOKENS=7 SM70_MAX_MODEL_LEN=16384 \
SM70_MAX_NUM_SEQS=16 SM70_MEMORY_UTILIZATION=0.80 \
SM70_KV_CACHE_DTYPE=auto WININTEL_SM70_TORCH_PREFILL=1 \
bash start.sh /path/to/Qwen3.8-27B-NVFP4
```

For a single full long-context request:

```bash
WININTEL_PROFILE=long SM70_MAX_NUM_SEQS=1 \
SM70_SPECULATIVE_TOKENS=11 bash start.sh /path/to/model
```

Reproduce the four-request 8K retrieval measurement with
`SM70_MAX_NUM_SEQS=4 SM70_MAX_BATCHED_TOKENS=6144` in the long profile.
Run the retrieval benchmark with `--concurrency 4 --rounds 1
--context-lengths 8192 --max-tokens 128`. Engine counters are available at
`/metrics`; subtract counters recorded before and after all four requests
finish to obtain the mean scheduler PREFILL phase time.

The maximum sequence count does not certify that all requests can use
the full context simultaneously. Actual capacity depends on the allocated
KV cache. Estimate TTFT from measured input throughput before increasing
input length or concurrency.

TP4:

```bash
SM70_TENSOR_PARALLEL_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
WININTEL_PROFILE=throughput bash start.sh /path/to/model
```

TP2 configuration, pending hardware validation:

```bash
SM70_TENSOR_PARALLEL_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 \
SM70_MAX_MODEL_LEN=4096 SM70_KV_CACHE_DTYPE=fp8_e4m3 \
bash start.sh /path/to/model
```

TP2 defaults to one sequence, a 92% memory budget, and 512-token prefill
chunks. TP4 defaults to an 80% memory budget.
Configuration overrides include `SM70_PORT`, `SM70_MAX_MODEL_LEN`,
`SM70_MAX_NUM_SEQS`, `SM70_MAX_BATCHED_TOKENS`, `SM70_KV_CACHE_DTYPE`,
`SM70_MEMORY_UTILIZATION`, `SM70_TENSOR_PARALLEL_SIZE`,
`SM70_SPECULATIVE_TOKENS`, `SM70_SPECULATIVE_METHOD`, `SM70_DRAFT_MODEL`,
and `SM70_SERVED_MODEL_NAME`.

## API

Default endpoint: `http://SERVER_IP:8037/v1`.
The TP8 model name is `winintel-qwen27-nvfp4-tp8`;
TP4 and TP2 names end with `tp4` and `tp2`.

```bash
curl http://127.0.0.1:8037/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "winintel-qwen27-nvfp4-tp8",
    "messages": [{"role": "user", "content": "Write a Python LRU cache class."}],
    "temperature": 0,
    "max_tokens": 512,
    "stream": true
  }'
```

## Reproduce benchmarks

Run from the installed `runtime` directory. Configure the server for at least
16 sequences, then test each concurrency level independently:

```bash
C=8
.venv/bin/python scripts/run_sm70_guarded.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --report "code-c${C}-gpu.json" -- \
  .venv/bin/python scripts/benchmark_sm70_http.py \
  --base-url http://127.0.0.1:8037/v1 \
  --model winintel-qwen27-nvfp4-tp8 --tokenizer /path/to/model \
  --workload english-code --coding-tasks lru \
  --concurrency "$C" --rounds 3 --first-token-timeout-seconds 180 \
  --max-tokens 2048 --report "code-c${C}.json"
```

Repeat with `C=1`, `2`, `4`, `8`, and `16`. Reports preserve actual inputs,
outputs, token counts, timing, validation, best rounds, and averages.
The guard records GPU memory, clock, and power without imposing
a power cap. The 180-second deadline applies to TTFT, not download time or
the entire benchmark.

For a full 8K English code input, add `--coding-context-tokens 8192`.
For the 256K service configuration, use `--coding-context-tokens 260096`
and reserve 2,048 tokens for output. For exact retrieval instead, use
`--workload context-retrieval --context-lengths 262016 --max-tokens 128`.
Keep concurrency at one until the relevant input-length configuration
has been measured.

## Community and attribution

We welcome reproducible improvements to V100 throughput, long-input prefill,
and model compatibility.

⭐ Star the repository to help more V100 owners find it.
🔔 Select **Watch → Custom → Releases** to receive new-version notifications.
🐛 [Report an issue](https://github.com/alexhmyang/winintel-vllm/issues) or
submit a pull request with your hardware, configuration, and measured results.
📺 Subscribe to our YouTube channel and follow Bilibili for demonstrations.

**数萌AI服务器 (winintel) · winintel AI Servers**
[winintel.com](https://winintel.com) · service@winintel.com

This is an adaptation of vLLM and existing open-source components.
AI assistance was used for development and documentation.
Licenses: Apache-2.0, with MIT components.
See [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) for attribution.
