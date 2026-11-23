# winintel-vllm V100 beginner deployment guide

Author: **数萌AI服务器 (winintel)** · **winintel AI**  
Qwen v0.1.0 · Documentation update: 2026-11-23

The [Chinese step-by-step guide](v100-beginner-guide.zh-CN.md)
covers Ubuntu, the V100 driver, CUDA, installation, model downloads,
single/dual/four/eight GPU deployment, API checks, systemd, and troubleshooting.
It is also displayed directly on the
[release page](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0).

| V100 32GB GPUs | Beginner choice | Qwen3.8-27B status | Other capacity candidates |
| ---: | --- | --- | --- |
| 1 | Qwen2.5-7B-Instruct / Qwen3-8B, FP16 | Does not fit this runtime | 14B FP16 is tight and untested |
| 2 × 16GB | Smaller models; TP2 requires separate validation | Does not fit this runtime: only 32GB total | Different from two 32GB GPUs |
| 2 | Small models; conservative 27B trial | TP2 configuration provided, GPU validation pending | 32B FP16 leaves too little space for a beginner setup |
| 4 | Qwen3.8-27B TP4 | TP4 measured | 32B FP16, capacity estimate only |
| 8 | Qwen3.8-27B TP8 | TP8 measured | 70B/72B FP16, capacity estimate only |

This runtime expands NVFP4 weights to FP16 during loading. A 27B model
needs approximately 50 GiB for weights before KV cache, CUDA Graph,
speculation and temporary buffers. Download size is not runtime memory usage.
Other model sizes are capacity estimates, not validated compatibility claims.
The single-GPU example has not undergone a new generation test.

Use native Ubuntu 22.04/24.04 x86_64, the proprietary R580-server driver,
and CUDA Toolkit 12.8. CUDA 13 cannot compile V100 code.
Install the driver/toolkit using the existing
[winintel-server-doctor v0.2.0](https://github.com/alexhmyang/winintel-server-doctor/releases/tag/v0.2.0),
reboot, then check both `nvidia-smi` and `nvcc --version`.
The CUDA version displayed by `nvidia-smi` is the driver's maximum supported
version, not the installed toolkit version.

Download the `winintel-vllm-0.1.0-qwen3.8-install.tar.gz` asset and verify SHA256:
`99bbbb519e1068efe034febde7a9a9af29f4fe683ee882c22598f4b480ec9d5e`.
Install with `bash install.sh` without a model argument, then download weights
using `runtime/.venv/bin/hf` and `HF_ENDPOINT=https://hf-mirror.com`.
The guide reuses the driver's uv executable and domestic package mirrors.

Use the generic CLI for the single-GPU small model. The 27B `start.sh`
accepts TP2/TP4/TP8. All documented launches use `WININTEL_SM70`
attention and explicit CUDA Graph. Small-model launches do not use 27B MTP.
API model names are `winintel-small` or `winintel-qwen27-nvfp4-tp2/tp4/tp8`.
The OpenAI-compatible base URL is `http://SERVER_IP:8037/v1`.

Published TP8 short-input coding results reached 592.92 aggregate decode
tokens/s at eight requests and 880.95 at sixteen, on eight V100-SXM2 32GB
GPUs. These results do not describe
single-user throughput or other GPU counts. TP4 results use a historical
checkpoint. See the
[performance tables on the release page](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)
for exact conditions.

This documentation update checks published artifacts, model configuration,
command consistency and Bash syntax. It adds no new clean-machine installation,
single-GPU or dual-GPU inference measurements. AI assistance was used.
Source and component licensing remain documented in
[NOTICE](https://github.com/alexhmyang/winintel-vllm/blob/main/NOTICE)
and [THIRD_PARTY.md](https://github.com/alexhmyang/winintel-vllm/blob/main/THIRD_PARTY.md).

[winintel.com](https://winintel.com) · service@winintel.com

## Connect a client

After checking the API, replace `192.168.1.100` with your server's reachable IP.

| Client | Setup | Endpoint | Protocol |
| --- | --- | --- | --- |
| WorkBuddy | Settings → Models → Custom | URL: `http://192.168.1.100:8037/v1/chat/completions`; a Base URL field takes the address ending in `/v1` | OpenAI-compatible chat |
| DeepSeek Harness | Settings → Models → Custom Model API | Base URL: `http://192.168.1.100:8037/v1` | OpenAI Chat Completions |

Use Provider ID `winintel` in Harness. Model IDs are `winintel-qwen27-nvfp4-tp8`
for eight GPUs and `winintel-qwen27-nvfp4-tp4` for four GPUs.
For other configurations, copy the exact ID from `/v1/models`.
Default startup does not enable authentication; use `winintel-local` if a client requires a key,
or your real gateway key when authentication is configured.

Save, select the model and test a short conversation.
For the default 8K service, use an 8,192-token context and 2,048-token output limit.
Input and output together must fit the server context; reduce client limits for smaller server configurations.
Disable image input. Default startup does not enable tool parsing, and complete automation workflows remain untested.

See the [README API examples](https://github.com/alexhmyang/winintel-vllm/blob/main/README.md#api)
for configuration files and troubleshooting.
