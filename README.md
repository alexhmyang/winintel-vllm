# winintel-vllm

Run local AI on NVIDIA V100 servers for conversations, coding and text processing.

Author: 数萌AI服务器 (winintel) · [winintel AI](https://winintel.com)

[Download v0.1.0](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0) ·
[Complete installation guide](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0) ·
[Performance tables](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告) ·
[中文](README.zh-CN.md)

[Bilibili](https://space.bilibili.com/429959998) ·
[YouTube · winintel AI](https://www.youtube.com/@winintelcom)

## What does it do?

winintel-vllm adapts vLLM 0.17.0 and existing open-source components for NVIDIA Tesla V100.
Install it on your server, download a model and start the service.
AI clients on your computer can then send requests to that server.
The model generates answers using your server's GPUs.

Release v0.1.0 focuses on Qwen3.8-27B text inference:

- Conversations, code generation and text processing.
- Multiple simultaneous requests.
- An OpenAI-compatible chat API for clients such as WorkBuddy and DeepSeek Harness.
- Faster answer generation using MTP prediction and CUDA Graph to reduce repeated scheduling work.
- Single-request inputs approaching 256K tested on eight GPUs.

This release provides text input and output. Images, video and complete client automation workflows have not been validated.

## Which GPUs do I need?

The NVFP4 download is compressed, but this release expands model weights to FP16 at load time.
27B weights alone need about 50 GiB, with additional memory for caches and acceleration.

| GPU configuration | This release's 27B model | Recommendation |
| --- | --- | --- |
| 1 × V100 32GB | Insufficient memory | Use the separate 7B/8B path in the beginner guide |
| 2 × V100 16GB | Insufficient memory | Total memory is still only 32GB |
| 2 × V100 32GB | Configuration provided; hardware validation pending | Validate startup and answers before increasing input length |
| 4 × V100 32GB | Tested | Recommended starting configuration for this 27B model |
| 8 × V100 32GB | Tested | Suitable for concurrent requests and longer inputs |

Use an Ubuntu/Linux server, a compatible NVIDIA driver and CUDA Toolkit 12.8.
Windows computers can run clients and connect to the server.
The installation commands below run in the Ubuntu server terminal.

## Install and start

If Ubuntu, drivers or CUDA are not ready, follow the
[complete beginner guide](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0).
It covers setup, GPU counts, startup at boot and troubleshooting.

On a prepared server:

```bash
mkdir -p "$HOME/winintel"
cd "$HOME/winintel"
curl -fL --retry 3 \
  https://github.com/alexhmyang/winintel-vllm/releases/download/v0.1.0/winintel-vllm-0.1.0-qwen3.8-install.tar.gz \
  -o winintel-vllm-0.1.0-qwen3.8-install.tar.gz
echo '99bbbb519e1068efe034febde7a9a9af29f4fe683ee882c22598f4b480ec9d5e  winintel-vllm-0.1.0-qwen3.8-install.tar.gz' \
  | sha256sum -c -
tar -xzf winintel-vllm-0.1.0-qwen3.8-install.tar.gz
cd winintel-vllm-0.1.0-qwen3.8
MAX_JOBS=4 bash install.sh
```

The checksum should show `OK`. The installer creates a Python environment, installs dependencies and builds V100 extensions.
It defaults to domestic package mirrors. Download the full `install` archive; the smaller `source` archive is not a replacement.

Download the model using the domestic mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com runtime/.venv/bin/hf download \
  nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir "$HOME/models/Qwen3.8-27B-NVFP4"
```

Start with four V100 32GB:

```bash
SM70_TENSOR_PARALLEL_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
WININTEL_PROFILE=throughput \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

Or eight V100 32GB:

```bash
WININTEL_PROFILE=throughput \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

Wait for model loading and service startup. These commands use an 8,192-token context.
Keep the terminal open, or configure startup at boot using the complete guide.

## Connection details

Replace `192.168.1.100` with your server's reachable IP address.
Run `hostname -I` on the server if you need to find it.

| Setting | Value |
| --- | --- |
| Base URL | `http://192.168.1.100:8037/v1` |
| Full chat endpoint | `http://192.168.1.100:8037/v1/chat/completions` |
| Eight-GPU model ID | `winintel-qwen27-nvfp4-tp8` |
| Four-GPU model ID | `winintel-qwen27-nvfp4-tp4` |
| API key | Default startup does not enable authentication; use `winintel-local` if the client requires a value |

The placeholder is not a server password. If you configured an authentication gateway, use its real key.
Check model IDs at `http://192.168.1.100:8037/v1/models`.
When the client runs on another computer, `127.0.0.1` points to that computer, not your server.

Test on the server before configuring a client:

```bash
curl http://127.0.0.1:8037/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "winintel-qwen27-nvfp4-tp8",
    "messages": [{"role": "user", "content": "Introduce yourself briefly."}],
    "max_tokens": 256,
    "stream": true
  }'
```

Change the model ID for a four-GPU service.

## WorkBuddy

1. Install [WorkBuddy](https://www.codebuddy.cn/) and open Settings → Models.
2. Add a custom model and select Custom.
3. Fill in the following fields and save.
4. Select the new model in the conversation model selector.
5. Send a short message to check the connection.

| Field | Value |
| --- | --- |
| Display name | `winintel Local AI` |
| Model ID | `winintel-qwen27-nvfp4-tp8`, or `winintel-qwen27-nvfp4-tp4` for four GPUs |
| URL / endpoint | `http://192.168.1.100:8037/v1/chat/completions` |
| API key | `winintel-local` for the default unauthenticated service, otherwise your actual key |
| Custom protocol | Off; use the standard chat endpoint |
| Images | Off |
| Input budget / maximum output, if available | Start with `6000` input tokens and `2048` output tokens |
| Tool calling, if available | Default server startup does not enable tool parsing; start with text conversations |

WorkBuddy's default mode validates and completes the standard chat path.
If your version asks specifically for a Base URL, enter the address ending in `/v1`.
If it asks for an endpoint URL, enter the full chat address above.
The final request path should contain `/v1/chat/completions` once.

Generating text and code does not establish support for editing files or executing commands.
Those workflows require server-side tool calling; a client checkbox cannot enable it on the server.
Complete WorkBuddy task workflows have not been tested with this release.

[Official WorkBuddy model configuration](https://www.codebuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Model)

## DeepSeek Harness

DeepSeek Harness can connect to providers other than DeepSeek.
These settings connect it to Qwen on your server.

Install and start it using the [official instructions](https://github.com/deepseek-ai/deepseek-harness), then:

1. Open Settings → Models.
2. Add a provider and choose Custom Model API.
3. Enter these settings and add at least one model.
4. Save and select the model under the `winintel` provider.
5. Test with a short conversation.

| Field | Value |
| --- | --- |
| Provider ID | `winintel`, lowercase |
| Display name | `winintel Local AI` |
| Base URL | `http://192.168.1.100:8037/v1` |
| API protocol | `OpenAI Chat Completions` |
| API key | `winintel-local` for the default unauthenticated service, otherwise your actual key |
| Model ID | `winintel-qwen27-nvfp4-tp8`, or `winintel-qwen27-nvfp4-tp4` |
| Context window | `8192`, matching the startup commands |
| Maximum output tokens | `2048` |
| Input type | Text only |
| Reasoning levels | Do not declare extra levels for the initial connection |

Do not append `/chat/completions` to this Base URL.
Do not select Responses, Anthropic or Gemini protocols.
If model discovery fails, add the model ID manually.

For errors involving the `developer` role or output length parameters, open the configuration file from Settings.
Under the existing `llm-pi-ai` entry, add these fields inside `providers.winintel`:

```yaml
compat:
  supportsDeveloperRole: false
  maxTokensField: max_tokens
```

This is a fragment inside the provider, not a replacement for your full configuration.
The default web profile uses `$DSH_HOME/profiles/web/cordis.patch.yml`.
Model changes normally take effect on the next request.

Harness automation also requires tool calling. The default release launcher does not enable tool parsing.
Complete Harness task workflows have not been tested; a working conversation does not validate file operations.

[Official Harness provider configuration](https://deepseek-harness.github.io/deepseek-harness/guide/providers)

## Performance

Eight V100 32GB GPUs, official Qwen3.8-27B-NVFP4 checkpoint, short English Python coding prompts:

| Simultaneous requests | MTP11 mean aggregate decode | MTP9 mean aggregate decode | MTP5 mean aggregate decode |
| ---: | ---: | ---: | ---: |
| 1 | 93.80 tokens/s | 94.03 tokens/s | 86.78 tokens/s |
| 8 | 516.89 tokens/s | 547.06 tokens/s | 592.26 tokens/s |
| 16 | 802.10 tokens/s | 831.58 tokens/s | 879.43 tokens/s |

Each setting ran twice. All 150 generated programs passed validation.
Best MTP5 rounds reached 592.92 tokens/s at eight requests and 880.95 tokens/s at sixteen.
Aggregate speed is the sum across requests, not the speed each user receives.
Tokens are text processing units, not a fixed number of characters.

Long inputs take additional time before the first answer:

| Task | Actual input tokens | Time to first answer | Decode speed | Validation |
| --- | ---: | ---: | ---: | --- |
| English code | 8,192 | 2.21 s | 82.37 tokens/s | Passed |
| English code | 260,096 | 160.66 s | 10.83 tokens/s | Passed |
| Exact clause retrieval | 32,768 | 9.68 s | 91.76 tokens/s | 3/3 |
| Exact clause retrieval | 262,016 | 163.64 s | 18.73 tokens/s | 3/3 |

One request, one measured round per long-input entry. Concurrent full 256K inputs remain unvalidated.
Start with 8K context and increase input length only after checking behavior and capacity.

[Complete results, GPU memory and validation tables](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)
are displayed directly on the page; no CSV download or JSON report reading is required.

## Troubleshooting

| Problem | Check |
| --- | --- |
| Connection refused | Server startup, reachable IP and port; access `/v1/models` from the client computer |
| Timeout | Network connectivity and firewall access to port `8037` |
| 404 | Missing `/v1` or a duplicated `/chat/completions` path |
| Model not found | Exact ID from `/v1/models`; TP4 and TP8 names differ |
| 401 | Actual gateway key if authentication is configured |
| Input too long | Clear conversation history or reduce input; client settings cannot expand server capacity |
| Conversation works, automation fails | Server-side tool calling; default startup does not enable tool parsing |
| Out of GPU memory | GPU count and memory, other GPU processes, context length and request count |

## Community and attribution

[Star the project](https://github.com/alexhmyang/winintel-vllm) ·
[Report an issue](https://github.com/alexhmyang/winintel-vllm/issues) ·
[Watch releases](https://github.com/alexhmyang/winintel-vllm/subscription)

数萌AI服务器 (winintel) · https://winintel.com · service@winintel.com

This project adapts vLLM and existing open-source components, with AI assistance for development and documentation.
Licenses: Apache-2.0, with MIT components. See [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
