<div align="center">

# winintel-vllm

### 🚀 数萌AI · V100 大模型推理

**💎 让经典算力，继续创造价值。**

**⚡ 880.9 token/s 十六路总吞吐 · 🔥 592.9 token/s 八路总吞吐 · 🧠 单路 256K 长输入已测**

[🌐 官方网站](https://winintel.com) · [📊 实测性能](#实测性能) · [🤖 适用模型](#适用模型) · [🛠️ 一键安装](#一键安装)

[🎬 Bilibili · 数萌AI视频主页](https://space.bilibili.com/429959998) ·
[▶️ YouTube · winintel AI](https://www.youtube.com/@winintelcom)

[⭐ 点亮 Star](https://github.com/alexhmyang/winintel-vllm) ·
[🔔 订阅更新](https://github.com/alexhmyang/winintel-vllm/subscription) ·
[📦 下载 v0.1.0](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0)

**v0.1.0 · Qwen3.8-27B · Volta / SM70**

</div>

---

🚀 **让现有 V100 服务器，成为实用的 AI 生产力。**
winintel-vllm 是数萌AI（**winintel AI**）面向 NVIDIA Tesla V100 的
Qwen3.8-27B 推理方案：十六路代码生成实测总解码吞吐最高
**880.95 token/s**，并已完成**单路完整 256K 长输入**测试。
基于 vLLM 0.17.0 与成熟开源组件，兼顾多用户服务、长文档处理和便捷部署，
适用于代码助手、私有知识库和企业文本应用。

- ⚡ 多人同时使用：8 张 V100 32GB，八路代码生成最佳总速度 **592.92 token/s**，十六路最佳 **880.95 token/s**。这是所有请求合计的速度。
- 🧠 处理长文档：一次输入与回答合计最多可配置 **262,144 token**，已测试单次长输入的代码生成与合同原文查找。
- 🏎️ 加快回答速度：使用 MTP 提前预测后续内容，再由主模型校验；使用 CUDA Graph 减少显卡重复调度的开销，配合为 V100 适配的 `WININTEL_SM70` 注意力计算后端。
- 💾 加载量化模型：支持 NVFP4/FP8 权重文件，加载后在 V100 上按 FP16 计算；对话计算缓存可使用 FP8，减少缓存占用的显存。
- 🔌 连接 AI 客户端：提供 OpenAI 兼容 API，可连接 WorkBuddy、DeepSeek Harness 等客户端；回答可以边生成边显示。
- 🛠️ 安装与启动：安装程序自动配置 Python 环境并编译 V100 所需的扩展；按显卡数量选择启动配置。
- ✅ 查看实测结果：提供测速脚本和网页结果表格，MTP 配置比较中 **150/150 份代码通过验证**。
- 🤝 参与项目：欢迎分享部署经验，帮助改进 V100 推理速度和模型兼容性。

## 核心能力

| 能力 | 配置 |
| --- | --- |
| V100 多卡推理 | TP2 / TP4 / TP8 配置；TP4、TP8 有实测记录 |
| 多请求服务 | 并发数可配置，支持独立请求与流式输出 |
| 解码加速 | 模型内 MTP3/5/7/9/11，主模型与草稿 CUDA Graph |
| 注意力后端 | `WININTEL_SM70` |
| 检查点加载 | 官方 ModelOpt FP8/NVFP4 混合格式、compressed-tensors NVFP4；加载后使用 FP16 权重与计算 |
| KV 缓存 | FP16 / FP8 E4M3 |
| 服务上下文 | 配置上限 262,144 token |
| 接入方式 | OpenAI 兼容 API，支持 SSE 流式响应 |

DFlash2 适配器仅供开发实验，当前未通过这套 V100 配置的草稿数值验证，
尚无已验证的加速收益。
已公布的性能成绩使用 MTP；生产部署推荐 MTP5/MTP9。

TP2 / TP4 / TP8 表示两张 / 四张 / 八张显卡共同运行一个模型。
上下文包括问题、历史对话和回答；KV 缓存用于保存对话计算的中间结果。
FP16、FP8 和 NVFP4 是数值格式，下载文件较小不代表运行时显存占用同样小。

## 实测性能

并发表示同时处理的请求数量；总吞吐表示这些请求合计的生成速度，不是每个人的速度。
token 是模型处理文字的单位，不能固定换算成汉字数。
首字等待是从发送请求到收到第一段回答的时间；Prefill 是模型处理输入资料的阶段，Decode 是生成回答的阶段。

### 官方检查点：英文代码生成

**硬件：**V100-SXM2 32GB × 8，TP8。  
**模型：**重新下载的 NVIDIA 官方 `Qwen3.8-27B-NVFP4`，
固定版本 `482ca0f3832238542f8f5295dde86b5f22711d80`，全部权重分片 SHA256 已核验。  
**配置：**FP16 权重与计算、FP16 KV、MTP11、主模型与草稿 CUDA Graph，
上下文 16,384 token，最大并发 16。

**单路已测最佳：93.56 token/s**，LRU 代码全部通过检查，
[网页测试报告](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)。各并发档位实测如下：

| 并发 | 轮数 | 最佳总解码吞吐 | 平均总解码吞吐 | 最大首字等待 | 代码检查 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | **93.26 token/s** | **93.06 token/s** | 188 ms | 3/3 |
| 2 | 3 | **191.64 token/s** | **191.38 token/s** | 460 ms | 6/6 |
| 4 | 3 | **332.96 token/s** | **332.76 token/s** | 410 ms | 12/12 |
| 8 | 3 | **518.46 token/s** | **518.19 token/s** | 455 ms | 24/24 |
| 16 | 2 | **802.81 token/s** | **802.10 token/s** | 599 ms | 32/32 |

1–8 路各三轮，16 路两轮，77 份输出全部通过语法、LRU 行为检查和至少八项模型生成的单元测试。
请求通过屏障同时发起，使用独立 HTTP 流式会话，正式计时前用四个输出 token 预热。
英文 LRU 提示词 100 token，
输出 1,677–1,746 token；贪心解码、关闭思考、正常 EOS、输出上限 2,048 token。
解码速度按客户端接收时间计算并排除首个流式块；16 路最佳轮次包含首字等待的
总输出吞吐为 **799.07 token/s**。

这些是短输入代码任务成绩；长上下文需按输入长度单独测试。

[查看逐轮结果、显存与模型校验表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)。

同次测试另测 CSV 解析器和 DAG 拓扑排序，单路分别为 **93.29、78.56 token/s**，
两份代码全部通过行为及单元测试检查。
[三类英文代码任务结果表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)。

### 优化配置比较

本轮补充 MTP11 复测（每档两轮）：
单路最佳 **93.83 token/s**，8 路最佳总吞吐 **517.92 token/s**，
16 路最佳总吞吐 **802.81 token/s**、平均 **802.10 token/s**。
50 份代码全部通过检查，
单卡最高显存占用 **29,723 MiB**。
[复测与优化配置比较表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

在相同提示词和服务配置下，比较 MTP 草稿长度：

| 并发 | MTP11 平均总解码吞吐 | MTP9 平均总解码吞吐 | MTP5 平均总解码吞吐 | MTP5 对比 MTP11 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 93.80 token/s | **94.03 token/s** | 86.78 token/s | −7.5% |
| 8 | 516.89 token/s | 547.06 token/s | **592.26 token/s** | +14.6% |
| 16 | 802.10 token/s | 831.58 token/s | **879.43 token/s** | +9.6% |

每档两轮，三种配置共 **150 份代码全部通过验证**。
MTP5 八路最佳 **592.92 token/s**，十六路最佳 **880.95 token/s**。
本任务单路推荐 MTP9，高并发推荐 MTP5；复现上述测速命令时，
将 `SM70_SPECULATIVE_TOKENS` 分别设为 `9` 或 `5`。
[MTP9 / MTP5 逐轮结果表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

### 长上下文：英文代码生成

官方检查点，TP8、FP8 KV、MTP11、CUTLASS 预填充、主模型与草稿
CUDA Graph，服务上下文 262,144 token，单请求。
沿用英文 LRU 实现及至少八项模型生成的单元测试：

| 实际输入 token | 首字等待 | 输入吞吐（输入 / 首字等待） | 解码速度 | 代码检查 |
| ---: | ---: | ---: | ---: | --- |
| 8,192 | **2.21 秒** | **3,704.29 token/s** | **82.37 token/s** | 通过 |
| 260,096（256K 配置，预留 2,048 输出） | **160.66 秒** | **1,618.89 token/s** | **10.83 token/s** | 通过 |

输入包含重复英文参考代码，每档一轮，关闭思考、贪心解码、正常 EOS，
输出上限 2,048 token。输入吞吐包含 HTTP 传输、预填充及首个生成块。
实际生成分别为 1,659、1,668 token。256K 英文代码轮次
单卡最高显存 **31,201 MiB**。
这些成绩对应完整长输入的英文代码任务。
[8K / 256K 长上下文与显存结果表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

### 追加长输入配置实测

MTP5、FP8 KV、262,144 上下文、最大四路、预填充批量 12,288 token，
CUTLASS 与 CUDA Graph：

| 任务 | 每路实际输入 token | 并发 | 最大首字等待 | 客户端总解码吞吐 | 验证 |
| --- | ---: | ---: | ---: | ---: | --- |
| 英文代码 | 8,192 | 4 | **8.56 秒** | **191.72 token/s** | 4/4 份代码 |
| 合同原文检索 | 32,768 | 1 | 9.48 秒 | 74.51 token/s | 3/3 条款 |
| 合同原文检索 | 262,016 | 1 | 171.77 秒 | 18.95 token/s | 3/3 条款 |

两档单路检索无预热，其引擎 PREFILL 阶段实测分别为：

| 输入 token | 引擎 PREFILL 耗时 | 输入 / 引擎 PREFILL 耗时 |
| ---: | ---: | ---: |
| 32,768 | **9.37 秒** | **3,498.94 token/s** |
| 262,016 | **171.25 秒** | **1,530.03 token/s** |

这是 vLLM 调度器记录的预填充阶段耗时，包含阶段内调度开销。
每项一轮。扩大预填充批量未改善 256K 首字等待，
原 MTP11 / 6,144 配置为 **163.64 秒**，本轮为 **171.77 秒**；
单路长输入优先保留原配置。四路完整 256K 尚未实测，预估首字时间超过 180 秒。

[四路 8K、32K / 256K 与引擎指标表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

补充四路独立 8K 原文检索会话，MTP11、预填充批量 6,144、FP8 KV、
服务上下文 262,144 token：

| 每路输入 token | 并发 | 平均每路引擎 PREFILL | 每路输入 / 平均 PREFILL | 最大首字等待 | 客户端总解码吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4 | **3.98 秒** | **2,059.69 token/s** | **9.24 秒** | **39.48 token/s** |

四路全部准确引用三条原文，无预热。引擎指标覆盖四个已完成请求，
PREFILL 阶段耗时之和为 15.91 秒；阶段时间可能重叠，因此每路速率
不能当成 GPU 总吞吐，首字等待还包含排队。每路仅输出 73 token，
不能直接与长代码输出成绩比较。

[四路检索与预填充指标表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

### 历史检查点：八路代码生成 373.4 token/s

**硬件：**8 张 V100-SXM2 32GB，TP8，GPU 核心时钟 1530 MHz。  
**配置：**`Qwen3.8-27B-NVFP4` 测试检查点，FP16 计算，FP8 E4M3 KV，MTP7，
主模型与草稿 CUDA Graph，服务上下文配置 262,144 token。

| 轮次 | 并发 | 总解码吞吐 | 包含首字等待的总输出吞吐 | 最大首字等待 | 代码检查 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一轮 | 8 | **373.97 token/s** | 371.47 token/s | 254 ms | 8/8 |
| 第二轮 | 8 | **372.83 token/s** | 371.04 token/s | 468 ms | 8/8 |
| 两轮均值 | 8 | **373.40 token/s** | **371.25 token/s** | — | **16/16** |

八个独立 HTTP 流式请求通过屏障同时发起，完成 LRU 缓存、CSV 解析、
DAG 拓扑排序三类英文代码任务。贪心解码，正常 EOS，输出上限 2048 token。
代码检查包含语法、行为检查和模型生成的单元测试。
总解码吞吐按客户端实际接收时间计算，排除首个流式块。
这里的吞吐对应短输入代码任务；262,144 是该服务的上下文配置。

### 历史检查点：单路与三路代码生成

同一硬件，TP8、MTP7、CUDA Graph，FP16 KV。每项重复两轮：

| 任务 | 并发 | 解码吞吐 |
| --- | ---: | ---: |
| LRU 缓存实现 | 1 | **129.90 token/s** |
| CSV 解析器实现 | 1 | **122.54 token/s** |
| DAG 拓扑排序实现 | 1 | **111.15 token/s** |
| 三类任务同时运行 | 3 | **288.54 token/s 总吞吐** |

### TP4 实测

4 张 V100-SXM2 32GB，TP4、MTP5、FP16 KV、CUDA Graph，
上下文 8,192 token，最大并发 3，显存预算 65%。每项两轮：

| 任务 | 并发 | 解码吞吐 |
| --- | ---: | ---: |
| LRU 缓存实现 | 1 | **97.77 token/s** |
| CSV 解析器实现 | 1 | **92.13 token/s** |
| DAG 拓扑排序实现 | 1 | **83.41 token/s** |
| 三类任务同时运行 | 3 | **220.54 token/s 总吞吐** |

12 份代码输出全部通过行为检查。
[测试结果表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)。

### 验证记录

| 项目 | 已完成验证 |
| --- | --- |
| 八路代码生成 | 16 份输出全部通过代码检查 |
| FP8 KV 与预填充 | 33 项算子测试通过 |
| NVFP4 权重解码 | 496 个量化矩阵首尾行与参考逐值一致 |
| CUDA Graph / eager 对照 | 1,024 个输出 token 一致 |
| 8K 输入原文检索 | 三段合同原文逐字通过，首字等待 2.353 秒 |

性能数据对应随包记录的实际测试检查点；权重文件标识见
[模型校验表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)。
下表的官方模型链接提供模型选择和配置参考。

[八路、单路与三路测试结果表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告) ·
[算子测试](docs/winintel/fp8-kv-prefill-tests-v100.txt)

## 适用模型

本版本面向 **Qwen3.8-27B 文本推理**，提供 NVFP4 检查点加载、
多卡服务与流式 API。

| 项目 | 模型与格式 |
| --- | --- |
| 目标模型系列 | Qwen3.8-27B |
| Qwen 官方基础模型原始链接 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| NVIDIA 官方 NVFP4 模型原始链接 | [Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |
| 国内基础模型镜像 | [HF Mirror · Qwen3.8-27B](https://hf-mirror.com/Qwen/Qwen3.8-27B) |
| 国内 NVFP4 模型镜像 | [HF Mirror · Qwen3.8-27B-NVFP4](https://hf-mirror.com/nvidia/Qwen3.8-27B-NVFP4) |
| 官方下载检查点格式 | ModelOpt MIXED_PRECISION：FP8 注意力投影 + NVFP4 MLP / 输出层，16 元素分组 |
| 历史测试检查点格式 | compressed-tensors NVFP4，FP4 E2M1，16 元素分组 |
| V100 执行精度 | NVFP4 权重加载时展开为 FP16；激活 FP16 |
| 默认推测解码 | 模型内 MTP，7 个草稿 token |
| 模型输入 | 文本 |

使用本地完整模型目录，包含 `config.json`、tokenizer 与权重分片。
安装程序会检查架构及量化格式；官方模型的配置可按上表链接查阅。

### 国内模型下载

在安装后的 `runtime` 目录执行，完整下载 tokenizer、配置和全部权重：

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/hf download \
  nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir /data/models/Qwen3.8-27B-NVFP4
```

镜像对应同一官方仓库，使用原始模型 ID。可用 `--revision <commit>` 固定版本。
模型下载由用户单独发起，耗时取决于网络带宽和模型大小。

## 一键安装

[从 Ubuntu 安装、驱动和 CUDA 配置到开机自启的完整教程](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0)。

**显存要求：**本版本将 27B 权重展开为 FP16，仅权重约需 50 GiB。
单张 V100 32GB 或两张 V100 16GB 无法部署此 27B 模型；建议选择 7B/8B。
四张、八张 V100 32GB 已实测；两张 V100 32GB 配置尚待硬件验证。

**推荐环境：**Linux x86_64、V100 32GB × 8、CUDA Toolkit 12.8、
NVIDIA 驱动、C/C++ 编译工具、Git、curl。安装包内置官方 vLLM 0.17.0
基础源码，自动创建 Python 3.12 环境、安装依赖并构建 SM70 扩展。

下载 `winintel-vllm-0.1.0-qwen3.8-install.tar.gz`，解压后执行：

```bash
bash install.sh /path/to/Qwen3.8-27B-NVFP4
```

安装完成后自动启动服务。模型使用本地目录；依赖使用 uv 安装，
Python 包默认使用清华镜像，PyTorch CUDA 12.8 包默认使用南京大学镜像。

| 安装资源 | 默认地址 |
| --- | --- |
| Python 依赖 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| PyTorch / torchvision / torchaudio | `https://mirrors.nju.edu.cn/pytorch/whl/cu128` |
| 模型镜像 | `https://hf-mirror.com` |
| uv 引导与 Python 3.12 | uv 官方渠道；可通过环境变量指定内部镜像 |

替换 Python 包源、PyTorch 包源：

```bash
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
WININTEL_TORCH_INDEX_URL=https://mirrors.nju.edu.cn/pytorch/whl/cu128 \
bash install.sh /path/to/model
```

使用官方源：

```bash
UV_INDEX_URL=https://pypi.org/simple \
WININTEL_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
HF_ENDPOINT=https://huggingface.co bash install.sh /path/to/model
```

已有 uv 时直接复用。引导脚本地址可设置 `WININTEL_UV_INSTALL_URL`；
Python 构建包镜像可设置 `UV_PYTHON_INSTALL_MIRROR`。

仅安装：

```bash
bash install.sh
```

后续启动：

```bash
bash start.sh /path/to/Qwen3.8-27B-NVFP4
```

### 启动配置

| 配置 | 上下文 | 最大并发 | KV | 显存预算 |
| --- | ---: | ---: | --- | ---: |
| `standard` 默认 | 8,192 | 8 | FP16 | 40% |
| `throughput` | 8,192 | 8 | FP8 E4M3 | 80% |
| `long` | 262,144 | 8 | FP8 E4M3 | 80% |

```bash
WININTEL_PROFILE=throughput bash start.sh /path/to/model
WININTEL_PROFILE=long bash start.sh /path/to/model
```

环境变量支持 `SM70_PORT`、`SM70_MAX_MODEL_LEN`、`SM70_MAX_NUM_SEQS`、
`SM70_KV_CACHE_DTYPE`、`SM70_MEMORY_UTILIZATION`、
`SM70_TENSOR_PARALLEL_SIZE`、`SM70_SERVED_MODEL_NAME`、
`SM70_SPECULATIVE_TOKENS`、`SM70_MAX_BATCHED_TOKENS` 和 `MAX_JOBS`。
上表为 TP8 默认值；TP4 默认显存预算 80%，TP2 为 92%。

### TP4：四卡运行

```bash
SM70_TENSOR_PARALLEL_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
WININTEL_PROFILE=throughput bash start.sh /path/to/model
```

复现上面的 TP4 历史测试配置：

```bash
SM70_TENSOR_PARALLEL_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
SM70_SPECULATIVE_TOKENS=5 SM70_MAX_NUM_SEQS=3 \
SM70_MEMORY_UTILIZATION=0.65 SM70_KV_CACHE_DTYPE=auto \
bash start.sh /path/to/model
```

API 模型名称：`winintel-qwen27-nvfp4-tp4`。

### TP2：双卡运行配置

```bash
SM70_TENSOR_PARALLEL_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 \
SM70_MAX_MODEL_LEN=4096 SM70_KV_CACHE_DTYPE=fp8_e4m3 \
bash start.sh /path/to/model
```

TP2 使用 V100 32GB × 2，默认最大并发 1、显存预算 92%、
预填充分块 512 token。权重加载后为 FP16，需为 KV、MTP 和 CUDA Graph
预留显存。此配置待 GPU 实测，实际可用上下文以启动时 KV 容量为准。
API 模型名称：`winintel-qwen27-nvfp4-tp2`。

### 16 并发服务

```bash
WININTEL_PROFILE=throughput SM70_MAX_NUM_SEQS=16 \
bash start.sh /path/to/model
```

启动程序按并发数自动生成 CUDA Graph 捕获批次。

复现官方检查点的 MTP11 测速配置：

```bash
sudo nvidia-smi -i 0,1,2,3,4,5,6,7 -ac 877,1530
SM70_SPECULATIVE_TOKENS=11 SM70_MAX_MODEL_LEN=16384 \
SM70_MAX_NUM_SEQS=16 SM70_MEMORY_UTILIZATION=0.80 \
SM70_KV_CACHE_DTYPE=auto WININTEL_SM70_TORCH_PREFILL=1 \
bash start.sh /path/to/model
```

使用下方英文 LRU 测速命令，分别设置 `C=1`、`2`、`4`、`8`、`16`。

### 长上下文运行

TP8 配置 256K 上下文，先以单路请求验证：

```bash
WININTEL_PROFILE=long SM70_MAX_NUM_SEQS=1 SM70_SPECULATIVE_TOKENS=11 \
bash start.sh /path/to/model
```

TP4 配置 32K 上下文：

```bash
SM70_TENSOR_PARALLEL_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
WININTEL_PROFILE=throughput SM70_MAX_MODEL_LEN=32768 \
SM70_MAX_NUM_SEQS=1 bash start.sh /path/to/model
```

官方检查点，TP8、FP8 KV、MTP11、CUTLASS 预填充、CUDA Graph，
单请求：

| 实际输入 token | 首字等待 | 输入吞吐（输入 / 首字等待） | 生成速度 | 原文检索 |
| ---: | ---: | ---: | ---: | --- |
| 8,192 | **2.48 秒** | **3,302.89 token/s** | **98.80 token/s** | 3/3 |
| 32,768 | **9.68 秒** | **3,385.23 token/s** | **91.76 token/s** | 3/3 |
| 262,016（接近 256K） | **163.64 秒** | **1,601.17 token/s** | **18.73 token/s** | 3/3 |

每项实测一轮，正文 5%、50%、95% 位置放入三段合同条款，
背景为重复的合同干扰资料；正常 EOS，实际生成 73 token，输出上限 128 token。
输入吞吐包含 HTTP 传输、预填充和首个生成块的时间。
256K 轮次单卡最高显存 **32,431 MiB**。

[8K / 32K / 256K 检索、显存与配置表](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

`long` 配置默认启用已有的 PyTorch CUTLASS 预填充路径，
配合 `WININTEL_SM70` 注意力后端与主模型、草稿 CUDA Graph。
上下文长度包含输入与输出；256K 为服务配置上限。
长输入吞吐与首字等待需按实际输入长度单独测量，
可用容量由启动时的 KV 缓存预算决定。

## API 接入

服务地址：`http://服务器IP:8037/v1`  
模型名称：`winintel-qwen27-nvfp4-tp8`

将“服务器IP”替换成客户端能访问的服务器地址，例如 `192.168.1.100`。
四卡服务的模型名称为 `winintel-qwen27-nvfp4-tp4`；
也可以访问 `http://服务器IP:8037/v1/models` 核对模型名称。
客户端和服务器在不同电脑上时，不能把服务器地址填成 `127.0.0.1`。

```bash
curl http://127.0.0.1:8037/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "winintel-qwen27-nvfp4-tp8",
    "messages": [{"role": "user", "content": "写一个 Python LRU 缓存类"}],
    "temperature": 0,
    "max_tokens": 512,
    "stream": true
  }'
```

### WorkBuddy：配置本地模型 API

打开 WorkBuddy 的“设置 → 模型”，添加“自定义 / Custom”模型，按下表填写。
保存后，在对话页面选择新模型，发送一句“你好，请介绍一下自己”测试。

| 设置项 | 八卡服务示例 |
| --- | --- |
| 显示名称 | `winintel 本地 AI` |
| 模型名称 / Model ID | `winintel-qwen27-nvfp4-tp8`；四卡改为 `winintel-qwen27-nvfp4-tp4` |
| URL / 接口地址 | `http://192.168.1.100:8037/v1/chat/completions`，替换 IP |
| API Key | 默认未鉴权服务可填 `winintel-local`；有鉴权网关时填真实密钥 |
| 自定义协议 | 关闭，使用标准聊天接口 |
| 输入预算 / 最大输出（如有） | 先设 `6000` / `2048`，适用于默认 8K 上下文 |
| 图片输入 / 工具调用（如有） | 先关闭；当前版本只提供文字推理，默认脚本未开启工具调用解析 |

如果你的版本只提供“Base URL”字段，填 `http://192.168.1.100:8037/v1`。
如果提供“URL / 接口地址”字段，使用上表的完整聊天地址。
最终请求路径应为 `/v1/chat/completions`，只出现一次。

使用配置文件的用户，可参考下面的模型条目。
旧版 `~/.codebuddy/models.json` 仍受支持；在已有 `models` 数组中加入条目，
不要覆盖其他模型配置。新手优先使用上面的界面方式。

```json
{
  "id": "winintel-qwen27-nvfp4-tp8",
  "name": "winintel 本地 AI",
  "url": "http://192.168.1.100:8037/v1/chat/completions",
  "apiKey": "winintel-local",
  "maxInputTokens": 6000,
  "maxOutputTokens": 2048,
  "supportsToolCall": false,
  "supportsImages": false
}
```

这里的 `winintel-local` 只是未鉴权服务的占位值，不是服务器密码。
[WorkBuddy 官方配置说明](https://www.codebuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Model)。

### DeepSeek Harness：配置本地模型 API

DeepSeek Harness 也可以连接自己部署的 Qwen 模型，不必使用 DeepSeek 云端 API。
按[官方说明](https://github.com/deepseek-ai/deepseek-harness)安装并启动后，
打开“设置 → 模型”，添加提供商，选择“自定义模型 API”。

| 设置项 | 八卡服务示例 |
| --- | --- |
| Provider ID | `winintel`，小写 |
| 显示名称 | `winintel 本地 AI` |
| 基础 URL / Base URL | `http://192.168.1.100:8037/v1`，替换 IP，不加 `/chat/completions` |
| API 协议 | `OpenAI Chat Completions` |
| API 密钥 | 默认未鉴权服务可填 `winintel-local`；有鉴权网关时填真实密钥 |
| 模型 ID | `winintel-qwen27-nvfp4-tp8`；四卡改为 `winintel-qwen27-nvfp4-tp4` |
| 上下文窗口 / 最大输出 | `8192` / `2048`，与默认服务上下文一致 |
| 输入类型 | 文本 |
| 推理等级 | 初次连接不声明额外等级，使用服务默认行为 |

保存后，在对话页面选择 `winintel` 下的模型并测试简短对话。
如果模型探测失败，可以手动填写模型 ID。
不要选择 OpenAI Responses、Anthropic Messages 或 Gemini 协议。

需要手动配置时，先在设置页点击“打开配置文件”。
默认网页配置位于 `$DSH_HOME/profiles/web/cordis.patch.yml`，
以下是 `llm-pi-ai` 条目的示例；已有该条目时只合并 `winintel` 提供商，保留其他配置。

```yaml
- id: llm-pi-ai
  config:
    providers:
      winintel:
        apiKeyEnv: WININTEL_API_KEY
        api: openai-completions
        baseURL: http://192.168.1.100:8037/v1
        compat:
          supportsDeveloperRole: false
          maxTokensField: max_tokens
        models:
          - id: winintel-qwen27-nvfp4-tp8
            name: winintel 本地 AI
            contextWindow: 8192
            maxTokens: 2048
            input: [text]
```

`apiKeyEnv` 表示从启动 Harness 的环境中读取密钥。
默认未鉴权服务，可在启动 Harness 前设置：

```powershell
$env:WININTEL_API_KEY = "winintel-local"
```

Linux/macOS 使用 `export WININTEL_API_KEY=winintel-local`；有鉴权时替换为真实密钥。
界面方式保存密钥即可，不需要另外设置环境变量。

两款客户端的完整自动执行流程尚未实测。
自动修改文件、执行命令还需要服务端工具调用支持，仅勾选客户端选项不能开启这项服务端能力。
如果连不上，先检查 IP、端口和 `/v1/models`；404 检查路径，找不到模型时核对 TP4 / TP8 名称。

[DeepSeek Harness 官方配置说明](https://deepseek-harness.github.io/deepseek-harness/guide/providers)。

### 复现并发测试

在安装后的 `runtime` 目录执行。每次只测一个并发档位，
`C=1` 可替换为 `2`、`4`、`8`、`16`；服务最大并发设为 16：

```bash
C=1
.venv/bin/python scripts/run_sm70_guarded.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --report "code-c${C}-gpu.json" -- \
  .venv/bin/python scripts/benchmark_sm70_http.py \
  --base-url http://127.0.0.1:8037/v1 \
  --model winintel-qwen27-nvfp4-tp8 \
  --tokenizer /path/to/model \
  --workload english-code --coding-tasks lru --concurrency "$C" --rounds 3 \
  --first-token-timeout-seconds 180 \
  --max-tokens 2048 --report "code-c${C}.json"
```

各请求通过屏障同时发起，保存输入输出、首字等待、解码速度、
总吞吐和代码行为检查结果；监控脚本记录显存和功耗，
每个请求的首字等待上限为 180 秒。模型下载与整轮测试时长不受此上限限制。
运行前按已有吞吐估算首字等待；英文 LRU 代码任务用于复现最快任务的成绩。
报告同时保存通过代码检查的最佳轮次与均值。

长输入检索，先从 8K 开始：

```bash
.venv/bin/python scripts/run_sm70_guarded.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --report context-8k-gpu.json -- \
  .venv/bin/python scripts/benchmark_sm70_http.py \
  --base-url http://127.0.0.1:8037/v1 \
  --model winintel-qwen27-nvfp4-tp8 --tokenizer /path/to/model \
  --workload context-retrieval --context-lengths 8192 \
  --concurrency 1 --rounds 1 --max-tokens 128 --report context-8k.json
```

服务需预留输出空间，例如输入 8,192 token 时，启动设置
`SM70_MAX_MODEL_LEN=16384`。逐档选择 `32768`、`65536`、`131072`、
`262016` 作为测试输入；接近 256K 的输入预留 128 token 输出。
报告保存实际输入 token 数、三处检索结果、首字等待与每路解码速度。

长上下文英文代码测速，使用相同的 LRU 实现与单元测试任务：

```bash
.venv/bin/python scripts/benchmark_sm70_http.py \
  --base-url http://127.0.0.1:8037/v1 \
  --model winintel-qwen27-nvfp4-tp8 --tokenizer /path/to/model \
  --workload english-code --coding-tasks lru --coding-context-tokens 8192 \
  --concurrency 1 --rounds 1 --max-tokens 2048 --report code-context-8k.json
```

将 `--coding-context-tokens` 改为 `260096`，在 262,144 token
上下文配置内预留 2,048 token 输出。背景为重复英文参考代码，
每次请求均记录实际输入长度，生成代码必须通过行为检查及至少八项单元测试。

## 社区与联系

欢迎共同优化 V100 多并发吞吐、长输入预填充和模型适配，
分享可复现的性能结果与部署经验。

⭐ 点亮 **Star**，让更多 V100 用户发现本项目。
🔔 点击 **Watch → Custom → Releases**，订阅新版本通知。
🐛 欢迎[提交 Issue](https://github.com/alexhmyang/winintel-vllm/issues)
或 Pull Request，附上硬件配置、复现步骤与实测结果。
📺 关注 Bilibili、订阅 YouTube，观看演示与部署视频。

**数萌AI服务器 (winintel)**  
官方网站：[winintel.com](https://winintel.com)  
联系邮箱：service@winintel.com

本项目基于 vLLM 及开源组件，使用 AI 辅助开发与整理。
许可证：Apache-2.0；部分组件 MIT。详见 [NOTICE](NOTICE) 与
[THIRD_PARTY.md](THIRD_PARTY.md)。
