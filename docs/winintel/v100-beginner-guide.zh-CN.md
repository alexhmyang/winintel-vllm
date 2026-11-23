# winintel-vllm：V100 32GB 从零安装与部署

[返回项目首页](https://github.com/alexhmyang/winintel-vllm/blob/main/README.zh-CN.md)

作者：**数萌AI服务器 (winintel)** · [winintel AI](https://winintel.com)  
适用版本：Qwen **v0.1.0** · 更新：2026-11-23  
[下载页](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0) ·
[网页测试表格](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)

这份教程从 Ubuntu、V100 驱动和 CUDA 开始，最后让模型通过 API 回答问题。
先完成公共安装步骤，再选择自己服务器的卡数。所有命令均在 **Ubuntu 服务器的终端**
执行；Windows 电脑可以使用 SSH 工具连接服务器。复制代码时，不要复制代码块外的说明。

## 1. 先看自己的显卡能跑什么

**这版 Qwen3.8-27B-NVFP4 在加载时会把量化权重展开成 FP16。**
NVFP4 下载文件较小，不代表运行时仍只占 4-bit 显存。
27B 的 FP16 权重约 50 GiB，此外还需要 KV 缓存、CUDA Graph、草稿模型和工作空间。
**单张 V100 32GB 装不下本版本的 27B；单卡入门请选择 7B/8B。**

| V100 配置 | 总标称显存 | 入门选择 | Qwen3.8-27B | 其他模型的容量参考 |
| ---: | ---: | --- | --- | --- |
| 1 张 32GB | 32GB | Qwen2.5-7B-Instruct / Qwen3-8B，FP16 | 本版本不适用 | 14B FP16 较紧，需要短上下文、低并发；未实测 |
| 2 张 **16GB** | 32GB | 较小模型，需另行验证 TP2 配置 | **显存不足，不能一键跑通本版 27B** | 不等同于两张 32GB |
| 2 张 32GB | 64GB | 7B/8B/14B FP16；27B 可尝试下方保守配置 | 提供 TP2 配置，**尚未 GPU 实测** | 32B FP16 权重约 60 GiB，余量太小，不推荐小白这样部署 |
| 4 张 32GB | 128GB | 本教程的 27B TP4 | **TP4 已实测** | Qwen2.5-32B-Instruct / Qwen3-32B FP16 为容量候选，未验证 |
| 8 张 32GB | 256GB | 本教程的 27B TP8，多用户服务优先 | **TP8 已实测** | 70B/72B FP16 约 130–134 GiB 权重，为容量候选，未验证 |

上表的其他模型按显存估算，**不代表本安装包已经完成这些模型的适配和正确率验证**。
单卡示例的模型 ID 与架构配置已核对，但新命令尚未进行单卡生成实测。
总显存也不会自动合并：需要模型支持张量并行，注意力头数等结构能够匹配卡数。
MoE 模型按总参数量估算权重，不能只看“每 token 激活参数量”。
AWQ/GPTQ/其他 FP4 格式需要相应的 SM70 算子支持，不能因文件小就假定兼容。

**推荐选法：**已有单卡，用 7B/8B 入门；打算专门跑本版 27B，优先四卡；
已有八卡并希望服务多人，用 TP8。双卡先验证能否启动和稳定回答，再增加上下文。

## 2. 准备服务器

| 项目 | 建议 |
| --- | --- |
| 系统 | 原生 Ubuntu Server 22.04 / 24.04，x86_64；本教程不适用于 Windows、WSL 或 ARM |
| 显卡 | NVIDIA Tesla V100 **32GB**；16GB 卡不能直接套用本表 |
| CPU / 内存 | 单卡小模型建议内存至少 64GB；27B 多卡建议至少 128GB，256GB 更宽裕 |
| 磁盘 | 建议 SSD 空闲至少 200GB，容纳模型、源码、环境和编译缓存 |
| 供电 / 散热 | 按整机厂商要求配置；被动散热 PCIe 卡需要服务器风道 |
| 网络 | 能访问国内 Ubuntu、Python、PyTorch 和模型镜像；GitHub 发布文件需能下载 |
| 多卡互联 | SXM2 NVLink 拓扑优先；PCIe 版本和不同拓扑的性能需单独测试 |

内存和磁盘是部署建议，不是已测的最低要求。
安装 Ubuntu 时建议把软件源选为国内镜像，便于下面安装基础工具。
确认服务器没有其他正在使用 GPU 的业务，再进行驱动安装或重启。
Secure Boot 开启时，应按服务器厂商方式完成模块签名/MOK，
或在 BIOS 中关闭 Secure Boot 后再安装；安装工具会检测并提示。
曾用 NVIDIA `.run` 安装过驱动的系统，需要先按照工具提示完成迁移。

## 3. 安装 V100 驱动与 CUDA 12.8

V100 使用 **R580-server 专有驱动**，不要选择仅支持新架构的 open 内核模块。
本教程使用 **CUDA Toolkit 12.8**；CUDA 13 已移除 V100 的编译支持。
复用 winintel-server-doctor 安装工具，Ubuntu 软件包走国内镜像，
CUDA 工具包使用 NVIDIA 中国下载地址。

先创建下载目录：

```bash
mkdir -p "$HOME/winintel-downloads"
cd "$HOME/winintel-downloads"
sudo apt-get update
sudo apt-get install -y curl ca-certificates
curl -fL --retry 3 \
  https://github.com/alexhmyang/winintel-server-doctor/releases/download/v0.2.0/winintel-server-doctor-v0.2.0.tar.gz \
  -o winintel-server-doctor-v0.2.0.tar.gz
echo '14556ad11405fb90ffaecc31e6899bfa138d8b2245564af2a901487710711925  winintel-server-doctor-v0.2.0.tar.gz' \
  | sha256sum -c -
tar -xzf winintel-server-doctor-v0.2.0.tar.gz
cd winintel-server-doctor
```

校验应显示 `OK`，失败时停止并重新下载。
先查看计划，下面第一条命令不修改系统；确认计划中的 CUDA 为 12.8：

```bash
sudo bash install.sh --cuda 12.8 --skip-tools --dry-run
sudo bash install.sh --cuda 12.8 --skip-tools
```

`--skip-tools` 表示跳过硬件压测程序编译，适合先部署推理。
工具不会自动重启。
如果已经安装了兼容的 R580 驱动，只需要工具包，
将实际安装命令改为 `sudo bash install.sh --cuda 12.8 --skip-driver --skip-tools`。

安装完成后手动重启；SSH 会断开，服务器重新开机后再连接：

```bash
sudo reboot
```

重新连接后，检查驱动、显卡和 CUDA：

```bash
nvidia-smi
source /opt/winintel-server-doctor/cuda-env.sh 12.8
nvcc --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv
nvidia-smi topo -m
```

应能看到所有 V100、每卡约 32GB 显存、580 系列驱动，
以及 `nvcc` 输出 `release 12.8`。`nvidia-smi` 顶部的 “CUDA Version”
是驱动支持的最高版本，可能显示 13.x；**实际工具包版本看 `nvcc --version`**。
`topo -m` 中 `NV...` 表示 NVLink，PCIe 拓扑可能显示 `PIX/PHB/SYS`。

## 4. 安装 winintel-vllm 推理环境

单卡、双卡、四卡和八卡共用这一步。**先安装环境，再下载模型。**
选择带 `install` 的安装包；`source` 包是源码覆盖层，不能单独代替完整安装包。

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
source /opt/winintel-server-doctor/cuda-env.sh 12.8
sudo apt-get -o Dir::Etc::sourcelist=/var/cache/winintel-server-doctor/ubuntu.sources.list \
  -o Dir::Etc::sourceparts=- install -y git
export PATH="/opt/winintel-server-doctor:$HOME/.local/bin:$PATH"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export WININTEL_TORCH_INDEX_URL=https://mirrors.nju.edu.cn/pytorch/whl/cu128
export HF_ENDPOINT=https://hf-mirror.com
export UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download
MAX_JOBS=4 bash install.sh
```

前一步 driver 工具已将 `uv` 安装到 `/opt/winintel-server-doctor`，
因此这里直接复用它。Python 3.12 下载经镜像代理，
普通依赖使用清华源，PyTorch 使用南京大学 CUDA 12.8 源。
第三方镜像不可用时，请换成可访问的国内镜像，不要改成不兼容的 CUDA/PyTorch 版本。
首次安装需要下载依赖并编译，耗时取决于 CPU、内存和网络。
看到 `winintel AI SM70 extensions are ready` 及安装完成信息才算成功。
CPU/内存紧张时用 `MAX_JOBS=2` 重试。

这里的 `bash install.sh` **不要附带模型目录**。
附带模型会校验 27B 并自动启动默认八卡配置，不适合按卡数分步安装。
后续不要另装普通版 `vllm` 覆盖此环境。
新开终端时，回到发布目录并再次加载 CUDA 环境：

```bash
cd "$HOME/winintel/winintel-vllm-0.1.0-qwen3.8"
source /opt/winintel-server-doctor/cuda-env.sh 12.8
export HF_ENDPOINT=https://hf-mirror.com
```

## 5. 单卡 32GB：先跑 7B/8B

这条路径使用同一推理环境的通用 CLI，**不使用 27B 专用 `start.sh`**。
示例选择官方 `Qwen/Qwen3-8B`，权重以 FP16 加载，
使用 `WININTEL_SM70` 注意力后端并显式启用 CUDA Graph。
这是按配置和容量提供的入门命令，尚未完成本轮单卡生成验证。

```bash
cd "$HOME/winintel/winintel-vllm-0.1.0-qwen3.8"
mkdir -p "$HOME/models"
HF_ENDPOINT=https://hf-mirror.com runtime/.venv/bin/hf download \
  Qwen/Qwen3-8B --local-dir "$HOME/models/Qwen3-8B"
CUDA_VISIBLE_DEVICES=0 WININTEL_SM70_CUDAGRAPH=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
runtime/.venv/bin/python -m vllm.winintel.cli serve "$HOME/models/Qwen3-8B" \
  --served-model-name winintel-small \
  --host 0.0.0.0 --port 8037 \
  --tensor-parallel-size 1 --dtype float16 \
  --attention-backend WININTEL_SM70 --no-enforce-eager \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --max-model-len 4096 --max-num-seqs 1 \
  --max-num-batched-tokens 512 --gpu-memory-utilization 0.80
```

模型文件下载完整后才启动服务。可以把下载与启动命令中的模型 ID/目录
一起改为 `Qwen/Qwen2.5-7B-Instruct` 和 `$HOME/models/Qwen2.5-7B-Instruct`，
其 API 名称仍为 `winintel-small`。
两者都不要添加 27B 的 MTP 推测解码配置。
14B 虽然可能装得下，但运行余量较少，先用 7B/8B 完成整个流程。

## 6. 双卡 32GB × 2：尝试 27B 保守配置

双卡/四卡/八卡共用下面的 **27B 下载步骤，只需下载一次**：

```bash
cd "$HOME/winintel/winintel-vllm-0.1.0-qwen3.8"
mkdir -p "$HOME/models"
HF_ENDPOINT=https://hf-mirror.com runtime/.venv/bin/hf download \
  nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir "$HOME/models/Qwen3.8-27B-NVFP4"
```

这是已测 TP8 所用的官方固定检查点，完整下载配置、tokenizer 和所有权重。
不要只下载一个 `.safetensors` 文件。下载可重试，保留同一模型目录。

**TP2 尚未实测，不能保证 4K 配置一定能分配出足够 KV 缓存。**
先使用两张空闲卡、单请求和小预填充批次：

```bash
CUDA_VISIBLE_DEVICES=0,1 SM70_TENSOR_PARALLEL_SIZE=2 \
SM70_MAX_MODEL_LEN=4096 SM70_MAX_NUM_SEQS=1 \
SM70_MAX_BATCHED_TOKENS=512 SM70_MEMORY_UTILIZATION=0.92 \
SM70_KV_CACHE_DTYPE=fp8_e4m3 SM70_SPECULATIVE_TOKENS=3 \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

API 模型名称：`winintel-qwen27-nvfp4-tp2`。
若启动显存不足，可先把 `SM70_MAX_MODEL_LEN` 降为 `2048` 并检查其他 GPU 进程；
如果仍不足，选择四卡或较小模型。不要靠不断提高显存预算来挤掉运行工作空间。

## 7. 四卡 32GB × 4：推荐的 27B 起步方案

先执行上一节的 27B 下载步骤，再启动。
下面复用已有 TP4 测试的 8K、三请求、MTP5、65% 显存配置：

```bash
cd "$HOME/winintel/winintel-vllm-0.1.0-qwen3.8"
CUDA_VISIBLE_DEVICES=0,1,2,3 SM70_TENSOR_PARALLEL_SIZE=4 \
SM70_MAX_MODEL_LEN=8192 SM70_MAX_NUM_SEQS=3 \
SM70_SPECULATIVE_TOKENS=5 SM70_KV_CACHE_DTYPE=auto \
SM70_MEMORY_UTILIZATION=0.65 \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

API 模型名称：`winintel-qwen27-nvfp4-tp4`。
TP4 历史检查点测试中，三类代码任务同时运行总解码吞吐 **220.54 token/s**，
12 份输出通过行为检查。这是历史检查点成绩，
上面固定官方模型在相同卡数的新部署不承诺获得相同速度。
先验证短请求，再按业务需要调整上下文和并发。

## 8. 八卡 32GB × 8：27B 多用户部署

先执行第 6 节的 27B 下载步骤，再启动。
新用户从短上下文、低并发开始，验证成功后再增大并发：

```bash
cd "$HOME/winintel/winintel-vllm-0.1.0-qwen3.8"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SM70_TENSOR_PARALLEL_SIZE=8 \
SM70_MAX_MODEL_LEN=8192 SM70_MAX_NUM_SEQS=4 \
SM70_SPECULATIVE_TOKENS=5 SM70_KV_CACHE_DTYPE=auto \
SM70_MEMORY_UTILIZATION=0.80 \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

API 模型名称：`winintel-qwen27-nvfp4-tp8`。
完成 API 验证后，可按 `Ctrl+C` 停止前台服务，重新启动 16 路配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SM70_TENSOR_PARALLEL_SIZE=8 \
SM70_MAX_MODEL_LEN=16384 SM70_MAX_NUM_SEQS=16 \
SM70_SPECULATIVE_TOKENS=5 SM70_KV_CACHE_DTYPE=auto \
SM70_MEMORY_UTILIZATION=0.80 WININTEL_SM70_TORCH_PREFILL=1 \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

在 **8 张 V100-SXM2 32GB** 的已测短输入代码任务中，
MTP5 八请求最佳总解码吞吐 **592.92 token/s**，
十六请求最佳 **880.95 token/s**。这是所有请求合计的速度，不是单个用户的速度。
不同 PCIe/NVLink 拓扑、提示词和输出长度都会影响结果。

## 9. 验证 API，接入聊天软件

启动命令会持续占用当前终端，等待日志出现服务启动成功。
**再打开一个 SSH 终端**，查询模型列表：

```bash
curl -f http://127.0.0.1:8037/health
curl -f http://127.0.0.1:8037/v1/models
```

第一个接口返回成功状态即可，可能没有正文；
第二个接口应列出模型名。先按卡数设置变量，再发起流式对话：

```bash
# 单卡用 winintel-small；双卡改为 ...tp2；四卡改为 ...tp4
MODEL=winintel-qwen27-nvfp4-tp8
curl -fN http://127.0.0.1:8037/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，请用中文介绍自己。\"}],\"max_tokens\":256,\"stream\":true}"
```

输出应逐步出现 `data:` 流式内容，结束时出现 `[DONE]`。
模型支持思考模式时，可能先输出思考内容；可按应用需要设置聊天模板参数。
聊天软件选择 OpenAI 兼容接口，填写：

| 设置 | 内容 |
| --- | --- |
| Base URL | `http://服务器IP:8037/v1`，把“服务器IP”替换成实际地址 |
| 模型名称 | `/v1/models` 返回的名称 |
| API Key | 当前示例未配置认证；客户端必填时可填 `local` |

本教程的服务监听 `0.0.0.0`，用于可信内网。
需要公网使用时配置认证与受保护的反向代理。
只在现有防火墙规则中允许可信客户端访问 8037；
不要为了连接方便直接关闭防火墙。
先让服务器本机 `curl` 成功，再处理电脑端连接问题。

## 10. 可选：配置开机自动启动

先完成前台启动和 API 验证，再设置系统服务。
下面以 **八卡 27B** 为例，四卡/双卡分别更改 GPU 列表和 `SM70_*` 参数，
使用对应章节已经验证可启动的值。单卡通用 CLI 需要单独填写其启动命令。
用日常部署账户执行，模型和安装目录沿用前面的路径：

```bash
cd "$HOME/winintel/winintel-vllm-0.1.0-qwen3.8"
DEPLOY_USER="$(id -un)"
DEPLOY_ROOT="$(pwd)"
DEPLOY_MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
sudo tee /etc/systemd/system/winintel-qwen.service >/dev/null <<EOF
[Unit]
Description=winintel AI Qwen on V100
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$DEPLOY_ROOT
Environment=CUDA_HOME=/usr/local/cuda-12.8
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Environment=SM70_TENSOR_PARALLEL_SIZE=8
Environment=SM70_MAX_MODEL_LEN=8192
Environment=SM70_MAX_NUM_SEQS=4
Environment=SM70_SPECULATIVE_TOKENS=5
Environment=SM70_MEMORY_UTILIZATION=0.80
Environment=SM70_KV_CACHE_DTYPE=auto
ExecStart=/bin/bash "$DEPLOY_ROOT/start.sh" "$DEPLOY_MODEL"
Restart=on-failure
RestartSec=15
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
EOF
```

先在原来的服务终端按 `Ctrl+C` 停止前台进程，避免占用同一批 GPU 和端口，再执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now winintel-qwen.service
sudo systemctl status winintel-qwen.service --no-pager
sudo journalctl -u winintel-qwen.service -n 80 --no-pager
```

服务加载模型可能需要几分钟，`active` 只表示进程启动；
再次完成第 9 节 API 检查才表示可以使用。
查看实时日志：`sudo journalctl -u winintel-qwen.service -f`。
停止服务：`sudo systemctl stop winintel-qwen.service`。
不再需要自启时：`sudo systemctl disable --now winintel-qwen.service`。
如部署路径包含空格，系统服务的 `WorkingDirectory` 也需正确引用；
小白建议保持教程中的目录命名。

## 11. 常见问题

| 现象 | 处理 |
| --- | --- |
| `nvidia-smi` 找不到卡 / 驱动通信失败 | 驱动安装后先重启；检查 Secure Boot、内核模块及卡的供电，不要跳过继续安装 |
| `nvcc: command not found` | 执行 `source /opt/winintel-server-doctor/cuda-env.sh 12.8`；确认安装的是 Toolkit |
| `uv: command not found` | 把 `/opt/winintel-server-doctor` 加入 `PATH`；检查 driver 工具是否安装完成 |
| 下载连接超时 | 检查对应镜像连通性；模型使用 `HF_ENDPOINT=https://hf-mirror.com`，保留目录重试 |
| 构建进程被杀 / 内存不足 | 降低 `MAX_JOBS`；检查主机内存和磁盘；继续使用 uv 环境 |
| 单卡 27B 显存不足 | 本版本加载为 FP16，单张 32GB 不支持；改用 7B/8B 或多卡 |
| 多卡启动显存不足 | 检查 `nvidia-smi` 中其他业务；降低并发与上下文；TP2 仍不足时选择四卡 |
| “必须 2、4、8 卡” | 27B 的 `start.sh` 仅接受这些卡数；单卡使用第 5 节通用 CLI |
| GPU 数量不足 | `CUDA_VISIBLE_DEVICES` 必须列出足够的真实空闲卡，编号以 `nvidia-smi` 为准 |
| 端口被占用 | 停止重复的前台/systemd 服务；27B 可用 `SM70_PORT=8038` 更换端口并同步客户端 |
| 模型名称不存在 | 从 `/v1/models` 复制准确模型名，单卡、TP2、TP4、TP8 名称不同 |
| 本机能访问，电脑不能访问 | 使用服务器实际 IP，检查两端路由与防火墙，并确认 Base URL 带 `/v1` |
| 输出速度与宣传不同 | 先核对卡数、模型检查点、NVLink、MTP、CUDA Graph、并发和提示词；总吞吐不是单路速度 |
| 长输入等待很久 | 首字前需要预填充，先做短输入验证；上下文长度包含输入和输出 |

不要一开始就设 256K。先让短对话稳定，再测试 8K 和中等上下文的并发，
最后参考项目原始记录逐步验证长输入。
单路长输入实测不能直接推成多路 256K；长输入应完整运行并单独记录首字和解码速度。

## 12. 连接 WorkBuddy 和 DeepSeek Harness

完成第 9 节的 API 测试后，再连接客户端。下面假设服务器 IP 为
`192.168.1.100`，请换成你实际能访问的服务器地址。

| 客户端 | 设置位置 | 接口地址 | API 协议 | 模型名称 |
| --- | --- | --- | --- | --- |
| WorkBuddy | 设置 → 模型 → 自定义 / Custom | URL 填 `http://192.168.1.100:8037/v1/chat/completions`；只有 Base URL 字段时填到 `/v1` | OpenAI 兼容聊天接口 | 八卡 `winintel-qwen27-nvfp4-tp8`，四卡 `winintel-qwen27-nvfp4-tp4` |
| DeepSeek Harness | 设置 → 模型 → 自定义模型 API | Base URL 填 `http://192.168.1.100:8037/v1` | `OpenAI Chat Completions` | 八卡 `winintel-qwen27-nvfp4-tp8`，四卡 `winintel-qwen27-nvfp4-tp4` |

Harness 的 Provider ID 可填小写 `winintel`。
默认服务没有开启密钥验证，客户端必须填 API Key 时可用占位值 `winintel-local`；
若另外配置了鉴权网关，则填写真实密钥。单卡、双卡用户应从 `/v1/models` 获取实际模型名称。

保存后，在客户端的模型选择器里选中新模型，发送“你好，请介绍一下自己”。
默认 8K 服务的客户端上下文设为 8,192 token、最大输出先设为 2,048 token；
输入与回答合计不能超过服务上限。单卡或双卡配置的上限更小时，应同步降低客户端设置。

本版本提供文字推理，先关闭图片输入。默认启动脚本未开启工具调用解析，
完整客户端自动执行流程尚未实测，聊天成功不代表自动修改文件、执行命令也已验证。

需要可复制的配置文件示例或遇到路径、模型名错误，请看
[首页的客户端 API 配置说明](https://github.com/alexhmyang/winintel-vllm/blob/main/README.zh-CN.md#api-接入)。

## 13. 验证范围与联系方式

本次更新核对了已发布安装器、驱动工具、模型配置、启动参数和校验值，
并对教程命令做静态与 Bash 语法检查；**未新增单卡、双卡或全新机器安装实测**。
TP4/TP8 和性能数字沿用项目已有记录，具体检查点及测试条件见
[网页测试表格](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)。
文档更新使用了 AI 辅助。源码与第三方组件的许可证及署名见
[NOTICE](https://github.com/alexhmyang/winintel-vllm/blob/main/NOTICE) 和
[THIRD_PARTY.md](https://github.com/alexhmyang/winintel-vllm/blob/main/THIRD_PARTY.md)。

**数萌AI服务器 (winintel)** · service@winintel.com  
[官网](https://winintel.com) ·
[Bilibili 视频](https://space.bilibili.com/429959998) ·
[YouTube](https://www.youtube.com/@winintelcom) ·
[反馈部署问题](https://github.com/alexhmyang/winintel-vllm/issues)
