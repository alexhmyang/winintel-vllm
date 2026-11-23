# winintel-vllm

让 V100 服务器运行本地 AI，用来聊天、写代码和处理文字资料。

作者：数萌AI服务器 (winintel) · [winintel AI 官网](https://winintel.com)

[下载安装包](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0) ·
[从零安装教程](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0) ·
[测试结果表格](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告) ·
[English](README.md)

[Bilibili 视频教程](https://space.bilibili.com/429959998) ·
[YouTube · winintel AI](https://www.youtube.com/@winintelcom)

## 这个项目是做什么的？

winintel-vllm 是为 NVIDIA Tesla V100 显卡适配的 AI 推理软件。
你把它安装到服务器上、下载模型并启动服务后，就可以让自己电脑上的 AI 客户端连接这台服务器。
模型生成回答的计算在你的服务器上完成。

当前发布版本为 v0.1.0，主要适配 Qwen3.8-27B 文本模型。
它基于 vLLM 0.17.0 和开源组件，适合已有 V100 服务器、希望部署本地 AI 的用户。

它可以帮助你：

- 聊天、写代码、解释代码、整理文字资料。
- 让多个人同时连接一台 AI 服务器。
- 连接支持 OpenAI 兼容接口的客户端，包括 WorkBuddy 和 DeepSeek Harness。
- 更快地生成回答：利用模型预测接下来可能生成的内容，并减少显卡重复执行任务的开销。
- 处理较长的文字：八卡配置已测试接近 256K 的单次输入；实际速度和容量取决于输入长度、显存和同时使用的人数。

本版本提供文字输入和文字输出。图片、视频和客户端的完整自动执行流程不属于已验证功能。

## 我的显卡能运行吗？

先看显卡数量和每张卡的显存，不要只看模型下载文件的大小。
本版本虽然下载的是 NVFP4 量化模型，但运行时会将权重展开成 FP16，
仅 27B 模型权重就约占 50 GiB，另外还要为对话缓存和加速功能预留显存。

| 显卡配置 | 能否运行本版本的 27B | 建议 |
| --- | --- | --- |
| 1 张 V100 32GB | 显存不足 | 选择 7B/8B 小模型，按完整教程的单卡路径部署 |
| 2 张 V100 16GB | 显存不足 | 总显存仍只有 32GB，不能套用两张 32GB 的配置 |
| 2 张 V100 32GB | 提供配置，尚未硬件实测 | 先验证启动和回答，再考虑增加输入长度 |
| 4 张 V100 32GB | 已实测 | 推荐用来开始部署本版本的 27B |
| 8 张 V100 32GB | 已实测 | 适合多人使用和较长文字输入 |

这里的 7B/8B 是入门选择，不代表 27B 专用启动脚本支持所有小模型。
完整教程提供单卡的另一条启动路径，并注明尚未实测的部分。

需要 Ubuntu/Linux 服务器、兼容的 NVIDIA 驱动和 CUDA Toolkit 12.8。
Windows 电脑可以用来连接服务器和运行客户端；本教程中的安装命令在 Ubuntu 服务器终端执行。

## 第一次安装，从哪里开始？

如果还没有安装 Ubuntu、显卡驱动或 CUDA，请先打开
[从 Ubuntu 安装到模型启动、开机自启的完整教程](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0)。
教程按单卡、双卡、四卡和八卡分别说明。

如果服务器环境已经准备好，下面是四卡/八卡部署的主要步骤。

### 1. 下载并安装软件

在服务器上打开终端，执行：

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

校验应显示 `OK`。安装程序会创建 Python 环境、安装依赖并编译 V100 所需的扩展。
安装包默认使用国内 Python 和 PyTorch 镜像。第一次安装需要等待下载与编译完成。

这里先安装软件，不附带模型目录。后面再根据显卡数量启动模型。
下载时选择带 `install` 的完整安装包；只有 `source` 的源码包不能代替它。

### 2. 下载模型

仍在刚才的安装目录执行：

```bash
HF_ENDPOINT=https://hf-mirror.com runtime/.venv/bin/hf download \
  nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir "$HOME/models/Qwen3.8-27B-NVFP4"
```

这条命令使用国内镜像，将完整模型保存到你的用户目录下。
下载必须包含配置、分词器和全部权重文件。

### 3. 按显卡数量启动

四张 V100 32GB：

```bash
SM70_TENSOR_PARALLEL_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
WININTEL_PROFILE=throughput \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

八张 V100 32GB：

```bash
WININTEL_PROFILE=throughput \
bash start.sh "$HOME/models/Qwen3.8-27B-NVFP4"
```

等待模型加载和服务启动。第一次启动还需要准备加速缓存。
默认端口为 `8037`，这些命令使用 8,192 token 的上下文配置。
终端窗口关闭可能导致服务停止；长期使用请按完整教程配置开机自启。

## 先确认服务能回答，再连接客户端

下面用服务器地址 `192.168.1.100` 举例。请把它替换成你自己的服务器 IP。
不知道 IP 时，在服务器终端执行 `hostname -I`，选择客户端能访问的地址。

| 要填写的内容 | 八卡示例 |
| --- | --- |
| 服务器 IP | `192.168.1.100`，需替换 |
| 接口基础地址（Base URL） | `http://192.168.1.100:8037/v1` |
| 完整聊天地址 | `http://192.168.1.100:8037/v1/chat/completions` |
| 模型名称 / Model ID | `winintel-qwen27-nvfp4-tp8` |
| 四卡时的模型名称 | `winintel-qwen27-nvfp4-tp4` |
| API Key | 默认启动未开启密钥验证；客户端必填时可用占位值 `winintel-local` |

占位值不是服务器的密码。如果你另外配置了鉴权网关，请填网关要求的真实密钥。
模型名称可以通过 `http://192.168.1.100:8037/v1/models` 查看，应与服务返回的一致。

在 Windows PowerShell 中执行下面的命令，先确认服务器可以回答：

```powershell
$request = @{
  model = "winintel-qwen27-nvfp4-tp8"
  messages = @(@{ role = "user"; content = "请用中文介绍一下你自己。" })
  max_tokens = 256
  stream = $false
} | ConvertTo-Json -Depth 5
Invoke-RestMethod `
  -Uri "http://192.168.1.100:8037/v1/chat/completions" `
  -Method Post -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($request)) |
  Select-Object -ExpandProperty choices |
  ForEach-Object { $_.message.content }
```

四卡用户把命令里的模型名称改成 `winintel-qwen27-nvfp4-tp4`。
能看到回答后，再配置下面的客户端。

## WorkBuddy 怎么连接？

WorkBuddy 是客户端，winintel-vllm 是提供模型回答的服务器。
无需购买 DeepSeek 或 OpenAI 的 API 来连接这台本地服务器。

1. 安装并打开 [WorkBuddy](https://www.codebuddy.cn/)，进入“设置 → 模型”。
2. 添加自定义模型，选择“自定义 / Custom”。
3. 按下面的表格填写，保存。
4. 回到对话页面，在模型选择器中选中刚添加的模型。
5. 先发送“请用中文介绍一下你自己”，确认能收到回答。

| WorkBuddy 设置项 | 填写内容 |
| --- | --- |
| 显示名称 | `winintel 本地 AI` |
| 模型名 / Model ID | `winintel-qwen27-nvfp4-tp8`；四卡改为 `winintel-qwen27-nvfp4-tp4` |
| URL / 接口地址 | `http://192.168.1.100:8037/v1/chat/completions` |
| API Key | 默认服务可填 `winintel-local`；有鉴权时填真实密钥 |
| 自定义协议 | 默认关闭，使用标准聊天接口 |
| 图片输入 | 关闭，本版本只提供文字推理 |
| 上下文 / 输出长度（如有） | 输入预算先用 `6000`，最大输出用 `2048` |
| 工具调用（如有） | 当前默认服务未开启工具调用解析，先按文字聊天配置 |

WorkBuddy 官方说明：默认模式会校验、补全标准聊天路径。
如果你的版本只提供“Base URL”字段，填到 `/v1`；
如果提供“URL / 接口地址”字段，使用上表的完整聊天地址。
最终请求应指向一次 `/v1/chat/completions`，不要重复添加路径。

先验证聊天和生成代码。自动修改文件、执行命令等任务还需要服务端工具调用支持；
仅在客户端勾选“工具调用”不能让服务器获得这项能力。
当前版本未完成 WorkBuddy 完整任务流程的实测。

[WorkBuddy 官方模型配置说明](https://www.codebuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Model)

## DeepSeek Harness 怎么连接？

DeepSeek Harness 是智能体客户端，也可以连接其他厂商和自己部署的模型。
这里连接的是你服务器上的 Qwen，不是 DeepSeek 云端模型。

先按 [DeepSeek Harness 官方安装说明](https://github.com/deepseek-ai/deepseek-harness)
安装并启动客户端，再在它的网页界面操作：

1. 打开“设置 → 模型”。
2. 添加模型提供商，切换为“自定义模型 API”。
3. 填写下表内容，并添加至少一个模型。
4. 保存后，回到对话页面，选择 `winintel` 提供商下的模型。
5. 先用简短文字测试连接。模型配置通常在下一次请求生效。

| DeepSeek Harness 设置项 | 填写内容 |
| --- | --- |
| Provider ID | `winintel`，使用小写 |
| 显示名称 | `winintel 本地 AI` |
| 基础 URL / Base URL | `http://192.168.1.100:8037/v1` |
| API 协议 | `OpenAI Chat Completions` |
| API 密钥 | 默认服务可填 `winintel-local`；有鉴权时填真实密钥 |
| 模型 ID | `winintel-qwen27-nvfp4-tp8`；四卡改为 `winintel-qwen27-nvfp4-tp4` |
| 上下文窗口 | `8192`，与上面的服务启动配置一致 |
| 最大输出 token | `2048` |
| 输入类型 | 只选择“文本” |
| 推理等级 | 初次接入不声明额外等级，使用服务默认行为 |

此处 Base URL 只填到 `/v1`，不要加 `/chat/completions`。
不要选择 OpenAI Responses、Anthropic Messages 或 Gemini API，这些不是本版本提供的协议。
如果“探测模型”没有找到结果，手动输入模型 ID 即可。

如果出现 `developer` 角色或输出长度参数不兼容的错误，
在设置页点击“打开配置文件”，找到已有的 `llm-pi-ai` 配置，
在 `providers` 下的 `winintel` 中加入下面两项：

```yaml
compat:
  supportsDeveloperRole: false
  maxTokensField: max_tokens
```

这段是已有 `winintel` 提供商内部的字段，不要用它覆盖整个配置文件或其他模型。
默认网页 profile 的配置文件位于 `$DSH_HOME/profiles/web/cordis.patch.yml`，
具体位置以客户端设置页打开的文件为准。

Harness 的自动执行任务也依赖工具调用。
当前默认启动脚本未开启工具调用解析，尚未完成 Harness 的完整任务流程验证。
聊天连接成功后，不能据此认为自动操作文件等功能也已通过测试。

[DeepSeek Harness 官方模型配置说明](https://deepseek-harness.github.io/deepseek-harness/guide/providers)

## 连不上或报错，怎么处理？

| 现象 | 检查方法 |
| --- | --- |
| 连接失败、连接被拒绝 | 确认模型服务已启动，IP 和端口正确；先在客户端电脑访问 `/v1/models` |
| 客户端在另一台电脑上，却填了 `127.0.0.1` | 改成服务器 IP；`127.0.0.1` 指的是客户端电脑自己 |
| 网络超时 | 检查两台机器是否互通，以及服务器防火墙是否允许客户端访问 `8037` |
| 404 / 找不到接口 | 检查是否漏了 `/v1`，或者重复写了 `/chat/completions` |
| 找不到模型 | 从 `/v1/models` 核对名称；四卡和八卡的名称不同 |
| 401 / 密钥错误 | 如果有网关或鉴权，填写其真实密钥，占位值只适用于默认未鉴权服务 |
| 提示输入太长 | 清理历史对话或减少资料长度；客户端设置不能超过服务器上下文上限 |
| 可以聊天，却不能自动改文件 | 检查服务端工具调用支持；默认发布配置没有启用工具调用解析 |
| 模型启动时显存不足 | 检查卡数、每卡显存、其他 GPU 进程；减少上下文或同时请求数 |

## 实测有多快？

下面是八张 V100 32GB、官方 Qwen3.8-27B-NVFP4 检查点的代码生成成绩。
测试使用短提示词生成英文 Python 代码，不能直接当作所有任务的速度。

“并发”表示同时有多少个请求，例如 8 路就是同时处理 8 个请求。
“总速度”是这些请求合计生成的 token 数，不是每个人都能达到这个速度。
token 是模型处理文字的单位，不能固定换算成汉字数。

| 同时请求数 | MTP11 平均总速度 | MTP9 平均总速度 | MTP5 平均总速度 |
| ---: | ---: | ---: | ---: |
| 1 | 93.80 token/s | 94.03 token/s | 86.78 token/s |
| 8 | 516.89 token/s | 547.06 token/s | 592.26 token/s |
| 16 | 802.10 token/s | 831.58 token/s | 879.43 token/s |

每档测试两轮，三种配置共 150 份代码全部通过检查。
MTP5 的最佳轮次为八路 592.92 token/s、十六路 880.95 token/s。
这里的 MTP 数字是加速配置，不是显卡数量；新手先使用默认设置即可。

长文字输入需要先阅读资料，因此第一段回答会更晚出现：

| 任务 | 实际输入 token | 等到第一段回答 | 回答生成速度 | 验证结果 |
| --- | ---: | ---: | ---: | --- |
| 英文代码生成 | 8,192 | 2.21 秒 | 82.37 token/s | 代码检查通过 |
| 英文代码生成 | 260,096 | 160.66 秒 | 10.83 token/s | 代码检查通过 |
| 合同原文检索 | 32,768 | 9.68 秒 | 91.76 token/s | 3/3 条款正确 |
| 合同原文检索 | 262,016 | 163.64 秒 | 18.73 token/s | 3/3 条款正确 |

长输入表为单次请求、每项一轮。完整 256K 多请求同时运行尚未验证。
小白建议从 8K 上下文开始，先确认回答正常，再增加资料长度。

[查看完整性能、首字等待、显存、逐轮结果和验证表格](https://github.com/alexhmyang/winintel-vllm/releases/tag/v0.1.0#测试报告)。
结果直接显示在网页上，无需下载 CSV 或阅读 JSON 报告。

## 技术名词，看不懂也不影响安装

| 名词 | 通俗说明 |
| --- | --- |
| 推理 | 模型根据你输入的内容生成回答 |
| 27B | 模型约有 270 亿个参数，会影响显存需求 |
| TP4 / TP8 | 将一个模型分配到四张 / 八张显卡共同运行 |
| 上下文 | 模型一次能处理的文字范围，包括问题、历史对话和回答 |
| MTP | 先预测接下来的几个 token，再校验，以加快回答生成 |
| CUDA Graph | 记录并重复执行显卡计算流程，减少重复调度开销 |
| SM70 / WININTEL_SM70 | V100 的架构标识，以及本项目为它选择的注意力计算后端 |
| FP16 / NVFP4 | 数值存储或计算格式；本版本下载量化权重后以 FP16 运行 |
| KV 缓存 | 保存对话计算的中间结果，输入越长、同时请求越多，通常越占显存 |
| OpenAI 兼容 API | 使用常见的接口格式，让客户端可以连接不同模型服务器 |

## 社区与联系

欢迎分享部署经验、提交问题和可复现的改进。

[Star 项目](https://github.com/alexhmyang/winintel-vllm) ·
[提交 Issue](https://github.com/alexhmyang/winintel-vllm/issues) ·
[订阅版本更新](https://github.com/alexhmyang/winintel-vllm/subscription)

数萌AI服务器 (winintel) · 官网：https://winintel.com · 邮箱：service@winintel.com

本项目基于 vLLM 及开源组件，使用 AI 辅助开发与整理。
许可证为 Apache-2.0，部分组件为 MIT；详见 [NOTICE](NOTICE) 与 [THIRD_PARTY.md](THIRD_PARTY.md)。
