# AirLLM 使用说明（已针对本机配置）

> 目标机器：Windows 11 家庭版 · i5-13420H (8核12线程) · 15.7 GB 内存 · **Intel UHD 核显（无 NVIDIA 独显）** · 954 GB NVMe SSD (C:400 / D:534)
> 环境位置：`D:\airllm-lab` · Python 3.12.6 · airllm 4.0.0 · torch 2.14.1+cpu · transformers 5.19.0

---

## 一、AirLLM 到底是怎么回事

### 1.1 它没有"压缩"模型

AirLLM 的官方卖点是 **不用量化、不用蒸馏、不用剪枝**，就能在 4GB 显存上跑 70B。这句话是真的，但它靠的不是把模型变小，而是**让模型从来不在同一时刻完整地待在显存里**。

### 1.2 真正的机制：按层流式加载

我从 airllm 4.0.0 源码里确认了它的实现（`airllm/airllm_base.py`，类 `AirLLMBaseModel` 的文档字符串写得很直白）：

```
The checkpoint is split into per-layer shards on disk. The real transformers model is
instantiated on the ``meta`` device (no memory used) and owns the full forward /
generation logic. AirLLM only attaches forward hooks to each big module (embeddings,
every decoder layer, the final norm and the lm_head) to stream that module's weights
disk -> GPU right before it runs and free them right after, prefetching the next
module on a worker thread.
```

翻译成人话，一共四步：

1. **离线切分**：第一次运行时，把 HuggingFace 的 checkpoint 拆成"一个模块一个文件"存到磁盘上（`model.embed_tokens.safetensors`、`model.layers.0.safetensors` … `model.norm.safetensors`、`lm_head.safetensors`）。
2. **模型挂在 meta 设备上**：真正的 transformers 模型对象用 PyTorch 的 `meta` device 实例化 —— 它只有结构、形状，**不占一个字节的内存**。
3. **前向 hook 流式搬运**（源码 `_pre_hook` / `_post_hook`）：
   - `_pre_hook`：这个模块马上要算了 → 从磁盘把这个模块的权重读进来，搬到计算设备；
   - 算完 → `_post_hook` 立刻 `module.to('meta')` 把权重扔掉 + `clean_memory()`。
4. **预取**：算第 N 层的时候，后台线程已经在读第 N+1 层了（`ThreadPoolExecutor(max_workers=1)`），用磁盘 I/O 和计算互相掩盖。

所以峰值占用 ≈ **最大的一层**，而不是整个模型。Llama-70B 有 80 层，最大的一层约 1.7 GB，这就是"4GB 显存跑 70B"的由来。

### 1.3 代价：每个 token 都要把整个模型从磁盘读一遍

这是理解 AirLLM 的**唯一关键**：

> 生成 **每一个 token**，都要把**全部**权重从磁盘读一遍。

因为算完就扔了，下一个 token 得重新读。于是有一个非常好用的估算公式：

```
每 token 秒数 ≈ 模型字节数 ÷ 磁盘读速度
```

我把这个公式和你的实测数字代入，见第三节。

### 1.4 顺带说明：压缩选项为什么在本机不可用

AirLLM 支持 `compression='4bit'/'8bit'`（块状量化）来加速约 3 倍。但它依赖 `bitsandbytes`，而 **bitsandbytes 需要 NVIDIA GPU**。本机没有 NVIDIA 显卡，所以这个加速选项直接不可用 —— 也就是说，在这台机器上你**只能用不量化的原始精度**，反而正好符合你"不量化"的诉求。

---

## 二、我为你做的配置

### 2.1 目录结构

```
D:\airllm-lab\
├── .venv\                    独立 Python 环境 (不动你的系统 Python)
├── airllm_env.ps1            环境变量配置 (镜像 / 缓存 / 线程数)
├── run_airllm.py             推理入口脚本 (已适配纯 CPU)
├── hf-cache\                 模型缓存 + 切片缓存 (在 D 盘，因为动辄上百 GB)
├── _pkg\ _src\               源码包，方便查证实现细节，可删
└── README-使用说明.md        本文件
```

### 2.2 关键配置项及原因

| 配置 | 值 | 为什么 |
|---|---|---|
| `HF_ENDPOINT` | `https://hf-mirror.com` | **huggingface.co 在本机网络下不可达**（实测超时），hf-mirror 通 |
| `HF_HOME` | `D:\airllm-lab\hf-cache` | 模型 + 切片副本约 2× 模型大小；C 盘只剩 203 GB，D 盘剩 515 GB |
| `HF_HUB_DISABLE_XET` | `1` | hf-mirror 不支持 Xet 传输协议，不关会卡住 |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | 进度条写 stderr，会被 PowerShell 误判成命令失败（exit 1） |
| `OMP_NUM_THREADS` | `8` | i5-13420H 是 4P+4E，物理核 8 个 |

### 2.3 对官方示例代码的两处必要修改

官方 README 的示例在**没有 NVIDIA 显卡的机器上会直接崩**：

```python
# ❌ 官方示例（需要 CUDA）
model = AutoModel.from_pretrained("Qwen/Qwen3-32B")          # device 默认 "cuda:0"
generation_output = model.generate(input_tokens['input_ids'].cuda(), ...)
```

`run_airllm.py` 里已经改成：

```python
# ✅ 本机可用
model = AutoModel.from_pretrained(args.model, device="cpu", ...)
input_ids = input_tokens["input_ids"].to(device)   # 不是 .cuda()
```

`device` 参数是自动探测的：有 CUDA 用 `cuda:0`，有 Apple MPS 用 `mps`，否则 `cpu`。

---

## 三、本机实测数据（不是估算）

我用 `Qwen/Qwen2.5-0.5B-Instruct`（0.49B，权重 942 MB，24 层）跑通了完整链路：

### 3.1 磁盘与网络基准

| 项目 | 实测值 |
|---|---|
| D 盘顺序写 | **1.94 GiB/s** |
| D 盘顺序读 | **1.88 GiB/s**（8 GiB 文件，部分命中缓存，冷读可能略低） |
| hf-mirror 下载速度 | **≈ 1.8 MB/s**（942 MB 权重花了 8 分 51 秒） |
| 介质 | NVMe SSD，C 盘和 D 盘是**同一块** 954 GB 物理盘 |

### 3.2 首次运行

| 阶段 | 耗时 |
|---|---|
| 下载 942 MB 权重 | 531 s（网络瓶颈） |
| 切分成 52 个模块文件 | ~3 s |
| 载入 + 切分合计 | 543.6 s |
| **生成 16 个 token** | **39.7 s → 2.48 s/token（0.40 token/s）** |

输出正确：`The capital of France is Paris.`

### 3.3 重要发现：冷/热缓存差一倍

我跑了第二次（切片已存在，载入只要 **2.3 s**，`exit code 0`），结果反而**更慢**：

| 运行 | 页缓存状态 | 速度 |
|---|---|---|
| 第 1 次 | **热**（切片刚写完，还在内存页缓存里） | **2.48 s/token** |
| 第 2 次 | **冷**（中间我跑了 8 GiB 磁盘读写基准，把缓存冲掉了） | **4.14 s/token** |

这是个关键信号，它解释了为什么 AirLLM 的实际表现明显差于"磁盘速度"的理论值：

- 冷读时有效吞吐只有 **942 MB ÷ 4.14 s ≈ 0.23 GB/s**，而裸磁盘顺序读是 **1.88 GiB/s** —— **差了约 8 倍**；
- 差距来自 safetensors 头部解析、52 个文件的 open/close、PyTorch 张量构造与 dtype 转换、CPU 前向计算；
- 模型越小、层文件越碎，这部分开销占比越高。

**对 70B 的意义**：141 GB 远超 15.7 GB 内存，页缓存完全失效 —— 70B **永远是冷读**。所以第四节的估算必须用冷读效率，不能乐观地拿裸磁盘速度去算。

### 3.4 磁盘占用实测

```
模型缓存总计          1895.6 MB
  ├─ 原始 model.safetensors   942.3 MB
  └─ splitted_model\          942.3 MB   (52 个文件)
```

**实打实的 2 倍占用**。原因我在源码里查到了（`utils.py:link_or_copy_file` 和 `:642`）：只有当"一个分片文件里恰好只装一个模块的权重"时，AirLLM 才会用硬链接省空间；否则老老实实复制。单文件模型（如 Qwen2.5-0.5B）和常规多分片模型（如 Llama-70B，分片是按大小打包的，一个分片里有好几层）都**不满足**这个条件，所以都是 2 倍。

> 省一半空间的办法：`--delete-original`，切分完删掉原始下载。

---

## 四、70B 到底能不能跑 —— 诚实结论

### 4.1 算给你看

以 **Llama-3.1-70B** 为例（fp16，约 **141 GB** 权重，80 层，每层约 1.75 GB）：

| 项目 | 数字 |
|---|---|
| 每个 token 要从磁盘读 | **141 GB** |
| 磁盘理论下限（裸顺序读 1.9 GB/s） | **74 s/token** |
| 按冷读实测效率（0.23 GB/s）外推 | **约 10 分钟/token** |
| **综合现实估计** | **约 1.5 ~ 3 分钟/token** |
| 生成 16 个 token 的一句话 | **约 25 ~ 50 分钟** |
| 下载 141 GB（1.8 MB/s） | **约 22 小时** |
| 磁盘需求 | **约 282 GB**（或加 `--delete-original` 后 141 GB） |
| 你的剩余磁盘 | D 盘 515 GB —— **够，但会占掉一大半** |

> 为什么不是 10 分钟：第 3.3 节那个 0.23 GB/s 是用 **0.5B 模型**测的，它的层文件只有十几 MB，碎文件开销占比极高。70B 的层文件是 **1.75 GB 的大块顺序读**，能把 open/解析/张量构造的开销摊薄，因此有效吞吐会明显好于 0.23 GB/s、但达不到裸盘的 1.9 GB/s。取 1.0~1.5 GB/s 比较合理 → 141 GB ÷ 1.25 GB/s ≈ **113 s/token**。
>
> CPU 算力不是瓶颈：70B 每 token 约 140 GFLOPs，i5-13420H 有效算力几十 GFLOPS 量级，约 2~5 s/token，会被预取掩盖在磁盘 I/O 之下。

### 4.2 所以结论是

**"能跑通"和"能用"是两回事。**

- ✅ 技术上确实能跑起来，输出也是正确的（对不量化的原始精度）；
- ❌ 但 **1.5~3 分钟一个字**，只能用于"挂一晚上跑一批文档"这种离线批处理，任何交互式使用都不可能；
- ❌ 而且 **本机没有 NVIDIA 显卡，AirLLM 的核心卖点对你完全不适用**。AirLLM 解决的是"显存不够"，不是"算力不够"。你的模型本来就没打算放进显存里 —— 它在内存里都放不下（141 GB vs 15.7 GB），所以 AirLLM 的磁盘流式方案虽然能work，但比"把一个小模型塞进内存"要慢 **几百到上千倍**。

### 4.3 一个更直观的对比

同样是这台笔记本：

| 方案 | 模型 | 速度 | 磁盘 |
|---|---|---|---|
| **Ollama + 4bit 量化** | Qwen2.5-7B | **约 5~15 token/s** | ~4.5 GB |
| AirLLM 不量化 | Qwen2.5-0.5B（玩具） | 0.40 token/s | 1.9 GB |
| AirLLM 不量化 | Llama-70B | **0.006~0.011 token/s** | 282 GB |

差距的来源就是"量化"本身：4bit 把权重压到 1/4，让模型**整个装进内存**，于是每个 token 不用再碰磁盘。你为了"不量化"付出的代价，就是每次生成都要重读磁盘。

---

## 五、怎么用

### 5.1 最省事：双击 `run.bat`

直接双击 `D:\airllm-lab\run.bat` 即可，环境变量、镜像、线程数都已经在里面设好了。

想换模型/提示词，在命令行里传参（会覆盖 bat 里的默认值）：

```powershell
cd D:\airllm-lab
.\run.bat --tokens 4                                        # 快速冒烟测试
.\run.bat --model Qwen/Qwen2.5-1.5B-Instruct --tokens 16    # 换大一点的模型
```

> 注意：`run.bat` 是**纯 ASCII + CRLF** 写的。如果你用编辑器改它，务必保持
> **CRLF 换行**和 **ANSI/ASCII 编码** —— cmd.exe 对 LF 换行和 UTF-8 BOM 的
> .bat 解析会错乱（实测会把 `HF_HUB_DISABLE_XET` 截断成 `LE_XET`）。
> 中文提示词请用下面的 PowerShell 方式传，或改 `run_airllm.py` 里的默认值。

### 5.2 手动方式（推荐，能传中文提示词）

```powershell
# 1. 进目录并加载环境变量（每开一个新终端都要做一次）
cd D:\airllm-lab
. .\airllm_env.ps1

# 2. 跑推理
& "D:\airllm-lab\.venv\Scripts\python.exe" run_airllm.py `
    --model "Qwen/Qwen2.5-0.5B-Instruct" `
    --prompt "用一句话解释什么是机器学习。" `
    --tokens 32
```

### 5.3 常用参数

| 参数 | 说明 |
|---|---|
| `--model` | HF repo id 或本地路径，如 `Qwen/Qwen2.5-1.5B-Instruct` |
| `--prompt` | 输入文本 |
| `--tokens` | `max_new_tokens`，默认 32 |
| `--device` | `auto`(默认) / `cpu` / `cuda:0` / `mps` |
| `--delete-original` | 切分后删原始权重，**省一半磁盘** |
| `--shards-path` | 指定切片存放目录 |
| `--no-prefetch` | 关预取（内存紧张时可试） |
| `--timing` | 打印每 token 耗时 |
| `--compression` | `4bit`/`8bit` —— **本机不可用**，会被自动忽略 |

### 5.4 建议先拿这个模型练手

```powershell
# 1.5B，约 3 GB 下载，切片后约 6 GB，是你这台机器上"能感受到 AirLLM 行为"的最大尺寸
& "D:\airllm-lab\.venv\Scripts\python.exe" run_airllm.py `
    --model "Qwen/Qwen2.5-1.5B-Instruct" --tokens 16 --timing
```

---

## 六、本机的已知风险与限制

1. **没有 CUDA，`compression` 不可用** —— 拿不到官方宣称的 3× 加速。
2. **内存只有 15.7 GB，偏紧**。70B 单层约 1.7 GB + 预取中的下一层，再加上 `embed_tokens`/`lm_head` 各约 2 GB，峰值会到 6~8 GB；再叠加 Windows 和浏览器，有 OOM 风险。
3. **Windows 上内存碎片可能累积**。AirLLM 的 `clean_memory()` 会调 `malloc_trim`，但那是 glibc 的函数，**Windows 上被 try/except 静默跳过**，只留下 `gc.collect()`。长文本生成时被释放的内存不一定还给操作系统。如果出现越跑越卡，用 `--no-prefetch`，或干脆缩短 `--tokens`。
4. **Windows 无符号链接权限**（未开开发者模式）导致 HF 缓存不能靠软链接去重，磁盘占用比 Linux 更大 —— 这也是第三节那个 2× 的一部分。
5. **初次运行必须预留 2× 模型大小的磁盘**，否则会报 `MetadataIncompleteBuffer`（官方 FAQ 第一条就是这个问题）。
6. `transformers 5.19.0` 与 airllm 4.0.0 目前实测可跑通，但运行时有 `torch_dtype is deprecated` 告警，属正常。

---

## 七、我的建议

| 如果你的真实目的是… | 建议 |
|---|---|
| 想**学明白 AirLLM 的原理** | 就用现在的配置跑 0.5B / 1.5B，看 `--timing` 输出，读 `_src\airllm-4.0.0\airllm\airllm_base.py` 的 `_pre_hook`/`_post_hook`。完全够用。 |
| 想**日常本地用大模型** | 装 **Ollama**，跑 `qwen2.5:7b` 的 4bit 版本。这台机器能到 5~15 token/s，是 AirLLM 的几百倍。 |
| 想**体验 70B 的效果** | 这台笔记本做不到。要 70B 就得有 ≥48 GB 内存/统一内存的机器，或者租一张云 GPU（A100 40GB 按小时几块钱）。 |
| 想**真的用 AirLLM 跑 70B** | 唯一合理场景是"挂机跑一整夜的离线批处理"。那就加 `--delete-original` 省磁盘，然后把 `--tokens` 设小、任务切碎、用 `Start-Process` 后台跑。 |
| 想**用好 AirLLM** | 它真正的主场是：**有 CUDA 显卡**（哪怕只有 4~12 GB）+ **高速 NVMe**，跑那些绝对塞不进显存的模型（405B / DeepSeek-V3 671B）。本机缺的正是前半句。 |

---

## 附：验证记录

```
$ nvidia-smi
(命令不存在 —— 本机没有 NVIDIA 显卡)

$ python -c "import torch; print(torch.cuda.is_available())"
False

--- 第 1 次运行 (冷启动, 含下载) ---
Fetching 1 files: 100%|██████████| 1/1 [08:51<00:00, 531.58s/it]     # 网络 1.8 MB/s
saved as: .../splitted_model/model.layers.0..23.safetensors          # 切分 52 个文件
[load+split] 耗时 543.6s
总耗时: 39.7s  生成 token: 16  平均: 2.48 s/token   (0.403 token/s)
输出: "What is the capital of France? ... The capital of France is Paris."   ✅

--- 第 2 次运行 (切片复用 + 页缓存被冲掉) ---
saved layers already found in .../splitted_model
[load+split] 耗时 2.3s                                               # 复用成功
总耗时: 66.3s  生成 token: 16  平均: 4.14 s/token   (0.241 token/s)  # 冷读更慢
输出: "用一句话说明什么是机器学习。机器学习是一种人工智能的分支..."   ✅ 中文正常
exit code: 0                                                         # 进度条修复生效

--- 磁盘基准 (D: NVMe) ---
写入 8 GiB 用时 4.1s  ->  1.94 GiB/s
第1次读  用时 4.2s  ->  1.88 GiB/s
第2次读  用时 3.3s  ->  2.43 GiB/s
```
