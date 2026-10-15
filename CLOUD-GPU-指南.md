# 云 GPU 使用指南（配合 AirLLM）

> 面向：本机无 NVIDIA 显卡、但想真正上手大模型（且预算不敏感）
> 配套脚本：`cloud_setup.sh`（已做 bash 语法检查 + 内嵌 Python 编译检查）

---

## 一、先分清你要的是哪个目标

这是全篇最重要的一张表。**"用云 GPU"不是一个目标，是两个完全不同的目标**，选错了会白花钱：

| | 目标 A：**学透 AirLLM** | 目标 B：**真正用上 70B** |
|---|---|---|
| 你想看到什么 | 「原来真能塞进小显存」 | 「能流畅对话、能干活」 |
| 显卡 | **便宜的 24GB 卡就够**（甚至 8GB） | **2× A100 80GB** 或 1× H200 141GB |
| 软件 | AirLLM | **vLLM**（不是 AirLLM） |
| 速度 | **分钟级 / token**（磁盘瓶颈） | **20~40 token/s** |
| 费用 | ≈ ¥2/小时 | ≈ ¥15/小时 |
| 一句话 | AirLLM 是**证明题**，不是生产力工具 | 要生产力，就别用 AirLLM |

### 为什么 AirLLM 在云上也不快

必须澄清一个常见误解：**AirLLM 的速度几乎不取决于显卡，而取决于磁盘。**

第 3 节实测的公式：每 token 秒数 ≈ 模型字节数 ÷ 磁盘读速度。云上 SSD 大概 1~3 GB/s，所以 141GB 的 70B 模型**无论配 4090 还是 H100，都是每分钟 1~2 个 token**。AirLLM 换来的是「能跑」，不是「跑得快」。

**推论（反直觉但很省钱）**：如果你只想体验 AirLLM，应该**把显卡往低了买、把磁盘往大了配**。24GB 的 4090 和 24GB 的 3090 对 AirLLM 没有任何区别，因为峰值显存只用 ~4GB。

---

## 二、平台怎么选

我从**你这台机器的网络**实测了各平台可达性（结果如下，`200` 即可达）：

| 平台 | 类型 | 你的网络 | 备注 |
|---|---|---|---|
| **AutoDL** (`autodl.com`) | 国内 | ✅ 200 | **首选**。支付宝充值、内置学术加速直连 HF |
| 恒源云 (`gpushare.com`) | 国内 | ✅ 200 | 类似 AutoDL |
| 矩池云 (`matpool.com`) | 国内 | ✅ 200 | 类似 AutoDL |
| 共绩算力 (`compshare.cn`) | 国内 | ✅ 200 | 潞晨科技，按秒计费 |
| RunPod (`runpod.io`) | 国际 | ✅ 200 | H100 $1.99~2.99/hr，需国际信用卡 |
| Vast.ai | 国际 | ✅ 200 | 最便宜（A100 80GB 低至 $0.68/hr），但机器质量参差 |
| Lambda | 国际 | ✅ 200 | 稳定但贵（H100 $3.29~3.99/hr） |

**给你的建议：用 AutoDL。** 理由有三：

1. **不用折腾网络**——AutoDL 内置学术加速，`source /etc/network_turbo` 之后 `huggingface.co` 直连，不需要 hf-mirror（见 [官方文档](https://api.autodl.com/docs/network_turbo/)）；
2. **支付方便**——支付宝/微信充值，不像 RunPod/Vast 需要国际信用卡；
3. **便宜且按量计费**——用完就关。

> 国际平台价格参考（2026 年，来自 [CloudZero 的对比](https://www.cloudzero.com/blog/cloud-gpu-pricing-comparison/)）：H100 80GB 从 $1.49(Vast) 到 $6.98(Azure)，A100 80GB 中位数约 $2.00/hr。国内平台通常更便宜。

---

## 三、AutoDL 实操：从注册到跑起来

### 3.1 准备（一次性）

1. 注册 [autodl.com](https://www.autodl.com)，完成**实名认证**（国内平台强制）
2. 充值（建议先充 ¥50，够跑 20+ 小时）
3. 学生的话做**学生认证**，有折扣

### 3.2 租实例

进入「算力市场」，按下面的思路选：

| 项目 | 目标 A（学 AirLLM） | 目标 B（真用 70B） |
|---|---|---|
| GPU | **RTX 3090 / 4090 24GB**（最便宜那档） | **A100 80GB × 2**（注意要选支持多卡的地区） |
| 镜像 | `PyTorch 2.x` + `CUDA 12.x` | 同上 |
| 数据盘 | **至少 300GB**（关键！见 3.5） | 100GB 以上（模型也就 141GB） |

> ⚠️ **镜像千万别选错**：一定要选带 CUDA 的 PyTorch 镜像。普通镜像装的是 CPU 版 torch，`torch.cuda.is_available()` 会是 `False`，云 GPU 就白租了。

### 3.3 先用「无卡模式」配环境（省钱技巧）

AutoDL 有**无卡模式**：不占用 GPU 时按极低价格（约 ¥0.1/小时）计费，专门用来配环境、传数据、下模型。**大模型下载动辄几小时，用无卡模式下载能省一大笔。**

流程：

```
1. 无卡模式开机 → source /etc/network_turbo → 下载模型到数据盘
2. 关机 → 切换成正常开机（有卡）→ 跑推理
```

### 3.4 登录并部署

用 AutoDL 提供的 SSH 命令登录（或直接用 JupyterLab 的终端）：

```bash
# 1) 打开学术加速, 之后 huggingface.co 直连
source /etc/network_turbo

# 2) 上传 cloud_setup.sh 后一键部署
bash cloud_setup.sh --test
```

`cloud_setup.sh` 会自动完成：检测数据盘 → 开加速 → 校验 CUDA → 设 `HF_HOME` 到数据盘 → 装 airllm → 装 bitsandbytes（解锁 4bit 压缩加速）→ 生成 `run.py` → 冒烟测试。

### 3.5 ⚠️ 云上最大的坑：磁盘

AirLLM 需要 **2× 模型大小**的磁盘（第 3.4 节实测确认）。而云实例的**系统盘通常只有 30~50GB 可用**。

| 模型 | 权重(fp16) | AirLLM 需要 | 加 `--delete-original` |
|---|---|---|---|
| Qwen2.5-7B | 15 GB | 30 GB | 15 GB |
| Qwen2.5-32B | 64 GB | **128 GB** | 64 GB |
| Llama-3.1-70B | 141 GB | **282 GB** | 141 GB |

**所以租实例时一定要挂数据盘**（AutoDL 的 `/root/autodl-tmp`），并在 `cloud_setup.sh` 里让它自动识别。

> 磁盘不够的报错是 `MetadataIncompleteBuffer`（官方 FAQ 第一条）。脚本里会提前警告。

### 3.6 用完必须关机

按量计费**关机才停 GPU 费**。AirDL 的「关机」和「删除」不同：关机保留数据盘（数据盘可能仍按容量计费），删除则清空。

---

## 四、两个可直接照抄的方案

### 方案 A：24GB 卡 + AirLLM 跑 32B —— 真正体会 AirLLM 的价值

**为什么这个组合最能说明问题**：Qwen2.5-32B 的 fp16 权重是 **64GB**，**塞不进 24GB 显存**。这正是 AirLLM 存在的理由。

```bash
source /etc/network_turbo
cd /root/autodl-tmp/airllm-lab

# 不量化, 原始精度
python run.py --model Qwen/Qwen2.5-32B-Instruct --tokens 32 --delete-original

# 想快一点: 加 4bit (在云 GPU 上 bitsandbytes 可用, 官方称 3x 加速)
python run.py --model Qwen/Qwen2.5-32B-Instruct --tokens 32 --compression 4bit
```

预期：约 10~30 s/token（磁盘瓶颈）。够你亲眼看到「64GB 模型在 24GB 卡上跑起来」。

费用：¥2/hr × 几小时 ≈ **¥10 以内**。

### 方案 B：2× A100 80GB + vLLM 跑 70B —— 真正用起来

**这个方案不用 AirLLM。** 既然有钱租大显存，就直接把模型整个装进显存：

```bash
pip install vllm

# 141GB 权重切到 2 张 80GB 卡上 (tensor parallel)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --port 8000
```

起来之后就是一个 **OpenAI 兼容的 API**，20~40 token/s：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-70B-Instruct",
       "messages":[{"role":"user","content":"你好"}]}'
```

费用：¥15/hr 左右，但你得到的是**能用的东西**。

> 💡 想让 API 从你本地电脑访问，用 SSH 隧道：
> `ssh -L 8000:localhost:8000 -p <端口> root@<实例地址>`

---

## 五、方案对比与成本

| 方案 | 硬件 | 速度 | 小时费 | 适合 |
|---|---|---|---|---|
| 本机 + AirLLM 0.5B | 你的笔记本 | 0.4 token/s | ¥0 | 学原理（已配好） |
| **A: 4090 + AirLLM 32B** | 24GB | 10~30 s/token | ≈¥2 | **理解 AirLLM 的价值** |
| A': 4090 + AirLLM 70B | 24GB + 300GB 盘 | 60~120 s/token | ≈¥2+盘 | 见证「4GB 跑 70B」 |
| **B: 2×A100 + vLLM 70B** | 160GB | 20~40 token/s | ≈¥15 | **真正干活** |
| C: H200 141GB + vLLM 70B | 141GB | 30~50 token/s | ≈¥25 | 单卡最省心 |

> 价格为大致档位，**以平台官网实时价为准**；国内平台普遍比国际便宜。

---

## 六、云上特有注意事项

1. **磁盘比显卡更容易成为瓶颈** —— AirLLM 是磁盘密集型的，务必挂数据盘。
2. **模型下载很慢，用无卡模式下** —— 141GB 在普通家庭宽带上要十几小时；云机房带宽大得多，但仍建议无卡模式慢慢下。
3. **一定要 `source /etc/network_turbo`** —— 不做这步，`huggingface.co` 会像你本机一样超时。
4. **不要重装 torch** —— 云镜像里的 torch 是配好 CUDA 的，随便 `pip install torch` 会把它换成 CPU 版。`cloud_setup.sh` 刻意规避了这点，并会在结尾校验 `cuda=True`。
5. **bitsandbytes 在云上可用** —— 这解锁了 AirLLM 的 `--compression 4bit/8bit`（3× 加速），在你本机是拿不到的。
6. **关机 ≠ 停止计费** —— 数据盘通常按容量单独计费，长期不用就删实例或转成镜像保存。
7. **多卡要选对地区** —— AutoDL 上不是所有地区都有 2 卡以上的机器，租之前看清楚。

---

## 七、建议的学习路径

```
第 1 步 (本机, 已完成, ¥0)
    跑 Qwen2.5-0.5B, 看 --timing 输出, 读源码 _pre_hook/_post_hook
    → 理解"每 token 重读整个磁盘"这个核心代价

第 2 步 (云, ¥10 以内)
    4090 + AirLLM 跑 Qwen2.5-32B(fp16, 不进显存)
    → 亲眼看到 AirLLM 解决的真实问题

第 3 步 (云, 可选)
    同一台机器加 --compression 4bit 再跑一次, 对比速度
    → 理解"量化为什么能提速" = 减少每 token 的磁盘读取量

第 4 步 (云, ¥50 左右)
    2×A100 + vLLM 跑 Llama-3.1-70B
    → 理解为什么"真正要快, 就该把模型整个放进显存", 从而明白
      AirLLM 的定位是"没办法时的办法", 而不是日常工具
```

走完这四步，你对「显存 / 内存 / 磁盘 / 量化」这四者的权衡会有非常具体的体感——这比读十篇文章都管用。
