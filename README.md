# airllm-lab

**在一台没有独立显卡的 Windows 笔记本上，把大模型跑起来——并搞清楚它到底有多慢、为什么慢。**

这个仓库是一份完整的实战记录：从硬件摸底、读 AirLLM 源码、踩坑、实测数据，到"如果你真要用 70B 该怎么办"的云 GPU 方案。

---

## 一句话结论

> AirLLM 用「每生成一个 token 就把整个模型从磁盘重读一遍」换来了「小显存跑大模型」。
> 因此它的瓶颈**永远是磁盘，不是显卡**——换来的是「能跑」，不是「跑得快」。

---

## 实测数据（本机：i5-13420H / 15.7GB 内存 / Intel 核显 / NVMe）

| 项目 | 实测值 |
|---|---|
| 测试模型 | Qwen2.5-0.5B-Instruct（942 MB） |
| 下载速度（hf-mirror） | 1.8 MB/s |
| 切分结果 | 52 个分片文件 |
| **生成速度（热缓存）** | **2.48 s/token**（0.40 token/s） |
| **生成速度（冷缓存）** | **4.14 s/token**（0.24 token/s） |
| 裸盘顺序读 | 1.88 GiB/s |
| 有效读取吞吐 | ≈ 0.23 GB/s（碎文件开销导致，比裸盘慢 8 倍） |
| **磁盘占用** | **严格 2×** 模型大小 |

### 对 Llama-3.1-70B 的推算

| 指标 | 估算 |
|---|---|
| 权重体积（fp16） | 141 GB |
| 每 token 耗时 | **1.5 ~ 3 分钟** |
| 下载耗时（@1.8MB/s） | 约 22 小时 |
| 所需磁盘 | 282 GB（加 `--delete-original` 可降到 141 GB） |

作为对比：同一台机器上用 Ollama 跑 4bit 量化的 Qwen2.5-7B 能到 **5~15 token/s**——差距约**一千倍**。差距来源就是"量化"：把模型压到能整个塞进内存，就不必每个 token 都去读穿硬盘。

---

## 快速开始（本机 CPU）

```bash
git clone https://github.com/anhaoXie/airllm-lab.git
cd airllm-lab

# 1) 建虚拟环境并装依赖（务必用 CPU 版 torch，别装成 CUDA 版）
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install airllm

# 2) 双击 run.bat   —— 或在 PowerShell 里：
.\run.bat --model Qwen/Qwen2.5-0.5B-Instruct --tokens 16 --timing
```

首次运行会自动从 hf-mirror 下载模型并切分成 per-layer 分片（耗时较长），之后就复用分片。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| [`README-使用说明.md`](README-使用说明.md) | **主文档**。原理、配置项、实测、70B 可行性、参数表、已知风险 |
| [`CLOUD-GPU-指南.md`](CLOUD-GPU-指南.md) | 云 GPU 平台选型、AutoDL 实操、两套方案与成本对比 |
| [`run.bat`](run.bat) | 一键启动器（双击即可，自动加载环境变量） |
| [`run_airllm.py`](run_airllm.py) | 推理入口。修掉了官方示例在无 CUDA 机器上会崩的两处问题 |
| [`airllm_env.ps1`](airllm_env.ps1) | 环境变量（HF 镜像、缓存路径、线程数等） |
| [`cloud_setup.sh`](cloud_setup.sh) | 云实例一键部署脚本（检测数据盘 → 开加速 → 装依赖 → 冒烟测试） |
| [`tools/gh_relay.py`](tools/gh_relay.py) | 见下方「已知问题」——绕过被污染 hosts 的 GitHub 中继 |

---

## 相对官方示例的两处必要修改

官方示例在**没有 NVIDIA 显卡**的机器上会直接崩溃，`run_airllm.py` 里修掉了：

1. **设备硬编码**——官方用 `.cuda()`，无 N 卡时会抛异常。改为自动探测 `auto → cuda:0 / mps / cpu`，并把张量显式 `.to(device)`。
2. **构造参数**——必须显式传 `device=device`，否则 AirLLM 内部仍按 `cuda:0` 初始化。

---

## 已知问题

### Windows 上无法使用硬链接

未开启「开发者模式」时，`huggingface_hub` 无法创建符号链接，且 AirLLM 的 `link_or_copy_file()`
在 NTFS 上会退化为**真实复制**（实测硬链接数 = 1）。后果是磁盘占用严格变成 2 倍。

→ 缓解：加 `--delete-original` 删掉原始权重，或者开启 Windows 开发者模式。

### Steam++ 遗留的 hosts 屏蔽会导致 GitHub / HuggingFace 完全不可达

如果你装过 Steam++（现叫 Watt Toolkit）后来卸载了，它写入 `C:\Windows\System32\drivers\etc\hosts`
的屏蔽块会残留：把 `github.com`、`api.github.com`、`raw.githubusercontent.com`、`huggingface.co`
等域名全部指向 `127.0.0.1`，而此时它的本地代理已不再运行 → 这些域名全部变成死地址。

判断方法：

```powershell
Resolve-DnsName github.com -Type A     # 若返回 127.0.0.1 即中招
```

两种解法：

```powershell
# 解法一（推荐，需管理员）：删掉 hosts 里 "# Steam++ Start" 到 "# Steam++ End" 之间的行
notepad C:\Windows\System32\drivers\etc\hosts
ipconfig /flushdns

# 解法二（无需管理员，临时）：跑 tools/gh_relay.py，它按 TLS SNI 分流到真实 IP
python tools/gh_relay.py
```

`tools/gh_relay.py` 在 `127.0.0.1:443` 监听，读出 TLS ClientHello 里的 SNI，再连到该域名的真实 IP
并把字节原样双向转发——SNI 原样透传，所以证书校验照常通过。它**不改任何系统文件，关掉即恢复原状**。

> 注意：GitHub 的部分 IP 在本网络会被 SNI 阻断（TCP 能连上，但 TLS 握手被丢弃）。
> 该脚本内置候选 IP 列表 + 故障转移，靠"能否收到 ServerHello"来判断某个 IP 是否可用。

---

## 环境

- Windows 11 家庭版（中文）
- Intel i5-13420H（4P+4E / 8 核 12 线程），15.7 GB 内存
- **无 NVIDIA 独立显卡**，仅 Intel UHD 核显
- 单块 NVMe SSD（954 GB）
- Python 3.12.6

依赖版本：`torch 2.14.1+cpu` · `airllm 4.0.0` · `transformers 5.19.0` · `accelerate 1.15.0`

---

## 给想真正用上 70B 的人

**别用 AirLLM，用 vLLM + 大显存。** 详细方案（平台选择、价格、AutoDL 实操、SSH 隧道）见
[`CLOUD-GPU-指南.md`](CLOUD-GPU-指南.md)。核心判断：

- **想理解 AirLLM** → 租最便宜的 24GB 卡（约 ¥2/小时），跑 Qwen2.5-32B——它 64GB 的 fp16 权重塞不进 24GB 显存，这时候 AirLLM 才真正有意义
- **想真正干活** → 租 2× A100 80GB（约 ¥15/小时），用 vLLM 把模型整个装进显存，20~40 token/s
