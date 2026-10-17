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
| [`tools/gh_relay.py`](tools/gh_relay.py) | 见下方「已知问题」——补上 Steam++ 缺失的 443 监听，让 GitHub 恢复可用 |
| [`tools/start_relay.bat`](tools/start_relay.bat) | 上面那个中继的一键启动器（双击即可） |

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

### Steam++ 的 hosts 屏蔽 + 加速器卡死 → GitHub 全灭（重启 Steam++ 即恢复）

> **2026-10-17 定位到真因：不是它天生坏，而是它的加速器会「卡死」。**
> 卡死时 `Steam++.Accelerator.exe` 不监听 443，它写的 47 条 hosts 就全指到死端口 → GitHub 全灭。
> **退出 Steam++ 再重新打开**（或杀掉三个 `Steam++*` 进程后重开）即可恢复：重启后 443 恢复监听，
> 实测 `github.com` HTTP 200 1.19s、`api.github.com` 200 0.21s、`git push` 成功。
> 所以遇到「GitHub 突然全打不开」时，**先重启 Steam++**，别急着重装或怀疑被墙。
>
> 两个踩过的坑记在这：
> 1. **`Set-Content -LiteralPath` 在这台机器上会静默失败**（`$ErrorActionPreference='Continue'`
>    把异常吞了，表现是"报写入成功但文件没变"）。改用
>    `[System.IO.File]::WriteAllLines($hp, [string[]]$lines, [System.Text.Encoding]::ASCII)` 才成功。
> 2. **给 `powershell.exe`（5.1）的 .ps1 必须纯 ASCII**。UTF-8 无 BOM 的脚本在中文系统上会被
>    按 GBK 解码，中文字符串里的字节会吃掉引号 → 解析报错。`pwsh`(PS7) 才默认识 UTF-8 无 BOM。
>
> 另外确认：**火绒（Huorong）没拦 hosts**，它的 ACL 也允许管理员写（`BUILTIN\Administrators`
> 是 FullControl）。hosts 写不进去纯粹是上面第 1 条那个工具问题。
> 注意 Steam++ 运行时会**实时把 hosts 写回自己的版本**，所以要改 hosts 必须先退出它。

这是本项目排查最久、最反直觉的一个坑。现象是 `github.com` 完全打不开，看着像被墙。

**实际机制**：装着 Steam++（现叫 Watt Toolkit）时，它会在
`C:\Windows\System32\drivers\etc\hosts` 写入一个屏蔽块，把 47 个域名（GitHub 全家、
`huggingface.co`、`hub.docker.com`、Steam 社区等）指向 `127.0.0.1`，
指望自己的本地反向代理在 **443** 端口接住并解密转发（它确实往系统里装了
`CN=SteamTools Certificate, O=BeyondDimension` 的根证书，就是干这个的）。

**它会卡死**：正常时 `Steam++.Accelerator.exe` 监听 80/443/444；一旦卡死，**443 就没人监听**。
此时这 47 个域名全部指向一个没人应答的端口 → 全部变成死地址，看起来就像被墙了。
本机遇到过两次这个故障，**重启 Steam++ 即恢复**。

判断方法：

```powershell
Resolve-DnsName github.com -Type A                                          # 返回 127.0.0.1 即中招
Get-NetTCPConnection -State Listen | Where-Object LocalPort -in 80,443,444  # 看谁在听哪个端口
```

**注意**：`huggingface.co` 是**额外**真有 GFW 封锁的（换任何真实 IP 直连都是 TLS 重置或超时），
所以它必须走镜像，别指望中继能救。GitHub 没被墙，只是被这个坏 hosts 拖累。

两种解法：

```powershell
# 解法一（最省事）：重启 Steam++ —— 退出后重新打开。
#   恢复后 443 会被它自己的加速器接住, GitHub 立刻可用。
#   验证: Get-NetTCPConnection -LocalPort 443 -State Listen

# 解法二（不用 Steam++）：退出它, 再把 hosts 里 "# Steam++ Start" 到 "# Steam++ End"
#   之间的行删掉, 然后 ipconfig /flushdns。
#   注意两点:
#     a) 它运行时会把 hosts 写回去, 必须先退干净 (三个进程都杀掉)
#     b) 直连 github.com 在本网络是间歇性的 (TLS 握手常被丢),
#        git push 这种长连接更容易断, 需要重试或改用解法三

# 解法三（兜底, 不改系统文件）：跑 tools/gh_relay.py 补上 443 监听
#   双击 tools\start_relay.bat 即可; 配合解法二留下的 hosts 条目使用
```

`tools/gh_relay.py` 在 `127.0.0.1:443` 监听，读出 TLS ClientHello 里的 SNI，再连到该域名的真实 IP
并把字节原样双向转发——SNI 原样透传，所以证书校验照常通过。它**不改任何系统文件，关掉即恢复原状**，
而且和 Steam++ 的 hosts 条目天然互补（那些条目说"去 127.0.0.1"，中继正好在那儿接住）。

> **两个实测出来的坑**：
> 1. GitHub 的部分 IP 在本网络会被 SNI 阻断（TCP 能连上，但 TLS 握手被丢弃）。
>    实测可用：`140.82.113.3` / `140.82.114.3` / `140.82.116.3` / `140.82.121.3` / `20.27.177.113` / `4.208.26.197`；
>    不可用：`140.82.112.4`、`20.205.243.166`。脚本内置候选列表 + 故障转移，
>    靠"能否收到 ServerHello"判断某个 IP 是否可用。
> 2. 脚本必须**拒绝回环地址**：hosts 里那些未被路由的域名（如 `store.steampowered.com`）
>    解析后仍是 `127.0.0.1`，若直连就会打到中继自己，每层再开一个连接，**无限递归**。


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
