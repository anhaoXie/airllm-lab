# ============================================================
#  AirLLM 环境变量配置  (在跑 airllm 之前 dot-source 本文件)
#      . .\airllm_env.ps1
# ============================================================

# --- 模型下载走国内镜像 (huggingface.co 在国内不可达) ---
$env:HF_ENDPOINT = "https://hf-mirror.com"

# --- 缓存放到 D 盘 (模型+切片副本动辄上百 GB，C 盘只有 203GB 剩余) ---
$env:HF_HOME = "D:\airllm-lab\hf-cache"

# --- 关闭 Xet 传输协议：hf-mirror 不支持 xet，不关会卡住/报错 ---
$env:HF_HUB_DISABLE_XET = "1"

# --- 关闭遥测，加快冷启动 ---
$env:HF_HUB_DISABLE_TELEMETRY = "1"

# --- 关闭下载进度条：进度条走 stderr，会让 PowerShell 误判命令失败(exit 1) ---
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

# --- 抑制 Windows 无符号链接权限的告警 (不影响功能) ---
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# --- 让 tokenizers 不要起并行线程，避免小内存机器上 fork 报错 ---
$env:TOKENIZERS_PARALLELISM = "false"

# --- PyTorch CPU 线程数 = 物理核数 (i5-13420H: 4P+4E = 8 核 12 线程) ---
$env:OMP_NUM_THREADS = "8"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# 自检
Write-Host "HF_ENDPOINT          = $env:HF_ENDPOINT"
Write-Host "HF_HOME              = $env:HF_HOME"
Write-Host "HF_HUB_DISABLE_XET   = $env:HF_HUB_DISABLE_XET"
Write-Host "OMP_NUM_THREADS      = $env:OMP_NUM_THREADS"
