#!/usr/bin/env bash
# =============================================================================
#  AirLLM 云 GPU 一键部署脚本
#  适用: AutoDL / 恒源云 / 矩池云 / RunPod / Vast.ai 等 Linux + NVIDIA 实例
#
#  用法:
#     bash cloud_setup.sh              # 只装环境, 不跑推理
#     bash cloud_setup.sh --test       # 装完立刻跑一个冒烟测试
#     MODEL=Qwen/Qwen2.5-7B-Instruct bash cloud_setup.sh --test
#
#  重要: 云镜像里通常已预装 CUDA 版 torch, 本脚本刻意不重装 torch.
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN_TEST=0
[[ "${1:-}" == "--test" ]] && RUN_TEST=1

echo "============================================================"
echo " AirLLM 云 GPU 部署"
echo "============================================================"

# ---------------------------------------------------------------------------
# [1/6] 选择存放模型的数据盘
#       镜像系统盘通常只有 30~50GB, 而 AirLLM 需要 2x 模型大小的磁盘
# ---------------------------------------------------------------------------
echo "[1/6] 检测数据盘..."
CANDIDATES=(/root/autodl-tmp /root/autodl-tmp2 /workspace /hy-tmp /root/data "$HOME")
DATA_DIR=""
for d in "${CANDIDATES[@]}"; do
    if [[ -d "$d" && -w "$d" ]]; then
        avail=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
        if [[ "${avail:-0}" -gt 50 ]]; then
            DATA_DIR="$d"; break
        fi
    fi
done
DATA_DIR="${DATA_DIR:-$HOME}"
AIRLLM_ROOT="$DATA_DIR/airllm-lab"
mkdir -p "$AIRLLM_ROOT"
FREE_GB=$(df -BG --output=avail "$AIRLLM_ROOT" | tail -1 | tr -dc '0-9')
echo "      数据目录: $AIRLLM_ROOT  (可用 ${FREE_GB}GB)"

# ---------------------------------------------------------------------------
# [2/6] 学术网络加速 (AutoDL 内置, 直接打通 huggingface.co)
# ---------------------------------------------------------------------------
echo "[2/6] 检测学术加速..."
if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo 2>/dev/null || true
    echo "      已启用 /etc/network_turbo -> huggingface.co 可直连"
else
    # 非 AutoDL 平台: 用国内镜像兜底
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    echo "      无内置加速, 改用镜像: $HF_ENDPOINT"
fi

# ---------------------------------------------------------------------------
# [3/6] 检查 GPU / CUDA / torch
# ---------------------------------------------------------------------------
echo "[3/6] 检查 GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/      /'
else
    echo "      !! 没找到 nvidia-smi, 这台机器可能没有 GPU"
fi

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

"$PY" - <<'PYCHECK'
import sys
try:
    import torch
    print(f"      torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    else:
        print("      !! CUDA 不可用 —— AirLLM 会退化成纯 CPU, 云 GPU 就白租了")
except ImportError:
    print("      !! 未安装 torch")
    sys.exit(3)
PYCHECK

# ---------------------------------------------------------------------------
# [4/6] 环境变量 (只影响当前 shell 及其子进程)
# ---------------------------------------------------------------------------
echo "[4/6] 配置环境变量..."
export HF_HOME="$AIRLLM_ROOT/hf-cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
echo "      HF_HOME=$HF_HOME"

# 持久化, 方便下次登录直接可用
PROFILE_LINE="export HF_HOME=$AIRLLM_ROOT/hf-cache HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false"
grep -qF "$PROFILE_LINE" "$HOME/.bashrc" 2>/dev/null || echo "$PROFILE_LINE" >> "$HOME/.bashrc"

# ---------------------------------------------------------------------------
# [5/6] 安装 airllm (刻意不升级 torch, 避免毁掉镜像里的 CUDA wheel)
# ---------------------------------------------------------------------------
echo "[5/6] 安装 airllm..."
"$PY" -m pip install -q --upgrade pip 2>/dev/null || true
"$PY" -m pip install -q airllm

"$PY" - <<'PYVER'
import torch, transformers, airllm, importlib.metadata as md
print(f"      airllm       {md.version('airllm')}")
print(f"      transformers {transformers.__version__}")
print(f"      torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
PYVER

# compression (4bit/8bit) 需要 bitsandbytes + NVIDIA GPU, 有卡就能用 -> 3x 加速
if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "      检测到 CUDA, 额外安装 bitsandbytes (解锁 compression='4bit' 的 3x 加速)..."
    "$PY" -m pip install -q bitsandbytes 2>/dev/null && echo "      bitsandbytes 安装完成" \
        || echo "      bitsandbytes 安装失败, compression 不可用 (不影响正常推理)"
fi

# ---------------------------------------------------------------------------
# [6/6] 生成运行脚本
# ---------------------------------------------------------------------------
echo "[6/6] 生成推理脚本..."
cat > "$AIRLLM_ROOT/run.py" <<'PYRUN'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AirLLM 云 GPU 推理脚本。用法:
     python run.py --model Qwen/Qwen2.5-7B-Instruct --tokens 64
     python run.py --model Qwen/Qwen2.5-7B-Instruct --compression 4bit   # 3x 加速
"""
import argparse, os, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from airllm import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--max-length", type=int, default=128)
ap.add_argument("--compression", default=None, choices=[None, "4bit", "8bit"])
ap.add_argument("--delete-original", action="store_true", help="省一半磁盘")
ap.add_argument("--no-prefetch", action="store_true")
a = ap.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("=" * 64)
print(f"模型 {a.model}")
print(f"设备 {device}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU  {p.name}  {p.total_memory/1024**3:.1f}GB")
print(f"compression {a.compression or 'None (原始精度, 不量化)'}")
print("=" * 64, flush=True)

t0 = time.time()
model = AutoModel.from_pretrained(
    a.model, device=device, max_seq_len=a.max_length,
    compression=a.compression, prefetching=not a.no_prefetch,
    delete_original=a.delete_original,
)
print(f"\n[载入+切分] {time.time()-t0:.1f}s\n", flush=True)

tok = model.tokenizer([a.prompt], return_tensors="pt", return_attention_mask=False,
                      truncation=True, max_length=a.max_length, padding=False)
ids = tok["input_ids"].to(device)

t1 = time.time()
out = model.generate(ids, max_new_tokens=a.tokens, use_cache=True, return_dict_in_generate=True)
el = time.time() - t1
n = out.sequences[0].shape[0] - ids.shape[1]

print("-" * 64)
print(model.tokenizer.decode(out.sequences[0]))
print("-" * 64)
print(f"{n} tokens / {el:.1f}s = {el/max(n,1):.2f} s/token  ({n/max(el,1e-9):.3f} token/s)")
PYRUN

echo
echo "============================================================"
echo " 部署完成"
echo "============================================================"
echo "  数据目录 : $AIRLLM_ROOT"
echo "  运行脚本 : $AIRLLM_ROOT/run.py"
echo
echo "  下一步:"
echo "    cd $AIRLLM_ROOT"
echo "    python run.py --model Qwen/Qwen2.5-7B-Instruct --tokens 64"
echo
if [[ "${FREE_GB:-0}" -lt 320 ]]; then
    echo "  !! 警告: 当前可用磁盘只有 ${FREE_GB}GB"
    echo "     AirLLM 需要 2x 模型大小 (70B fp16 约 141GB -> 需 282GB)"
    echo "     磁盘不够时请加 --delete-original, 或扩容数据盘"
fi
echo "============================================================"

# ---------------------------------------------------------------------------
if [[ "$RUN_TEST" == "1" ]]; then
    echo
    echo ">>> 冒烟测试: $MODEL"
    cd "$AIRLLM_ROOT"
    "$PY" run.py --model "$MODEL" --tokens 16
fi
