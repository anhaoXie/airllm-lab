#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AirLLM 运行脚本 —— 已适配「无 NVIDIA 显卡 / 纯 CPU」的机器。

用法:
    python run_airllm.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "你好，介绍一下你自己"
    python run_airllm.py --model Qwen/Qwen2.5-0.5B-Instruct --tokens 40 --timing

关键点说明:
  * AirLLM 默认 device="cuda:0"，本机没有 CUDA，必须显式传 device="cpu"。
  * 官方 README 里的 input_tokens['input_ids'].cuda() 在 CPU 机器上会直接报错，
    这里改成 .to(device)。
  * 首次运行会把整个 checkpoint「按层切分」再存一份，所以需要约 2 倍模型大小的磁盘。
    用 delete_original=True 可以在切分后删掉原始下载，省一半空间。
"""
import argparse
import os
import sys
import time

# ---- 环境兜底：即使忘了 dot-source airllm_env.ps1 也能跑 ----
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\airllm-lab\hf-cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from airllm import AutoModel  # noqa: E402


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run AirLLM inference (CPU/GPU auto-adapting).")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HuggingFace repo id 或本地路径")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--tokens", type=int, default=32, help="max_new_tokens")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | mps")
    ap.add_argument("--shards-path", default=None,
                    help="切分后按层存放的目录 (默认放 HF cache 旁边)")
    ap.add_argument("--delete-original", action="store_true",
                    help="切分完成后删除原始 checkpoint，省一半磁盘")
    ap.add_argument("--compression", default=None, choices=[None, "4bit", "8bit"],
                    help="块状量化加速 (需要 bitsandbytes + NVIDIA GPU，本机不可用)")
    ap.add_argument("--no-prefetch", action="store_true", help="关闭预取")
    ap.add_argument("--timing", action="store_true", help="打印每 token 耗时")
    args = ap.parse_args()

    device = pick_device(args.device)
    print("=" * 64)
    print(f"模型          : {args.model}")
    print(f"设备          : {device}")
    print(f"torch         : {torch.__version__}   cuda_available={torch.cuda.is_available()}")
    print(f"HF_ENDPOINT   : {os.environ.get('HF_ENDPOINT')}")
    print(f"HF_HOME       : {os.environ.get('HF_HOME')}")
    print(f"预取 prefetch : {not args.no_prefetch}")
    print(f"压缩 compress : {args.compression or 'None (不量化，原始精度)'}")
    print("=" * 64, flush=True)

    if args.compression and not torch.cuda.is_available():
        print("[!] compression 依赖 bitsandbytes，且需要 NVIDIA GPU；本机没有，忽略该选项。")
        args.compression = None

    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.model,
        device=device,
        max_seq_len=args.max_length,
        layer_shards_saving_path=args.shards_path,
        prefetching=not args.no_prefetch,
        delete_original=args.delete_original,
        compression=args.compression,
    )
    print(f"\n[load+split] 耗时 {time.time() - t0:.1f}s\n", flush=True)

    input_tokens = model.tokenizer(
        [args.prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=args.max_length,
        padding=False,
    )
    input_ids = input_tokens["input_ids"].to(device)  # 注意: 不是 .cuda()

    print(f"[prompt] {args.prompt}\n", flush=True)
    t1 = time.time()
    generation_output = model.generate(
        input_ids,
        max_new_tokens=args.tokens,
        use_cache=True,
        return_dict_in_generate=True,
    )
    elapsed = time.time() - t1

    out = model.tokenizer.decode(generation_output.sequences[0])
    n_new = generation_output.sequences[0].shape[0] - input_ids.shape[1]

    print("-" * 64)
    print(out)
    print("-" * 64)
    print(f"总耗时      : {elapsed:.1f}s")
    if n_new > 0:
        print(f"生成 token  : {n_new}")
        print(f"平均        : {elapsed / n_new:.2f} s/token   ({n_new / elapsed:.3f} token/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
