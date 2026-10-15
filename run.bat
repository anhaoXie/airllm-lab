@echo off
rem ============================================================
rem  AirLLM one-click launcher
rem  Extra args are passed through to run_airllm.py, e.g.:
rem     run.bat --model Qwen/Qwen2.5-1.5B-Instruct --tokens 16
rem ============================================================
setlocal

rem ---- defaults (overridable by CLI args) ----
set "MODEL=Qwen/Qwen2.5-0.5B-Instruct"
set "PROMPT=What is the capital of France? Answer in one sentence."
set "TOKENS=32"

rem ---- environment ----
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HOME=D:\airllm-lab\hf-cache"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "TOKENIZERS_PARALLELISM=false"
set "OMP_NUM_THREADS=8"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo  MODEL  : %MODEL%
echo  PROMPT : %PROMPT%
echo  TOKENS : %TOKENS%
echo ============================================================
echo.

"D:\airllm-lab\.venv\Scripts\python.exe" "D:\airllm-lab\run_airllm.py" --model "%MODEL%" --prompt "%PROMPT%" --tokens %TOKENS% --timing %*

echo.
echo ============================================================
echo  exit code: %ERRORLEVEL%
echo ============================================================
pause
