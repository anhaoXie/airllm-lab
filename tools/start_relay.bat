@echo off
REM ============================================================
REM  启动 GitHub/HuggingFace 中继
REM
REM  为什么需要它:
REM    你装过 Steam++ (Watt Toolkit), 它在 hosts 里把 github.com /
REM    api.github.com / huggingface.co 等 47 个域名指向 127.0.0.1,
REM    指望自己的本地代理在 443 端口接住。但它的 HTTPS 加速器
REM    (Steam++.Accelerator.exe) 只监听了 80 和 444 端口, 443 没人接,
REM    于是这些域名全部变成死地址。
REM
REM  这个中继就补上那个缺失的 443 监听: 读出 TLS 握手里的域名,
REM  连到真实 IP 原样转发。SNI 原样透传, 证书校验照常通过。
REM
REM  用法: 双击本文件。保持这个窗口开着, 关掉就失效。
REM  停止: 在这个窗口按 Ctrl+C
REM ============================================================
setlocal

set "PY="
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.8+ or create ..\.venv
    pause
    exit /b 1
)

echo Starting relay with: %PY%
echo.
"%PY%" -u "%~dp0gh_relay.py"

echo.
echo Relay stopped.
pause
