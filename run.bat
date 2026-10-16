@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if exist tabpfn_token.txt set /p TABPFN_TOKEN=<tabpfn_token.txt
set TABPFN_NO_BROWSER=1
if exist .weights_ready set HF_HUB_OFFLINE=1
set TABPFN_DISABLE_TELEMETRY=1
set LLM_MODEL=qwen3:8b
start http://127.0.0.1:8000
python server.py
pause