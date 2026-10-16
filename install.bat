@echo off
cd /d "%~dp0"
where python >nul 2>&1 || (echo Python 3.10-3.12 not found. Install from python.org and tick "Add to PATH". & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo [1/3] Installing PyTorch with CUDA (needs only the NVIDIA driver, NOT the CUDA Toolkit)...
pip install torch --index-url https://download.pytorch.org/whl/cu128 || (echo torch install failed & exit /b 1)
echo [2/3] Installing Airgap dependencies...
pip install -r requirements.txt || (echo pip install failed & exit /b 1)
if not exist .env copy env.example .env >nul
echo [3/3] Checking GPU and downloading TabPFN weights (one-time, needs internet)...
python check_gpu.py || echo. & echo [!] GPU/weights check FAILED. Fix the message above, then run:  .venv\Scripts\activate ^& python check_gpu.py
echo.
echo Done. Double-click run.bat to start.
pause
