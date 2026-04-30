@echo off
REM ============================================================
REM vLLM does NOT run on native Windows.
REM Please run start_vllm.sh inside WSL2 (Windows Subsystem for Linux):
REM
REM   wsl bash /path/to/local/start_vllm.sh
REM
REM Or run directly in a Linux terminal:
REM
REM   bash local/start_vllm.sh
REM ============================================================

echo.
echo ERROR: vLLM requires Linux. Please use WSL2 to run start_vllm.sh
echo.
echo   wsl bash local/start_vllm.sh
echo.
pause
