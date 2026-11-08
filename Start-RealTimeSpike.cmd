@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m realtimespike %*
) else (
  py -3.12 -m realtimespike %*
)
