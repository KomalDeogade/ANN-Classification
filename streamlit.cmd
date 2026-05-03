@echo off
cd /d "%~dp0"
"%~dp0venv\python.exe" -m streamlit %*
