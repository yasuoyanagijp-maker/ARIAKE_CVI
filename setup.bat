@echo off
chcp 65001 >nul
title ARIAKE_CVI - Setup

echo ========================================
echo   ARIAKE_CVI - Initial Setup
echo ========================================
echo.

REM Check Python
echo([1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found
    echo.
    echo Please install Python from the following URL:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo [OK] Python found
echo.

REM Install dependencies
echo([2/4] Installing required packages...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)
echo [OK] Package installation complete
echo.

REM Create desktop shortcut
echo([3/4] Creating desktop shortcut...
python create_shortcut.py
if %errorlevel% neq 0 (
    echo [WARN] Failed to create shortcut (please run run.bat manually)
) else (
    echo [OK] Desktop shortcut created
)
echo.

REM Final check
echo([4/4] Final check...
echo [OK] Setup is complete!
echo.

echo ========================================
echo           Setup Complete!
echo ========================================
echo.
echo You can start the application by:
echo  • Double-clicking the "ARIAKE_CVI" icon on your desktop
echo  • Or double-clicking run.bat
echo.
pause