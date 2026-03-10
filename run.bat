@echo off
chcp 65001 >nul
title ARIAKE_CVI

cd /d "%~dp0"
echo Launching ARIAKE_CVI...
echo (ブラウザを閉じると、このウィンドウも自動的に閉じます)

REM 直接 streamlit を呼ぶのではなく、ラッパーを python で実行する
python app_launcher.py

REM pythonが終了（os.kill）すれば、ここへ戻ってきます
exit