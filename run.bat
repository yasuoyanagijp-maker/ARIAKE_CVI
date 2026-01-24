@echo off
chcp 65001 >nul
title ARIAKE_CVI

REM カレントディレクトリを移動
cd /d "%~dp0"

REM Streamlitを起動
echo 起動中... ブラウザが自動で開きます
echo このウィンドウを閉じるとアプリも終了します
echo.
streamlit run main_st.py

pause