@echo off
chcp 65001 >nul
title 画像ROI選択ツール

REM カレントディレクトリを移動
cd /d "%~dp0"

REM Streamlitを起動
echo 起動中... ブラウザが自動で開きます
echo このウィンドウを閉じるとアプリも終了します
echo.
streamlit run app.py

pause