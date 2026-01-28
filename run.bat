@echo off
chcp 65001 >nul
title ARIAKE_CVI

REM Launch Streamlit
cd /d "%~dp0"
echo Launching... The browser will open automatically.
echo If you close this window, the app will also exit.
echo.
streamlit run main_st.py
pause