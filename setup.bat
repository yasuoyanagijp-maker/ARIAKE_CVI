@echo off
chcp 65001 >nul
title 画像ROI選択ツール - セットアップ

echo ╔════════════════════════════════════════╗
echo ║  画像ROI選択ツール - 初期セットアップ  ║
echo ╚════════════════════════════════════════╝
echo.

REM Pythonの確認
echo [1/4] Pythonを確認中...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Pythonが見つかりません
    echo.
    echo 以下のURLからPythonをインストールしてください:
    echo https://www.python.org/downloads/
    echo.
    echo インストール時に「Add Python to PATH」にチェックを入れてください
    pause
    exit /b 1
)
python --version
echo ✅ Pythonが見つかりました
echo.

REM 依存関係のインストール
echo [2/4] 必要なパッケージをインストール中...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ❌ パッケージのインストールに失敗しました
    pause
    exit /b 1
)
echo ✅ パッケージのインストール完了
echo.

REM デスクトップショートカットの作成
echo [3/4] デスクトップショートカットを作成中...
python create_shortcut.py
if %errorlevel% neq 0 (
    echo ⚠️  ショートカット作成に失敗しました（手動で run.bat を起動してください）
) else (
    echo ✅ デスクトップにショートカットを作成しました
)
echo.

REM 完了確認
echo [4/4] 完了確認...
echo ✅ セットアップが完了しました！
echo.

echo ╔════════════════════════════════════════╗
echo ║           セットアップ完了！           ║
echo ╚════════════════════════════════════════╝
echo.
echo 次回からは以下の方法で起動できます:
echo  • デスクトップの「画像ROIツール」アイコンをダブルクリック
echo  • または、run.bat をダブルクリック
echo.
pause