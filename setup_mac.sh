#!/bin/bash
# setup_mac.sh — 既配布 ARIAKE_CVI フォルダ向け（ローカル Python 3.12 + venv）
# 使い方: このスクリプトを main_st.py と同じ階層に置き、bash setup_mac.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=========================================="
echo " ARIAKE_CVI macOS セットアップ"
echo " 作業ディレクトリ: $ROOT"
echo "=========================================="
echo ""

# --- 必須ファイル確認 ---
if [[ ! -f "$ROOT/main_st.py" ]] || [[ ! -f "$ROOT/requirements.txt" ]]; then
	echo "エラー: main_st.py または requirements.txt が見つかりません。"
	echo "このスクリプトは、配布フォルダ（main_st.py と同じ階層）に置いて実行してください。"
	exit 1
fi

# --- Python 3.12 検出 ---
PYTHON312=""
if command -v python3.12 >/dev/null 2>&1; then
	PYTHON312="$(command -v python3.12)"
elif [[ -x /opt/homebrew/bin/python3.12 ]]; then
	PYTHON312="/opt/homebrew/bin/python3.12"
elif [[ -x /usr/local/bin/python3.12 ]]; then
	PYTHON312="/usr/local/bin/python3.12"
elif [[ -x /opt/homebrew/opt/python@3.12/bin/python3.12 ]]; then
	PYTHON312="/opt/homebrew/opt/python@3.12/bin/python3.12"
elif [[ -x /usr/local/opt/python@3.12/bin/python3.12 ]]; then
	PYTHON312="/usr/local/opt/python@3.12/bin/python3.12"
fi

if [[ -z "$PYTHON312" ]]; then
	echo "エラー: python3.12 が見つかりません。"
	echo ""
	echo "Homebrew で Python 3.12 をインストールしてください。"
	echo ""
	echo "  # Apple Silicon (M1/M2/M3/M4)"
	echo "  brew install python@3.12"
	echo "  echo 'export PATH=\"/opt/homebrew/opt/python@3.12/bin:\$PATH\"' >> ~/.zprofile"
	echo "  source ~/.zprofile"
	echo ""
	echo "  # Intel Mac"
	echo "  brew install python@3.12"
	echo "  echo 'export PATH=\"/usr/local/opt/python@3.12/bin:\$PATH\"' >> ~/.zprofile"
	echo "  source ~/.zprofile"
	echo ""
	echo "インストール後、もう一度このスクリプトを実行してください:"
	echo "  bash setup_mac.sh"
	exit 1
fi

echo "使用する Python: $PYTHON312 ($("$PYTHON312" --version 2>&1))"
echo ""

# --- 古い venv を削除 ---
if [[ -d "$ROOT/venv" ]]; then
	echo "既存の venv/ を削除します..."
	rm -rf "$ROOT/venv"
fi

# --- venv 作成 ---
echo "venv を作成しています..."
"$PYTHON312" -m venv "$ROOT/venv"

# --- pip / requirements ---
echo "pip をアップグレードしています..."
"$ROOT/venv/bin/python" -m pip install --upgrade pip

echo "requirements.txt をインストールしています（数分かかることがあります）..."
"$ROOT/venv/bin/python" -m pip install -r "$ROOT/requirements.txt"

# --- 検疫属性の解除（.app） ---
if [[ -d "$ROOT/ARIAKE_CVI.app" ]]; then
	echo "ARIAKE_CVI.app の検疫属性を解除しています..."
	xattr -cr "$ROOT/ARIAKE_CVI.app" 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo " セットアップ完了"
echo "=========================================="
echo ""
echo "起動方法（どちらか）:"
echo ""
echo "  1) アプリから起動"
echo "       open \"$ROOT/ARIAKE_CVI.app\""
echo "     または Finder で ARIAKE_CVI.app をダブルクリック"
echo ""
echo "  2) ターミナルから起動"
echo "       cd \"$ROOT\""
echo "       source venv/bin/activate"
echo "       streamlit run main_st.py"
echo ""
echo "Gatekeeper で「開けない」と出る場合:"
echo "  ・ARIAKE_CVI.app を右クリック →「開く」"
echo "  ・または システム設定 → プライバシーとセキュリティ →「このまま開く」"
echo ""
