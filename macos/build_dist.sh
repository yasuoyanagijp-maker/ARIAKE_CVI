#!/bin/bash
# ARIAKE_CVI — macOS 配布用ビルド
# 埋め込み Python 3.12 に依存関係を直接インストールし、配布先に
# Homebrew / pyenv / 開発用 venv を不要にする。
#
# 使い方:
#   cd /path/to/ARIAKE_CVI-1
#   bash macos/build_dist.sh
#
# 成果物:
#   dist/ARIAKE_CVI-macos-<arch>-v<version>.zip
#   （展開して ARIAKE_CVI.app を起動。runtime/ は同フォルダ必須）

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' ARIAKE_CVI.app/Contents/Info.plist 2>/dev/null || echo '1.0')"
ARCH_UNAME="$(uname -m)"
case "$ARCH_UNAME" in
  arm64)  PBS_ARCH="aarch64-apple-darwin" ;;
  x86_64) PBS_ARCH="x86_64-apple-darwin" ;;
  *)
    echo "未対応アーキテクチャ: $ARCH_UNAME"
    exit 1
    ;;
esac

PYTHON_TAG="${PYTHON_TAG:-20260718}"
PYTHON_VER="${PYTHON_VER:-3.12.13}"
ASSET="cpython-${PYTHON_VER}+${PYTHON_TAG}-${PBS_ARCH}-install_only.tar.gz"
ASSET_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_TAG}/${ASSET}"

RUNTIME_DIR="$ROOT/runtime"
PYTHON_DIR="$RUNTIME_DIR/python"
PY="$PYTHON_DIR/bin/python3.12"
DIST_DIR="$ROOT/dist"
STAGE_NAME="ARIAKE_CVI"
STAGE_DIR="$DIST_DIR/$STAGE_NAME"
ZIP_NAME="ARIAKE_CVI-macos-${ARCH_UNAME}-v${VERSION}.zip"

echo "==> プロジェクト: $ROOT"
echo "==> 対象: macOS $ARCH_UNAME / Python $PYTHON_VER (standalone, no Homebrew)"
echo "==> 成果物: dist/$ZIP_NAME"

# --- 埋め込み Python（venv は使わない＝配布先でパスが変わっても壊れにくい）---
if [[ ! -x "$PY" ]]; then
  echo "==> Standalone Python を取得: $ASSET"
  mkdir -p "$RUNTIME_DIR"
  TMP_TGZ="$(mktemp -t ariake-cpython).tar.gz"
  curl -fL --retry 3 -o "$TMP_TGZ" "$ASSET_URL"
  rm -rf "$PYTHON_DIR"
  tar -xzf "$TMP_TGZ" -C "$RUNTIME_DIR"
  rm -f "$TMP_TGZ"
  if [[ ! -x "$PY" && -x "$RUNTIME_DIR/bin/python3.12" ]]; then
    mkdir -p "$PYTHON_DIR"
    mv "$RUNTIME_DIR/bin" "$RUNTIME_DIR/lib" "$RUNTIME_DIR/include" "$PYTHON_DIR/" 2>/dev/null || true
    mv "$RUNTIME_DIR/share" "$PYTHON_DIR/" 2>/dev/null || true
  fi
  test -x "$PY"
  echo "    OK: $($PY --version)"
else
  echo "==> 既存の埋め込み Python を使用: $($PY --version)"
fi

# 古い配布用 venv が残っていれば削除（混乱防止）
if [[ -d "$RUNTIME_DIR/venv" ]]; then
  echo "==> 旧 runtime/venv を削除"
  rm -rf "$RUNTIME_DIR/venv"
fi

echo "==> 埋め込み Python に requirements をインストール（数分〜十数分）"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -r "$ROOT/requirements.txt"
"$PY" -c "import streamlit, torch; print('streamlit', streamlit.__version__); print('torch', torch.__version__)"

# --- .app 起動スクリプト（runtime/python 優先）---
echo "==> ARIAKE_CVI.app 起動スクリプトを更新"
cat > "$ROOT/ARIAKE_CVI.app/Contents/MacOS/launch" <<'LAUNCH'
#!/bin/bash
# ARIAKE_CVI.app — 配布用 runtime/python 優先、なければ開発用 ./venv
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/runtime/python/bin/python3.12" ]]; then
	PYTHON="$PROJECT_ROOT/runtime/python/bin/python3.12"
elif [[ -x "$PROJECT_ROOT/runtime/venv/bin/python" ]]; then
	PYTHON="$PROJECT_ROOT/runtime/venv/bin/python"
elif [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
	PYTHON="$PROJECT_ROOT/venv/bin/python"
else
	PYTHON=""
fi

LAUNCHER="$PROJECT_ROOT/app_launcher.py"
LOG_DIR="$HOME/Library/Logs"
LOG_FILE="$LOG_DIR/ARIAKE_CVI.log"

alert() {
	osascript -e "display alert \"ARIAKE_CVI\" message \"$1\" as warning" 2>/dev/null || true
}

if [[ -z "$PYTHON" ]]; then
	alert "実行環境が見つかりません。

配布 ZIP を展開したフォルダ一式（ARIAKE_CVI.app と runtime/）が揃っているか確認してください。

開発者向け再ビルド:
cd \"$PROJECT_ROOT\"
bash macos/build_dist.sh"
	exit 1
fi

if [[ ! -f "$LAUNCHER" ]]; then
	alert "app_launcher.py が見つかりません: $LAUNCHER"
	exit 1
fi

mkdir -p "$LOG_DIR"

if lsof -iTCP:8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
	open "http://localhost:8501"
	exit 0
fi

osascript <<APPLESCRIPT
tell application "Terminal"
	activate
	do script "cd $(printf '%q' "$PROJECT_ROOT") && clear && echo 'ARIAKE_CVI を起動しています...' && echo 'ブラウザをすべて閉じるとサーバーが終了します。' && echo 'ログ: $LOG_FILE' && echo '' && exec $(printf '%q' "$PYTHON") $(printf '%q' "$LAUNCHER") 2>&1 | tee -a $(printf '%q' "$LOG_FILE")"
end tell
APPLESCRIPT
LAUNCH
chmod +x "$ROOT/ARIAKE_CVI.app/Contents/MacOS/launch"

echo "==> codesign (ad-hoc) + xattr クリア"
xattr -cr "$ROOT/ARIAKE_CVI.app" 2>/dev/null || true
xattr -cr "$ROOT/runtime" 2>/dev/null || true
codesign --force --deep --sign - "$ROOT/ARIAKE_CVI.app" 2>/dev/null || true

echo "==> 配布 ZIP を作成"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

rsync -a \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude 'venv/' \
  --exclude 'dist/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.devcontainer/' \
  --exclude 'runtime/venv/' \
  "$ROOT/ARIAKE_CVI.app" \
  "$ROOT/runtime" \
  "$ROOT/app_launcher.py" \
  "$ROOT/main_st.py" \
  "$ROOT/requirements.txt" \
  "$ROOT/README.md" \
  "$ROOT/choroid_radiomics" \
  "$ROOT/deepgpet" \
  "$ROOT/macos" \
  "$ROOT/icon.ico" \
  "$STAGE_DIR/"

cat > "$STAGE_DIR/起動方法.txt" <<EOF
ARIAKE_CVI（macOS）

1. このフォルダごと任意の場所に置く（中の runtime/ を消さない・.app だけ移さない）
2. ARIAKE_CVI.app を右クリック → 開く（初回）
3. ブロックされたら「システム設定 → プライバシーとセキュリティ」で「このまま開く」

ターミナルから quarantine を外す場合:
  xattr -cr "$(pwd)/ARIAKE_CVI"

※ Apple Silicon 用と Intel 用は別ビルドです。
※ このパッケージは Python 3.12 を同梱しています（Homebrew 不要）。
EOF

(
  cd "$DIST_DIR"
  rm -f "$ZIP_NAME"
  ditto -c -k --sequesterRsrc --keepParent "$STAGE_NAME" "$ZIP_NAME"
)

echo ""
echo "完了: $DIST_DIR/$ZIP_NAME"
echo "サイズ: $(du -h "$DIST_DIR/$ZIP_NAME" | awk '{print $1}')"
echo ""
echo "ローカル確認: open \"$ROOT/ARIAKE_CVI.app\""
