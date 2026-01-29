        ARIAKE_CVI - 使用方法                 
        Developer: Team Yanagi                


【初回セットアップ】

1. フォルダ構成を確認
        
```
   ARIAKE_CVI/
   ├── main_st.py
   ├── requirements.txt
   ├── setup.bat
   ├── run.bat
   ├── create_shortcut.py
   ├── deepgpet/          ← 重要！このフォルダが必要です
   │   └── choseg/
   │       ├── inference.py
   │       ├── utils.py
   │       └── ...
   └── README.txt
```                

3. setup.bat をダブルクリック
   
   ※ Pythonがインストールされていない場合：
   ```
     https://www.python.org/downloads/
   ```
     からダウンロードしてインストール
     （インストール時に「Add Python to PATH」に✓）

5. セットアップが完了すると、デスクトップに
   「ARIAKE_CVI」アイコンが作成されます

【使い方】

◆ 起動方法
  • デスクトップの「ARIAKE_CVI」アイコンをダブルクリック
  • または run.bat をダブルクリック

◆ ログイン
  デフォルトアクセスキー: ******

◆ 処理モード

  【File Uploadモード】
  1. サイドバーから画像をアップロード（複数選択可）
  2. 各画像でFovea（中心窩）をクリック選択
  3. 確認画像が表示されたら:
     • "Confirm & Analyze": 解析を実行して次へ
     • "Skip This Image": この画像をスキップ
     • "Reselect": クリック位置をやり直す
  4. 全画像完了後、結果をZIPダウンロード

  【Folder Batchモード】
  1. 入力フォルダと出力フォルダをBrowseボタンから選択
     （またはパスを直接入力）
  2. "Found XX images" が表示されたら、サイドバーが自動的に閉じます
  3. 各画像でFoveaをクリック選択
  4. 解析完了後、指定した出力フォルダに結果が保存されます
     • フォルダ名: [入力フォルダ名]_CVI_[日時]
     • 中身: cvi_results.csv, パラメータログ, 可視化画像

◆ パラメータ設定
  • Scan Width: スキャン幅（デフォルト: 12.0mm）
  • Depth Range: 深度範囲（デフォルト: 2.6mm）
  • Author: 解析者名

◆ 結果
  ZIPファイルに以下が含まれます:
  • cvi_results.csv - 解析結果データ
  • parameters.csv - パラメータログ
  • CVI_*.jpg - CVI可視化画像
  • DCVI_*.jpg - D-CVI可視化画像

【重要な注意事項】

⚠️ deepgpetフォルダが必須です
  - DeepGPETモデルが含まれているフォルダ
  - main_st.pyと同じディレクトリに配置してください

⚠️ 初回解析時のモデルロード
  - 最初の解析時に自動でAIモデルを読み込みます
  - 数秒〜数十秒かかる場合があります
  - モデル読み込み後は高速に動作します

【トラブルシューティング】

Q: 起動しない
A: setup.bat をもう一度実行してください

Q: "DeepGPETモジュールが見つかりません"と表示される
A: deepgpetフォルダがmain_st.pyと同じ場所にあるか確認

Q: モデルの読み込みに失敗する
A: deepgpetフォルダの中身が正しいか確認
   特にchoseg/inference.pyが存在するか

Q: 画像のクリック位置がずれる
A: ブラウザのズームを100%に設定してください

Q: フォルダパスが認識されない
A: パスの例: C:/Users/YourName/Documents/images
   スラッシュ(/)またはバックスラッシュ2つ(\\)を使用

【システム要件】

• OS: Windows 10/11, macOS, Linux
• Python: 3.11 or 3.12 推奨（3.14はPyTorch未対応）
• メモリ: 8GB以上推奨
• ストレージ: 2GB以上の空き容量（モデル含む）
• ブラウザ: Chrome, Firefox, Edge など最新版

【バージョン情報】
Version: 2.0.0
更新日: 2026-01-24
Developer: Team Yanagi

主な機能:
• Streamlit UIへの完全移行
• Folder Batch処理モード追加
• フォルダーブラウザー機能
• Skip機能追加

ARIAKE_CVI is a Streamlit-based web app for automated choroidal vascularity index (CVI) and related metrics from OCT B‑scan images using a DeepGPET segmentation model.

## Overview

- **ARIAKE_CVI** runs as a Streamlit app and loads a DeepGPET-based model (with a MockModel fallback) to segment the choroid on grayscale B‑scan images.
- The app computes TCA, CT, CVI, and D‑CVI within 1.5 mm and 3.0 mm fovea-centered regions and exports both metrics and visualization images.

## Features

- Simple access control via a shared access key and login screen.
- Multi-file upload of B‑scan images (png/jpg/jpeg/tif/tiff) with manual fovea selection per image via `streamlit_image_coordinates`.
- Automatic first-time model initialization plus optional manual re-initialization from the sidebar.
- Per-image computation of:
  - TCA 1.5/3.0 mm (mm²), CT 1.5/3.0 mm (µm), CVI 1.5/3.0 mm (%), D‑CVI 1.5/3.0 mm (%).
- On-the-fly visualization:
  - Original and denoised images with overlaid choroid segmentation, fovea marker, and 3.0‑mm ROI, saved as JPEGs (CVI_*, DCVI_*).
- One-click download of:
  - `cvi_results.csv` with all image metrics, `parameters.csv` with app/author/date/scan width/depth, and all visualization images packed into a ZIP file.

## Requirements

- **Python Version**: 3.11 or 3.12 recommended (PyTorch does not support Python 3.14 yet as of 2026-01)
- **Python packages**: `streamlit`, `numpy`, `opencv-python`, `Pillow`, `pandas`, `streamlit-image-coordinates`, `torch`, `torchvision`, `scikit-image`, etc.
- **Project structure** (expected):  
  - `main_st.py` (this app file) and a `deepgpet/` directory containing `choseg/inference.py` and related utilities for `DeepGPET`.
- If `choseg.inference` cannot be imported, the app falls back to a dummy `MockModel` that outputs an empty mask (for UI testing only, not for real analysis).

## How to Run

### 1. Setup Python Environment (Important!)

**For Python 3.14 users:** PyTorch does not support Python 3.14 yet. Please use Python 3.11 or 3.12:

```bash
# Check available Python versions
ls /usr/local/bin/python3*

# If you have Python 3.11 or 3.12, create virtual environment with it
python3.11 -m venv venv
# or
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

1. Place `main_st.py` and the `deepgpet` directory in the same folder (so `deepgpet_path = Path(__file__).parent / 'deepgpet'` is valid).
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
   
### 3. Start the App

3. Start the app:  
   ```bash
   streamlit run main_st.py
   ```  
4. Open the URL shown in the terminal, enter the access key on the login screen, and proceed to the main interface.

## Usage Workflow

1. In the sidebar, upload B‑scan images and set scan width (mm), depth range (mm), and author name.
2. For each uploaded image:
   - Click on the foveal center in the displayed B‑scan.
   - Click “Confirm & Analyze” to trigger DeepGPET-based segmentation and CVI computation.
3. After all images are processed, review the results table on the main page.
4. Click “Download All Results (Zip)” to obtain the metrics CSVs and visualization images; use “Reset Session” to clear state and start a new batch.

• Enterキー対応（開発中）

【サポート】
問題がある場合は開発チームにお問い合わせください
