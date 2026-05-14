# ARIAKE_CVI — 使用方法

Developer: Team Yanagi

## 初回セットアップ

1. **フォルダ構成**（`deepgpet/` が必須）

```
ARIAKE_CVI/
├── main_st.py
├── app_launcher.py      （Windows の run.bat から起動する場合）
├── requirements.txt
├── setup.bat / run.bat  （Windows）
├── deepgpet/
│   └── choseg/
│       ├── inference.py
│       └── ...
└── README.md
```

2. **Python** 3.11 または 3.12 を推奨（PyTorch 未対応の 3.14 は非推奨）

3. 依存関係のインストール

```bash
python3.12 -m venv venv
source venv/bin/activate   # macOS / Linux
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **起動**

```bash
streamlit run main_st.py
```

Windows では `run.bat`（`app_launcher.py` 経由）でも可。

---

## ログインと画面

- ログイン前も、画面上部の **Settings** で「処理モード」「解析モード」、パス、スキャン条件、**Analyst** 名を指定できます（ログイン後のサイドバーと同じ項目・同じ状態が引き継がれます）。
- ログイン後のメインタイトルは **「CVI Results」** です。
- アクセスキーは管理者から共有された値を使用してください（ソース内の既定値は運用に合わせて変更してください）。

---

## 処理モード（サイドバー）

| モード | 内容 |
|--------|------|
| **File Upload** | 画像をアップロード。完了後は ZIP でダウンロード。 |
| **Folder Batch** | 入力フォルダ・出力フォルダを指定。完了後は出力フォルダに保存。 |

---

## 解析モード（サイドバー）

| モード | 内容 |
|--------|------|
| **自動解析**（既定） | **Fovea のクリックなし**で、リスト内の全画像を一括処理。 |
| **マニュアル解析** | 従来どおり、**画像ごとに Fovea をクリック**してから解析。 |

### マニュアル解析の流れ

1. 画像を用意（アップロードまたはフォルダ指定）。
2. 各画像で **Fovea** をクリック → 確認画面で **Confirm & Analyze** / **Skip** / **Reselect**。
3. 全件完了後、結果テーブルと ZIP またはフォルダ出力。

### 自動解析の流れ

1. 画像を用意（同上）。
2. **「▶ 自動解析を実行」** をクリック。初回のみモデル読み込みあり。
3. 完了後、結果テーブルと ZIP またはフォルダ出力。  
   - **同じ一覧で再実行** / **Reset Session** /（Folder 時）**Start New Batch** で状態をリセットできます。

---

## パラメータ（サイドバー）

- **Scan Width (mm)** — 既定 12.0  
- **Depth Range (mm)** — 既定 2.6  
- **Analyst** — 既定 `Yasuo Yanagi`（`parameters.csv` / フォルダ側 Parameter ログの項目名も **Analyst**）

---

## 結果の違い（マニュアル vs 自動）

### マニュアル解析

**`cvi_results.csv`** に含まれる例:

- Image ID, TCA / CT / CVI / D-CVI（1.5 mm / 3.0 mm）, Total mask area (mm2), Fluid area (mm2), D-Fluid area (mm2)

**画像（JPEG）**

- `CVI_*`, `DCVI_*` … 3 mm ROI 縦線・脈絡膜輪郭・Fovea マーカー付き  
- `CVI_plain_*`, `DCVI_plain_*` … 全幅の内腔二値化のみ黒オーバーレイ（ROI 線・輪郭・Fovea なし）  
- `CVI_Dlumen_overlay_*` … 元 B-scan に D-CVI 系内腔二値を重ねた図  

**ZIP / フォルダ名（Folder）**

- ZIP: `CVI_Results_YYYYMMDD_HHMM.zip`  
- フォルダ: `[入力フォルダ名]_CVI_[日時]/`

### 自動解析

**`cvi_results.csv`** の列のみ:

- **Image ID**, **Total mask area (mm2)**, **Fluid area (mm2)**, **D-Fluid area (mm2)**  
  （Fovea 依存の TCA / CVI 等は **出力しません**）

**画像（JPEG）**

- `CVI_plain_*`, `DCVI_plain_*`, `CVI_Dlumen_overlay_*` のみ（**`CVI_*` / `DCVI_*` ROI 本図は含みません**）

**ZIP / フォルダ名（Folder）**

- ZIP: `CVI_Auto_Results_YYYYMMDD_HHMM.zip`  
- フォルダ: `[入力フォルダ名]_CVI_auto_[日時]/`

**`parameters.csv`（および Folder の Parameter ログ）**

- **Analysis mode** に `自動解析` または `マニュアル解析` が記録されます。

---

## 注意事項

- **`deepgpet/`** が `main_st.py` と同階層に必要です。  
- `choseg.inference` が読めない環境では **MockModel** にフォールバックします（実解析には使えません）。  
- Fovea クリック位置がずれる場合はブラウザの表示倍率を 100% にしてください。

---

## システム要件

- OS: Windows 10/11, macOS, Linux  
- Python: 3.11 / 3.12 推奨  
- メモリ: 8 GB 以上推奨（大量画像・高解像度ではさらに余裕があると安心）  
- ストレージ: モデル含め約 2 GB 以上の空き推奨  

---

## バージョン

- Version: **2.1.0**  
- 更新日: **2026-05-14**  
- 主な更新: 自動解析モード、ログイン時 Settings、プレーン／ハイブリッド画像と面積系 CSV、UI 文言整理  

---

# ARIAKE_CVI (English)

Streamlit web app for choroidal vascularity index (CVI) and related metrics from OCT B-scan images, using a **DeepGPET** choroid mask and lumen binarization inside that mask.

## Features

- Access key on login; optional **Settings** on the login page (same widgets as the sidebar after login).
- **Processing mode**: File Upload (ZIP download) or Folder Batch (write results to disk).
- **Analysis mode**:
  - **Automatic (default)**: batch all images **without** fovea clicks; compact CSV and three JPEG types per image.
  - **Manual**: per-image fovea selection via `streamlit_image_coordinates`, full ROI metrics and `CVI_*` / `DCVI_*` figures.
- **Analyst** field (default `Yasuo Yanagi`) stored in `parameters.csv` / batch parameter logs.
- DeepGPET loads on first analysis; **MockModel** fallback if `torch` / `choseg` is missing (UI/testing only).

## Run

```bash
pip install -r requirements.txt
streamlit run main_st.py
```

Or on Windows: `run.bat` → `app_launcher.py` (browser disconnect can stop the server).

## Outputs (summary)

| Item | Manual | Automatic |
|------|--------|-----------|
| **cvi_results.csv** | Full metrics + mask/fluid areas | Image ID + Total mask area + Fluid + D-Fluid (mm²) only |
| **Figures** | `CVI_*`, `DCVI_*`, plains, `CVI_Dlumen_overlay_*` | Plains + `CVI_Dlumen_overlay_*` only |
| **ZIP name** | `CVI_Results_*.zip` | `CVI_Auto_Results_*.zip` |
| **Folder suffix** | `_CVI_` | `_CVI_auto_` |

---

*Developed by Team Yanagi*
