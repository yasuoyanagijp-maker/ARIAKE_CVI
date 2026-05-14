"""
ARIAKE_CVI (DeepGPET based) - Streamlit Cloud Ready Version
Developer: Team Yanagi
特定のURLを知っているメンバーのみが利用できるよう、簡易認証とファイルアップロード機能を実装。
初回の解析時にモデルを自動初期化するユーザビリティ向上版。
"""
import sys
import os
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import io
import zipfile
from streamlit_image_coordinates import streamlit_image_coordinates

# DeepGPET module path setup
deepgpet_path = Path(__file__).resolve().parent / 'deepgpet'
sys.path.insert(0, str(deepgpet_path))

try:
    from choseg import inference
except ImportError as e:
    error_msg = str(e)
    if 'torch' in error_msg:
        st.error(f"""
        ⚠️ **PyTorchが見つかりません**
        
        現在のPythonバージョン（{sys.version.split()[0]}）ではPyTorchがインストールできない可能性があります。
        
        **解決方法:**
        1. Python 3.11または3.12を使用してください
        2. 仮想環境を再作成してください:
           ```
           python3.11 -m venv venv
           source venv/bin/activate
           pip install -r requirements.txt
           ```
        
        詳細エラー: {error_msg}
        """)
    else:
        st.error(f"⚠️ DeepGPETモジュールが見つかりません。deepgpetフォルダが同じディレクトリにあることを確認してください。\n\nエラー: {error_msg}")
    
    class MockModel:
        def __call__(self, img):
            # Ensure numpy array output consistent with real model
            if isinstance(img, Image.Image):
                img = np.array(img)
            return np.zeros(img.shape, dtype=np.float32)
    inference = type('obj', (object,), {'DeepGPET': MockModel})

APP_NAME = "ARIAKE_CVI"
DEVELOPER = "Team Yanagi"
ACCESS_KEY = "ariake2024"  # 公開時に共有するアクセスキー（必要に応じて変更してください）

# --- 深さ依存 Niblack（ROI = DeepGPET の脈絡膜マスクのみ。fovea 周辺の数 mm 解析帯とは別物）---
# True: マスク内で浅い側（画像上端寄り＝通常 RPE 側、d が小さい）から下方向へ NIBLACK_INNER_DEPTH_FRAC の範囲に k_inner
# False: 縦反転 B-scan 等で浅いが d 大のとき用に、d > (1 - NIBLACK_INNER_DEPTH_FRAC) に k_inner
NIBLACK_STRICT_K_ON_SHALLOW_LOW_D = True
# 脈絡膜マスク（ROI）内の厚み方向で、浅い側から何割を k_inner とするか（上から 1/3 = 0.333...）
NIBLACK_INNER_DEPTH_FRAC = 1.0 / 3.0
# マスク内浅い帯 k_inner / それ以外 k_outer（inner だけ上げる場合はここを調整）
NIBLACK_K_OUTER = 0.2
NIBLACK_K_INNER = 0.12
# 二値化後の median。0 または 1 以下でスキップ（現在はスキップ）
LUMEN_BINARY_MEDIAN_KSIZE = 0

# ==========================================
# 解析ロジック (CVIProcessor)
# ==========================================
class CVIProcessor:
    def __init__(self):
        self.model = None
        
    def initialize_model(self):
        if self.model is None:
            self.model = inference.DeepGPET()

    @staticmethod
    def choroid_depth_fraction_map(mask):
        """
        DeepGPET 脈絡膜マスク（本アプリでいう ROI＝セグメンテーション領域）内の正規化深さ [0,1]。
        各列 x でマスクの上端 y_top から下端 y_bot までを 1 としたときの相対深さ。
        0 = マスク上端（画像では浅い方＝通常は RPE 寄り）、1 = マスク下端（深い方）。
        画像全体の高さは使わない。
        """
        h, w = mask.shape
        depth = np.zeros((h, w), dtype=np.float32)
        y_top = np.full(w, -1, np.int32)
        y_bot = np.full(w, -1, np.int32)
        m = mask > 0
        for x in range(w):
            ys = np.flatnonzero(m[:, x])
            if ys.size:
                y_top[x], y_bot[x] = int(ys.min()), int(ys.max())
        last_t, last_b = -1, -1
        for x in range(w):
            if y_top[x] >= 0:
                last_t, last_b = y_top[x], y_bot[x]
            elif last_t >= 0:
                y_top[x], y_bot[x] = last_t, last_b
        last_t, last_b = -1, -1
        for x in range(w - 1, -1, -1):
            if y_top[x] >= 0:
                last_t, last_b = y_top[x], y_bot[x]
            elif last_t >= 0:
                y_top[x], y_bot[x] = last_t, last_b
        rows = np.arange(h, dtype=np.float32)
        for x in range(w):
            if y_top[x] < 0:
                continue
            th = max(float(y_bot[x] - y_top[x]), 1.0)
            depth[:, x] = np.clip((rows - float(y_top[x])) / th, 0.0, 1.0)
        depth[~m] = 0.0
        return depth

    def niblack_threshold_depth_adaptive_k(
        self,
        img,
        mask,
        radius=15,
        k_outer=NIBLACK_K_OUTER,
        k_inner=NIBLACK_K_INNER,
        inner_depth_frac=(1.0 / 3.0),
        strict_k_on_shallow_low_d=True,
    ):
        """
        Niblack の k を、DeepGPET 脈絡膜マスク（ROI）内の相対深さ d で切替する。
        d は choroid_depth_fraction_map（マスク上端=0、下端=1）。画像全体や fovea 数 mm 帯ではない。

        strict_k_on_shallow_low_d=True: マスク内の浅い側から inner_depth_frac（例: 上 1/3）に k_inner。
        False: 縦反転等で浅いが d が大きいとき、d > (1-inner_depth_frac) に k_inner。

        k_map はマスク内だけ上書き（マスク外は k_outer のまま計算し、最後に mask で除去）。
        """
        img_float = img.astype(np.float32)
        kernel_size = 2 * radius + 1
        mean = cv2.blur(img_float, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.blur(img_float ** 2, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
        variance = mean_sq - (mean ** 2)
        std_dev = np.sqrt(np.maximum(variance, 0))
        d = self.choroid_depth_fraction_map(mask)
        m = mask > 0
        frac = float(inner_depth_frac)
        k_map = np.full(img_float.shape[:2], k_outer, dtype=np.float32)
        if strict_k_on_shallow_low_d:
            inner_sel = m & (d < frac)
        else:
            inner_sel = m & (d > (1.0 - frac))
        k_map[inner_sel] = float(k_inner)
        threshold = mean + k_map * std_dev
        threshold = np.clip(threshold, 0.0, 255.0)
        binary = (img_float < threshold).astype(np.uint8) * 255
        binary[mask == 0] = 0
        return binary
    
    def niblack_threshold(self, img, mask, radius=15, k=0.2):
        img_float = img.astype(np.float32)
        kernel_size = 2 * radius + 1
        mean = cv2.blur(img_float, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.blur(img_float ** 2, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
        variance = mean_sq - (mean ** 2)
        std_dev = np.sqrt(np.maximum(variance, 0))
        threshold = mean + k * std_dev
        binary = (img_float < threshold).astype(np.uint8) * 255
        binary[mask == 0] = 0
        return binary

    def denoise_image_global(self, img):
        smoothed = cv2.GaussianBlur(img, (3, 3), 0)
        img_float = smoothed.astype(np.float32) / 255.0
        j_raw = np.power(img_float, 4)
        sum_j_raw = np.cumsum(np.power(j_raw, 2)[::-1], axis=0)[::-1]
        sum_j_raw[sum_j_raw == 0] = 1e-6
        j_raw_squared = np.power(j_raw, 2)
        enhanced = 255 * np.power(j_raw_squared / (2 * sum_j_raw), 0.25)
        result = np.clip(enhanced, 0, 255).astype(np.uint8)
        result = cv2.medianBlur(result, 3)
        return result

    def get_roi_metrics(self, global_lum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, width_mm):
        height, width = mask.shape
        
        # Validate fovea position
        fovea_x = max(0, min(fovea_x, width - 1))
        
        roi_width_px = int(width_mm * scale_px_per_mm_h)
        roi_start_x = max(0, int(fovea_x - roi_width_px / 2))
        roi_end_x = min(width, roi_start_x + roi_width_px)
        actual_roi_width_px = roi_end_x - roi_start_x
        if actual_roi_width_px <= 0:
            return 0.0, 0.0, 0.0, 0.0, None
        roi_mask = mask[:, roi_start_x:roi_end_x]
        roi_lum = global_lum_mask[:, roi_start_x:roi_end_x]
        tca_px = np.sum(roi_mask > 0)
        la_px = np.sum(roi_lum > 0)
        pixel_area_mm2 = (1.0 / scale_px_per_mm_h) * (1.0 / scale_px_per_mm_v)
        tca_mm2 = tca_px * pixel_area_mm2
        la_mm2 = la_px * pixel_area_mm2
        cvi = (la_mm2 / tca_mm2 * 100) if tca_mm2 > 0 else 0.0
        ct_um = (tca_mm2 / (actual_roi_width_px / scale_px_per_mm_h)) * 1000
        return tca_mm2, ct_um, cvi, la_mm2, roi_lum

    def _prepare_deepgpet_lumen(self, pil_img, scan_width_mm, depth_range_mm):
        """
        DeepGPET セグメンテーションと全幅内腔二値マスクまで（fovea 非依存）。
        Returns:
            (img_gray, denoised_img, mask, global_lum_mask, global_dlum_mask,
             height, width, scale_px_per_mm_h, scale_px_per_mm_v) or None on failure.
        """
        img_gray = np.array(pil_img.convert('L'))
        height, width = img_gray.shape
        scale_px_per_mm_h = width / scan_width_mm
        scale_px_per_mm_v = height / depth_range_mm
        img_seg = self.model(Image.fromarray(img_gray))
        if isinstance(img_seg, np.ndarray):
            mask = (img_seg > 0.5).astype(np.uint8) * 255
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        denoised_img = self.denoise_image_global(img_gray)
        global_lum_mask = self.niblack_threshold_depth_adaptive_k(
            img_gray,
            mask,
            k_outer=NIBLACK_K_OUTER,
            k_inner=NIBLACK_K_INNER,
            inner_depth_frac=NIBLACK_INNER_DEPTH_FRAC,
            strict_k_on_shallow_low_d=NIBLACK_STRICT_K_ON_SHALLOW_LOW_D,
        )
        global_dlum_mask = self.niblack_threshold_depth_adaptive_k(
            denoised_img,
            mask,
            k_outer=NIBLACK_K_OUTER,
            k_inner=NIBLACK_K_INNER,
            inner_depth_frac=NIBLACK_INNER_DEPTH_FRAC,
            strict_k_on_shallow_low_d=NIBLACK_STRICT_K_ON_SHALLOW_LOW_D,
        )
        if LUMEN_BINARY_MEDIAN_KSIZE and LUMEN_BINARY_MEDIAN_KSIZE >= 3:
            global_lum_mask = cv2.medianBlur(global_lum_mask, int(LUMEN_BINARY_MEDIAN_KSIZE))
            global_dlum_mask = cv2.medianBlur(global_dlum_mask, int(LUMEN_BINARY_MEDIAN_KSIZE))
        global_lum_mask[mask == 0] = 0
        global_dlum_mask[mask == 0] = 0
        return (
            img_gray,
            denoised_img,
            mask,
            global_lum_mask,
            global_dlum_mask,
            height,
            width,
            scale_px_per_mm_h,
            scale_px_per_mm_v,
        )

    @staticmethod
    def encode_binarization_overlay_jpg(base_img, global_lumen_binary):
        """内腔二値を黒で重ねた JPEG（全スキャン幅、ROI 線・輪郭・fovea なし）。"""
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
        vis[global_lumen_binary > 0] = [0, 0, 0]
        ok, buf = cv2.imencode(".jpg", vis)
        return buf.tobytes() if ok else None

    def process_image_automatic(self, pil_img, filename, scan_width_mm, depth_range_mm):
        """
        自動解析: fovea / 3mm ROI 指標・CVI_* / DCVI_* 本図は出さない。
        CSV は Image ID + Total mask area + Fluid / D-Fluid (mm2) のみ。
        画像は CVI_plain / DCVI_plain / CVI_Dlumen_overlay のみ。
        """
        try:
            prep = self._prepare_deepgpet_lumen(pil_img, scan_width_mm, depth_range_mm)
            img_gray, denoised_img, mask, global_lum_mask, global_dlum_mask, height, width, scale_px_per_mm_h, scale_px_per_mm_v = prep
            pixel_area_mm2 = (1.0 / scale_px_per_mm_h) * (1.0 / scale_px_per_mm_v)
            total_mask_area_mm2 = float(np.sum(mask > 0)) * pixel_area_mm2
            fluid_area_mm2 = float(np.sum(global_lum_mask > 0)) * pixel_area_mm2
            d_fluid_area_mm2 = float(np.sum(global_dlum_mask > 0)) * pixel_area_mm2
            plain_cvi = self.encode_binarization_overlay_jpg(img_gray, global_lum_mask)
            plain_dcvi = self.encode_binarization_overlay_jpg(denoised_img, global_dlum_mask)
            hybrid = self.encode_binarization_overlay_jpg(img_gray, global_dlum_mask)
            stats = {
                'Image ID': filename,
                'Total mask area (mm2)': round(total_mask_area_mm2, 4),
                'Fluid area (mm2)': round(fluid_area_mm2, 4),
                'D-Fluid area (mm2)': round(d_fluid_area_mm2, 4),
            }
            return stats, None, None, plain_cvi, plain_dcvi, hybrid
        except Exception as e:
            st.error(f"❌ Error (automatic) {filename}: {str(e)}")
            return None, None, None, None, None, None

    def process_image(self, pil_img, filename, fovea_x, fovea_y, scan_width_mm, depth_range_mm):
        try:
            prep = self._prepare_deepgpet_lumen(pil_img, scan_width_mm, depth_range_mm)
            img_gray, denoised_img, mask, global_lum_mask, global_dlum_mask, height, width, scale_px_per_mm_h, scale_px_per_mm_v = prep

            # Validate fovea coordinates
            fovea_x = max(0, min(int(fovea_x), width - 1))
            fovea_y = max(0, min(int(fovea_y), height - 1))
            
            # Get Metrics
            tca15, ct15, cvi15, _, _ = self.get_roi_metrics(global_lum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 1.5)
            tca30, ct30, cvi30, _, lum30_roi = self.get_roi_metrics(global_lum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 3.0)
            dtca15, dct15, dcvi15, _, _ = self.get_roi_metrics(global_dlum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 1.5)
            dtca30, dct30, dcvi30, _, dlum30_roi = self.get_roi_metrics(global_dlum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 3.0)

            # CVI_plain / DCVI_plain の黒オーバーレイ面積（脈絡膜マスク内・スキャン全幅の内腔二値化）
            pixel_area_mm2 = (1.0 / scale_px_per_mm_h) * (1.0 / scale_px_per_mm_v)
            total_mask_area_mm2 = float(np.sum(mask > 0)) * pixel_area_mm2
            fluid_area_mm2 = float(np.sum(global_lum_mask > 0)) * pixel_area_mm2
            d_fluid_area_mm2 = float(np.sum(global_dlum_mask > 0)) * pixel_area_mm2
        
            # Generate visualization images for download/preview
            def roi_3mm_horizontal_bounds():
                roi_w_px = int(3.0 * scale_px_per_mm_h)
                rx_start = max(0, int(fovea_x - roi_w_px / 2))
                rx_end = min(width, rx_start + roi_w_px)
                return rx_start, rx_end

            def apply_lumen_binarization_overlay(vis_rgb, l_mask_roi):
                """3.0mm ROI 内の内腔二値化を黒で重ねる（CVI/DCVI 本図用）"""
                rx_start, rx_end = roi_3mm_horizontal_bounds()
                full_v_mask = np.zeros_like(mask)
                if l_mask_roi is not None:
                    full_v_mask[:, rx_start:rx_end] = l_mask_roi
                vis_rgb[full_v_mask > 0] = [0, 0, 0]

            def create_vis_buffer(base_img, l_mask_roi):
                vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
                apply_lumen_binarization_overlay(vis, l_mask_roi)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
                cv2.circle(vis, (int(fovea_x), int(fovea_y)), 8, (0, 255, 0), -1)
                rx_start, rx_end = roi_3mm_horizontal_bounds()
                cv2.line(vis, (rx_start, 0), (rx_start, height), (255, 0, 0), 2)
                cv2.line(vis, (rx_end, 0), (rx_end, height), (255, 0, 0), 2)
                is_success, buffer = cv2.imencode(".jpg", vis)
                return buffer.tobytes() if is_success else None

            vis_cvi = create_vis_buffer(img_gray, lum30_roi)
            vis_dcvi = create_vis_buffer(denoised_img, dlum30_roi)

            plain_cvi = self.encode_binarization_overlay_jpg(img_gray, global_lum_mask)
            plain_dcvi = self.encode_binarization_overlay_jpg(denoised_img, global_dlum_mask)
            # 元B-scan（CVI系）上に D-CVI 系の全幅内腔二値化のみ黒で重ねた図
            cvi_base_dlumen_overlay = self.encode_binarization_overlay_jpg(img_gray, global_dlum_mask)

            stats = {
                'Image ID': filename,
                'TCA 1.5mm (mm2)': round(tca15, 4),
                'CT 1.5mm (um)': round(ct15, 2),
                'CVI 1.5mm (%)': round(cvi15, 2),
                'D-CVI 1.5mm (%)': round(dcvi15, 2),
                'TCA 3.0mm (mm2)': round(tca30, 4),
                'CT 3.0mm (um)': round(ct30, 2),
                'CVI 3.0mm (%)': round(cvi30, 2),
                'D-CVI 3.0mm (%)': round(dcvi30, 2),
                'Total mask area (mm2)': round(total_mask_area_mm2, 4),
                'Fluid area (mm2)': round(fluid_area_mm2, 4),
                'D-Fluid area (mm2)': round(d_fluid_area_mm2, 4),
            }
            return stats, vis_cvi, vis_dcvi, plain_cvi, plain_dcvi, cvi_base_dlumen_overlay
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {str(e)}")
            return None, None, None, None, None, None

# ==========================================
# Streamlit UI
# ==========================================
# サイドバーの初期状態を設定（画像処理中は自動的に閉じる）
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False

initial_sidebar_state = "collapsed" if st.session_state.processing_started else "expanded"
st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state=initial_sidebar_state)

# カスタムCSS: 画像の位置を完全に統一
st.markdown("""
<style>
    /* streamlit_image_coordinatesとst.imageの表示位置を統一 */
    div[data-testid="stImage"],
    div[data-testid="stImageContainer"] {
        display: flex !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stImage"] > img,
    div[data-testid="stImageContainer"] > img {
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態初期化
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'results' not in st.session_state:
    st.session_state.results = []
if 'vis_files' not in st.session_state:
    st.session_state.vis_files = {} # filename -> bytes
if 'input_path' not in st.session_state:
    st.session_state.input_path = ""
if 'output_path' not in st.session_state:
    st.session_state.output_path = ""
if 'browsing_for' not in st.session_state:
    st.session_state.browsing_for = None  # "input" or "output"
if 'current_browse_path' not in st.session_state:
    st.session_state.current_browse_path = str(Path.home())
if 'fovea_clicked' not in st.session_state:
    st.session_state.fovea_clicked = {}  # idx -> (fx, fy)


def uploaded_files_fingerprint(files):
    """入力リストの同一性判定（自動解析でファイル差し替え時に状態をリセット）。"""
    if not files:
        return ""
    if isinstance(files[0], Path):
        return "\x00".join(sorted(str(p.resolve()) for p in files))
    return "\x00".join(sorted(getattr(f, "name", str(id(f))) for f in files))


def render_app_settings(*, show_header=True):
    """
    サイドバーとログイン画面上部で共通の設定 UI。
    同一の widget key を使うため、ログイン前に選んだモード・パス等はログイン後のサイドバーに引き継がれる。
    """
    if show_header:
        st.header("⚙️ Settings")

    mode = st.radio(
        "Processing Mode",
        ["File Upload", "Folder Batch"],
        key="settings_processing_mode",
    )

    analysis_mode = st.radio(
        "解析モード",
        ["マニュアル解析", "自動解析"],
        key="settings_analysis_mode",
        help=(
            "自動解析: fovea・3mm ROI を指定せず、Total mask area / Fluid / D-Fluid (mm²) と "
            "CVI_plain / DCVI_plain / CVI_Dlumen_overlay のみを出力します。"
        ),
    )

    if mode == "File Upload":
        uploaded_files = st.file_uploader(
            "Upload B-scan Images",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key="settings_file_uploader",
        )
        output_folder = None
        input_folder = None
    else:
        st.markdown("**📁 Folder Paths**")

        col1, col2 = st.columns([3, 1])
        with col1:
            input_folder = st.text_input(
                "Input Folder",
                value=st.session_state.input_path,
                placeholder="/Users/y/Github/input",
                key="input_folder_text",
            )
        with col2:
            if st.button("📂 Browse", key="browse_input"):
                st.session_state.browsing_for = "input"
                st.session_state.current_browse_path = (
                    st.session_state.input_path if st.session_state.input_path else str(Path.home())
                )

        col1, col2 = st.columns([3, 1])
        with col1:
            output_folder = st.text_input(
                "Output Folder",
                value=st.session_state.output_path,
                placeholder="/Users/y/Github/output",
                key="output_folder_text",
            )
        with col2:
            if st.button("📂 Browse", key="browse_output"):
                st.session_state.browsing_for = "output"
                st.session_state.current_browse_path = (
                    st.session_state.output_path if st.session_state.output_path else str(Path.home())
                )

        if st.session_state.browsing_for:
            st.markdown("---")
            st.markdown(
                f"### 📂 Select {'Input' if st.session_state.browsing_for == 'input' else 'Output'} Folder"
            )

            quick_paths = {
                "🏠 Home": str(Path.home()),
                "🖥️ Desktop": str(Path.home() / "Desktop"),
                "📄 Documents": str(Path.home() / "Documents"),
                "📥 Downloads": str(Path.home() / "Downloads"),
            }

            cols = st.columns(len(quick_paths))
            for idx, (label, path) in enumerate(quick_paths.items()):
                with cols[idx]:
                    if st.button(label, key=f"quick_{idx}", use_container_width=True):
                        st.session_state.current_browse_path = path
                        st.rerun()

            browse_path = Path(st.session_state.current_browse_path)
            st.text_input("Current Path:", value=str(browse_path), disabled=True, key="current_path_display")

            if browse_path.parent != browse_path:
                if st.button("⬆️ Parent Folder", use_container_width=True):
                    st.session_state.current_browse_path = str(browse_path.parent)
                    st.rerun()

            try:
                subdirs = sorted(
                    [d for d in browse_path.iterdir() if d.is_dir()],
                    key=lambda x: x.name.lower(),
                )
                if subdirs:
                    st.markdown("**Folders:**")
                    for subdir in subdirs[:20]:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(f"📁 {subdir.name}")
                        with col2:
                            if st.button("Open", key=f"open_{subdir.name}"):
                                st.session_state.current_browse_path = str(subdir)
                                st.rerun()
                else:
                    st.info("No subfolders found")
            except PermissionError:
                st.error("❌ Permission denied")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select This Folder", type="primary", use_container_width=True):
                    if st.session_state.browsing_for == "input":
                        st.session_state.input_path = st.session_state.current_browse_path
                    else:
                        st.session_state.output_path = st.session_state.current_browse_path
                    st.session_state.browsing_for = None
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.browsing_for = None
                    st.rerun()

            st.markdown("---")

        uploaded_files = []
        if input_folder and os.path.exists(input_folder):
            try:
                image_files = sorted(
                    list(Path(input_folder).glob("*.png"))
                    + list(Path(input_folder).glob("*.jpg"))
                    + list(Path(input_folder).glob("*.jpeg"))
                    + list(Path(input_folder).glob("*.tif"))
                    + list(Path(input_folder).glob("*.tiff"))
                )
                st.info(f"✅ Found {len(image_files)} images")
                uploaded_files = image_files
            except Exception as e:
                st.error(f"❌ Error reading folder: {str(e)}")
        elif input_folder:
            st.error(f"❌ Folder path does not exist: {input_folder}")

        if output_folder and not os.path.exists(output_folder):
            st.warning(f"⚠️ Output folder will be created: {output_folder}")

    scan_w = st.number_input("Scan Width (mm)", value=12.0, min_value=0.1, key="settings_scan_w")
    depth_r = st.number_input("Depth Range (mm)", value=2.6, min_value=0.1, key="settings_depth_r")
    author = st.text_input("Author", value="YY", key="settings_author")

    st.divider()

    if "processor" in st.session_state:
        st.success("✅ AI Model Loaded")
    else:
        st.info("ℹ️ Model will be loaded automatically on first analysis")

    return mode, uploaded_files, input_folder, output_folder, scan_w, depth_r, author, analysis_mode


# --- 認証画面 ---
if not st.session_state.authenticated:
    with st.expander("⚙️ Settings", expanded=True):
        st.caption("ログイン前でもモード・パス・スキャン条件を指定できます。ログイン後も同じ内容がサイドバーに表示されます。")
        render_app_settings(show_header=False)
    st.divider()
    st.title(f"🔐 {APP_NAME} Login")
    pwd = st.text_input("Enter Access Key", type="password")
    if st.button("Login"):
        if pwd == ACCESS_KEY:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Access Key.")
    st.stop()

# --- メイン画面 ---
st.title(f"🚀 {APP_NAME}")
st.markdown(f"*Developed by {DEVELOPER}*")

# サイドバー設定（ログイン画面の expander と同一 UI）
with st.sidebar:
    mode, uploaded_files, input_folder, output_folder, scan_w, depth_r, author, analysis_mode = (
        render_app_settings(show_header=True)
    )

if "last_settings_analysis_mode" not in st.session_state:
    st.session_state.last_settings_analysis_mode = analysis_mode
elif st.session_state.last_settings_analysis_mode != analysis_mode:
    st.session_state.last_settings_analysis_mode = analysis_mode
    for key in ("current_idx", "results", "vis_files", "fovea_clicked", "auto_analysis_complete", "_auto_fp"):
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.results = []
    st.session_state.vis_files = {}
    st.session_state.current_idx = 0
    st.session_state.fovea_clicked = {}
    st.session_state.processing_started = False
    st.rerun()

# 解析処理
if uploaded_files:
    if not st.session_state.processing_started:
        st.session_state.processing_started = True

    _fp = uploaded_files_fingerprint(uploaded_files)
    if analysis_mode == "自動解析" and st.session_state.get("_auto_fp") != _fp:
        st.session_state._auto_fp = _fp
        st.session_state.auto_analysis_complete = False
        st.session_state.results = []
        st.session_state.vis_files = {}

    if analysis_mode == "自動解析":
        if st.session_state.get("auto_analysis_complete"):
            st.success("🎉 自動解析が完了しました（fovea 指定・3mm ROI 本図・ROI 依存 CSV 項目は出力されません）")
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df)

            _reset_keys = [
                "current_idx",
                "results",
                "vis_files",
                "fovea_clicked",
                "auto_analysis_complete",
                "_auto_fp",
            ]

            if mode == "Folder Batch" and output_folder:
                try:
                    input_folder_name = Path(input_folder).name if input_folder else "output"
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_root = Path(output_folder) / f"{input_folder_name}_CVI_auto_{now_str}"
                    out_root.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out_root / "cvi_results.csv", index=False)
                    log_df = pd.DataFrame(
                        {
                            "Parameter": [
                                "Macro Name",
                                "Developer",
                                "Date",
                                "Time",
                                "Original Folder",
                                "Analyzed by",
                                "Analysis mode",
                                "Scan Width (H-mm)",
                                "Depth Range (V-mm)",
                                "Total Images",
                                "Success",
                            ],
                            "Value": [
                                APP_NAME,
                                DEVELOPER,
                                datetime.now().strftime("%Y-%m-%d"),
                                datetime.now().strftime("%H:%M:%S"),
                                input_folder_name,
                                author,
                                "自動解析",
                                scan_w,
                                depth_r,
                                len(uploaded_files),
                                len(st.session_state.results),
                            ],
                        }
                    )
                    log_df.to_csv(out_root / f"{input_folder_name}_Parameter.csv", index=False)
                    for fname, fdata in st.session_state.vis_files.items():
                        if fdata is not None:
                            with open(out_root / fname, "wb") as f:
                                f.write(fdata)
                    st.success(f"✅ Results saved to: `{out_root}`")
                    if st.button("🔄 Start New Batch", key="auto_folder_new_batch"):
                        for key in _reset_keys:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving to output folder: {str(e)}")

            elif mode == "File Upload":
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    zip_file.writestr("cvi_results.csv", csv_data)
                    log_df = pd.DataFrame(
                        {
                            "Parameter": [
                                "App",
                                "Author",
                                "Date",
                                "Time",
                                "Analysis mode",
                                "Scan Width",
                                "Depth Range",
                            ],
                            "Value": [
                                APP_NAME,
                                author,
                                datetime.now().strftime("%Y-%m-%d"),
                                datetime.now().strftime("%H:%M:%S"),
                                "自動解析",
                                scan_w,
                                depth_r,
                            ],
                        }
                    )
                    zip_file.writestr("parameters.csv", log_df.to_csv(index=False).encode("utf-8"))
                    for fname, fdata in st.session_state.vis_files.items():
                        if fdata is not None:
                            zip_file.writestr(fname, fdata)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button(
                        label="📥 Download Results (Zip)",
                        data=zip_buffer.getvalue(),
                        file_name=f"CVI_Auto_Results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip",
                        type="primary",
                    )
                with c2:
                    if st.button("🔄 同じ一覧で再実行", key="auto_rerun_btn"):
                        st.session_state.auto_analysis_complete = False
                        st.session_state.results = []
                        st.session_state.vis_files = {}
                        st.rerun()
                with c3:
                    if st.button("🔄 Reset Session", key="auto_reset_session"):
                        for key in _reset_keys:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
        else:
            st.subheader("自動解析")
            st.info(
                "DeepGPET で脈絡膜マスクを推定し、マスク面積（mm²）・全幅内腔面積・"
                "プレーン可視化のみを一括出力します。fovea のクリックは不要です。"
            )
            st.caption(
                "CSV: Image ID, Total mask area (mm2), Fluid area (mm2), D-Fluid area (mm2) のみ。"
                " 画像: CVI_plain_*, DCVI_plain_*, CVI_Dlumen_overlay_* のみ（CVI_* / DCVI_* ROI 本図は含みません）。"
            )
            st.markdown(f"**対象画像数:** {len(uploaded_files)}")
            if st.button("▶ 自動解析を実行", type="primary", key="auto_run_all_button"):
                if "processor" not in st.session_state:
                    with st.spinner("🔄 Loading AI model for the first time..."):
                        st.session_state.processor = CVIProcessor()
                        st.session_state.processor.initialize_model()
                results = []
                vis_files = {}
                n_total = len(uploaded_files)
                progress = st.progress(0)
                for i, file_item in enumerate(uploaded_files):
                    try:
                        if isinstance(file_item, Path):
                            pil_img = Image.open(file_item)
                            filename = file_item.name
                        else:
                            pil_img = Image.open(file_item)
                            filename = file_item.name
                    except Exception as e:
                        st.error(f"❌ Cannot open image: {str(e)}")
                        progress.progress(min(1.0, (i + 1) / max(n_total, 1)))
                        continue
                    res, _v1, _v2, p_cvi, p_dcvi, hybrid = st.session_state.processor.process_image_automatic(
                        pil_img, filename, scan_w, depth_r
                    )
                    if res is not None:
                        results.append(res)
                        if p_cvi is not None:
                            vis_files[f"CVI_plain_{filename}"] = p_cvi
                        if p_dcvi is not None:
                            vis_files[f"DCVI_plain_{filename}"] = p_dcvi
                        if hybrid is not None:
                            vis_files[f"CVI_Dlumen_overlay_{filename}"] = hybrid
                    progress.progress(min(1.0, (i + 1) / max(n_total, 1)))
                st.session_state.results = results
                st.session_state.vis_files = vis_files
                st.session_state.auto_analysis_complete = True
                st.rerun()
    else:
        idx = st.session_state.current_idx

        if idx < len(uploaded_files):
            # ファイル取得（File UploadとPathオブジェクト両対応）
            file_item = uploaded_files[idx]
            if isinstance(file_item, Path):
                try:
                    pil_img = Image.open(file_item)
                    filename = file_item.name
                except Exception as e:
                    st.error(f"❌ Cannot open image: {str(e)}")
                    st.stop()
            else:
                try:
                    pil_img = Image.open(file_item)
                    filename = file_item.name
                except Exception as e:
                    st.error(f"❌ Cannot open image: {str(e)}")
                    st.stop()
            
            st.subheader(f"Step {idx+1}/{len(uploaded_files)}: Select Fovea for `{filename}`")
            
            orig_w, orig_h = pil_img.size
            
            # クリック確定前か確定後かで表示を切り替え
            if idx not in st.session_state.fovea_clicked:
                # まだクリックしていない場合: 画像全体を表示してクリックを受け付ける
                st.info("👆 Click the center of the fovea on the image")
                # 確認画像と同じサイズで表示（一瞬で切り替わるように）
                value = streamlit_image_coordinates(pil_img, key=f"fovea_{idx}", width=orig_w)
                
                if value is not None:
                    display_w = value.get('width', orig_w)
                    display_h = value.get('height', orig_h)
                    if not display_w:
                        display_w = orig_w
                    if not display_h:
                        display_h = orig_h
                    
                    # Check for zero division
                    if display_w <= 0 or display_h <= 0:
                        st.error("❌ Invalid image display dimensions")
                        st.stop()
                    
                    x_click = value.get('x', 0)
                    y_click = value.get('y', 0)
                    fx = int(x_click * (orig_w / display_w))
                    fy = int(y_click * (orig_h / display_h))
                    
                    # クリック位置を保存して画面を再描画
                    st.session_state.fovea_clicked[idx] = (fx, fy)
                    st.rerun()
            
            else:
                # クリック済み: 確認画像とボタンを表示
                fx, fy = st.session_state.fovea_clicked[idx]
                
                # ROI範囲を描画した確認画像を作成
                preview_img = np.array(pil_img.convert('RGB'))
                scale_px_per_mm_h = orig_w / scan_w
                
                # 3.0mm ROI範囲を描画
                roi_width_px = int(3.0 * scale_px_per_mm_h)
                roi_start_x = max(0, int(fx - roi_width_px / 2))
                roi_end_x = min(orig_w, roi_start_x + roi_width_px)
                
                # ROI範囲を青い線で描画
                cv2.line(preview_img, (roi_start_x, 0), (roi_start_x, orig_h), (255, 0, 0), 3)
                cv2.line(preview_img, (roi_end_x, 0), (roi_end_x, orig_h), (255, 0, 0), 3)
                
                # 中心点を緑の円で描画
                cv2.circle(preview_img, (fx, fy), 10, (0, 255, 0), -1)
                
                st.success("✅ Fovea position selected!")
                
                # 画像とボタンを重ねて表示するためのコンテナ
                img_container = st.container()
                with img_container:
                    # 画像を表示
                    st.image(preview_img, caption="Confirmation: Blue lines = 3.0mm ROI range, Green circle = Fovea center", width=orig_w)
                
                # Reselectボタンを画像の右上に重ねる（ネガティブマージンを使用）
                st.markdown("""
                <style>
                    .reselect-overlay {
                        margin-top: -50px;
                        margin-bottom: 10px;
                        display: flex;
                        justify-content: flex-end;
                        padding-right: 10px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="reselect-overlay">', unsafe_allow_html=True)
                if st.button("🔄 Reselect", key=f"reselect_{idx}", type="secondary"):
                    del st.session_state.fovea_clicked[idx]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confirm & Analyze と Skip This Image ボタンを配置
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm & Analyze", type="primary", key=f"confirm_{idx}", use_container_width=True):
                        # --- Auto-initialization logic ---
                        if 'processor' not in st.session_state:
                            with st.spinner("🔄 Loading AI model for the first time..."):
                                st.session_state.processor = CVIProcessor()
                                st.session_state.processor.initialize_model()
                        
                        with st.spinner(f"🔬 Analyzing {filename}..."):
                            res, v_cvi, v_dcvi, p_cvi, p_dcvi, hybrid = st.session_state.processor.process_image(
                                pil_img, filename, fx, fy, scan_w, depth_r
                            )
                            
                            if res is not None:
                                st.session_state.results.append(res)
                                if v_cvi is not None:
                                    st.session_state.vis_files[f"CVI_{filename}"] = v_cvi
                                if v_dcvi is not None:
                                    st.session_state.vis_files[f"DCVI_{filename}"] = v_dcvi
                                if p_cvi is not None:
                                    st.session_state.vis_files[f"CVI_plain_{filename}"] = p_cvi
                                if p_dcvi is not None:
                                    st.session_state.vis_files[f"DCVI_plain_{filename}"] = p_dcvi
                                if hybrid is not None:
                                    st.session_state.vis_files[f"CVI_Dlumen_overlay_{filename}"] = hybrid
                                st.session_state.current_idx += 1
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to process image")
                
                with col2:
                    if st.button("⏭️ Skip This Image", key=f"skip_{idx}", use_container_width=True):
                        st.warning(f"⏭️ Skipped: {filename}")
                        st.session_state.current_idx += 1
                        st.rerun()
        else:
            st.success("🎉 Analysis Complete!")
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df)
        
            # --- Folder Batchモード: ファイルシステムに保存 ---
            if mode == "Folder Batch" and output_folder:
                try:
                    # 出力フォルダー作成（入力フォルダー名を含むサブフォルダー）
                    input_folder_name = Path(input_folder).name if input_folder else "output"
                    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_root = Path(output_folder) / f"{input_folder_name}_CVI_{now_str}"
                    out_root.mkdir(parents=True, exist_ok=True)
                
                    # CSV保存
                    df.to_csv(out_root / "cvi_results.csv", index=False)
                
                    # パラメータログ保存
                    log_df = pd.DataFrame({
                        'Parameter': ['Macro Name', 'Developer', 'Date', 'Time',
                                      'Original Folder', 'Analyzed by', 'Analysis mode',
                                      'Scan Width (H-mm)', 'Depth Range (V-mm)',
                                      'Total Images', 'Success'],
                        'Value': [APP_NAME, DEVELOPER,
                                  datetime.now().strftime('%Y-%m-%d'),
                                  datetime.now().strftime('%H:%M:%S'),
                                  input_folder_name, author, 'マニュアル解析',
                                  scan_w, depth_r,
                                  len(uploaded_files), len(st.session_state.results)]
                    })
                    log_df.to_csv(out_root / f"{input_folder_name}_Parameter.csv", index=False)
                
                    # 可視化画像保存
                    for fname, fdata in st.session_state.vis_files.items():
                        if fdata is not None:
                            with open(out_root / fname, 'wb') as f:
                                f.write(fdata)
                
                    st.success(f"✅ Results saved to: `{out_root}`")
                
                    if st.button("🔄 Start New Batch"):
                        for key in ['current_idx', 'results', 'vis_files', 'fovea_clicked',
                                    'auto_analysis_complete', '_auto_fp']:
                            if key in st.session_state: 
                                del st.session_state[key]
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error saving to output folder: {str(e)}")
        
            # --- File Uploadモード: Zipダウンロード ---
            elif mode == "File Upload":
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    # CSV追加
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    zip_file.writestr("cvi_results.csv", csv_data)
                
                    # パラメータログ追加
                    log_df = pd.DataFrame({
                        'Parameter': ['App', 'Author', 'Date', 'Time', 'Analysis mode',
                                      'Scan Width', 'Depth Range'],
                        'Value': [APP_NAME, author, datetime.now().strftime('%Y-%m-%d'),
                                 datetime.now().strftime('%H:%M:%S'), 'マニュアル解析',
                                 scan_w, depth_r]
                    })
                    zip_file.writestr("parameters.csv", log_df.to_csv(index=False).encode('utf-8'))
                
                    # 可視化画像追加
                    for fname, fdata in st.session_state.vis_files.items():
                        if fdata is not None:
                            zip_file.writestr(fname, fdata)
            
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download All Results (Zip)",
                        data=zip_buffer.getvalue(),
                        file_name=f"CVI_Results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip",
                        type="primary"
                    )
            
                with col2:
                    if st.button("🔄 Reset Session"):
                        for key in ['current_idx', 'results', 'vis_files', 'fovea_clicked',
                                    'auto_analysis_complete', '_auto_fp']:
                            if key in st.session_state: 
                                del st.session_state[key]
                        st.rerun()
else:
    if mode == "Folder Batch":
        st.warning("⚠️ Please specify input and output folder paths in the sidebar")
    else:
        st.warning("⚠️ Please upload B-scan images in the sidebar")