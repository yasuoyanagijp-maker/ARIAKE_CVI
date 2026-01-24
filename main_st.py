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
    from choseg import inference, utils
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

# ==========================================
# 解析ロジック (CVIProcessor)
# ==========================================
class CVIProcessor:
    def __init__(self):
        self.model = None
        
    def initialize_model(self):
        if self.model is None:
            self.model = inference.DeepGPET()
    
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

    def process_image(self, pil_img, filename, fovea_x, fovea_y, scan_width_mm, depth_range_mm):
        try:
            img_gray = np.array(pil_img.convert('L'))
            height, width = img_gray.shape
            
            # Validate fovea coordinates
            fovea_x = max(0, min(int(fovea_x), width - 1))
            fovea_y = max(0, min(int(fovea_y), height - 1))
            
            scale_px_per_mm_h = width / scan_width_mm
            scale_px_per_mm_v = height / depth_range_mm
            
            # Segmentation
            img_seg = self.model(Image.fromarray(img_gray))
            if isinstance(img_seg, np.ndarray):
                mask = (img_seg > 0.5).astype(np.uint8) * 255
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
            
            # Denoising & Thresholding
            denoised_img = self.denoise_image_global(img_gray)
            global_lum_mask = self.niblack_threshold(img_gray, mask)
            global_dlum_mask = self.niblack_threshold(denoised_img, mask)
            
            global_lum_mask = cv2.medianBlur(global_lum_mask, 3)
            global_dlum_mask = cv2.medianBlur(global_dlum_mask, 3)
            global_lum_mask[mask == 0] = 0
            global_dlum_mask[mask == 0] = 0
            
            # Get Metrics
            tca15, ct15, cvi15, _, _ = self.get_roi_metrics(global_lum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 1.5)
            tca30, ct30, cvi30, _, lum30_roi = self.get_roi_metrics(global_lum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 3.0)
            dtca15, dct15, dcvi15, _, _ = self.get_roi_metrics(global_dlum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 1.5)
            dtca30, dct30, dcvi30, _, dlum30_roi = self.get_roi_metrics(global_dlum_mask, mask, fovea_x, scale_px_per_mm_h, scale_px_per_mm_v, 3.0)
        
            # Generate Visualization Images for Memory
            def create_vis_buffer(base_img, l_mask_roi):
                vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
                roi_w_px = int(3.0 * scale_px_per_mm_h)
                rx_start = max(0, int(fovea_x - roi_w_px / 2))
                rx_end = min(width, rx_start + roi_w_px)
                full_v_mask = np.zeros_like(mask)
                if l_mask_roi is not None: full_v_mask[:, rx_start:rx_end] = l_mask_roi
                vis[full_v_mask > 0] = [0, 0, 0] 
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 255), 1) 
                cv2.circle(vis, (int(fovea_x), int(fovea_y)), 8, (0, 255, 0), -1) 
                cv2.line(vis, (rx_start, 0), (rx_start, height), (255, 0, 0), 2)
                cv2.line(vis, (rx_end, 0), (rx_end, height), (255, 0, 0), 2)
                is_success, buffer = cv2.imencode(".jpg", vis)
                return buffer.tobytes() if is_success else None

            vis_cvi = create_vis_buffer(img_gray, lum30_roi)
            vis_dcvi = create_vis_buffer(denoised_img, dlum30_roi)

            stats = {
                'Image ID': filename,
                'TCA 1.5mm (mm2)': round(tca15, 4),
                'CT 1.5mm (um)': round(ct15, 2),
                'CVI 1.5mm (%)': round(cvi15, 2),
                'D-CVI 1.5mm (%)': round(dcvi15, 2),
                'TCA 3.0mm (mm2)': round(tca30, 4),
                'CT 3.0mm (um)': round(ct30, 2),
                'CVI 3.0mm (%)': round(cvi30, 2),
                'D-CVI 3.0mm (%)': round(dcvi30, 2)
            }
            return stats, vis_cvi, vis_dcvi
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {str(e)}")
            return None, None, None

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title=APP_NAME, layout="wide")

# セッション状態初期化
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'results' not in st.session_state:
    st.session_state.results = []
if 'vis_files' not in st.session_state:
    st.session_state.vis_files = {} # filename -> bytes

# --- 認証画面 ---
if not st.session_state.authenticated:
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

# サイドバー設定
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 処理モード選択
    mode = st.radio("Processing Mode", ["File Upload", "Folder Batch"])
    
    if mode == "File Upload":
        uploaded_files = st.file_uploader(
            "Upload B-scan Images", 
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], 
            accept_multiple_files=True
        )
    else:
        input_folder = st.text_input("Input Folder Path", placeholder="/path/to/input")
        output_folder = st.text_input("Output Folder Path", placeholder="/path/to/output")
        uploaded_files = []
        if input_folder and os.path.exists(input_folder):
            try:
                image_files = sorted(
                    list(Path(input_folder).glob("*.png")) + 
                    list(Path(input_folder).glob("*.jpg")) + 
                    list(Path(input_folder).glob("*.jpeg")) +
                    list(Path(input_folder).glob("*.tif")) +
                    list(Path(input_folder).glob("*.tiff"))
                )
                st.info(f"✅ Found {len(image_files)} images")
                uploaded_files = image_files
            except Exception as e:
                st.error(f"❌ Error reading folder: {str(e)}")
        elif input_folder:
            st.error(f"❌ Folder path does not exist: {input_folder}")
    
    scan_w = st.number_input("Scan Width (mm)", value=12.0, min_value=0.1)
    depth_r = st.number_input("Depth Range (mm)", value=2.6, min_value=0.1)
    author = st.text_input("Author", value="YY")
    
    st.divider()
    
    # モデルのロード状態を表示
    if 'processor' in st.session_state:
        st.success("✅ AI Model Loaded")
    else:
        st.warning("⏳ Model loads on first analysis")
    
    # 手動初期化ボタンもオプションとして残す
    if st.button("Re-initialize AI Model"):
        with st.spinner("Loading DeepGPET..."):
            st.session_state.processor = CVIProcessor()
            st.session_state.processor.initialize_model()
            st.success("✅ Model Initialized!")

# 解析処理
if uploaded_files:
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
        
        st.info("👆 Click the center of the fovea on the image")
        
        # 画像を全幅で表示
        value = streamlit_image_coordinates(pil_img, key=f"fovea_{idx}", width=orig_w)
        
        if value is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clicked X", value['x'])
            with col2:
                st.metric("Clicked Y", value['y'])
        
        # ROIプレビューを追加
        if value is not None:
            display_w = value['width']
            display_h = value['height']
            
            # Check for zero division
            if display_w <= 0 or display_h <= 0:
                st.error("❌ Invalid image display dimensions")
                st.stop()
            
            fx = int(value['x'] * (orig_w / display_w))
            fy = int(value['y'] * (orig_h / display_h))
            
            # ROI範囲を描画したプレビュー画像を作成
            preview_img = np.array(pil_img.convert('RGB'))
            scale_px_per_mm_h = orig_w / scan_w
            
            # 3.0mm ROI範囲を描画
            roi_width_px = int(3.0 * scale_px_per_mm_h)
            roi_start_x = max(0, int(fx - roi_width_px / 2))
            roi_end_x = min(orig_w, roi_start_x + roi_width_px)
            
            # ROI範囲を青い線で描画
            cv2.line(preview_img, (roi_start_x, 0), (roi_start_x, orig_h), (255, 0, 0), 2)
            cv2.line(preview_img, (roi_end_x, 0), (roi_end_x, orig_h), (255, 0, 0), 2)
            
            # 中心点を緑の円で描画
            cv2.circle(preview_img, (fx, fy), 8, (0, 255, 0), -1)
            
            st.image(preview_img, caption="ROI Preview (Blue: 3.0mm range, Green: Fovea center)", width=orig_w)
            
            if st.button("✅ Confirm & Analyze", type="primary"):
                # --- Auto-initialization logic ---
                if 'processor' not in st.session_state:
                    with st.spinner("🔄 Loading AI model for the first time..."):
                        st.session_state.processor = CVIProcessor()
                        st.session_state.processor.initialize_model()
                
                with st.spinner(f"🔬 Analyzing {filename}..."):
                    res, v_cvi, v_dcvi = st.session_state.processor.process_image(
                        pil_img, filename, fx, fy, scan_w, depth_r
                    )
                    
                    if res is not None:
                        st.session_state.results.append(res)
                        if v_cvi is not None:
                            st.session_state.vis_files[f"CVI_{filename}"] = v_cvi
                        if v_dcvi is not None:
                            st.session_state.vis_files[f"DCVI_{filename}"] = v_dcvi
                        st.session_state.current_idx += 1
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to process image")
    else:
        st.success("🎉 Analysis Complete!")
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df)
        
        # --- Zip ファイル作成とダウンロード ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            # CSV追加
            csv_data = df.to_csv(index=False).encode('utf-8')
            zip_file.writestr("cvi_results.csv", csv_data)
            
            # パラメータログ追加
            log_df = pd.DataFrame({
                'Parameter': ['App', 'Author', 'Date', 'Time', 'Scan Width', 'Depth Range'],
                'Value': [APP_NAME, author, datetime.now().strftime('%Y-%m-%d'), 
                         datetime.now().strftime('%H:%M:%S'), scan_w, depth_r]
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
                for key in ['current_idx', 'results', 'vis_files']:
                    if key in st.session_state: 
                        del st.session_state[key]
                st.rerun()
else:
    st.warning("⚠️ Please upload B-scan images or specify input folder in the sidebar")