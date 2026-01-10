ARIAKE_CVI is a Streamlit-based web app for automated choroidal vascularity index (CVI) and related metrics from OCT B‑scan images using a DeepGPET segmentation model.[1]

## Overview

- **ARIAKE_CVI** runs as a Streamlit app and loads a DeepGPET-based model (with a MockModel fallback) to segment the choroid on grayscale B‑scan images.[1]
- The app computes TCA, CT, CVI, and D‑CVI within 1.5 mm and 3.0 mm fovea-centered regions and exports both metrics and visualization images.[1]

## Features

- Simple access control via a shared access key (`ACCESS_KEY = "ariake2024"`) and login screen.[1]
- Multi-file upload of B‑scan images (png/jpg/jpeg/tif/tiff) with manual fovea selection per image via `streamlit_image_coordinates`.[1]
- Automatic first-time model initialization plus optional manual re-initialization from the sidebar.[1]
- Per-image computation of:
  - TCA 1.5/3.0 mm (mm²), CT 1.5/3.0 mm (µm), CVI 1.5/3.0 mm (%), D‑CVI 1.5/3.0 mm (%).[1]
- On-the-fly visualization:
  - Original and denoised images with overlaid choroid segmentation, fovea marker, and 3.0‑mm ROI, saved as JPEGs (CVI_*, DCVI_*).[1]
- One-click download of:
  - `cvi_results.csv` with all image metrics, `parameters.csv` with app/author/date/scan width/depth, and all visualization images packed into a ZIP file.[1]

## Requirements

- **Python packages**: `streamlit`, `numpy`, `opencv-python`, `Pillow`, `pandas`, `streamlit-image-coordinates`.[1]
- **Project structure** (expected):  
  - `main_st.py` (this app file) and a `deepgpet/` directory containing `choseg/inference.py` and related utilities for `DeepGPET`.[1]
- If `choseg.inference` cannot be imported, the app falls back to a dummy `MockModel` that outputs an empty mask (for UI testing only, not for real analysis).[1]

## How to Run

1. Place `main_st.py` and the `deepgpet` directory in the same folder (so `deepgpet_path = Path(__file__).parent / 'deepgpet'` is valid).[1]
2. Install dependencies, for example:  
   ```bash
   pip install streamlit numpy opencv-python pillow pandas streamlit-image-coordinates
   ```  
3. Start the app:  
   ```bash
   streamlit run main_st.py
   ```  
4. Open the URL shown in the terminal, enter the access key (`ariake2024`) on the login screen, and proceed to the main interface.[1]

## Usage Workflow

1. In the sidebar, upload B‑scan images and set scan width (mm), depth range (mm), and author name.[1]
2. For each uploaded image:
   - Click on the foveal center in the displayed B‑scan.
   - Click “Confirm & Analyze” to trigger DeepGPET-based segmentation and CVI computation.[1]
3. After all images are processed, review the results table on the main page.[1]
4. Click “Download All Results (Zip)” to obtain the metrics CSVs and visualization images; use “Reset Session” to clear state and start a new batch.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/18140865/df7ecf91-3bdd-4a23-be64-106d18106427/main_st.py)
