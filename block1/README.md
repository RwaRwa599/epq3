# Block 1 — Document Normalization and ROI Extraction Framework

A standalone, high-precision document normalization engine for clinical laboratory request forms.

---

## 🌟 Key Features

- **Automatic Quad Detection & Homography Warp**: Detects page boundaries using adaptive multi-scale morphological filtering and projects distorted phone photos / scans into standard canonical space.
- **Orientation Correction**: Automatically detects and fixes upside-down pages (0° vs 180° orientation scoring).
- **Dual-Template Support**: Supports both **v0 digital forms** (124 checkboxes, 9 handwriting fields) and **v1 clinic printed forms** (138 checkboxes, 12 handwriting fields) with automatic revision detection (`_pick_revision`).
- **Sub-Pixel Anchor Snapping (`snap_overlay`)**: Snaps individual checkboxes to actual detected ink on the page, eliminating drift caused by lens distortion, paper folds, or lighting shadows.
- **CLAHE Illumination Normalization**: Normalizes shadows, flash gradients, and uneven ambient lighting across individual field crops.
- **Google Colab Ready**: 100% self-contained with no external path dependencies; includes ready-to-run Jupyter notebook.

---

## 📁 Folder Structure

```
block1/
├── Block_1_Document_Normalization.ipynb  # Interactive Google Colab notebook
├── README.md                              # This file
├── requirements.txt                       # Standalone pip dependencies
├── pyproject.toml                         # Standard package configuration
├── run_demo.py                            # Standalone CLI entrypoint
├── templates/                             # Canonical coordinate templates
│   ├── lab_request_canonical.json         # v0 digital template (2048x1720)
│   └── lab_request_v1_canonical.json      # v1 clinic print template (2048x1754)
├── samples/                               # Sample images
│   └── lab_request_v0_blank.png           # Clean synthetic blank sample
└── med_doc/                               # Core Python package
    ├── __init__.py
    ├── paths.py
    ├── schemas.py                         # Pydantic models (FieldCrop, NormalizedDocumentResult)
    ├── template.py                        # Template loading & coordinate calculations
    └── normalization/
        ├── __init__.py
        ├── warp.py                        # Page quad detection & perspective transform
        ├── align.py                       # Anchor matching & sub-pixel fine snapping
        ├── detect.py                      # Robust checkbox detection on photos
        ├── crops.py                       # CLAHE normalization & crop extraction
        ├── viz.py                         # Visual bounding box overlays
        └── pipeline.py                    # Main public API (normalize_document)
```

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
cd block1
pip install -r requirements.txt
```

### 2. Run Demo CLI
```bash
python run_demo.py samples/lab_request_v0_blank.png
```

This will output:
- Canonical canvas size and orientation
- Total extracted checkboxes and handwriting ROIs
- Bounding box debug overlay to `outputs/lab_request_v0_blank_overlay.jpg`
- Extracted crops to `outputs/lab_request_v0_blank_crops/`

---

## ☁️ Running on Google Colab

1. Upload the `block1/` folder to Google Colab, or clone the repository directly:
   ```python
   !git clone https://github.com/RwaRwa599/epq3.git
   %cd epq3/block1
   !pip install -r requirements.txt
   ```
2. Open `Block_1_Document_Normalization.ipynb`.
3. Run all cells: you can upload your own clinic document photo or use the built-in sample to visualize overlays and download extracted crops as a ZIP.

---

## 💻 Python API Usage

```python
from PIL import Image
from med_doc.normalization.pipeline import normalize_document

# Load any image (PIL Image, numpy array, or file path)
img = Image.open("path/to/clinic_photo.png")

# Run normalization
result = normalize_document(img, document_id="patient_form_01")

print(f"Confidence: {result.alignment_confidence:.2f}")
print(f"Checkboxes: {len(result.checkbox_crops)}")
print(f"Handwriting: {len(result.handwriting_crops)}")

# Access individual crops
hba1c_crop = result.checkbox_crops["hba1c"]
raw_rgb = hba1c_crop.raw_image
norm_rgb = hba1c_crop.normalized_image
```
