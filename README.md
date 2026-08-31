# 🔬 DermaScan AI: Skin Cancer Diagnostic & Clinical Risk Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning Computer Vision web application built with **PyTorch**, **Vision Transformers (ViT-16)**, and **Streamlit**. The system detects and classifies 7 clinical skin lesion pathologies from dermatoscopic imagery and provides multi-class risk assessment stratification.

---

## 🌐 Live Application
👉 **[Click here to test the Live Demo](https://YOUR-APP-NAME.streamlit.app)**

---

## ✨ Key Capabilities
- **Vision Transformer Architecture:** Fine-tuned on the **HAM10000 / ISIC** clinical skin lesion benchmark dataset.
- **7-Class Pathology Screening:**
  - `Melanoma (MEL)` (🚨 High Risk)
  - `Melanocytic Nevus (NV)` (🟢 Benign)
  - `Basal Cell Carcinoma (BCC)` (⚠️ Moderate Risk)
  - `Actinic Keratosis (AKIEC)` (⚠️ Moderate Risk)
  - `Benign Keratosis (BKL)` (🟢 Benign)
  - `Dermatofibroma (DF)` (🟢 Benign)
  - `Vascular Lesion (VASC)` (🟢 Benign)
- **Clinical Risk Assessment:** Categorizes conditions into High Risk, Moderate Risk, or Benign.
- **Multi-Class Confidence Distributions:** Real-time probability bar charts across all condition classes.

---

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch, Hugging Face Transformers (`google/vit-base-patch16-224`)
- **Computer Vision & Data:** Pillow, NumPy
- **Frontend & Deployment:** Streamlit Community Cloud
