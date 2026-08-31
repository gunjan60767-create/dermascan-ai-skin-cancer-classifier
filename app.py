import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

# Page setup
st.set_page_config(
    page_title="DermaScan AI | Clinical Pathology Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Clinical Styling
st.markdown("""
<style>
    /* Medical Background Theme with Soft Grid */
    .stApp {
        background-color: #0d1520;
        background-image: 
            radial-gradient(at 10% 20%, rgba(13, 148, 136, 0.15) 0px, transparent 50%),
            radial-gradient(at 90% 80%, rgba(14, 165, 233, 0.12) 0px, transparent 50%),
            linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 30px 30px, 30px 30px;
        color: #e2e8f0;
    }

    /* Glassmorphism Header Container */
    .medical-header {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(13, 148, 136, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .medical-badge {
        display: inline-block;
        background: rgba(13, 148, 136, 0.2);
        color: #2dd4bf;
        border: 1px solid #0d9488;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Clinical Cards */
    .diag-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Risk Badges */
    .badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-mod {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-benign {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }

    /* File Uploader Box Styling */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 2px dashed rgba(13, 148, 136, 0.4);
        border-radius: 14px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Diagnostic Class Data
CONDITION_INFO = {
    "melanoma": {
        "title": "Melanoma (MEL)",
        "badge": '<span class="badge-high">🚨 High Risk • Malignant</span>',
        "desc": "An aggressive cutaneous malignancy derived from melanocytes. Immediate biopsy and surgical excision consult recommended."
    },
    "melanocytic nevi": {
        "title": "Melanocytic Nevus (NV)",
        "badge": '<span class="badge-benign">🟢 Benign • Common Mole</span>',
        "desc": "Standard benign melanocytic proliferation. Periodic self-checks via the clinical ABCDE criteria recommended."
    },
    "basal cell carcinoma": {
        "title": "Basal Cell Carcinoma (BCC)",
        "badge": '<span class="badge-mod">⚠️ Moderate Risk • Malignant</span>',
        "desc": "Slow-growing non-melanoma skin cancer. Locally invasive; evaluate for Mohs micrographic surgery or topical therapy."
    },
    "actinic keratoses": {
        "title": "Actinic Keratosis (AKIEC)",
        "badge": '<span class="badge-mod">⚠️ Pre-Malignant Patch</span>',
        "desc": "Dysplastic epidermal lesion induced by chronic ultraviolet radiation. Precursor to Squamous Cell Carcinoma."
    },
    "benign keratosis-like lesions": {
        "title": "Benign Keratosis (BKL)",
        "badge": '<span class="badge-benign">🟢 Benign Growth</span>',
        "desc": "Seborrheic keratosis or solar lentigo. Non-neoplastic keratinocyte proliferation without malignant potential."
    },
    "dermatofibroma": {
        "title": "Dermatofibroma (DF)",
        "badge": '<span class="badge-benign">🟢 Benign Nodule</span>',
        "desc": "Benign dermal histiocytoma/fibroma commonly presenting after minor trauma or insect bites."
    },
    "vascular lesions": {
        "title": "Vascular Lesion (VASC)",
        "badge": '<span class="badge-benign">🟢 Benign Vascular</span>',
        "desc": "Angiomas, pyogenic granulomas, or vascular malformations. Generally benign vascular ectasias."
    }
}

@st.cache_resource
def load_medical_pipeline():
    return pipeline(
        "image-classification",
        model="Anwarkh1/Skin_Cancer-Image_Classification",
        device=-1
    )

pipe = load_medical_pipeline()

# Hospital/Clinical Header
st.markdown("""
<div class="medical-header">
    <div class="medical-badge">Clinical Diagnostic Assistant • ISIC AI-07</div>
    <h1 style='margin: 0; font-size: 2.2rem; color: #f8fafc;'>🏥 DermaScan AI: Skin Pathology Screening</h1>
    <p style='margin-top: 8px; color: #94a3b8; font-size: 1rem;'>Deep Vision Transformer (ViT) Diagnostic Model Trained on the HAM10000 Dataset</p>
</div>
""", unsafe_allow_html=True)

# Sidebar clinical guide
with st.sidebar:
    st.markdown("### 📋 Diagnostic Telemetry")
    st.markdown("""
    - **Engine:** `ViT-Base-Patch16-224`
    - **Validation Modality:** Dermoscopy (Polarized / Contact)
    - **Training Standard:** Harvard Dataverse / ISIC Archive
    """)
    st.divider()
    st.markdown("### ⚠️ Clinical Advisory")
    st.caption("This system is engineered for secondary triage and educational screening. It does not replace a histology biopsy performed by a certified pathologist or dermatologist.")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload Dermoscopy Lesion Image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 Input Dermoscopic Scan")
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Executing tissue feature extraction & pathology classification..."):
        results = pipe(img)
        top_prediction = results[0]
        pred_label_raw = top_prediction['label'].lower().strip()
        top_confidence = top_prediction['score'] * 100

        matched_info = None
        for key in CONDITION_INFO:
            if key in pred_label_raw or pred_label_raw in key:
                matched_info = CONDITION_INFO[key]
                break
        
        if not matched_info:
            matched_info = {
                "title": top_prediction['label'].title(),
                "badge": '<span class="badge-mod">ℹ️ Review Required</span>',
                "desc": "Classified via ISIC feature correlation."
            }

    with col2:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        st.markdown("#### 🩺 Clinical Diagnostic Assessment")
        st.markdown(f"<h2 style='color: #38bdf8; margin: 4px 0;'>{matched_info['title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Classification Status:** {matched_info['badge']}", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Model Confidence", f"{top_confidence:.2f}%")
        metric_col2.metric("Target Resolution", "224x224 RGB")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Pathology Profile:** {matched_info['desc']}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="diag-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Differential Diagnosis Confidence Spectrum")
    chart_data = {r['label'].title(): round(r['score'] * 100, 2) for r in results}
    st.bar_chart(chart_data)
    st.markdown('</div>', unsafe_allow_html=True)
