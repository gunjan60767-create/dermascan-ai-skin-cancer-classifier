import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="DermaScan AI - Clinical Skin Lesion Diagnostic",
    page_icon="🔬",
    layout="wide"
)

# Medical clinical context mapping for HAM10000 / ISIC classes
CONDITION_INFO = {
    "melanoma": {
        "title": "Melanoma (MEL)",
        "risk": "🚨 High Risk (Malignant)",
        "desc": "A serious, invasive form of skin cancer originating in melanocytes. Urgent dermatologist evaluation and biopsy are recommended."
    },
    "melanocytic nevi": {
        "title": "Melanocytic Nevus (NV)",
        "risk": "🟢 Benign (Non-Cancerous)",
        "desc": "A standard, benign mole composed of melanocytes. Common and generally harmless; monitor periodically using the ABCDE criteria."
    },
    "basal cell carcinoma": {
        "title": "Basal Cell Carcinoma (BCC)",
        "risk": "⚠️ Moderate Risk (Malignant)",
        "desc": "The most common form of skin cancer. Locally destructive but rarely spreads to distant organs if treated early."
    },
    "actinic keratoses": {
        "title": "Actinic Keratosis / Bowen's Disease (AKIEC)",
        "risk": "⚠️ Moderate Risk (Pre-Malignant)",
        "desc": "A rough, scaly patch caused by long-term UV exposure. Has potential to evolve into squamous cell carcinoma if untreated."
    },
    "benign keratosis-like lesions": {
        "title": "Benign Keratosis (BKL)",
        "risk": "🟢 Benign (Non-Cancerous)",
        "desc": "Includes seborrheic keratoses and solar lentigines. Harmless skin growths common in older adults."
    },
    "dermatofibroma": {
        "title": "Dermatofibroma (DF)",
        "risk": "🟢 Benign (Non-Cancerous)",
        "desc": "A harmless, firm fibrous nodule commonly found on the arms and legs."
    },
    "vascular lesions": {
        "title": "Vascular Lesion (VASC)",
        "risk": "🟢 Benign (Non-Cancerous)",
        "desc": "Benign proliferation of blood vessels (such as cherry angiomas or pyogenic granulomas)."
    }
}

@st.cache_resource
def load_medical_pipeline():
    # Pre-trained Vision Transformer fine-tuned on HAM10000 skin dataset
    return pipeline(
        "image-classification",
        model="Anwarkh1/Skin_Cancer-Image_Classification",
        device=-1
    )

pipe = load_medical_pipeline()

# UI Layout
st.markdown("<h1 style='text-align: center;'>🔬 DermaScan AI: Skin Cancer Diagnostic</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Pretrained Vision Transformer (ViT) Benchmark on HAM10000 / ISIC Medical Data</p>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("ℹ️ Model Architecture")
    st.info(
        "**Core Engine:** Google Vision Transformer (ViT-16)\n\n"
        "**Dataset:** HAM10000 Skin Lesion Dataset\n\n"
        "**Scope:** 7 Clinical Pathologies\n\n"
        "⚠️ *Educational demo only. Always consult a doctor for diagnosis.*"
    )

uploaded_file = st.file_uploader("Upload a Dermoscopic Skin Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Uploaded Lesion")
        st.image(img, use_container_width=True)

    with st.spinner("Classifying tissue patterns via Vision Transformer..."):
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
                "risk": "ℹ️ Diagnostic Review",
                "desc": "Classified via ISIC benchmark feature extraction."
            }

    with col2:
        st.subheader("🩺 Diagnostic Findings")
        st.metric(label="Predicted Condition", value=matched_info["title"])
        st.metric(label="Model Confidence", value=f"{top_confidence:.2f}%")
        st.metric(label="Risk Assessment", value=matched_info["risk"])
        st.info(f"**Description:** {matched_info['desc']}")

    st.divider()
    st.subheader("📊 Full Probability Breakdown Across All 7 Pathologies")
    
    chart_data = {r['label'].title(): round(r['score'] * 100, 2) for r in results}
    st.bar_chart(chart_data)
