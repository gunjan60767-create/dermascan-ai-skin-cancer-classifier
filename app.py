import streamlit as st
import torch
from PIL import Image
from transformers import pipeline
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="AuraHealth • AI Clinical Dermatology OS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical-Grade UI CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hospital Dark Operating System Backdrop */
    .stApp {
        background-color: #060b13;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(13, 148, 136, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(14, 165, 233, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 32px 32px, 32px 32px;
        color: #f1f5f9;
    }

    /* Top Hospital Navigation Bar */
    .hospital-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(13, 22, 38, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(45, 212, 191, 0.2);
        border-radius: 16px;
        padding: 16px 28px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2dd4bf 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #34d399;
    }
    .status-pulse {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Glassmorphic Clinical Cards */
    .clinical-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        transition: border 0.3s ease;
    }
    .clinical-card:hover {
        border-color: rgba(45, 212, 191, 0.4);
    }
    .card-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Severity Badges */
    .badge-malignant {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.15));
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 8px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 0 14px rgba(239, 68, 68, 0.25);
    }
    .badge-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(180, 83, 9, 0.15));
        color: #fde68a;
        border: 1px solid #f59e0b;
        padding: 8px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.25);
    }
    .badge-benign {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(4, 120, 87, 0.15));
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 8px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.25);
    }

    /* Stat Highlight Boxes */
    .stat-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Hide standard Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Diagnostic Pathology Metadata
CONDITION_INFO = {
    "melanoma": {
        "title": "Melanoma (MEL)",
        "badge": '<span class="badge-malignant">🚨 Critical • High Risk Malignant</span>',
        "icd": "ICD-10-CM C43.9",
        "action": "Immediate dermatological referral, staging assessment, and histological punch/excisional biopsy required.",
        "desc": "Aggressive cutaneous malignancy originating from atypical melanocytes. Marked by cellular pleomorphism and high metastatic potential."
    },
    "melanocytic nevi": {
        "title": "Melanocytic Nevus (NV)",
        "badge": '<span class="badge-benign">🟢 Normal • Benign Common Mole</span>',
        "icd": "ICD-10-CM D22.9",
        "action": "Routine annual skin check. Instruct patient on standard ABCDE self-monitoring.",
        "desc": "Benign melanocytic proliferation characterized by orderly pigment architecture and symmetric boundary margins."
    },
    "basal cell carcinoma": {
        "title": "Basal Cell Carcinoma (BCC)",
        "badge": '<span class="badge-moderate">⚠️ Warning • Moderate Risk Malignant</span>',
        "icd": "ICD-10-CM C44.91",
        "action": "Schedule for standard surgical excision or Mohs micrographic surgery.",
        "desc": "Slow-growing non-melanoma skin cancer arising from the basal layer of epidermis with low distant metastasis rates."
    },
    "actinic keratoses": {
        "title": "Actinic Keratosis (AKIEC)",
        "badge": '<span class="badge-moderate">⚠️ Warning • Pre-Malignant Lesion</span>',
        "icd": "ICD-10-CM L57.0",
        "action": "Consider cryotherapy, topical 5-fluorouracil, or photodynamic field treatment.",
        "desc": "Dysplastic keratinocyte lesion induced by cumulative ultraviolet irradiation; recognized precursor to invasive squamous cell carcinoma."
    },
    "benign keratosis-like lesions": {
        "title": "Benign Keratosis (BKL)",
        "badge": '<span class="badge-benign">🟢 Normal • Benign Epidermal Growth</span>',
        "icd": "ICD-10-CM L82.1",
        "action": "No medical intervention indicated unless mechanically irritated or for cosmetic preference.",
        "desc": "Non-malignant epidermal proliferation comprising seborrheic keratosis, solar lentigines, and lichenoid keratoses."
    },
    "dermatofibroma": {
        "title": "Dermatofibroma (DF)",
        "badge": '<span class="badge-benign">🟢 Normal • Benign Fibrous Nodule</span>',
        "icd": "ICD-10-CM D21.9",
        "action": "Clinical observation only. Reassurance of non-malignant nature.",
        "desc": "Benign cutaneous fibrohistiocytic lesion featuring hyperpigmentation and typical positive pinch/dimple sign."
    },
    "vascular lesions": {
        "title": "Vascular Lesion (VASC)",
        "badge": '<span class="badge-benign">🟢 Normal • Benign Vascular Malformation</span>',
        "icd": "ICD-10-CM D18.0",
        "action": "Monitor for acute changes or ulceration. Elective laser ablation if indicated.",
        "desc": "Benign vascular ectasias comprising cherry angiomas, hemangiomas, and pyogenic granulomas."
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

# Hospital OS Top Navigation Bar
st.markdown("""
<div class="hospital-nav">
    <div class="brand-title">
        <span>🧬</span>
        <span>AuraHealth Clinical OS <span style="font-size: 0.8rem; font-weight: 500; color: #94a3b8;">| DermatoPath v2.4</span></span>
    </div>
    <div class="status-pill">
        <div class="status-pulse"></div>
        <span>ViT-16 Neural Pipeline Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hospital Telemetry Sidebar
with st.sidebar:
    st.markdown("### 🏥 System Telemetry")
    st.markdown("""
    - **Inference Engine:** Google Vision Transformer (`ViT-16`)
    - **Resolution Tensor:** `224 × 224 × 3 RGB`
    - **Clinical Dataset:** Harvard HAM10000 / ISIC
    - **PACS Protocol:** DICOM & Standard RGB
    """)
    st.divider()
    st.markdown("### 🩺 Diagnostic Guide (ABCDE)")
    st.markdown("""
    - **A:** Asymmetry
    - **B:** Border Irregularity
    - **C:** Color Variation
    - **D:** Diameter (>6mm)
    - **E:** Evolving Morphology
    """)
    st.divider()
    st.caption("🔒 **HIPAA/GDPR Advisory:** Local client inference session. No patient biometrics or image data are permanently stored.")

# Main Interface
st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
st.markdown('<div class="card-heading">📂 Clinical Dermoscopy Upload Interface</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Select high-resolution dermatoscopic lesion capture (JPEG, PNG, DICOM-export)", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([5, 7], gap="large")

    with col1:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🔬 High-Resolution Lesion Scan</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(f"<p style='color: #64748b; font-size: 0.8rem; text-align: center; margin-top: 8px;'>Image Dimensions: {img.size[0]} × {img.size[1]} px</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Processing dermoscopic attention maps & multi-class probabilities..."):
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
                "badge": '<span class="badge-moderate">⚠️ Diagnostic Review</span>',
                "icd": "ICD-10-CM R23.8",
                "action": "Manual review required by clinician.",
                "desc": "Classified via ISIC benchmark feature distribution."
            }

    with col2:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🩺 Primary Diagnostic Classification</div>', unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: #38bdf8; font-size: 1.85rem; margin: 0 0 10px 0;'>{matched_info['title']}</h1>", unsafe_allow_html=True)
        st.markdown(f"{matched_info['badge']} &nbsp; <code style='background: rgba(30,41,59,0.8); color: #cbd5e1; padding: 4px 10px; border-radius: 6px; font-family: monospace;'>{matched_info['icd']}</code>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clinical Metrics Grid
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{top_confidence:.1f}%</div>
                <div class="stat-label">Model Confidence Index</div>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">ViT-16</div>
                <div class="stat-label">Transformer Architecture</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Pathology Description:** <span style='color: #cbd5e1;'>{matched_info['desc']}</span>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin-top: 10px; background: rgba(15, 23, 42, 0.8); border-left: 3px solid #38bdf8; padding: 12px; border-radius: 0 8px 8px 0; color: #94a3b8; font-size: 0.9rem;'><strong style='color: #f1f5f9;'>Suggested Protocol:</strong> {matched_info['action']}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Plotly Interactive Modern Healthcare Distribution Chart
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">📊 Multi-Class Probability Spectrum</div>', unsafe_allow_html=True)
    
    labels = [r['label'].title() for r in results]
    scores = [round(r['score'] * 100, 2) for r in results]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[
                [0.0, 'rgba(56, 189, 248, 0.3)'],
                [0.5, 'rgba(45, 212, 191, 0.7)'],
                [1.0, '#38bdf8']
            ],
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{s:.2f}%" for s in scores],
        textposition='auto',
        textfont=dict(color='#ffffff', family='JetBrains Mono')
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(
            title="Probability Percentage (%)",
            titlefont=dict(color="#94a3b8", size=12),
            tickfont=dict(color="#64748b"),
            gridcolor='rgba(255, 255, 255, 0.05)',
            range=[0, 100]
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#f1f5f9", size=13, family="Plus Jakarta Sans")
        ),
        height=320
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
