import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgroScan — Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS) matching reference mockup
# ---------------------------------------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Overall page background */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* Hide standard Streamlit header & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    color: #f8fafc;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #f1f5f9 !important;
}

/* Top Navigation Header */
.top-header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
}
.top-header-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.top-header-date {
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748b;
}

/* Streamlit Container / Card Styling */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 16px !important;
    border: 1px solid #e2e8f0 !important;
    background-color: #ffffff !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
    padding: 24px !important;
}

/* Card Header */
.card-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}
.card-icon {
    font-size: 1.25rem;
}
.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}
.card-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 18px;
    margin-top: 2px;
}

/* Image Preview styling */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    margin-bottom: 16px;
}
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    max-height: 300px;
    width: auto;
    max-width: 100%;
    object-fit: contain;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
}

/* File Uploader styling */
[data-testid="stFileUploader"] {
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
    padding: 12px;
    background: #f8fafc;
    transition: all 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #15803d;
    background: #f0fdf4;
}

/* Results Card styling matching mockup */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 25px -4px rgba(0, 0, 0, 0.06);
    margin-top: 20px;
}

/* Header row in results card */
.result-header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
}
.badges-group {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}
.badge-disease {
    background-color: #fee2e2;
    color: #dc2626;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 9999px;
    letter-spacing: 0.02em;
}
.badge-healthy {
    background-color: #dcfce7;
    color: #16a34a;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 9999px;
    letter-spacing: 0.02em;
}
.badge-crop {
    background-color: #f1f5f9;
    color: #475569;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
}

/* Right side confidence score */
.score-container {
    text-align: right;
}
.score-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
}
.score-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Disease diagnosis section */
.diagnosis-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    margin-top: 6px;
    margin-bottom: 16px;
}
.diagnosis-icon-wrapper {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1.25rem;
}
.icon-disease {
    background-color: #fef3c7;
    color: #b45309;
}
.icon-healthy {
    background-color: #dcfce7;
    color: #15803d;
}
.diagnosis-content {
    flex: 1;
}
.diagnosis-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 6px 0;
}
.diagnosis-desc {
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.5;
    margin: 0;
}

/* Progress Bar */
.progress-section {
    margin: 18px 0 6px 0;
    padding-top: 14px;
    border-top: 1px solid #f1f5f9;
}
.progress-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 8px;
}
.progress-bar-bg {
    width: 100%;
    height: 10px;
    background-color: #f1f5f9;
    border-radius: 9999px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 9999px;
}
.fill-disease {
    background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
}
.fill-healthy {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%);
}

/* Action buttons */
div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.25rem !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"] {
    background-color: #15803d !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(21, 128, 61, 0.2) !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #166534 !important;
    box-shadow: 0 6px 16px rgba(21, 128, 61, 0.3) !important;
    transform: translateY(-1px);
}
div.stButton > button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
}
div.stButton > button[kind="secondary"]:hover {
    background-color: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

/* Home Page Hero & Stats */
.hero-container {
    text-align: center;
    padding: 1.5rem 1rem 2rem 1rem;
    max-width: 860px;
    margin: 0 auto;
}
.hero-badge {
    display: inline-block;
    background: #dcfce7;
    color: #15803d;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 6px 14px;
    border-radius: 9999px;
    margin-bottom: 16px;
    letter-spacing: 0.03em;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 12px;
}
.hero-desc {
    font-size: 1.05rem;
    color: #475569;
    max-width: 680px;
    margin: 0 auto 24px auto;
    line-height: 1.6;
}

.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}
.stat-number {
    font-size: 1.8rem;
    font-weight: 800;
    color: #15803d;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Class Metadata & Diagnostic Knowledge Base
# ---------------------------------------------------------
CLASS_LABELS = [
    'Corn___Common_Rust',
    'Corn___Gray_Leaf_Spot',
    'Corn___Healthy',
    'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight',
    'Potato___Healthy',
    'Potato___Late_Blight',
    'Rice___Brown_Spot',
    'Rice___Healthy',
    'Rice___Leaf_Blast',
    'Rice___Neck_Blast',
    'Sugarcane_Bacterial Blight',
    'Sugarcane_Healthy',
    'Sugarcane_Red Rot',
    'Wheat___Brown_Rust',
    'Wheat___Healthy',
    'Wheat___Yellow_Rust'
]

CLASS_METADATA = {
    'Corn___Common_Rust': {
        'crop': 'Corn',
        'condition': 'Common Rust',
        'is_healthy': False,
        'symptom': 'Reddish-brown oval pustules appear scattered across both upper and lower leaf surfaces.',
        'mitigation': 'Apply foliar fungicides early if infection threatens upper leaves. Plant resistant hybrids and rotate crops.'
    },
    'Corn___Gray_Leaf_Spot': {
        'crop': 'Corn',
        'condition': 'Gray Leaf Spot',
        'is_healthy': False,
        'symptom': 'Rectangular, tan-to-gray necrotic lesions running strictly parallel to leaf veins.',
        'mitigation': 'Rotate crops, till infected residues, and apply foliar fungicide if lesions reach upper leaves before flowering.'
    },
    'Corn___Healthy': {
        'crop': 'Corn',
        'condition': 'Healthy',
        'is_healthy': True,
        'symptom': 'Clean, vigorous foliage with no fungal or bacterial lesions observed.',
        'mitigation': 'Maintain balanced nitrogen fertilization, regular field scouting, and optimal irrigation intervals.'
    },
    'Corn___Northern_Leaf_Blight': {
        'crop': 'Corn',
        'condition': 'Northern Leaf Blight',
        'is_healthy': False,
        'symptom': 'Long, elliptical cigar-shaped grayish-green lesions that become tan as fungal spores mature.',
        'mitigation': 'Utilize resistant cultivars, manage infected crop debris, and spray protective fungicides.'
    },
    'Potato___Early_Blight': {
        'crop': 'Potato',
        'condition': 'Early Blight',
        'is_healthy': False,
        'symptom': 'Dark brown to black circular spots showing characteristic concentric rings (target-board pattern).',
        'mitigation': 'Avoid overhead sprinkler watering, maintain plant vigor through balanced potassium, and apply protectant fungicides.'
    },
    'Potato___Healthy': {
        'crop': 'Potato',
        'condition': 'Healthy',
        'is_healthy': True,
        'symptom': 'Robust green foliage, active growth, and no signs of blights or rot.',
        'mitigation': 'Continue regular soil hilling, weed management, and monitor soil moisture.'
    },
    'Potato___Late_Blight': {
        'crop': 'Potato',
        'condition': 'Late Blight',
        'is_healthy': False,
        'symptom': 'Water-soaked dark lesions spreading rapidly on leaves, often accompanied by white fungal mold underneath.',
        'mitigation': 'Destroy infected foliage immediately to prevent tuber rot. Apply systemic copper or metalaxyl fungicides.'
    },
    'Rice___Brown_Spot': {
        'crop': 'Rice',
        'condition': 'Brown Spot',
        'is_healthy': False,
        'symptom': 'Small, circular to oval brown spots with gray centers and yellowish halos on leaves and grains.',
        'mitigation': 'Ensure balanced potassium and nitrogen soil nutrition, treat seeds before planting, and spray mancozeb.'
    },
    'Rice___Healthy': {
        'crop': 'Rice',
        'condition': 'Healthy',
        'is_healthy': True,
        'symptom': 'Upright, clean emerald-green leaf blades with no spot discoloration.',
        'mitigation': 'Maintain scheduled water depth, weed clearance, and timely top-dressing with nitrogen.'
    },
    'Rice___Leaf_Blast': {
        'crop': 'Rice',
        'condition': 'Leaf Blast',
        'is_healthy': False,
        'symptom': 'Diamond or spindle-shaped lesions with grayish centers and dark reddish-brown borders on leaf blades.',
        'mitigation': 'Avoid excessive nitrogen fertilizers; spray tricyclazole or azoxystrobin fungicides promptly.'
    },
    'Rice___Neck_Blast': {
        'crop': 'Rice',
        'condition': 'Neck Blast',
        'is_healthy': False,
        'symptom': 'Grayish-brown rot around the panicle node (neck), causing lodging and sterile white heads.',
        'mitigation': 'Apply preventive systemic fungicides at 5% heading and flowering stage to secure yield.'
    },
    'Sugarcane_Bacterial Blight': {
        'crop': 'Sugarcane',
        'condition': 'Bacterial Blight',
        'is_healthy': False,
        'symptom': 'Reddish-brown water-soaked streaks along leaf veins, leading to leaf desiccation and top rot.',
        'mitigation': 'Use certified disease-free setts, sanitize knives during harvesting, and avoid waterlogged conditions.'
    },
    'Sugarcane_Healthy': {
        'crop': 'Sugarcane',
        'condition': 'Healthy',
        'is_healthy': True,
        'symptom': 'Sturdy cane stalks with lush green leaf canopy and no vascular streaks.',
        'mitigation': 'Continue inter-row cultivation, adequate irrigation cycles, and stem borer surveillance.'
    },
    'Sugarcane_Red Rot': {
        'crop': 'Sugarcane',
        'condition': 'Red Rot',
        'is_healthy': False,
        'symptom': 'Discoloration of midribs with red lesions; internal stalk tissue exhibits red discoloration with white patches.',
        'mitigation': 'Uproot and burn infected clumps immediately. Practice crop rotation and plant certified disease-resistant setts.'
    },
    'Wheat___Brown_Rust': {
        'crop': 'Wheat',
        'condition': 'Brown Rust',
        'is_healthy': False,
        'symptom': 'Brown Rust. Small, orange-brown pustules scattered irregularly across the upper leaf surface.',
        'mitigation': 'Early mitigation is advised to prevent spread across the field. Apply triazole fungicides promptly.'
    },
    'Wheat___Healthy': {
        'crop': 'Wheat',
        'condition': 'Healthy',
        'is_healthy': True,
        'symptom': 'Clean green tillers and flag leaves showing no pustules or chlorotic streaking.',
        'mitigation': 'Maintain regular field monitoring, adequate soil moisture, and timely top-dressing through flag leaf emergence.'
    },
    'Wheat___Yellow_Rust': {
        'crop': 'Wheat',
        'condition': 'Yellow Rust',
        'is_healthy': False,
        'symptom': 'Yellow-orange pustules arranged in prominent linear stripes along leaf veins (stripe rust).',
        'mitigation': 'Spray systemic fungicides (propiconazole/tebuconazole) upon initial observation. Sow rust-resistant cultivars.'
    }
}

def format_class_name(raw_label):
    if raw_label in CLASS_METADATA:
        meta = CLASS_METADATA[raw_label]
        return f"{meta['crop']} — {meta['condition']}"
    cleaned = raw_label.replace("___", " — ").replace("_", " ")
    return cleaned

# ---------------------------------------------------------
# Model Loading & Prediction Logic
# ---------------------------------------------------------
@st.cache_resource
def load_disease_model():
    model_path = 'crop_disease_model_final.keras'
    return tf.keras.models.load_model(model_path)

def model_prediction(uploaded_file):
    model = load_disease_model()
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((180, 180))
    input_arr = tf.keras.preprocessing.image.img_to_array(image_resized)
    input_arr = np.expand_dims(input_arr, axis=0)
    
    raw_prediction = model.predict(input_arr)[0]
    
    # Calculate probabilities
    if np.isclose(np.sum(raw_prediction), 1.0, atol=1e-2):
        probabilities = raw_prediction
    else:
        exp_p = np.exp(raw_prediction - np.max(raw_prediction))
        probabilities = exp_p / np.sum(exp_p)
        
    top_index = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_index] * 100)
        
    return top_index, top_confidence

# ---------------------------------------------------------
# Sidebar Navigation (Landing page defaults to "Home")
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #334155;">
            <span style="font-size: 1.8rem;">🌿</span>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700;">AgroScan AI</h3>
                <span style="font-size: 0.75rem; color: #94a3b8;">Crop Health Intelligence</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # "Home" is default landing page
    if "app_nav_choice" not in st.session_state:
        st.session_state["app_nav_choice"] = "Home"
        
    def on_nav_change():
        st.session_state["app_nav_choice"] = st.session_state["sidebar_radio"]
        
    app_mode = st.radio(
        "Navigation",
        ["Home", "Detect Disease"],
        index=0 if st.session_state["app_nav_choice"] == "Home" else 1,
        key="sidebar_radio",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )
    app_mode = st.session_state["app_nav_choice"]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Specs Badge
    st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; margin-bottom: 20px;">
            <div style="font-size: 0.72rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px;">Model Engine</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">Deep CNN Classifier</div>
            <div style="display: flex; align-items: center; gap: 6px; margin-top: 8px;">
                <span style="background: #15803d; color: #ffffff; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 6px;">93.5% Accuracy</span>
                <span style="color: #94a3b8; font-size: 0.75rem;">17 Classes</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px;">Supported Crops</div>
        <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 24px;">
            <span style="background: #334155; color: #e2e8f0; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px;">🌽 Corn</span>
            <span style="background: #334155; color: #e2e8f0; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px;">🥔 Potato</span>
            <span style="background: #334155; color: #e2e8f0; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px;">🌾 Rice</span>
            <span style="background: #334155; color: #e2e8f0; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px;">🎋 Sugarcane</span>
            <span style="background: #334155; color: #e2e8f0; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px;">🌾 Wheat</span>
        </div>
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            Capture high-resolution leaf images under natural light for optimal prediction precision.
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# App Mode: Home (Landing Page)
# ---------------------------------------------------------
if app_mode == "Home":
    st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">AI Agricultural Diagnosis</span>
            <h1 class="hero-title">Smart Crop Disease Detection & Health Monitoring</h1>
            <p class="hero-desc">
                Identify crop leaf infections instantly with deep computer vision. Protect your crop yields through rapid symptoms diagnosis and expert mitigation guidance.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # CTA to jump straight to Detect Disease
    _, cta_col, _ = st.columns([0.3, 0.4, 0.3])
    with cta_col:
        def go_to_detection():
            st.session_state["app_nav_choice"] = "Detect Disease"
        st.button("🌿 Start Disease Detection Now", use_container_width=True, type="primary", on_click=go_to_detection)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3 Stat Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">93.5%</div>
                <div class="stat-label">Model Validation Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">17 Classes</div>
                <div class="stat-label">Diseases & Health States</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">5 Crops</div>
                <div class="stat-label">Corn, Potato, Rice, Sugarcane, Wheat</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero Image Display if available
    if os.path.exists("Home_page.jpg"):
        st.markdown("""
            <div style="border-radius: 16px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 32px;">
        """, unsafe_allow_html=True)
        st.image("Home_page.jpg", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # How It Works 3-Step Section
    st.markdown("""
        <h3 style="font-weight: 800; color: #0f172a; margin-bottom: 16px; text-align: center;">How It Works</h3>
    """, unsafe_allow_html=True)
    
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("""
            <div class="stat-card" style="text-align: left; height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">📸</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">1. Upload Image</div>
                <div style="font-size: 0.88rem; color: #64748b; line-height: 1.5;">
                    Drag and drop or browse a leaf photograph showing clear symptoms.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with step2:
        st.markdown("""
            <div class="stat-card" style="text-align: left; height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">⚡</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">2. AI Diagnosis</div>
                <div style="font-size: 0.88rem; color: #64748b; line-height: 1.5;">
                    Deep neural network extracts visual patterns and computes confidence probabilities.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with step3:
        st.markdown("""
            <div class="stat-card" style="text-align: left; height: 100%;">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">🛡️</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;">3. Mitigation Plan</div>
                <div style="font-size: 0.88rem; color: #64748b; line-height: 1.5;">
                    Receive clear disease identification and actionable steps to prevent crop damage.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# App Mode: Detect Disease (Mockup Design)
# ---------------------------------------------------------
elif app_mode == "Detect Disease":
    current_date_str = datetime.now().strftime("%A, %B %d, %Y")
    
    # Top Header Bar
    st.markdown(f"""
        <div class="top-header-container">
            <h1 class="top-header-title">
                Detect Disease
            </h1>
            <div class="top-header-date">
                {current_date_str}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Centered Column layout matching mockup width
    _, center_col, _ = st.columns([0.15, 0.7, 0.15])
    
    with center_col:
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0

        # Native container with card border
        with st.container(border=True):
            st.markdown("""
                <div class="card-title-row">
                    <span class="card-icon">🔎</span>
                    <span class="card-title">Detect Disease</span>
                </div>
                <div class="card-subtitle">
                    Capture or upload a crop leaf photo to identify diseases and health condition in real time.
                </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload an Image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key=f"crop_uploader_{st.session_state['uploader_key']}"
            )
            
            if uploaded_file is not None:
                # Clear previous results if a different file is selected
                if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
                    st.session_state["last_uploaded_filename"] = uploaded_file.name
                    st.session_state.pop("detection_results", None)

                # Display image preview
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                # Action Buttons Row
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    choose_another = st.button("🔄 Choose Another Image", use_container_width=True, type="secondary", key="btn_another")
                with btn_col2:
                    detect_clicked = st.button("🔍 Detect Disease", use_container_width=True, type="primary", key="btn_detect")
                
                if choose_another:
                    st.session_state.pop("detection_results", None)
                    st.session_state.pop("last_uploaded_filename", None)
                    st.session_state["uploader_key"] += 1
                    st.rerun()
                    
                if detect_clicked:
                    with st.spinner("Analyzing plant leaf symptoms..."):
                        top_idx, top_confidence = model_prediction(uploaded_file)
                        st.session_state["detection_results"] = {
                            "top_idx": top_idx,
                            "confidence": top_confidence,
                            "raw_label": CLASS_LABELS[top_idx]
                        }
            else:
                st.session_state.pop("detection_results", None)
                st.info("👆 Please drag and drop or browse a crop leaf photo above.")

        # Render Results Card (Showing ONLY the highest predicted disease)
        if "detection_results" in st.session_state and uploaded_file is not None:
            res = st.session_state["detection_results"]
            top_class_raw = res["raw_label"]
            meta = CLASS_METADATA.get(top_class_raw, {
                'crop': 'Crop',
                'condition': top_class_raw,
                'is_healthy': 'Healthy' in top_class_raw,
                'symptom': 'Pattern recognized by deep classification network.',
                'mitigation': 'Consult regional agronomy advisors for specific recommendations.'
            })
            
            is_healthy = meta['is_healthy']
            crop_name = meta['crop']
            top_conf_pct = res["confidence"]
            conf_formatted = f"{top_conf_pct:.2f}%"
            
            status_badge_html = '<span class="badge-healthy">Healthy Plant</span>' if is_healthy else '<span class="badge-disease">Disease Detected</span>'
            status_icon = "🌿" if is_healthy else "⚠️"
            icon_class = "icon-healthy" if is_healthy else "icon-disease"
            fill_class = "fill-healthy" if is_healthy else "fill-disease"
            formatted_title = f"{crop_name} — {meta['condition']}"
            
            # Flush-left HTML string with zero leading indentation to prevent Markdown code block parsing
            result_card_html = f"""<div class="result-card">
<div class="result-header-row">
<div class="badges-group">
{status_badge_html}
<span class="badge-crop">Crop: {crop_name}</span>
</div>
<div class="score-container">
<div class="score-value">{conf_formatted}</div>
<div class="score-label">CONFIDENCE</div>
</div>
</div>
<div class="diagnosis-row">
<div class="diagnosis-icon-wrapper {icon_class}">
{status_icon}
</div>
<div class="diagnosis-content">
<h2 class="diagnosis-title">{formatted_title}</h2>
<p class="diagnosis-desc">
<strong>Identified symptom:</strong> {meta['symptom']}
<br>
<strong>Action:</strong> {meta['mitigation']}
</p>
</div>
</div>
<div class="progress-section">
<div class="progress-labels">
<span>Model Confidence</span>
<span>{conf_formatted}</span>
</div>
<div class="progress-bar-bg">
<div class="progress-bar-fill {fill_class}" style="width: {min(max(top_conf_pct, 4.0), 100.0):.1f}%;"></div>
</div>
</div>
</div>"""
            st.markdown(result_card_html, unsafe_allow_html=True)