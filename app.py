import streamlit as st
import pickle
import numpy as np
from src.processor import preprocess_text
import time
import base64

# Page Config
st.set_page_config(
    page_title="SentiMentX | AI-Powered Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get logo
def get_logo_base64():
    try:
        with open("logo.jpeg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"
    except:
        return ""

LOGO_BASE64 = get_logo_base64()

# Clean & Attractive CSS
st.markdown('''
<style>
    /* Clean Dark Theme */
    :root {
        --primary: #7B68EE;
        --primary-light: #9370DB;
        --accent: #00D4AA;
        --bg-dark: #0A0A0F;
        --card-bg: rgba(25, 25, 40, 0.9);
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0C0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0A0A0F 0%, #1A1A2E 100%);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Hide Streamlit elements */
    .stApp > header, #MainMenu, footer, .stDeployButton { 
        display: none !important; 
    }
    
    /* Clean Card Design */
    .clean-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 35px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Input Field */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 15px !important;
        color: white !important;
        font-size: 16px !important;
        padding: 20px !important;
        min-height: 150px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(123, 104, 238, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-light)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(123, 104, 238, 0.3) !important;
    }
    
    /* Progress Bar */
    .simple-progress {
        width: 100%;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 8px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .simple-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #7B68EE, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric Card */
    .simple-metric {
        background: rgba(123, 104, 238, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(123, 104, 238, 0.2);
        transition: transform 0.3s ease;
    }
    
    .simple-metric:hover {
        transform: translateY(-3px);
        background: rgba(123, 104, 238, 0.15);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 25, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header Glow Effect */
    .header-glow {
        text-shadow: 0 0 30px rgba(123, 104, 238, 0.5);
    }
    
    /* Tagline Animation */
    .tagline-animation {
        background: linear-gradient(90deg, #7B68EE, #00D4AA, #9370DB);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite alternate;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 100% center; }
    }
    
    /* Pulse Animation */
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Remove button icon duplicates */
    button div[data-testid="stMarkdownContainer"] p {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
</style>
''', unsafe_allow_html=True)

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Load Models
@st.cache_resource
def load_assets():
    try:
        with open('models/best_emotion_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/emotion_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        return model, vectorizer, mappings['numbers_to_emotions']
    except Exception as e:
        st.error(f"Error loading models: {str(e)[:100]}")
        return None, None, None

model, vectorizer, num_to_emo = load_assets()

# Emotion Configuration
EMOTION_CONFIG = {
    'joy': {'color': '#FFD700', 'emoji': '😊', 'name': 'Joy'},
    'sadness': {'color': '#4169E1', 'emoji': '😢', 'name': 'Sadness'},
    'anger': {'color': '#FF4500', 'emoji': '😠', 'name': 'Anger'},
    'love': {'color': '#FF69B4', 'emoji': '❤️', 'name': 'Love'},
    'surprise': {'color': '#9370DB', 'emoji': '😲', 'name': 'Surprise'},
    'fear': {'color': '#32CD32', 'emoji': '😨', 'name': 'Fear'}
}

# ======================
# ENHANCED HEADER SECTION
# ======================

with st.container():
    # Decorative Elements
    st.markdown('''
    <div style="position: absolute; top: 20px; right: 20px; opacity: 0.1; font-size: 4rem; z-index: 0;">
        🧠
    </div>
    <div style="position: absolute; top: 80px; left: 20px; opacity: 0.1; font-size: 3rem; z-index: 0;">
        ⚡
    </div>
    ''', unsafe_allow_html=True)
    
    # Main Header Container
    st.markdown('<div style="position: relative; z-index: 1;">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Animated Logo Container
        if LOGO_BASE64:
            st.markdown(f'''
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div class="pulse-animation" style="
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #7B68EE, #00D4AA);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 0 40px rgba(123, 104, 238, 0.4);
                    border: 3px solid rgba(255, 255, 255, 0.2);
                ">
                    <img src="{LOGO_BASE64}" style="width: 70px; height: 70px; border-radius: 50%;">
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                <div class="pulse-animation" style="
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #7B68EE, #00D4AA);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 0 40px rgba(123, 104, 238, 0.4);
                    border: 3px solid rgba(255, 255, 255, 0.2);
                    font-size: 3rem;
                ">
                    🧠
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Company Name with Glow Effect
        st.markdown('''
        <h1 class="header-glow" style="
            text-align: center; 
            font-size: 3.5rem; 
            margin: 10px 0; 
            background: linear-gradient(135deg, #FFFFFF, #B0B0C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            letter-spacing: -0.5px;
        ">
            SentiMentX
        </h1>
        ''', unsafe_allow_html=True)
        
        # Animated Tagline
        st.markdown('''
        <p class="tagline-animation" style="
            text-align: center; 
            font-size: 1.4rem; 
            margin-bottom: 40px;
            font-weight: 600;
            letter-spacing: 0.5px;
        ">
            AI-Powered Sentiment Intelligence
        </p>
        ''', unsafe_allow_html=True)
        
        # Decorative Line
        st.markdown('''
        <div style="
            height: 2px;
            width: 200px;
            margin: 0 auto 30px auto;
            background: linear-gradient(90deg, transparent, #7B68EE, #00D4AA, transparent);
        "></div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# INPUT SECTION
# ======================

st.markdown('<div class="clean-card">', unsafe_allow_html=True)

# Text Input
user_input = st.text_area(
    "Share your thoughts or feelings:",
    value=st.session_state.user_input,
    height=150,
    placeholder="Type your text here...\n\nExample: 'I feel incredibly happy today! Everything is going perfectly.'",
    label_visibility="collapsed"
)

# Analyze Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button(
        "🔍 Analyze Emotions",
        type="primary",
        use_container_width=True,
        key="analyze_btn"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Handle analysis
if analyze_clicked and user_input.strip():
    st.session_state.user_input = user_input
    
    with st.spinner("🤖 Analyzing emotions with AI..."):
        time.sleep(0.5)
        
        if model and vectorizer:
            cleaned_text = preprocess_text(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            emotion = num_to_emo[prediction]
            probs = model.predict_proba(vectorized_text)[0]
            
            confidence = np.max(probs) * 100
            complexity = np.std(probs) * 100
            
            st.session_state.analysis_result = {
                'emotion': emotion,
                'confidence': confidence,
                'probs': probs,
                'cleaned_text': cleaned_text,
                'complexity': complexity,
                'word_count': len(user_input.split())
            }

# ======================
# RESULTS SECTION
# ======================

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    emotion = result['emotion']
    config = EMOTION_CONFIG[emotion]
    
    # Primary Emotion Result
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Large emoji
        st.markdown(f'<div style="font-size: 4.5rem; text-align: center; margin: 10px 0;">{config["emoji"]}</div>', 
                   unsafe_allow_html=True)
        
        # Emotion name
        st.markdown(f'<h2 style="text-align: center; font-size: 2rem; margin-bottom: 5px; color: {config["color"]};">'
                   f'{config["name"]}</h2>', unsafe_allow_html=True)
        
        # Confidence
        st.markdown(f'<p style="text-align: center; color: #B0B0C0; font-size: 1rem; margin-bottom: 20px;">'
                   f'Confidence: <span style="color: {config["color"]}; font-weight: 600;">{result["confidence"]:.1f}%</span></p>', 
                   unsafe_allow_html=True)
        
        # Progress bar
        st.markdown(f'''
        <div class="simple-progress">
            <div class="simple-progress-bar" style="width: {result['confidence']}%"></div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Simple Metrics
    metric_cols = st.columns(3)
    
    with metric_cols[0]:
        st.markdown(f'''
        <div class="simple-metric">
            <div style="font-size: 1.8rem; margin-bottom: 10px; color: #7B68EE;">🎯</div>
            <div style="font-weight: 700; font-size: 1.6rem; margin-bottom: 5px;">{result['confidence']:.1f}%</div>
            <div style="color: #B0B0C0; font-size: 0.9rem;">Confidence</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with metric_cols[1]:
        complexity_label = "Complex" if result['complexity'] > 30 else "Moderate"
        complexity_color = "#FF4500" if result['complexity'] > 30 else "#FFD700"
        st.markdown(f'''
        <div class="simple-metric">
            <div style="font-size: 1.8rem; margin-bottom: 10px; color: {complexity_color};">🌀</div>
            <div style="font-weight: 700; font-size: 1.4rem; margin-bottom: 5px;">{complexity_label}</div>
            <div style="color: #B0B0C0; font-size: 0.9rem;">Complexity</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown(f'''
        <div class="simple-metric">
            <div style="font-size: 1.8rem; margin-bottom: 10px; color: #32CD32;">📝</div>
            <div style="font-weight: 700; font-size: 1.6rem; margin-bottom: 5px;">{result['word_count']}</div>
            <div style="color: #B0B0C0; font-size: 0.9rem;">Words</div>
        </div>
        ''', unsafe_allow_html=True)

# ======================
# ENHANCED SIDEBAR WITH EMOJIS
# ======================

with st.sidebar:
    # Sidebar Header
    if LOGO_BASE64:
        st.markdown(f'''
        <div style="display: flex; justify-content: center; margin-bottom: 15px;">
            <img src="{LOGO_BASE64}" style="
                width: 60px; 
                height: 60px; 
                border-radius: 50%;
                border: 2px solid rgba(123, 104, 238, 0.3);
                box-shadow: 0 0 20px rgba(123, 104, 238, 0.3);
            ">
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="
            display: flex; 
            justify-content: center; 
            margin-bottom: 15px;
            font-size: 2.5rem;
            color: #7B68EE;
            text-shadow: 0 0 15px rgba(123, 104, 238, 0.5);
        ">
            🧠
        </div>
        ''', unsafe_allow_html=True)
    
    # Company Name in Sidebar
    st.markdown('<h3 style="text-align: center; margin-bottom: 5px; color: white;">SentiMentX</h3>', unsafe_allow_html=True)
    
    # Tagline in Sidebar
    st.markdown('<p style="text-align: center; color: #B0B0C0; font-size: 0.9rem; margin-bottom: 30px;">AI-Powered Sentiment Intelligence</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Emotion Emojis Section
    st.markdown('<p style="color: #B0B0C0; margin-bottom: 15px; font-weight: 600;">🎭 Detected Emotions</p>', unsafe_allow_html=True)
    
    # Create a grid of emotion emojis
    emotion_cols = st.columns(3)
    
    emotions_display = [
        ('😊', 'Joy', '#FFD700'),
        ('😢', 'Sadness', '#4169E1'),
        ('😠', 'Anger', '#FF4500'),
        ('❤️', 'Love', '#FF69B4'),
        ('😲', 'Surprise', '#9370DB'),
        ('😨', 'Fear', '#32CD32')
    ]
    
    for idx, (emoji, name, color) in enumerate(emotions_display):
        col = emotion_cols[idx % 3]
        with col:
            st.markdown(f'''
            <div style="
                text-align: center;
                padding: 10px 0;
                margin-bottom: 10px;
                border-radius: 10px;
                background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
                border: 1px solid rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3);
            ">
                <div style="font-size: 1.8rem; margin-bottom: 5px;">{emoji}</div>
                <div style="color: {color}; font-size: 0.8rem; font-weight: 600;">{name}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Examples
    st.markdown('<p style="color: #B0B0C0; margin-bottom: 15px; font-weight: 600;">💡 Try these examples:</p>', unsafe_allow_html=True)
    
    examples = [
        ("I'm so happy right now!", "😊 Joy"),
        ("Feeling really sad today", "😢 Sadness"),
        ("This makes me angry!", "😠 Anger"),
        ("I love this so much!", "❤️ Love"),
        ("Wow, that surprised me!", "😲 Surprise"),
        ("I'm scared about this", "😨 Fear")
    ]
    
    for text, emoji_label in examples:
        if st.button(f"{emoji_label}: {text[:20]}...", key=f"ex_{text[:10]}", use_container_width=True):
            st.session_state.user_input = text
            st.session_state.analysis_result = None
            st.rerun()
    
    st.divider()
    
    # Clear button
    if st.button("🗑️ Clear Analysis", use_container_width=True, key="clear_btn"):
        st.session_state.user_input = ""
        st.session_state.analysis_result = None
        st.rerun()
    
    st.divider()
    
    # Enhanced Footer
    st.markdown('''
    <div style="margin-top: 20px; padding: 15px; text-align: center;">
        <div style="
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 10px;
            opacity: 0.7;
        ">
            <span>⚡</span>
            <span>🤖</span>
            <span>🧠</span>
        </div>
        <p style="color: #B0B0C0; font-size: 0.8rem; margin: 0;">
            Powered by SentiMentX AI
        </p>
        <p style="color: #666; font-size: 0.7rem; margin: 5px 0 0 0;">
            v2.0 | Real-time Emotion Detection
        </p>
    </div>
    ''', unsafe_allow_html=True)

# ======================
# FOOTER WITH MODEL INFO
# ======================

st.markdown('<div class="clean-card">', unsafe_allow_html=True)

# Model Information Section
st.markdown('<h2 class="gradient-text" style="text-align: center; margin-bottom: 30px;">🧠 Powered by SentiMentX AI</h2>', 
           unsafe_allow_html=True)

# Model Details
col1, col2 = st.columns(2)

with col1:
    st.markdown('''
    <div style="background: rgba(123, 104, 238, 0.1); border-radius: 15px; padding: 20px; height: 100%;">
        <h3 style="color: #7B68EE; margin-bottom: 15px;">📊 AI Architecture</h3>
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 1.2rem; margin-right: 10px;">🤖</div>
            <div>
                <div style="font-weight: 600; color: white;">Logistic Regression</div>
                <div style="color: #B0B0C0; font-size: 0.9rem;">Advanced ML Classification</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 1.2rem; margin-right: 10px;">🔤</div>
            <div>
                <div style="font-weight: 600; color: white;">TF-IDF Vectorizer</div>
                <div style="color: #B0B0C0; font-size: 0.9rem;">Context-Aware Processing</div>
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem; margin-right: 10px;">🧹</div>
            <div>
                <div style="font-weight: 600; color: white;">NLP Preprocessing</div>
                <div style="color: #B0B0C0; font-size: 0.9rem;">Intelligent Text Cleaning</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div style="background: rgba(0, 212, 170, 0.1); border-radius: 15px; padding: 20px; height: 100%;">
        <h3 style="color: #00D4AA; margin-bottom: 15px;">📈 AI Performance</h3>
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 1.2rem; margin-right: 10px;">🎯</div>
            <div>
                <div style="font-weight: 600; color: white;">88.6% Accuracy</div>
                <div style="color: #B0B0C0; font-size: 0.9rem;">Industry-Leading Precision</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 1.2rem; margin-right: 10px;">⚡</div>
            <div>
                <div style="font-weight: 600; color: white;">0.2s Processing</div>
                <div style="color: #B0B0C0; font-size: 0.9rem;">Real-time Analysis</div>
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.2rem; margin-right: 10px;">🎭</div>
            <div>
                <div style="font-weight: 600; color: white;">6 Emotions</div>
                <div style="color: #B0B0C0; font-size: 0.9rem;">Comprehensive Detection</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# How It Works Section
st.markdown('''
<div style="margin-top: 30px; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px;">
    <h3 style="color: white; margin-bottom: 15px;">🔍 How SentiMentX Works</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
        <div style="padding: 15px; background: linear-gradient(135deg, rgba(123, 104, 238, 0.1), rgba(0, 212, 170, 0.1)); border-radius: 10px;">
            <div style="font-size: 1.5rem; margin-bottom: 10px; color: #7B68EE;">1️⃣</div>
            <div style="font-weight: 600; color: white; margin-bottom: 5px;">Text Input</div>
            <div style="color: #B0B0C0; font-size: 0.9rem;">User provides text for AI analysis</div>
        </div>
        <div style="padding: 15px; background: linear-gradient(135deg, rgba(0, 212, 170, 0.1), rgba(255, 215, 0, 0.1)); border-radius: 10px;">
            <div style="font-size: 1.5rem; margin-bottom: 10px; color: #00D4AA;">2️⃣</div>
            <div style="font-weight: 600; color: white; margin-bottom: 5px;">AI Processing</div>
            <div style="color: #B0B0C0; font-size: 0.9rem;">Advanced NLP & vectorization</div>
        </div>
        <div style="padding: 15px; background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 105, 180, 0.1)); border-radius: 10px;">
            <div style="font-size: 1.5rem; margin-bottom: 10px; color: #FFD700;">3️⃣</div>
            <div style="font-weight: 600; color: white; margin-bottom: 5px;">ML Prediction</div>
            <div style="color: #B0B0C0; font-size: 0.9rem;">Intelligent emotion classification</div>
        </div>
        <div style="padding: 15px; background: linear-gradient(135deg, rgba(255, 105, 180, 0.1), rgba(147, 112, 219, 0.1)); border-radius: 10px;">
            <div style="font-size: 1.5rem; margin-bottom: 10px; color: #FF69B4;">4️⃣</div>
            <div style="font-weight: 600; color: white; margin-bottom: 5px;">Insight Delivery</div>
            <div style="color: #B0B0C0; font-size: 0.9rem;">Detailed emotion detection report</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Final Metrics
st.markdown('''
<div style="margin-top: 30px; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
    <div class="simple-metric">
        <div style="font-size: 1.8rem; margin-bottom: 10px; color: #7B68EE;">🚀</div>
        <div style="font-weight: 700; font-size: 1.4rem; margin-bottom: 5px;">SentiMentX</div>
        <div style="color: #B0B0C0; font-size: 0.85rem;">AI Engine</div>
    </div>
    <div class="simple-metric">
        <div style="font-size: 1.8rem; margin-bottom: 10px; color: #00D4AA;">🎯</div>
        <div style="font-weight: 700; font-size: 1.4rem; margin-bottom: 5px;">88.6%</div>
        <div style="color: #B0B0C0; font-size: 0.85rem;">AI Accuracy</div>
    </div>
    <div class="simple-metric">
        <div style="font-size: 1.8rem; margin-bottom: 10px; color: #FFD700;">⚡</div>
        <div style="font-weight: 700; font-size: 1.4rem; margin-bottom: 5px;">0.2s</div>
        <div style="color: #B0B0C0; font-size: 0.85rem;">Real-time AI</div>
    </div>
    <div class="simple-metric">
        <div style="font-size: 1.8rem; margin-bottom: 10px; color: #4169E1;">🤖</div>
        <div style="font-weight: 700; font-size: 1.4rem; margin-bottom: 5px;">6 Emotions</div>
        <div style="color: #B0B0C0; font-size: 0.85rem;">AI Detection</div>
    </div>
</div>
''', unsafe_allow_html=True)

# Footer
st.markdown('''
<div style="margin-top: 40px; padding: 20px; text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <p style="color: #B0B0C0; font-size: 0.9rem; margin: 0;">
        Powered by <span style="color: #7B68EE; font-weight: 600;">SentiMentX</span> • AI-Powered Sentiment Intelligence
    </p>
    <p style="color: #666; font-size: 0.8rem; margin: 5px 0 0 0;">
        © 2026 SentiMentX AI • Real-time Emotion Analysis Technology
    </p>
</div>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Error handling
if model is None or vectorizer is None or num_to_emo is None:
    st.error("""
    ⚠️ **AI Models not loaded properly!** 
    
    Please ensure you have the following files in your `models` directory:
    - `best_emotion_model.pkl`
    - `tfidf_vectorizer.pkl` 
    - `emotion_mappings.pkl`
    
    SentiMentX interface is ready, but AI analysis requires model files.
    """)