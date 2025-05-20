import streamlit as st
import logging
import random
import re
import time
import pandas as pd
from datetime import datetime
import math
import base64
import nltk
from nltk import bigrams

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom CSS for a clean, professional look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #333333;
    }
    .main-header {
        background: linear-gradient(to right, #007bff, #0056b3);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextArea textarea {
        border: 2px solid #007bff;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .highlight-phishing {
        background-color: #ff9999;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .highlight-link {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f5;
        border-right: 1px solid #ddd;
    }
    .stProgress .st-bo {
        background-color: #007bff;
    }
    .section-header {
        color: #007bff;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        color: #666666;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Function to calculate URL entropy
def calculate_entropy(text):
    if not text:
        return 0
    prob = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)

# Advanced heuristic for real-time phishing detection
def simulate_prediction(text, history=None):
    phishing_keywords = {
        'password': 0.35, 'login': 0.3, 'bank': 0.35, 'paypal': 0.35, 'account': 0.25,
        'credential': 0.3, 'credit': 0.25, 'payment': 0.25, 'urgent': 0.3, 'immediate': 0.25,
        'verify': 0.3, 'confirm': 0.25, 'alert': 0.25, 'security': 0.25, 'admin': 0.2,
        'support': 0.2, 'update': 0.25, 'click here': 0.25, 'prize': 0.2, 'scam': 0.3, 'phish': 0.3
    }
    phishing_bigrams = {
        ('update', 'password'): 0.3, ('bank', 'login'): 0.3, ('verify', 'account'): 0.3,
        ('click', 'here'): 0.25, ('urgent', 'action'): 0.25
    }
    suspicious_patterns = [
        (r'\.org-[a-zA-Z0-9]', 0.45), (r'http[s]?://[^\s]*\.[a-z]{2,}/[^\s]*', 0.35),
        (r'[0-9]{6,}', 0.25), (r'[^a-zA-Z0-9\s]{3,}', 0.2), (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 0.4),
        (r'[-]{2,}', 0.2), (r'\.[a-z]{2,3}\.[a-z]{2,3}', 0.25)
    ]
    blacklist_domains = ['etcsw.win', 'login-verify.com', 'secure-update.net']
    whitelist_domains = ['google.com', 'microsoft.com', 'apple.com']

    is_url = bool(re.match(r'^http[s]?://', text.lower()) or ('.' in text and ' ' not in text.strip()))
    input_type = "URL" if is_url else "Email"
    
    text_lower = text.lower().strip()
    keyword_score = 0
    bigram_score = 0
    pattern_score = 0
    structural_score = 0
    keyword_triggers = []
    bigram_triggers = []
    pattern_triggers = []
    structural_triggers = []
    
    highlighted_text = text
    for keyword in phishing_keywords:
        if keyword in text_lower:
            highlighted_text = re.sub(
                rf'\b{re.escape(keyword)}\b',
                f'<span class="highlight-phishing">{keyword}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )
    urls = re.findall(r'http[s]?://[^\s]+', text)
    for url in urls:
        highlighted_text = highlighted_text.replace(
            url, f'<span class="highlight-link">{url}</span>'
        )
    
    for keyword, weight in phishing_keywords.items():
        if keyword in text_lower:
            keyword_score += weight
            keyword_triggers.append(keyword)
    
    tokens = nltk.word_tokenize(text_lower)
    text_bigrams = [' '.join(bg) for bg in bigrams(tokens)]
    for bigram, weight in phishing_bigrams.items():
        if ' '.join(bigram) in text_bigrams:
            bigram_score += weight
            bigram_triggers.append(' '.join(bigram))
    
    for pattern, weight in suspicious_patterns:
        if re.search(pattern, text_lower):
            pattern_score += weight
            pattern_triggers.append(pattern)
    
    if input_type == "Email":
        word_count = len(tokens)
        if word_count < 20:
            structural_score += 0.15
            structural_triggers.append("short length")
        if word_count > 500:
            structural_score += 0.1
            structural_triggers.append("excessive length")
        if text_lower.count('http') > 2:
            structural_score += 0.2
            structural_triggers.append("multiple links")
        if not re.search(r'dear|hello|hi', text_lower):
            structural_score += 0.15
            structural_triggers.append("no salutation")
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text_lower))
        if special_chars / max(len(text_lower), 1) > 0.1:
            structural_score += 0.1
            structural_triggers.append("high special characters")
    else:
        if len(text_lower) > 50:
            structural_score += 0.15
            structural_triggers.append("long URL")
        if not re.search(r'\.[com|org|net|edu]$', text_lower):
            structural_score += 0.2
            structural_triggers.append("unusual TLD")
        entropy = calculate_entropy(text_lower)
        if entropy > 4.0:
            structural_score += 0.15
            structural_triggers.append("high entropy")
        for domain in blacklist_domains:
            if domain in text_lower:
                structural_score += 0.3
                structural_triggers.append(f"blacklisted domain: {domain}")
        for domain in whitelist_domains:
            if domain in text_lower:
                structural_score -= 0.2
                structural_triggers.append(f"whitelisted domain: {domain}")
    
    history_bias = 0
    if history and len(history) > 5:
        recent_phishing = sum(1 for h in history[-5:] if h['result'] == 'Phishing') / 5
        history_bias = recent_phishing * 0.1
        if recent_phishing > 0.6:
            structural_triggers.append("recent phishing trend")
    
    total_score = (keyword_score + bigram_score + pattern_score + structural_score + history_bias)
    confidence = 1 / (1 + math.exp(-10 * (total_score - 0.5)))
    confidence = max(0.5, min(0.95, confidence + random.uniform(-0.05, 0.05)))
    
    threshold = 0.5 if input_type == "Email" and len(tokens) > 50 else 0.55
    is_phishing = total_score > threshold
    
    if total_score > 0.7:
        risk_level = "High"
    elif total_score > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    explanation = []
    if keyword_triggers:
        explanation.append(f"Suspicious words: {', '.join(keyword_triggers)}")
    if bigram_triggers:
        explanation.append(f"Suspicious phrases: {', '.join(bigram_triggers)}")
    if pattern_triggers:
        explanation.append(f"Suspicious patterns: {', '.join([p if isinstance(p, str) else 'complex structure' for p in pattern_triggers])}")
    if structural_triggers:
        explanation.append(f"Structural issues: {', '.join(structural_triggers)}")
    if history_bias:
        explanation.append("Recent phishing trend detected")
    if not explanation:
        explanation.append("No suspicious indicators found.")
    
    return {
        "result": "Phishing" if is_phishing else "Legitimate",
        "confidence": confidence,
        "input_type": input_type,
        "explanation": "; ".join(explanation),
        "highlighted_text": highlighted_text,
        "risk_level": risk_level,
        "scores": {
            "keyword": keyword_score,
            "bigram": bigram_score,
            "pattern": pattern_score,
            "structural": structural_score,
            "history": history_bias
        }
    }

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'confidence_history' not in st.session_state:
    st.session_state.confidence_history = []

# Sidebar for prediction history
with st.sidebar:
    st.header("Prediction History")
    if st.session_state.prediction_history:
        with st.expander("Recent Predictions"):
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df[["Time", "Input", "Result", "Confidence", "Type", "Risk"]], use_container_width=True)
        
        csv = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download History</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("No predictions yet.")

# Main content
st.markdown('<div class="main-header"><h1>Phishing Detection System</h1><p>Instantly check if an email or URL is suspicious</p></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Live Detection", "Help"])

with tab1:
    st.markdown('<h2 class="section-header">Predict</h2>', unsafe_allow_html=True)
    st.markdown("Enter an email or URL to check for phishing risks.")
    
    user_input = st.text_area("Input Text", placeholder="Enter an email or URL (e.g., 'Urgent: Verify your account' or 'https://example.com')", height=200, key="predict_input")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Predict", key="predict_button"):
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    time.sleep(0.5)
                    prediction = simulate_prediction(user_input, st.session_state.prediction_history)
                    
                    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                    st.success(f"Prediction: **{prediction['result']}** (Confidence: {prediction['confidence']:.2%})")
                    st.write(f"**Type**: {prediction['input_type']}")
                    st.write(f"**Risk Level**: {prediction['risk_level']}")
                    st.write(f"**Why?** {prediction['explanation']}")
                    st.markdown(f"**Input Preview**: {prediction['highlighted_text']}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.progress(prediction['confidence'])
                    st.caption(f"Confidence Level: {prediction['confidence']:.2%}")
                    
                    st.subheader("Score Breakdown")
                    score_df = pd.DataFrame({
                        "Category": ["Words", "Phrases", "Patterns", "Structure", "History"],
                        "Score": [
                            prediction['scores']['keyword'],
                            prediction['scores']['bigram'],
                            prediction['scores']['pattern'],
                            prediction['scores']['structural'],
                            prediction['scores']['history']
                        ]
                    })
                    st.bar_chart(score_df.set_index("Category")["Score"])
                    
                    logger.info(f"Prediction for '{user_input[:50]}...': {prediction['result']}, Confidence: {prediction['confidence']}, Type: {prediction['input_type']}, Risk: {prediction['risk_level']}")
                    
                    st.session_state.prediction_history.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Input": user_input[:50] + "..." if len(user_input) > 50 else user_input,
                        "Result": prediction['result'],
                        "Confidence": f"{prediction['confidence']:.2%}",
                        "Type": prediction['input_type'],
                        "Risk": prediction['risk_level']
                    })
                    st.session_state.confidence_history.append(prediction['confidence'])
    
    with col2:
        if st.button("Clear Input", key="clear_button"):
            st.session_state.predict_input = ""

with tab2:
    st.markdown('<h2 class="section-header">Live Detection</h2>', unsafe_allow_html=True)
    st.markdown("Monitor input for phishing risks in real-time.")
    
    live_toggle = st.checkbox("Enable Live Detection", key="live_toggle")
    live_input = st.text_area("Input Text", placeholder="Start typing to detect phishing in real-time", height=200, key="live_input")
    
    if live_toggle and live_input.strip() and live_input != st.session_state.last_input:
        with st.spinner("Analyzing in real-time..."):
            prediction = simulate_prediction(live_input, st.session_state.prediction_history)
            st.session_state.last_input = live_input
            
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.success(f"Live Prediction: **{prediction['result']}** (Confidence: {prediction['confidence']:.2%})")
            st.write(f"**Type**: {prediction['input_type']}")
            st.write(f"**Risk Level**: {prediction['risk_level']}")
            st.write(f"**Why?** {prediction['explanation']}")
            st.markdown(f"**Input Preview**: {prediction['highlighted_text']}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.progress(prediction['confidence'])
            st.caption(f"Confidence Level: {prediction['confidence']:.2%}")
            
            st.session_state.prediction_history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Input": live_input[:50] + "..." if len(live_input) > 50 else live_input,
                "Result": prediction['result'],
                "Confidence": f"{prediction['confidence']:.2%}",
                "Type": prediction['input_type'],
                "Risk": prediction['risk_level']
            })
            st.session_state.confidence_history.append(prediction['confidence'])
            
            if len(st.session_state.confidence_history) > 1:
                st.subheader("Confidence Trend")
                confidence_df = pd.DataFrame({
                    "Prediction": range(len(st.session_state.confidence_history)),
                    "Confidence": st.session_state.confidence_history
                })
                st.line_chart(confidence_df.set_index("Prediction")["Confidence"])

with tab3:
    st.markdown('<h2 class="section-header">Help & Examples</h2>', unsafe_allow_html=True)
    with st.expander("How to Use"):
        st.markdown("""
        - **Predict Tab**: Enter an email or URL and click "Predict" to check for phishing risks.
        - **Live Detection Tab**: Enable live detection to monitor input as you type.
        - **Prediction History**: View past predictions in the sidebar and download them as a CSV.
        - **Indicators**:
          - Suspicious words are **highlighted in red**.
          - Links are **highlighted in yellow**.
        """)
    with st.expander("Example Inputs"):
        st.markdown("""
        - **Phishing URL**: `https://bayareafastrak.org-etcsw.win/`
        - **Legitimate Email**: `Meeting tomorrow at 10 AM to discuss project updates.`
        - **Phishing Email**: `Urgent: Verify your account password at bank-login.com`
        - **Legitimate URL**: `https://www.google.com`
        - **Suspicious URL**: `http://192.168.1.1/login?verify=123456789`
        - **Phishing Email**: `Alert: Your PayPal account is locked. Click here to unlock.`
        """)
    with st.expander("About"):
        st.markdown("""
        This system uses advanced rules to detect phishing in real-time, analyzing:
        - Suspicious words and phrases (e.g., "urgent", "click here")
        - URL structures and randomness
        - Email patterns and links
        
        Designed for real-time cybersecurity with a user-friendly interface.
        """)

# Footer
st.markdown('<div class="footer">Phishing Detection System | Developed for Cybersecurity Applications | 2025</div>', unsafe_allow_html=True)