import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    model = BertForSequenceClassification.from_pretrained('models/bert_phishing')
    tokenizer = BertTokenizer.from_pretrained('models/bert_phishing')
    model.eval()
    logger.info("Model and tokenizer loaded")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("Phishing Detection System")
st.markdown("Enter an email or URL to check if it's phishing or legitimate.")

user_input = st.text_area("Input Text (Email or URL)", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = probs.argmax(-1).item()
                confidence = probs[0][pred].item()
            result = "Phishing" if pred == 1 else "Legitimate"
            st.success(f"Prediction: **{result}** (Confidence: {confidence:.2%})")
            logger.info(f"Prediction for '{user_input[:50]}...': {result}, Confidence: {confidence}")
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            st.error(f"Prediction failed: {e}")

st.markdown("### Example Inputs")
st.write("- Phishing URL: `https://bayareafastrak.org-etcsw.win/`")
st.write("- Legitimate Email: `Meeting tomorrow at 10 AM to discuss project updates.`")