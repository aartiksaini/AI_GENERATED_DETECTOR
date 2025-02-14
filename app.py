import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Load Model & Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertForSequenceClassification.from_pretrained("AI_DETECTOR").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("AI_DETECTOR")

bert_model.eval()  # Set model to evaluation mode

def encode_text(text):
    """Tokenizes text and returns input IDs & attention mask."""
    encoding = bert_tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    return encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

def predict(text):
    """Predicts AI-generated and human-written probabilities."""
    input_ids, attention_mask = encode_text(text)
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
        ai_prob = probs[0, 1].item() * 100  # AI-generated probability
        human_prob = probs[0, 0].item() * 100  # Human-written probability
    
    return ai_prob, human_prob

# Streamlit UI
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4A90E2;
    }
    .subtext {
        font-size: 18px;
        text-align: center;
        color: #333;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
    }
    .stTextArea textarea {
        border: 1px solid #ccc !important;
        border-radius: 5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 AI Text Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Detect whether the given text is AI-generated or human-written.</div>", unsafe_allow_html=True)

user_input = st.text_area("Enter Text:", height=400)

if st.button("Analyze", help="Click to analyze the text"):
    if user_input.strip():
        ai_percentage, human_percentage = predict(user_input)
        st.markdown(f"<div class='result'>🤖 AI-Generated Probability: {ai_percentage:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'>📝 Human-Written Probability: {human_percentage:.2f}%</div>", unsafe_allow_html=True)
        
        if ai_percentage > 50:
            st.error("⚠️ This text is likely AI-generated!")
        else:
            st.success("✅ This text appears to be human-written!")
    else:
        st.warning("Please enter some text to analyze.")
