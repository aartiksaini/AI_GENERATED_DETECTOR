import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

# Load Model and Tokenizer from Hugging Face Hub

pipe = pipeline("text-classification", model="aartik001/ai_text_detector", framework="pt")  # Force PyTorch

tokenizer = AutoTokenizer.from_pretrained("aartik001/ai_text_detector")
model = AutoModelForSequenceClassification.from_pretrained("aartik001/ai_text_detector")

def predict(text):
    """Predict AI-generated and human-written probabilities."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        ai_prob = probs[0, 1].item() * 100  # AI-generated probability
        human_prob = probs[0, 0].item() * 100  # Human-written probability
    return ai_prob, human_prob

# Streamlit UI
st.title("🧠 AI Text Detector")
st.write("Detect whether the given text is AI-generated or human-written.")

user_input = st.text_area("Enter Text:", height=300)

if st.button("Analyze"):
    if user_input.strip():
        ai_percentage, human_percentage = predict(user_input)
        st.write(f"🤖 AI-Generated Probability: {ai_percentage:.2f}%")
        st.write(f"📝 Human-Written Probability: {human_percentage:.2f}%")
        
        if ai_percentage > 50:
            st.error("⚠️ This text is likely AI-generated!")
        else:
            st.success("✅ This text appears to be human-written!")
    else:
        st.warning("Please enter some text to analyze.")
