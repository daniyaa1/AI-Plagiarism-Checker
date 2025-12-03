import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re
from nltk.probability import FreqDist
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import string
import nltk
import os

# Setup NLTK path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

for res in ['stopwords', 'brown']:
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res, download_dir=nltk_data_path)

# Load GPT-2 (called only when needed)
@st.cache_resource
def load_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32)
        model.to('cpu')
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# Tokenization without punkt
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def calculate_perplexity(text, tokenizer, model):
    if tokenizer is None or model is None:
        return float('inf')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def calculate_burstiness(text):
    tokens = simple_tokenize(text)
    word_freq = FreqDist(tokens)
    repeated = sum(1 for count in word_freq.values() if count > 1)
    return repeated / len(word_freq) if len(word_freq) > 0 else 0

def plot_top_repeated_words(text):
    tokens = simple_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    if top_words:
        words, counts = zip(*top_words)
        fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title='Top 10 Most Repeated Words')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough words to plot.")

# Layout & styles
st.set_page_config(layout="wide", page_title="AI Plagiarism Checker")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #ddefff, #ffffff);
}
h1 {
    text-align: center;
    font-size: 3em;
    color: #4a47a3;
    font-weight: bold;
    margin-top: 0.5em;
}
textarea {
    background-color: #fdfdfd !important;
    border: 1px solid #ccc !important;
    border-radius: 10px !important;
    padding: 12px !important;
    font-size: 16px !important;
    color: #333 !important;
}
button[kind="primary"] {
    background-color: #4a47a3 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    transition: transform 0.2s ease-in-out;
}
button[kind="primary"]:hover {
    background-color: #3d3999 !important;
    transform: scale(1.05);
}
.st-success, .st-error, .st-warning {
    border-radius: 10px;
    padding: 10px;
    margin-top: 10px;
}
h2, h3 {
    color: #333;
    font-weight: 600;
    margin-top: 1em;
}
.stCaption {
    font-size: 12px;
    color: #888;
    font-style: italic;
}
</style>

<script>
document.addEventListener("DOMContentLoaded", function() {
    const btn = window.parent.document.querySelector('button[kind="primary"]');
    if (btn) {
        btn.innerHTML = "🚀 Analyze";
        btn.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.2)";
    }
});
</script>

<!-- Header Branding -->
<div style="text-align: center; margin-bottom: 20px;">
    <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="80">
    <p style="color:#4a47a3; font-size: 20px; font-weight: 500;">Detect AI-generated content with burstiness & perplexity</p>
</div>
""", unsafe_allow_html=True)

# App Title
st.title("AI Plagiarism Checker")
text_input = st.text_area("Enter the text you want to analyze", height=200)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # Load model only when needed
        tokenizer, model = load_model()
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Your Input Text")
            st.success(text_input)

        with col2:
            st.subheader("Detection Score")
            perplexity = calculate_perplexity(text_input, tokenizer, model)
            burstiness = calculate_burstiness(text_input)
            st.write("Perplexity:", round(perplexity, 2))
            st.write("Burstiness Score:", round(burstiness, 2))

            # Detection Logic
            if perplexity < 50 and burstiness < 0.3:
                st.error("Text Analysis Result: AI Generated Content")
            else:
                st.success("Text Analysis Result: Likely Human-Written")

            st.caption("Disclaimer: This tool assists in detection but is not 100% accurate. Always use manual review as well.")

        with col3:
            st.subheader("Top Words")
            plot_top_repeated_words(text_input)

