import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

@st.cache_resource
def load_models():
    en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    hi_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    return en_tok, en_model, hi_tok, hi_model

def translate(text, tok, model):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs, max_length=512)
    return tok.decode(translated[0], skip_special_tokens=True)

st.title("🔄 English ↔ Hindi Translator")
input_text = st.text_area("Enter text (auto-detects language):", height=150)

if st.button("Translate 🚀"):
    with st.spinner("Translating..."):
        en_tok, en_model, hi_tok, hi_model = load_models()
        
        # Auto-detect Hindi (Devanagari script)
        is_hindi = any(0x0900 <= ord(c) <= 0x097F for c in input_text)
        
        if is_hindi:
            result = translate(input_text, hi_tok, hi_model)
            st.success("✅ Hindi → English")
        else:
            result = translate(input_text, en_tok, en_model)
            st.success("✅ English → Hindi")
        
        st.subheader("Translation:")
        st.write(result)
