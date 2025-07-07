import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import spacy
import re
import pandas as pd

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Load model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_model")
le_classes = pd.read_json("label_encoder.json")[0].tolist()

# Text preprocessing
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def preprocess(text):
    text = clean_text(text).lower()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Prediction function
def predict_sentiment(text):
    model.eval()
    cleaned_text = preprocess(text)
    encoded = tokenizer(cleaned_text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).numpy()[0]
    return le_classes[pred]

# Streamlit UI
st.title("Sentiment Analysis with BERT")
st.write("Enter a sentence to predict its sentiment (positive, negative, or neutral).")

user_input = st.text_area("Input your sentence here:")
if st.button("Predict"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.write("Please enter a sentence.")