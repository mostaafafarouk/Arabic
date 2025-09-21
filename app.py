import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load pre-trained Arabic sentiment model
model_name = "aubmindlab/bert-base-arabertv02-twitter-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Streamlit UI
st.title("Arabic Sentiment Analysis App")

user_input = st.text_area("Enter Arabic text here:")

if user_input:
    result = sentiment_pipeline(user_input)
    sentiment = result[0]['label']
    score = result[0]['score']
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence score:** {score:.2f}")

