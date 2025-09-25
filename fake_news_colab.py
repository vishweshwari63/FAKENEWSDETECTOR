# fake_news_colab.py
import re
import joblib
import nltk
from nltk.corpus import stopwords
from transformers import pipeline

# --- NLTK setup ---
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# --- Load ML model and vectorizer ---
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Load summarizer (CPU) ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# --- Helper function to clean text ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# --- Predict function ---
def predict_article(text):
    clean_input = clean_text(text)
    vect_input = vectorizer.transform([clean_input])
    prediction = model.predict(vect_input)[0]
    confidence = model.predict_proba(vect_input).max() * 100
    label = "Real" if prediction == 0 else "Fake"
    return label, confidence

# --- Summarize function ---
def summarize_text(text, max_chars=1000):
    if len(text) < 100:
        return text
    summaries = []
    start = 0
    while start < len(text):
        chunk = text[start:start + max_chars]
        try:
            chunk_summary = summarizer(
                chunk,
                max_length=150,
                min_length=40,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(chunk_summary)
        except Exception:
            summaries.append(chunk)
        start += max_chars
    return " ".join(summaries)
