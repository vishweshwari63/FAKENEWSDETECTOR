# app.py
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import newspaper

# --- NLTK setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# --- Load ML model and vectorizer ---
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Summarizer with caching ---
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1  # CPU
    )

summarizer = load_summarizer()

# --- Helper function to clean text ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# --- Robust summarization function ---
def summarize_long_text(text, max_chars=1000):
    # Very short text: return as is
    if len(text) < 100:
        return text

    summaries = []
    start = 0
    while start < len(text):
        chunk = text[start:start+max_chars]
        try:
            chunk_summary = summarizer(
                chunk,
                max_length=150,
                min_length=40,
                do_sample=False
            )[0]['summary_text']
            summaries.append(chunk_summary)
        except Exception:
            # fallback: use the chunk itself
            summaries.append(chunk)
        start += max_chars
    final_summary = " ".join(summaries)
    return final_summary

# --- Streamlit Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --- CSS for UI ---
st.markdown("""
<style>
.main { background: linear-gradient(145deg, #f5f7fa, #e0f7fa); padding: 30px 60px; }
.title { text-align: center; font-size: 52px; font-weight: 900; color: #0D47A1; 
        background: linear-gradient(90deg,#0D47A1,#42A5F5); -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; animation: slidein 2s ease-in-out; }
@keyframes slidein { 0% {opacity: 0; transform: translateY(-30px);} 100% {opacity: 1; transform: translateY(0);} }
.subtitle { text-align: center; font-size: 20px; color: #34495E; margin-bottom: 40px; }
.glass-card { background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(15px); border-radius: 25px;
             padding: 25px; margin: 10px 0; box-shadow: 0 15px 40px rgba(0,0,0,0.2);
             transition: transform 0.3s, box-shadow 0.3s; }
.glass-card:hover { transform: scale(1.03); box-shadow: 0 20px 50px rgba(0,0,0,0.3); }
.prediction-real { background: linear-gradient(135deg,#28B463,#82E0AA); color: white; border-radius: 25px;
                  padding: 30px; text-align: center; font-size: 22px; font-weight: 700; 
                  box-shadow: 0 10px 35px rgba(0,0,0,0.2); animation: glow 1.5s infinite alternate; }
.prediction-fake { background: linear-gradient(135deg,#CB4335,#F1948A); color: white; border-radius: 25px;
                  padding: 30px; text-align: center; font-size: 22px; font-weight: 700;
                  box-shadow: 0 10px 35px rgba(0,0,0,0.2); animation: glow 1.5s infinite alternate; }
@keyframes glow { from { box-shadow: 0 0 15px rgba(255,255,255,0.3);} to { box-shadow: 0 0 35px rgba(255,255,255,0.6);} }
.stButton>button { background: linear-gradient(90deg, #1E88E5, #42A5F5); color: white; font-size: 18px;
                   font-weight: bold; border-radius: 15px; padding: 14px 35px; transition: transform 0.2s, box-shadow 0.2s; }
.stButton>button:hover { transform: scale(1.05); box-shadow: 0 5px 20px rgba(0,0,0,0.2); cursor: pointer; }
.footer { text-align: center; color: #566573; font-size: 14px; margin-top: 50px; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='title'>📰 Fake News Detector & Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Paste a news article or provide a URL to check if it is Real or Fake and get a concise summary.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Section ---
st.subheader("Enter News Article or URL:")
choice = st.radio("Choose input type:", ("Paste Article", "Provide URL"))

user_input = ""
if choice == "Paste Article":
    user_input = st.text_area("Paste your article here:", height=300, placeholder="Copy and paste the news article text here...")
else:
    url = st.text_input("Enter news article URL:")
    if url:
        with st.spinner("Fetching article..."):
            try:
                article = newspaper.Article(url)
                article.download()
                article.parse()
                user_input = article.text
                st.success("Article fetched successfully!")
                with st.expander("Preview Fetched Article"):
                    st.write(user_input)
            except Exception as e:
                st.error(f"Failed to fetch article. Error: {e}")

# --- Analyze Button ---
analyze_disabled = True if user_input.strip() == "" else False
if st.button("Analyze News", disabled=analyze_disabled):
    # --- Fake News Detection ---
    clean_input = clean_text(user_input)
    vect_input = vectorizer.transform([clean_input])
    prediction = model.predict(vect_input)[0]
    confidence = model.predict_proba(vect_input).max() * 100
    label = "Real" if prediction == 0 else "Fake"

    col1, col2 = st.columns([1, 2])

    # Prediction Card
    with col1:
        icon = "✔️" if label == "Real" else "❌"
        card_class = "prediction-real" if label == "Real" else "prediction-fake"
        st.markdown(f"<div class='{card_class}'>{icon} Prediction: {label}<br>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)

    # Summary Card
    with col2:
        try:
            summary = summarize_long_text(user_input, max_chars=1000)
            with st.expander("View Summary"):
                st.markdown(f"<div class='glass-card'>{summary}</div>", unsafe_allow_html=True)
            st.text_area("Copy Summary Here:", summary, height=120)
            st.download_button(
                label="Download Result as TXT",
                data=f"Prediction: {label}\nConfidence: {confidence:.2f}%\n\nSummary:\n{summary}",
                file_name="news_analysis.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Summarization failed: {e}")

    st.markdown("<script>window.scrollTo(0,document.body.scrollHeight);</script>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p class='footer'>Designed for Students | Internship Project Demo</p>", unsafe_allow_html=True)
