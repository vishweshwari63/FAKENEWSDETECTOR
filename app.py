# app.py
import streamlit as st
import re
import requests
from bs4 import BeautifulSoup

# --- Transformers import ---
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception as e:
    HAS_TRANSFORMERS = False
    TRANSFORMERS_IMPORT_ERROR = str(e)

# --- NLTK setup ---
import nltk
def ensure_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)

ensure_nltk_resources()
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# --- Helper functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# --- Load HuggingFace detectors ---
@st.cache_resource
def load_hf_detector():
    if not HAS_TRANSFORMERS:
        return None, f"Transformers not available: {TRANSFORMERS_IMPORT_ERROR}"
    try:
        clf = pipeline("text-classification",
                       model="mrm8488/bert-tiny-finetuned-fake-news-detection",
                       top_k=1, device=-1)
        return clf, None
    except Exception as e:
        return None, f"HF detector load failed: {e}"

@st.cache_resource
def load_summarizer():
    if not HAS_TRANSFORMERS:
        return None, f"Transformers not available: {TRANSFORMERS_IMPORT_ERROR}"
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        return summarizer, None
    except Exception as e:
        return None, f"Summarizer load failed: {e}"

hf_detector, hf_detector_err = load_hf_detector()
summarizer, summarizer_err = load_summarizer()

# --- Fetch article text ---
def fetch_article_text(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")
        paragraphs = soup.find_all('p')
        text = "\n".join(p.get_text() for p in paragraphs)
        if text.strip():
            return text, None
        else:
            return None, "No text found on page."
    except Exception as e:
        return None, f"Failed to fetch URL: {e}"

# --- Summarize ---
def summarize_text(text, max_chunk=800):
    if summarizer is None:
        return text[:1000] + ("\n\n[Summarizer not available]" if summarizer_err else "")
    summaries = []
    start = 0
    while start < len(text):
        chunk = text[start:start+max_chunk]
        try:
            s = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(s)
        except Exception:
            summaries.append(chunk[:300])
        start += max_chunk
    return " ".join(summaries)

# --- Predict ---
def predict_with_hf(text):
    if hf_detector is None:
        return None, None, hf_detector_err or "HF detector not available."
    try:
        results = hf_detector(text[:512])
        # Handle list of lists (top_k)
        if isinstance(results, list) and len(results) > 0:
            first_item = results[0]
            if isinstance(first_item, list) and len(first_item) > 0:
                r = first_item[0]
            elif isinstance(first_item, dict):
                r = first_item
            else:
                return None, None, "HF detector returned unexpected format."
        else:
            return None, None, "HF detector returned empty result."

        lab = r.get("label", "")
        score = r.get("score", 0)*100
        if "FAKE" in lab.upper():
            label = "Fake"
        elif "REAL" in lab.upper():
            label = "Real"
        else:
            label = lab
        return label, round(score,2), None
    except Exception as e:
        return None, None, f"HF predict failed: {e}"


# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --- CSS ---
st.markdown("""
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1 { text-align:center; font-size:48px; background: linear-gradient(90deg,#0D47A1,#42A5F5);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { color:#34495E; }
.stButton>button { background: linear-gradient(90deg, #1E88E5, #42A5F5); color:white;
                   font-size:18px; font-weight:bold; border-radius:12px; padding:12px 30px;
                   transition: transform 0.2s, box-shadow 0.2s; }
.stButton>button:hover { transform: scale(1.05); box-shadow:0 5px 15px rgba(0,0,0,0.3); }
.footer { text-align:center; font-size:14px; color:#566573; margin-top:50px; }
.summary-box { background: rgba(245,245,245,0.4); padding:15px; border-radius:15px; 
               transition: transform 0.2s; }
.summary-box:hover { transform: scale(1.02); background: rgba(245,245,245,0.6); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📰 Fake News Detector & Summarizer</h1>", unsafe_allow_html=True)

# --- Input ---
choice = st.radio("Input type:", ("Paste Article", "Provide URL"))
user_input = ""

if choice == "Paste Article":
    user_input = st.text_area("Paste article text here:", height=300)
else:
    url = st.text_input("Article URL:")
    if url:
        with st.spinner("Fetching article..."):
            txt, err = fetch_article_text(url)
            if err:
                st.error(err)
            else:
                user_input = txt
                st.success("Fetched article successfully.")
                with st.expander("Preview Article"):
                    st.write(user_input[:5000])

# --- CSV upload option ---
st.subheader("Batch Testing (Upload CSV)")
csv_file = st.file_uploader("Upload CSV with a column 'article_text'", type=["csv"])
if csv_file:
    import pandas as pd
    df = pd.read_csv(csv_file)
    if 'article_text' not in df.columns:
        st.error("CSV must have a column named 'article_text'")
    else:
        st.write("Analyzing CSV articles...")
        results = []
        for text in df['article_text']:
            label, conf, err = predict_with_hf(text)
            results.append({"article": text[:50]+"...", "Prediction": label, "Confidence": conf})
        st.dataframe(pd.DataFrame(results))

# --- Analyze button ---
if st.button("Analyze News"):
    if not user_input or not user_input.strip():
        st.warning("Paste or fetch an article first.")
    else:
        with st.spinner("Classifying and summarizing..."):
            label, conf, err = predict_with_hf(user_input)
            if err:
                st.error(err)
            else:
                conf_text = f"{conf:.2f}%" if conf else "N/A"

                # Map any label to True/False
                label_clean = "True" if str(label).upper() in ["REAL", "LABEL_0"] else "False"

                # Prediction card
                if label_clean == "False":
                    st.markdown(
                        f"<div style='padding:18px;border-radius:12px;"
                        f"background:linear-gradient(90deg,#F1948A,#CB4335);"
                        f"color:white;font-weight:700'> ❌ Prediction: {label_clean} — Confidence: {conf_text} </div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='padding:18px;border-radius:12px;"
                        f"background:linear-gradient(90deg,#82E0AA,#28B463);"
                        f"color:white;font-weight:700'> ✅ Prediction: {label_clean} — Confidence: {conf_text} </div>",
                        unsafe_allow_html=True
                    )
                # Word counts
                word_count = len(user_input.split())
                st.write(f"Article word count: {word_count}")

               # --- Summarize ---
summary = summarize_text(user_input)
st.write(f"Summary word count: {len(summary.split())}")

# Summary box
st.markdown("<div class='summary-box'>"+summary+"</div>", unsafe_allow_html=True)

# --- Copy & Download Summary Buttons ---
col1, col2 = st.columns(2)

with col1:
    copy_code = f"""
    <script>
    function copyToClipboard(text) {{
        navigator.clipboard.writeText(text).then(function() {{
            alert("✅ Summary copied to clipboard!");
        }}, function(err) {{
            alert("❌ Failed to copy text: " + err);
        }});
    }}
    </script>
    <button onclick="copyToClipboard(`{summary}`)" 
        style="background: linear-gradient(90deg, #1E88E5, #42A5F5);
               color: white; font-size: 18px; font-weight: bold;
               border-radius: 12px; padding: 12px 30px;
               border: none; cursor: pointer; margin-top: 10px;">
        📋 Copy Summary to Clipboard
    </button>
    """
    st.markdown(copy_code, unsafe_allow_html=True)

with col2:
    download_code = f"""
    <a href="data:text/plain;charset=utf-8,{summary}" download="summary.txt">
    <button 
        style="background: linear-gradient(90deg, #1E88E5, #42A5F5);
               color: white; font-size: 18px; font-weight: bold;
               border-radius: 12px; padding: 12px 30px;
               border: none; cursor: pointer; margin-top: 10px;">
        ⬇️ Download Summary
    </button>
    </a>
    """
    st.markdown(download_code, unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div class='footer'>Designed for Students | Internship Project Demo | Powered by HuggingFace</div>", unsafe_allow_html=True)
