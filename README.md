📰 Fake News Detector & Summarizer

An AI-powered web application built with Streamlit, HuggingFace Transformers, and NLTK that helps detect fake news and generate concise article summaries.
Deploy Link : https://fakenewsdetector-oe649bpchqpkqesui2xn7g.streamlit.app/
🚀 Features

✅ Classifies news as Real or Fake with confidence score.

📑 Generates summaries of long articles using a transformer summarization model.

🔗 Accepts both pasted article text and URLs (auto-fetches article content).

📂 Batch testing – Upload a CSV file with multiple articles for prediction.

📥 Option to download summary or copy to clipboard.

🎨 Enhanced UI/UX with hover effects, styled prediction cards, and gradient buttons.

🛠️ Tech Stack

Frontend/UI: Streamlit

NLP Models: HuggingFace Transformers (bert-tiny-finetuned-fake-news-detection, distilbart-cnn-12-6)

Text Preprocessing: NLTK (stopwords, tokenization)

Web Scraping: BeautifulSoup4

Data Handling: Pandas

📂 Project Structure
FakeNewsDetector/
│── app.py              # Main Streamlit app
│── requirements.txt    # Python dependencies
│── demo_articles.csv   # Sample CSV for batch testing
│── README.md           # Project documentation

🔧 Installation

Clone the repository

git clone https://github.com/your-username/FakeNewsDetector.git
cd FakeNewsDetector


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt

▶️ Run the App
streamlit run app.py


Then open your browser at:
👉 http://localhost:8501

📊 Demo
Example Input:
NASA’s OSIRIS-REx spacecraft successfully delivered samples from the asteroid Bennu to Earth on September 24, 2023...

Example Output:

✅ Prediction: Real — Confidence: 88.24%

📝 Summary: "NASA’s OSIRIS-REx mission successfully returned asteroid samples to Earth, marking a historic first for the U.S. and offering insights into solar system formation."

📂 CSV Batch Testing

Upload a CSV file with a column named article_text:

id,article_text
1,"NASA’s OSIRIS-REx spacecraft delivered asteroid samples..."
2,"A viral post claims aliens were spotted on the Moon..."


The app will return predictions with confidence scores for each row.

📥 Download

Download summaries as .txt

Copy to clipboard directly from the app

📌 Future Improvements

Fine-tune fake news detection with larger datasets

Add multilingual support

Provide more detailed analytics (e.g., source credibility, fact-checking links)

👩‍💻 Author

Developed by VISHWESHWARI
🎓 Student Project | Internship Demo | AI for Social Good

✨ If you like this project, don’t forget to star ⭐ the repo!
