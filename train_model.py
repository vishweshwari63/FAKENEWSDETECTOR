import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dataset
data = {
    "text": [
        "Breaking news: Stock market crashes!",
        "Celebrity X spotted at local cafe",
        "Scientists discover cure for disease",
        "Fake news: Aliens invade city",
        "Election results announced today",
        "Click here to win $1000 now!",
        "Government plans new tax reforms",
        "You won’t believe this shocking news!"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
}
df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
