# demo_model_setup.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample news data
news_texts = [
    "The economy is booming and unemployment is falling",
    "Scientists discover cure for common cold",
    "Aliens have landed in the city center",
    "Celebrity caught in secret conspiracy",
    "New tech startup raises millions in funding",
    "Government secretly controls all minds",
    "Local sports team wins championship",
    "Fake news claims the world is flat",
    "Education system improves in rural areas",
    "Politician cloned using DNA experiments"
]

# Labels: 0 = Real, 1 = Fake
labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Create vectorizer and transform text
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(news_texts)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_vect, labels)

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Demo model and vectorizer saved! ✅")
