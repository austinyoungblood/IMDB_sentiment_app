import streamlit as st
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"


@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review and this app will predict whether the sentiment "
    "is positive or negative using a trained TF-IDF + Naive Bayes pipeline."
)

model = load_model()

user_text = st.text_area("Enter a movie review to analyze:")

analyze_clicked = st.button("Analyze")

if analyze_clicked:
    if not user_text.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        review = user_text.strip()
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]

        class_labels = model.classes_

        # Find probability for predicted class
        predicted_index = list(class_labels).index(prediction)
        confidence = probabilities[predicted_index]

        if prediction == "positive":
            st.success(f"Predicted Sentiment: Positive 👍")
        else:
            st.error(f"Predicted Sentiment: Negative 👎")

        st.write(f"Confidence: {confidence:.2%}")
