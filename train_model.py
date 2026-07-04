import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "imdb-dataset-of-50k-movie-reviews.zip"
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"


def main():
    # 1. Load the dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Inspect assumptions before training
    print(df.head())
    print(df.columns)
    print(df["sentiment"].value_counts())

    # 3. Split into features and labels
    X = df["review"].fillna("").tolist()
    y = df["sentiment"].str.lower()

    # 4. Create the pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", MultinomialNB()),
    ])

    # 5. Grid search over max_features
    param_grid = {
        "tfidf__max_features": [5_000, 20_000, 50_000, None],
    }

    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X, y)

    print(f"Best params:    {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    # 6. Save best estimator
    joblib.dump(search.best_estimator_, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
