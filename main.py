from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
DATA_PATHS = [
    BASE_DIR / "IMDB Dataset.csv",
    BASE_DIR / "data" / "imdb-dataset-of-50k-movie-reviews.zip",
]


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _load_dataset():
    for data_path in DATA_PATHS:
        if data_path.exists():
            return pd.read_csv(data_path)
    expected_paths = ", ".join(str(path) for path in DATA_PATHS)
    raise FileNotFoundError(f"Data file not found. Expected one of: {expected_paths}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load eagerly so a broken model/dataset fails the container at startup,
    # not on whichever request happens to hit it first.
    app.state.model = _load_model()
    app.state.dataset = _load_dataset()
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="A FastAPI backend that serves a trained IMDB sentiment analysis model.",
    version="1.0.0",
    lifespan=lifespan,
)


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        examples=["This movie was a masterpiece!"],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped


class SentimentResponse(BaseModel):
    sentiment: str


class ProbabilityResponse(BaseModel):
    sentiment: str
    probability: float


class ExampleResponse(BaseModel):
    review: str


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentResponse)
def predict(payload: TextRequest, request: Request):
    model = request.app.state.model
    sentiment = str(model.predict([payload.text])[0])
    return {"sentiment": sentiment}


@app.post("/predict_proba", response_model=ProbabilityResponse)
def predict_proba(payload: TextRequest, request: Request):
    model = request.app.state.model
    sentiment = str(model.predict([payload.text])[0])
    probabilities = model.predict_proba([payload.text])[0]
    classes = [str(label) for label in model.classes_]
    predicted_index = classes.index(sentiment)
    probability = float(probabilities[predicted_index])
    return {
        "sentiment": sentiment,
        "probability": probability,
    }


@app.get("/example", response_model=ExampleResponse)
def example(request: Request):
    df = request.app.state.dataset

    if "review" not in df.columns:
        raise HTTPException(status_code=500, detail="Dataset must contain a 'review' column.")

    reviews = df["review"].dropna()
    if reviews.empty:
        raise HTTPException(status_code=500, detail="Dataset does not contain any reviews.")

    review = str(reviews.sample(n=1).iloc[0])

    return {"review": review}
