# Movie Review Sentiment API

This project wraps a trained IMDB movie review sentiment model in a FastAPI
backend. The API accepts review text and returns whether the model predicts a
positive or negative sentiment.

## Project Files

```text
.
├── .gitignore
├── Dockerfile
├── Makefile
├── README.md
├── main.py
├── requirements.txt
├── sentiment_model.pkl
└── IMDB Dataset.csv
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Confirms that the API is running. |
| POST | `/predict` | Returns the predicted sentiment for input text. |
| POST | `/predict_proba` | Returns the predicted sentiment and confidence score. |
| GET | `/example` | Returns one random review from the IMDB training dataset. |

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API:

```bash
make dev
```

This runs the API at:

```text
http://127.0.0.1:8001
```

Port `8001` is used by default so it does not conflict with another local
service on port `8000`. To use port `8000` instead, run:

```bash
make dev PORT=8000
```

## Run with Docker

Build the Docker image:

```bash
make build
```

Run the container:

```bash
make run
```

The API will be available at:

```text
http://127.0.0.1:8001
```

To use host port `8000` instead:

```bash
make run PORT=8000
```

Remove the Docker image:

```bash
make clean
```

## FastAPI Documentation

Once the API is running, open the auto-generated FastAPI docs:

```text
http://127.0.0.1:8001/docs
```

If you are running on port `8000`, use:

```text
http://127.0.0.1:8000/docs
```

## Example Requests

Health check:

```bash
curl http://127.0.0.1:8001/health
```

Predict sentiment:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was a masterpiece!"}'
```

Predict sentiment with probability:

```bash
curl -X POST http://127.0.0.1:8001/predict_proba \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was a complete waste of time."}'
```

Get a random training example:

```bash
curl http://127.0.0.1:8001/example
```
