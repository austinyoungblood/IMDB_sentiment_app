# Movie Review Sentiment Analyzer

This project packages a Streamlit sentiment analysis application in a Docker container. The app loads a trained machine learning model from `sentiment_model.pkl` and predicts whether user-entered movie review text has positive or negative sentiment.

## Prerequisites

- Docker installed and running
- Make installed

## How to Run with Docker

1. Open a terminal in the `sentiment-streamlit-app` directory:

   ```bash
   cd sentiment-streamlit-app
   ```

2. Build the Docker image:

   ```bash
   make build
   ```

3. Run the app container:

   ```bash
   make run
   ```

4. Open the app in your browser:

   ```text
   http://localhost:8501
   ```

5. To remove the Docker image when you are finished:

   ```bash
   make clean
   ```

## How to Run Locally Without Docker

1. Create a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

If `sentiment_model.pkl` is missing, download the IMDB Dataset of 50K Movie Reviews from Kaggle, place the zip file in a `data/` folder, then run:

```bash
python train_model.py
```
