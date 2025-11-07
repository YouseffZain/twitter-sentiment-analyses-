# Twitter Sentiment Analysis (RNN/LSTM) & Streamlit

This repo mirrors the structure and purpose of the example you shared. It contains:
- A Jupyter notebook for training and comparing deep-learning models (Simple RNN, LSTM, BiLSTM).
- A Streamlit web app (`app.py`) to run an interactive predictor (using a small BiLSTM by default).
- Minimal training & test CSVs (placeholders) to show the expected schema.

## Goal
Classify tweets into **Positive**, **Negative**, **Neutral**, or **Irrelevant**.

## Tech Stack
Python 3.10+, TensorFlow/Keras, Pandas, Scikit-learn, Streamlit, Jupyter.

## Project Structure
```
.
├── app.py                             # Streamlit app (quick demo)
├── twitter_sentiment_analysis.ipynb   # Notebook: model training & comparison
├── twitter_training.csv               # Training data (placeholder schema)
├── twitter_test.csv                   # Test data (placeholder schema)
├── requirements.txt                   # Python dependencies
├── documentation.md                   # Extra notes / how it works
└── README.md                          # This file
```

## Quickstart (Streamlit App)
> The app expects two CSV columns: `text` (string) and `label` (one of: Positive, Negative, Neutral, Irrelevant).

```bash
# 1) Clone your repo (after you create it on GitHub)
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_FOLDER>

# 2) (Recommended) Create & activate a virtualenv
python -m venv venv
# Windows: venv\\Scripts\\activate
# macOS/Linux:
source venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Run the demo app
streamlit run app.py
```

The first run will quickly fit a tiny model **only if no saved model exists**. Replace the placeholder CSVs with your real data when ready.

## Notebook
Open `twitter_sentiment_analysis.ipynb` and run it cell-by-cell to train and compare Simple RNN, LSTM, and BiLSTM. It saves a model into `models/bilstm_demo.h5` by default (created on demand).

## Creating a GitHub repo like the example
1. Create an empty repo on GitHub (no README) or via GitHub CLI:
   ```bash
   gh repo create <your-username>/<repo-name> --public --source=. --remote=origin --push
   ```
2. Initialize & push:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: sentiment analysis scaffold"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

## License
MIT (feel free to change).
