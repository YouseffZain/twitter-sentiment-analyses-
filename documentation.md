# Documentation / Notes

## Data Format
CSV with:
- `text`: the tweet or short text.
- `label`: one of `Positive`, `Negative`, `Neutral`, `Irrelevant`.

## Pipeline (Simplified)
1. Clean text (lowercase, basic punctuation stripping).
2. Tokenize & pad sequences.
3. Encode labels using scikit-learn.
4. Train an RNN family model (SimpleRNN / LSTM / BiLSTM).
5. Evaluate on a hold-out split.
6. (Optional) Export the best model and load it in Streamlit.

## Streamlit
- If `models/bilstm_demo.h5` exists, the app loads it.
- Otherwise, a tiny demo model is trained on the provided `twitter_training.csv`.
