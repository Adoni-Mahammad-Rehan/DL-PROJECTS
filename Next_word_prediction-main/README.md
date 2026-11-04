# LSTM Next Word Predictor — Streamlit UI

## Files

Keep these files together in one folder:

- `app.py`
- `lstm_model.h5` — your trained LSTM model
- `tokenizer.pkl` — the exact tokenizer used during training
- `max_len.pkl` — the sequence length used when training the model
- `requirements.txt`

The model, tokenizer, and max_len must come from the same training run.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\\Scripts\\activate    # Windows

pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually http://localhost:8501.

## Important

- Do not retrain the model in `app.py`; it loads the saved model for inference.
- `load_model(..., compile=False)` is used because the UI only performs predictions.
- Keep `tokenizer.pkl` and `max_len.pkl` from the exact same run as `lstm_model.h5`.
- Only load pickle files that you created and trust.
