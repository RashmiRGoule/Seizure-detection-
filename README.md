# NeuroScan — Early Seizure Detection from EEG (1D-CNN)

A full web app for screening EEG reports for seizure activity:
**login/signup → upload EEG CSV → 1D-CNN prediction → history of past results.**

Built on the UCI **Epileptic Seizure Recognition** dataset (11,500 rows, 178
EEG signal samples per row, balanced across 5 original classes).

---

## Why this replaces the Random Forest version

Your original Random Forest setup gave **inconsistent results on every run**.
That symptom (not just "low accuracy", but *changing* accuracy) almost always
comes from missing reproducibility controls, not the model itself:

| Cause | Fix applied in this project |
|---|---|
| No fixed `random_state` | `SEED = 42` is set for Python, NumPy, and PyTorch, and used in every train/test split |
| No stratified split | `train_test_split(..., stratify=y)` used for both the train/val and val/test splits, so the seizure:non-seizure ratio (1:4 after binarizing) is identical across all three sets every run |
| Plain accuracy on imbalanced data | We report **Precision, Recall, F1, ROC-AUC, and a confusion matrix** — accuracy is barely mentioned, since a model predicting "non-seizure" every time would still score ~80% accuracy on this data |
| Random Forest on raw tabular features | Replaced with a **1D Convolutional Neural Network** that treats the 178 values as an ordered time-series and learns local waveform shapes (spikes, sharp waves) via convolution — closer to how EEG is actually structured |

On top of that, the CNN uses **class-weighted loss** (`pos_weight` in
`BCEWithLogitsLoss`) so the 1:4 class imbalance doesn't bias it toward
always predicting "non-seizure", plus **early stopping** on validation F1
so it doesn't overfit.

---

## Project structure

```
seizure_app/
├── app.py                  # Flask app factory, run this to start the server
├── extensions.py           # shared SQLAlchemy db object
├── models.py                # User + PredictionHistory database models
├── auth.py                  # signup / login / logout routes
├── predict.py                # dashboard, upload, prediction, history routes
├── inference.py              # loads the trained CNN and runs predictions
├── requirements.txt
├── model/
│   ├── train_cnn.py          # training script — run this FIRST
│   ├── seizure_cnn.pt         # (generated after training)
│   ├── scaler.pkl              # (generated after training)
│   ├── training_report.json    # (generated after training) metrics
│   └── training_curve.png       # (generated after training) loss/F1 plot
├── templates/                 # HTML pages (Jinja2)
├── static/css/style.css        # all styling
├── uploads/                    # uploaded EEG CSVs are saved here temporarily
└── instance/app.db             # SQLite database (created on first run)
```

---

## Setup

### 1. Install dependencies

```bash
cd seizure_app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on PyTorch:** if `pip install torch` is slow or fails on your
> machine, get the right command for your OS/CUDA version from
> https://pytorch.org/get-started/locally/ and run that instead, then
> `pip install -r requirements.txt` again for the rest.

### 2. Train the model (do this before running the app)

```bash
python model/train_cnn.py --csv /path/to/Epileptic_Seizure_Recognition.csv
```

This will:
- Load and stratified-split the dataset (70% train / 15% val / 15% test)
- Train the CNN with early stopping (up to 30 epochs)
- Print Precision / Recall / F1 / ROC-AUC / confusion matrix for the test set
- Save `model/seizure_cnn.pt`, `model/scaler.pkl`, `model/training_report.json`,
  and `model/training_curve.png`

Takes roughly 2-5 minutes on CPU for this dataset size.

### 3. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

- If you run the app **before** training, the dashboard will show a clear
  "model not ready" message instead of crashing — train first, then refresh.

---

## Using the app

1. **Sign up** with an email + password (stored as a salted hash, never
   plain text).
2. **Log in.**
3. On the dashboard, **upload an EEG CSV**. Accepted formats:
   - A single row of 178 numeric EEG values (no header)
   - A single row of 178 values with a header row above it
   - A full original-dataset-style row (ID column + X1..X178 + trailing
     label column) — the label column, if present, is ignored, since the
     whole point is to predict it
   - If your CSV happens to contain multiple rows, only the **first** row
     is used (one EEG report = one prediction)
4. View the **result**: Seizure / Non-Seizure + model confidence %.
5. Check **History** any time to see all of your past screenings.

---

## Re-evaluating / improving accuracy further

If you want to push accuracy higher after this baseline:
- Try a deeper CNN (more conv blocks) or add a `BiLSTM` after the conv
  layers to capture longer-range temporal dependencies.
- Try data augmentation on the signal (small time-shifts, noise injection)
  since EEG datasets are often small.
- Compare against **XGBoost on FFT/band-power features** as a second model
  and report both in your project — "classical ML vs deep learning"
  comparisons are also good content for a project report or viva.
- Use k-fold cross-validation instead of a single train/val/test split for
  a more robust accuracy estimate to report.

---

## Important: scope and limitations

This is a **student/portfolio project**, not a certified medical device:

- The dataset is single-channel EEG segments from a research dataset, not
  real multi-channel clinical EEG recordings — results won't directly
  transfer to real hospital EEG equipment without re-training on
  appropriate clinical data.
- Auth here is intentionally simple (no email verification, no password
  reset, no CSRF tokens) — fine for a demo/college project, but you'd want
  to add `Flask-WTF` (CSRF), email verification, and rate-limiting before
  any real deployment.
- Treat all predictions as a **screening aid**, not a diagnosis — the UI
  copy reflects this on purpose, keep that framing if you present this
  project.
