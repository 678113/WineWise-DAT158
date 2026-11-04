"""Inference helpers for the website."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import ROOT, FEATURES

MODEL_PATH = ROOT / 'models' / 'wine_quality_pipeline.joblib'

def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Model missing: {path}. Run python -m src.train first.')
    # Only load model artifacts produced by this project; joblib files can execute code.
    return joblib.load(path)

def predict(model, measurements):
    missing = set(FEATURES) - set(measurements)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    row = {}
    for name in FEATURES:
        value = float(measurements[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite, nonnegative number')
        row[name] = value
    result = float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
    if not np.isfinite(result):
        raise ValueError('Model returned an invalid prediction')
    return result
