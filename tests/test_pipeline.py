import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from src.data import FEATURES, load_data
from src.predict import predict
from src.train import pipeline, metrics

def test_prediction_on_toy_fixture():
    # Synthetic values are used ONLY for unit testing, never for claimed model results.
    X = pd.DataFrame(np.ones((12, len(FEATURES))), columns=FEATURES)
    y = np.array([5, 6] * 6)
    model = pipeline(DummyRegressor())
    model.fit(X, y)
    assert predict(model, X.iloc[0].to_dict()) == pytest.approx(5.5)

def test_rejects_invalid_input():
    with pytest.raises(ValueError, match='Missing features'):
        predict(None, {'alcohol': 10})

def test_metric_values():
    result = metrics(np.array([4, 6]), np.array([5, 5]))
    assert result['MAE'] == 1.0
    assert result['RMSE'] == 1.0

def test_data_loading(tmp_path):
    df = pd.DataFrame([{**{f: 1.0 for f in FEATURES}, 'quality': 6}])
    file = tmp_path / 'red.csv'
    df.to_csv(file, sep=';', index=False)
    assert len(load_data('red', file)) == 1
