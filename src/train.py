"""Train and evaluate models with a held-out test set."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import ROOT, FEATURES, TARGET, load_data

SEED = 42

def candidates():
    return {
        'Dummy (mean)': DummyRegressor(strategy='mean'),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=SEED, n_jobs=1),
    }

def pipeline(estimator):
    prep = ColumnTransformer([('numeric', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ]), FEATURES)], remainder='drop')
    return Pipeline([('preprocess', prep), ('model', estimator)])

def metrics(y_true, pred):
    return {'MAE': float(mean_absolute_error(y_true, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, pred))),
            'R2': float(r2_score(y_true, pred))}

def train(kind='red', data_path=None, output_dir=None):
    df = load_data(kind, data_path)
    X, y = df[FEATURES], df[TARGET]
    if len(df) < 30:
        raise ValueError('At least 30 labeled observations are required')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scores = {}
    for name, model in candidates().items():
        p = pipeline(model)
        cv = cross_validate(p, X_train, y_train, cv=3,
                            scoring={'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'},
                            n_jobs=1)
        scores[name] = {'CV_MAE': float(-cv['test_mae'].mean()),
                        'CV_RMSE': float(-cv['test_rmse'].mean())}
    winner = min(scores, key=lambda n: scores[n]['CV_MAE'])
    final = pipeline(candidates()[winner])
    final.fit(X_train, y_train)
    test = metrics(y_test, final.predict(X_test))
    output = Path(output_dir) if output_dir else ROOT
    (output / 'models').mkdir(parents=True, exist_ok=True)
    (output / 'results').mkdir(parents=True, exist_ok=True)
    joblib.dump(final, output / 'models' / 'wine_quality_pipeline.joblib')
    report = {'dataset': kind, 'rows': len(df), 'train_rows': len(X_train),
              'test_rows': len(X_test), 'seed': SEED, 'selection_metric': 'cross-validation MAE',
              'cross_validation': scores, 'selected_model': winner, 'test_metrics': test,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_source': 'https://archive.ics.uci.edu/dataset/186/wine+quality'}
    (output / 'results' / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['red', 'white'], default='red')
    a = p.parse_args()
    train(a.kind)
