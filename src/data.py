"""UCI Wine Quality dataset acquisition and validation."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
FEATURES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
TARGET = 'quality'
URLS = {
    'red': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    'white': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
}

def download_data(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for kind, url in URLS.items():
        dest = DATA_DIR / f'winequality-{kind}.csv'
        if not dest.exists() or force:
            try:
                urlretrieve(url, dest)
                print(f'Downloaded {dest}')
            except Exception as exc:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f'Could not download {url}. Download winequality-{kind}.csv manually from the UCI Wine Quality dataset and put it in data/.') from exc

def load_data(kind='red', path=None):
    if kind not in URLS:
        raise ValueError('kind must be red or white')
    file = Path(path) if path is not None else DATA_DIR / f'winequality-{kind}.csv'
    if not file.exists():
        raise FileNotFoundError(f'Missing {file}. Run: python -m src.data --download')
    df = pd.read_csv(file, sep=';')
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Dataset missing columns: {sorted(missing)}')
    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    if df.empty:
        raise ValueError('No valid labeled rows')
    return df

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    if a.download:
        download_data(force=a.force)
    else:
        print(load_data().describe().to_string())
