# WineWise — DAT158 Wine Quality Predictor

A reproducible ML regression project and Streamlit website predicting the sensory quality score of red wine from physicochemical laboratory measurements. The website uses a saved model trained by the project itself.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m src.data --download
python -m src.train --kind red
python -m pytest -q
streamlit run app.py
```

If the download command fails, download `winequality-red.csv` and `winequality-white.csv` manually from https://archive.ics.uci.edu/dataset/186/wine+quality and place them under `data/`.

`python -m src.train` trains a mean DummyRegressor baseline, LinearRegression, and RandomForestRegressor. It selects the model with lowest 3-fold cross-validated MAE on the training split, refits that model on the full training split, and evaluates it **once** on the 20% held-out test split. It saves `models/wine_quality_pipeline.joblib` and `results/metrics.json`. The model pipeline includes missing-value imputation and scaling.

The Streamlit website loads that saved pipeline and accepts the 11 laboratory inputs used by the model. A trained model is required before the website can make predictions.

## Project structure

- `src/data.py`: download, load, validate dataset
- `src/train.py`: preprocessing, model comparison, evaluation, serialization
- `src/predict.py`: inference and validation
- `app.py`: Streamlit website
- `tests/`: unit tests with isolated toy fixtures (not used for actual model evaluation)
- `results/metrics.json`: generated evaluation metrics
- `models/wine_quality_pipeline.joblib`: generated trained pipeline
- `reports/`: DAT158 report to be completed using actual evaluation results

## Limitations

The quality score is a subjective sensory assessment, not a safety measure. Model generalization is limited to the characteristics and distribution of the selected UCI red-wine dataset. The user interface assumes inputs in the same units as the UCI dataset. Do not load arbitrary joblib files from untrusted sources.

## References

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.
- UCI Wine Quality dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
- scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/

## AI assistance

Initial project code and documentation were generated with assistance from ChatGPT. Review, understand, and test all outputs before submitting. Add your own development and source-use disclosures as appropriate.

## Deployment

Push this project to a public GitHub repository, with the trained model artifact and results included. Deploy `app.py` through Streamlit Community Cloud and verify the live URL. Public deployment has not been performed in this environment.
