# WineWise — DAT158 Wine Quality Predictor

Try the WineWise web application:

https://winewise-dat158-aq76uxvru6mqgpaonzympd.streamlit.app

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

Initial project code and documentation were developed with assistance from ChatGPT.

The project was subsequently set up, executed, tested, and deployed by the student. The machine learning models were trained using the UCI Wine Quality dataset, and their performance was evaluated using cross-validation and a held-out test set.

The Streamlit application was tested locally and deployed publicly through Streamlit Community Cloud. The student has verified that the application runs and produces wine quality predictions.

AI assistance was used as a development and learning tool. The final project, testing, results, and documentation remain the student's responsibility.

## Deployment

The WineWise application is deployed publicly using Streamlit Community Cloud.

Live application: https://winewise-dat158-aq76uxvru6mqgpaonzympd.streamlit.app

Source code, trained model, evaluation metrics, and tests are available in this GitHub repository. The application runs `app.py` from the `main` branch.

Final Project Report

The DAT158 final project report is available in the reports folder.

The report documents the problem definition, dataset, machine learning methodology, evaluation results, deployment, and references.
