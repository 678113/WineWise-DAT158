"""Streamlit application for WineWise."""
import json
import streamlit as st
from src.data import FEATURES, ROOT
from src.predict import load_model, predict

st.set_page_config(page_title='WineWise | Quality Predictor', page_icon='🍷', layout='centered')
st.title('🍷 WineWise')
st.subheader('Wine quality prediction using machine learning')
st.write('Enter laboratory measurements of a wine to estimate its quality score. This educational model predicts the UCI dataset’s expert-assigned quality rating, not taste preferences or safety.')

@st.cache_resource
def cached_model():
    return load_model()

try:
    model = cached_model()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(str(exc))
    st.info('First download the dataset and train the model. See README.md for instructions.')
    st.stop()

metrics_path = ROOT / 'results' / 'metrics.json'
if metrics_path.exists():
    info = json.loads(metrics_path.read_text(encoding='utf-8'))
    st.caption(f"Trained on {info['dataset']} wine data • Model: {info['selected_model']}")

# Defaults are illustrative form values, not a real sample or a guaranteed wine profile.
defaults = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
with st.form('wine_form'):
    st.markdown('### Wine measurements')
    measurements = {}
    left, right = st.columns(2)
    for idx, (feature, default) in enumerate(zip(FEATURES, defaults)):
        with (left if idx % 2 == 0 else right):
            measurements[feature] = st.number_input(feature.title(), min_value=0.0,
                                                     value=float(default), format='%.4f',
                                                     help='Use the units provided in the UCI Wine Quality dataset documentation.')
    submitted = st.form_submit_button('Predict wine quality', type='primary', use_container_width=True)
if submitted:
    try:
        result = predict(model, measurements)
        st.metric('Predicted quality score', f'{result:.2f} / 10')
        st.caption('This is a regression estimate. Scores may be fractional and are not guaranteed to fall between 0 and 10.')
    except (ValueError, TypeError) as exc:
        st.error(f'Could not predict: {exc}')

with st.expander('About the model and its limitations'):
    st.write('The app uses a model trained from scratch on UCI Wine Quality data. It compares a mean baseline, linear regression, and random forest using cross-validation; the final model is evaluated on a held-out test set.')
    if metrics_path.exists():
        m = info['test_metrics']
        st.write(f"Held-out test MAE: {m['MAE']:.3f} | RMSE: {m['RMSE']:.3f} | R²: {m['R2']:.3f}")
    st.write('Predictions are estimates and may be less reliable for measurements unlike those in the training data. Not suitable for health or safety decisions.')
