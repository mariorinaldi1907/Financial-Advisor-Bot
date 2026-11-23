"""SPY research dashboard. Run: python -m streamlit run app.py."""
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Financial Advisor Bot', page_icon='📈', layout='wide')
st.title('📈 Financial Advisor Bot')
st.caption('SPY · 10 trading day direction · educational research prototype')


@st.cache_resource
def load_bundle(path, modified):
    return joblib.load(path)


@st.cache_data
def load_csv(path, modified):
    return pd.read_csv(path, index_col=0, parse_dates=True)


paths = {name: ROOT / path for name, path in {
    'model': 'models/forecast.joblib', 'features': 'data/spy_features.csv',
    'predictions': 'reports/holdout_predictions.csv',
}.items()}
if any(not p.exists() for p in paths.values()):
    st.error('Training artifacts are missing. Run: python -m src.train')
    st.stop()
try:
    bundle = load_bundle(str(paths['model']), paths['model'].stat().st_mtime_ns)
    data = load_csv(str(paths['features']), paths['features'].stat().st_mtime_ns)
    historical = load_csv(str(paths['predictions']), paths['predictions'].stat().st_mtime_ns)
    model, meta = bundle['model'], bundle['metadata']
    if str(data.index.max().date()) != meta['raw_end']:
        raise ValueError('Model and feature snapshot dates disagree. Retrain all artifacts together.')
    if data[meta['features']].isna().any().any():
        raise ValueError('Feature snapshot contains missing model inputs.')
except Exception as exc:
    st.error(f'Could not load matching training artifacts: {exc}')
    st.stop()

age = (pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize() - data.index.max()).days
st.caption(f"Data: {meta['raw_start']} to {meta['raw_end']} · Selected model: {meta['selected_model']} · {meta['raw_rows']:,} market sessions")
if age > 7:
    st.warning(f'This snapshot is {age} calendar days old. Refresh data and retrain before interpreting it as a current forecast.')

st.sidebar.header('Explore forecasts')
mode = st.sidebar.radio('View', ['Latest forecast', 'Historical holdout'])
if mode == 'Historical holdout':
    picked = st.sidebar.date_input('As-of date', value=historical.index.max().date(),
                                   min_value=historical.index.min().date(), max_value=historical.index.max().date())
    # Use only a session at or before the chosen date, never a future Monday.
    selected = historical.index[historical.index <= pd.Timestamp(picked)][-1]
    p_up = float(historical.loc[selected, f"{meta['selected_model']}_p_up"])
    st.info(f"Historical evaluation: predictions were saved from a model whose training labels ended on {meta['development_label_end']}. It was not fitted on this holdout.")
else:
    selected = data.index.max()
    row_input = data.loc[[selected], meta['features']]
    p_up = float(model.predict_proba(row_input)[0, list(model.classes_).index(1)])
    st.info(f"Latest snapshot forecast: the deployed model was refitted using outcomes known through {meta['live_label_end']}. This refitted model is used only for the latest session.")

row = data.loc[selected]
left, right = st.columns([1.5, 1])
with left:
    st.subheader('Market history')
    recent = data.loc[:selected].tail(180)
    fig = go.Figure()
    for col, label in [('Close', 'Adjusted close'), ('SMA_30', 'SMA 30'), ('SMA_100', 'SMA 100')]:
        fig.add_trace(go.Scatter(x=recent.index, y=recent[col], name=label))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=15, b=10),
                      yaxis_title='Adjusted USD', legend=dict(orientation='h'))
    st.plotly_chart(fig, width='stretch')
    st.caption(f'As of market close on {selected.date()}. Features include that session’s closing price and volume; this is not an intraday forecast.')
with right:
    st.subheader('Direction estimate')
    a, b = st.columns(2)
    a.metric('P(UP)', f'{p_up:.1%}')
    b.metric('P(DOWN / flat)', f'{1-p_up:.1%}')
    if meta['selected_model'] == 'prior_baseline':
        st.warning('No validated timing signal. The selected baseline reports the training-period frequency of UP outcomes and does not use indicators to time the market.')
    elif abs(p_up - .5) < .05:
        st.warning('No clear directional signal: model estimate is close to 50%.')
    else:
        st.write(f"Estimated direction: **{'UP' if p_up >= .5 else 'DOWN / flat'}** over the next {meta['horizon']} trading sessions.")
    st.caption('These are uncalibrated model estimates, not guaranteed odds or a measure of investment safety. The 5 percentage point neutral band is a display heuristic, not an optimized trading rule.')
    if mode == 'Historical holdout':
        actual = 'UP' if historical.loc[selected, 'Target'] == 1 else 'DOWN / flat'
        st.write(f'Observed 10-session outcome: **{actual}**')
    st.subheader('Indicator context')
    st.write(f"• Trend: SMA 30 {'above' if row.SMA_30 > row.SMA_100 else 'at or below'} SMA 100.")
    st.write(f'• RSI (14): {row.RSI_14:.1f}.')
    st.write(f'• Previous 10-session return: {row.Mom_10:+.2%}.')
    st.write(f'• Daily return volatility (20 sessions): {row.Volatility_20:.2%}.')
    st.write(f'• Volume relative to 20-session average: {row.Relative_volume:.2f}×.')
    st.caption('These describe market conditions. They do not establish why this specific model made its prediction.')

with st.expander('Evaluation and model selection', expanded=False):
    table = []
    for name, metrics in meta['holdout'].items():
        table.append({'Model': name, 'Validation log loss': meta['validation'][name]['mean_log_loss'],
                      'Holdout accuracy': metrics['accuracy'], 'Holdout balanced accuracy': metrics['balanced_accuracy'],
                      'Holdout log loss': metrics['log_loss'], 'Holdout Brier score': metrics['brier']})
    st.dataframe(pd.DataFrame(table).set_index('Model').style.format('{:.3f}'), width='stretch')
    st.write(f"Holdout: {meta['holdout_start']} to {meta['holdout_end']}. Always-UP accuracy: {meta['always_up_holdout']['accuracy']:.1%}.")
    st.caption('Selection uses the lowest mean log loss across four expanding development folds, with a 10-session gap. The final 20% is held out of selection. Lower log loss and Brier score are better. Overlapping target windows mean rows are dependent; these results do not establish trading profitability.')
    st.download_button('Download historical predictions', paths['predictions'].read_bytes(), 'holdout_predictions.csv', 'text/csv')

st.caption('Academic prototype for research and education. It does not assess your finances, determine suitable allocations, or provide personalized investment advice.')
