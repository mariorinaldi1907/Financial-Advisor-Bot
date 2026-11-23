"""Multi-asset forecasting research dashboard. Run: python -m streamlit run app.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.context import ASSETS, input_hashes

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Market Research', page_icon='📈', layout='wide')
st.title('📈 Market forecasting research')
st.caption('22 ETFs · market and interest-rate context · calibrated probabilities · return ranges')
report_path, predictions_path = ROOT / 'data/research_results.json', ROOT / 'data/research_predictions.csv'
if not report_path.exists() or not predictions_path.exists():
    st.error('Train the research models first: python -m src.research')
    st.stop()


@st.cache_data
def load_report(path, modified):
    return json.loads(Path(path).read_text())


@st.cache_data
def load_predictions(path, modified):
    return pd.read_csv(path, parse_dates=['Date', 'Label_end'])


report = load_report(str(report_path), report_path.stat().st_mtime_ns)
if report['input_hashes'] != input_hashes():
    st.error('The market or environment snapshot changed after training. Retrain before using these results.')
    st.stop()
ticker = st.sidebar.selectbox('ETF', list(ASSETS), format_func=lambda t: f'{t} — {ASSETS[t][0]}')
horizons = sorted(int(h) for h in report['horizons'])
horizon = st.sidebar.selectbox('Forecast horizon', horizons, index=horizons.index(10) if 10 in horizons else 0,
                               format_func=lambda h: f'{h} trading session{ "s" if h != 1 else ""}')
mode = st.sidebar.radio('View', ['Latest forecast', 'Historical evaluation'])
result = report['horizons'][str(horizon)]
metrics = result['metrics'][ticker]
all_predictions = load_predictions(str(predictions_path), predictions_path.stat().st_mtime_ns)
history = all_predictions.loc[(all_predictions.Ticker == ticker) & (all_predictions.horizon == horizon)].sort_values('Date')
if history.empty:
    st.error('This horizon has no saved evaluation predictions. Finish its training run.')
    st.stop()
latest = next(row for row in result['latest'] if row['ticker'] == ticker)
st.subheader(f'{ticker} · {ASSETS[ticker][0]} · {horizon}-session horizon')
a,b,c,d = st.columns(4)
a.metric('Evaluation accuracy', f"{metrics['accuracy']:.1%}")
b.metric('Balanced accuracy', f"{metrics['balanced_accuracy']:.1%}")
c.metric('Prior baseline accuracy', f"{metrics['baseline_accuracy']:.1%}")
d.metric('Probability log loss', f"{metrics['log_loss']:.3f}")
st.caption(f"Evaluation: {result['test_start']} to {result['test_end']}. Selected using 2018–2023 validation: {result['selected_model']}.")
st.info('These are retrospective research results. Earlier experiments inspected overlapping dates. Daily outcomes and asset returns are correlated; they do not establish a new, independent trading advantage.')
if mode == 'Historical evaluation':
    picked = st.sidebar.date_input('As-of date', value=history.Date.max().date(),
        min_value=history.Date.min().date(), max_value=history.Date.max().date())
    forecast = history.loc[history.Date <= pd.Timestamp(picked)].iloc[-1]
    as_of, p_up = forecast.Date.date(), float(forecast.p_up)
    expected, low, median, high = [float(forecast[c]) for c in ['expected_return','lower_return','median_return','upper_return']]
    st.caption(f"Model fitting and calibration outcomes ended before {forecast.training_label_end}; prediction as of {as_of}.")
else:
    forecast = latest
    as_of, p_up = pd.Timestamp(latest['as_of']).date(), latest['p_up']
    expected, low, median, high = [latest[c] for c in ['expected_return','lower_return','median_return','upper_return']]
    st.caption(f"Latest snapshot: {as_of}. Fitting/calibration used only outcomes known through {result['live_trace']['calibration_label_end']}.")
    age = (pd.Timestamp.now(tz='America/New_York').date() - as_of).days
    if age > 7:
        st.warning(f'The saved forecast is {age} calendar days old. Refresh and retrain for a current snapshot.')

left,right = st.columns([1.4,1])
with left:
    st.subheader('Price history')
    market = pd.read_csv(ROOT / 'data/market_prices.csv', parse_dates=['Date'])
    series = market.loc[(market.Ticker == ticker) & (market.Date <= pd.Timestamp(as_of))].tail(180)
    chart = go.Figure(go.Scatter(x=series.Date, y=series.Close, name='Adjusted close'))
    chart.update_layout(height=340, margin=dict(l=5,r=5,t=15,b=5), yaxis_title='Adjusted USD')
    st.plotly_chart(chart, width='stretch')
with right:
    st.subheader('Forecast and uncertainty')
    a,b = st.columns(2)
    a.metric('Calibrated P(UP)', f'{p_up:.1%}')
    b.metric('Expected return estimate', f'{expected:+.2%}')
    st.write(f'**Nominal 80% return range:** {low:+.2%} to {high:+.2%}')
    st.write(f'Median return estimate: {median:+.2%}')
    st.caption('The return and direction models are separate. Interval adjustment uses earlier calibration dates; market dependence means 80% future coverage is not guaranteed.')
    if result['selected_model'] == 'prior':
        st.warning('The historical-prior baseline won validation. No learned timing signal is established for this horizon.')
    elif abs(p_up-.5) < .1:
        st.warning('The estimated direction is uncertain. A high probability does not establish investment suitability.')
    if mode == 'Historical evaluation':
        st.write(f"Observed {horizon}-session return: **{forecast.Forward_return:+.2%}**")
        st.write(f"Observed direction: **{'UP' if forecast.Target == 1 else 'DOWN / flat'}**")

with st.expander('All assets and model comparisons'):
    rows = []
    for asset, m in result['metrics'].items():
        rows.append(dict(ETF=asset, Accuracy=m['accuracy'], Balanced_accuracy=m['balanced_accuracy'],
            Baseline_accuracy=m['baseline_accuracy'], Log_loss=m['log_loss'], Baseline_log_loss=m['baseline_log_loss'],
            Interval_coverage=m['interval_80_coverage'], Nonoverlapping_windows=m['nonoverlapping_rows']))
    st.dataframe(pd.DataFrame(rows).set_index('ETF').style.format({c:'{:.3f}' for c in rows[0] if c not in ['ETF','Nonoverlapping_windows']}), width='stretch')
    comparison = [{'Model':name, 'Validation log loss':value,
                   'Evaluation log loss':result['candidates'][name]['mean_asset_log_loss']}
                  for name,value in result['validation_log_loss'].items()]
    st.dataframe(pd.DataFrame(comparison).set_index('Model').style.format('{:.4f}'), width='stretch')
    bootstrap = result['bootstrap']
    lo,hi = bootstrap['bootstrap_95_interval']
    st.write(f"Mean log-loss difference versus prior: {bootstrap['mean_log_loss_difference']:+.4f}; exploratory date-block interval [{lo:+.4f}, {hi:+.4f}]. Negative favors the selected model.")
    st.caption('Dates, with all assets retained together, are resampled in blocks. This interval does not correct for previous research or selection across horizons.')

with st.expander('Out-of-sample stability by time period', expanded=True):
    period_rows = []
    for period in result['evaluation_period_metrics']:
        values = period['models'][result['selected_model']][ticker]
        baseline = period['models']['prior'][ticker]
        period_rows.append({'Period':period['start'] + ' to ' + period['end'],
            'Accuracy':values['accuracy'], 'Balanced accuracy':values['balanced_accuracy'],
            'Log loss':values['log_loss'], 'Baseline log loss':baseline['log_loss'],
            'Brier':values['brier'], 'ECE':values['ece'], 'Rows':values['rows']})
    st.dataframe(pd.DataFrame(period_rows).set_index('Period').style.format({c:'{:.3f}' for c in period_rows[0] if c not in ['Period','Rows']}),width='stretch')
    st.caption('Five-year rolling training windows, separate chronological calibration, purged target windows, and a 20-session embargo after each completed validation/evaluation period. Later fits exclude embargo observations. Stability is measured rather than assumed.')

with st.expander('Calibration, return estimates, and execution study'):
    st.write(f"Observed interval coverage: {metrics['interval_80_coverage']:.1%}; expected-return MAE: {metrics['expected_return_mae']:.2%}; zero-return MAE: {metrics['zero_return_mae']:.2%}.")
    st.write(f"Raw probability Brier score: {metrics['raw_brier']:.4f}; calibrated: {metrics['brier']:.4f}. Lower is better.")
    study = result['execution'].get(ticker)
    if study:
        rows = [{'Method':name, **study[name]} for name in ['strategy','buy_hold']]
        st.dataframe(pd.DataFrame(rows).set_index('Method'), width='stretch')
        st.caption(f"{study['periods']} non-overlapping holding periods; strategy invested {study['invested_fraction']:.1%} of periods. Signals require P(UP) ≥ 60% and expected return above assumed round-trip costs. Entry is next adjusted open, exit after {horizon} sessions, 5 basis points each way for each separate holding window; gaps are cash and earn zero. Drawdown is measured at holding-period endpoints, so intraperiod drawdowns are not shown. Market impact, taxes, financing, and operational slippage are not modeled. These assumptions are not live execution results.")

with st.expander('Data, methods, and forward tracking'):
    st.write(f"Features: {report['feature_counts']['market']} market features, {report['feature_counts']['with_environment']} including the environment.")
    st.write('Context includes backward-looking changes and rolling Z-scores of volatility and Treasury-yield indices, yield-curve changes, credit/commodity/dollar proxies, market correlations, relative strength, and mature historical outcome frequencies.')
    st.caption('Environment observations are delayed by two market sessions and stale values are removed. Vendor snapshots are not original economic-release vintages. No historical news or fundamental-release archive is included.')
    ledger = ROOT / 'data/forecast_ledger.csv'
    if ledger.exists():
        st.caption('An append-only ledger preserves the first forecasts issued for each date, horizon, and model version, including forecasts issued before later model refits. Those future outcomes can support forward evaluation as they mature; historical simulations are kept separate.')
        st.download_button('Download issued forecasts', ledger.read_bytes(), 'forecast_ledger.csv', 'text/csv')
    st.download_button('Download evaluation predictions', predictions_path.read_bytes(), 'research_predictions.csv', 'text/csv')
st.caption('Research and education. Forecasts are uncertain and do not determine suitable investments or portfolio allocations.')
