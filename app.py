"""Beginner-friendly local dashboard. Run: python -m streamlit run app.py."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.context import ASSETS, input_hashes
from src.presentation import TIMEFRAMES, outlook_summary, direction_text, plain_change

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Market Outlook', page_icon='📈', layout='wide')
st.title('Market Outlook')
st.write('Choose a fund and a timeframe. Start with the simple outlook below.')


@st.cache_data
def load_report(path, modified):
    return json.loads(Path(path).read_text())


@st.cache_data
def load_table(path, modified):
    return pd.read_csv(path, parse_dates=['Date'])


report_path = ROOT / 'data/research_results.json'
predictions_path = ROOT / 'data/research_predictions.csv'
if not report_path.exists() or not predictions_path.exists():
    st.error('The forecast data is unavailable. Please try again after it has been updated.')
    st.stop()
report = load_report(str(report_path), report_path.stat().st_mtime_ns)
if report['input_hashes'] != input_hashes():
    st.error('The forecasts are being updated. Please check back when the update is complete.')
    st.stop()

st.subheader('Explore a fund')
fund_control, time_control = st.columns(2)
with fund_control:
    ticker = st.selectbox('Choose a fund', list(ASSETS), format_func=lambda t: f'{t} — {ASSETS[t][0]}',
                          help='An ETF is a fund traded on a stock exchange, often holding a basket of investments.')
horizons = sorted(int(h) for h in report['horizons'])
with time_control:
    horizon = st.selectbox('Look ahead', horizons, index=horizons.index(10) if 10 in horizons else 0,
                           format_func=lambda h: TIMEFRAMES[h],
                           help='Trading days exclude weekends and market holidays.')
mode = st.radio('View', ['Latest outlook', 'Past forecasts'], horizontal=True)
result = report['horizons'][str(horizon)]
metrics = result['metrics'][ticker]
latest = next(row for row in result['latest'] if row['ticker'] == ticker)
forecast = latest
history = None
if mode == 'Past forecasts':
    all_predictions = load_table(str(predictions_path), predictions_path.stat().st_mtime_ns)
    history = all_predictions.loc[(all_predictions.Ticker == ticker) & (all_predictions.horizon == horizon)].sort_values('Date')
    picked = st.date_input('Choose a past date', value=history.Date.max().date(),
        min_value=history.Date.min().date(), max_value=history.Date.max().date())
    stored = history.loc[history.Date <= pd.Timestamp(picked)].iloc[-1]
    forecast = dict(as_of=stored.Date.date().isoformat(), p_up=float(stored.p_up),
        expected_return=float(stored.expected_return), lower_return=float(stored.lower_return),
        upper_return=float(stored.upper_return), median_return=float(stored.median_return))
    st.info('You are exploring a past forecast. The reliability explanation uses the tests completed so far.')

as_of = pd.Timestamp(forecast['as_of']).date()
age = (pd.Timestamp.now(tz='America/New_York').date() - as_of).days
summary = outlook_summary(forecast['p_up'], metrics, result['bootstrap'], stale=mode == 'Latest outlook' and age > 7)
st.subheader(f'{ticker} · {ASSETS[ticker][0]}')
st.caption(f"{TIMEFRAMES[horizon]} · Saved market data: {as_of.strftime('%d %b %Y')}")
message = f"### {summary['label']}\n\n**{summary['meaning']}**\n\n{summary['reason']}"
{'Uncertain':st.warning,'Positive':st.success,'Negative':st.info}[summary['label']](message)
st.caption('Positive describes the model’s price outlook. It does not mean “buy”. The app does not know your savings, debts or goals.')

left,right = st.columns([1.3,1])
with left:
    st.subheader('How the price has moved')
    market_path = ROOT / 'data/market_prices.csv'
    market = load_table(str(market_path), market_path.stat().st_mtime_ns)
    series = market.loc[(market.Ticker == ticker) & (market.Date <= pd.Timestamp(as_of))].tail(180)
    chart = go.Figure(go.Scatter(x=series.Date, y=series.Close, line=dict(color='#254ca6', width=2.5), name='Past price'))
    chart.update_layout(height=300, margin=dict(l=5,r=5,t=10,b=5), yaxis_title='Adjusted price, USD', showlegend=False)
    st.plotly_chart(chart, width='stretch')
    st.caption('This chart shows the past. It does not show what will happen next.')
with right:
    st.subheader('What the forecast suggests')
    with st.container(border=True):
        st.write('**Model direction**')
        st.write(direction_text(forecast['p_up']))
        st.caption('A model estimate. The outlook above also considers its reliability.')
    with st.container(border=True):
        st.write('**Possible price movement**')
        st.write(f"From {plain_change(forecast['lower_return'])} to {plain_change(forecast['upper_return'])}")
        st.caption('The model’s estimated range. Actual prices can move outside it.')

with st.expander('How reliable has it been?', expanded=True):
    a,b = st.columns(2)
    a.metric('Model correct in past tests', f"{metrics['accuracy']*100:.0f} out of 100")
    b.metric('Simple comparison correct', f"{metrics['baseline_accuracy']*100:.0f} out of 100")
    st.write('The simple comparison uses only how often this fund rose in the past.')
    st.write('A higher score is better, but these figures are past results—not your chances of making money.')
    rows = []
    for period in result['evaluation_period_metrics']:
        m = period['models'][result['selected_model']][ticker]
        comparison = period['models']['prior'][ticker]
        rows.append({'Year':period['start'][:4], 'Model correct':f"{m['accuracy']:.0%}",
                     'Simple comparison':f"{comparison['accuracy']:.0%}"})
    st.dataframe(pd.DataFrame(rows).set_index('Year'), width='stretch')
if mode == 'Past forecasts':
    st.info(f"What actually happened: the adjusted price was {plain_change(float(stored.Forward_return))} over {horizon} trading days.")

with st.expander('A few simple terms'):
    st.write('**Positive:** the model leans toward a rise, with supporting evidence from its tests.')
    st.write('**Negative:** the model leans toward a fall, with supporting evidence from its tests.')
    st.write('**Uncertain:** the direction is unclear, the evidence is weak, or the data is old.')
    st.write('**Possible movement:** a range of estimated changes. It can include both a gain and a loss.')

with st.expander('Detailed research results (optional)'):
    st.caption('These details are for inspecting the research. You do not need them to use the outlook.')
    st.write(f"Evaluation: {result['test_start']} to {result['test_end']}; selected model: {result['selected_model']}.")
    st.write(f"Estimated P(UP): {forecast['p_up']:.1%}; expected return: {forecast['expected_return']:+.2%}.")
    st.caption('Probabilities are estimates. Historical labels are retrospective explanations, not original investment signals. The return and direction models are separate.')
    rows = [{'Model':n, 'Validation log loss':v, 'Evaluation log loss':result['candidates'][n]['mean_asset_log_loss']}
            for n,v in result['validation_log_loss'].items()]
    st.dataframe(pd.DataFrame(rows).set_index('Model').style.format('{:.4f}'), width='stretch')
    rows = []
    for period in result['evaluation_period_metrics']:
        m = period['models'][result['selected_model']][ticker]
        rows.append({'Period':period['start']+' to '+period['end'], 'Accuracy':m['accuracy'],
                     'Balanced accuracy':m['balanced_accuracy'], 'Log loss':m['log_loss'], 'Brier':m['brier'], 'ECE':m['ece']})
    st.dataframe(pd.DataFrame(rows).set_index('Period').style.format('{:.4f}'), width='stretch')
    lo,hi = result['bootstrap']['bootstrap_95_interval']
    st.write(f"Exploratory date-block interval for model-minus-prior log loss: [{lo:+.4f}, {hi:+.4f}].")
    st.caption('Outlook labels require a model-favoring interval, lower per-asset log loss, balanced accuracy above 50%, and an estimated UP probability of at least 60% or at most 40%. These are display rules, not an optimized trading strategy. Historical tests reuse previously inspected periods, outcomes overlap, and vendor data is not original-release vintage data.')
    st.write(f"Observed range coverage: {metrics['interval_80_coverage']:.1%}. Nominal 80% coverage is not guaranteed.")
    st.download_button('Download research predictions', predictions_path.read_bytes(), 'research_predictions.csv', 'text/csv')
st.caption('Learning demo · Saved forecasts · No trading or personal financial assessment')
