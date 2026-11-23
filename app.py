"""Beginner-friendly local dashboard. Run: python -m streamlit run app.py."""
import json
from html import escape
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.context import ASSETS, input_hashes
from src.presentation import TIMEFRAMES, outlook_summary, direction_text, plain_change

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Market Outlook', page_icon='📈', layout='wide')
st.html("""
<style>
:root { color-scheme: light; }
.stApp { background: #f4f7fb; color: #172b43; }
.stApp [data-testid="stHeader"] { background: rgba(244,247,251,.96); }
.stApp [data-testid="stMainBlockContainer"] { max-width: 1200px; padding-top: 4.5rem; padding-bottom: 3rem; }
.stApp h1, .stApp h2, .stApp h3 { color: #142d4e; letter-spacing: -.025em; }
.stApp h1 { font-size: clamp(2rem,5vw,3rem); padding-bottom: .3rem; }
.stApp h3 { font-size: 1.15rem; }
.stApp [data-testid="stVerticalBlock"] { gap: 1rem; }
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stExpander"] summary { color: #172b43; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] [data-testid="stMarkdownContainer"] { color: #52657b; }
.stApp [data-baseweb="select"] > div { background: #fff; border-color: #cbd6e3; border-radius: 12px; color: #172b43; }
.stApp [data-testid="stExpander"] details > summary { background: #fff; color: #172b43; border-radius: 15px; }
.stApp [data-testid="stExpander"] details > summary:hover { background: #edf3fa; color: #172b43; }
.stApp [data-testid="stExpander"] details > summary:focus-visible { outline: 2px solid #2876c7; outline-offset: 2px; }
.stApp [data-testid="stExpander"] details[open] > summary { border-bottom: 1px solid #dde5ef; border-radius: 15px 15px 0 0; }
.stApp [data-testid="stTable"] { overflow-x: auto; }
.stApp [data-testid="stTable"] table { width: 100%; font-size: .9rem; border-collapse: collapse; }
.stApp [data-testid="stTable"] th, .stApp [data-testid="stTable"] td { color: #172b43; background: #fff; border-color: #dde5ef; padding: 10px 12px; }
.stApp [data-testid="stTable"] thead th { background: #edf3fa; color: #254b72; }
.stApp [data-testid="stTable"] tbody tr:nth-child(even) td, .stApp [data-testid="stTable"] tbody tr:nth-child(even) th { background: #f8fafd; }
.stApp input, .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"] { background: #fff; color: #172b43; color-scheme: light; }
.stApp [data-testid="stBaseButton-secondary"] { background: #fff; color: #173f70; border-color: #cbd6e3; }
.stApp [data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color: inherit; }
.stApp [data-testid="stAlertContentInfo"] [data-testid="stMarkdownContainer"] { color: #21476f; }
.stApp [data-testid="stAlertContentError"] [data-testid="stMarkdownContainer"] { color: #882b35; }
.stApp button[aria-label^="Help for"] { color: #52657b; }
.stApp [data-baseweb="select"] svg { color: #52657b; fill: currentColor; }
.stApp label[data-baseweb="radio"] > div:first-child { background: #fff; border: 1px solid #8494a7; }
.stApp label[data-baseweb="radio"] > div:first-child > div { background: transparent; }
.stApp label[data-baseweb="radio"]:has(input:checked) > div:first-child { background: #16776d; border-color: #16776d; }
.stApp label[data-baseweb="radio"]:has(input:checked) > div:first-child > div { background: #fff; }
.stApp label[data-baseweb="radio"]:has(input:focus-visible) > div:first-child { outline: 2px solid #2876c7; outline-offset: 3px; }
.stApp [data-testid="stExpander"] { background: #fff; border: 1px solid #dde5ef; border-radius: 16px; }
.stApp [data-testid="stMetric"] { background: #f4f7fb; border: 1px solid #e4ebf3; border-radius: 14px; padding: 16px; }
.stApp [data-testid="stMetricValue"] { color: #173f70; font-size: 1.9rem; }
.stApp .st-key-fund-controls, .stApp .st-key-price-card, .stApp .st-key-direction-card, .stApp .st-key-range-card, .stApp .st-key-risk-card { background: #fff; border: 1px solid #dde5ef; border-radius: 18px; padding: 22px; box-shadow: 0 4px 18px #17314e04; }
.st-key-direction-card, .st-key-range-card, .st-key-risk-card { min-height: 260px; }
.intro { color: #52657b; font-size: 1.04rem; margin: 0 0 1rem; max-width: 650px; }
.outlook-card { --accent: #936012; --tint: #fff8e9; padding: 24px 28px; background: var(--tint); border: 1px solid #dfd5bc; border-left: 5px solid var(--accent); border-radius: 16px; margin: .2rem 0 .6rem; }
.outlook-card.positive { --accent: #126b56; --tint: #edf9f3; border-color: #b9ddcd; }
.outlook-card.negative { --accent: #a33d4b; --tint: #fff1f3; border-color: #edc7cd; }
.outlook-top { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
.outlook-icon { display: inline-flex; align-items: center; justify-content: center; width: 38px; height: 38px; border-radius: 50%; background: #ffffffb3; font-size: 1.4rem; color: var(--accent); }
.outlook-card h3 { color: var(--accent); font-size: 1.6rem; margin: 0; padding: 0; }
.outlook-card p { color: #34465c; margin: 6px 0 0; line-height: 1.6; }
.takeaway { margin-top: 18px; padding-top: 16px; border-top: 1px solid #172b4326; }
.takeaway strong { display: block; font-size: 1.2rem; color: #172b43; margin-top: 4px; }
.outlook-card .outlook-lead { color: #172b43; font-weight: 650; }
.mini-label { color: #61728a; font-size: .78rem; font-weight: 650; margin: 0 0 5px; }
.direction { font-size: 1.4rem; font-weight: 650; color: #183d65; margin: 0 0 2px; }
.range-value { font-size: 1.1rem; font-weight: 650; color: #183d65; margin: 0; }
.footer { border-top: 1px solid #dde5ef; padding-top: 18px; color: #61728a; font-size: .8rem; margin-top: 16px; }
@media (max-width: 640px) {
 .stApp [data-testid="stMainBlockContainer"] { padding-top: 4.5rem; }
 .stApp .st-key-fund-controls, .stApp .st-key-price-card, .stApp .st-key-direction-card, .stApp .st-key-range-card, .stApp .st-key-risk-card { padding: 16px; }
 .outlook-card { padding: 20px; }
 .st-key-direction-card, .st-key-range-card, .st-key-risk-card { min-height: 0; }
 .stApp [data-testid="stMetricValue"] { font-size: 1.55rem; }
}
</style>
""")
st.title('Market Outlook')
st.html('<p class="intro">Explore a fund. Understand the outlook. See the evidence behind it.</p>')


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

with st.container(key='fund-controls'):
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
    if 'downside_p' in stored:
        forecast.update(downside_p=float(stored.downside_p), downside_threshold=float(stored.downside_threshold))
    st.info('You are exploring a past forecast. The reliability explanation uses the tests completed so far.')

as_of = pd.Timestamp(forecast['as_of']).date()
age = (pd.Timestamp.now(tz='America/New_York').date() - as_of).days
summary = outlook_summary(forecast['p_up'], metrics, result['bootstrap'], stale=mode == 'Latest outlook' and age > 7)
st.subheader(f'{ticker} · {ASSETS[ticker][0]}')
st.caption(f"{TIMEFRAMES[horizon]} · Saved market data: {as_of.strftime('%d %b %Y')}")
symbol = {'Uncertain': '≈', 'Positive': '↗', 'Negative': '↘'}[summary['label']]
st.html(f"""<section class="outlook-card {summary['tone']}" aria-label="Market outlook">
<div class="outlook-top"><span class="outlook-icon" aria-hidden="true">{symbol}</span><h3>{escape(summary['label'])}</h3></div>
<p class="outlook-lead">{escape(summary['meaning'])}</p><p>{escape(summary['reason'])}</p>
<div class="takeaway"><span class="mini-label">{'HOW TO READ THIS PAST RESULT' if mode == 'Past forecasts' else 'WHAT THIS MEANS'}</span>
<strong>{escape(summary['takeaway'])}</strong><p>{escape(summary['next_step'])}</p></div></section>""")
st.caption('A model-based research takeaway, not a personal instruction to buy, hold or sell. The app does not know your savings, debts or goals.')

with st.container(key='price-card'):
    st.subheader('Price history')
    st.caption('Last 180 trading days · adjusted price in USD')
    market_path = ROOT / 'data/market_prices.csv'
    market = load_table(str(market_path), market_path.stat().st_mtime_ns)
    series = market.loc[(market.Ticker == ticker) & (market.Date <= pd.Timestamp(as_of))].tail(180)
    chart = go.Figure(go.Scatter(x=series.Date, y=series.Close,
        line=dict(color='#2876c7', width=2.8), name='Past price',
        hovertemplate='%{x|%d %b %Y}<br>$%{y:,.2f}<extra></extra>'))
    chart.add_trace(go.Scatter(x=series.Date.tail(1), y=series.Close.tail(1),
        mode='markers', marker=dict(size=8,color='#16776d',line=dict(color='white',width=2)),
        hoverinfo='skip', showlegend=False))
    chart.update_layout(height=285, margin=dict(l=0,r=12,t=12,b=0), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#52657b',family='Arial'), hovermode='x unified',
        xaxis=dict(showgrid=False, zeroline=False, tickformat='%b %Y', fixedrange=True),
        yaxis=dict(gridcolor='#edf1f6', zeroline=False, tickprefix='$', fixedrange=True))
    st.plotly_chart(chart, width='stretch', config={'displayModeBar':False}, theme=None)
    st.caption('Past prices help show context. They do not predict what happens next.')
st.subheader('Forecast at a glance')
direction_card, movement_card, risk_card = st.columns(3, gap='medium')
with direction_card, st.container(key='direction-card'):
    st.html(f'<p class="mini-label">MODEL DIRECTION</p><p class="direction">{escape(direction_text(forecast["p_up"]))}</p>')
    st.caption('An estimate of direction; the outlook above also considers reliability.')
with movement_card, st.container(key='range-card'):
    st.html(f'<p class="mini-label">POSSIBLE PRICE MOVEMENT</p><p class="range-value">{escape(plain_change(forecast["lower_return"]))} to {escape(plain_change(forecast["upper_return"]))}</p>')
    lower, upper = forecast['lower_return']*100, forecast['upper_return']*100
    spread = max(upper-lower, 1)
    range_chart = go.Figure(go.Scatter(x=[lower,upper],y=[0,0],mode='lines+markers',
        line=dict(color='#2876c7',width=9), marker=dict(size=11,color='#2876c7'),
        hovertemplate='%{x:+.1f}%<extra>Estimated range</extra>'))
    range_chart.add_vline(x=0,line_color='#8494a7',line_dash='dot')
    range_chart.update_layout(height=85, margin=dict(l=12,r=12,t=10,b=20),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False,
        font=dict(color='#52657b',family='Arial'),
        xaxis=dict(range=[min(lower,0)-spread*.15,max(upper,0)+spread*.15],
                   ticksuffix='%',zeroline=False,showgrid=False,fixedrange=True,nticks=4),
        yaxis=dict(visible=False,range=[-1,1],fixedrange=True))
    st.plotly_chart(range_chart,width='stretch',config={'displayModeBar':False},theme=None)
    st.caption('Dotted line = no price change. Actual prices can move outside the blue range.')
with risk_card, st.container(key='risk-card'):
    if 'downside_p' in forecast:
        st.html(f'<p class="mini-label">LARGER-LOSS ESTIMATE</p><p class="direction">{forecast["downside_p"]:.0%}</p>')
        st.caption(f'Estimated probability of a fall greater than {abs(forecast["downside_threshold"]):.1%} over this timeframe.')
        source = 'Based on historical frequency.' if result.get('downside', {}).get('selected_model') == 'prior' else 'Based on the risk model.'
        st.caption(f'{source} This estimate can be wrong; the loss threshold adjusts to recent volatility.')
    else:
        st.write('Risk estimate unavailable for this forecast.')


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
    st.table(pd.DataFrame(rows).set_index('Year'))
if mode == 'Past forecasts':
    st.info(f"What actually happened: the adjusted price was {plain_change(float(stored.Forward_return))} over {horizon} trading days.")

with st.expander('A few simple terms'):
    st.write('**Positive:** the model leans toward a rise, with supporting evidence from its tests.')
    st.write('**Negative:** the model leans toward a fall, with supporting evidence from its tests.')
    st.write('**Uncertain → Wait / review:** the direction is unclear, the evidence is weak, or the data is old. This is not an automatic hold or sell signal.')
    st.write('**Consider buying / reducing:** research prompts shown only when the outlook meets the evidence checks. They do not assess whether a trade suits you.')
    st.write('**Possible movement:** a range of estimated changes. It can include both a gain and a loss.')

with st.expander('Detailed research results (optional)'):
    st.caption('These details are for inspecting the research. You do not need them to use the outlook.')
    st.write(f"Evaluation: {result['test_start']} to {result['test_end']}; selected model: {result['selected_model']}.")
    st.write(f"Estimated P(UP): {forecast['p_up']:.1%}; expected return: {forecast['expected_return']:+.2%}.")
    st.caption('Probabilities are estimates. Historical labels are retrospective explanations, not original investment signals. The return and direction models are separate.')
    rows = [{'Model':n, 'Validation log loss':v, 'Evaluation log loss':result['candidates'][n]['mean_asset_log_loss']}
            for n,v in result['validation_log_loss'].items()]
    st.table(pd.DataFrame(rows).set_index('Model').style.format('{:.4f}'))
    rows = []
    for period in result['evaluation_period_metrics']:
        m = period['models'][result['selected_model']][ticker]
        rows.append({'Period':period['start']+' to '+period['end'], 'Accuracy':m['accuracy'],
                     'Balanced accuracy':m['balanced_accuracy'], 'Log loss':m['log_loss'], 'Brier':m['brier'], 'ECE':m['ece']})
    st.table(pd.DataFrame(rows).set_index('Period').style.format('{:.4f}'))
    lo,hi = result['bootstrap']['bootstrap_95_interval']
    st.write(f"Exploratory date-block interval for model-minus-prior log loss: [{lo:+.4f}, {hi:+.4f}].")
    st.caption('Outlook labels require a model-favoring interval, lower per-asset log loss, balanced accuracy above 50%, and an estimated UP probability of at least 60% or at most 40%. These are display rules, not an optimized trading strategy. Historical tests reuse previously inspected periods, outcomes overlap, and vendor data is not original-release vintage data.')
    st.write(f"Observed range coverage: {metrics['interval_80_coverage']:.1%}. Nominal 80% coverage is not guaranteed.")
    if 'promotion' in result:
        st.write('**Model selection checks**')
        st.caption('A candidate needs at least two validation-fold wins, average log-loss improvement of 0.0005, and no fold deterioration greater than 0.002. Otherwise the historical baseline is used. Evaluation results do not select the model.')
        decisions = [dict(Model=name, **{f'Fold {i+1} gain':f'{gain:+.4f}' for i,gain in enumerate(decision['fold_gains'])},
                          Eligible='Yes' if decision['eligible'] else 'No') for name,decision in result['promotion'].items()]
        st.table(pd.DataFrame(decisions).set_index('Model'))
    if 'downside' in result:
        risk_metrics = result['downside']['metrics'][ticker]
        risk_baseline = result['downside']['baseline_metrics'][ticker]
        st.write(f"**Larger-loss model:** {result['downside']['selected_model']}. Out-of-sample Brier score {risk_metrics['brier']:.4f}; historical baseline {risk_baseline['brier']:.4f}. Lower is better.")
    forward = report.get('forward_evaluation')
    if forward:
        st.write('**Forward forecasts**')
        st.write(f"Scored: {forward['status_counts'].get('evaluated',0)} · Pending: {forward['status_counts'].get('pending',0)}")
        st.caption(forward['note'])
        if forward['groups']:
            st.json(forward['groups'], expanded=False)

    st.download_button('Download research predictions', predictions_path.read_bytes(), 'research_predictions.csv', 'text/csv')
st.html('<footer class="footer">Market Outlook &nbsp;·&nbsp; Learning demo &nbsp;·&nbsp; Saved forecasts &nbsp;·&nbsp; No personal financial assessment</footer>')
