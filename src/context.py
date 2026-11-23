"""Shared market panel and conservatively delayed environment observations."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .features import CORE_FEATURES, EXTRA_FEATURES, build_features, build_market_context
from .data import ROOT, MARKET_PATH

MACRO_PATH = ROOT / 'data/macro_vintages.csv'
ENVIRONMENT_PATH = ROOT / 'data/environment.csv'
HORIZONS = [1, 5, 10, 20]
ASSETS = {
    'SPY': ('US large-cap stocks', 'equity'), 'QQQ': ('Nasdaq 100', 'equity'),
    'IWM': ('US small-cap stocks', 'equity'), 'VEU': ('International stocks', 'international'),
    'AGG': ('US aggregate bonds', 'bonds'), 'TLT': ('Long US Treasuries', 'bonds'),
    'GLD': ('Gold', 'commodity'), 'XLF': ('Financials', 'equity'),
    'XLK': ('Technology', 'equity'), 'XLE': ('Energy', 'equity'),
    'XLV': ('Healthcare', 'equity'), 'XLI': ('Industrials', 'equity'),
    'XLU': ('Utilities', 'equity'), 'XLP': ('Consumer staples', 'equity'),
    'XLY': ('Consumer discretionary', 'equity'), 'XLB': ('Materials', 'equity'),
    'EEM': ('Emerging markets', 'international'), 'EFA': ('Developed international markets', 'international'),
    'HYG': ('High-yield corporate bonds', 'credit'), 'LQD': ('Investment-grade corporate bonds', 'credit'),
    'IEF': ('Intermediate US Treasuries', 'bonds'), 'DBC': ('Broad commodities', 'commodity'),
}
CONTEXT_ASSETS = ['SPY', 'QQQ', 'IWM', 'VEU', 'AGG', 'TLT', 'GLD', 'UUP', 'HYG', 'LQD', 'DBC']
ENV_SYMBOLS = {'^VIX': 'VIX', '^IRX': 'TBill13WeekIndex', '^FVX': 'Treasury5YIndex',
               '^TNX': 'Treasury10YIndex', '^TYX': 'Treasury30YIndex'}


def environment_features(environment, sessions, lag=2):
    """Past observations only; expose them two completed market sessions later."""
    environment = environment.copy().sort_index()
    if environment.index.has_duplicates:
        raise ValueError('Environment observations must have unique dates.')
    result = pd.DataFrame(index=sessions)
    for name in environment:
        known = environment[name].dropna()
        if known.empty:
            raise ValueError(f'No environment observations for {name}.')
        aligned = known.reindex(sessions, method='ffill')
        observed = pd.Series(known.index, index=known.index).reindex(sessions, method='ffill')
        # Both the observation and its age are delayed by actual market sessions.
        available = aligned.shift(lag)
        source_date = observed.shift(lag)
        age = (pd.Series(sessions, index=sessions) - source_date).dt.days
        available = available.where(age <= 10)
        result[f'Env_{name}'] = available
        result[f'Env_{name}_change20'] = available - available.shift(20)
        result[f'Env_{name}_age'] = age
    result['Env_yield_curve'] = result.Env_Treasury10YIndex - result.Env_TBill13WeekIndex
    result['Env_curve_30_5'] = result.Env_Treasury30YIndex - result.Env_Treasury5YIndex
    result['Env_vix_change5'] = result.Env_VIX - result.Env_VIX.shift(5)
    # Level features use only prior observations to define their rolling reference.
    levels = [f'Env_{name}' for name in environment] + ['Env_yield_curve', 'Env_curve_30_5']
    for column in levels:
        past = result[column].shift(1).rolling(252, min_periods=63)
        result[column + '_z252'] = ((result[column] - past.mean()) / past.std().replace(0, np.nan)).clip(-8, 8)
    return result.drop(columns=levels)


def macro_features(vintages, sessions):
    """Use monthly historical vintages only after their entire vintage month ends."""
    vintages = vintages.copy()
    for column in ['Date', 'vintage_month', 'observation_date']:
        vintages[column] = pd.to_datetime(vintages[column], errors='raise')
    if vintages.duplicated(['Date', 'indicator']).any():
        raise ValueError('Duplicate macro vintage availability dates.')
    expected = vintages.vintage_month + pd.offsets.MonthBegin(1)
    if not (vintages.Date == expected).all() or (vintages.observation_date > vintages.vintage_month).any():
        raise ValueError('Macro availability precedes its conservative vintage cutoff.')
    result = pd.DataFrame(index=sessions)
    for name in ['inflation', 'employment']:
        rows = vintages.loc[vintages.indicator == name].sort_values('Date').set_index('Date')
        if rows.empty:
            raise ValueError(f'Missing macro indicator: {name}')
        aligned = rows.reindex(sessions, method='ffill')
        availability = pd.Series(rows.index, index=rows.index).reindex(sessions, method='ffill')
        age = (pd.Series(sessions, index=sessions) - availability).dt.days
        for column in ['growth_1m', 'growth_12m']:
            result[f'Macro_{name}_{column}'] = aligned[column].where(age <= 62)
        result[f'Macro_{name}_age'] = age.clip(upper=63)
    return result


def market_conditions(markets):
    """Credit appetite proxies and breadth of this fixed, disclosed sector universe."""
    result = pd.DataFrame(index=markets.index)
    ratio = markets.HYG / markets.LQD
    result['Credit_relative_return20'] = ratio.pct_change(20, fill_method=None)
    result['Credit_relative_return60'] = ratio.pct_change(60, fill_method=None)
    past = ratio.shift(1).rolling(252, min_periods=63)
    result['Credit_relative_z252'] = (ratio - past.mean()) / past.std().replace(0, np.nan)
    sectors = markets[['XLF','XLK','XLE','XLV','XLI','XLU','XLP','XLY','XLB']]
    available = sectors.notna() & sectors.shift(20).notna()
    result['Breadth_positive20'] = (sectors.pct_change(20, fill_method=None) > 0).where(available).mean(axis=1)
    average = sectors.rolling(50).mean()
    result['Breadth_above_average50'] = (sectors > average).where(average.notna()).mean(axis=1)
    result['Breadth_dispersion20'] = sectors.pct_change(20, fill_method=None).std(axis=1)
    return result


def build_panel(market_path=MARKET_PATH, environment_path=ENVIRONMENT_PATH, macro_path=MACRO_PATH):
    table = pd.read_csv(market_path, parse_dates=['Date'], float_precision='round_trip')
    if table.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Duplicate asset sessions in market snapshot.')
    markets = table.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    sessions = markets.index
    environment = pd.read_csv(environment_path, index_col=0, parse_dates=True, float_precision='round_trip')
    env = environment_features(environment, sessions)
    macro = macro_features(pd.read_csv(macro_path, float_precision='round_trip'), sessions)
    conditions = market_conditions(markets)
    context = build_market_context(markets[CONTEXT_ASSETS])
    spy_return = markets.SPY.pct_change(fill_method=None)
    tlt_return = markets.TLT.pct_change(fill_method=None)
    frames = []
    for ticker, (name, group) in ASSETS.items():
        raw = table.loc[table.Ticker == ticker].drop(columns='Ticker').set_index('Date').sort_index()
        if raw.empty:
            raise ValueError(f'Missing required ETF: {ticker}')
        expected = sessions[(sessions >= raw.index.min()) & (sessions <= raw.index.max())]
        if not expected.equals(raw.index):
            raise ValueError(f'{ticker} has missing market sessions.')
        frame = build_features(raw).join(context).join(env).join(macro).join(conditions)
        close, returns = raw.Close, raw.Close.pct_change(fill_method=None)
        frame['Corr_SPY60'] = returns.rolling(60).corr(spy_return)
        frame['Corr_TLT60'] = returns.rolling(60).corr(tlt_return)
        frame['Beta_SPY60'] = returns.rolling(60).cov(spy_return) / spy_return.rolling(60).var().replace(0, np.nan)
        if ticker == 'SPY':
            frame['Corr_SPY60'], frame['Beta_SPY60'] = 1., 1.
        if ticker == 'TLT':
            frame['Corr_TLT60'] = 1.
        frame['Vol_regime'] = frame.Volatility_20 / returns.rolling(252).std().replace(0, np.nan)
        frame['Relative_momentum20'] = frame.Mom_20 - markets.SPY.pct_change(20, fill_method=None)
        frame['Intraday_return'] = close / raw.Open - 1
        frame['Opening_gap'] = raw.Open / close.shift() - 1
        frame['Range_pct'] = (raw.High - raw.Low) / close
        illiquidity = returns.abs() / (close * raw.Volume).replace(0, np.nan)
        frame['Liquidity_proxy'] = illiquidity.rolling(20).mean() / illiquidity.shift(1).rolling(252).mean().replace(0, np.nan) - 1
        frame['Volatility50_relative'] = frame.Volatility_50 / returns.rolling(252).std().replace(0, np.nan)
        frame['Volatility_log_change20'] = np.log(frame.Volatility_20 / frame.Volatility_20.shift(20).replace(0, np.nan))
        frame['Weekday_sin'] = np.sin(2 * np.pi * frame.index.dayofweek / 5)
        frame['Weekday_cos'] = np.cos(2 * np.pi * frame.index.dayofweek / 5)
        frame['Month_sin'] = np.sin(2 * np.pi * frame.index.month / 12)
        frame['Month_cos'] = np.cos(2 * np.pi * frame.index.month / 12)
        frame['Ticker'], frame['Group'] = ticker, group
        for symbol in ASSETS:
            frame[f'Asset_{symbol}'] = float(symbol == ticker)
        for category in sorted({g for _, g in ASSETS.values()}):
            frame[f'Class_{category}'] = float(category == group)
        for horizon in HORIZONS:
            realized = close.pct_change(horizon, fill_method=None)
            frame[f'Past_up_rate_{horizon}'] = (realized > 0).astype(float).where(realized.notna()).rolling(126, min_periods=63).mean()
            frame[f'Forward_return_{horizon}'] = close.shift(-horizon) / close - 1
            frame[f'Label_end_{horizon}'] = pd.Series(frame.index, index=frame.index).shift(-horizon)
            frame[f'Scale_{horizon}'] = (frame.Volatility_20 * np.sqrt(horizon)).clip(lower=.0001)
            # Execution study: signal after close, entry next open, exit h sessions later.
            frame[f'Entry_open_{horizon}'] = raw.Open.shift(-1)
            frame[f'Exit_open_{horizon}'] = raw.Open.shift(-(horizon + 1))
        frames.append(frame)
    panel = pd.concat(frames).sort_index(kind='stable').replace([np.inf, -np.inf], np.nan)
    panel['Cross_rank_momentum20'] = panel.groupby(level=0).Mom_20.rank(pct=True)
    market_extra = ['Corr_SPY60', 'Corr_TLT60', 'Beta_SPY60', 'Vol_regime', 'Relative_momentum20',
                    'Intraday_return', 'Opening_gap', 'Range_pct', 'Liquidity_proxy',
                    'Weekday_sin', 'Weekday_cos', 'Month_sin', 'Month_cos', 'Cross_rank_momentum20',
                    'Volatility50_relative', 'Volatility_log_change20']
    feature_cols = [c for c in CORE_FEATURES if c not in ['Volatility_20', 'Volatility_50']] + EXTRA_FEATURES + list(context) + market_extra
    feature_cols += list(conditions)
    feature_cols += [c for c in panel if c.startswith(('Asset_', 'Class_', 'Past_up_rate_'))]
    # A full trailing year is required; intermediate missing inputs are handled within each fit.
    panel = panel.loc[panel.Vol_regime.notna() & panel.Mom_60.notna()]
    all_features = feature_cols + list(env) + list(macro)
    # Prevent tree splits from learning floating-point cancellation artifacts.
    panel[all_features] = panel[all_features].round(8)
    return panel, feature_cols, all_features


def input_hashes(market_path=MARKET_PATH, environment_path=ENVIRONMENT_PATH, macro_path=MACRO_PATH):
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name,path in [('market', market_path), ('environment', environment_path), ('macro', macro_path)]}
