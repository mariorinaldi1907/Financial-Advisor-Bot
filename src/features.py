"""Causal end-of-day features. No backward filling or future prices in inputs."""
import numpy as np
import pandas as pd

HORIZON = 10
LEGACY_FEATURES = [
    'SMA_10', 'SMA_30', 'SMA_50', 'SMA_100', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_signal', 'RSI_14', 'Volatility_20', 'Volatility_50',
    'Mom_5', 'Mom_10', 'Mom_20', 'VROC_5', 'Volume',
]
CORE_FEATURES = [
    'SMA_gap_10', 'SMA_gap_30', 'SMA_gap_50', 'SMA_gap_100',
    'EMA_gap_12', 'EMA_gap_26', 'MACD_pct', 'MACD_signal_pct',
    'RSI_14', 'Volatility_20', 'Volatility_50', 'Mom_5', 'Mom_10',
    'Mom_20', 'VROC_5', 'Relative_volume',
]
EXTRA_FEATURES = ['Return_1', 'Mom_60', 'BB_zscore', 'Drawdown_60',
                  'Volatility_ratio', 'ATR_pct', 'Stochastic_14']


def validate_prices(raw):
    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.has_duplicates or df.index.isna().any():
        raise ValueError('Dates must be unique and nonmissing.')
    df = df.sort_index()
    required = ['Close', 'Volume', 'High', 'Low']
    if not set(required).issubset(df):
        raise ValueError(f'Required market columns: {required}')
    df[required] = df[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError('Market data contains missing or infinite values.')
    if (df[['Close', 'High', 'Low']] <= 0).any().any() or (df.Volume < 0).any():
        raise ValueError('Prices must be positive and volumes nonnegative.')
    if (df.High < df.Low).any():
        raise ValueError('High cannot be below Low.')
    return df


def build_features(raw):
    df = validate_prices(raw)
    close = df.Close
    returns = close.pct_change(fill_method=None)
    for n in [10, 30, 50, 100]:
        df[f'SMA_{n}'] = close.rolling(n).mean()
        df[f'SMA_gap_{n}'] = close / df[f'SMA_{n}'] - 1
    for n in [12, 26]:
        df[f'EMA_{n}'] = close.ewm(span=n, adjust=False).mean()
        df[f'EMA_gap_{n}'] = close / df[f'EMA_{n}'] - 1
    df['MACD'] = df.EMA_12 - df.EMA_26
    df['MACD_signal'] = df.MACD.ewm(span=9, adjust=False).mean()
    df['MACD_pct'] = df.MACD / close
    df['MACD_signal_pct'] = df.MACD_signal / close
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    df['RSI_14'] = rsi.mask((loss == 0) & (gain == 0), 50)
    for n in [20, 50]:
        df[f'Volatility_{n}'] = returns.rolling(n).std()
    for n in [5, 10, 20, 60]:
        df[f'Mom_{n}'] = close.pct_change(n, fill_method=None)
    df['VROC_5'] = df.Volume / df.Volume.shift(5).replace(0, np.nan) - 1
    df['Relative_volume'] = df.Volume / df.Volume.rolling(20).mean().replace(0, np.nan)
    df['Return_1'] = returns
    df['BB_zscore'] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
    df['Drawdown_60'] = close / close.rolling(60).max() - 1
    df['Volatility_ratio'] = df.Volatility_20 / df.Volatility_50.replace(0, np.nan)
    tr = pd.concat([df.High - df.Low, (df.High - close.shift()).abs(),
                    (df.Low - close.shift()).abs()], axis=1).max(axis=1)
    df['ATR_pct'] = tr.rolling(14).mean() / close
    low, high = df.Low.rolling(14).min(), df.High.rolling(14).max()
    df['Stochastic_14'] = 100 * (close - low) / (high - low).replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan)


def add_target(features, horizon=HORIZON):
    df = features.copy()
    df['Future_Close'] = df.Close.shift(-horizon)
    df['Label_end'] = pd.Series(df.index, index=df.index).shift(-horizon)
    df['Target'] = (df.Future_Close > df.Close).astype(float).where(df.Future_Close.notna())
    return df
