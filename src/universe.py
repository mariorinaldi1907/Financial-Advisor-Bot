"""Refresh the shared snapshots and train the pooled multi-horizon models."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from .context import ASSETS, ENV_SYMBOLS, HORIZONS
from .data import ROOT
from .features import validate_prices

TICKERS = list(ASSETS)
CONTEXT_TICKERS = TICKERS + ['UUP']


def download(start, end):
    import yfinance as yf
    raw = yf.download(CONTEXT_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError('Market download failed; existing snapshots were left untouched.')
    snapshots = {}
    for ticker in CONTEXT_TICKERS:
        frame = validate_prices(raw.xs(ticker, level=1, axis=1).dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume']))
        if len(frame) < 1000:
            raise ValueError(f'Insufficient history for {ticker}.')
        snapshots[ticker] = frame
    sessions = snapshots['SPY'].index
    for frame in snapshots.values():
        sessions = sessions.union(frame.index)
    for ticker, frame in snapshots.items():
        expected = sessions[(sessions >= frame.index.min()) & (sessions <= frame.index.max())]
        if len(expected.difference(frame.index)):
            raise ValueError(f'{ticker} is missing market sessions; existing data was not changed.')
    if len({frame.index.max() for frame in snapshots.values()}) != 1:
        raise ValueError('ETF snapshots end on different sessions; retry after all markets close.')
    environment = yf.download(list(ENV_SYMBOLS), start=start, end=end, auto_adjust=False, progress=False)
    environment = environment['Close'].rename(columns=ENV_SYMBOLS)
    if environment.notna().sum().min() < 1000 or len(environment.columns) != len(ENV_SYMBOLS):
        raise ValueError('Incomplete environment download; existing data was not changed.')
    combined = pd.concat([frame.assign(Ticker=ticker) for ticker, frame in snapshots.items()])
    market_tmp, environment_tmp = ROOT / 'data/market_prices.tmp', ROOT / 'data/environment.tmp'
    combined.to_csv(market_tmp, index_label='Date')
    environment.to_csv(environment_tmp, index_label='Date')
    market_tmp.replace(ROOT / 'data/market_prices.csv')
    environment_tmp.replace(ROOT / 'data/environment.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--start', default='2007-01-01')
    parser.add_argument('--end', default=datetime.now(ZoneInfo('America/New_York')).date().isoformat())
    parser.add_argument('--horizons', nargs='+', type=int, choices=HORIZONS, default=HORIZONS)
    args = parser.parse_args()
    if args.download:
        download(args.start, args.end)
    from .research import run
    run(args.horizons)
