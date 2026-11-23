"""Refresh and train each supported ETF. Run: python -m src.universe [--download]."""
import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from .features import validate_prices
from .train import train, ROOT

TICKERS = ['SPY', 'QQQ', 'IWM', 'VEU', 'AGG', 'TLT', 'GLD']
CONTEXT_TICKERS = TICKERS + ['UUP']


def download(start, end):
    import yfinance as yf
    raw = yf.download(CONTEXT_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError('Market download failed; existing snapshots were left untouched.')
    snapshots = {}
    for ticker in CONTEXT_TICKERS:
        frame = raw.xs(ticker, level=1, axis=1).dropna(subset=['Close', 'High', 'Low', 'Volume'])
        frame = validate_prices(frame)
        if len(frame) < 800:
            raise ValueError(f'Insufficient market history for {ticker}.')
        snapshots[ticker] = frame
    sessions = snapshots['SPY'].index
    for frame in snapshots.values():
        sessions = sessions.union(frame.index)
    for ticker, frame in snapshots.items():
        expected = sessions[(sessions >= frame.index.min()) & (sessions <= frame.index.max())]
        if len(expected.difference(frame.index)):
            raise ValueError(f'{ticker} is missing market sessions; refusing to change the snapshot.')
    latest = {frame.index.max() for frame in snapshots.values()}
    if len(latest) != 1:
        raise ValueError('Market snapshots end on different sessions. Retry after all markets close.')
    combined = pd.concat([frame.assign(Ticker=ticker) for ticker, frame in snapshots.items()])
    temporary = ROOT / 'data/market_prices.tmp'
    combined.to_csv(temporary, index_label='Date')
    temporary.replace(ROOT / 'data/market_prices.csv')


def train_universe(tickers):
    path = ROOT / 'data/training_results.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    for ticker in tickers:
        print(f'Training {ticker}', flush=True)
        results[ticker] = train(ticker)
        path.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--start', default='2007-01-01')
    parser.add_argument('--end', default=datetime.now(ZoneInfo('America/New_York')).date().isoformat())
    parser.add_argument('--tickers', nargs='+', choices=TICKERS, default=TICKERS)
    args = parser.parse_args()
    if args.download:
        download(args.start, args.end)
    train_universe(args.tickers)
