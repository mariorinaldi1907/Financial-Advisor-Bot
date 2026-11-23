"""One market snapshot shared by training, inference, and research."""
import hashlib
import csv
import io
from pathlib import Path
import pandas as pd
from .features import build_features, build_market_context

ROOT = Path(__file__).resolve().parents[1]
MARKET_PATH = ROOT / 'data/market_prices.csv'


def load_market(ticker, path=MARKET_PATH):
    table = pd.read_csv(path, parse_dates=['Date'])
    if table.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Market snapshot contains duplicate asset sessions.')
    raw = table.loc[table.Ticker == ticker].drop(columns='Ticker').set_index('Date').sort_index()
    if raw.empty:
        raise ValueError(f'No market observations for {ticker}.')
    markets = table.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    return raw, markets


def raw_digest(ticker, path=MARKET_PATH):
    # Hash exact source records, not platform-dependent floating-point formatting.
    with Path(path).open(newline='') as source:
        rows = [row for row in csv.DictReader(source) if row['Ticker'] == ticker]
    if not rows:
        raise ValueError(f'No market observations for {ticker}.')
    columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows([[row[c] for c in columns] for row in sorted(rows, key=lambda r: r['Date'])])
    return hashlib.sha256(output.getvalue().encode()).hexdigest()


def snapshot_digest(path=MARKET_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_frame(ticker, path=MARKET_PATH):
    raw, markets = load_market(ticker, path)
    return build_features(raw).join(build_market_context(markets, exclude=ticker), how='left')
