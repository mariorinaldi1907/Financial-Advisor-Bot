"""Explicit market-data refresh; training itself never accesses the network."""
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
from .features import validate_prices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2014-01-01')
    parser.add_argument('--end', default=datetime.now(ZoneInfo('America/New_York')).date().isoformat(), help='Exclusive end date; defaults to today in New York to exclude incomplete sessions.')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/spy.csv')
    args = parser.parse_args()
    raw = yf.download('SPY', start=args.start, end=args.end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise SystemExit('Yahoo returned no data. Existing snapshot was left untouched.')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = validate_prices(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    raw.to_csv(temporary, index_label='Date')
    temporary.replace(args.output)
    print(f'Saved {len(raw)} adjusted SPY sessions through {raw.index.max().date()} to {args.output}')
