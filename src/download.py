"""Refresh the shared ETF snapshot without training. Use src.universe to do both."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from .universe import download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2007-01-01')
    parser.add_argument('--end', default=datetime.now(ZoneInfo('America/New_York')).date().isoformat())
    args = parser.parse_args()
    download(args.start, args.end)
