"""Validate the saved demo and serve it without training or downloading data."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def get_port(environ=None):
    value = (os.environ if environ is None else environ).get('PORT', '8501')
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError('PORT must be an integer.') from None
    if not 1024 <= port <= 65535:
        raise ValueError('PORT must be between 1024 and 65535.')
    return port


def validate_snapshot(root=ROOT):
    root = Path(root)
    data = root / 'data'
    report = json.loads((data / 'research_results.json').read_text())
    for key,name in [('market','market_prices.csv'),('environment','environment.csv'),('macro','macro_vintages.csv')]:
        digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
        if report['input_hashes'].get(key) != digest:
            raise ValueError(f'The {key} snapshot does not match the saved forecasts.')
    with (data / 'research_predictions.csv').open(newline='') as source:
        columns = csv.reader(source).__next__()
    required = {'Date','Ticker','horizon','p_up','Forward_return'}
    if not required.issubset(columns):
        raise ValueError('The saved prediction table is incomplete.')
    expected = set(map(str,report['config']['horizons']))
    if expected != set(report['horizons']):
        raise ValueError('Finish all configured horizons before deploying.')
    for horizon,result in report['horizons'].items():
        if {row['ticker'] for row in result['latest']} != set(report['assets']):
            raise ValueError(f'Horizon {horizon} is missing asset forecasts.')
        for row in result['latest']:
            if not math.isfinite(row['p_up']) or not 0 <= row['p_up'] <= 1:
                raise ValueError('A forecast has an invalid probability.')
            if not all(math.isfinite(row[c]) for c in ['lower_return','upper_return','expected_return']):
                raise ValueError('A forecast has an invalid return estimate.')
            for name in ['prior_p_up','downside_p','downside_prior_p']:
                if not math.isfinite(row[name]) or not 0 <= row[name] <= 1:
                    raise ValueError('A baseline or downside probability is invalid.')
            if not math.isfinite(row['downside_threshold']) or row['downside_threshold'] >= 0:
                raise ValueError('A downside threshold must be a finite negative return.')
            if row['lower_return'] > row['upper_return']:
                raise ValueError('A forecast return range is reversed.')
    return report


def healthcheck(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=5) as response:
        if response.status != 200 or response.read().strip() != b'ok':
            raise ValueError('The web server is not healthy.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate snapshot files and exit.')
    parser.add_argument('--healthcheck', action='store_true', help='Check the running server and snapshot.')
    args = parser.parse_args()
    validate_snapshot()
    if args.check:
        print('Saved demo data is consistent and ready.')
    elif args.healthcheck:
        healthcheck(get_port())
    else:
        command = [sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
                   '--server.address=0.0.0.0', f'--server.port={get_port()}',
                   '--server.headless=true', '--server.fileWatcherType=none',
                   '--browser.gatherUsageStats=false', '--server.enableCORS=true',
                   '--server.enableXsrfProtection=true']
        os.execv(sys.executable, command)
