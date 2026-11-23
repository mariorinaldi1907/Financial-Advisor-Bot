"""Resolve issued forecasts only when later market outcomes have actually arrived."""
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
from .data import ROOT, MARKET_PATH


def resolve_forecasts(ledger, market):
    resolved = ledger.copy()
    for column in ['status', 'outcome_date', 'realized_return', 'actual_up', 'correct']:
        if column not in resolved:
            resolved[column] = None
    for index, forecast in resolved.iterrows():
        prices = market.loc[market.Ticker == forecast.ticker].set_index('Date').sort_index()
        as_of = pd.Timestamp(forecast.as_of)
        issued = pd.Timestamp(forecast.issued_utc)
        if issued.tzinfo is None or as_of not in prices.index:
            resolved.loc[index, 'status'] = 'ineligible_missing_timestamp_or_session'
            continue
        position = prices.index.get_loc(as_of)
        if position + 1 >= len(prices):
            resolved.loc[index, 'status'] = 'pending'
            continue
        next_session = prices.index[position+1].date()
        last_close = pd.Timestamp(datetime.combine(as_of.date(), time(16), ZoneInfo('America/New_York')))
        next_open = pd.Timestamp(datetime.combine(next_session, time(9,30), ZoneInfo('America/New_York')))
        # Conservatively use the normal close even on early-close dates.
        if not last_close <= issued < next_open:
            resolved.loc[index, 'status'] = 'ineligible_issue_time'
            continue
        target = position + int(forecast.horizon)
        if target >= len(prices):
            resolved.loc[index, 'status'] = 'pending'
            continue
        realized = float(prices.Close.iloc[target] / prices.Close.iloc[position] - 1)
        actual = int(realized > 0)
        resolved.loc[index, ['status','outcome_date','realized_return','actual_up','correct']] = [
            'evaluated', str(prices.index[target].date()), realized, actual, int((forecast.p_up >= .5) == actual)]
    return resolved


if __name__ == '__main__':
    path = ROOT / 'data/forecast_ledger.csv'
    if not path.exists():
        raise SystemExit('No issued forecasts. Train the models first.')
    ledger = pd.read_csv(path)
    market = pd.read_csv(MARKET_PATH, parse_dates=['Date'])
    result = resolve_forecasts(ledger, market)
    result.to_csv(path, index=False)
    print(result.status.value_counts().to_string())
