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



def summarize_forecasts(ledger):
    """Score only genuinely issued, matured records; retain versions separately."""
    import numpy as np
    result = dict(status_counts={str(k):int(v) for k,v in ledger.status.value_counts().items()}, groups=[],
                  note='Forward outcomes only. Pending and ineligible rows are not scored; model versions are separate. Small and overlapping samples are not proof of an investment edge.')
    evaluated = ledger.loc[ledger.status == 'evaluated'].copy()
    if evaluated.empty:
        return result
    evaluated['period'] = pd.to_datetime(evaluated.outcome_date).dt.year.astype(str)
    def probability_scores(y, p):
        y, p = np.asarray(y,dtype=float), np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8)
        return dict(accuracy=float(((p >= .5) == y).mean()),
                    brier=float(np.mean((p-y)**2)), log_loss=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))))
    if 'model_sha256' not in evaluated:
        evaluated['model_sha256'] = 'legacy_unversioned'
    for (horizon, version, period), rows in evaluated.groupby(['horizon','model_sha256','period']):
        y, p = rows.actual_up.astype(float), rows.p_up.astype(float)
        entry = dict(horizon=int(horizon), model_sha256=version, period=period, rows=len(rows),
                     first_outcome=rows.outcome_date.min(), last_outcome=rows.outcome_date.max(),
                     direction=probability_scores(y,p))
        if 'prior_p_up' in rows:
            paired = rows.dropna(subset=['prior_p_up'])
            if len(paired):
                entry['baseline_comparison'] = dict(rows=len(paired),
                    model=probability_scores(paired.actual_up,paired.p_up),
                    baseline=probability_scores(paired.actual_up,paired.prior_p_up))
        if {'lower_return','upper_return'}.issubset(rows):
            paired = rows.dropna(subset=['lower_return','upper_return'])
            if len(paired):
                entry['interval'] = dict(rows=len(paired), coverage=float(((paired.realized_return >= paired.lower_return) & (paired.realized_return <= paired.upper_return)).mean()),
                    mean_width=float((paired.upper_return-paired.lower_return).mean()))
        if {'downside_p','downside_threshold','downside_prior_p'}.issubset(rows):
            paired = rows.dropna(subset=['downside_p','downside_threshold','downside_prior_p'])
            if len(paired):
                target = (paired.realized_return < paired.downside_threshold).astype(int)
                entry['downside'] = dict(rows=len(paired), model=probability_scores(target,paired.downside_p),
                                          baseline=probability_scores(target,paired.downside_prior_p))
        result['groups'].append(entry)
    return result


def update_forward_evaluation():
    import json
    path = ROOT / 'data/forecast_ledger.csv'
    ledger = pd.read_csv(path)
    market = pd.read_csv(MARKET_PATH, parse_dates=['Date'])
    result = resolve_forecasts(ledger, market)
    result.to_csv(path,index=False)
    report_path = ROOT / 'data/research_results.json'
    report = json.loads(report_path.read_text())
    report['forward_evaluation'] = summarize_forecasts(result)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False) + '\n')
    return report['forward_evaluation']


if __name__ == '__main__':
    result = update_forward_evaluation()
    print(result['status_counts'])
    for group in result['groups']:
        print(group)
