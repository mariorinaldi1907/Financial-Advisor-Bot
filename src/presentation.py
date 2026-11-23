"""Plain-language outlooks shared by the local app and the public snapshot demo."""
import math

TIMEFRAMES = {1: 'Next trading day', 5: 'About 1 week', 10: 'About 2 weeks', 20: 'About 1 month'}


def outlook_summary(p_up, metrics, bootstrap, stale=False):
    """Fail closed on weak evidence; labels describe direction, never suitability."""
    uncertain = dict(label='Uncertain', tone='uncertain',
        meaning='This forecast does not give a clear reason to invest.',
        takeaway='Wait / review',
        next_step='No reliable buying or selling signal. If you already own this fund, review your plan rather than treating uncertainty as a reason to hold or sell.',
        reason='Past tests have not shown a reliable advantage over a simple comparison.')
    if stale:
        return {**uncertain, 'reason': 'The saved data is out of date. A fresh outlook is needed.',
                'next_step': 'Wait for updated data before using this forecast to consider a purchase or sale.'}
    bounds = bootstrap.get('bootstrap_95_interval', [])
    valid = (math.isfinite(float(p_up)) and len(bounds) == 2 and
             math.isfinite(float(bounds[1])) and bounds[1] < 0 and
             metrics.get('balanced_accuracy', 0) > .5 and
             metrics.get('log_loss', float('inf')) < metrics.get('baseline_log_loss', -float('inf')))
    if not valid:
        return uncertain
    if p_up >= .6:
        return dict(label='Positive', tone='positive',
            takeaway='Consider buying',
            next_step='A possible buying opportunity to research. Check the risk, costs and your own investment plan before deciding.',
            meaning='The model leans toward a price rise over this timeframe.',
            reason='A loss is still possible. This outlook does not check whether investing suits your finances.')
    if p_up <= .4:
        return dict(label='Negative', tone='negative',
            takeaway='Consider reducing',
            next_step='If you own this fund, review whether to reduce your exposure. If you do not own it, review the downside before buying.',
            meaning='The model leans toward a price fall over this timeframe.',
            reason='This is an estimate of direction, not a prediction of what will definitely happen.')
    return {**uncertain, 'reason': 'The model does not clearly favor a price rise or a price fall.'}


def direction_text(p_up):
    return 'Leans higher' if p_up >= .6 else 'Leans lower' if p_up <= .4 else 'No clear direction'


def plain_change(value):
    if abs(value) < .00005:
        return 'about unchanged'
    return f"{'up' if value > 0 else 'down'} {abs(value):.1%}"
