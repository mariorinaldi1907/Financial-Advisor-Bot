"""Compact, reproducible Philadelphia Fed vintage snapshots; no latest-revised history."""
import argparse
import hashlib
import io
import re
import ssl
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import pandas as pd
from .data import ROOT

SOURCES = {name: f'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/real-time-data/data-files/xlsx/{code}MvMd.xlsx'
           for name, code in [('inflation', 'pcpi'), ('employment', 'employ')]}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_vintages(content, indicator):
    """Read the source's simple value-only XLSX without adding a web dependency."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        strings = [''.join(n.itertext()) for n in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        rows = ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('.//s:row', NS)
        def cells(row):
            output = {}
            for cell in row:
                node = cell.find('s:v', NS)
                value = node.text if node is not None else ''
                if value and cell.get('t') == 's':
                    value = strings[int(value)]
                output[re.sub(r'\d', '', cell.get('r'))] = value
            return output
        header, observations = cells(rows[0]), {}
        if header.get('A') != 'DATE':
            raise ValueError('Unexpected vintage workbook layout.')
        for row in rows[1:]:
            values = cells(row)
            if re.fullmatch(r'\d{4}:\d{2}', values.get('A', '')):
                observations[pd.Timestamp(values['A'].replace(':', '-') + '-01')] = values
        records = []
        for column, name in header.items():
            match = re.fullmatch(r'(PCPI|EMPLOY)(\d{2})M(\d{1,2})', name)
            if not match:
                continue
            year = int(match[2]) + (1900 if int(match[2]) >= 50 else 2000)
            vintage = pd.Timestamp(year, int(match[3]), 1)
            if vintage < pd.Timestamp('2005-01-01'):
                continue
            series = pd.Series({date: pd.to_numeric(values.get(column, ''), errors='coerce')
                                for date, values in observations.items()}).sort_index()
            series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq='MS'))
            known = series.dropna()
            if known.empty or known.index.max() > vintage:
                raise ValueError('Invalid observation dates in a vintage.')
            date = known.index.max()
            first = series.pct_change(fill_method=None).loc[date]
            annual = series.pct_change(12, fill_method=None).loc[date]
            if not np.isfinite([first, annual]).all():
                continue
            records.append(dict(Date=(vintage + pd.offsets.MonthBegin(1)).date().isoformat(),
                indicator=indicator, vintage_month=vintage.date().isoformat(), observation_date=date.date().isoformat(),
                growth_1m=float(first), growth_12m=float(annual), source_url=SOURCES[indicator],
                source_sha256=hashlib.sha256(content).hexdigest()))
        if len(records) < 200:
            raise ValueError('Insufficient vintage history.')
        return records


def download_vintages():
    import certifi
    records = []
    for indicator, url in SOURCES.items():
        with urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where()), timeout=60) as response:
            records.extend(read_vintages(response.read(), indicator))
    output = pd.DataFrame(records).sort_values(['Date', 'indicator'])
    path = ROOT / 'data/macro_vintages.csv'
    temporary = path.with_suffix('.tmp')
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    return output


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = download_vintages()
    print(f'Saved {len(result)} vintage snapshots; latest availability {result.Date.max()}.')
