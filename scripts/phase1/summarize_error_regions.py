"""Count reviewed unmatched detections by region, preserving the unreviewed count."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from common import REPORTS, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=REPORTS/'phase1_predictions/false_positive_region_review.csv')
    a=p.parse_args()
    counts=Counter();allowed={'eyes','lips','mole','hair_beard','other'}
    with a.csv.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        region=row['region'].strip()
        if region and region not in allowed:
            raise ValueError(f'Unknown region {region} in {row["file_name"]}')
        counts[region or 'unreviewed']+=1
    result=dict(total_unmatched_detections=len(rows),region_counts=dict(counts),review_complete=counts['unreviewed']==0,note='Regions describe unmatched predictions, which may include unannotated lesions; inspect notes before calling them false positives.')
    write_json(a.csv.parent/'region_error_summary.json',result)
    print(result)


if __name__=='__main__':
    main()
