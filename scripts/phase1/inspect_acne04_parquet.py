import argparse
import json
import pyarrow.parquet as pq
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description='Inspect schemas and 3 records without dumping image bytes')
    parser.add_argument('--parquet-dir', type=__import__('pathlib').Path, default=ROOT/'ACNE04/data')
    args = parser.parse_args()
    files = sorted(args.parquet_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(args.parquet_dir)
    for path in files:
        parquet = pq.ParquetFile(path)
        print(path.name, 'rows:', parquet.metadata.num_rows)
        print(parquet.schema_arrow)
        for row in next(parquet.iter_batches(batch_size=3)).to_pylist():
            image = row.get('image')
            if isinstance(image, dict):
                row['image'] = {'path': image.get('path'), 'bytes_length': len(image.get('bytes') or b'')}
            print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
