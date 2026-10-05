"""Idempotent extraction; verify existing dimensions/content, preserve original encoded bytes."""
import argparse
import io
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageChops, ImageStat
from common import ROOT, RAW, AUDIT, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet-dir', type=Path, default=ROOT/'ACNE04/data')
    parser.add_argument('--output', type=Path, default=RAW)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    counters = dict(exported=0, existing=0, duplicated=0, failed=0)
    files = sorted(args.parquet_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(args.parquet_dir)
    seen = set()
    global_row = 0
    for path in files:
        row_id = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                filename = ''
                status, error = '', ''
                try:
                    entry = row['image']
                    filename = row.get('file_name') or row.get('filename') or Path(entry.get('path') or '').name or f'image_{global_row:05d}.jpg'
                    filename = Path(filename).name
                    if Path(filename).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                        filename += '.png'
                    if filename in seen:
                        raise ValueError('Duplicate source filename; refusing overwrite')
                    seen.add(filename)
                    source = Image.open(io.BytesIO(entry['bytes'])) if entry.get('bytes') else Image.open(path.parent / entry['path'])
                    with source:
                        rgb = source.convert('RGB')
                        target = args.output/filename
                        if target.exists():
                            # Old script recompressed JPEG; permit lossy differences, never overwrite it.
                            with Image.open(target) as existing:
                                existing.verify()
                            with Image.open(target) as existing:
                                if existing.size != rgb.size:
                                    raise ValueError('Existing file dimensions differ from source')
                                existing_thumb = existing.convert('RGB').resize((64,64))
                            source_thumb = rgb.resize((64,64))
                            if max(ImageStat.Stat(ImageChops.difference(existing_thumb,source_thumb)).mean) > 3:
                                raise ValueError('Existing image content differs from source; refusing overwrite')
                            status = 'existing'
                        else:
                            # Preserve source bytes when source format matches extension.
                            matching = (source.format == 'JPEG' and target.suffix.lower() in {'.jpg', '.jpeg'}) or (source.format == 'PNG' and target.suffix.lower() == '.png')
                            if entry.get('bytes') and matching:
                                with target.open('xb') as f:
                                    f.write(entry['bytes'])
                            else:
                                with target.open('xb') as f:
                                    rgb.save(f, format='PNG' if target.suffix.lower()=='.png' else 'JPEG', quality=95)
                            status = 'exported'
                    counters[status] += 1
                except Exception as exc:
                    status = 'duplicated' if 'Duplicate source filename' in str(exc) else 'failed'
                    counters[status] += 1
                    error = str(exc)
                rows.append(dict(shard=path.name, row_id=row_id, global_row=global_row, file_name=filename, status=status, error=error))
                row_id += 1
                global_row += 1
        print(path.name, counters, flush=True)
    write_csv(AUDIT/'extraction_mapping.csv', rows, ['shard','row_id','global_row','file_name','status','error'])
    write_json(AUDIT/'extraction_summary.json', counters)
    print(counters, 'sample:', sorted(seen)[:10])
    if counters['failed'] or counters['duplicated']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
