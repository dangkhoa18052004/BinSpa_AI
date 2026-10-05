"""Compatibility entrypoint; implementation lives in scripts/phase1."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent / 'scripts' / 'phase1'
    sys.path.insert(0, str(script_dir))
    runpy.run_path(str(script_dir / 'extract_acne04_images.py'), run_name='__main__')