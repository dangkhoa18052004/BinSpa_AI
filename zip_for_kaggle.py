from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
source = root / "datasets" / "phase1_acne_detection"
output = root / "phase1_acne_detection.zip"
if not source.is_dir():
    raise FileNotFoundError("Prepare datasets/phase1_acne_detection before exporting the private archive")

with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in source.rglob("*"):
        if file.is_file():
            arcname = file.relative_to(source.parent).as_posix()
            zf.write(file, arcname)

print("Created:", output)
