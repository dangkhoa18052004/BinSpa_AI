import os
import io
import pandas as pd
from PIL import Image

PARQUET_DIR = r"D:\BinSpa_AI\ACNE04\data"
OUTPUT_DIR = r"D:\BinSpa_AI\raw\ACNE04_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

parquet_files = sorted([
    os.path.join(PARQUET_DIR, f)
    for f in os.listdir(PARQUET_DIR)
    if f.endswith(".parquet")
])

print("Parquet files:")
for f in parquet_files:
    print(" -", f)

count = 0

for parquet_file in parquet_files:
    print(f"\nReading: {parquet_file}")

    df = pd.read_parquet(parquet_file)

    print("Columns:", list(df.columns))

    for idx, row in df.iterrows():
        image_data = row["image"]

        # Hugging Face image thường lưu dạng dict:
        # {'bytes': ..., 'path': ...}
        if isinstance(image_data, dict):
            image_bytes = image_data.get("bytes")
            image_path = image_data.get("path")

            if image_bytes is not None:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif image_path:
                img = Image.open(image_path).convert("RGB")
            else:
                print("Không đọc được image ở row:", idx)
                continue
        else:
            print("Kiểu image lạ:", type(image_data))
            continue

        # cố gắng giữ filename gốc nếu dataset có
        filename = None

        for col in ["filename", "file_name", "image_name", "name"]:
            if col in df.columns:
                value = row[col]
                if pd.notna(value):
                    filename = str(value)
                    break

        if filename is None:
            if isinstance(image_data, dict) and image_data.get("path"):
                filename = os.path.basename(image_data["path"])
            else:
                filename = f"image_{count:05d}.jpg"

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            filename += ".jpg"

        output_path = os.path.join(OUTPUT_DIR, filename)
        img.save(output_path, quality=95)

        count += 1

        if count % 100 == 0:
            print(f"Exported {count} images")

print(f"\nDONE - Total exported images: {count}")
print("Saved to:", OUTPUT_DIR)