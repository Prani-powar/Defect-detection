# Train Apple AI On Kaggle Free GPU

Use this when your laptop is slow. Kaggle runs the training in the cloud and gives you downloadable model files at the end.

## Steps

1. Open Kaggle and sign in.
2. Click **Create** -> **New Notebook**.
3. In the right panel, click **Settings**.
4. Set **Accelerator** to **GPU T4 x2** or **GPU P100**. If both are unavailable, use **GPU T4**.
5. Click **Add Input** and add this dataset:
   `sriramr/fruits-fresh-and-rotten-for-classification`
6. Copy the full code below into the first Kaggle notebook cell.
7. Click **Run All**.
8. When it finishes, open the Kaggle output files and download:
   - `product_classifier.keras`
   - `labels.json`
9. Put those files into your laptop folder:
   - `C:\apple-ai\models\product_classifier.keras`
   - `C:\apple-ai\models\labels.json`
10. Restart your Streamlit app.

## Kaggle Notebook Cell

```python
import os
import shutil
import subprocess
from pathlib import Path

repo_url = "https://github.com/Prani-powar/Defect-detection.git"
project_dir = Path("/kaggle/working/apple-ai")

if project_dir.exists():
    shutil.rmtree(project_dir)

subprocess.run(["git", "clone", repo_url, str(project_dir)], check=True)
os.chdir(project_dir)

subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

dataset_root = Path("/kaggle/input/fruits-fresh-and-rotten-for-classification")
raw_root = project_dir / "data" / "raw"
fresh_dir = raw_root / "fresh"
rotten_dir = raw_root / "rotten"
not_fruit_dir = raw_root / "not_fruit"
for folder in [fresh_dir, rotten_dir, not_fruit_dir]:
    folder.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def copy_matching(source_word, target_dir):
    copied = 0
    for path in dataset_root.rglob("*"):
        if path.suffix.lower() not in image_exts:
            continue
        parts = [part.lower() for part in path.parts]
        filename = path.name.lower()
        if "apple" not in filename and not any("apple" in part for part in parts):
            continue
        if source_word not in filename and not any(source_word in part for part in parts):
            continue
        target = target_dir / f"{copied:06d}_{path.name}"
        shutil.copy2(path, target)
        copied += 1
    return copied

fresh_count = copy_matching("fresh", fresh_dir)
rotten_count = copy_matching("rotten", rotten_dir)
print("Fresh apple images:", fresh_count)
print("Rotten apple images:", rotten_count)

subprocess.run(["python", "src/create_starter_not_fruit.py"], check=True)
subprocess.run(["python", "src/split_dataset.py"], check=True)
subprocess.run(["python", "src/train_model.py"], check=True)

output_dir = Path("/kaggle/working/trained_model")
output_dir.mkdir(exist_ok=True)
shutil.copy2(project_dir / "models" / "product_classifier.keras", output_dir / "product_classifier.keras")
shutil.copy2(project_dir / "models" / "labels.json", output_dir / "labels.json")
print("DONE. Download files from:", output_dir)
```
