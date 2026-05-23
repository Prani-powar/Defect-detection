# Google Colab Training Steps

Use this if your laptop is slow. Google Colab gives you a free GPU sometimes.

## 1. Upload Project

Upload this project folder to Google Drive or GitHub.

## 2. Open Colab

Go to:

```text
https://colab.research.google.com
```

Enable GPU:

```text
Runtime -> Change runtime type -> T4 GPU
```

## 3. Install Libraries

```python
!pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn pillow kaggle
```

## 4. Download Dataset

Upload your `kaggle.json` to Colab, then run:

```python
import os
os.makedirs("/root/.kaggle", exist_ok=True)
!cp kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p downloads/apples --unzip
```

## 5. Prepare And Train

Run these from the project folder:

```python
!python src/prepare_kaggle_apples.py --source downloads/apples
!python src/split_dataset.py
!python src/train_model.py
```

## 6. Download Model

After training, download:

```text
models/product_classifier.keras
models/labels.json
logs/classification_report.txt
logs/training_curves.png
```
