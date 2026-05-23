# AI-Based Smart Manufacturing Apple Defect Detection

Beginner-friendly computer vision prototype for classifying apples as:

- `fresh`
- `rotten`

The project follows the DPR workflow:

```text
Apple -> Camera/Image -> AI Model -> Prediction -> Dashboard -> Conveyor Simulation
```

This is a prototype. Real industrial sorting needs stronger lighting control, better datasets, physical trigger sensors, calibration, and safety testing.

## Features

- Kaggle apple dataset import helper
- MobileNetV2 transfer-learning classifier
- Train/validation/test split script
- Test accuracy, confusion matrix, and classification report
- Single-image prediction
- Live webcam prediction
- Queue/background prediction mode
- Streamlit dashboard
- Production metrics: total, fresh, rotten, uncertain, defect percentage
- Optional ESP32 servo rejection starter sketch

## Folder Structure

```text
conveyor-vision-sorter/
├── config.py
├── requirements.txt
├── README.md
├── DPR_CONTEXT.md
├── data/
│   ├── raw/fresh/
│   ├── raw/rotten/
│   ├── train/fresh/
│   ├── train/rotten/
│   ├── val/fresh/
│   ├── val/rotten/
│   ├── test/fresh/
│   ├── test/rotten/
│   └── captured/
├── models/
├── logs/
├── src/
│   ├── prepare_kaggle_apples.py
│   ├── split_dataset.py
│   ├── train_model.py
│   ├── predict_image.py
│   ├── live_predict.py
│   ├── queue_predict.py
│   ├── dashboard.py
│   └── utils.py
├── notebooks/
└── hardware/
```

## Recommended Kaggle Dataset

Use this first:

```text
sriramr/fruits-fresh-and-rotten-for-classification
```

It contains fresh and rotten fruit images, including apples. The importer keeps apple images and ignores other fruit.

Other usable datasets:

- `nourabdoun/fruits-quality-fresh-vs-rotten`
- `zlatan599/fruitquality1`
- `muhriddinmuxiddinov/fruits-and-vegetables-dataset`

## Run Steps

Open PowerShell in this folder:

```powershell
cd C:\Users\prani\Documents\Codex\2026-05-22\files-mentioned-by-the-user-agents\conveyor-vision-sorter
```

Create virtual environment:

```powershell
python -m venv .venv
```

Do not activate it if PowerShell blocks scripts. Use `.venv\Scripts\python.exe` directly.

Install libraries:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create Kaggle credential file:

```powershell
mkdir $env:USERPROFILE\.kaggle
notepad $env:USERPROFILE\.kaggle\kaggle.json
```

Paste this into Notepad, using your own Kaggle username and key:

```json
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
```

Save and close Notepad.

Test Kaggle:

```powershell
.venv\Scripts\python.exe -m kaggle datasets list
```

Download apple dataset:

```powershell
.venv\Scripts\python.exe -m kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p downloads\apples --unzip
```

Prepare only apple images:

```powershell
.venv\Scripts\python.exe src\prepare_kaggle_apples.py --source downloads\apples
```

Check images:

```powershell
dir data\raw\fresh
dir data\raw\rotten
```

Split dataset:

```powershell
.venv\Scripts\python.exe src\split_dataset.py
```

Train model:

```powershell
.venv\Scripts\python.exe src\train_model.py
```

Test one image:

```powershell
dir data\test\fresh
.venv\Scripts\python.exe src\predict_image.py --image data\test\fresh\IMAGE_NAME.jpg
```

Run dashboard:

```powershell
.venv\Scripts\python.exe -m streamlit run src\dashboard.py
```

Run live webcam prediction:

```powershell
.venv\Scripts\python.exe src\live_predict.py
```

Press `q` to quit webcam windows.

## Outputs

Training creates:

```text
models/product_classifier.keras
models/labels.json
logs/training_curves.png
logs/classification_report.txt
```

The dashboard shows:

- Total apples inspected
- Fresh apples
- Rotten apples
- Uncertain predictions
- Defect percentage
- Recent prediction table
- Uploaded-image prediction

## Google Colab

If training is slow on your laptop, use:

```text
notebooks/colab_training_steps.md
```

## GitHub

Do not upload datasets, trained models, logs, or `.venv`. They are ignored by `.gitignore`.

Upload only the code project. After creating a GitHub repository, run:

```powershell
git init
git add .
git commit -m "Initial apple defect detection prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

If Git asks for login, use GitHub sign-in or a personal access token.

## Future Improvements

- ESP32 serial trigger
- Servo rejection for rotten apples
- YOLO detection after classifier works
- TensorFlow Lite export
- SQLite prediction database
- Better dashboard with live camera feed
