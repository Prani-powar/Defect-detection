# Apple AI Dashboard - Run In VS Code

Everything for this project is inside this folder:

```text
C:\apple-ai
```

Important files:

```text
C:\apple-ai\src\dashboard.py              Streamlit dashboard
C:\apple-ai\src\train_model.py            Model training script
C:\apple-ai\src\split_dataset.py          Dataset split script
C:\apple-ai\config.py                     Project settings/classes
C:\apple-ai\models\product_classifier.keras  Trained model
C:\apple-ai\models\labels.json            Labels: fresh, rotten, not_fruit
C:\apple-ai\data\raw                      Training images
C:\apple-ai\logs                          Prediction and training logs
C:\apple-ai\run_dashboard.ps1             Easy dashboard runner
```

## Fastest Way

1. Open **VS Code**.
2. Click **File > Open Folder**.
3. Select:

```text
C:\apple-ai
```

4. Open the VS Code terminal:

```text
Terminal > New Terminal
```

5. Run:

```powershell
.\run_dashboard.ps1
```

6. Open:

```text
http://localhost:8501
```

## If PowerShell Blocks The Script

Run this in the same VS Code terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_dashboard.ps1
```

This only unlocks scripts for the current terminal window.

## Manual Commands

Use these if you do not want the runner script.

```powershell
cd C:\apple-ai
$env:PYTHONPATH="C:\apple-ai\.train_packages"
C:\Users\prani\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -B -m streamlit run src\dashboard.py --global.developmentMode=false --server.port 8501
```

## Normal Fresh Setup Commands

Use these only if you install Python 3.11 and want a clean virtual environment.

```powershell
cd C:\apple-ai
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src\dashboard.py --global.developmentMode=false --server.port 8501
```

## Train Again Later

Run these only when you have added more feedback images.

```powershell
cd C:\apple-ai
$env:PYTHONPATH="C:\apple-ai\.train_packages"
C:\Users\prani\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -B src\split_dataset.py
C:\Users\prani\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -B src\train_model.py
```

## Stop The Dashboard

In the terminal where Streamlit is running, press:

```text
Ctrl + C
```

If it is running in the background, find the process:

```powershell
netstat -ano | Select-String ":8501"
```

Then stop it:

```powershell
Stop-Process -Id PROCESS_ID_HERE
```
