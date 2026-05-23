# DPR Context Applied To This Project

The supplied DPR describes an AI-based smart manufacturing defect detection prototype for apples. This project now follows that scope directly.

## Active Scope

- Product: apple
- Classes: `fresh`, `rotten`
- Fresh/non-defective equivalent: `fresh`
- Defective/rotten equivalent: `rotten`

## DPR Requirements Covered

- Python implementation
- TensorFlow/Keras model training
- MobileNetV2 transfer learning
- Kaggle dataset workflow
- OpenCV webcam prediction
- Streamlit dashboard
- Real-time prediction display
- Total, fresh, rotten, uncertain, and defect percentage counts
- Conveyor-style inspection simulation
- Optional ESP32/servo path for future hardware
- Beginner-friendly modular files

## Dataset Mapping

Mixed fruit Kaggle datasets often include folders like:

```text
freshapples/
rottenapples/
fresh/apple/
rotten/apple/
```

The script below extracts only apple images:

```bash
python src/prepare_kaggle_apples.py --source downloads/apples
```

## Why MobileNetV2

The DPR recommends MobileNetV2 because it is fast, beginner-friendly, easier to train than a custom CNN, and suitable for webcam/conveyor prototypes.
