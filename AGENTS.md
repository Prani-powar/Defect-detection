# Conveyor Vision Sorter Agent Notes

Build a beginner-friendly Python computer vision prototype for apple quality sorting.

Current first version:

- Product: apple
- Classes: `fresh`, `rotten`
- Model: TensorFlow/Keras MobileNetV2 transfer learning
- Runtime modes: single image, live webcam, background queue, Streamlit dashboard
- DPR metrics: total inspected, fresh count, rotten count, uncertain count, defect percentage

Important constraints:

- Keep paths relative to the project.
- Keep code local; do not require paid services or cloud APIs.
- Prefer simple working scripts over complex research code.
- Use `config.py` for shared settings.
- Make the architecture extensible for future tasks such as bottle label detection.
- Treat this as a prototype; real sorting requires controlled lighting, calibration, and safety testing.
