# Validation

## Automated checks

- `python -m pytest -q`: **13 passed**.
- `python -m ruff check .`: **passed**.
- `python -m compileall -q src app.py`: **passed**.
- Streamlit `AppTest.from_file('app.py').run()`: page title verified; no startup exceptions.

Tests cover catalog checks, ranking abstention, invalid/clipped boxes, filtering non-target classes, detection thresholds, small crops, detection-only behavior, detection limits, non-mutating annotations, JSON serialization, IoU, one-to-one matching, and sampled video frame/JSONL exports. Tests use deterministic detector outputs and do not download weights.

## Real pretrained inference smoke checks

Both the TorchVision detector and CLIP weights downloaded and loaded successfully on CPU. The following were exercised outside the automated test suite:

| Input | Observed behavior |
| --- | --- |
| Blank white image | Zero target detections |
| Aircraft composite image | Seven airplane detections at threshold 0.6; ranked aircraft candidates and both tentative/unknown outputs |
| Car sample | One car detection; model identification abstained with three candidate labels |
| Gondola scene | Three boat detections; model identification abstained; a small crop skipped classification |

These checks demonstrate functioning inference, not verified detection precision, recall, or model-identification accuracy. The boat catalog lacks gondolas, illustrating why candidate estimates require validation. No labeled project benchmark has been run. The README documents how to evaluate a labeled set.

Public smoke-check inputs (downloaded temporarily and **not redistributed**):

- [Aircraft composite](https://github.com/kerrgarr/May-2022-Aircraft-Image-Classification/blob/main/pics/DifferentLooking.png)
- [Car sample](https://github.com/fishman2008/stanford-cars-classification/blob/master/images/samples/02381.jpg)
- [Gondola scene](https://github.com/Riccorl/CNN-Venice-Boat/blob/master/resources/gondola_ex.jpg)

Environment: Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu, transformers 4.57.6, OpenCV 5.0.0, Pillow 12.3.0, Streamlit 1.66.1.
