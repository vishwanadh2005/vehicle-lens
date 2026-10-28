# Vehicle Lens

**Detect airplanes, boats, and cars, then explore their likely models.**

A local computer vision application combining a COCO-pretrained Faster R-CNN detector with CLIP image–text matching. It locates vehicles in images or sampled video frames, crops each vehicle, ranks model names, and exports annotated images and structured JSON.

## Features

- **Three vehicle categories:** airplanes (including sufficiently large aircraft photographed in flight), boats, and cars.
- **Model estimates:** a starter catalog of aircraft families, car models, and boat models/types; add your own labels without retraining.
- **Abstention:** small crops and ambiguous rankings return `unknown`; top candidates remain available for inspection.
- **Interactive Streamlit app:** per-vehicle crops, category counts, relative candidate scores, and downloadable results.
- **Image and video CLI:** annotated images, JSON reports, sampled frames, and timestamped JSONL.
- **Evaluation:** one-to-one IoU matching, detection precision/recall, candidate top-1 accuracy, and identification coverage.
- **Automated checks:** unit tests and GitHub Actions on Python 3.11 and 3.12.

## Quick start

Python 3.11+ is required. A GPU is optional. The first inference downloads about 700 MB of model weights; later runs use the local cache. CPU inference is slower than GPU inference. No API key or paid service is required.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# CPU installation (Windows/Linux):
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# macOS: python -m pip install torch torchvision
# For CUDA, use the matching command from https://pytorch.org/get-started/locally/

python -m pip install -e '.[demo,dev]'
python -m streamlit run app.py
```

Upload a JPG, PNG, or WebP photo, choose a detection threshold, and select **Analyze image**. Images are processed on the machine running the app.

### Command line

```bash
vehicle-lens aircraft.jpg --output outputs/aircraft
vehicle-lens harbor.jpg --device cpu --output outputs/harbor
vehicle-lens street.jpg --detect-only
vehicle-lens clip.mp4 --video --stride 15 --output outputs/video
```

Image results are `annotated.jpg` and `report.json`. Video results are annotated JPEGs for every Nth frame and `frames.jsonl` with source frame indices and timestamps. The video mode does not track identities, count unique vehicles, or export an annotated movie.

## How it works

1. **Detect:** TorchVision Faster R-CNN with a MobileNetV3 backbone returns COCO bounding boxes. Keep `airplane`, `boat`, and `car` above the selected threshold.
2. **Crop:** clip each box to the image and preserve the original image coordinates.
3. **Identify:** compare CLIP image embeddings against cached text embeddings of the category's candidate labels.
4. **Rank:** apply softmax to CLIP similarity logits; return the top three labels.
5. **Abstain:** return `unknown` if the crop's smaller side is under 32 pixels, the leading relative score is below 0.35, the top-two gap is under 0.08, or an unknown label ranks first.

Detection confidence and candidate relative score mean different things. CLIP scores sum to one **within the supplied candidate list**; they are not calibrated probabilities of a specific model. The abstention thresholds are initial heuristics and need validation on your own data. An unknown text prompt does not guarantee reliable open-set rejection.

## Customize the catalog

Copy `src/vehicle_lens/catalog.json` and edit the labels under `airplane`, `boat`, and `car`. Each category needs at least two unique, non-empty labels.

```bash
vehicle-lens photo.jpg --catalog my_catalog.json
```

Aircraft examples include Boeing 737, Airbus A320, Cessna 172, and C-130. Cars include Tesla Model 3, Toyota Camry, and Porsche 911. The boat list mixes named models/families and vessel types because detailed manufacturer/model identification from distant hulls is often underdetermined. Limit labels to plausible local vehicles when appropriate, while retaining an unknown option.

## Evaluate on your own images

Create a JSON manifest next to your labeled images. Boxes use `[x_min, y_min, x_max, y_max]` in original, EXIF-corrected image pixel coordinates. Model labels should match catalog strings exactly. Every visible target vehicle should be annotated to avoid counting unlabeled objects as false positives.

```json
[
  {
    "image": "images/aircraft.jpg",
    "objects": [
      {"category": "airplane", "box": [42, 80, 520, 310], "model": "Boeing 747"}
    ]
  }
]
```

```bash
python -m vehicle_lens.evaluate evaluation.json --output metrics.json
```

Matching is confidence-ordered, category-aware, one-to-one, at IoU ≥ 0.5. This computes single-threshold precision and recall, **not COCO mAP**. Model accuracy and abstention coverage are reported over matched objects with model labels. Missed objects affect recall and are excluded from conditional model accuracy. No project-level accuracy claim is made without a labeled evaluation set.

## Development

```bash
python -m pytest -q
python -m ruff check .
```

Tests run without downloading pretrained weights. They cover catalog validation, candidate abstention, bounding-box clipping, annotations and reports, and duplicate/category-aware evaluation matching. See `VALIDATION.md` for checks performed during development.

## Limitations and next steps

- Tiny aircraft, blur, darkness, occlusion, rear views, and unusual viewpoints reduce detection and identification quality. Cropping cannot recover missing visual details.
- Airplanes on the ground are detected too; the app does not infer airborne status, altitude, flight number, registration, or location.
- Similar aircraft variants, car years/trims, and boat models often cannot be distinguished reliably with generic CLIP. Estimates should be checked against visible details.
- This is pretrained-model integration, not a custom-trained fine-grained classifier. Adding candidate names changes ranking; it does not teach new visual distinctions.
- A stronger next iteration would fine-tune dedicated classifiers on licensed aircraft/car/vessel datasets, hold out validation/test splits, and calibrate rejection thresholds. Dataset and model licenses should be checked before redistribution or deployment.

## Model references

- [TorchVision Faster R-CNN MobileNetV3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html)
- [CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32)
- [OpenAI CLIP](https://github.com/openai/CLIP)

Project code is MIT licensed. Pretrained model weights and third-party dependencies retain their respective licenses and terms.
