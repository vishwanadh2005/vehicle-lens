"""Evaluate a user-labeled image set without claiming benchmark performance."""

import argparse
import json
from pathlib import Path

from PIL import Image

from .pipeline import VehicleLens


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_predictions(predictions, objects, threshold=0.5):
    """Confidence-ordered, one-to-one matching within the same vehicle category."""
    available = set(range(len(objects)))
    matches = []
    for prediction in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        options = [(iou(prediction.box, objects[i]["box"]), i) for i in available
                   if objects[i]["category"] == prediction.category]
        overlap, index = max(options, default=(0, -1))
        if index >= 0 and overlap >= threshold:
            matches.append((prediction, objects[index]))
            available.remove(index)
    return matches


def evaluate(engine, manifest, threshold=0.6):
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Evaluation manifest must contain a non-empty list")
    predicted = expected = matched = labeled = correct_top1 = accepted = correct_accepted = 0
    for entry in entries:
        with Image.open(manifest.parent / entry["image"]) as image:
            _, detections = engine.predict(image, threshold)
        objects = entry["objects"]
        matches = match_predictions(detections, objects)
        predicted += len(detections)
        expected += len(objects)
        matched += len(matches)
        for detection, truth in matches:
            if not truth.get("model"):
                continue
            labeled += 1
            if detection.candidates and detection.candidates[0]["label"] == truth["model"]:
                correct_top1 += 1
            if detection.identification != "unknown":
                accepted += 1
                correct_accepted += detection.identification == truth["model"]
    return {
        "images": len(entries), "predicted_objects": predicted, "ground_truth_objects": expected,
        "matched_objects": matched, "iou_threshold": 0.5, "detection_threshold": threshold,
        "precision": matched / predicted if predicted else None,
        "recall": matched / expected if expected else None,
        "matched_objects_with_model_labels": labeled,
        "candidate_top1_accuracy": correct_top1 / labeled if labeled else None,
        "identification_coverage": accepted / labeled if labeled else None,
        "accepted_identification_accuracy": correct_accepted / accepted if accepted else None,
        "note": "Single-threshold metrics; not COCO mAP. Model metrics condition on matched detections.",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate labeled vehicle images")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = evaluate(VehicleLens(args.device), args.manifest)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
