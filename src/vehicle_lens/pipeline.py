"""Local inference; model weights are downloaded on first use."""

import json
import math
from dataclasses import asdict, dataclass
from importlib.resources import files

from PIL import ImageDraw, ImageOps


@dataclass
class Detection:
    category: str
    confidence: float
    box: list[int]
    identification: str
    candidates: list[dict]
    reason: str


def load_catalog(path=None):
    text = path.read_text() if path else files("vehicle_lens").joinpath("catalog.json").read_text()
    catalog = json.loads(text)
    for category in ("airplane", "boat", "car"):
        labels = catalog.get(category)
        if (not isinstance(labels, list) or len(labels) < 2
                or any(not isinstance(s, str) or not s.strip() for s in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError(f"{category} needs at least two unique non-empty labels")
    return catalog


def choose_identification(ranked, min_share=0.35, min_margin=0.08):
    """A heuristic abstention rule, not a calibrated confidence estimate."""
    if not ranked:
        return "unknown", "No candidates"
    best = ranked[0]
    gap = best["relative_score"] - (ranked[1]["relative_score"] if len(ranked) > 1 else 0)
    if "unknown" in best["label"].lower():
        return "unknown", "Unknown candidate ranked highest"
    if best["relative_score"] < min_share or gap < min_margin:
        return "unknown", "Ambiguous candidate ranking"
    return best["label"], "Tentative visual match"


def clip_box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Expected four finite box coordinates")
    x1, y1, x2, y2 = map(float, box)
    return [max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
            max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))]


class VehicleLens:
    def __init__(self, device="auto", catalog=None, identify=True):
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval().to(self.device)
        self.categories = weights.meta["categories"]
        self.transform = weights.transforms()
        self.catalog = catalog or load_catalog()
        self.identify = identify
        self.clip = None
        self.processor = None
        self.text_features = {}

    def _load_clip(self):
        if self.clip is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=True
        ).eval().to(self.device)
        for category, labels in self.catalog.items():
            inputs = self.processor(
                text=[f"a photo of a {label}, a {category}" for label in labels],
                return_tensors="pt", padding=True,
            ).to(self.device)
            with self.torch.inference_mode():
                features = self.clip.get_text_features(**inputs)
                self.text_features[category] = features / features.norm(dim=-1, keepdim=True)

    def rank(self, crop, category):
        self._load_clip()
        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.clip.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = self.clip.logit_scale.exp() * features @ self.text_features[category].T
            scores = logits.softmax(dim=-1)[0].cpu().tolist()
        return sorted(
            [{"label": label, "relative_score": float(score)}
             for label, score in zip(self.catalog[category], scores)],
            key=lambda row: row["relative_score"], reverse=True,
        )

    def predict(self, image, threshold=0.6, max_detections=30):
        if not 0 <= threshold <= 1 or max_detections < 1:
            raise ValueError("threshold must be in [0, 1] and max_detections must be positive")
        image = ImageOps.exif_transpose(image).convert("RGB")
        with self.torch.inference_mode():
            raw = self.detector([self.transform(image).to(self.device)])[0]
        detections = []
        for box, label, score in zip(raw["boxes"], raw["labels"], raw["scores"]):
            category = self.categories[int(label)]
            if category not in ("airplane", "boat", "car") or float(score) < threshold:
                continue
            bounds = clip_box(box.tolist(), *image.size)
            x1, y1, x2, y2 = bounds
            if x2 <= x1 or y2 <= y1:
                continue
            ranked = []
            identification, reason = "unknown", "Model identification disabled"
            if self.identify:
                if min(x2 - x1, y2 - y1) < 32:
                    reason = "Crop too small for model identification"
                else:
                    ranked = self.rank(image.crop(tuple(bounds)), category)
                    identification, reason = choose_identification(ranked)
            detections.append(Detection(category, float(score), bounds, identification,
                                        ranked[:3], reason))
            if len(detections) >= max_detections:
                break
        return image, detections


def annotate(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    colors = {"airplane": "#58bfff", "boat": "#35dfb0", "car": "#ffbd59"}
    for item in detections:
        x1, y1, x2, y2 = item.box
        color = colors[item.category]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{item.category} {item.confidence:.0%} | {item.identification} (estimate)"
        text_box = draw.textbbox((0, 0), label)
        y = max(0, y1 - text_box[3] - 8)
        draw.rectangle((x1, y, min(image.width, x1 + text_box[2] + 8), y + text_box[3] + 8),
                       fill=color)
        draw.text((x1 + 4, y + 3), label, fill="black")
    return result


def report(image, detections):
    return {"schema_version": 1, "image_size": list(image.size),
            "detector": "torchvision/fasterrcnn_mobilenet_v3_large_fpn",
            "identifier": "openai/clip-vit-base-patch32",
            "score_note": "Candidate scores are relative similarities, not model probabilities.",
            "detections": [asdict(item) for item in detections]}
