"""Exercise orchestration with deterministic detections; no model downloads."""

import torch
from PIL import Image

from vehicle_lens.pipeline import VehicleLens


def engine_with_outputs(boxes, labels, scores):
    engine = VehicleLens.__new__(VehicleLens)
    engine.torch = torch
    engine.device = torch.device('cpu')
    engine.categories = ['background', 'person', 'car', 'airplane', 'boat']
    engine.transform = lambda image: torch.zeros(3, image.height, image.width)
    engine.detector = lambda images: [{
        'boxes': torch.tensor(boxes), 'labels': torch.tensor(labels), 'scores': torch.tensor(scores)
    }]
    engine.identify = True
    engine.rank = lambda image, category: [
        {'label': 'Boeing 747', 'relative_score': 0.9},
        {'label': 'Boeing 737', 'relative_score': 0.1}
    ]
    return engine


def test_filters_categories_scores_and_degenerate_boxes():
    engine = engine_with_outputs(
        [[10, 10, 80, 80], [0, 0, 10, 10], [10, 10, 80, 80],
         [80, 80, 20, 20], [-10, 5, 60, 70]],
        [1, 2, 3, 4, 3], [0.99, 0.9, 0.1, 0.9, 0.95]
    )
    _, detections = engine.predict(Image.new('RGB', (100, 100)))
    assert [d.category for d in detections] == ['car', 'airplane']
    assert detections[0].identification == 'unknown'
    assert detections[0].reason == 'Crop too small for model identification'
    assert detections[1].box == [0, 5, 60, 70]
    assert detections[1].identification == 'Boeing 747'


def test_detection_only_never_calls_rank():
    engine = engine_with_outputs([[0, 0, 70, 70]], [2], [0.95])
    engine.identify = False
    engine.rank = lambda *args: (_ for _ in ()).throw(AssertionError('rank called'))
    _, detections = engine.predict(Image.new('RGB', (100, 100)))
    assert detections[0].candidates == []
    assert detections[0].identification == 'unknown'


def test_detection_limit():
    engine = engine_with_outputs([[0, 0, 70, 70]] * 3, [2, 2, 2], [0.9, 0.85, 0.8])
    _, detections = engine.predict(Image.new('RGB', (100, 100)), max_detections=1)
    assert len(detections) == 1
