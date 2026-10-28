from vehicle_lens.evaluate import iou, match_predictions
from vehicle_lens.pipeline import Detection


def test_iou_overlapping_disjoint_and_empty():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    assert round(iou([0, 0, 10, 10], [5, 0, 15, 10]), 3) == 0.333


def test_matching_one_to_one_and_category_aware():
    truth = [{"category": "car", "box": [0, 0, 50, 50]}]
    detections = [Detection("car", 0.8, [0, 0, 50, 50], "unknown", [], ""),
                  Detection("car", 0.9, [0, 0, 50, 50], "unknown", [], ""),
                  Detection("boat", 0.99, [0, 0, 50, 50], "unknown", [], "")]
    matches = match_predictions(detections, truth)
    assert len(matches) == 1
    assert matches[0][0].confidence == 0.9
