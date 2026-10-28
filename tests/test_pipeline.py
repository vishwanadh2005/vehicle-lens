import json

import pytest
from PIL import Image

from vehicle_lens.pipeline import (
    Detection,
    annotate,
    choose_identification,
    clip_box,
    load_catalog,
    report,
)


def test_catalog_supports_all_categories():
    catalog = load_catalog()
    assert set(catalog) == {"airplane", "boat", "car"}
    assert "Boeing 737" in catalog["airplane"]
    assert "Tesla Model 3" in catalog["car"]
    assert any("unknown" in label for label in catalog["boat"])


def test_ambiguous_models_abstain():
    rows = [{"label": "Boeing 737", "relative_score": 0.44},
            {"label": "Airbus A320", "relative_score": 0.42}]
    assert choose_identification(rows)[0] == "unknown"


def test_distinctive_match_is_tentative():
    rows = [{"label": "Boeing 747", "relative_score": 0.75},
            {"label": "Airbus A320", "relative_score": 0.1}]
    assert choose_identification(rows) == ("Boeing 747", "Tentative visual match")


def test_unknown_and_empty_candidates():
    assert choose_identification([])[0] == "unknown"
    assert choose_identification([{"label": "unknown car model", "relative_score": 0.9}])[0] == "unknown"


def test_box_clipped_to_image():
    assert clip_box([-2.5, 3.2, 105, 60], 100, 50) == [0, 3, 100, 50]
    with pytest.raises(ValueError):
        clip_box([0, 0, float("nan"), 5], 100, 100)


def test_report_is_json_and_annotation_preserves_input():
    image = Image.new("RGB", (100, 100), "white")
    detection = Detection("car", 0.92, [10, 30, 90, 80], "unknown", [], "Ambiguous")
    output = annotate(image, [detection])
    assert output.getpixel((10, 40)) != image.getpixel((10, 40))
    assert image.getpixel((10, 40)) == (255, 255, 255)
    data = json.loads(json.dumps(report(image, [detection])))
    assert data["detections"][0]["box"] == [10, 30, 90, 80]
    assert data["image_size"] == [100, 100]


def test_invalid_catalog_rejected(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text('{"car": ["same", "same"]}')
    with pytest.raises(ValueError):
        load_catalog(file)
