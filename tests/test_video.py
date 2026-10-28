import json

import cv2
import numpy as np

from vehicle_lens.cli import process_video


class EmptyEngine:
    def predict(self, image, threshold):
        return image, []


def test_video_sampling_exports_original_indices(tmp_path):
    source = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((64, 64, 3), i * 40, dtype=np.uint8))
    writer.release()
    output = tmp_path / 'results'
    output.mkdir()
    process_video(EmptyEngine(), source, output, 0.6, stride=2)
    frames = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
    assert [frame['frame_index'] for frame in frames] == [0, 2, 4]
    assert len(list(output.glob('frame_*.jpg'))) == 3
    assert all(frame['detections'] == [] for frame in frames)
