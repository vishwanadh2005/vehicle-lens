"""Image and sampled-video command line entry points."""

import argparse
import json
from pathlib import Path

from PIL import Image

from .pipeline import VehicleLens, annotate, load_catalog, report


def process_video(engine, source, destination, threshold, stride):
    import cv2
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    frame_index = 0
    try:
        with (destination / "frames.jsonl").open("w") as stream:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    image, detections = engine.predict(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold
                    )
                    data = report(image, detections)
                    data.update(frame_index=frame_index, timestamp_ms=capture.get(cv2.CAP_PROP_POS_MSEC))
                    stream.write(json.dumps(data) + "\n")
                    annotate(image, detections).save(destination / f"frame_{frame_index:06d}.jpg")
                frame_index += 1
    finally:
        capture.release()
    if frame_index == 0:
        raise ValueError("Video contained no decodable frames")


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles and estimate their models")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--video", action="store_true", help="Process sampled video frames")
    parser.add_argument("--stride", type=int, default=15, help="Analyze every Nth video frame")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("input must be an existing file")
    if not 0 <= args.threshold <= 1 or args.stride < 1:
        parser.error("threshold must be in [0, 1]; stride must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        engine = VehicleLens(args.device, load_catalog(args.catalog), not args.detect_only)
        if args.video:
            process_video(engine, args.input, args.output, args.threshold, args.stride)
        else:
            with Image.open(args.input) as source:
                image, detections = engine.predict(source, args.threshold)
            annotate(image, detections).save(args.output / "annotated.jpg")
            (args.output / "report.json").write_text(json.dumps(report(image, detections), indent=2))
            print(f"Found {len(detections)} vehicles")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Inference failed: {error}\n")
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
