"""Real-time object detection and tracking with YOLO (Ultralytics) and OpenCV.

CodeAlpha Artificial Intelligence Internship - Task 4: Object Detection and Tracking.

Examples:
    python object_detection.py                              # default webcam
    python object_detection.py --source video.mp4 --save output/result.mp4
    python object_detection.py --source video.mp4 --no-show --save output/result.mp4   # headless
    python object_detection.py --source image.jpg --save output/image.jpg
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_NAME = "CodeAlpha - Object Detection and Tracking"


# ---------------------------------------------------------------- configuration
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object detection and tracking with YOLO + OpenCV.")
    p.add_argument("--source", default="0", help="webcam index (0, 1, ...), video file, image file or stream URL")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (downloaded automatically if missing)")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (0-1)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="tracker config: bytetrack.yaml or botsort.yaml")
    p.add_argument("--classes", nargs="*", default=None, help="only keep these class names, e.g. person car")
    p.add_argument("--save", default=None, help="save the annotated video/image to this path")
    p.add_argument("--no-show", action="store_true", help="do not open a window (servers, CI, notebooks)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = p.parse_args(argv)
    if not 0 < args.conf <= 1:
        p.error("--conf must be in ]0, 1]")
    return args


def parse_source(source: str) -> int | str:
    """'0' -> webcam index 0; anything else is kept as a path or URL."""
    return int(source) if str(source).isdigit() else str(source)


def is_image(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES


def class_ids_from_names(names: dict[int, str], wanted: list[str] | None) -> list[int] | None:
    """Translate class names (e.g. ['person', 'car']) into YOLO class indices."""
    if not wanted:
        return None
    lookup = {v.lower(): k for k, v in names.items()}
    unknown = [w for w in wanted if w.lower() not in lookup]
    if unknown:
        raise ValueError(f"Unknown class name(s): {', '.join(unknown)}")
    return [lookup[w.lower()] for w in wanted]


# ---------------------------------------------------------------- counting
class ObjectCounter:
    """Counts objects in the current frame and unique tracking IDs per class over the session."""

    def __init__(self) -> None:
        self.unique_ids: dict[str, set[int]] = defaultdict(set)
        self.current = 0

    def update(self, track_ids, class_ids, names: dict[int, str], n_boxes: int | None = None) -> None:
        self.current = len(class_ids) if n_boxes is None else n_boxes
        if track_ids is None:
            return
        for track_id, class_id in zip(track_ids, class_ids):
            self.unique_ids[names[int(class_id)]].add(int(track_id))

    @property
    def total(self) -> int:
        return sum(len(ids) for ids in self.unique_ids.values())

    def per_class(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in sorted(self.unique_ids.items())}


def draw_stats(frame: np.ndarray, counter: ObjectCounter, fps: float | None = None,
               tracking: bool = True) -> np.ndarray:
    """Overlay the counts (and FPS) in place on the frame, with a dark background for readability."""
    if not tracking:  # single image: there is nothing to track
        lines = [f"Objects detected: {counter.current}"]
    else:
        lines = [f"Objects in frame: {counter.current}", f"Total tracked: {counter.total}"]
        lines += [f"  {name}: {n}" for name, n in counter.per_class().items()]
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    scale = max(0.5, frame.shape[1] / 1600)
    thick = max(1, round(scale * 2))
    step = int(32 * scale) + 4
    width = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines)) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 16 + step * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (16, 8 + step * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 0) if i == 0 else (0, 255, 255), thick, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------- pipeline
def load_model(weights: str):
    from ultralytics import YOLO  # imported lazily so helpers and tests do not need torch
    return YOLO(weights)


def process_frame(model, frame: np.ndarray, counter: ObjectCounter, conf: float, tracker: str,
                  classes: list[int] | None, track: bool = True) -> np.ndarray:
    """Detect (and track) objects in one frame, update the counter and return the annotated frame."""
    if track:
        results = model.track(frame, persist=True, conf=conf, tracker=tracker, classes=classes, verbose=False)
    else:
        results = model.predict(frame, conf=conf, classes=classes, verbose=False)
    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        counter.update(None, [], model.names)
    else:
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else None
        counter.update(ids, boxes.cls.int().cpu().tolist(), model.names)
    return result.plot()


def run(args: argparse.Namespace) -> int:
    source = parse_source(args.source)
    if isinstance(source, str) and "://" not in source and not Path(source).exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        return 1
    model = load_model(args.model)
    try:
        classes = class_ids_from_names(model.names, args.classes)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    counter = ObjectCounter()

    if is_image(source):  # a single image: detection only, no tracking across frames
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: could not read image: {source}", file=sys.stderr)
            return 1
        annotated = draw_stats(process_frame(model, frame, counter, args.conf, args.tracker, classes, track=False),
                               counter, tracking=False)
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.save, annotated)
        if not args.no_show:
            cv2.imshow(WINDOW_NAME, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Objects detected: {counter.current}")
        return 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open the video source: {source}", file=sys.stderr)
        return 1
    writer = None
    frames, start, fps = 0, time.perf_counter(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = process_frame(model, frame, counter, args.conf, args.tracker, classes)
            frames += 1
            elapsed = time.perf_counter() - start
            fps = frames / elapsed if elapsed > 0 else None
            draw_stats(annotated, counter, fps)
            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"Frames processed: {frames}" + (f" ({fps:.1f} FPS)" if fps else ""))
    print(f"Total unique objects tracked: {counter.total}")
    for name, n in counter.per_class().items():
        print(f"  {name}: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
