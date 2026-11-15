"""Tests that do not need the YOLO weights: a tiny fake model replaces Ultralytics."""
import numpy as np
import pytest

import object_detection as od


def test_parse_source_webcam_and_path():
    assert od.parse_source("0") == 0
    assert od.parse_source("2") == 2
    assert od.parse_source("video.mp4") == "video.mp4"
    assert od.is_image("photo.JPG") and not od.is_image("clip.mp4") and not od.is_image(0)


def test_parse_args_defaults_and_validation():
    args = od.parse_args([])
    assert args.source == "0" and args.tracker == "bytetrack.yaml" and not args.no_show
    with pytest.raises(SystemExit):
        od.parse_args(["--conf", "1.5"])


def test_class_ids_from_names():
    names = {0: "person", 1: "bicycle", 2: "car"}
    assert od.class_ids_from_names(names, None) is None
    assert od.class_ids_from_names(names, ["Car", "person"]) == [2, 0]
    with pytest.raises(ValueError):
        od.class_ids_from_names(names, ["dragon"])


def test_counter_counts_unique_ids_per_class():
    names = {0: "person", 2: "car"}
    c = od.ObjectCounter()
    c.update([1, 2], [0, 2], names)
    c.update([1, 3], [0, 0], names)          # ID 1 seen again: not counted twice
    assert c.current == 2
    assert c.total == 3
    assert c.per_class() == {"car": 1, "person": 2}
    c.update(None, [0], names)               # detections without IDs: frame count only
    assert c.current == 1 and c.total == 3


def test_draw_stats_keeps_shape_and_draws():
    frame = np.zeros((240, 320, 3), np.uint8)
    c = od.ObjectCounter()
    c.update([7], [0], {0: "person"})
    out = od.draw_stats(frame, c, fps=12.3)
    assert out.shape == (240, 320, 3) and out.any()
    single = od.draw_stats(np.zeros((240, 320, 3), np.uint8), c, tracking=False)
    assert single.any()


class _Tensor(list):
    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, ids, cls):
        self.id = _Tensor(ids) if ids is not None else None
        self.cls = _Tensor(cls)

    def __len__(self):
        return len(self.cls)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes

    def plot(self):
        return np.full((10, 10, 3), 255, np.uint8)


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, frames):
        self.frames, self.calls = list(frames), []

    def track(self, frame, **kw):
        self.calls.append(("track", kw))
        return [_Result(self.frames.pop(0))]

    def predict(self, frame, **kw):
        self.calls.append(("predict", kw))
        return [_Result(self.frames.pop(0))]


def test_process_frame_tracks_and_counts():
    model = FakeModel([_Boxes([1, 2], [0, 2]), _Boxes([2], [2]), _Boxes(None, [0])])
    c = od.ObjectCounter()
    frame = np.zeros((10, 10, 3), np.uint8)
    for _ in range(3):
        out = od.process_frame(model, frame, c, 0.4, "bytetrack.yaml", None)
    assert out.shape == (10, 10, 3)
    assert c.total == 2 and c.current == 1
    kind, kw = model.calls[0]
    assert kind == "track" and kw["persist"] is True and kw["tracker"] == "bytetrack.yaml"


def test_process_frame_image_mode_uses_predict():
    model = FakeModel([_Boxes(None, [0, 0, 2])])
    c = od.ObjectCounter()
    od.process_frame(model, np.zeros((10, 10, 3), np.uint8), c, 0.4, "bytetrack.yaml", None, track=False)
    assert model.calls[0][0] == "predict" and c.current == 3 and c.total == 0


def test_run_reports_missing_source(capsys):
    assert od.main(["--source", "does_not_exist.mp4", "--no-show"]) == 1
    assert "source not found" in capsys.readouterr().err
