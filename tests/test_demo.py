"""Tests de la démo allégée (demo/) : détecteur ONNX et ByteTrack numpy. Ignorés si onnxruntime/scipy absents."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
DEMO = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO))

from tracker import ByteTracker, iou_matrix  # noqa: E402


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def _box(x):
    return np.array([[x, 50, x + 40, 130]], float)


def test_tracker_keeps_id_on_moving_object_and_confirms_after_two_hits():
    tr = ByteTracker(track_thresh=0.5, new_track_thresh=0.6)
    first = tr.update(_box(0), np.array([0.9]), np.array([0]))
    assert first == []                               # pas encore confirmé
    ids = set()
    for x in range(4, 80, 4):
        out = tr.update(_box(x), np.array([0.9]), np.array([0]))
        ids |= {t.track_id for t in out}
    assert ids == {1}


def test_low_score_detection_keeps_track_alive():
    tr = ByteTracker(track_thresh=0.5, low_thresh=0.1)
    for x in (0, 4, 8):
        tr.update(_box(x), np.array([0.9]), np.array([0]))
    out = tr.update(_box(12), np.array([0.3]), np.array([0]))     # score faible : 2e association
    assert [t.track_id for t in out] == [1]


def test_lost_track_recovered_within_buffer_and_classes_separate():
    tr = ByteTracker(buffer=5)
    for x in (0, 4, 8):
        tr.update(_box(x), np.array([0.9]), np.array([0]))
    for _ in range(3):                                             # objet caché 3 images
        assert tr.update(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) == []
    out = tr.update(_box(20), np.array([0.9]), np.array([0]))
    assert [t.track_id for t in out] == [1]
    # même position, autre classe : nouvelle piste, pas de mélange
    tr.update(np.vstack([_box(24), _box(24)]), np.array([0.9, 0.9]), np.array([0, 2]))
    out = tr.update(np.vstack([_box(28), _box(28)]), np.array([0.9, 0.9]), np.array([0, 2]))
    assert sorted((t.cls, t.track_id) for t in out) == [(0, 1), (2, 2)]


def test_onnx_detector_on_example_image():
    pytest.importorskip("onnxruntime")
    cv2 = pytest.importorskip("cv2")
    from detector import OnnxYolo, letterbox
    img = cv2.imread(str(DEMO / "examples" / "bus.jpg"))
    padded, r, _ = letterbox(img, 640)
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0 and max(padded.shape[:2]) == 640
    model = OnnxYolo(str(DEMO / "model" / "yolo11n.onnx"))
    det = model(img, conf=0.5)
    names = [model.names[int(c)] for c in det.cls]
    assert names.count("bus") == 1 and names.count("person") >= 4
    assert (det.xyxy[:, 2] <= img.shape[1]).all() and (det.xyxy[:, 3] <= img.shape[0]).all()
