"""YOLO11n exporté en ONNX, exécuté avec onnxruntime (sans PyTorch ni Ultralytics)."""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detections:
    xyxy: np.ndarray    # (N, 4) en pixels de l'image d'origine
    conf: np.ndarray    # (N,)
    cls: np.ndarray     # (N,) indices de classes COCO

    def __len__(self) -> int:
        return len(self.conf)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


def letterbox(image: np.ndarray, size: int, stride: int = 32) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensionne (côté le plus long = size) et complète au multiple de 32 le plus proche (gris 114).

    Le modèle ONNX est exporté avec des dimensions dynamiques : comme Ultralytics, on évite un carré inutile.
    """
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else image
    tw, th = int(np.ceil(nw / stride) * stride), int(np.ceil(nh / stride) * stride)
    left, top = (tw - nw) // 2, (th - nh) // 2
    out = cv2.copyMakeBorder(resized, top, th - nh - top, left, tw - nw - left, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Suppression des non-maxima (numpy), renvoie les indices gardés."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ratio = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ratio <= iou]
    return np.array(keep, dtype=np.int64)


class OnnxYolo:
    def __init__(self, path: str, threads: int | None = None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or int(os.environ.get("ORT_THREADS", "1"))
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False      # moins de mémoire réservée (Render gratuit : 512 Mo)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}

    def __call__(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45, size: int = 640,
                 classes: list[int] | None = None) -> Detections:
        img, r, (left, top) = letterbox(image_bgr, size)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = self.session.run(None, {self.input_name: blob})[0][0].T     # (anchors, 4 + classes)
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        mask = best >= conf
        if classes is not None:
            mask &= np.isin(cls, classes)
        if not mask.any():
            return Detections.empty()
        xywh, best, cls = pred[mask, :4], best[mask], cls[mask]
        xyxy = np.empty_like(xywh)
        xyxy[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
        xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
        offset = cls[:, None] * 4096.0                 # NMS par classe
        keep = nms(xyxy + offset, best, iou)[:300]
        xyxy, best, cls = xyxy[keep], best[keep], cls[keep]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        h, w = image_bgr.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        return Detections(xyxy.astype(np.float32), best.astype(np.float32), cls.astype(np.int64))
