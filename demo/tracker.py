"""ByteTrack allégé (numpy + scipy) : filtre de Kalman, association en deux temps (scores hauts puis bas).

D'après Zhang et al., « ByteTrack: Multi-Object Tracking by Associating Every Detection Box » (ECCV 2022).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class KalmanBox:
    """Vitesse constante sur (cx, cy, rapport largeur/hauteur, hauteur), comme SORT / ByteTrack."""
    POS, VEL = 1 / 20, 1 / 160
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.eye(4, 8)

    def __init__(self, xyxy: np.ndarray):
        z = self.to_z(xyxy)
        self.x = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.POS * h, 2 * self.POS * h, 1e-2, 2 * self.POS * h,
               10 * self.VEL * h, 10 * self.VEL * h, 1e-5, 10 * self.VEL * h]
        self.P = np.diag(np.square(std))

    @staticmethod
    def to_z(b):
        w, h = b[2] - b[0], max(b[3] - b[1], 1e-6)
        return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h], float)

    def predict(self):
        h = self.x[3]
        q = np.square([self.POS * h, self.POS * h, 1e-2, self.POS * h,
                       self.VEL * h, self.VEL * h, 1e-5, self.VEL * h])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(q)

    def update(self, xyxy):
        z = self.to_z(xyxy)
        h = self.x[3]
        R = np.diag(np.square([self.POS * h, self.POS * h, 1e-1, self.POS * h]))
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def xyxy(self):
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    kf: KalmanBox
    cls: int
    score: float
    track_id: int = 0          # 0 tant que la piste n'est pas confirmée
    hits: int = 1
    lost: int = 0
    xyxy: np.ndarray = field(default=None)

    def __post_init__(self):
        self.xyxy = self.kf.xyxy


class ByteTracker:
    def __init__(self, track_thresh=0.5, low_thresh=0.1, new_track_thresh=0.6, match_thresh=0.8,
                 buffer=30, min_hits=2):
        self.track_thresh, self.low_thresh, self.new_track_thresh = track_thresh, low_thresh, new_track_thresh
        self.match_thresh, self.buffer, self.min_hits = match_thresh, buffer, min_hits
        self.tracks: list[Track] = []
        self.next_id = 1

    def _match(self, tracks, boxes, classes, max_cost):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        cost = 1 - iou_matrix(np.array([t.kf.xyxy for t in tracks]), boxes)
        cost[np.array([t.cls for t in tracks])[:, None] != classes[None, :]] = 1.0   # même classe
        rows, cols = linear_sum_assignment(cost)
        pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
        mr, mc = {r for r, _ in pairs}, {c for _, c in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mr], [j for j in range(len(boxes)) if j not in mc]

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> list[Track]:
        """Met à jour les pistes avec les détections d'une image ; renvoie les pistes confirmées visibles."""
        for t in self.tracks:
            t.kf.predict()
        high = conf >= self.track_thresh
        low = (conf >= self.low_thresh) & ~high
        hb, hs, hc = xyxy[high], conf[high], cls[high]
        lb, ls, lc = xyxy[low], conf[low], cls[low]

        # 1) toutes les pistes contre les détections de score élevé
        pairs, rest_t, rest_d = self._match(self.tracks, hb, hc, self.match_thresh)
        matched = set()
        for r, c in pairs:
            self._hit(self.tracks[r], hb[c], hs[c])
            matched.add(r)
        # 2) pistes actives restantes contre les détections de score faible
        active = [i for i in rest_t if self.tracks[i].lost == 0]
        pairs2, _, _ = self._match([self.tracks[i] for i in active], lb, lc, 0.5)
        for r, c in pairs2:
            self._hit(self.tracks[active[r]], lb[c], ls[c])
            matched.add(active[r])
        # pistes non associées : perdues, supprimées après `buffer` images (ou tout de suite si non confirmées)
        for i, t in enumerate(self.tracks):
            if i not in matched:
                t.lost += 1
        self.tracks = [t for t in self.tracks if (t.lost <= self.buffer if t.track_id else t.lost == 0)]
        # nouvelles pistes à partir des détections hautes non associées
        for j in rest_d:
            if hs[j] >= self.new_track_thresh:
                self.tracks.append(Track(KalmanBox(hb[j]), int(hc[j]), float(hs[j])))
        for t in self.tracks:
            if not t.track_id and t.hits >= self.min_hits:
                t.track_id, self.next_id = self.next_id, self.next_id + 1
        return [t for t in self.tracks if t.track_id and t.lost == 0]

    def _hit(self, t: Track, box, score):
        t.kf.update(box)
        t.xyxy, t.score, t.hits, t.lost = t.kf.xyxy, float(score), t.hits + 1, 0
