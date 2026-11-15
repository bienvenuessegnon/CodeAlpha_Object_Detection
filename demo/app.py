"""Démo en ligne (version allégée) : détection YOLO11n en ONNX + suivi ByteTrack, interface Gradio en français.

Sans PyTorch ni Ultralytics à l'exécution, pour tenir dans 512 Mo de mémoire (Render, offre gratuite).
Lancer : pip install -r demo/requirements.txt && python demo/app.py
"""
from __future__ import annotations

import os

# Un seul fil de calcul : l'offre gratuite n'a qu'une fraction de CPU, et chaque fil OpenBLAS / OpenCV
# réserve sa propre mémoire (512 Mo au total).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

try:  # glibc : au plus 2 zones d'allocation (sinon une par fil et par cœur de la machine hôte)
    import ctypes
    ctypes.CDLL("libc.so.6").mallopt(-8, 2)   # M_ARENA_MAX
except (OSError, AttributeError):
    pass

import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np

from detector import OnnxYolo
from tracker import ByteTracker

cv2.setNumThreads(1)
HERE = Path(__file__).parent
MODEL = OnnxYolo(str(HERE / "model" / "yolo11n.onnx"))
NAMES = MODEL.names
CLASS_CHOICES = sorted(NAMES.values())
MAX_SECONDS, MAX_FRAMES = 15, 450           # durée analysée au plus
VIDEO_FPS_CAP, VIDEO_SIZE, VIDEO_WIDTH = 10, 640, 768
LOW_CONF = 0.1                              # ByteTrack garde aussi les détections faibles (2e association)


def memory_mb() -> str:
    """Mémoire résidente du processus et de ses fils ffmpeg (journal du service)."""
    def rss(pid: str) -> int:
        try:
            with open(f"/proc/{pid}/status") as fh:
                return next(int(l.split()[1]) for l in fh if l.startswith("VmRSS")) // 1024
        except (OSError, StopIteration, ValueError):
            return 0
    me, kids = str(os.getpid()), []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/stat") as fh:
                if fh.read().rsplit(")", 1)[1].split()[1] == me:
                    kids.append(pid)
        except (OSError, IndexError):
            pass
    try:
        st = {l.split(":")[0]: l.split()[1] for l in open("/proc/self/status") if ":" in l and len(l.split()) > 1}
        extra = [f"anon {int(st['RssAnon']) // 1024} Mo", f"fils d'exécution {st['Threads']}"]
    except (OSError, KeyError, ValueError):
        extra = []
    parts = [f"python {rss(str(os.getpid()))} Mo"] + extra + [f"fils {rss(k)} Mo" for k in kids]
    try:
        parts.append(f"cgroup {int(open('/sys/fs/cgroup/memory.current').read()) // 2**20} Mo")
    except (OSError, ValueError):
        pass
    return ", ".join(parts)


def log(msg: str) -> None:
    print(f"[démo] {msg} | mémoire : {memory_mb()}", flush=True)


def color(cls: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(cls * 7 + 3)
    return tuple(int(v) for v in rng.integers(60, 230, 3))


def class_ids(selected: list[str] | None) -> list[int] | None:
    if not selected:
        return None
    lookup = {v: k for k, v in NAMES.items()}
    return [lookup[s] for s in selected if s in lookup]


def draw_box(img, box, label, cls):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    c = color(cls)
    t = max(2, round(img.shape[1] / 500))
    cv2.rectangle(img, (x1, y1), (x2, y2), c, t)
    scale = max(0.5, img.shape[1] / 1400)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t - 1))
    y = max(y1, th + 6)
    cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 6, y), c, -1)
    cv2.putText(img, label, (x1 + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, t - 1),
                cv2.LINE_AA)


def draw_panel(img, lines):
    scale = max(0.5, img.shape[1] / 1600)
    thick = max(1, round(scale * 2))
    step = int(30 * scale) + 6
    width = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for s in lines) + 20
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 14 + step * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (16, 6 + step * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (80, 230, 80) if i == 0 else (60, 220, 240), thick, cv2.LINE_AA)


def table(counts: dict[str, int]) -> list[list]:
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[k, int(v)] for k, v in rows] or [["(aucun objet)", 0]]


# ------------------------------------------------------------------ image
def detect_image(image: np.ndarray | None, conf: float, classes: list[str] | None):
    if image is None:
        raise gr.Error("Ajoutez une image.")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    det = MODEL(bgr, conf=conf, classes=class_ids(classes))
    log(f"image {image.shape[1]}x{image.shape[0]} : {len(det.cls)} objets")
    for box, s, c in zip(det.xyxy, det.conf, det.cls):
        draw_box(bgr, box, f"{NAMES[int(c)]} {s:.2f}", int(c))
    draw_panel(bgr, [f"Objets detectes : {len(det)}"])
    counts = Counter(NAMES[int(c)] for c in det.cls)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), table(counts)


# ------------------------------------------------------------------ vidéo
class H264Writer:
    """Écrit une vidéo MP4 H.264 lisible par les navigateurs via le ffmpeg fourni par imageio-ffmpeg.

    Un seul fil d'encodage et le préréglage « ultrafast » : peu de mémoire et peu de CPU.
    """

    def __init__(self, path: Path, fps: float, width: int, height: int):
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "ultrafast", "-crf", "26", "-threads", "1", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=120)


def track_video(video: str | None, conf: float, classes: list[str] | None, progress=gr.Progress()):
    if not video:
        raise gr.Error("Ajoutez une vidéo courte (les 15 premières secondes sont analysées).")
    # Lecture par le ffmpeg d'imageio-ffmpeg : un seul fil de décodage, cadence et taille réduites par ffmpeg
    # (le décodeur intégré à OpenCV ouvre un fil par cœur de la machine hôte, trop gourmand en mémoire).
    try:
        log("vidéo reçue")
        probe = imageio_ffmpeg.read_frames(video, input_params=["-threads", "1"],
                                           output_params=["-threads", "1", "-filter_threads", "1", "-frames:v", "1"])
        meta = next(probe)
        probe.close()
    except Exception as exc:  # fichier illisible
        raise gr.Error("Impossible de lire cette vidéo.") from exc
    src_fps = meta.get("fps") or 25
    step = max(1, round(src_fps / VIDEO_FPS_CAP))        # on traite au plus ~10 images/s
    out_fps = src_fps / step
    duration = min(MAX_SECONDS, meta.get("duration") or MAX_SECONDS)
    limit = max(1, min(MAX_FRAMES, int(duration * out_fps)))
    frames = imageio_ffmpeg.read_frames(
        video, pix_fmt="bgr24", input_params=["-threads", "1", "-t", str(MAX_SECONDS)],
        output_params=["-threads", "1", "-filter_threads", "1", "-an", "-vf",
                       f"fps={out_fps:.4f},scale='trunc(min({VIDEO_WIDTH},iw)/2)*2':-2"])
    w, h = next(frames)["size"]
    log(f"vidéo {meta.get('source_size')} {src_fps:.1f} i/s -> {w}x{h} {out_fps:.1f} i/s, {limit} images")
    tracker = ByteTracker(track_thresh=conf, low_thresh=min(LOW_CONF, conf), new_track_thresh=conf + 0.1)
    seen: dict[str, set[int]] = defaultdict(set)
    out_path = Path(tempfile.mkdtemp(prefix="suivi_")) / "suivi.mp4"
    writer, done, t0 = H264Writer(out_path, out_fps, w, h), 0, time.perf_counter()
    try:
        for raw in frames:
            if done >= limit:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            det = MODEL(frame, conf=min(LOW_CONF, conf), size=VIDEO_SIZE, classes=class_ids(classes))
            tracks = tracker.update(det.xyxy, det.conf, det.cls)
            for t in tracks:
                seen[NAMES[t.cls]].add(t.track_id)
                draw_box(frame, t.xyxy, f"id:{t.track_id} {NAMES[t.cls]} {t.score:.2f}", t.cls)
            total_ids = sum(len(v) for v in seen.values())
            draw_panel(frame, [f"Objets dans l'image : {len(tracks)}", f"Total suivi : {total_ids}"]
                       + [f"  {k} : {len(v)}" for k, v in sorted(seen.items())])
            writer.write(frame)
            done += 1
            if done % 10 == 0:
                log(f"image {done}/{limit}")
            if done % 10 == 0:
                progress(done / limit, desc=f"Image {done}/{limit}")
    finally:
        frames.close()
        writer.release()
    if done == 0:
        raise gr.Error("Aucune image lue dans cette vidéo.")
    elapsed = time.perf_counter() - t0
    log(f"vidéo terminée : {done} images en {elapsed:.0f} s")
    counts = {k: len(v) for k, v in seen.items()}
    summary = (f"**{done / out_fps:.1f} s analysées** ({done} images traitées, {out_fps:.0f} images/s, "
               f"calcul {elapsed:.0f} s). **{sum(counts.values())} objets suivis** : identifiants uniques "
               f"ByteTrack ; un objet perdu puis retrouvé peut recevoir un nouvel identifiant.")
    return str(out_path), table(counts), summary


# ------------------------------------------------------------------ interface
INTRO = """# 👁️ Détection et suivi d'objets
YOLO11n (pré-entraîné sur COCO, 80 classes, exporté en ONNX) + suivi **ByteTrack** · stage IA CodeAlpha (tâche 4) ·
[code source](https://github.com/bienvenuessegnon/CodeAlpha_Object_Detection)

<small>Hébergement gratuit : le premier chargement peut prendre une minute et les calculs sont lents (CPU partagé).</small>"""

with gr.Blocks(title="Détection et suivi d'objets") as demo:
    gr.Markdown(INTRO)
    conf = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Seuil de confiance")
    classes = gr.Dropdown(CLASS_CHOICES, multiselect=True, label="Classes à garder (vide = toutes)")
    headers = ["Classe", "Nombre"]
    with gr.Tab("Image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Image", sources=["upload", "clipboard"])
            img_out = gr.Image(label="Résultat")
        img_counts = gr.Dataframe(headers=headers, label="Objets détectés par classe", interactive=False)
        gr.Button("Détecter", variant="primary").click(detect_image, [img_in, conf, classes],
                                                       [img_out, img_counts], api_name="image",
                                                       concurrency_limit=1)
        gr.Examples([[str(HERE / "examples" / "bus.jpg")], [str(HERE / "examples" / "zidane.jpg")]],
                    inputs=[img_in], label="Exemples")
    with gr.Tab("Vidéo"):
        gr.Markdown(f"Vidéo courte : seules les **{MAX_SECONDS} premières secondes** sont analysées, "
                    f"à {VIDEO_FPS_CAP} images/s au plus. Comptez une à deux minutes de calcul.")
        with gr.Row():
            vid_in = gr.Video(label="Vidéo", sources=["upload"])
            vid_out = gr.Video(label="Résultat (identifiants de suivi)")
        vid_summary = gr.Markdown()
        vid_counts = gr.Dataframe(headers=headers, label="Objets uniques suivis par classe", interactive=False)
        gr.Button("Suivre les objets", variant="primary").click(
            track_video, [vid_in, conf, classes], [vid_out, vid_counts, vid_summary], api_name="video",
            concurrency_limit=1)
        gr.Examples([[str(HERE / "examples" / "parking_intel.mp4")]], inputs=[vid_in], label="Exemple")
        gr.Markdown("<small>Exemple : extrait de *person-bicycle-car-detection.mp4*, "
                    "[Intel IoT DevKit sample videos](https://github.com/intel-iot-devkit/sample-videos), "
                    "CC BY 4.0.</small>")
    gr.Markdown("<small>Modèle YOLO11n © Ultralytics, licence AGPL-3.0. Images d'exemple : dépôt Ultralytics.</small>")

if __name__ == "__main__":
    try:
        flags = next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split()
        cpu = " ".join(f for f in flags if f.startswith(("avx512f", "amx", "avx2")))
    except (OSError, StopIteration):
        cpu = "?"
    log(f"démarrage ({os.cpu_count()} cœurs visibles, {cpu})")
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="green"), show_error=True, max_file_size="40mb")
