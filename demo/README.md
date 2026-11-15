# Démo en ligne (version allégée)

Application Gradio en français : onglet **Image** (détection, nombre d'objets par classe) et onglet **Vidéo**
(suivi ByteTrack sur les 15 premières secondes, vidéo annotée avec les identifiants, objets uniques par classe).

Pour tenir dans 512 Mo de mémoire (offre gratuite Render), la démo n'utilise ni PyTorch ni Ultralytics à l'exécution :

| Élément | Choix |
| --- | --- |
| Détection | YOLO11n exporté en ONNX (`model/yolo11n.onnx`, 11 Mo, dimensions dynamiques), exécuté avec onnxruntime ; prétraitement et NMS en numpy / OpenCV (`detector.py`) |
| Suivi | ByteTrack allégé : filtre de Kalman + association en deux temps (scores hauts puis bas), algorithme hongrois de SciPy (`tracker.py`) |
| Vidéo | 15 s au plus, 10 images/s traitées au plus, largeur ≤ 768 px, lecture et sortie MP4 H.264 par le ffmpeg d’imageio-ffmpeg, un seul fil |
| Mémoire mesurée sur Render | ≈ 235 Mo au repos ; pendant une vidéo de 15 s, ≈ 256 Mo (Python) + ≈ 90 Mo (deux ffmpeg), soit ≈ 345 Mo, stable d’une image à l’autre (journal `[démo]` du service) |

Sur l'exemple, les détections sont identiques à celles d'Ultralytics (mêmes boîtes et scores sur `bus.jpg`).

```bash
pip install -r demo/requirements.txt
python demo/app.py            # http://localhost:7860
```

Export du modèle (une fois, avec Ultralytics) :

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="onnx", dynamic=True, simplify=True, opset=17)
```

Crédits : YOLO11n © Ultralytics (AGPL-3.0) ; images d'exemple du dépôt Ultralytics ; vidéo d'exemple extraite de
*person-bicycle-car-detection.mp4*, [Intel IoT DevKit sample videos](https://github.com/intel-iot-devkit/sample-videos) (CC BY 4.0).
