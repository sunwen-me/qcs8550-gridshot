import json
from pathlib import Path

import torch
from ultralytics import YOLO, settings

root = Path(__file__).resolve().parent
dataset = root / "data" / "train-config.yaml"
dataset.write_text(
    f"path: {root / 'data'}\n"
    "train: train/images\n"
    "val: valid/images\n"
    "test: test/images\n"
    "names:\n"
    "  0: Turquoise Ball\n",
    encoding="utf-8",
)
settings.update({"sync": False})
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the training environment")
print(f"CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}", flush=True)

model = YOLO(str(root / "yolo11n.pt"))
model.train(
    data=str(dataset),
    project=str(root / "runs"),
    name="ball",
    device=0,
    imgsz=640,
    batch=16,
    epochs=30,
    patience=10,
    workers=4,
    fliplr=0.0,
    seed=42,
    plots=False,
)
best_path = Path(model.trainer.best)
best = YOLO(str(best_path))
if best.names != {0: "Turquoise Ball"}:
    raise RuntimeError(f"Unexpected classes: {best.names}")
test = best.val(
    data=str(dataset),
    split="test",
    device=0,
    imgsz=640,
    batch=16,
    workers=4,
    plots=False,
    project=str(root / "runs"),
    name="test",
)
metrics = {key: float(value) for key, value in test.results_dict.items()}
(root / "test-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(f"Verified {best_path}: {metrics}", flush=True)
