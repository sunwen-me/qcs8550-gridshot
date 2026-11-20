import csv
import json
from pathlib import Path

from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect import DetectionTrainer

root = Path(__file__).resolve().parent
dataset = str(root / "data/train-config.yaml")
baseline = root / "runs/ball/weights/best.pt"
settings.update({"sync": False})
common = {"data": dataset, "device": 0, "imgsz": 640, "batch": 16, "workers": 4, "plots": False}

class LocalDetectionTrainer(DetectionTrainer):
    def read_results_csv(self):
        try:
            with self.csv.open(newline="", encoding="utf-8") as handle:
                rows = csv.DictReader(handle, skipinitialspace=True)
                columns = {name.strip(): [] for name in (rows.fieldnames or [])}
                for row in rows:
                    for name, value in row.items():
                        columns[name.strip()].append(float(value))
            return columns
        except (OSError, ValueError):
            return {}


model = YOLO(str(baseline))
model.train(
    trainer=LocalDetectionTrainer,
    **common,
    project=str(root / "runs"),
    name="ball_refine_v2",
    epochs=35,
    patience=12,
    optimizer="AdamW",
    lr0=0.0003,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=0.0,
    fliplr=0.0,
    mosaic=0.25,
    close_mosaic=10,
    seed=42,
    verbose=False,
)
refined = Path(model.trainer.best)
comparison = {}
for label, checkpoint in {"baseline": baseline, "refined": refined}.items():
    current = YOLO(str(checkpoint))
    assert current.names == {0: "Turquoise Ball"}
    comparison[label] = {}
    for split in ("val", "test"):
        result = current.val(
            **common,
            split=split,
            project=str(root / "runs"),
            name=f"compare_{label}_{split}",
            verbose=False,
        )
        comparison[label][split] = {key: float(val) for key, val in result.results_dict.items()}

(root / "refine-metrics.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
print(json.dumps(comparison, indent=2), flush=True)
