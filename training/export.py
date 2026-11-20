from pathlib import Path
import shutil

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO, settings
from ultralytics.data.augment import LetterBox

root = Path(__file__).resolve().parent
target = root / "deployment"
target.mkdir(exist_ok=True)
checkpoint = target / "aimlabs-yolo11n"
shutil.copyfile(root / "runs/ball/weights/best.pt", checkpoint.with_suffix(".pt"))
settings.update({"sync": False})
model = YOLO(str(checkpoint.with_suffix(".pt")))
exported = Path(model.export(
    format="onnx", imgsz=640, batch=1, dynamic=False, half=False,
    simplify=False, opset=13, nms=False, device="cpu",
))
onnx.checker.check_model(str(exported))
session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
assert session.get_inputs()[0].shape == [1, 3, 640, 640]
assert session.get_outputs()[0].shape == [1, 5, 8400]
assert model.names == {0: "Turquoise Ball"}
model.model.eval()
for image_path in sorted((root / "data/test/images").glob("*.jpg"))[:3]:
    image = LetterBox(new_shape=(640,640))(image=cv2.imread(str(image_path)))
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    with torch.inference_mode():
        expected = model.model(torch.from_numpy(tensor))[0].cpu().numpy()
    actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    np.testing.assert_allclose(actual[:, :4], expected[:, :4], rtol=1e-4, atol=0.015)
    np.testing.assert_allclose(actual[:, 4:], expected[:, 4:], rtol=1e-3, atol=1e-4)
    print(image_path.name, actual.shape, "PT/ONNX matched")
print(exported)
