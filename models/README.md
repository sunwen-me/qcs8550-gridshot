# 模型格式

把模型放在本目录，或用 `AIMLABS_MODEL` 指定路径。

| 后端 | 输入 | 输出 |
|---|---|---|
| OpenCV DNN `.onnx` | FP32 RGB `[1,3,640,640]`，归一化到 0–1 | FP32 `[1,5,8400]` |
| QNN context `.bin` | FP16 RGB `[1,640,640,3]`，归一化到 0–1 | `[1,5,8400]`，由适配器读为 FP32 |

输出为 `(cx, cy, w, h, score)`，坐标按补边后的 640×640 输入计算，一个类别，模型不做 NMS。运行时按扩展名选择后端：`.onnx` 走 OpenCV，其余走 QNN。

## 训练和导出

```text
training/
  yolo11n.pt
  data/{train,valid,test}/
    images/
    labels/               # YOLO 格式；类别 0 = Turquoise Ball
```

用 CUDA、PyTorch 和 Ultralytics 跑 `python training/train.py`。`training/refine-v2.py` 是可选的微调脚本。然后用 `python training/export.py` 从 `training/runs/ball/weights/best.pt` 导出 ONNX。导出检查依赖 NumPy、OpenCV、ONNX 和 ONNX Runtime。

QNN 版需要另行生成与板上 QAIRT/HTP 匹配的 FP16 NHWC context。
