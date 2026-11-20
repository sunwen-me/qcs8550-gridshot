# 模型接口

将自行训练或获准使用的模型放在本目录，或设置 `AIMLABS_MODEL` 指向其他路径。权重、数据和厂商运行时不会提交。

| 路径 | 输入 | 输出 |
|---|---|---|
| OpenCV DNN `.onnx` | FP32 RGB，`[1,3,640,640]`，归一化到 0–1 | FP32 `[1,5,8400]` |
| QNN context `.bin` | FP16 RGB，`[1,640,640,3]`，归一化到 0–1 | `[1,5,8400]`，适配器可读取为 FP32 |

输出五个通道为 `(cx, cy, w, h, score)`，坐标是补边后 640×640 模型输入上的像素坐标。一个类别、未做 NMS。模型文件扩展名为 `.onnx` 时走 OpenCV；其他扩展名走 QNN 构建。

`training/train.py`、`refine-v2.py` 与 `export.py` 是原型训练、可选微调和 ONNX 导出的参考脚本。训练目录预期如下：

```text
training/
  yolo11n.pt              # 自行合法获取，需兼容 Ultralytics YOLO11
  data/{train,valid,test}/
    images/               # 使用有权训练和分享的数据
    labels/               # YOLO 格式；类别 0 = Turquoise Ball
```

原训练脚本需要 CUDA、PyTorch 和 Ultralytics。ONNX 导出检查还需要 NumPy、OpenCV、ONNX 和 ONNX Runtime。先训练，随后执行 `python training/export.py`；脚本从 `training/runs/ball/weights/best.pt` 导出固定 `[1,3,640,640]` ONNX。不附带数据，因此不能从仓库直接重建原权重。QNN 需要另行转换为适配板上 QAIRT/HTP 的 context binary；把 ONNX 改后缀并不能运行，也不要把 NCHW context 当成 NHWC 使用。
