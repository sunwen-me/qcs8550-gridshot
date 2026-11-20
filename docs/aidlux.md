# 在 AidLux 部署

## 接线与环境

1. 电脑的视频输出接板上的 **HDMI IN**（不是 HDMI OUT）。让靶场显示在这一路，或镜像画面。板上 `cat /sys/class/hdmi_in1/lt6911_detect/resolution` 应报告 `1920x1080`。
2. 板上的 USB-C **OTG 数据口**用数据线接电脑。不要把两个 USB Host 口相连。HID 测试前先保留独立的 SSH 连接。
3. 已验证的平台是犀牛派 X1 / QCS8550，AidLux 的 QMMF 插件和具备 GStreamer 支持的 OpenCV。可先运行 `gst-inspect-1.0 qtiqmmfsrc`。USB Gadget 脚本假定系统用 `g1/configs/c.1`、控制器名 `a600000.dwc3`；其他固件需自行核对，不能直接照跑。

## QNN 构建（可选）

本仓库不包含 `qnn_runtime` 适配器。如果你已有兼容的本地 `qnn_runtime/session.hpp` 和静态库，可以复用；否则先使用 README 中的 ONNX 路径。QNN 适配器需提供代码中使用的 `Session::load`、张量访问、`run` 和 `read_f32`。原型在板上使用 QAIRT 2.50 的配套 HTP 运行库及 FP16 context。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGRIDSHOT_WITH_QNN=ON \
  -DQNN_RUNTIME_INCLUDE_DIR=/path/to/include \
  -DQNN_RUNTIME_LIBRARY=/path/to/libqnn_runtime.a
cmake --build build -j2
ctest --test-dir build --output-on-failure

export AIMLABS_QNN_RUNTIME=/path/to/qairt250
export AIMLABS_MODEL=/path/to/ball-detector-context.bin
./live/run-live.sh --hdmi
```

只读检测不会打开 `/dev/hidg0`。先确认日志中出现正确的 `detections` 与 `centers`，然后停掉只读进程，再配置鼠标。

## HID 与校准

```bash
sudo bash scripts/usb-mouse.sh enable
ls -l /dev/hidg*
# 确认 hidg0 是脚本新建的鼠标接口之后，才执行：
sudo setfacl -m "u:$(id -un):-w-" /dev/hidg0
python3 live/calibrate-gridshot.py
./live/run-live.sh --active
```

脚本配置三字节相对报文 `[buttons, dx, dy]`，重绑时 USB 会短暂断开；原 ADB 和诊断配置会保留。重启后可能需要重新启用和赋权。校准会在三个静止球靶的画面上分别移动 X/Y 再移回，写入本地 `live/gridshot-calibration.json`；请只在离线靶场操作。改游戏灵敏度、分辨率或输入配置后重新校准。校准里的反馈测量是参考记录；启动器当前默认超时为 120 ms。

想停用 USB 鼠标：先 Ctrl+C 停程序，再执行 `sudo bash scripts/usb-mouse.sh disable`。只想看画面检测则用 `./live/run-live.sh --hdmi`；它不需要鼠标设备。
