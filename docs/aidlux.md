# 在 AidLux 部署

## 接线

1. 电脑的视频输出接犀牛派 X1 的 **HDMI IN**，把 Gridshot 放到这一路显示或镜像过去。用 `cat /sys/class/hdmi_in1/lt6911_detect/resolution` 查看输入分辨率。
2. 板上的 USB-C **OTG 数据口**接电脑。用 SSH 操作板子，重绑 USB 时这一路会短暂断开。
3. 用 `gst-inspect-1.0 qtiqmmfsrc` 检查采集插件。USB 脚本对应 `g1/configs/c.1` 和控制器 `a600000.dwc3`。

## 编译 QNN 版

板上实测用 QAIRT 2.50、HTP 和 FP16 context。构建时传入本地 `qnn_runtime` 适配器的头文件和静态库；没有这套依赖就走 [README 的 ONNX 路径](../README.md#怎么跑)。

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

确认日志里的 `detections` 和 `centers` 对得上画面，再退出检测进程。

## 配置 HID 和校准

```bash
sudo bash scripts/usb-mouse.sh enable
ls -l /dev/hidg*
# 确认 hidg0 对应新增的鼠标接口
sudo setfacl -m "u:$(id -un):-w-" /dev/hidg0
python3 live/calibrate-gridshot.py
./live/run-live.sh --active
```

脚本保留原来的 ADB 和诊断接口，新增的鼠标用三字节报文 `[buttons, dx, dy]`。校准时保持三个球靶静止；脚本会沿 X、Y 各移一次再移回，结果写到 `live/gridshot-calibration.json`。改分辨率或鼠标灵敏度后重新校准。启动器默认反馈超时为 120 ms。

`Ctrl+C` 停程序；`sudo bash scripts/usb-mouse.sh disable` 移除鼠标接口。重启后如果接口消失，重新启用并赋权即可。
