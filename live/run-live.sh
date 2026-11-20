#!/usr/bin/env bash
set -euo pipefail
directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime="${AIMLABS_QNN_RUNTIME:-$HOME/rbt-work/runtime/yolo/qairt250}"
model="${AIMLABS_MODEL:-$directory/../models/aimlabs-yolo11n-qnn250-fp16-nhwc.bin}"
binary="${AIMLABS_BIN:-$directory/../build/aim-live}"
[[ -x $binary ]] || { printf 'Executable not found: %s\n' "$binary" >&2; exit 1; }
[[ -r $model ]] || { printf 'Model not found: %s\n' "$model" >&2; exit 1; }
if [[ $model != *.onnx ]]; then
  export LD_LIBRARY_PATH="$runtime/lib/aarch64-oe-linux-gcc11.2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ADSP_LIBRARY_PATH="$runtime/lib/hexagon-v73/unsigned;/usr/lib/rfsa/adsp;/dsp${ADSP_LIBRARY_PATH:+;$ADSP_LIBRARY_PATH}"
fi
export OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT=1
# Read-only by default. Require an explicit choice before writing to USB HID.
if (( $# == 0 )); then set -- --hdmi; fi
if [[ $1 == --active ]]; then
  shift
  (( $# == 0 )) || { echo "Usage: $0 --active" >&2; exit 2; }
  if [[ $(< /sys/class/udc/a600000.dwc3/state) != configured || ! -w /dev/hidg0 ]]; then
    echo 'USB HID is not connected or /dev/hidg0 is not writable.' >&2
    exit 1
  fi
  calibration=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["counts_x"], c["counts_y"])' "$directory/gridshot-calibration.json")
  read -r counts_x counts_y <<< "$calibration"
  set -- --hdmi /dev/hidg0 "$counts_x" "$counts_y" 120
fi
if [[ ${1:-} == --hdmi ]]; then
  resolution=$(cat /sys/class/hdmi_in1/lt6911_detect/resolution)
  [[ $resolution =~ ^([0-9]+)x([0-9]+)$ ]] || { printf 'HDMI input has no active resolution
' >&2; exit 1; }
  width=${BASH_REMATCH[1]}
  height=${BASH_REMATCH[2]}
  (( width > 0 && height > 0 )) || { echo 'HDMI input has no active signal.' >&2; exit 1; }
  shift
  if (( width == 1920 && height == 1080 )); then
    set -- --gst-hdmi1080-nv12 "qtiqmmfsrc name=src camera=0 video_0::type=preview src.video_0 ! video/x-raw,format=NV12,width=$width,height=$height,framerate=60/1 ! videoscale ! video/x-raw,format=NV12,width=640,height=360 ! appsink max-buffers=1 drop=true sync=false wait-on-eos=false enable-last-sample=false" "$@"
  else
    set -- --gst "qtiqmmfsrc name=src camera=0 video_0::type=preview src.video_0 ! video/x-raw,format=NV12,width=$width,height=$height,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false" "$@"
  fi
fi
exec "$binary" "$runtime" "$model" "$@"
