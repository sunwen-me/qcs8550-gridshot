import itertools
import json
import math
import os
from pathlib import Path
import select
import signal
import statistics
import struct
import subprocess
import time

root = Path(__file__).resolve().parent
assert Path("/sys/class/udc/a600000.dwc3/state").read_text().strip() == "configured"
with (root / "gridshot-calibration.err").open("w") as log:
    process = subprocess.Popen(
        ["bash", str(root / "run-live.sh"), "--hdmi"], stdout=subprocess.PIPE,
        stderr=log, bufsize=0,
    )
    mouse = None
    try:
        fd = process.stdout.fileno()
        buffer = b""

        def read_frame(deadline):
            global buffer
            while time.monotonic() < deadline:
                if b"\n" not in buffer:
                    if not select.select([fd], [], [], max(0, deadline-time.monotonic()))[0]:
                        break
                    data = os.read(fd, 8192)
                    if not data:
                        break
                    buffer += data
                    continue
                line, buffer = buffer.split(b"\n", 1)
                if line.startswith(b'{"detections"'):
                    return json.loads(line)
            raise RuntimeError("No further frames from HDMI")

        def points(frame):
            return [(x, y) for x, y, score in frame["centers"] if score >= 0.55]

        def align(previous, new):
            return min(itertools.permutations(new), key=lambda sequence: sum(
                math.dist(a, b)**2 for a, b in zip(previous, sequence)))

        def wait_stable():
            deadline = time.monotonic() + 6
            stable = 0
            previous = []
            while time.monotonic() < deadline:
                new = points(read_frame(deadline))
                if len(new) == len(previous) == 3:
                    deltas = [math.dist(a, b) for a, b in zip(previous, align(previous, new))]
                    stable = stable + 1 if max(deltas) <= 3 else 0
                else:
                    stable = 0
                previous = new
                if stable >= 4:
                    return new
            raise RuntimeError("Need three still Gridshot targets before moving the mouse")

        def pulse(x, y):
            assert os.write(mouse, struct.pack("Bbb", 0, x, y)) == 3

        def measure(axis):
            baseline = wait_stable()
            step = 16
            vectors = [(step, 0), (0, step)]
            vector = vectors[axis]
            pulse(*vector)
            started = time.monotonic()
            deadline = started + 0.6
            measurements = []
            try:
                while time.monotonic() < deadline and len(measurements) < 3:
                    new = points(read_frame(deadline))
                    if len(new) != 3:
                        continue
                    new = align(baseline, new)
                    deltas = [b[axis]-a[axis] for a,b in zip(baseline,new)]
                    offset = statistics.median(deltas)
                    if 3 <= abs(offset) <= 180 and max(abs(d-offset) for d in deltas) <= 12:
                        measurements.append((offset, 1000*(time.monotonic()-started)))
            finally:
                pulse(*(-v for v in vector))
            if len(measurements) != 3:
                raise RuntimeError(f"No reliable Gridshot response on axis {axis}")
            shift = statistics.median(x[0] for x in measurements)
            response_ms = measurements[0][1]
            print(f"axis {axis}: {step} counts -> {shift:.2f} px, observed in {response_ms:.1f} ms", flush=True)
            return -step/shift, response_ms

        wait_stable()
        mouse = os.open("/dev/hidg0", os.O_WRONLY|os.O_NONBLOCK)
        x, latency_x = measure(0)
        y, latency_y = measure(1)
        if min(abs(x),abs(y)) < 0.08 or max(abs(x),abs(y)) > 5:
            raise RuntimeError("Mouse response fell outside the calibration range")
        feedback = round(min(250,max(35,latency_x+10,latency_y+10)))
        result={"counts_x":round(x,4),"counts_y":round(y,4),"feedback_ms":feedback}
        (root/"gridshot-calibration.json").write_text(json.dumps(result)+"\n")
        print(json.dumps(result),flush=True)
    finally:
        if mouse is not None:
            os.close(mouse)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
