# Compile best.onnx → best.hef

The Hailo Dataflow Compiler (DFC) is **x86-64 Linux only** and cannot run on
the Pi's ARM64 CPU. Use one of the options below.

---

## Option A — Google Colab (easiest, free)

1. Open `colab_compile.ipynb` in Google Colab
2. Get the Hailo DFC wheel from https://hailo.ai/developer-zone/
   - Free account required
   - Downloads → Hailo Dataflow Compiler → Python 3.10 wheel
3. Upload `best.onnx` and the `.whl` when prompted
4. Run all cells → download `best.hef`
5. Copy `best.hef` to the project root on the Pi

---

## Option B — Docker on any x86-64 machine

```bash
# 1. Download the DFC wheel from https://hailo.ai/developer-zone/
#    Put it in this hailo_compile/ directory.

# 2. From the project root:
docker build -t hailo-compile hailo_compile/

docker run --rm \
  -v "$(pwd)/best.onnx:/work/best.onnx:ro" \
  -v "$(pwd):/out" \
  hailo-compile

# best.hef will appear in the project root.
```

---

## Option C — Bare x86-64 Linux machine

```bash
# Python 3.8–3.10 required on x86-64 Linux

# 1. Get wheel from https://hailo.ai/developer-zone/
pip install hailo_dataflow_compiler-*.whl
pip install opencv-python numpy

# 2. From the project root:
python hailo_compile/compile.py

# Optional — pass calibration images for better INT8 accuracy:
python hailo_compile/compile.py --calib /path/to/weed_images/

# Outputs: best.hef in the project root
```

---

## After compilation

Copy `best.hef` to the project root on the Pi:

```bash
scp best.hef pi@<PI_IP>:"/home/dev/Weeding robot 2.0/best.hef"
```

Then run:

```bash
source "/home/dev/Weeding robot 2.0/venv/bin/activate"
python "/home/dev/Weeding robot 2.0/weed_detection_hailo.py"
```

---

## Notes

- **Calibration images**: The `--calib` option accepts a folder of `.jpg`/`.png`
  images. Use a representative mix of weed and plant images from your field.
  Without calibration, random data is used which may slightly reduce accuracy.
- **Hardware target**: `hailo8` (26 TOPS). This is the Hailo-8 chip on the
  AI Hat+ (not Hailo-8L which is 13 TOPS).
- **Classes**: `{0: 'Plant', 1: 'weeds'}` — must match training order.
