"""
compile.py  —  Compile best.onnx → best.hef for Hailo-8 (26 TOPS)

Run this on an x86-64 Linux machine that has hailo-dataflow-compiler installed.
See README.md in this folder for setup instructions.

Usage:
    python compile.py                       # uses ../best.onnx → ../best.hef
    python compile.py --onnx /path/to/best.onnx --out /path/to/best.hef
    python compile.py --calib /path/to/images  # better quantisation accuracy
"""

import argparse
import os
import sys
from pathlib import Path

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_NAME   = "best"
HW_ARCH      = "hailo8"        # Hailo-8 = 26 TOPS
NUM_CLASSES  = 2               # 0=Plant  1=weeds
INPUT_H      = 640
INPUT_W      = 640
BATCH_SIZE   = 1
MAX_DETS     = 100             # max detections per class
# ─────────────────────────────────────────────────────────────────────────────

# ALLS post-processing script injected during parsing.
# This adds the standard YOLOv8 NMS layer directly into the HEF so the
# Pi only receives clean (num_classes, 5, max_dets) FLOAT32 tensors.
ALLS_SCRIPT = f"""
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
nms_postprocess("", max_proposals_per_class={MAX_DETS}, classes={NUM_CLASSES}, \\
                regression_length=15, \\
                background_removal=False, \\
                background_removal_index=0)
"""


def parse_args():
    here = Path(__file__).resolve().parent
    root = here.parent

    p = argparse.ArgumentParser(description="Compile best.onnx to best.hef")
    p.add_argument("--onnx",  default=str(root / "best.onnx"),
                   help="Path to input ONNX model")
    p.add_argument("--out",   default=str(root / "best.hef"),
                   help="Path for output HEF file")
    p.add_argument("--calib", default=None,
                   help="(optional) Folder of calibration images for INT8 "
                        "quantisation. If omitted, random data is used "
                        "(lower accuracy).")
    p.add_argument("--har-dir", default=str(here),
                   help="Directory to write intermediate .har files")
    return p.parse_args()


def check_dfc():
    try:
        from hailo_sdk_client import ClientRunner  # noqa: F401
    except ImportError:
        print("ERROR: hailo-dataflow-compiler is not installed.")
        print("  Install from https://hailo.ai/developer-zone/")
        print("  pip install hailo_dataflow_compiler-*.whl")
        sys.exit(1)


def compile_model(onnx_path: str, hef_path: str,
                  calib_dir: str | None, har_dir: str):
    from hailo_sdk_client import ClientRunner, InferenceContext  # noqa: F401

    onnx_path = str(Path(onnx_path).resolve())
    hef_path  = str(Path(hef_path).resolve())
    har_dir   = Path(har_dir).resolve()
    har_dir.mkdir(parents=True, exist_ok=True)

    parsed_har = str(har_dir / f"{MODEL_NAME}_parsed.har")
    optim_har  = str(har_dir / f"{MODEL_NAME}_optimized.har")

    # ── Step 1: Parse ONNX ───────────────────────────────────────────────────
    print(f"\n[1/3] Parsing {onnx_path} …")
    runner = ClientRunner(hw_arch=HW_ARCH)
    runner.translate_onnx_model(
        onnx_path,
        MODEL_NAME,
        net_input_shapes={"images": [BATCH_SIZE, 3, INPUT_H, INPUT_W]},
        alls_script=ALLS_SCRIPT,
    )
    runner.save_har(parsed_har)
    print(f"      Saved: {parsed_har}")

    # ── Step 2: Optimize (quantise) ──────────────────────────────────────────
    print("\n[2/3] Optimising (INT8 quantisation) …")
    runner = ClientRunner(hw_arch=HW_ARCH, har=parsed_har)

    if calib_dir:
        import numpy as np
        import cv2  # pip install opencv-python

        calib_path = Path(calib_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = [p for p in calib_path.iterdir() if p.suffix.lower() in exts]
        if not imgs:
            print(f"  WARNING: No images found in {calib_dir}. "
                  "Falling back to random data.")
            calib_data = None
        else:
            print(f"  Loading {len(imgs)} calibration images …")
            data = []
            for img_p in imgs[:200]:        # cap at 200
                img = cv2.imread(str(img_p))
                img = cv2.resize(img, (INPUT_W, INPUT_H))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append(img.astype(np.float32))
            calib_data = np.stack(data)     # (N, H, W, 3)
    else:
        calib_data = None

    if calib_data is not None:
        runner.optimize(calib_data)
    else:
        print("  No calibration data — using random data "
              "(may reduce detection accuracy slightly).")
        runner.optimize_full_precision()

    runner.save_har(optim_har)
    print(f"      Saved: {optim_har}")

    # ── Step 3: Compile → HEF ────────────────────────────────────────────────
    print("\n[3/3] Compiling to HEF …")
    runner = ClientRunner(hw_arch=HW_ARCH, har=optim_har)
    hef_bytes = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef_bytes)
    size_mb = len(hef_bytes) / 1024 / 1024
    print(f"      Saved: {hef_path}  ({size_mb:.1f} MB)")
    print("\nDone! Copy best.hef to the Pi project directory and run:")
    print("  python weed_detection_hailo.py")


if __name__ == "__main__":
    args = parse_args()
    check_dfc()
    compile_model(args.onnx, args.out, args.calib, args.har_dir)
