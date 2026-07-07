"""Phase-2 CPU latency benchmark for the exported ONNX model.

GPU backends (PyTorch FP32, TensorRT FP16/INT8) run on Colab T4 instead - see
notebooks/benchmark_colab.ipynb. This script produces the two CPU rows of the
final reports/benchmark.md table: unrestricted ONNX Runtime CPU threading
(this machine's default thread pool), and a 2-thread configuration as a rough
proxy for Hugging Face's free CPU-Basic Space tier (2 vCPU).

    uv run --extra train --group eval python scripts/benchmark_cpu.py
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pcb_defect.e2e_onnx import OnnxYoloModel

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "exports" / "best.onnx"
TEST_IMAGES_DIR = ROOT / "data" / "pcb" / "images" / "test"
REPORTS_DIR = ROOT / "reports"
FIDELITY_PATH = REPORTS_DIR / "export_fidelity.json"

N_IMAGES = 100
N_CYCLES = 2  # -> 200 timed inferences, per plan.md SS 2.4
WARMUP = 30
CONF = 0.25

# intra_op=0 leaves onnxruntime's SessionOptions untouched (its own auto-detected
# default thread pool for this machine). intra_op=2 caps both intra- and inter-op
# threads to approximate a 2 vCPU box - not a measurement ON that tier, just a proxy.
CONFIGS = [
    {"tag": "ort_cpu_unrestricted", "label": "ORT CPU (unrestricted)", "intra_op": 0},
    {"tag": "ort_cpu_2thread", "label": "ORT CPU (2-thread, HF free-tier proxy)", "intra_op": 2},
]


def _load_images() -> list[Image.Image]:
    paths = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:N_IMAGES]
    if len(paths) < N_IMAGES:
        print(f"WARNING: only {len(paths)} test images available (wanted {N_IMAGES})")
    # Decode fully into RAM up front - the timed loop below measures inference,
    # not disk I/O or JPEG decode.
    return [Image.open(p).convert("RGB").copy() for p in paths]


def _make_model(intra_op: int) -> OnnxYoloModel:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if intra_op:
        opts.intra_op_num_threads = intra_op
        opts.inter_op_num_threads = 1
    return OnnxYoloModel(ONNX_PATH, sess_options=opts)


def _time_config(model: OnnxYoloModel, images: list[Image.Image]) -> dict:
    for i in range(WARMUP):
        model.predict(images[i % len(images)], conf=CONF)

    n_runs = len(images) * N_CYCLES
    latencies_ms = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        model.predict(image, conf=CONF)  # end-to-end: preprocess + session.run + postprocess
        latencies_ms[i] = (time.perf_counter() - t0) * 1000

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return {
        "n_runs": n_runs,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "mean_ms": round(float(latencies_ms.mean()), 3),
        "fps_from_p50": round(1000.0 / p50, 2),
    }


def main() -> int:
    import onnxruntime as ort

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not ONNX_PATH.exists():
        print(f"ERROR: {ONNX_PATH} not found - run scripts/export_models.py first")
        return 1

    fidelity_delta = None
    if FIDELITY_PATH.exists():
        fidelity_delta = json.loads(FIDELITY_PATH.read_text(encoding="utf-8"))["delta_map5095"]

    images = _load_images()
    print(f"loaded {len(images)} test images into RAM")
    print(f"onnxruntime {ort.__version__}")

    hardware = {
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "logical_cores": os.cpu_count(),
    }

    results = {}
    header = f"{'config':<40}{'p50 (ms)':>10}{'p95 (ms)':>10}{'FPS':>8}"
    print(header)
    for cfg in CONFIGS:
        model = _make_model(cfg["intra_op"])
        stats = _time_config(model, images)
        results[cfg["tag"]] = {
            "label": cfg["label"],
            "intra_op_num_threads": cfg["intra_op"],
            **stats,
        }
        print(
            f"{cfg['label']:<40}{stats['p50_ms']:>10.3f}"
            f"{stats['p95_ms']:>10.3f}{stats['fps_from_p50']:>8.2f}"
        )

    report = {
        "backend": "onnxruntime",
        "precision": "fp32",
        "model": str(ONNX_PATH.relative_to(ROOT)),
        "n_images": len(images),
        "n_cycles": N_CYCLES,
        "warmup": WARMUP,
        "conf": CONF,
        "map5095_fidelity_delta": fidelity_delta,
        "hardware": hardware,
        "configs": results,
        "note": (
            "Timed end-to-end (preprocess + ONNX Runtime session.run + postprocess) via "
            "time.perf_counter, matching what a real CPU deployment (e.g. this project's "
            "Hugging Face Space) actually pays per request rather than just the graph- "
            "execution step. mAP50-95 fidelity is inherited from reports/export_fidelity.json: "
            "both configs run the identical best.onnx artifact, so thread count changes "
            "latency, not accuracy. The 2-thread config is a rough proxy for HF's free "
            "CPU-Basic tier (2 vCPU), not a measurement taken on that tier directly."
        ),
    }
    (REPORTS_DIR / "benchmark_cpu.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    print("\nreports/benchmark_cpu.json written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
