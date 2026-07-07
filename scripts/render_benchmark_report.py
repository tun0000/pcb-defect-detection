"""Phase-2 benchmark report: merges CPU (local) + GPU (Colab T4) results into
reports/benchmark.md.

Depends on three JSON files already existing:
- reports/benchmark_cpu.json      (scripts/benchmark_cpu.py)
- reports/benchmark_gpu.json      (notebooks/benchmark_colab.ipynb's final cell,
                                    pasted back and saved verbatim to this path)
- reports/export_fidelity.json    (scripts/export_models.py - source of the ONNX
                                    fidelity numbers the CPU rows share, since both
                                    CPU configs run the identical best.onnx artifact)

All numbers below are read from those files, never retyped - see plan.md SS 2.4.

    uv run python scripts/render_benchmark_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
CPU_JSON = REPORTS_DIR / "benchmark_cpu.json"
GPU_JSON = REPORTS_DIR / "benchmark_gpu.json"
FIDELITY_JSON = REPORTS_DIR / "export_fidelity.json"

FIDELITY_THRESHOLD = 0.02  # same 2-point bar used in export_models.py


def _row(backend: str, precision: str, device: str, stats: dict, map5095: float) -> str:
    return (
        f"| {backend} | {precision} | {device} | {stats['p50_ms']:.2f} "
        f"| {stats['p95_ms']:.2f} | {stats['fps_from_p50']:.2f} | {map5095:.4f} |"
    )


def main() -> int:
    for path, hint in (
        (CPU_JSON, "run scripts/benchmark_cpu.py first"),
        (GPU_JSON, "run notebooks/benchmark_colab.ipynb on Colab T4, save its JSON here"),
        (FIDELITY_JSON, "run scripts/export_models.py first"),
    ):
        if not path.exists():
            print(f"ERROR: {path} not found - {hint}")
            return 1

    cpu = json.loads(CPU_JSON.read_text(encoding="utf-8"))
    gpu = json.loads(GPU_JSON.read_text(encoding="utf-8"))
    fidelity = json.loads(FIDELITY_JSON.read_text(encoding="utf-8"))

    onnx_map50 = fidelity["onnx"]["map50"]
    onnx_map5095 = fidelity["onnx"]["map5095"]

    pt = gpu["results"]["pytorch_fp32_t4"]
    fp16 = gpu["results"]["tensorrt_fp16_t4"]
    int8 = gpu["results"]["tensorrt_int8_t4"]
    cpu_unrestricted = cpu["configs"]["ort_cpu_unrestricted"]
    cpu_2thread = cpu["configs"]["ort_cpu_2thread"]

    rows = [
        _row("ONNX Runtime", "fp32", "CPU（本機，全執行緒）", cpu_unrestricted, onnx_map5095),
        _row("ONNX Runtime", "fp32", "CPU（本機，2-thread，HF proxy）", cpu_2thread, onnx_map5095),
        _row("PyTorch", "fp32", "T4 GPU（Colab）", pt, pt["map5095"]),
        _row("TensorRT", "fp16", "T4 GPU（Colab）", fp16, fp16["map5095"]),
        _row("TensorRT", "int8", "T4 GPU（Colab）", int8, int8["map5095"]),
    ]

    fp16_fps_gain = (fp16["fps_from_p50"] / pt["fps_from_p50"] - 1) * 100
    int8_fps_gain = (int8["fps_from_p50"] / pt["fps_from_p50"] - 1) * 100
    int8_vs_fp16_speed = (int8["fps_from_p50"] / fp16["fps_from_p50"] - 1) * 100
    int8_map_delta = int8["map5095"] - pt["map5095"]
    fp16_map_delta = fp16["map5095"] - pt["map5095"]
    cpu_vs_t4_pt = (cpu_unrestricted["fps_from_p50"] / pt["fps_from_p50"] - 1) * 100

    lines = [
        "# Benchmark 對照表\n",
        "本機 CPU（ONNX Runtime）與 Colab T4 GPU（PyTorch / TensorRT FP16 / TensorRT INT8）"
        "五個後端的延遲與精度對照。方法學一致：100 張固定 test 圖、30 次 warmup、"
        "循環 2 輪＝200 次計時推論、batch=1、`time.perf_counter` 量端到端"
        "（前處理＋推論＋後處理），GPU 端每次呼叫前後都插 `torch.cuda.synchronize()` barrier。"
        "詳見 `scripts/benchmark_cpu.py` / `notebooks/benchmark_colab.ipynb`（plan.md SS 2.4）。\n",
        "| backend | precision | device | p50 (ms) | p95 (ms) | FPS (1/p50) | mAP50-95 |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## 精度-速度取捨\n",
        f"- **TensorRT FP16 vs PyTorch FP32（T4）**：快 {fp16_fps_gain:+.1f}%，"
        f"mAP50-95 幾乎不變（{fp16_map_delta:+.4f}）—— FP16 在這個模型上幾乎是「免費的加速」。",
        f"- **TensorRT INT8 vs PyTorch FP32（T4）**：快 {int8_fps_gain:+.1f}%，"
        f"但 mAP50-95 掉了 {abs(int8_map_delta):.4f}"
        + (
            "（超過 2 個百分點門檻，同 export_models.py）。"
            if abs(int8_map_delta) > FIDELITY_THRESHOLD
            else "。"
        ),
        f"- **INT8 vs FP16（T4）**：INT8 並沒有比 FP16 快（{int8_vs_fp16_speed:+.1f}%，"
        "在雜訊範圍內，甚至略慢），卻要犧牲更多精度——這個模型/硬體組合下，"
        "**INT8 被 FP16 全面壓過，沒有部署理由**。這不是預期中「量化一定更快」的結果，"
        "但實測數字就是如此，如實記錄。",
        f"- **本機 CPU（全執行緒）vs T4 PyTorch FP32**：本機 CPU 反而快 {cpu_vs_t4_pt:+.1f}%。"
        "這不是公平的硬體成本對比（筆電 CPU vs 資料中心 GPU），但對「batch=1 單張推論」"
        "這個部署場景來說是真實可信的數字：模型夠小、單張推論的 GPU 呼叫額外開銷"
        "（kernel 啟動、PCIe 傳輸）稀釋掉了 GPU 的算力優勢，GPU 的優勢通常要更大 batch "
        "或更大模型才會顯現。這也支持這個專案把 Hugging Face Space 部署在免費 CPU 層"
        "的選擇——不只是省錢，這個工作負載形狀下 CPU 本來就是合理選項。",
        "",
        "## 誠實聲明\n",
        f"- **本機 CPU**：{cpu['hardware']['cpu']}，{cpu['hardware']['logical_cores']} 邏輯核心，"
        f"{cpu['hardware']['platform']}（`scripts/benchmark_cpu.py` 實測）。",
        f"- **Colab T4**：{gpu['hardware']['colab_tier']}，"
        f"ultralytics {gpu['hardware']['ultralytics_version']}、"
        f"torch {gpu['hardware']['torch_version']}、"
        f"TensorRT {gpu['hardware']['tensorrt_version']}"
        "（`notebooks/benchmark_colab.ipynb` 實測）。",
        "- 這些數字**不能**直接拿來跟 Ultralytics 官方發布的 T4 benchmark 數字比較："
        "batch size、TensorRT/驅動版本、量測方式（端到端 vs 純推論）都可能不同。",
        "- 「2-thread proxy」只是粗略模擬 Hugging Face 免費 CPU-Basic 層（2 vCPU），"
        "不是在該租戶上實測——真正的 Space 上線後延遲可能不同。",
        "- INT8 的校準資料來自 `data.yaml` 的 train split（`fraction=1.0`，官方預設值），"
        "與這裡拿來算 fidelity 的 test split 分開，calibration 沒有碰過 test 資料。",
        f"- ONNX（CPU 兩列共用）mAP50={onnx_map50:.4f}／PyTorch T4 mAP50={pt['map50']:.4f}："
        "兩者本該相同（同一組 `.pt` 權重、同一份 test split），實際些微差異"
        "（{:.4f}）已在 `reports/export_fidelity.json` 記錄並判斷為 FP32 ONNX 匯出的"
        "已知數值特性，不是新問題。".format(abs(onnx_map50 - pt["map50"])),
    ]

    (REPORTS_DIR / "benchmark.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    print("\n".join(lines))
    print("\nreports/benchmark.md written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
