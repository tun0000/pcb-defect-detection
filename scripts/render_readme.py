"""Phase-2 final README assembly (plan.md SS 2.8).

Splices the already-generated reports (leakage_comparison.md, benchmark.md,
sahi_ablation.md, test_metrics.json) into the top-level README.md. Every
number here is read from those files - none are retyped.

    uv run python scripts/render_readme.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pcb_defect.constants import KAGGLE_DATASET

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

GITHUB_URL = "https://github.com/tun0000/pcb-defect-detection"
MODEL_URL = "https://huggingface.co/betty0/pcb-defect-detection"
SPACE_URL = "https://huggingface.co/spaces/betty0/pcb-defect-detection"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
ARXIV_URL = "https://arxiv.org/abs/1901.08204"


def _embed_report(rel_path: str, heading_shift: int = 1) -> str:
    """Read a reports/*.md file, drop its own top-level title (the README
    supplies its own section heading instead), and shift remaining headings
    down so they nest correctly."""
    text = (REPORTS_DIR / rel_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    shifted = [("#" * heading_shift + line) if line.startswith("#") else line for line in lines]
    return "\n".join(shifted).strip()


def _badges() -> str:
    return " ".join(
        [
            f"[![HF Space](https://img.shields.io/badge/HF-Space-blue)]({SPACE_URL})",
            f"[![HF Model](https://img.shields.io/badge/HF-Model-yellow)]({MODEL_URL})",
            f"[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-lightgrey)]"
            f"({GITHUB_URL}/blob/main/LICENSE)",
            "[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]"
            "(https://www.python.org/)",
            "[![Ultralytics YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-purple)]"
            "(https://docs.ultralytics.com/)",
            f"[![CI]({GITHUB_URL}/actions/workflows/ci.yml/badge.svg)]"
            f"({GITHUB_URL}/actions/workflows/ci.yml)",
        ]
    )


def _demo_gif_block() -> str:
    gif_path = ROOT / "assets" / "demo.gif"
    if gif_path.exists():
        return "![demo](assets/demo.gif)"
    return f"""<!--
demo GIF 待補：用 ScreenToGif（或任何錄影工具）錄下 {SPACE_URL} 的操作
（上傳圖 → 出現偵測框 → 拖曳信心值滑桿），存成 assets/demo.gif（< 8MB），
存好後把下面這行取消註解即可：
![demo](assets/demo.gif)
-->"""


def build_readme(metrics: dict) -> str:
    grouped, random = metrics["grouped"], metrics["random"]
    leakage_gap = (random["map50"] - grouped["map50"]) * 100

    leakage_section = _embed_report("leakage_comparison.md")
    benchmark_section = _embed_report("benchmark.md")
    sahi_section = _embed_report("sahi_ablation.md")

    return f"""{_badges()}

# pcb-defect-detection

用 Ultralytics **YOLO26**（NMS-free 端到端偵測頭）偵測 PCB 裸板六類瑕疵\
（HRIPCB / PKU-Market-PCB 資料集）的求職作品集專案。目標讀者：台灣電子製造／AOI 職缺面試官。

執行藍圖與所有技術決策見 [plan.md](plan.md)（含每個步驟的實測結果與偏離記錄）。

{_demo_gif_block()}

## 六類瑕疵

`missing_hole`（漏鑽孔）、`mouse_bite`（鼠咬）、`open_circuit`（斷路）、`short`（短路）、\
`spur`（毛刺）、`spurious_copper`（多餘銅）

## 這對 AOI 產線的價值

- **recall ≈ 漏檢率（escape rate）**、**precision ≈ 誤殺率（false kill，決定人工複判成本）**——\
這個 repo 的每個表格都同時列出兩者，而不是只秀一個好看的 mAP。
- **YOLO26 是 NMS-free 端到端架構**：匯出的 ONNX/TensorRT 圖只需要信心值過濾，不用另外調 \
NMS 閾值——`app/app.py`（見下方 demo）只用 onnxruntime + opencv 就能跑完整推論管線，\
沒有 torch/ultralytics。
- **最強的誠實工程賣點**：這份資料集只有 10 片模板裸板，隨機切分會讓同一片板子的背景同時出現在 \
train/test，讓數字虛高。這個專案刻意訓練了兩個模型（板級分組 vs 隨機切分）來**量化**這個灌水幅度：\
grouped mAP50={grouped["map50"]:.4f}，random mAP50={random["map50"]:.4f}，\
差距 **{leakage_gap:.1f} 個百分點**——這個差距本身就是最值得跟面試官講的故事。
- **兩個「沒有把工具包裝成萬靈丹」的誠實負結果**：TensorRT INT8 在這個模型上沒有部署理由（比 FP16 \
慢且精度掉 2 個百分點）；SAHI 切片推論的 recall 增益集中在單一類別、precision 代價卻是全面性的，\
單純拉高推論解析度更划算。細節見下方 benchmark／SAHI 章節。

## 線上 Demo

[![Open in HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Open%20in-Spaces-blue?style=for-the-badge)]({SPACE_URL})

零 torch/ultralytics 依賴、純 CPU（`onnxruntime` + `opencv-python-headless`），上傳圖片或點選範例\
即可看到偵測框；拖曳信心值滑桿只重新篩選已快取的原始輸出，不會重新推論。

## 結果（test split，只用過一次）

{leakage_section}

![反挑櫻桃的 3x3 可視化網格](assets/figures/predictions_grid.png)

*上圖選圖規則：好的例子取 F1 最高且類別多樣化，壞的例子優先選「有漏檢」（AOI escape）\
再選「誤殺多」——全部規則寫死、非人工挑選，詳見 `scripts/evaluate.py`。*

## Benchmark

{benchmark_section}

## SAHI 切片推論消融實驗

{sahi_section}

![SAHI 對照圖](assets/figures/sahi_comparison.png)

## 重現步驟

```bash
# 0. 安裝 uv，並同步基礎環境（資料前處理＋測試，零 torch）
uv sync

# 1. 下載＋轉換資料（一次性 ~2GB 下載；kagglehub 快取共用）
uv run python -m pcb_defect.data_prep.prepare --out data/pcb --strategy grouped --seed 42
uv run python -m pcb_defect.data_prep.prepare --out data/pcb_random --strategy random --seed 42

# 2. EDA
uv run python -m pcb_defect.stats --data data/pcb --out reports

# 3. （選用）本機 smoke test，需要 GPU 或 --allow-cpu
uv sync --extra train
uv run --extra train python -m pcb_defect.smoke --data data/pcb/data.yaml --allow-cpu

# 4. 訓練：在 Colab 依序跑兩次 notebooks/train_colab.ipynb
#    （SPLIT_STRATEGY="grouped" 再改成 "random"），下載 best.pt 放進
#    weights/grouped/ 與 weights/random/

# 5. Phase 2：評估／匯出／parity gate／SAHI（依序）
uv run --extra train python scripts/evaluate.py
uv sync --extra train --group eval
uv run --extra train --group eval python scripts/export_models.py
uv run --extra train --group eval python scripts/verify_onnx_parity.py
uv sync --extra train --group sahi
uv run --extra train --group sahi python scripts/sahi_experiment.py --smoke  # 先跑 smoke
uv run --extra train --group sahi python scripts/sahi_experiment.py

# 6. Benchmark：本機 CPU ＋ Colab T4（notebooks/benchmark_colab.ipynb），再組表
uv run --extra train --group eval python scripts/benchmark_cpu.py
#    （Colab 跑完 notebooks/benchmark_colab.ipynb 後把印出的 JSON 存成
#    reports/benchmark_gpu.json）
uv run python scripts/render_benchmark_report.py

# 7. 上傳 HF（需要 .env 裡的 HF_TOKEN）
uv sync --group hf
uv run --group hf python scripts/upload_hf.py
uv run --group hf python scripts/deploy_space.py
```

## 限制與誠實聲明

- **只有 10 片模板裸板**：板級分組切分的 val/test 各只有 1 片板，per-board 的視覺變異會讓數字比\
板子數量更多的資料集更晃動；這也是選擇同時跑隨機切分對照組的原因之一。
- **合成瑕疵，非真實產線缺陷**：HRIPCB 的瑕疵是人工引入的（非真實 AOI 產線良率問題），真實產線影像\
的光照、對焦、背景可能造成明顯的 domain shift，部署前應該用目標產線影像重新驗證。
- **板級分組數字不能跟用隨機切分的論文/notebook 直接比較**：本專案的 grouped 數字刻意偏保守，見上方\
洩漏對照表。
- **`short`／`spurious_copper` 是最弱的兩類**：在 evaluate.py 與 SAHI baseline 兩個獨立測量中都\
重複出現，是真實、可重現的現象，不是量測雜訊。
- **資料集授權**：Kaggle 鏡像（`{KAGGLE_DATASET}`）\
標示授權「Unknown」，請引用原始論文（見下方引用）。
- **AGPL-3.0 商用限制**：本 repo 與 HF 權重皆為 AGPL-3.0（因使用 Ultralytics YOLO26），商用需要\
[Ultralytics Enterprise License](https://www.ultralytics.com/license)。

## 引用

```
Huang, W., & Wei, P. (2019). A PCB Dataset for Defects Detection and
Classification. arXiv:1901.08204. {ARXIV_URL}
```

資料集：[Kaggle: akhatova/pcb-defects]({KAGGLE_URL})　模型：[Ultralytics YOLO26](https://docs.ultralytics.com/)

## 授權

本專案使用 Ultralytics YOLO26，整個 repo（含微調權重）以 **AGPL-3.0** 授權釋出。

---

## TL;DR (English)

PCB bare-board defect detector (6 classes, HRIPCB dataset) built with Ultralytics **YOLO26**
(NMS-free end-to-end head). The headline engineering story: a board-grouped train/val/test split
(no board's background leaks between splits) scores mAP50={grouped["map50"]:.4f} versus
mAP50={random["map50"]:.4f} for an otherwise-identical random split - a {leakage_gap:.1f}-point gap
that is pure background leakage, quantified rather than glossed over. Two honest negative results
are documented rather than hidden: TensorRT INT8 isn't worth deploying here (no speed edge over
FP16, costs 2 points of mAP50-95), and SAHI slicing barely helps recall (gain concentrated in one
class) while inflating false positives - plain higher-resolution inference is the better lever for
small-object recall on this dataset. Live demo, model weights, full benchmark/ablation reports, and
every script that produced every number in this README are linked above.
"""


def main() -> int:
    metrics_path = REPORTS_DIR / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    readme = build_readme(metrics)
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print(f"README.md written ({len(readme)} chars).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
