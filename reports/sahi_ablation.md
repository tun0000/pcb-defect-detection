# SAHI 切片推論消融實驗

三種推論策略在同一份 test split（120 張圖，全部來自板號 04）上的 recall/precision 對照：baseline（整張圖 imgsz=640）、SAHI（640 切片／0.2 重疊）、hires（整張圖 imgsz=1280，回答「SAHI 的增益是不是只靠解析度」）。詳見 `scripts/sahi_experiment.py`（plan.md SS 2.5）。

## 整體指標

| arm | recall | precision | mean sec/img |
|---|---|---|---|
| baseline (imgsz=640) | 0.7654 | 0.4215 | 0.187 |
| SAHI (640 slices / 0.2 overlap) | 0.7737 | 0.3424 | 4.688 |
| hires (imgsz=1280) | 0.7682 | 0.8540 | 0.417 |

## 每類別 recall

| class | baseline (imgsz=640) | SAHI (640 slices / 0.2 overlap) | hires (imgsz=1280) |
|---|---|---|---|
| missing_hole | 0.9833 | 0.9833 | 1.0000 |
| mouse_bite | 0.9000 | 0.9000 | 0.9333 |
| open_circuit | 0.7288 | 0.7288 | 0.8475 |
| short | 0.6102 | 0.6102 | 0.3729 |
| spur | 0.7500 | 0.8000 | 0.8667 |
| spurious_copper | 0.6167 | 0.6167 | 0.5833 |

## GT 面積三分位分桶 recall（換算至 imgsz=640，桶界線 180px² / 243px²）

COCO 的絕對面積分桶（small/medium/large）對這個資料集無意義——全部瑕疵都遠小於 COCO 的門檻，所以改用資料集自己的相對三分位。

| bucket | baseline (imgsz=640) | SAHI (640 slices / 0.2 overlap) | hires (imgsz=1280) |
|---|---|---|---|
| 小 (bottom third) | 0.8000 | 0.8250 | 0.9250 |
| 中 (middle third) | 0.8487 | 0.8487 | 0.8739 |
| 大 (top third) | 0.6471 | 0.6471 | 0.5042 |

## 分析

- **SAHI 抓到的額外 TP 高度集中在單一類別**：SAHI 比 baseline 多抓到 3 個 GT，全部集中在「spur」（其餘類別 recall 完全沒變），代價是多了 156 個 FP（分散在所有類別）——換算成 AOI 的說法：多背 156 次誤殺（人工複判成本）只換到 3 次少漏檢，這筆帳不划算。
- **hires 在小物件桶的挽回幅度比 SAHI 大，成本卻低很多**：小三分位 recall baseline 0.8000 → hires 0.9250（+0.1250）vs SAHI 0.8250（+0.0250）；hires 只慢 2.2x（SAHI 慢 25.1x），precision 也遠高於 SAHI （0.8540 vs 0.3424）。
- **結論：這個資料集/模型組合下，SAHI 不值得部署**——recall 增益小且集中在少數類別，precision 代價（AOI 誤殺率／人工複判成本）卻是全面性的，跑起來還慢一個數量級；如果目標是挽回小物件 recall，提高推論解析度（hires）是更划算的選擇。
- **一個留意但未深究的反例**：hires 在 short 類別的 recall 反而下降（baseline 0.6102 → hires 0.3729），跟其他類別的走勢相反——可能與 640 訓練／1280 推論的解析度不匹配有關，但這裡沒有進一步驗證，如實記錄，不做未經檢驗的定論。
- **單一板號的限制**：這份 test split 全部來自板號 04（板級分組切分的既定結果），所以「每板秒數」等同「每張圖秒數」，沒有另外做逐板拆解的意義；上表的 mean sec/img 已經是這批圖唯一有意義的時間單位。

## 對照圖

`assets/figures/sahi_comparison.png` — 04_spur_05：baseline 漏掉的一個瑕疵（GT area@640 ≈ 100px²，三分位裡最小的一批），SAHI 抓到了。挑選規則：所有「baseline 漏、SAHI 抓到」的候選中，取 GT 面積最小的一個（決定性選圖，不是挑最好看的）。
