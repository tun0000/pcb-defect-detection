# Benchmark 對照表

本機 CPU（ONNX Runtime）與 Colab T4 GPU（PyTorch / TensorRT FP16 / TensorRT INT8）五個後端的延遲與精度對照。方法學一致：100 張固定 test 圖、30 次 warmup、循環 2 輪＝200 次計時推論、batch=1、`time.perf_counter` 量端到端（前處理＋推論＋後處理），GPU 端每次呼叫前後都插 `torch.cuda.synchronize()` barrier。詳見 `scripts/benchmark_cpu.py` / `notebooks/benchmark_colab.ipynb`（plan.md SS 2.4）。

| backend | precision | device | p50 (ms) | p95 (ms) | FPS (1/p50) | mAP50-95 |
|---|---|---|---|---|---|---|
| ONNX Runtime | fp32 | CPU（本機，全執行緒） | 81.48 | 86.38 | 12.27 | 0.3867 |
| ONNX Runtime | fp32 | CPU（本機，2-thread，HF proxy） | 141.31 | 153.42 | 7.08 | 0.3867 |
| PyTorch | fp32 | T4 GPU（Colab） | 102.87 | 152.84 | 9.72 | 0.3881 |
| TensorRT | fp16 | T4 GPU（Colab） | 77.71 | 119.78 | 12.87 | 0.3887 |
| TensorRT | int8 | T4 GPU（Colab） | 78.96 | 122.94 | 12.66 | 0.3681 |

## 精度-速度取捨

- **TensorRT FP16 vs PyTorch FP32（T4）**：快 +32.4%，mAP50-95 幾乎不變（+0.0005）—— FP16 在這個模型上幾乎是「免費的加速」。
- **TensorRT INT8 vs PyTorch FP32（T4）**：快 +30.2%，但 mAP50-95 掉了 0.0200（超過 2 個百分點門檻，同 export_models.py）。
- **INT8 vs FP16（T4）**：INT8 並沒有比 FP16 快（-1.6%，在雜訊範圍內，甚至略慢），卻要犧牲更多精度——這個模型/硬體組合下，**INT8 被 FP16 全面壓過，沒有部署理由**。這不是預期中「量化一定更快」的結果，但實測數字就是如此，如實記錄。
- **本機 CPU（全執行緒）vs T4 PyTorch FP32**：本機 CPU 反而快 +26.2%。這不是公平的硬體成本對比（筆電 CPU vs 資料中心 GPU），但對「batch=1 單張推論」這個部署場景來說是真實可信的數字：模型夠小、單張推論的 GPU 呼叫額外開銷（kernel 啟動、PCIe 傳輸）稀釋掉了 GPU 的算力優勢，GPU 的優勢通常要更大 batch 或更大模型才會顯現。這也支持這個專案把 Hugging Face Space 部署在免費 CPU 層的選擇——不只是省錢，這個工作負載形狀下 CPU 本來就是合理選項。

## 誠實聲明

- **本機 CPU**：Intel64 Family 6 Model 154 Stepping 3, GenuineIntel，16 邏輯核心，Windows-11-10.0.26200-SP0（`scripts/benchmark_cpu.py` 實測）。
- **Colab T4**：Colab Pro (T4 runtime)，ultralytics 8.4.89、torch 2.11.0+cu128、TensorRT 11.1.0.106（`notebooks/benchmark_colab.ipynb` 實測）。
- 這些數字**不能**直接拿來跟 Ultralytics 官方發布的 T4 benchmark 數字比較：batch size、TensorRT/驅動版本、量測方式（端到端 vs 純推論）都可能不同。
- 「2-thread proxy」只是粗略模擬 Hugging Face 免費 CPU-Basic 層（2 vCPU），不是在該租戶上實測——真正的 Space 上線後延遲可能不同。
- INT8 的校準資料來自 `data.yaml` 的 train split（`fraction=1.0`，官方預設值），與這裡拿來算 fidelity 的 test split 分開，calibration 沒有碰過 test 資料。
- ONNX（CPU 兩列共用）mAP50=0.8094／PyTorch T4 mAP50=0.8390：兩者本該相同（同一組 `.pt` 權重、同一份 test split），實際些微差異（0.0296）已在 `reports/export_fidelity.json` 記錄並判斷為 FP32 ONNX 匯出的已知數值特性，不是新問題。
