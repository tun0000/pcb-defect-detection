---
title: PCB 裸板瑕疵偵測 - YOLO26 Demo
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
python_version: "3.11"
license: agpl-3.0
models:
  - tun0000/pcb-defect-detection
tags:
  - object-detection
  - pcb
  - defect-detection
  - yolo
short_description: YOLO26 對 PCB 裸板 6 類瑕疵的 NMS-free 即時偵測 demo
pinned: false
---

# PCB 裸板瑕疵偵測 — YOLO26 Demo

用 [Ultralytics YOLO26](https://docs.ultralytics.com/) 的 NMS-free e2e 偵測頭，對 PCB 裸板影像抓 6 類瑕疵：
`missing_hole`、`mouse_bite`、`open_circuit`、`short`、`spur`、`spurious_copper`。

- 完整專案（訓練、評估、benchmark、SAHI 消融實驗）：[GitHub repo](https://github.com/tun0000/pcb-defect-detection)
- 模型權重與 model card：見上方 `models:` 連結

## 這個 Space 怎麼跑

純 CPU、零 torch/ultralytics 依賴——只用 ONNX Runtime 做推論。上傳圖片後，模型只跑一次推論；
拖曳信心值滑桿只是重新篩選同一份已快取的原始輸出，不會重新呼叫模型。

## 已知限制

- 訓練資料（HRIPCB）只有 10 片模板裸板，這個 demo 的模型是在其中 8 片上訓練、於「沒看過的板子」
  評估，數字比隨機切分（會有背景洩漏）保守但更誠實——細節見 GitHub repo 的 `plan.md`／`reports/`。
- 真實產線影像的光照、對焦、背景可能與這份資料集差異很大（domain shift），實際部署前應該用
  目標產線的影像重新驗證。

## 授權

程式碼與權重皆為 **AGPL-3.0**（因使用 Ultralytics YOLO26）。商用需求請參考
[Ultralytics Enterprise License](https://www.ultralytics.com/license)。
