# 變更紀錄

本檔案記錄 `voice-spectral-classifier` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

### 新增

- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。
- 流程圖分英文與繁體中文兩版：英文版是 `docs/workflow.html` 與 `docs/images/workflow.png`，繁體中文版是 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。

### 修正

- 流程圖「全資料訓練」節點的模型檔名：預設的 CNN 存成 `model.pt`，MLP 與 logistic regression 存成 `model.joblib`。

---

## [0.1.0] - 2026-10-09

### 新增

- 首次釋出。
- 錄音前處理：讀入 WAV 後以多相濾波器重取樣到統一的 `segmentation.sample_rate`，可選擇以端點偵測去除頭尾靜音，再切成固定長度（預設 1 秒）的片段；不足一段的尾巴依 `min_fraction` 補零保留或捨棄，並可用 `max_segments` 均勻挑選片段。
- 40 × 98 的 8 位元灰階頻譜圖：毫秒為單位的分析參數、Hamming 等四種窗函數、單邊功率譜密度、等寬（`linear`）或 mel 頻帶、相對於片段峰值的 dB 與線性量化，第 0 列是最低頻。
- 六層卷積的頻譜圖 CNN（batch normalization、dropout、時間軸全域 max pooling、訓練集平均影像存為 buffer），包成 scikit-learn 相容的估計器；PyTorch 放在 `dl` extra，延遲匯入。
- 不需要 PyTorch 的基線模型：標準化（可選 PCA）加 MLP 或 logistic regression；`model.class_weight = balanced` 以加權損失或少數類別過取樣處理類別不平衡。
- 分層、分組的 k 折交叉驗證（同一位受試者的錄音不跨折），同時回報片段層級與錄音層級（分數平均）的 UAR、accuracy、macro F1、one-vs-rest macro ROC-AUC、每類 recall／specificity／precision／F1 與混淆矩陣。
- typer CLI：`synthesize`、`index`、`prepare`、`evaluate`、`train`、`predict`；除了 `index` 以外的指令每次執行都寫出 `config.yaml` 與 `manifest.json`，資料檔在同一環境下逐位元組可重現。
- 三類合成持續母音（`breathy`、`modal`、`rough`），以多種取樣率與頭尾靜音寫出，與真實資料同樣的磁碟格式。
- 範例腳本：端到端流程、linear 與 mel 頻譜圖在各模型上的比較、片段層級與錄音層級分數的對照。
