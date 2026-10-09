# voice-spectral-classifier

[![CI](https://github.com/recklight/voice-spectral-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/recklight/voice-spectral-classifier/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

以固定大小的頻譜圖與卷積神經網路（Convolutional Neural Network, CNN）分類病理嗓音的研究工具。

本專案（Python 套件 `voice_spectral_classifier`，命令列工具 `voice-spectral-classifier`）把持續母音的錄音重取樣、去除頭尾靜音並切成 1 秒片段，每個片段算成 40 × 98 的灰階頻譜圖，再交給小型 CNN 分類；沒有 PyTorch 時也能用 MLP 或 logistic regression 當基線。評估採分層、分組的 k 折交叉驗證，片段層級與錄音層級各自回報未加權平均召回率（unweighted average recall, UAR）、混淆矩陣與各類指標。

作者：RL

---

## 目錄

- [專案簡介](#專案簡介)
- [方法說明](#方法說明)
- [安裝步驟](#安裝步驟)
- [快速開始](#快速開始)
- [使用範例](#使用範例)
- [CLI 指令對照表](#cli-指令對照表)
- [專案結構](#專案結構)
- [資料準備](#資料準備)
- [設定檔](#設定檔)
- [結果與評估指標](#結果與評估指標)
- [已知限制](#已知限制)
- [設計重點](#設計重點)
- [開發與測試](#開發與測試)
- [References](#references)
- [License](#license)

---

## 專案簡介

[![voice-spectral-classifier 流程圖](docs/images/workflow.png)](https://raw.githack.com/recklight/voice-spectral-classifier/master/docs/workflow.html?theme=dark)

互動式流程圖：[線上瀏覽](https://raw.githack.com/recklight/voice-spectral-classifier/master/docs/workflow.html?theme=dark)，或在本機用瀏覽器開啟 [`docs/workflow.html`](docs/workflow.html)。

病理嗓音（例如喉部腫瘤、聲帶麻痺、聲帶結節或息肉造成的發聲異常）會在頻譜上留下痕跡：諧波結構變得模糊、高頻出現氣息雜訊、音高與音量出現顫動，或多出次諧波。把一小段持續母音畫成時頻圖，再交給影像分類器，是常見的自動偵測作法（Alhussein & Muhammad, 2018）；另一條路線是以倒頻譜特徵搭配深度神經網路（Fang et al., 2019）。

一段錄音通常能切出好幾個片段，預設全部拿來訓練與評估，同一段錄音的片段分數再平均成錄音層級的判斷。評估以受試者分組做交叉驗證，主要看 UAR，因為臨床資料的類別通常不平衡：一律猜多數類別就能拿到不低的 accuracy，UAR 卻只有 1／類別數。

分框、端點偵測、梅爾（mel）濾波器組、dB 換算與 WAV 讀寫來自 [`speechdsp`](https://pypi.org/project/speechdsp/)，UAR 也用它的定義；重取樣使用 SciPy 的多相濾波器。

| 階段 | 做什麼 | 對應模組 |
| --- | --- | --- |
| 1. 資料集索引 | 讀取並驗證 `dataset.csv`（路徑、類別、受試者分組），或由 `<資料夾>/<類別>/<檔案>.wav` 產生 | `dataset` |
| 2. 切段 | 讀 WAV、以多相濾波器重取樣到統一取樣率、去除頭尾靜音、切成固定長度片段 | `segmentation` |
| 3. 頻譜圖 | 短時功率譜密度、等寬或 mel 頻帶、相對於片段峰值的 dB、線性量化成 40 × 98 的 8 位元影像 | `spectrogram` |
| 4. 模型 | 頻譜圖 CNN（PyTorch，選用）、MLP 與 logistic regression 基線、類別不平衡處理 | `cnn`、`models` |
| 5. 評估 | 分層、分組的 k 折交叉驗證，片段與錄音兩個層級的指標，混淆矩陣圖表 | `evaluation`、`figures` |
| 6. 流程與輸出 | `prepare`／`train`／`predict` 高階函式、執行資料夾、`config.yaml` 與 `manifest.json` | `pipeline`、`manifest`、`cli` |
| 7. 合成資料 | 三種嗓音特質的合成持續母音，供測試、範例與快速開始使用 | `synthetic` |

---

## 方法說明

### 1. 重取樣、去除靜音與固定長度切段

每段錄音先以 `speechdsp.read_wav` 讀成單聲道 `float64`，再用多相（polyphase）抗混疊濾波器重取樣到 `segmentation.sample_rate`（預設 16 kHz；Crochiere & Rabiner, 1983）。44.1 kHz 與 48 kHz 混用的錄音因此有相同的時間與頻率解析度，之後所有以毫秒為單位的參數才有一致的意義。8 kHz 以上的成分不會用到，16 kHz 正好保留 0–8 kHz。

接著可選擇以雙門檻端點偵測（Rabiner & Sambur, 1975；`speechdsp.trim_silence`）去除頭尾靜音，再切成長度 $L$、位移 $H$ 的片段：第 $i$ 段從第 $iH$ 個樣本開始，完整片段的數量為

$$n = \left\lfloor \frac{N - L}{H} \right\rfloor + 1 .$$

最後一個完整片段之後的殘段，若長度至少是 `min_fraction` × $L$，就補零保留，否則捨棄。每個片段記錄它在原錄音中的序號（`segment`，從 0 起算），並帶著錄音路徑與受試者分組 `group`。

| 設定鍵 | 預設 | 作用 |
| --- | --- | --- |
| `segmentation.sample_rate` | `16000` | 所有錄音重取樣到這個取樣率（Hz） |
| `segmentation.trim_silence` | `true` | 先去除頭尾靜音 |
| `segmentation.segment_s` | `1.0` | 片段長度（秒） |
| `segmentation.hop_s` | `1.0` | 片段位移（秒）；小於 `segment_s` 即為重疊切段 |
| `segmentation.min_fraction` | `0.5` | 殘段至少要填滿片段的這個比例才補零保留 |
| `segmentation.max_segments` | `null` | 每段錄音最多保留幾個片段（均勻挑選；只留一個時取中間那段）；`null` 表示全部保留 |

### 2. 短時功率譜密度

片段以 `speechdsp.enframe` 切成音框，乘上分析窗（預設 Hamming；Harris, 1978），補零到 $N_{\mathrm{FFT}}$ 點後做實數 FFT（Allen, 1977）。第 $t$ 個音框的單邊功率譜密度為

$$P_t[k] = \frac{c_k \, |X_t[k]|^2}{f_s \sum_n w[n]^2}, \qquad c_k = \begin{cases} 1 & k = 0 \text{ 或 Nyquist} \\ 2 & \text{其他} \end{cases}$$

把它對頻率積分就等於訊號的變異數（測試以白雜訊驗證這個性質）。1 秒是關鍵詞辨識資料集慣用的片段長度（Warden, 2018）；以預設的 25 ms 音框、10 ms 位移分析，1 秒片段恰好得到 $\lfloor (1000 - 25) / 10 \rfloor + 1 = 98$ 個音框，這就是 98 欄的由來。$N_{\mathrm{FFT}}$ 預設為不小於音框長度的最小 2 的冪次（16 kHz 下 400 點音框補零到 512 點）。

| 設定鍵 | 預設 | 作用 |
| --- | --- | --- |
| `spectrogram.frame_ms` | `25.0` | 音框長度（毫秒） |
| `spectrogram.hop_ms` | `10.0` | 音框位移（毫秒） |
| `spectrogram.n_fft` | `null` | FFT 點數；`null` 為不小於音框長度的最小 2 的冪次 |
| `spectrogram.window` | `hamming` | 窗函數：`hamming`、`hann`、`blackman` 或 `boxcar` |

### 3. 頻帶網格與 8 位元量化

功率譜在 0 Hz 到 `fmax_hz` 之間合併成 `n_bands` 個頻帶：

- `linear`：等寬頻帶（預設 40 個，每個 200 Hz）。每個 FFT bin 依它落在頻帶內的寬度比例加權，每一列的權重和為 1，所以頻帶值是平均功率；頻帶比 bin 還窄時也不會出現空列。
- `mel`：`speechdsp.mel_filterbank` 的三角形 mel 濾波器（Stevens et al., 1937；Davis & Mermelstein, 1980），低頻解析度較高。

接著以片段自己的峰值為基準換成 dB（`speechdsp.power_to_db`），並以 `top_db` 截底：

$$L = \max\!\left(10 \log_{10} \frac{E}{E_{\max}},\; -\mathrm{top\_db}\right),$$

音框數不等於 `n_frames` 時沿時間軸線性內插（剛好 98 個音框時不變），最後線性量化成灰階值

$$g = \mathrm{round}\!\left(255 \cdot \frac{L + \mathrm{top\_db}}{\mathrm{top\_db}}\right) \in \{0, \dots, 255\}.$$

影像第 0 列是最低頻帶、第 0 欄是最早的音框；255 是片段的頻譜峰值，0 是峰值以下 `top_db` dB，中間是線性對應。一個灰階等於 `top_db`／255 dB（預設約 0.31 dB），所以灰階值可以換算回 dB，只有被截底的部分無法還原。這一步沒有任何亂數，同樣的訊號與設定一定得到同一張影像。dB 的下限是相對於峰值，所以小聲的錄音與大聲的錄音得到相同的影像；全靜音的片段（例如補零的部分）是全黑。

| 設定鍵 | 預設 | 作用 |
| --- | --- | --- |
| `spectrogram.scale` | `linear` | `linear` 或 `mel` |
| `spectrogram.n_bands` | `40` | 影像列數（頻帶數） |
| `spectrogram.n_frames` | `98` | 影像欄數（音框數） |
| `spectrogram.fmax_hz` | `8000.0` | 最高頻帶的上緣（Hz），不可超過取樣率的一半 |
| `spectrogram.top_db` | `80.0` | 動態範圍：灰階 0 對應峰值以下幾 dB |

### 4. 頻譜圖 CNN

輸入表示沿用小型關鍵詞辨識（keyword spotting）的作法：40 個頻帶、25 ms 音框、10 ms 位移的時頻圖（Sainath & Parada, 2015），長度取 1 秒（Warden, 2018）。網路本身是 3 × 3 小卷積核的堆疊（Simonyan & Zisserman, 2015）：六個卷積區塊，「same」補邊，每個區塊後接 batch normalization（Ioffe & Szegedy, 2015）與 ReLU 活化函數（Nair & Hinton, 2010），後四個區塊之前加 dropout（Srivastava et al., 2014）。

| 區塊 | 通道 | 區塊之後 | 輸出（40 × 98 輸入） |
| --- | --- | --- | --- |
| 1 | 16 | 2 × 2 max pooling | 16 × 20 × 49 |
| 2 | 32 | 2 × 2 max pooling，左右各補一欄 | 32 × 10 × 25 |
| 3 | 64 | — | 64 × 10 × 25 |
| 4 | 64 | 2 × 2 max pooling，左右各補一欄 | 64 × 5 × 13 |
| 5 | 64 | — | 64 × 5 × 13 |
| 6 | 64 | 沿整個時間軸取最大值 | 64 × 5 × 1 |

剩下的 320 個值接全連接層與 softmax，以交叉熵損失函數和 Adam（Kingma & Ba, 2015）訓練（LeCun et al., 1998），可訓練參數共 135,651 個（`parameter_count(3)`）。最後一層沿時間軸取全域最大值，全連接層的輸入大小因此與欄數無關，任何 `spectrogram.n_frames` 設定都能用同一個架構；訓練好的模型只接受訓練時的影像大小。輸入先除以 255，再減去訓練資料的逐像素平均影像；這張平均影像是網路的 buffer，跟權重一起存進 `state_dict`，推論時自動套用。

每個 epoch 用固定種子的亂數產生器重新打亂訓練資料，最後不足一批的資料照樣使用；權重初值由 `torch.manual_seed(seed)` 固定。`model.class_weight = balanced` 時，類別 $c$ 的交叉熵權重為 $w_c = N / (K N_c)$（$N$ 為訓練片段數、$K$ 為類別數、$N_c$ 為該類片段數）。預設學習率是 $10^{-3}$、100 個 epoch；要比較其他學習率時，以 `--set model.learning_rate=...` 覆寫即可。

| 設定鍵 | 預設 | 作用 |
| --- | --- | --- |
| `model.name` | `cnn` | `cnn`（需要 `dl` extra）、`mlp` 或 `logistic` |
| `model.n_epochs` | `100` | CNN 的訓練回合數；MLP 則是上限，訓練損失連續 10 個回合改善不到 $10^{-4}$ 就提早停止 |
| `model.learning_rate` | `0.001` | Adam 步長（CNN 與 MLP） |
| `model.batch_size` | `60` | mini-batch 大小（CNN 與 MLP） |
| `model.weight_decay` | `0.0001` | L2 懲罰：CNN 的 Adam weight decay、MLP 的 `alpha` |
| `model.dropout` | `0.2` | 後四個卷積區塊之前的 dropout 機率 |
| `model.device` | `cpu` | `cpu` 或 `cuda` |

### 5. 基線模型與類別不平衡

沒有安裝 PyTorch 時，整個流程仍可用 scikit-learn 的模型跑完，它們也是 CNN 的比較基準：把 40 × 98 影像攤平成 3,920 維向量，先標準化（可選 PCA 投影），再接以反向傳播訓練的多層感知器（MLP；Rumelhart et al., 1986）或 multinomial logistic regression。所有會從資料學參數的步驟都包在同一個 `Pipeline` 裡，交叉驗證時只在訓練折上擬合。

`class_weight = balanced` 對 logistic regression 是加權損失；MLP 不支援類別權重，所以在訓練折裡把少數類別的片段隨機重複抽樣（有放回），直到每一類都和最大類一樣多；推論不受影響。

| 設定鍵 | 預設 | 作用 |
| --- | --- | --- |
| `model.class_weight` | `balanced` | `balanced` 或 `null` |
| `model.hidden_units` | `[128]` | MLP 隱藏層大小 |
| `model.n_components` | `null` | 先投影到幾個主成分；`null` 表示不投影 |
| `model.logistic_c` | `1.0` | logistic regression 正規化（regularization）強度的倒數 |

### 6. 分層分組交叉驗證與錄音層級彙整

折由 `StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)` 產生（Kohavi, 1995）：每一折維持類別比例，而同一個 `group`（受試者；索引沒有 `group` 欄時就是錄音本身）的所有片段一定落在同一折。每一折都重新建立模型、只在訓練折上擬合，所以每個片段恰好得到一筆 out-of-fold 預測。

主要指標是 UAR（Schuller et al., 2009），即各類召回率的平均：

$$\mathrm{UAR} = \frac{1}{K} \sum_{c=1}^{K} \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c}.$$

它不會因為一律猜多數類別而變高；三類問題中，隨機猜測的 UAR 約為 1/3。另外回報 accuracy、macro F1 與 one-vs-rest macro ROC-AUC，以及合併所有 out-of-fold 預測後的混淆矩陣與每一類的召回率（sensitivity）、specificity、precision、F1 與樣本數。每折指標的摘要是平均與樣本標準差（`ddof=1`）。

錄音層級的判斷，是把同一段錄音所有片段的類別機率取平均，再選機率最大的類別：

$$\bar{p}_r(c) = \frac{1}{|S_r|} \sum_{s \in S_r} p_s(c), \qquad \hat{y}_r = \arg\max_c \bar{p}_r(c).$$

| 設定鍵 | 預設 | 作用 |
| --- | --- | --- |
| `evaluation.n_splits` | `5` | 交叉驗證折數 |
| `seed` | `0` | 主種子：分折、模型初值、過取樣、合成資料都由它決定 |

### 7. 合成資料

`synthesize` 指令與測試使用的合成資料，是以諧波聲源加上 /a/ 的四個共振峰產生的持續母音，分成三種嗓音特質：

| 類別 | 生成方式 | 在頻譜圖上的痕跡 |
| --- | --- | --- |
| `modal` | 穩定的諧波、適中的頻譜斜率、極低的雜訊 | 清楚的諧波橫紋 |
| `breathy` | 較陡的頻譜斜率（高次諧波弱）、2 kHz 以上的強氣息雜訊 | 高頻區整片變亮 |
| `rough` | 4–6 Hz 的音高與音量顫動、基頻 0.5 與 1.5 倍處的次諧波 | 時間方向的週期起伏、多出的頻譜線 |

我挑這三種特質，是因為它們在頻譜圖上留下的痕跡各不相同；它們只是替代訊號，並不模擬任何臨床病症。每位合成受試者有自己的基頻、共振峰比例與「嚴重度」（`synthetic.severity_range`，0 等於 modal、1 是特質最明顯），輕微的受試者會和 modal 重疊，所以分類不是輕而易舉。同一位受試者的多段錄音共用這些參數（只有基頻每段錄音在 ±3% 內各自變動），交叉驗證若不依受試者分組，測試折裡就會出現訓練時已經聽過的聲音。錄音輪流使用不同取樣率（預設 22,050 與 24,000 Hz），頭尾各有一小段近乎無聲的訊號，重取樣與去除靜音因此都會實際跑到。預設各類人數是 8／10／20，刻意不平衡。所有亂數都來自同一個產生器，所以同一個種子寫出的檔案逐位元組相同。

| 設定鍵 | 預設 | 作用 |
| --- | --- | --- |
| `synthetic.class_names` | `[breathy, modal, rough]` | 要合成的嗓音特質（至少兩類） |
| `synthetic.n_speakers_per_class` | `[8, 10, 20]` | 各類的合成受試者人數，順序同 `class_names` |
| `synthetic.n_recordings_per_speaker` | `2` | 每位受試者的錄音數（同一位受試者的錄音必在同一折） |
| `synthetic.severity_range` | `[0.2, 1.0]` | 非 modal 受試者的嚴重度範圍：0 等於 modal，1 是特質最明顯 |
| `synthetic.duration_s` | `2.5` | 有聲部分的長度（秒） |
| `synthetic.silence_s` | `0.25` | 頭尾近乎無聲部分的長度（秒） |
| `synthetic.sample_rates` | `[22050, 24000]` | 輪流使用的取樣率（Hz） |

---

## 安裝步驟

需要 Python 3.10 以上。

```bash
git clone https://github.com/recklight/voice-spectral-classifier.git
cd voice-spectral-classifier
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e .                 # 核心（會一併從 PyPI 安裝 speechdsp）
pip install -e ".[dl]"           # PyTorch 模型（選用）
pip install -e ".[dev]"          # 開發工具：pytest、ruff
```

PyTorch 只有 CNN 用得到。沒有安裝 `dl` extra 時，合成資料、前處理、MLP 與 logistic regression 的交叉驗證、訓練與推論、圖表都能照常使用。預設的 `model.name` 是 `cnn`，這時錯誤訊息會提示在專案資料夾內執行 `pip install -e ".[dl]"`（或直接 `pip install torch`）；不想裝 PyTorch 就加上 `--set model.name=mlp`，`configs/quick_demo.yaml` 已經這樣設定。

套件本身也可以不安裝，在專案資料夾內以 `PYTHONPATH=src python -m voice_spectral_classifier --help` 直接執行。

---

## 快速開始

不需要任何資料，以下指令會先產生合成資料集。

```bash
# 1. 產生合成資料集
voice-spectral-classifier synthesize --config configs/quick_demo.yaml --out data/synthetic
# 2. 由資料集索引準備模型輸入
voice-spectral-classifier prepare --config configs/quick_demo.yaml --out outputs/demo/prepare
# 3. 交叉驗證
voice-spectral-classifier evaluate outputs/demo/prepare/features.npz --config configs/quick_demo.yaml --out outputs/demo/evaluate
# 4. 以全部資料訓練
voice-spectral-classifier train outputs/demo/prepare/features.npz --config configs/quick_demo.yaml --out outputs/demo/train
# 5. 對新的輸入檔推論
voice-spectral-classifier predict data/synthetic/rough/rough_000.wav --model outputs/demo/train/model.joblib --out outputs/demo/predict
```

或一次跑完整個流程：`python examples/end_to_end_synthetic.py`。

> 這份合成資料只有 18 段錄音，用來確認每個指令都跑得通；跑出來的分數不代表真實資料上的表現。

---

## 使用範例

### CLI

```bash
# 1. 由 <資料夾>/<類別>/<檔案>.wav 的結構產生資料集索引（之後可手動編輯 group 欄）
voice-spectral-classifier index data/voices
# 2. 切段並計算頻譜圖；這裡用 mel 頻帶
voice-spectral-classifier prepare --dataset data/voices/dataset.csv --set spectrogram.scale=mel --out outputs/voices/prepare
# 3. 5 折交叉驗證 CNN（需要 dl extra）；特徵設定必須與 prepare 相同
voice-spectral-classifier evaluate outputs/voices/prepare/features.npz --set spectrogram.scale=mel --seed 1 --out outputs/voices/evaluate
# 4. 換成 MLP 基線、關掉類別平衡，做對照
voice-spectral-classifier evaluate outputs/voices/prepare/features.npz --set spectrogram.scale=mel --set model.name=mlp --set model.class_weight=null --out outputs/voices/mlp
# 5. 以全部資料訓練並存檔
voice-spectral-classifier train outputs/voices/prepare/features.npz --set spectrogram.scale=mel --out outputs/voices/train
# 6. 推論新的錄音：同時寫出逐片段與逐錄音的結果
voice-spectral-classifier predict new/visit_01.wav new/visit_02.wav --model outputs/voices/train/model.pt --out outputs/voices/predict
```

上面的 `prepare`、`evaluate` 與 `train` 都得各寫一次 `--set spectrogram.scale=mel`。固定不變的設定可以寫進自己的 YAML（例如只列出與預設不同的鍵的 `configs/my_study.yaml`），再以 `--config` 指定。

### Python API

```python
import numpy as np

from voice_spectral_classifier import (
    Config,
    cross_validate,
    load_model,
    predict_inputs,
    prepare_features,
    save_model,
    spectrogram_image,
    train_model,
    write_synthetic_dataset,
)

# 1. 低階函式：1.1 kHz 正弦波的 40 × 98 頻譜圖，最亮的列是 1000–1200 Hz 的第 5 個頻帶
t = np.arange(16_000) / 16_000
image = spectrogram_image(np.sin(2 * np.pi * 1100 * t), 16_000)
print(image.shape, image.dtype, image.mean(axis=1).argmax())  # (40, 98) uint8 5

# 2. 設定：讀快速示範的設定，再以點號鍵覆寫（換成 mel 頻帶與 logistic regression）
config = Config.from_yaml("configs/quick_demo.yaml").override(
    {"spectrogram.scale": "mel", "model.name": "logistic"}
)

# 3. 合成資料 → 片段影像
index = write_synthetic_dataset("data/synthetic", config.synthetic, rng=config.seed)
features = prepare_features(index, config)
print(features.n_items, features.image_shape)  # 36 (40, 98)

# 4. 分組交叉驗證：片段層級與錄音層級的 UAR（3 折的平均 ± 樣本標準差）
result = cross_validate(features, config)
for level in (result.metrics, result.metrics["recordings"]):
    uar = level["summary"]["uar"]
    print(f"{uar['mean']:.3f} +/- {uar['std']:.3f}")  # 0.722 +/- 0.096，接著 0.778 +/- 0.096
print(result.predictions.columns.tolist())  # path, segment, group, fold, label, predicted, score_<class>

# 5. 訓練、存檔、讀回並推論
model_path = save_model("outputs/model.joblib", train_model(features, config), config)
bundle = load_model(model_path)
table = predict_inputs(["data/synthetic/rough/rough_000.wav"], bundle)
print(table[["segment", "predicted"]])  # 每個片段一列
```

### 範例腳本

| 腳本 | 示範什麼 | 執行時間 |
| --- | --- | --- |
| `examples/end_to_end_synthetic.py` | 合成資料 → 頻譜圖 → 交叉驗證 → 訓練存檔 → 讀回推論，全部經由公開 API | 約 20 秒（`--quick` 約 3 秒） |
| `examples/compare_spectrogram_scales.py` | 同一份資料以 `linear` 與 `mel` 頻譜圖分別交叉驗證 logistic regression、MLP（有 PyTorch 時加上 CNN），寫出比較表與長條圖 | 約 30 秒（`--quick` 約 3 秒） |
| `examples/aggregate_segment_scores.py` | 每段錄音只留一個片段與保留全部片段的對照，以及片段層級與錄音層級分數的差別；預設資料上，只留一個片段時 UAR 為 0.795 ± 0.109，保留全部片段為 0.833 ± 0.054（錄音層級 0.837 ± 0.099） | 約 25 秒（`--quick` 約 3 秒） |

每支腳本都接受 `--out`、`--seed` 與 `--quick`，預設輸出到 `examples/output/<腳本名稱>/`。

---

## CLI 指令對照表

| 指令 | 用途 | 主要輸入 | 主要輸出 |
| --- | --- | --- | --- |
| `synthesize` | 寫出與真實資料相同格式的合成資料集 | 設定 | `<out>/<類別>/<類別>_<NNN>.wav`、`dataset.csv` |
| `index` | 由 `<資料夾>/<類別>/<檔案>.wav` 產生資料集索引 | 資料夾 | `<資料夾>/dataset.csv` |
| `prepare` | 讀資料集索引、切段、計算頻譜圖 | `dataset.csv` | `features.npz`、`items.csv`、`class_spectrograms.*` |
| `evaluate` | 分層、分組的 k 折交叉驗證 | `features.npz` | `metrics.json`、`folds.csv`、`predictions.csv`、`recording_predictions.csv`、混淆矩陣圖表 |
| `train` | 以全部片段訓練一次並存檔 | `features.npz` | `model.pt`（CNN）或 `model.joblib` |
| `predict` | 以模型檔推論新的錄音，只用模型檔裡的設定 | WAV 檔、模型檔 | `predictions.csv`、`recording_predictions.csv` |

| 選項 | 適用指令 | 說明 |
| --- | --- | --- |
| `--log-level LEVEL` | 全域（寫在子命令之前） | `DEBUG`、`INFO`（預設）、`WARNING` 或 `ERROR`；記錄輸出到 stderr |
| `--version` | 全域 | 顯示版本後結束 |
| `--config/-c PATH` | `synthesize`、`prepare`、`evaluate`、`train` | YAML 設定檔；沒給時使用預設值 |
| `--set/-s KEY=VALUE` | 同上 | 覆寫一個設定，例如 `--set model.name=mlp`；可重複 |
| `--seed INT` | `synthesize`、`evaluate`、`train` | 等同 `--set seed=INT` |
| `--dataset/-d PATH` | `prepare` | 等同 `--set data.dataset=PATH` |
| `--out/-o PATH` | `synthesize`、`prepare`、`evaluate`、`train`、`predict` | 輸出資料夾，不存在時自動建立；沒給時在 `output.root` 底下建立 `<指令>_<UTC 時間>`（`synthesize` 預設為 `data/synthetic`）。`index` 沒有這個選項，一律寫到 `<資料夾>/dataset.csv` |
| `--model/-m PATH` | `predict` | `train` 寫出的模型檔（必填） |
| `--figures/--no-figures` | `prepare` | 是否畫出各類平均頻譜圖（預設會畫） |

除了 `index`，每個指令都會在輸出資料夾另外寫出 `config.yaml` 與 `manifest.json`，事後查得到這次用了哪些設定和輸入檔。

exit code：`0` 表示成功；`2` 表示輸入或設定有誤（只印一行 `error: ...`，加上 `--log-level DEBUG` 可看到完整的 traceback）；`1` 表示被中斷或程式本身的錯誤。

---

## 專案結構

```text
voice-spectral-classifier/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml               # 錯誤回報表單
│   │   ├── config.yml                   # 停用空白 issue
│   │   └── feature_request.yml          # 功能建議表單
│   └── workflows/
│       └── ci.yml                       # lint、3.10 語法閘門、Linux／Windows × Python 3.10–3.13、最舊相依版本、PyTorch
├── configs/
│   ├── default.yaml                     # 與 Config() 完全相同，逐鍵附註解
│   └── quick_demo.yaml                  # 小型合成資料與 MLP，幾秒內跑完
├── data/
│   └── .gitkeep                         # 資料的預設位置，內容不進版控
├── outputs/
│   └── .gitkeep                         # 執行資料夾的預設根目錄，內容不進版控
├── docs/
│   ├── images/workflow.png              # README 的流程圖
│   └── workflow.html                    # 互動式流程圖（瀏覽器開啟）
├── examples/
│   ├── end_to_end_synthetic.py          # 端到端流程
│   ├── compare_spectrogram_scales.py    # linear 與 mel 頻譜圖、各模型的比較
│   └── aggregate_segment_scores.py      # 片段層級與錄音層級分數的對照
├── src/
│   └── voice_spectral_classifier/
│       ├── __init__.py                  # 公開 API
│       ├── __main__.py                  # python -m voice_spectral_classifier
│       ├── py.typed                     # 型別資訊標記
│       ├── cli.py                       # typer 命令列
│       ├── config.py                    # 設定 dataclass 與 YAML 讀寫
│       ├── errors.py                    # 例外類別
│       ├── logging_utils.py             # logging 設定
│       ├── manifest.py                  # 執行資料夾與 manifest.json
│       ├── dataset.py                   # dataset.csv 與 features.npz
│       ├── synthetic.py                 # 合成持續母音
│       ├── segmentation.py              # 重取樣、去除靜音、固定長度切段
│       ├── spectrogram.py               # 功率譜密度、頻帶網格、8 位元量化
│       ├── cnn.py                       # 頻譜圖 CNN（延遲匯入 PyTorch）
│       ├── models.py                    # 模型註冊表、類別平衡、模型檔
│       ├── pipeline.py                  # prepare／train／predict 高階函式
│       ├── evaluation.py                # 分組交叉驗證與指標
│       └── figures.py                   # 混淆矩陣、各類平均頻譜圖與對應表格
├── tests/                               # 各模組都有對應的測試，另有 CLI、範例、3.10 相容性與文件檢查
├── .gitattributes
├── .gitignore
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## 資料準備

**本專案不附任何資料。** repository 裡沒有錄音、頻譜圖影像或模型檔，測試與範例只用程式合成的資料。請自行準備錄音，並把 `data.dataset`（或 `--dataset`）指向你的資料集索引。

### 1. 資料集索引 dataset.csv

| 欄位 | 必要 | 說明 |
| --- | --- | --- |
| `path` | 是 | WAV 檔路徑，相對於 `dataset.csv` 所在的資料夾，以 `/` 分隔（`\` 也接受） |
| `label` | 是 | 類別名稱，例如 `neoplasm`、`phonotrauma`、`palsy`；類別順序一律依字母排序 |
| `group` | 否 | 受試者代號。同一位受試者的所有錄音一定在同一折；沒有這一欄時，每段錄音自成一組 |

```csv
path,label,group
neoplasm/case_0001.wav,neoplasm,case_0001
palsy/case_0002_a.wav,palsy,case_0002
palsy/case_0002_b.wav,palsy,case_0002
phonotrauma/case_0003.wav,phonotrauma,case_0003
```

檔名不承載任何意義，標籤與分組只從欄位讀取。若錄音已經依類別放在子資料夾，`voice-spectral-classifier index <資料夾>` 會寫出一份 `dataset.csv`（每段錄音一組），再依需要編輯 `group` 欄即可。索引列出的檔案若不存在，程式會列出前 5 個找不到的路徑並停止。

### 2. 檔案格式

| 項目 | 要求與處理方式 |
| --- | --- |
| 容器 | RIFF／WAVE；8、16、32 位元整數或浮點 PCM 都可以 |
| 聲道 | 多聲道會平均成單聲道 |
| 取樣率 | 任意；一律重取樣到 `segmentation.sample_rate`（預設 16 kHz） |
| 長度 | 至少要有 `min_fraction` × `segment_s`（預設 0.5 秒）的有聲部分；更短的錄音會被略過並記錄警告 |
| 數值 | 浮點 WAV 裡只要有一個 NaN 或無限大的樣本，程式就指出檔名並停止 |
| 內容 | 持續母音（例如 /a/）最理想；本流程不處理連續語句的斷詞 |

### 3. 公開資料集

FEMH Voice Data Challenge 2018（IEEE BigData 2018 的競賽資料）含有 neoplasm、phonotrauma 與 vocal palsy 等類別的持續母音錄音，需向主辦單位申請使用。取得後依上述格式建立 `dataset.csv` 即可套用本流程。本專案不包含也不散布該資料集。

---

## 設定檔

- `configs/default.yaml`：與 `Config()` 完全相同，每個鍵都附英文註解，適合複製後修改。
- `configs/quick_demo.yaml`：只列與預設不同的鍵（小型合成資料、MLP、3 折），讓「快速開始」幾秒內跑完；`data.dataset` 指向 `data/synthetic/dataset.csv`。

程式不會自動載入任何設定檔：沒給 `--config` 時使用預設值。設定裡的相對路徑以目前的工作目錄為基準。

| 區段 | 重要欄位 | 說明 |
| --- | --- | --- |
| （頂層） | `seed` | 主種子 |
| `data` | `dataset` | 資料集索引（預設為佔位值 `data/dataset.csv`） |
| `synthetic` | `class_names`、`n_speakers_per_class`、`n_recordings_per_speaker`、`severity_range`、`duration_s`、`silence_s`、`sample_rates` | 合成資料的大小與生成參數 |
| `segmentation` | `sample_rate`、`trim_silence`、`segment_s`、`hop_s`、`min_fraction`、`max_segments` | 重取樣、去除靜音與切段 |
| `spectrogram` | `scale`、`n_bands`、`n_frames`、`frame_ms`、`hop_ms`、`n_fft`、`window`、`fmax_hz`、`top_db` | 頻譜圖 |
| `model` | `name`、`n_epochs`、`learning_rate`、`batch_size`、`weight_decay`、`class_weight`、`dropout`、`hidden_units`、`n_components`、`logistic_c`、`device` | 模型與訓練 |
| `evaluation` | `n_splits` | 交叉驗證折數 |
| `output` | `root`、`figure_format`、`dpi` | 執行資料夾的位置與圖檔格式 |

`segmentation` 與 `spectrogram` 決定 `prepare` 的輸出，會存進 `features.npz`；`evaluate` 與 `train` 讀入時若這兩個區段與目前的設定不同，會直接報錯，避免拿不同設定的特徵訓練。`predict` 只使用模型檔裡的設定。

未知的鍵、型別不對或不是有限值的數字（例如 `.inf`）都會直接報錯；未知的鍵還會列出合法的鍵，例如：

```text
$ voice-spectral-classifier prepare --set spectrogram.n_band=20
error: unknown configuration key 'spectrogram.n_band'; valid keys here are ['fmax_hz', 'frame_ms', 'hop_ms', 'n_bands', 'n_fft', 'n_frames', 'scale', 'top_db', 'window']
```

---

## 結果與評估指標

一次 `evaluate` 的執行資料夾：

```text
outputs/demo/evaluate/
├── config.yaml                      # 實際使用的完整設定
├── manifest.json                    # 指令、時間、版本、輸入檔的 SHA-256、種子
├── metrics.json                     # 每折、摘要與合併後的指標（片段與錄音兩個層級）
├── folds.csv                        # 每折一列
├── predictions.csv                  # 每個片段一筆 out-of-fold 預測
├── recording_predictions.csv        # 每段錄音一筆（片段機率平均）
├── confusion_matrix.png             # 片段層級的混淆矩陣
├── confusion_matrix.csv             # 同一張圖的數字
├── recording_confusion_matrix.png   # 錄音層級的混淆矩陣
└── recording_confusion_matrix.csv
```

| 檔案 | 內容 |
| --- | --- |
| `metrics.json` | `task`、`n_items`、`n_splits`、`seed`、`folds`、`summary`（每個指標的 `mean` 與 `std`）、`pooled`（含 `classes`、`confusion_matrix`、`per_class`）、`recordings`（以錄音為單位的同樣結構） |
| `folds.csv` | `fold`、`n_train`、`n_test`、`uar`、`accuracy`、`f1_macro`、`roc_auc_ovr` |
| `predictions.csv` | `path`、`segment`、`group`、`fold`、`label`、`predicted`、`score_<類別>`，列順序與 `items.csv` 相同 |
| `recording_predictions.csv` | `path`、`group`、`fold`、`label`、`predicted`、`score_<類別>` |
| `confusion_matrix.csv` | `label`（真實類別）、`predicted`、`count`、`fraction`（占該真實類別的比例） |

| 指標 | 定義 | 為什麼看它 |
| --- | --- | --- |
| UAR | 各類召回率的平均 | 類別不平衡時的主要指標；一律猜多數類別只會得到 $1/K$ |
| accuracy | 預測正確的比例 | 直觀，但會被多數類別主導，只作參考 |
| `f1_macro` | 各類 F1 的平均 | 同時反映 precision 與 recall |
| `roc_auc_ovr` | 每一類對其他類的 ROC-AUC 平均 | 不依賴決策門檻，評估機率排序的品質；測試折缺少某一類時記為 `null` |
| 每類 recall／specificity | 該類被正確找出的比例／其他類沒被誤判成該類的比例 | 即臨床上的 sensitivity 與 specificity（one-vs-rest） |
| 混淆矩陣 | 列是真實類別、欄是預測類別 | 看錯誤集中在哪兩類之間 |

以下數字都是在合成資料上實際執行指令得到的輸出（Windows 11、Python 3.13、scikit-learn 1.7、NumPy 2.3，CNN 用 CPU 版 PyTorch 2.14），平均 ± 樣本標準差，`ddof=1`。換作業系統或套件版本時，末幾位可能略有不同。

「快速開始」的結果（`quick_demo.yaml`：18 段錄音、36 個片段、MLP、3 折）：

```bash
voice-spectral-classifier synthesize --config configs/quick_demo.yaml --out data/synthetic
voice-spectral-classifier prepare --config configs/quick_demo.yaml --out outputs/demo/prepare
voice-spectral-classifier evaluate outputs/demo/prepare/features.npz --config configs/quick_demo.yaml --out outputs/demo/evaluate
```

| 層級 | 筆數 | UAR | accuracy | macro F1 | ROC-AUC（OvR） |
| --- | --- | --- | --- | --- | --- |
| 片段 | 36 | 0.889 ± 0.096 | 0.889 ± 0.096 | 0.888 ± 0.098 | 0.927 ± 0.079 |
| 錄音 | 18 | 0.889 ± 0.096 | 0.889 ± 0.096 | 0.888 ± 0.098 | 0.934 ± 0.082 |

片段層級合併後的混淆矩陣（`confusion_matrix.csv`）與每類指標：

| 真實＼預測 | breathy | modal | rough | recall | specificity |
| --- | --- | --- | --- | --- | --- |
| breathy | 10 | 0 | 0 | 1.000 | 1.000 |
| modal | 0 | 8 | 2 | 0.800 | 0.923 |
| rough | 0 | 2 | 14 | 0.875 | 0.900 |

預設大小的合成資料（38 位受試者、76 段錄音、228 個片段，5 折）上，比較頻帶尺度、模型與類別平衡：

```bash
voice-spectral-classifier synthesize --out data/synthetic_full
voice-spectral-classifier prepare --dataset data/synthetic_full/dataset.csv --out outputs/full/linear
voice-spectral-classifier prepare --dataset data/synthetic_full/dataset.csv --set spectrogram.scale=mel --out outputs/full/mel
voice-spectral-classifier evaluate outputs/full/linear/features.npz --set model.name=cnn --out outputs/full/linear_cnn
voice-spectral-classifier evaluate outputs/full/linear/features.npz --set model.name=logistic --out outputs/full/linear_logistic
voice-spectral-classifier evaluate outputs/full/linear/features.npz --set model.name=mlp --out outputs/full/linear_mlp
voice-spectral-classifier evaluate outputs/full/mel/features.npz --set spectrogram.scale=mel --set model.name=cnn --out outputs/full/mel_cnn
voice-spectral-classifier evaluate outputs/full/mel/features.npz --set spectrogram.scale=mel --set model.name=logistic --out outputs/full/mel_logistic
voice-spectral-classifier evaluate outputs/full/mel/features.npz --set spectrogram.scale=mel --set model.name=mlp --out outputs/full/mel_mlp
voice-spectral-classifier evaluate outputs/full/linear/features.npz --set model.name=mlp --set model.class_weight=null --out outputs/full/linear_mlp_unweighted
```

| 頻譜圖 | 模型 | `class_weight` | UAR（片段） | accuracy（片段） | UAR（錄音） |
| --- | --- | --- | --- | --- | --- |
| `linear` | `cnn` | `balanced` | 0.843 ± 0.080 | 0.807 ± 0.100 | 0.834 ± 0.099 |
| `linear` | `logistic` | `balanced` | 0.933 ± 0.070 | 0.921 ± 0.072 | 0.933 ± 0.070 |
| `linear` | `mlp` | `balanced` | 0.833 ± 0.054 | 0.826 ± 0.084 | 0.837 ± 0.099 |
| `linear` | `mlp` | `null` | 0.827 ± 0.070 | 0.826 ± 0.091 | 0.820 ± 0.066 |
| `mel` | `cnn` | `balanced` | 0.945 ± 0.069 | 0.938 ± 0.062 | 0.945 ± 0.069 |
| `mel` | `logistic` | `balanced` | 0.841 ± 0.130 | 0.846 ± 0.096 | 0.838 ± 0.134 |
| `mel` | `mlp` | `balanced` | 0.827 ± 0.112 | 0.840 ± 0.099 | 0.827 ± 0.118 |

這份合成資料的類別差異集中在 2 kHz 以上（氣息雜訊）與基頻附近的細節（次諧波）。logistic regression 用 `linear` 明顯較好（0.933 對 0.841），MLP 兩者差不多，原因可能是等寬頻帶在高頻的解析度比 mel 頻帶好；CNN 則是 `mel` 較好（0.945 對 0.843），也是整張表最高的 UAR。沒有哪一種尺度對每個模型都比較好，真實資料的結論也可能不同，所以兩種尺度都保留。CNN 那兩列需要安裝 `dl` extra。

---

## 已知限制

- 合成母音只用來確認每個環節都正確運作，上面的分數不能拿來推估臨床資料的準確度；請在自己的資料上以分組交叉驗證評估。
- 這是研究用的程式，沒有經過臨床驗證，不能用於診斷。
- 模型檔只載入自己信任的來源。`model.joblib` 是 pickle，讀取時可以執行任意程式碼；`model.pt` 以 `weights_only=True` 讀取，風險較低，但同樣建議只用自己訓練的檔案。
- 受試者很少時，`StratifiedGroupKFold` 不保證每一折都有每一類。缺類的那一折 ROC-AUC 記為 `null`，摘要只平均有定義的折；UAR 只平均這一折實際出現的類別（`speechdsp.uar` 的定義），macro F1 則仍把被預測到、但沒有出現的類別算進去。某類的受試者少於折數時會記錄警告，這時請增加資料或降低 `evaluation.n_splits`。
- 機率沒有經過校準。MLP 與 CNN 的 softmax 輸出常過度自信，錄音層級平均的也是這些未校準的機率；需要風險分數時請另做校準。
- 只處理持續母音，片段之間沒有語音內容的對齊；連續語句需要另外做斷句或音素切割。
- 端點偵測靠錄音開頭估計雜訊。錄音一開始就在發聲時，偵測器會保留整段錄音；已經剪好的 1 秒片段建議設定 `segmentation.trim_silence=false`，以免被裁短。

---

## 設計重點

### 評估

- `group` 欄讓同一位受試者（或同一段錄音）的所有片段只出現在一折；同一個人的聲音如果同時出現在訓練折與測試折，分數會偏高。`predictions.csv` 每一列都有所屬的折、真實標籤與預測，事後可以逐筆檢查。
- 分折、權重初值、每個 epoch 的打亂順序、過取樣與合成資料都由 `seed` 決定。同一個種子在同一個環境重跑，CSV、`metrics.json` 與 `features.npz` 逐位元組相同（測試會比對；`manifest.json` 記有執行時間，每次都不同）。

### 訊號處理與輸入表示

- 每個片段都帶著錄音路徑與序號，錄音層級的分數就是靠路徑把同一段錄音的片段併回去。以 `max_segments` 只留部分片段時，序號仍是它在整段錄音所有片段中的編號。
- `linear` 與 `mel` 只差在頻帶合併矩陣，其餘分析相同，兩者的比較才公平；在相同的窗函數與音框設定下，mel 版本與 `speechdsp.spectrogram_image` 的差距不超過一個灰階（測試會比對）。

### 模型與資料流程

- CNN 每 10 個 epoch（以及最後一個）以 INFO 記錄一次訓練損失，學習率或 epoch 數不合適時從 log 就看得出來。
- 模型檔連同完整設定一起存：scikit-learn 模型存成 `model.joblib`，CNN 存成只含張量與基本型別的 `model.pt`。`predict` 只用模型檔裡的設定重新切段與計算頻譜圖，所以訓練與推論的前處理一致。在 GPU 上訓練的 CNN 拿到沒有 CUDA 的電腦上，會記錄警告並在 CPU 上推論。
- 標籤與分組只從 `dataset.csv` 的 `label` 與 `group` 欄讀取，不從檔名解析，所以檔名可以完全不帶標籤或身分資訊。類別依字母排序，數量就是 `label` 欄實際出現的名稱數；不管索引的列怎麼排，類別編號、混淆矩陣與 `score_<類別>` 欄的順序都一樣。錄音路徑以 `dataset.csv` 所在的資料夾為基準，整個資料夾搬到別處、或從別的工作目錄執行，索引都不用改。
- 每次執行的輸出集中在一個資料夾，檔名固定，旁邊就是這次的 `config.yaml` 與 `manifest.json`；要比較兩次執行，對照兩個資料夾就好。
- 資料有問題時（缺欄位、找不到檔案、無法解碼或含 NaN 的 WAV、特徵設定與 `prepare` 不一致），錯誤訊息只有一行，直接指出是哪個檔案、欄位或設定，不必從 traceback 裡找；太短的錄音則略過並記錄警告。

---

## 開發與測試

```bash
pip install -e ".[dev]"
python -m pytest -q                 # 全部測試，含 doctest
python -m ruff check .
python -m ruff format --check .
# Python 3.10 語法閘門（任何 Python 版本都能跑）
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8'), str(p), feature_version=(3, 10)) for p in pathlib.Path('.').rglob('*.py') if not {'.venv', 'build', 'dist', '__pycache__'} & set(p.parts)]"
```

除了各模組的功能測試，另有兩個檢查專案本身的測試：

- `tests/test_python310_compat.py` 以 Python 3.10 的語法解析每個檔案，掃描 3.11 以後才有的標準函式庫名稱、NumPy 1.x 與 2.x 之間不相容的名稱，以及比宣告的相依版本下限更新的 API；也確認 `import voice_spectral_classifier` 與命令列不會載入 PyTorch 等選用套件。
- `tests/test_documentation.py` 檢查文件用字（繁體字、台灣用語）與 README 的章節順序，也擋下不小心寫進檔案的本機路徑或 e-mail 地址。

examples 的三支腳本也由測試以 `--quick` 實際跑過一次。

CNN 的測試需要 PyTorch。安裝 `dl` extra（CPU 版即可）後，這些測試會在本機實際訓練與推論，沒有安裝時自動略過。CI 在 Linux 與 Windows 上跑 Python 3.10–3.13，另外有一個 job 用最舊的相依版本（這些下限都能在 Python 3.10 安裝），一個 job 裝 CPU 版 PyTorch 跑 CNN 的測試。

---

## References

### Signal analysis

- J. B. Allen, "Short term spectral analysis, synthesis, and modification by discrete Fourier transform," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 25, no. 3, pp. 235–238, 1977.
- F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform," *Proceedings of the IEEE*, vol. 66, no. 1, pp. 51–83, 1978.
- S. S. Stevens, J. Volkmann, and E. B. Newman, "A scale for the measurement of the psychological magnitude pitch," *The Journal of the Acoustical Society of America*, vol. 8, no. 3, pp. 185–190, 1937.
- S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 357–366, 1980.
- L. R. Rabiner and M. R. Sambur, "An algorithm for determining the endpoints of isolated utterances," *Bell System Technical Journal*, vol. 54, no. 2, pp. 297–315, 1975.
- R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal Processing*. Englewood Cliffs, NJ, USA: Prentice-Hall, 1983.

### Neural networks and training

- Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.
- T. N. Sainath and C. Parada, "Convolutional neural networks for small-footprint keyword spotting," in *Proc. Interspeech 2015*, 2015, pp. 1478–1482.
- K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. 3rd International Conference on Learning Representations (ICLR)*, 2015, arXiv:1409.1556.
- S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd International Conference on Machine Learning (ICML)*, 2015, pp. 448–456.
- N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.
- V. Nair and G. E. Hinton, "Rectified linear units improve restricted Boltzmann machines," in *Proc. 27th International Conference on Machine Learning (ICML)*, 2010, pp. 807–814.
- D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd International Conference on Learning Representations (ICLR)*, 2015, arXiv:1412.6980.
- D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533–536, 1986.

### Classification and evaluation

- R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in *Proc. 14th International Joint Conference on Artificial Intelligence (IJCAI)*, 1995, pp. 1137–1143.
- B. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009 Emotion Challenge," in *Proc. Interspeech 2009*, 2009, pp. 312–315.

### Pathological voice

- S.-H. Fang, Y. Tsao, M.-J. Hsiao, J.-Y. Chen, Y.-H. Lai, F.-C. Lin, and C.-T. Wang, "Detection of pathological voice using cepstrum vectors: A deep learning approach," *Journal of Voice*, vol. 33, no. 5, pp. 634–641, 2019.
- M. Alhussein and G. Muhammad, "Voice pathology detection using deep learning on mobile healthcare framework," *IEEE Access*, vol. 6, pp. 41034–41041, 2018.

### Datasets

- P. Warden, "Speech Commands: A dataset for limited-vocabulary speech recognition," arXiv:1804.03209, 2018.
- "FEMH Voice Data Challenge 2018," IEEE International Conference on Big Data (IEEE BigData 2018), 2018.

---

## License

MIT License，Copyright (c) 2026 RL。詳見 [LICENSE](LICENSE)。
