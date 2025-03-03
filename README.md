# GAN - Let's Make Some Blots
本專案是基於 GAN（生成對抗網路）的作業，利用類似 Pix2Pix 的架構進行影像到影像的轉換，主要應用於 Western Blot 影像的生成與處理。

## 目錄
- 專案簡介
- 資料說明
- 實作流程
- 模型架構
- 訓練流程
- 評估與結果
- 實驗結果及分析
- 心得與未來展望
- 使用方式
- 檔案結構
- 參考資料

---

## 專案簡介
本專案採用條件式 GAN 架構，利用 U-Net 風格的生成器 (Generator) 與 PatchGAN 鑑別器 (Discriminator) 對成對資料進行影像生成任務。輸入由模板影像與模板2（經過直角化處理的 BasePattern）組成 6 通道資料，生成器負責將該輸入轉換成 3 通道的 RGB 影像，並與真實影像比對以達到影像生成的目的。

---

## 資料說明
- 原始資料：存放於 `data/targets`，包含各尺寸不一的目標影像。
- 模板1：存放於 `data/templates`，提供 BandMask 資訊。
- 模板2：存放於 `data/template2`（或資料夾名稱為 `templates2`），提供經處理後的 BasePattern。
資料皆為成對對應，可直接應用於 Pix2Pix 架構進行條件式影像轉換。（參考作業報告內容 `Assignment _4- GAN - Le…`）

---

## 實作流程
1. **資料預處理**
   - 尺寸調整：使用 `transforms.Resize((256,256))` 將所有影像調整為固定尺寸。
   - 資料增強與歸一化：將影像轉為 tensor 並利用 `transforms.Normalize([0.5], [0.5])` 將像素值映射到 [-1,1] 區間。
   - Dataset 設計：實作 `WesternBlotDataset`（見 dataset.py `dataset`），自動載入並驗證 template、target 與（可選的）template2 之間的對應性。
2. **模型架構**
   - 生成器 (UNetGenerator)：採用 encoder-decoder 結構，輸入 6 通道（template 與 template2 拼接），中間經由多層卷積與轉置卷積處理，最終輸出 3 通道 RGB 影像，並以 Tanh 激活（參見 model.py `model`）。
   - 鑑別器 (PatchGAN)：採用 PatchGAN 結構，輸入 6 通道（template 與 target 或生成影像拼接），通過多層卷積與 LeakyReLU 判斷影像真偽。
3. **訓練流程**
   - 損失函數：
     - GAN 損失：使用二元交叉熵 (`BCELoss`) 判斷真假。
     - L1 損失：使用 `nn.L1Loss()` 衡量生成影像與真實影像的像素差異，並以權重 50 加權。
   - 優化器：均採用 Adam 優化器（學習率 1e-4，betas=(0.5, 0.999)），Batch size 設為 16。
   - 訓練步驟：
     - 先訓練鑑別器，分別對真實影像（template + target）與生成影像（template + fake）計算損失。
     - 固定鑑別器，再訓練生成器，使生成影像既能“騙過”鑑別器，又能與真實影像接近。（詳見 train.py `train`）

---

## 模型架構
- 生成器 (UNetGenerator)：
  - 使用多層卷積與 Batch Normalization 進行編碼，再利用轉置卷積逐步恢復空間尺寸，最後使用 Tanh 將輸出映射至 [-1,1] 範圍。
- 鑑別器 (PatchGAN)：
  - 透過多層卷積（含 LeakyReLU 與 Batch Normalization）進行局部判斷，最終輸出一張判斷圖以區分真偽。
 
---

## 評估與結果
- 評估指標：
  - PSNR（峰值信噪比）
  - SSIM（結構相似度）
- 評估工具：
  - evaluate.py 用於讀取指定 epoch 的生成影像與真實影像，計算 PSNR 與 SSIM。（​`evaluate`）
  - evaluate_all.py 會遍歷整個資料集，依據 PSNR 值排序後輸出 PSNR 前 10 及後 10 筆的對比圖。（`​evaluate_all`）
    
在 500 個 epoch 後，最終評估結果大致為：
- 平均 PSNR：約 13.23~13.28
- 平均 SSIM：約 0.59

---

## 實驗結果及分析
- 生成影像效果：
  - 生成影像在條帶位置與整體結構上已能大致對應目標影像，但在細節（例如灰階層次與邊緣處理）上仍有差異。
- 數據統計：
  - PSNR 前 10 筆圖片的值約介於 24.04 至 27.94 之間。
  - PSNR 後 10 筆圖片的值約介於 3.44 至 6.89 之間。
- 分析結論：
  - 由於生成過程本身存在隨機性，若要進一步提高生成品質與 PSNR/SSIM 數值，可能需要更多資料、進一步調整網路結構或引入更複雜的技術。

---

## 心得與未來展望
- 心得：
  - 本次作業中，從資料前處理到 GAN 模型訓練，均面臨不少挑戰，尤其在處理影像尺寸不一致與生成器與鑑別器的平衡問題上收穫頗多。
- 未來展望：
  - 可嘗試引入 WGAN-GP、Gradient Penalty、Spectral Normalization 等方法以提升訓練穩定性。
  - 探索 Conditional GAN 與更多 Perceptual Loss 的應用，以進一步改善生成影像細節品質。

---

## 使用方式
1. 環境安裝
   請確認已安裝 Python 3.7 以上版本，並安裝下列必要套件：
   ```bash
   pip install torch torchvision scikit-image matplotlib pillow
   ```
2. 資料準備
   請依照下列結構準備資料：
   ```bash
   data/
   ├── targets/      # 真實影像
   ├── templates/    # 模板1 (BandMask)
   └── template2/    # 模板2 (BasePattern)
   ```
3. 訓練模型
   執行 `train.py` 開始訓練：
   ```bash
   python train.py
   ```
4. 評估模型
   執行以下指令進行模型評估：
   - 單一影像評估:
     ```bash
     python evaluate.py
     ```
   - 全資料集評估並生成對比圖：
     ```bash
     python evaluate_all.py
     ```

---

## 檔案結構
```makefile
├── data/
│   ├── targets/         # 真實影像
│   ├── templates/       # 模板1影像 (BandMask)
│   └── template2/       # 模板2影像 (BasePattern)
├── output/              # 訓練過程中生成的影像與模型 checkpoint
├── eval_output/         # 評估時生成的對比圖片
├── main.py              # 測試 Dataset 載入
├── dataset.py           # 定義 WesternBlotDataset
├── model.py             # 定義 UNetGenerator 與 PatchGAN 模型
├── train.py             # GAN 模型訓練流程
├── evaluate.py          # 單次生成影像評估
├── evaluate_all.py      # 全資料集評估與結果可視化
└── Assignment _4- GAN - Let's make some blots.docx  # 作業報告
```

---

## 參考資料
- PyTorch 官方文件
- Pix2Pix 論文
- PatchGAN 論文

---

此專案由 蔡晉瑋 完成，感謝教授與所有提供資源的人員支持與指導！
