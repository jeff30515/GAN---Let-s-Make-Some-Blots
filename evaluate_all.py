import os
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from dataset import WesternBlotDataset
from model import UNetGenerator

# ============== 1. 基本設定 ==============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路徑設定（請根據實際路徑做調整）
template_dir = "data/templates"
target_dir = "data/targets"
template2_dir = None  # 如果有 template2，請改成資料夾路徑；沒有就保持 None

# 載入最後訓練好的生成器權重路徑 (例如 epoch_500)
CHECKPOINT_PATH = "測試紀錄/output4/generator_epoch_500.pth"

# 輸出結果的資料夾
EVAL_OUTPUT_DIR = "eval_output"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# ============== 2. 建立 Dataset & DataLoader ==============
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1,1]
])

dataset = WesternBlotDataset(
    template_dir=template_dir,
    target_dir=target_dir,
    template2_dir=template2_dir,  # 若有 template2 路徑就放這裡
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ============== 3. 載入已訓練好的生成器 ==============
generator = UNetGenerator().to(DEVICE)
generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
generator.eval()


# 封裝函式：將 (C,H,W) 的張量從 [-1,1] 轉到 [0,255] 並回傳 numpy
def tensor_to_np255(tensor_img):
    """將 (C,H,W) or (1,C,H,W) 的張量轉為 (H,W,C) 的 [0..255] np.uint8"""
    if tensor_img.ndim == 4:
        tensor_img = tensor_img[0]  # 只取 batch=1
    arr = tensor_img.cpu().numpy()  # shape=(C,H,W)
    arr = (arr + 1) * 127.5  # [-1,1] -> [0,255]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # (C,H,W)->(H,W,C)
    return arr


# ============== 4. 遍歷資料集，計算每張影像的 PSNR ==============
psnr_list = []  # 用來存 (idx, psnr)

for idx, data in enumerate(dataloader):
    if len(data) == 3:
        template, target, template2 = data
        template = template.to(DEVICE)
        target = target.to(DEVICE)
        template2 = template2.to(DEVICE)
        generator_input = torch.cat((template, template2), dim=1)  # (B,6,H,W)
    else:
        template, target = data
        template = template.to(DEVICE)
        target = target.to(DEVICE)
        # 假 6 通道 (template 重複拼接)
        generator_input = torch.cat((template, template), dim=1)

    with torch.no_grad():
        fake_target = generator(generator_input)  # (B,3,H,W)

    # 計算 PSNR
    real_np = tensor_to_np255(target)
    fake_np = tensor_to_np255(fake_target)
    psnr_val = peak_signal_noise_ratio(real_np, fake_np, data_range=255)
    psnr_list.append((idx, psnr_val))

# 依 PSNR 由大到小排序
psnr_list.sort(key=lambda x: x[1], reverse=True)

# 取前 10 與後 10 筆
top_10 = psnr_list[:10]
bottom_10 = psnr_list[-10:]

print("=== Top 10 PSNR ===")
for rank, (idx, psnr_val) in enumerate(top_10, start=1):
    print(f"Rank {rank} | Index={idx} | PSNR={psnr_val:.2f}")

print("\n=== Bottom 10 PSNR ===")
for rank, (idx, psnr_val) in enumerate(bottom_10, start=1):
    print(f"Rank {rank} | Index={idx} | PSNR={psnr_val:.2f}")


# ============== 5. 輸出對應的 Template/Real/Fake ==============
# 可視化函式：將三張圖(模板,真實,生成)拼成一張
def save_comparison(idx, out_dir):
    """依據 dataset[idx], 生成對應 fake, 並將三張圖拼成一張圖片存檔。"""
    item = dataset[idx]
    if len(item) == 3:
        template, target, template2 = item
        template = template.unsqueeze(0).to(DEVICE)
        target = target.unsqueeze(0).to(DEVICE)
        template2 = template2.unsqueeze(0).to(DEVICE)
        gen_input = torch.cat((template, template2), dim=1)
    else:
        template, target = item
        template = template.unsqueeze(0).to(DEVICE)
        target = target.unsqueeze(0).to(DEVICE)
        gen_input = torch.cat((template, template), dim=1)

    with torch.no_grad():
        fake_target = generator(gen_input)

    # 還原到 [0,1] 範圍
    inv_norm = lambda x: (x * 0.5 + 0.5)  # [-1,1] -> [0,1]

    template_01 = inv_norm(template.squeeze(0).cpu())
    target_01 = inv_norm(target.squeeze(0).cpu())
    fake_01 = inv_norm(fake_target.squeeze(0).cpu())

    # 拼接 (B, C, H, W) => 3張: (template, real, fake)
    comparison = torch.stack([template_01, target_01, fake_01], dim=0)
    # shape = (3, 3, H, W)

    out_path = os.path.join(out_dir, f"compare_idx{idx}.png")
    # 每行顯示 3 張 => template, real, fake 在同一行
    save_image(comparison, out_path, nrow=3)
    return out_path


print("\n--- Saving top 10 images ---")
for (idx, psnr_val) in top_10:
    out_path = save_comparison(idx, EVAL_OUTPUT_DIR)
    print(f"Saved: {out_path}, PSNR={psnr_val:.2f}")

print("\n--- Saving bottom 10 images ---")
for (idx, psnr_val) in bottom_10:
    out_path = save_comparison(idx, EVAL_OUTPUT_DIR)
    print(f"Saved: {out_path}, PSNR={psnr_val:.2f}")

print("\nDone! Check your eval_output/ folder for results.")
