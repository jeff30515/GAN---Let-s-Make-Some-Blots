import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import numpy as np

# 設定影像路徑
SAVE_PATH = "output/"
fake_image_path = os.path.join(SAVE_PATH, "epoch_500_fake.png")
real_image_path = os.path.join(SAVE_PATH, "epoch_500_real.png")

# 加載生成影像和真實影像
fake_image = Image.open(fake_image_path).convert('RGB')
real_image = Image.open(real_image_path).convert('RGB')

# 轉為 numpy 格式
fake_image = np.array(fake_image)
real_image = np.array(real_image)

# 計算 PSNR
psnr = peak_signal_noise_ratio(real_image, fake_image)

# 計算 SSIM，明確指定 win_size
min_side = min(real_image.shape[0], real_image.shape[1])
win_size = min(7, min_side) if min_side >= 7 else min_side  # 確保窗口大小合法
ssim = structural_similarity(real_image, fake_image, win_size=win_size, channel_axis=-1)

# 打印結果
print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.2f}")
