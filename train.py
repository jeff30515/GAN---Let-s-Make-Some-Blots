import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import WesternBlotDataset
from model import UNetGenerator, PatchGAN
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# 新增：匯入 matplotlib 以便繪圖
import matplotlib.pyplot as plt

# 參數設置
EPOCHS = 500
BATCH_SIZE = 16
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = "output/"
os.makedirs(SAVE_PATH, exist_ok=True)

# 數據加載
template_dir = "data/templates"
target_dir = "data/targets"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 將影像值歸一化到 [-1, 1]
])

dataset = WesternBlotDataset(template_dir, target_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定義模型
generator = UNetGenerator().to(DEVICE)
discriminator = PatchGAN().to(DEVICE)

# 損失函數與優化器
criterion_gan = nn.BCELoss()  # 用於 GAN 的真假損失
criterion_l1 = nn.L1Loss()    # 用於生成影像和目標影像的像素差異
optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# ----------------------------------
# 新增：用來記錄每個 Epoch 的 Generator / Discriminator 損失
g_losses = []
d_losses = []
# ----------------------------------

# 訓練過程
for epoch in range(EPOCHS):
    # 先準備累加器，用來計算該 Epoch 的總損失
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        if len(data) == 3:
            template, target, template2 = data
            template, target, template2 = template.to(DEVICE), target.to(DEVICE), template2.to(DEVICE)
            generator_input = torch.cat((template, template2), dim=1)  # 6 通道輸入生成器
        else:
            template, target = data
            template, target = template.to(DEVICE), target.to(DEVICE)
            generator_input = torch.cat((template, template), dim=1)  # 假 6 通道

        # ----------------------
        # 訓練判別器
        # ----------------------
        real_input = torch.cat((template, target), dim=1)  # 判別器輸入是 template + target，6 通道
        real_output = discriminator(real_input)

        real_labels = torch.ones_like(real_output, device=DEVICE)
        fake_labels = torch.zeros_like(real_output, device=DEVICE)

        d_loss_real = criterion_gan(real_output, real_labels)
        fake_target = generator(generator_input)
        fake_input = torch.cat((template, fake_target), dim=1)  # 判別器輸入是 template + fake_target
        fake_output = discriminator(fake_input.detach())
        d_loss_fake = criterion_gan(fake_output, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # ----------------------
        # 訓練生成器
        # ----------------------
        optimizer_g.zero_grad()
        fake_output = discriminator(fake_input)
        g_loss_gan = criterion_gan(fake_output, real_labels)
        g_loss_l1 = criterion_l1(fake_target, target) * 50
        g_loss = g_loss_gan + g_loss_l1
        g_loss.backward()
        optimizer_g.step()

        # 把該批次的損失加總到 epoch 累加器
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    # 每個 Epoch 結束，計算平均損失
    avg_d_loss = epoch_d_loss / len(dataloader)
    avg_g_loss = epoch_g_loss / len(dataloader)
    d_losses.append(avg_d_loss)
    g_losses.append(avg_g_loss)

    # 每個 Epoch 保存生成影像 (只取最後一批)
    save_image(fake_target, os.path.join(SAVE_PATH, f"epoch_{epoch+1}_fake.png"))
    save_image(target,      os.path.join(SAVE_PATH, f"epoch_{epoch+1}_real.png"))
    save_image(template,    os.path.join(SAVE_PATH, f"epoch_{epoch+1}_template.png"))

    # 保存模型
    torch.save(generator.state_dict(), os.path.join(SAVE_PATH, f"generator_epoch_{epoch+1}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_PATH, f"discriminator_epoch_{epoch+1}.pth"))

    print(f"[Epoch {epoch+1}] Average D Loss: {avg_d_loss:.4f}, Average G Loss: {avg_g_loss:.4f}")

print("訓練完成！")

# ----------------------------------
# 取得訓練結束後的損失曲線
# ----------------------------------
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, EPOCHS+1), g_losses, label='Generator Loss', color='blue')
plt.plot(range(1, EPOCHS+1), d_losses, label='Discriminator Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH, 'loss_curve.png'))  # 也可改成 plt.show()

print(f"損失曲線已保存於 {os.path.join(SAVE_PATH, 'loss_curve.png')}")
