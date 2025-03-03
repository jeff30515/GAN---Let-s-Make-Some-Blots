from dataset import WesternBlotDataset
from torchvision import transforms

# 設置目錄
template_dir = "data/templates"
target_dir = "data/targets"
template2_dir = "data/template2"  # 確保目錄存在

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = WesternBlotDataset(template_dir, target_dir, template2_dir, transform=transform)


# 測試加載
for i, data in enumerate(dataset):
    if len(data) == 3:
        template, target, template2 = data
    else:
        template, target = data
    print(f"Loaded sample {i+1}")
