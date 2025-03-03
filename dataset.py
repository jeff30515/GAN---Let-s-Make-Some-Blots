import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class WesternBlotDataset(Dataset):
    def __init__(self, template_dir, target_dir, template2_dir=None, transform=None):
        self.template_dir = template_dir
        self.target_dir = target_dir
        self.template2_dir = template2_dir
        self.template_files = sorted(os.listdir(template_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.template2_files = sorted(os.listdir(template2_dir)) if template2_dir else None
        self.transform = transform if transform else transforms.ToTensor()

        # 驗證文件名對應
        self.verify_matching()

    def verify_matching(self):
        # 提取模板和目標的公共 ID
        template_ids = [f.split('BandMask_')[-1].split('.')[0] for f in self.template_files]
        target_ids = [f.split('.')[0] for f in self.target_files]

        # 驗證模板和目標是否匹配
        assert template_ids == target_ids, f"模板和目標的文件名不匹配: {template_ids} vs {target_ids}"

        # 如果有 template2，進一步驗證
        if self.template2_dir:
            template2_ids = [f.split('BasePattern_')[-1].split('.')[0] for f in self.template2_files]
            assert template2_ids == target_ids, f"模板2和目標的文件名不匹配: {template2_ids} vs {target_ids}"

    def __len__(self):
        return len(self.template_files)

    def __getitem__(self, idx):
        template_path = os.path.join(self.template_dir, self.template_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        template = Image.open(template_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        if self.transform:
            template = self.transform(template)
            target = self.transform(target)

        if self.template2_dir and self.template2_files:
            template2_path = os.path.join(self.template2_dir, self.template2_files[idx])
            template2 = Image.open(template2_path).convert('RGB')
            if self.transform:
                template2 = self.transform(template2)
            return template, target, template2

        # 如果未提供 template2，返回兩個值
        return template, target
