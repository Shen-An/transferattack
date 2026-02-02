import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
# 使用 HuggingFace 的 diffusers 库来简化扩散模型调用
# 如果之前的报错，请改用以下方式：
from diffusers.models import UNet2DModel
from diffusers.schedulers import DDPMScheduler
# =================配置区域=================
ADV_IMAGES_DIR = './SSA-TI-DIM'
CSV_PATH = './labels_adv.csv'
BATCH_SIZE = 4  # 扩散模型非常吃显存，建议设小一点
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DiffPure 参数
# t 代表加噪的步数。t 越大，净化能力越强，但图像内容改变也越大。
# 论文中常用的标准是总步数的 10% 左右（例如 1000 步里的第 100 步）。
PURIFY_STEP = 100


# =========================================

class DiffPureDefense(nn.Module):
    """
    简化的 DiffPure 防御逻辑
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        # 加载一个预训练的扩散模型 (这里以 ImageNet 预训练的 DDPM 为例)
        # 注意：实际学术复现通常使用 Guided Diffusion 的预训练权重
        model_id = "google/ddpm-celebahq-256"  # 示例 ID，实际应使用 ImageNet 权重
        try:
            self.unet = UNet2DModel.from_pretrained(model_id).to(device)
            self.scheduler = DDPMScheduler.from_pretrained(model_id)
        except:
            print("请联网或检查 diffusers 库以加载预训练权重")
            self.unet = None

    def forward(self, x):
        if self.unet is None: return x

        batch_size = x.shape[0]
        # 1. 确定净化的起始步数 (Forward Add Noise)
        t = torch.tensor([PURIFY_STEP] * batch_size).long().to(self.device)
        noise = torch.randn_like(x).to(self.device)
        x_noisy = self.scheduler.add_noise(x, noise, t)

        # 2. 逆向扩散净化 (Reverse Diffusion Loop)
        # 为了速度，这里演示简单的 1-step 净化逻辑 (实际 DiffPure 会跑多步循环)
        with torch.no_grad():
            for i in reversed(range(0, PURIFY_STEP)):
                step_t = torch.tensor([i] * batch_size).long().to(self.device)
                model_output = self.unet(x_noisy, step_t).sample
                x_noisy = self.scheduler.step(model_output, i, x_noisy).prev_sample

        return x_noisy


class AdvDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        self.f_col = 'filename' if 'filename' in self.data_info.columns else self.data_info.columns[0]
        self.l_col = 'label' if 'label' in self.data_info.columns else self.data_info.columns[1]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = str(self.data_info.iloc[idx][self.f_col])
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.data_info.iloc[idx][self.l_col])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_diffpure():
    print(f"当前设备: {DEVICE}")

    # 预处理 (扩散模型通常期望输入在 [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 扩散模型通常是 256x256
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = AdvDataset(img_dir=ADV_IMAGES_DIR, csv_file=CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载底座分类模型
    classifier = models.resnet50(pretrained=True).to(DEVICE).eval()

    # 加载净化器
    purifier = DiffPureDefense(DEVICE)

    correct = 0
    total = 0

    print("开始 DiffPure 净化评估 (这可能比较慢)...")
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 第一步：净化 (DiffPure)
        purified_images = purifier(images)

        # 第二步：将净化后的图 Resize 回分类器的大小 (224) 并预测
        input_for_cls = F.interpolate(purified_images, size=(224, 224), mode='bilinear')

        with torch.no_grad():
            outputs = classifier(input_for_cls)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    print(f"DiffPure ASR: {100 - (100 * correct / total):.2f}%")


if __name__ == '__main__':
    import torch.nn.functional as F

    evaluate_diffpure()