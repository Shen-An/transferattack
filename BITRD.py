import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import timm

# =================配置区域=================
ADV_IMAGES_DIR = './4block_flip_pf_SITIDIM/images'
CSV_PATH = './labels.csv'
BATCH_SIZE = 16
BIT_DEPTH = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 选择模型权重版本：'inception_v3.tf_in1k' 是 TF 移植版
MODEL_NAME = 'inception_v3.tf_in1k'


# =========================================

class BitReduction(object):
    def __init__(self, bits=3):
        self.bits = bits
        self.levels = 2 ** bits

    def __call__(self, img_tensor):
        return torch.round(img_tensor * (self.levels - 1)) / (self.levels - 1)


class AdvDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir, self.data_info, self.transform = img_dir, pd.read_csv(csv_file), transform
        self.f_col, self.l_col = self.data_info.columns[0], self.data_info.columns[1]

    def __len__(self): return len(self.data_info)

    def __getitem__(self, idx):
        img_name = str(self.data_info.iloc[idx][self.f_col])
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.data_info.iloc[idx][self.l_col])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label


def evaluate():
    print(f"正在加载 timm 模型: {MODEL_NAME}...")

    # 1. 加载模型 (timm 会自动处理 TF 权重的转换和命名)
    # 这样你就不用去弄那个该死的 .npy 了
    model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
    model.eval()

    # 2. 获取该模型指定的归一化参数 (tf_in1k 通常是 mean=0.5, std=0.5)
    config = timm.data.resolve_model_data_config(model)
    print(f"检测到模型预处理配置: {config}")

    # 手动定义 transform 确保 Bit-RD 在归一化之前
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        BitReduction(bits=BIT_DEPTH)
    ])

    # 注意：TF 风格模型的均值和标准差
    normalize = transforms.Normalize(mean=config['mean'], std=config['std'])

    dataset = AdvDataset(img_dir=ADV_IMAGES_DIR, csv_file=CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct, total = 0, 0
    print("开始评估...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # 先 Bit-RD (已在 transform)，再归一化
            norm_images = torch.stack([normalize(img) for img in images])

            outputs = model(norm_images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0: print(f"进度: {i + 1}/{len(dataloader)}")

    acc = 100 * correct / total
    print("\n" + "=" * 50)
    print(f"使用 timm ({MODEL_NAME}) 的结果:")
    print(f"Accuracy: {acc:.2f}%")
    print(f"攻击成功率 (ASR): {100 - acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    evaluate()