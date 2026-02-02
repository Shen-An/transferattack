import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from robustbench.utils import load_model

# =================配置区域=================
# 你的图片文件夹路径
ADV_IMAGES_DIR = './SSA-TI-DIM'
# 你的 label.csv 路径
CSV_PATH = './labels_adv.csv'
# 批次大小 (如果显存不够，改小一点，比如 32 或 16)
BATCH_SIZE = 16
# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================

class AdvDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        # 读取 csv，不包含表头的话 header=None，有表头则 header=0
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

        # 简单处理列名逻辑
        if 'filename' in self.data_info.columns:
            self.f_col = 'filename'
            self.l_col = 'label'
        else:
            self.f_col = self.data_info.columns[0]
            self.l_col = self.data_info.columns[1]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = str(self.data_info.iloc[idx][self.f_col])
        img_path = os.path.join(self.img_dir, img_name)
        label = int(self.data_info.iloc[idx][self.l_col])

        # 确保转为 RGB
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_at():
    print(f"当前设备: {DEVICE}")

    # 1. 定义预处理
    # RobustBench 的模型期望输入是 [0, 1] 的 Tensor
    # 它会在模型内部做 Normalization，所以这里只需要 Resize 和 ToTensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 2. 准备数据
    if not os.path.exists(ADV_IMAGES_DIR):
        print(f"错误：找不到文件夹 {ADV_IMAGES_DIR}")
        return

    dataset = AdvDataset(img_dir=ADV_IMAGES_DIR, csv_file=CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"已加载 {len(dataset)} 张图片。")

    # 3. 自动下载并加载 AT 模型
    print("正在通过 RobustBench 加载 AT 模型 (Salman2020Do_R50)...")
    print("如果是第一次运行，会自动下载模型权重 (约 100MB+)...")

    # model_name='Salman2020Do_R50' 是最常用的 ImageNet AT 基准
    # threat_model='Linf' 代表这是针对 Linf 攻击训练的模型
    try:
        # model = load_model(model_name='Salman2020Do_R50', dataset='imagenet', threat_model='Linf')
        model = load_model(model_name='Engstrom2019Robustness', dataset='imagenet', threat_model='Linf')
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已安装 robustbench: pip install robustbench")
        return

    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    print("开始评估...")

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{len(dataloader)} 处理完毕")

    acc = 100 * correct / total
    print("\n========================================")
    print(f"AT 模型 (Salman2020) 评估结果:")
    print(f"样本总数: {total}")
    print(f"模型鲁棒准确率 (Robust Accuracy): {acc:.2f}%")
    print(f"攻击成功率 (ASR): {100 - acc:.2f}%")
    print("========================================")


if __name__ == '__main__':
    evaluate_at()