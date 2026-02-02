import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import io

# =================配置区域=================
ADV_IMAGES_DIR = './4block_flip_pf_SITIDIM/images'
CSV_PATH = './labels.csv'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# JPEG 压缩质量：通常设为 75（防御与画质的平衡点）
JPEG_QUALITY = 75


# =========================================

# 1. 定义 JPEG 防御变换
class JPEGDefense(object):
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, img_tensor):
        """
        输入: [C, H, W] Tensor (0-1)
        输出: 经过 JPEG 压缩后的 Tensor
        """
        # 1. Tensor 转 PIL (JPEG 是基于图像域的操作)
        unloader = transforms.ToPILImage()
        img_pil = unloader(img_tensor)

        # 2. 在内存中进行 JPEG 压缩
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)

        # 3. 重新读取并转回 Tensor
        compressed_img = Image.open(buffer)
        loader = transforms.ToTensor()
        return loader(compressed_img)


# 2. 数据集类
class AdvDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        self.f_col = self.data_info.columns[0]
        self.l_col = self.data_info.columns[1]

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


# 3. 评估入口
def evaluate():
    print(f"当前设备: {DEVICE} | JPEG 质量: {JPEG_QUALITY}")

    # 加载 PyTorch Hub 的 InceptionV3
    print("正在从 PyTorch Hub 加载 InceptionV3...")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model = model.to(DEVICE)
    model.eval()

    # 标准 Normalize 参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 数据预处理：Resize -> ToTensor -> JPEG 防御
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        JPEGDefense(quality=JPEG_QUALITY)  # 加入 JPEG 变换
    ])

    dataset = AdvDataset(img_dir=ADV_IMAGES_DIR, csv_file=CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0

    print(f"开始评估 JPEG 防御，样本数: {len(dataset)}")

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 归一化后送入模型
            # 注意：transform 中已经完成了 JPEG 压缩
            norm_images = torch.stack([normalize(img) for img in images])

            outputs = model(norm_images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{len(dataloader)} 处理完毕")

    accuracy = 100 * correct / total
    print("\n" + "=" * 40)
    print(f"防御方法: JPEG Compression (Quality={JPEG_QUALITY})")
    print(f"最终 Accuracy: {accuracy:.2f}%")
    print(f"攻击成功率 (ASR): {100 - accuracy:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    evaluate()