import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import os
import random

# =================配置区域=================
ADV_IMAGES_DIR = './4block_flip_pf_SITIDIM/images'
CSV_PATH = './labels.csv'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# R&P 参数 (对应你代码中的默认值)
IMAGE_RESIZE = 331


# =========================================

# 1. 适配你的 labels_adv.csv 的数据集类
class AdvDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        # 自动识别列名 (第一列文件名，第二列标签)
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
        return image, label, img_name


# 2. 维持你原文的 R&P 核心逻辑
def process_and_defend_batch(images):
    """
    输入是 [B, 3, 299, 299] 的 Tensor (已经 ToTensor 但未 Normalize)
    """
    batch_size = images.shape[0]
    defended_batch = []

    for i in range(batch_size):
        img = images[i]

        # --- 原文逻辑: 随机水平翻转 ---
        if random.random() > 0.5:
            img = torch.flip(img, [2])

        # --- 原文逻辑: 随机缩放 (310 到 331) ---
        resize_shape = random.randint(310, IMAGE_RESIZE)
        # 注意: interpolate 需要 [1, C, H, W]
        resized_image = F.interpolate(img.unsqueeze(0), size=(resize_shape, resize_shape),
                                      mode='nearest').squeeze(0)

        # --- 原文逻辑: 随机填充到 331x331 ---
        padding_size = 331 - resize_shape
        padding_left = random.randint(0, padding_size)
        padding_top = random.randint(0, padding_size)
        padding_right = padding_size - padding_left
        padding_bottom = padding_size - padding_top

        padded_image = F.pad(resized_image,
                             (padding_left, padding_right, padding_top, padding_bottom),
                             "constant", 0)

        # --- 关键补充: InceptionV3 Hub 模型必须接收 299x299 ---
        # 原文逻辑中填充到了 331，输入模型前必须 Resize 回 299
        final_image = F.interpolate(padded_image.unsqueeze(0), size=(299, 299),
                                    mode='nearest').squeeze(0)

        defended_batch.append(final_image)

    return torch.stack(defended_batch)


# 3. 评估入口
def evaluate():
    print(f"当前设备: {DEVICE}")

    # 加载 PyTorch Hub 的 InceptionV3 模型
    # 对应 https://pytorch.org/hub/pytorch_vision_inception_v3/
    print("正在从 PyTorch Hub 加载 InceptionV3...")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model = model.to(DEVICE)
    model.eval()

    # 标准 Normalize 参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 数据加载 (只做 Resize 和 ToTensor，防御逻辑在循环里做)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = AdvDataset(img_dir=ADV_IMAGES_DIR, csv_file=CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0

    print(f"开始评估，样本总数: {len(dataset)}")

    with torch.no_grad():
        for i, (images, labels, filenames) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # --- 应用 R&P 防御 ---
            # 直接在 Tensor 上操作，保留你原文的随机翻转、缩放、填充逻辑
            defended_images = process_and_defend_batch(images)

            # --- 归一化后送入模型 ---
            norm_images = torch.stack([normalize(img) for img in defended_images])

            # InceptionV3 输出包含主输出和辅助输出(测试时只需主输出)
            outputs = model(norm_images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{len(dataloader)} 处理完毕, 当前 Acc: {100 * correct / total:.2f}%")

    final_acc = 100 * correct / total
    print("\n" + "=" * 40)
    print(f"防御方法: R&P (Random Resizing & Padding)")
    print(f"基础模型: PyTorch Hub InceptionV3")
    print(f"最终 Accuracy: {final_acc:.2f}%")
    print(f"攻击成功率 (ASR): {100 - final_acc:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    evaluate()