import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import timm
from scipy.stats import mode
import numpy as np

# =================配置区域=================
ADV_IMAGES_DIR = './SSA-TI-DIM'
CSV_PATH = './labels_adv.csv'
BATCH_SIZE = 1  # 因为每张图要重复 N 次，建议设小一点防止显存溢出
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RS 核心参数
SIGMA = 0.25  # 高斯噪声的标准差 (常用 0.12, 0.25, 0.5)
N_SAMPLES = 100  # 采样次数 (评估 ASR 时 100 即可，理论认证通常用 10000)
MODEL_NAME = 'inception_v3.tf_in1k'


# =========================================

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


def evaluate_rs():
    print(f"正在加载模型: {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
    model.eval()

    # 获取模型归一化配置
    config = timm.data.resolve_model_data_config(model)
    normalize = transforms.Normalize(mean=config['mean'], std=config['std'])

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = AdvDataset(img_dir=ADV_IMAGES_DIR, csv_file=CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct, total = 0, 0
    print(f"开始 RS 评估 (Sigma={SIGMA}, N={N_SAMPLES})...")

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)  # [B, 3, 299, 299]
            curr_batch_size = images.size(0)

            # --- RS 核心逻辑：对 Batch 中的每张图进行采样 ---
            # 为了提速，我们将单张图扩展为 N 张，一次性并行通过模型
            # 扩展形状: [B * N, 3, 299, 299]
            expanded_images = images.repeat_interleave(N_SAMPLES, dim=0)

            # 加入高斯噪声
            noise = torch.randn_like(expanded_images) * SIGMA
            noisy_images = expanded_images + noise
            noisy_images = torch.clamp(noisy_images, 0, 1)  # 保证像素合法性

            # 归一化并推理
            norm_images = normalize(noisy_images)
            outputs = model(norm_images)
            _, predicted = torch.max(outputs, 1)  # [B * N]

            # 重新整理预测结果并进行多数投票
            # 将 [B * N] 变回 [B, N]
            votes = predicted.view(curr_batch_size, N_SAMPLES).cpu().numpy()

            # 统计每一行的众数 (即投票结果)
            final_predictions, _ = mode(votes, axis=1)
            final_predictions = torch.from_numpy(final_predictions.flatten()).to(DEVICE)

            total += labels.size(0)
            correct += (final_predictions == labels.to(DEVICE)).sum().item()

            if (i + 1) % 5 == 0:
                print(f"进度: {i + 1}/{len(dataloader)} | 当前 Acc: {100 * correct / total:.2f}%")

    acc = 100 * correct / total
    print("\n" + "=" * 50)
    print(f"Randomized Smoothing (RS) 防御结果:")
    print(f"噪声强度 Sigma: {SIGMA}")
    print(f"采样次数 N: {N_SAMPLES}")
    print(f"最终 Accuracy: {acc:.2f}%")
    print(f"攻击成功率 (ASR): {100 - acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    evaluate_rs()