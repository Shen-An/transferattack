import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os, pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

paths = {
    "img_root": r"../../BSRpurified_images/",
    "label_csv": r"./labels_adv.csv",
    # 如果你确实要加载本地权重文件，填入路径；否则留空
    "weights_file": ""
}

# 使用 torchvision 官方 ImageNet 预训练权重
source_model = models.inception_v3(
    weights=models.Inception_V3_Weights.IMAGENET1K_V1,  # 官方权重
    aux_logits=True
).to(device).eval()

# 如果需要加载自定义权重文件，取消注释并提供路径
# if paths["weights_file"]:
#     state_dict = torch.load(paths["weights_file"], map_location=device)
#     source_model.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

label_df = pd.read_csv(paths["label_csv"])
total_num = len(label_df)
print(f"Successfully read CSV, total {total_num} images")

success_count, correct_count = 0, 0

with torch.no_grad():
    for _, row in tqdm(label_df.iterrows(), desc='Predicting', total=total_num):
        img_filename = row["filename"]
        true_label = int(row["label"])  # 注意：应是 ImageNet 的 0..999 标签
        img_path = os.path.join(paths["img_root"], img_filename)

        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            logits = source_model(x)  # 单输出
            pred_label = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())
            success_count += 1
            if pred_label == true_label:
                correct_count += 1
        except Exception:
            continue

pred_accuracy = (correct_count / success_count) * 100 if success_count > 0 else 0
print("\n=== 核心预测指标 ===")
print(f"总样本数: {total_num}")
print(f"成功预测数: {success_count}")
print(f"预测准确率: {pred_accuracy:.2f}%")