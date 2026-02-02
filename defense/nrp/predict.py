from tqdm import tqdm
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径配置（修正原路径多余符号）
paths = {
    "img_root": r"E:/TransferAttack/TransferAttack/help/purified",
    "label_csv": r"E:/TransferAttack/TransferAttack/help/labels_adv.csv"
}

# 加载InceptionV3模型
source_model = models.inception_v3(pretrained=False, aux_logits=True).to(device)
try:
    state_dict = torch.load(
        r"E:/TransferAttack/TransferAttack/help/model/inception_v3_google-0cc3c7bd.pth",
        map_location=device,
        weights_only=False
    )
    source_model.load_state_dict(state_dict)
    print("Model weights loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError("Model weights file not found! Please check the path.")
source_model.eval()

# InceptionV3专用图像预处理
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取标签文件

label_df = pd.read_csv(paths["label_csv"])
# label_df = label_df.sample(n=2)  # 测试时可取消注释
total_num = len(label_df)
print(f"Successfully read CSV, total {total_num} images")


# 仅统计核心指标
success_count = 0  # 成功完成预测的样本数
correct_count = 0  # 预测正确的样本数

# 批量预测
with torch.no_grad():
    for idx, row in tqdm(label_df.iterrows(), desc='Predicting', total=total_num):
        img_filename = row["filename"]
        true_label = int(row["label"])
        img_path = os.path.join(paths["img_root"], img_filename)

        try:
            # 图像读取与预处理
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            # 模型推理
            outputs = source_model(img_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 取主输出

            # 计算预测结果
            pred_label = torch.argmax(torch.softmax(outputs, dim=1), dim=1).item()
            success_count += 1

            # 统计正确数
            if pred_label == true_label:
                correct_count += 1

        except Exception:
            # 忽略预测失败的样本（不统计到success_count）
            continue

pred_accuracy = (correct_count / success_count) * 100 if success_count > 0 else 0  # 预测准确率

print("\n=== 核心预测指标 ===")
print(f"总样本数: {total_num}")
print(f"成功预测数: {success_count}")
print(f"预测准确率: {pred_accuracy:.2f}%")