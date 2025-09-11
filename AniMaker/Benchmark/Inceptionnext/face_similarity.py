import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.models import create_model, load_checkpoint

output_dir = 'Benchmark/results'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'face-sim.txt')

# 定义数据加载器
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='Benchmark/Inceptionnext/data/eval', transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 初始化人脸检测和特征提取模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_model('inception_next_tiny', pretrained=False)
checkpoint_path = 'Benchmark/Inceptionnext/output/train/20250308-124611-inceptionnext_tiny-224/checkpoint-50.pth.tar'
load_checkpoint(model, checkpoint_path)
model = model.to(device)
model.eval()

# 定义数据加载器
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='Benchmark/Inceptionnext/data/eval', transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
aligned = []
names = []


with torch.no_grad():
    for images, y in loader:
        aligned.append(images)
        names.append(y.item())

# Extract features
aligned = torch.cat(aligned).to(device)
embeddings = model(aligned).detach().cpu()

# 计算相似度
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
dists_array = np.array(dists)
df_dists = pd.DataFrame(dists_array, columns=[f"Class_{label}" for label in names], index=[f"Class_{label}" for label in names])
print("相似度矩阵 (dists_array):")
print(df_dists)

# 类内相似度计算
inter_class_similarity = []
inter_class_similarity.append(np.mean(dists_array[:5, :5]))  # 计算类内平均相似度
inter_class_similarity.append(np.mean(dists_array[5:, 5:]))  # 计算类内平均相似度

# 类间相似度计算
intra_class_similarity = np.mean(dists_array[:5, 5:])  # 计算类间平均相似度

# 计算比值
inter_class_mean = sum(inter_class_similarity) / len(inter_class_similarity)  # 类内相似度的平均值
intra_class_mean = intra_class_similarity

similarity_ratio = inter_class_mean / intra_class_mean

# 打印结果
print(f"类内相似度平均值: {inter_class_mean:.4f}")
print(f"类间相似度平均值: {intra_class_mean:.4f}")
print(f"类内相似度与类间相似度的比值: {similarity_ratio:.4f}")

# 保存结果到文件
with open(output_path, 'w') as f:
    f.write("相似度矩阵:\n")
    f.write(df_dists.to_string())
    f.write("\n\n")
    f.write(f"类内相似度平均值: {inter_class_mean:.4f}\n")
    f.write(f"类间相似度平均值: {intra_class_mean:.4f}\n")
    f.write(f"类内相似度与类间相似度的比值: {similarity_ratio:.4f}\n")

print(f"结果已保存到: {output_path}")