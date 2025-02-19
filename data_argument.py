
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

data_dir = 'path_to_your_dataset'  

# 定义数据增强transforms的总类
def get_data_transforms():
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 将图像调整到 512x512
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色变换，感觉不太重要，我先注掉
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则归一化
    ])

    # 验证集无需数据增强，只需调整大小并归一化
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# 加载数据集并进行数据增强
def load_data(data_dir, train_transform, val_transform, batch_size=32, val_split=0.1, random_seed=42):
    
    # 加载数据集，并使用 ImageFolder
    dataset = ImageFolder(root=data_dir, transform=None)  # 先加载原始数据，不应用转换
    dataset_classes = dataset.classes  # 获取类别列表

    # 获取数据总量
    total_samples = len(dataset)
    indices = list(range(total_samples))

    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed)

    # 定义采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 创建数据集加载器
    train_dataset = ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = ImageFolder(root=data_dir, transform=val_transform)

    # 使用 sampler 重新生成加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader, dataset_classes

def main():
    return 0

if __name__ == "__main__":
    main()

'''
数据集要求是以下格式
dataset/
├── 开心/
│   ├── cat0000.jpg
│   ├── cat0001.jpg
│   ├── cat0002.jpg
│   ├── cat0003.jpg
│   └── ...
├── ...
├── 伤心/
│   ├── cat0F38.jpg
│   ├── cat0F39.jpg
│   ├── cat0F3A.jpg
│   ├── cat0F3B.jpg
│   └── ...
└── ...
'''
