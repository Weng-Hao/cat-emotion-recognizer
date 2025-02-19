import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from data_argument import load_data, get_data_transforms  # 从data_argument.py导入数据处理函数

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#这边我在苹果电脑上是"cuda:0",之前在Windows上面跑貌似是"cuda:1"，请注意！！！！

data_dir = 'path_to_your_dataset'  
# 替换为你的数据集路径

input_size = (512, 512)  # 图像尺寸
num_classes = 8  # 情绪的类别数，作为一个向量，我这里先假设设置8种情绪

# CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1) # 这里有 8*3*3*3 = 216 (+8) 个参数
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2) # 这里有 8*16*5*5 = 3200 (+16)个参数
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # 这里有 16*8*3*3 = 1152 (+8)个参数
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0) # 完成/2的池化
        self.fc1 = nn.Linear(8 * 8 * 8, 16)  # 根据卷积层后的输出尺寸调整，这里有 8*8*8*16 = 8192个参数
        self.fc2 = nn.Linear(16, num_classes) #这里参数量大致在64到256之间
        self.relu = nn.ReLU()
    '''
    根据以上计算，参数量一共有13000左右，我们用5000张图片进行数据增强到3-4w即可
    '''

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 输入3个通道的RGB，输出8个通道的卷积，再依次池化变成8*128*128的内容
        x = self.pool(self.relu(self.conv2(x)))  # 输入8个通道，输出16个通道的卷积，再池化变成16*32*32的内容
        x = self.pool(self.relu(self.conv3(x)))  # 输入16个通道，输出8个通道的卷积，再池化变成8*8*8的内容
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    since = time.time()

    val_acc_history = []
    best_model_wts = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个 epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = dataloaders['train']
            else:
                model.eval()   # 设置模型为评估模式
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad() # 清空梯度

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# 主函数
def main():
    batch_size = 32 # 32个作为一组进行训练，batch越大GPU占用越多，训练时间越短

    train_transform, val_transform = get_data_transforms()
    train_loader, val_loader = load_data(data_dir, train_transform, val_transform, batch_size)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    model = SimpleCNN(num_classes).to(device)  # 使用 CNN，把CNN加载到GPU上

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 开始训练
    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")

if __name__ == "__main__":
    main()
