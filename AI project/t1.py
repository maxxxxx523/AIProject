import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
set_seed(42)

# 检查设备
device = torch.device("cuda")
print(f"使用设备: {device}")

# 修复后的模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入: 1x28x28, 输出: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 输入: 32x28x28, 输出: 64x28x28
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 每次池化将尺寸减半
        
        # 计算全连接层的输入维度
        # 输入: 1x28x28
        # 经过 conv1 + pool: 28/2 = 14
        # 经过 conv2 + pool: 14/2 = 7
        # 最终特征图尺寸: 64 x 7 x 7
        self.fc_input_size = 64 * 7 * 7  # = 3136
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # 第一层: Conv -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv1(x)))  # 28x28 -> 14x14
        # 第二层: Conv -> ReLU -> Pool  
        x = self.pool(torch.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # 展平特征图
        x = x.view(-1, self.fc_input_size)  # 展平为 [batch_size, 3136]
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 验证模型维度
def test_model_dimensions():
    """测试模型输入输出维度是否正确"""
    model = SimpleCNN()
    test_input = torch.randn(64, 1, 28, 28)  # batch_size=64
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    assert output.shape == (64, 10), "输出维度错误！"
    print("✓ 模型维度测试通过")

# 数据加载函数
def load_data(batch_size=64, val_split=0.1):
    # Training transform: augmentation to handle real-world handwriting variations
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,           # rotate ±15°
            translate=(0.15, 0.15),  # shift up to 15%
            scale=(0.8, 1.2),     # scale 80%-120%
            shear=10              # shear ±10°
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Validation/test transform: no augmentation, just normalize
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    val_size = int(len(full_train_dataset) * val_split)

    # Shuffle indices for a consistent split, then assign val its own transform
    indices = list(range(len(full_train_dataset)))
    rng = random.Random(42)
    rng.shuffle(indices)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    val_dataset_base = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=eval_transform
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(val_dataset_base, val_indices)
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=eval_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# 训练函数
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{total_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'\n测试集准确率: {accuracy:.2f}%')
    
    return accuracy

# 可视化预测结果
def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    
    data, target = next(iter(test_loader))
    data, target = data[:num_images].to(device), target[:num_images]
    
    with torch.no_grad():
        output = model(data)
        _, predicted = output.max(1)
    
    # 计算网格布局
    cols = 5
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = data[i].cpu().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'target: {target[i].item()}\n prediction: {predicted[i].item()}')
        axes[i].axis('off')
        
        if target[i].item() == predicted[i].item():
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training And Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Training Accuracy', marker='o')
    ax2.plot(val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training And Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 手动实现混淆矩阵
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true, pred] += 1
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, num_classes=10):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=[str(i) for i in range(num_classes)],
           yticklabels=[str(i) for i in range(num_classes)],
           xlabel='prediction',
           ylabel='target')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            if value > 0:
                ax.text(j, i, value, ha="center", va="center", 
                       color="white" if value > confusion_matrix.max() / 2 else "black")
    
    ax.set_title('混淆矩阵')
    fig.tight_layout()
    plt.show()

def show_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    cm = compute_confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)
    
    # 打印每个类别的准确率
    print("\nAccuracy of different types:")
    for i in range(10):
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            acc = 100. * correct / total
            print(f"digit {i}: {acc:.2f}% ({correct}/{total})")

# 显示错误分类的样本
def show_misclassified(model, test_loader, device, num_samples=10):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            incorrect_mask = predicted != target
            incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0]
            
            for idx in incorrect_indices:
                if len(misclassified) < num_samples:
                    misclassified.append({
                        'image': data[idx],
                        'true': target[idx].item(),
                        'pred': predicted[idx].item()
                    })
                else:
                    break
            if len(misclassified) >= num_samples:
                break
    
    if misclassified:
        cols = 5
        rows = (len(misclassified) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        axes = axes.ravel()
        
        for i, sample in enumerate(misclassified):
            img = sample['image'].cpu().squeeze()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'target:{sample["true"]} prediction:{sample["pred"]}', color='red')
            axes[i].axis('off')
        
        for i in range(len(misclassified), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'错误分类样本 (共{len(misclassified)}个)', fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("没有发现错误分类的样本！")

# 主函数
def main():
    print("="*50)
    print("手写数字识别项目")
    print("="*50)
    
    # 测试模型维度
    print("\n0. 测试模型维度...")
    test_model_dimensions()
    
    # 超参数
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # 加载数据
    print("\n1. 加载数据...")
    train_loader, val_loader, test_loader = load_data(BATCH_SIZE)
    
    # 创建模型
    print("\n2. 创建模型...")
    model = SimpleCNN().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练
    print("\n3. 开始训练...")
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ 保存最佳模型 (准确率: {val_acc:.2f}%)")
    
    # 测试
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    print("\n4. 测试集评估...")
    test_acc = test_model(model, test_loader, device)
    
    # 可视化
    print("\n5. 生成可视化结果...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    visualize_predictions(model, test_loader, device)
    show_confusion_matrix(model, test_loader, device)
    show_misclassified(model, test_loader, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("\n模型已保存为 'final_model.pth'")
    
    # 总结
    print("\n" + "="*50)
    print("项目总结")
    print("="*50)
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"训练轮数: {EPOCHS}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")

# 极简版
def minimal_main():
    """最简化版本"""
    print("简化版训练开始...")
    
    BATCH_SIZE = 64
    EPOCHS = 5
    
    # 加载数据 (带数据增强)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15),
                                scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 测试
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        accuracy = 100. * correct / len(test_dataset)
        test_accs.append(accuracy)
        print(f'Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}, 测试准确率: {accuracy:.2f}%')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('测试准确率')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n最终测试准确率: {test_accs[-1]:.2f}%")
    print("训练完成！")

# 运行选择
if __name__ == "__main__":
    print("选择运行模式：")
    print("1. 完整版（包含所有可视化）")
    print("2. 极简版（快速训练）")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        main()
    else:
        minimal_main()