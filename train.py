import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from data import download_and_prepare_data
from model import (
    StockPredictor,
    prepare_feature_groups
)

class StockDataset(Dataset):
    def __init__(self, data, seq_length=10, prediction_horizon=1):
        self.data = data
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        
        # 生成序列起始索引
        self.indices = range(len(data) - seq_length - prediction_horizon + 1)
        
        # 模拟事件数据 (这里需要根据实际情况修改)
        self.events = np.zeros((len(data), 1))  # 简单起见先用零
        self.time_distances = np.ones((len(data), 1))  # 简单起见先用1
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length
        
        # 获取输入序列
        sequence = torch.FloatTensor(
            self.data.iloc[start_idx:end_idx].values
        )
        
        # 获取目标值 (预测未来的收盘价)
        target_idx = end_idx + self.prediction_horizon - 1
        target = torch.FloatTensor([
            self.data.iloc[target_idx]['Close']
        ])
        
        # 获取事件数据
        events = torch.LongTensor(
            self.events[start_idx:end_idx]
        )
        
        time_distances = torch.FloatTensor(
            self.time_distances[start_idx:end_idx]
        )
        
        return sequence, events, time_distances, target

# 新添加：组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta   # 方向预测权重
        self.gamma = gamma # 连续性权重
        
    def forward(self, predictions, targets, prev_price):
        batch_size = predictions.size(0)
        
        # MSE损失
        mse_loss = F.mse_loss(predictions[:, -1], targets)
        
        # 方向预测损失
        pred_diff = predictions[:, -1] - prev_price
        target_diff = targets - prev_price
        pred_direction = (pred_diff > 0).float()
        target_direction = (target_diff > 0).float()
        direction_loss = F.binary_cross_entropy_with_logits(
            pred_direction, target_direction
        )
        
        # 连续性损失
        smoothness_loss = torch.mean(torch.abs(predictions[:, -1] - prev_price))
        
        return (self.alpha * mse_loss + 
                self.beta * direction_loss + 
                self.gamma * smoothness_loss)

# 新添加：早停回调
class EarlyStoppingCallback:
    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.stop = False
        
    def __call__(self, val_loss, val_acc):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            if val_acc > self.best_acc:  # 同时考虑准确率的提升
                self.best_loss = val_loss
                self.best_acc = val_acc
                self.counter = 0
            
        return self.stop

# 修改：训练函数
def train_model(model, train_loader, val_loader, num_epochs=200, device='cuda'):
    model = model.to(device)
    
    # 修改：使用新的优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # 修改：使用新的学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    
    # 修改：使用新的损失函数
    criterion = CombinedLoss()
    
    # 修改：使用新的早停策略
    early_stopping = EarlyStoppingCallback(patience=30)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_direction_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader)
        for batch_idx, (data, events, time_distances, targets) in enumerate(progress_bar):
            data = data.to(device)
            events = events.to(device)
            time_distances = time_distances.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(data, events, time_distances)
            
            # 修改：使用组合损失
            loss = criterion(
                predictions,
                targets,
                data[:, -1, 3]  # 使用最后一个时间步的收盘价
            )
            
            # 计算方向准确率
            pred_direction = (predictions[:, -1] > data[:, -1, 3]).float()
            target_direction = (targets > data[:, -1, 3]).float()
            direction_accuracy = (pred_direction == target_direction).float().mean()
            
            loss.backward()
            # 修改：增加梯度裁剪的阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 修改：更新学习率
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            train_loss += loss.item()
            train_direction_correct += direction_accuracy.item()
            train_total += 1
            
            progress_bar.set_description(
                f'Epoch {epoch} | Loss: {loss.item():.4f} | '
                f'Direction Acc: {direction_accuracy.item():.4f}'
            )
        
        avg_train_loss = train_loss / train_total
        avg_train_direction = train_direction_correct / train_total
        
        # 验证阶段（使用相同的损失函数）
        model.eval()
        val_loss = 0
        val_direction_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, events, time_distances, targets in val_loader:
                data = data.to(device)
                events = events.to(device)
                time_distances = time_distances.to(device)
                targets = targets.to(device)
                
                predictions = model(data, events, time_distances)
                loss = criterion(predictions, targets, data[:, -1, 3])
                
                pred_direction = (predictions[:, -1] > data[:, -1, 3]).float()
                target_direction = (targets > data[:, -1, 3]).float()
                direction_accuracy = (pred_direction == target_direction).float().mean()
                
                val_loss += loss.item()
                val_direction_correct += direction_accuracy.item()
                val_total += 1
        
        avg_val_loss = val_loss / val_total
        avg_val_direction = val_direction_correct / val_total
        
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {avg_train_loss:.4f} | '
              f'Train Direction Acc: {avg_train_direction:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Direction Acc: {avg_val_direction:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_direction_acc': avg_val_direction,
            }, 'best_model.pth')
        
        # 修改：使用新的早停策略
        if early_stopping(avg_val_loss, avg_val_direction):
            print(f'Early stopping triggered after {epoch} epochs')
            break

# 主训练流程
def main():
    # 准备数据
    data = download_and_prepare_data('AAPL', '1980-01-01', '2024-01-01')
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        data, test_size=0.2, shuffle=False
    )
    
    # 创建数据加载器
    train_dataset = StockDataset(train_data)
    val_dataset = StockDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # 创建模型
    model = StockPredictor(
        input_dim=21,         # 总特征数
        hidden_dim=128,      # 隐藏层维度
        event_dim=32,        # 事件嵌入维度
        num_event_types=10,  # 事件类型数量
        feature_groups=prepare_feature_groups()
    )
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device.type)
    print("String training")
    train_model(model, train_loader, val_loader, device=device)

if __name__ == '__main__':
    main()