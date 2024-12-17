import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

class StockDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame, 
                 seq_length: int = 10, 
                 prediction_horizon: int = 1, 
                 scalers: Optional[Dict] = None,
                 is_training: bool = True):
        """
        初始化股票数据集，包含归一化处理
        
        Args:
            data: 输入的DataFrame，包含所有特征
            seq_length: 序列长度
            prediction_horizon: 预测时间跨度
            scalers: 已有的scaler字典（用于验证集）
            is_training: 是否为训练模式
        """
        self.data = data
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.is_training = is_training
        
        # 定义特征组
        self.price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        self.volume_cols = ['Volume', 'Volume_MA5']
        self.technical_cols = [col for col in data.columns 
                             if col not in self.price_cols + self.volume_cols]
        
        # 初始化或加载 scalers
        if scalers is None and is_training:
            self.scalers = self._init_scalers()
        else:
            self.scalers = scalers
            
        # 归一化数据
        self.normalized_data = self._normalize_data()
        
        # 生成序列起始索引
        self.indices = range(len(data) - seq_length - prediction_horizon + 1)
        
        # 模拟事件数据 (实际使用时需要替换为真实数据)
        self.events = np.zeros((len(data), 1))
        self.time_distances = np.ones((len(data), 1))
    
    def _init_scalers(self) -> Dict:
        """初始化所有特征的 scalers"""
        scalers = {
            'price': RobustScaler().fit(self.data[self.price_cols]),
            'volume': RobustScaler().fit(self.data[self.volume_cols]),
            'technical': StandardScaler().fit(self.data[self.technical_cols])
        }
        return scalers
    
    def _normalize_data(self) -> np.ndarray:
        """对所有特征进行归一化处理"""
        normalized_prices = self.scalers['price'].transform(
            self.data[self.price_cols]
        )
        normalized_volume = self.scalers['volume'].transform(
            self.data[self.volume_cols]
        )
        normalized_technical = self.scalers['technical'].transform(
            self.data[self.technical_cols]
        )
        
        return np.concatenate(
            [normalized_prices, normalized_volume, normalized_technical], 
            axis=1
        )
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length
        target_idx = end_idx + self.prediction_horizon - 1
        
        # 获取输入序列
        sequence = torch.FloatTensor(
            self.normalized_data[start_idx:end_idx]
        )
        
        # 获取目标值（收盘价）
        # 修复：创建一个与price_cols相同维度的数组，全部填充收盘价
        close_price = self.data.iloc[target_idx]['Close']
        target_prices = np.full(len(self.price_cols), close_price)
        normalized_target = self.scalers['price'].transform(
            target_prices.reshape(1, -1)
        )[0, 3]  # 只取收盘价对应的索引
        
        target = torch.FloatTensor([normalized_target])
        
        # 获取事件数据
        events = torch.LongTensor(
            self.events[start_idx:end_idx]
        )
        
        time_distances = torch.FloatTensor(
            self.time_distances[start_idx:end_idx]
        )
        
        return sequence, events, time_distances, target
    
    def get_scalers(self) -> Dict:
        """返回所有scalers，用于验证集和测试集"""
        return self.scalers

    def inverse_transform_price(self, normalized_price: np.ndarray) -> np.ndarray:
        """将归一化的价格转换回原始价格"""
        # 确保输入是 2D 数组
        if normalized_price.ndim == 1:
            normalized_price = normalized_price.reshape(-1, 1)
        
        # 创建完整的价格特征数组
        batch_size = normalized_price.shape[0]
        price_array = np.zeros((batch_size, len(self.price_cols)))
        price_array[:, 3] = normalized_price.flatten()  # 将收盘价放在正确的位置
        
        # 使用 RobustScaler 进行反转换
        original_prices = self.scalers['price'].inverse_transform(price_array)
        return original_prices[:, 3]  # 返回收盘价列

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                train_dataset: StockDataset,  # 添加dataset以便使用scaler
                num_epochs: int = 200,
                device: str = 'cuda'):
    """
    训练模型的主函数，包含对归一化数据的处理
    """
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    
    criterion = CombinedLoss()
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
            
            # 使用归一化后的数据计算损失
            loss = criterion(
                predictions,
                targets,
                data[:, -1, 3]  # 使用归一化后的最后一个时间步的收盘价
            )
            
            # 计算方向准确率（使用归一化后的数据）
            pred_direction = (predictions[:, -1] > data[:, -1, 3]).float()
            target_direction = (targets > data[:, -1, 3]).float()
            direction_accuracy = (pred_direction == target_direction).float().mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            # 转换回原始价格尺度用于展示
            pred_price = train_dataset.inverse_transform_price(
                predictions[:, -1].cpu().detach().numpy()
            )
            target_price = train_dataset.inverse_transform_price(
                targets.cpu().numpy()
            )
            
            train_loss += loss.item()
            train_direction_correct += direction_accuracy.item()
            train_total += 1
            
            progress_bar.set_description(
                f'Epoch {epoch} | Loss: {loss.item():.4f} | '
                f'Direction Acc: {direction_accuracy.item():.4f} | '
                f'Pred Price: {pred_price.mean():.4f} | '
                f'True Price: {target_price.mean():.4f}'
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
                'scalers': train_dataset.get_scalers(),
            }, 'best_model_normalized.pth')
    
        # 早停检查
        if early_stopping(avg_val_loss, avg_val_direction):
            print(f'Early stopping triggered after {epoch} epochs')
            break

def main():

    warnings.filterwarnings("ignore")
    
    # 准备数据
    data = download_and_prepare_data('^IXIC', '2000-01-01', '2024-01-01')
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        data, test_size=0.2, shuffle=False
    )
    
    # 创建训练集，初始化scalers
    train_dataset = StockDataset(train_data, is_training=True)
    
    # 使用训练集的scalers创建验证集
    val_dataset = StockDataset(
        val_data, 
        scalers=train_dataset.get_scalers(),
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型并训练
    model = StockPredictor(
        input_dim=21,
        hidden_dim=128,
        event_dim=32,
        num_event_types=10,
        feature_groups=prepare_feature_groups()
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Starting training...")
    
    train_model(
        model, 
        train_loader, 
        val_loader,
        train_dataset,  # 传入dataset以便使用scaler
        device=device
    )

if __name__ == '__main__':
    main()