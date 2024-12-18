# fusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
import numpy as np
import pandas as pd
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN import ResidualFinancialBlock, EnhancedStockDataset
from model import StockLSTMCell, EventProcessor, EnhancedTMDO, prepare_feature_groups
from train import EnhancedCombinedLoss, train_enhanced_model, generate_event_data, combine_stock_data
from data import load_data_from_csv, download_and_prepare_data
from sklearn.model_selection import train_test_split

class FusionStockDataset(Dataset):
    """
    融合数据集类,同时支持CNN和LSTM特性
    """
    def __init__(self, data, events, sequence_length=250, prediction_horizon=5):
        """
        Args:
            data: DataFrame containing stock data
            events: Event data matrix
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of days to predict ahead
        """
        # 定义特征顺序
        self.feature_order = [
            'Close', 'Open', 'High', 'Low',  # 价格指标
            'MA5', 'MA20',                   # 均线指标
            'MACD', 'MACD_Signal', 'MACD_Hist',  # MACD族
            'RSI', 'Upper', 'Middle', 'Lower',    # RSI和布林带
            'CRSI', 'Kalman_Price', 'Kalman_Trend',  # 高级指标
            'FFT_21', 'FFT_63',              # FFT指标
            'Volume', 'Volume_MA5'           # 成交量指标
        ]
        
        self.data = data
        self.events = events
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 组织2D特征数据
        self.feature_data = self._organize_features()
        
        # 计算时间距离矩阵
        self.time_distances = self._compute_time_distances()
        
    def _organize_features(self):
        """组织特征为2D张量格式"""
        # 检查特征完整性
        for feature in self.feature_order:
            if feature not in self.data.columns:
                raise ValueError(f"Missing feature: {feature}")
        
        # 提取并堆叠特征
        feature_data = []
        for feature in self.feature_order:
            feature_data.append(self.data[feature].values)
            
        return torch.FloatTensor(np.stack(feature_data, axis=0))
    
    def _compute_time_distances(self):
        """计算事件时间距离"""
        distances = np.zeros((len(self.data), 1))
        last_event_idx = -1
        
        for i in range(len(self.data)):
            if self.events[i].any():
                last_event_idx = i
            distances[i] = i - last_event_idx if last_event_idx != -1 else 999
            
        return distances
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # 获取特征序列
        feature_seq = self.feature_data[:, idx:idx + self.sequence_length]
        
        # 获取DataFrame格式的序列(用于LSTM)
        df_seq = self.data.iloc[idx:idx + self.sequence_length]
        sequence = torch.FloatTensor(df_seq.values)
        
        # 获取事件数据
        events = torch.LongTensor(
            self.events[idx:idx + self.sequence_length]
        )
        
        # 获取时间距离
        time_distances = torch.FloatTensor(
            self.time_distances[idx:idx + self.sequence_length]
        )
        
        # 计算目标值
        future_idx = idx + self.sequence_length + self.prediction_horizon - 1
        future_price = self.feature_data[0, future_idx]  # Close price
        current_price = self.feature_data[0, idx + self.sequence_length - 1]
        
        # 计算收益率
        returns = (future_price - current_price) / current_price
        
        # 生成分类标签
        if returns < -0.02:
            label = 0  # 下跌
        elif returns > 0.02:
            label = 2  # 上涨
        else:
            label = 1  # 横盘
            
        # 回归目标
        target = torch.FloatTensor([self.data.iloc[future_idx]['Close']])
        
        return {
            'feature_seq': feature_seq,      # CNN使用
            'sequence': sequence,            # LSTM使用
            'events': events,                # 事件数据
            'time_distances': time_distances,  # 时间距离
            'label': label,                  # 分类标签
            'target': target,                # 回归目标
            'current_price': current_price   # 当前价格
        }

class ATLASCNNFusion(nn.Module):
    """
    ATLAS-CNN融合模型
    结合CNN的空间特征提取能力和ATLAS(改进版LSTM)的时序建模能力
    """
    def __init__(self, input_dim, hidden_dim, event_dim, num_event_types, feature_groups):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # CNN特征提取
        self.cnn_branch = nn.Sequential(
            ResidualFinancialBlock(hidden_dim, input_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # ATLAS分支
        self.atlas = StockLSTMCell(input_dim, hidden_dim)
        
        # 事件处理器
        self.event_processor = EventProcessor(
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            num_event_types=num_event_types
        )
        
        # TMDO特征提取
        self.tmdo = EnhancedTMDO(input_dim)
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 市场状态感知
        self.market_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def _apply_cnn(self, x):
        """CNN特征提取"""
        batch_size, seq_len, features = x.shape
        # 调整维度以适应CNN
        x = x.unsqueeze(1)  # [batch, 1, seq, features]
        cnn_features = self.cnn_branch(x)
        return cnn_features.squeeze(1)  # [batch, seq, hidden]
        
    def forward(self, x, events, time_distances):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # CNN特征提取
        cnn_features = self._apply_cnn(x)
        
        # TMDO特征提取
        tmdo_features, lap_features = self.tmdo(x)
        
        # 初始化ATLAS状态
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        combined_features = []
        
        for t in range(seq_len):
            # 获取当前时间步的特征
            current_x = x[:, t]
            current_cnn = cnn_features[:, t]
            current_tmdo = tmdo_features[:, t]
            current_events = events[:, t]
            current_distances = time_distances[:, t]
            
            # 处理事件影响
            event_impact = self.event_processor(
                current_events,
                h.unsqueeze(1),  # 添加seq_len维度
                current_distances
            )
            
            # ATLAS步进
            h, c = self.atlas(
                current_x,
                h, c,
                current_cnn,  # 使用CNN特征作为indicators
                event_impact
            )
            
            # 交叉注意力增强
            h_enhanced, _ = self.cross_attention(
                h.unsqueeze(1),
                current_cnn.unsqueeze(1),
                current_cnn.unsqueeze(1)
            )
            h_enhanced = h_enhanced.squeeze(1)
            
            # 动态特征融合
            fusion_weight = self.fusion_gate(
                torch.cat([h_enhanced, current_cnn], dim=-1)
            )
            
            # 市场状态调制
            market_impact = self.market_state(
                torch.cat([h_enhanced, current_cnn], dim=-1)
            )
            
            # 最终特征组合
            combined = fusion_weight * h_enhanced + (1 - fusion_weight) * current_cnn
            combined = combined * market_impact
            
            # 预测
            pred = self.predictor(combined)
            
            outputs.append(pred)
            combined_features.append(combined)
        
        predictions = torch.stack(outputs, dim=1)
        features = torch.stack(combined_features, dim=1)
        
        return predictions, tmdo_features, features

# 训练函数
def train_fusion_model(model, train_loader, val_loader, 
                      n_epochs=50, device='cuda', learning_rate=0.001):
    """
    训练融合模型的函数
    
    Args:
        model: ATLASCNNFusion模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        n_epochs: 训练轮数
        device: 训练设备
        learning_rate: 学习率
    """
    criterion = EnhancedCombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 使用Cosine退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    return train_enhanced_model(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        device=device
    )

# 主函数示例
def main():
    # 参数设置
    input_dim = 21  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    event_dim = 32  # 事件嵌入维度
    num_event_types = 10  # 事件类型数量
    
    # 数据准备和训练过程
    symbols = ['AAPL', 'MSFT']
    data = combine_stock_data(symbols, '1980-01-01', '2024-01-01')
    events = generate_event_data(data)  # 需要实现这个函数
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_events, val_events = train_test_split(events, test_size=0.2, shuffle=False)
    
    # 创建数据集
    train_dataset = FusionStockDataset(train_data, train_events)
    val_dataset = FusionStockDataset(val_data, val_events)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32768,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32768,
        shuffle=False,
        num_workers=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)

    # 创建模型实例
    model = ATLASCNNFusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_dim=event_dim,
        num_event_types=num_event_types,
        feature_groups=prepare_feature_groups()
    ).to(device=device)

    train_fusion_model(model, train_loader, val_loader)
    
if __name__ == "__main__":
    main()