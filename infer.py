import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from data import download_and_prepare_data
from model import StockPredictor, prepare_feature_groups
from torch.utils.data import Dataset, DataLoader

class PredictionDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length
        
        # 生成序列起始索引
        self.indices = range(len(data) - seq_length + 1)
        
        # 模拟事件数据 (与训练保持一致)
        self.events = np.zeros((len(data), 1))
        self.time_distances = np.ones((len(data), 1))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length
        
        sequence = torch.FloatTensor(
            self.data.iloc[start_idx:end_idx].values
        )
        
        events = torch.LongTensor(
            self.events[start_idx:end_idx]
        )
        
        time_distances = torch.FloatTensor(
            self.time_distances[start_idx:end_idx]
        )
        
        return sequence, events, time_distances

def load_model(checkpoint_path, device='cuda'):
    # 创建模型实例
    model = StockPredictor(
        input_dim=21,         # 与训练时保持一致
        hidden_dim=128,      
        event_dim=32,        
        num_event_types=10,  
        feature_groups=prepare_feature_groups()
    )
    
    # 加载模型参数
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def predict_next_day(model, data, device='cuda'):
    dataset = PredictionDataset(data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for sequence, events, time_distances in dataloader:
            sequence = sequence.to(device)
            events = events.to(device)
            time_distances = time_distances.to(device)
            
            prediction = model(sequence, events, time_distances)
            # 只取最后一个预测值
            predictions.append(prediction[:, -1].cpu().numpy())
    
    return np.array(predictions[:-1])  # 去掉最后一个预测，因为它对应的是未来数据

def plot_predictions(actual_data, predictions, title="Stock Price Predictions"):
    plt.figure(figsize=(15, 7))
    
    # 绘制实际价格
    plt.plot(actual_data.index, actual_data['Close'], 
             label='Actual Price', color='blue')
    
    # 确保预测和日期长度匹配
    pred_dates = actual_data.index[9:-1]
    assert len(pred_dates) == len(predictions), \
           f"Mismatch in lengths: dates={len(pred_dates)}, predictions={len(predictions)}"
    
    plt.plot(pred_dates, predictions, 
             label='Predicted Price', color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model('best_model.pth', device)
    
    # 获取最新数据
    symbol = 'AAPL'  # 可以改为其他股票
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 获取一年的数据
    
    data = download_and_prepare_data(symbol, start_date.strftime('%Y-%m-%d'), 
                                   end_date.strftime('%Y-%m-%d'))
    
    # 进行预测
    predictions = predict_next_day(model, data, device)
    
    # 反归一化预测结果
    # 注意：这里需要根据你的标准化方法来调整
    # 如果使用了StandardScaler，需要用相同的scaler进行反变换
    
    # 绘制结果
    plot_predictions(data, predictions.flatten(), 
                    title=f"{symbol} Stock Price Predictions")
    
    # 打印最新的预测结果
    latest_prediction = predictions[-1][0]
    print(f"\nPrediction for next day: ", latest_prediction)
    
    # 输出一些基本的统计信息
    recent_data = data.tail(5)
    print("\nRecent price movements:")
    print(recent_data[['Close', 'Volume', 'RSI', 'MACD']].to_string())
    
    # 计算一些预测指标
    actual_prices = data['Close'].values[10:]
    predicted_prices = predictions.flatten()
    
    mse = np.mean((actual_prices - predicted_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    
    print("\nPrediction metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 方向准确率
    actual_direction = np.diff(actual_prices) > 0
    predicted_direction = np.diff(predicted_prices) > 0
    direction_accuracy = np.mean(actual_direction == predicted_direction)
    print(f"Direction Accuracy: {direction_accuracy:.2%}")

if __name__ == "__main__":
    main()