import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch

# 导入ATLAS系统
from atlas_system import (
    StockDataStructure, 
    ImprovedAttentionMechanism,
    WeightedLaplacian
)

class MarketPredictor:
    def __init__(self):
        # 初始化权重配置
        self.weights = {
            'momentum': 0.4,
            'volume': 0.15,
            'technical': 0.35,
            'volatility': 0.1
        }
        
        # 初始化ATLAS系统组件
        self.attention_mechanism = ImprovedAttentionMechanism(feature_dim=10)
        self.laplacian_operator = WeightedLaplacian(self.attention_mechanism)
        
        # 设置预测参数
        self.window_size = 20
        self.volatility_threshold = {
            'high': 1.5,
            'low': 0.5
        }
        
    def download_latest_data(self, symbol='AAPL', lookback_days=100):
        end_date = '2024-12-13'
        start_date = (pd.to_datetime(end_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # 下载数据
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        # 计算技术指标
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['Signal'] = self.calculate_macd(data['Close'])
        
        return data
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def prepare_features(self, data):
        # 准备输入特征
        features = pd.DataFrame({
            'Open': data['Open'],
            'High': data['High'],
            'Low': data['Low'],
            'Close': data['Close'],
            'Volume': data['Volume'],
            'SMA_20': data['SMA_20'],
            'SMA_50': data['SMA_50'],
            'RSI': data['RSI'],
            'MACD': data['MACD'],
            'Signal': data['Signal']
        })
        
        # 标准化
        for col in features.columns:
            if col != 'Volume':
                features[col] = (features[col] - features[col].mean()) / features[col].std()
            else:
                features[col] = (features[col] - features[col].min()) / (features[col].max() - features[col].min())
        
        return features
    
    def predict_next_day(self, data):
        # 准备特征
        features = self.prepare_features(data)
        stock_data = StockDataStructure(features)
        
        # 获取最后一个时间窗口
        last_window = stock_data.get_time_window(len(features)-1, window_size=20)
        
        # 计算拉普拉斯特征
        laplacian_output = self.laplacian_operator(last_window)
        
        # 分析市场状态
        market_state = self.analyze_market_state(laplacian_output, last_window)
        
        return market_state
    
    def analyze_market_state(self, laplacian_output, window_data):
        latest_features = laplacian_output[0, -1].detach().numpy()
        
        # 计算基础分数
        momentum_score = np.mean(latest_features[:4])
        volume_score = latest_features[4]
        technical_score = np.mean(latest_features[5:])
        
        # 计算波动性分数
        volatility = np.std(window_data[0, -5:, 3].numpy())
        volatility_score = -1 if volatility > self.volatility_threshold['high'] else \
                          1 if volatility < self.volatility_threshold['low'] else 0
        
        # 加权计算最终分数
        final_score = (
            self.weights['momentum'] * momentum_score +
            self.weights['volume'] * volume_score +
            self.weights['technical'] * technical_score +
            self.weights['volatility'] * volatility_score
        )
        
        # 计算置信区间
        scores = [momentum_score, volume_score, technical_score, volatility_score]
        confidence_range = np.std(scores) * 2
        
        return {
            'direction': 'Up' if final_score > 0 else 'Down',
            'confidence': abs(final_score),
            'confidence_range': confidence_range,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'technical_score': technical_score,
            'volatility_score': volatility_score,
            'final_score': final_score,
            'risk_level': 'High' if volatility_score < 0 else 'Low' if volatility_score > 0 else 'Medium'
        }


def main(ticker='AAPL'):
    predictor = MarketPredictor()
    
    # 下载数据
    print("Downloading latest market data...")
    data = predictor.download_latest_data(symbol=ticker)
    
    # 生成预测
    print("\nGenerating prediction for next trading day...")
    
    predictor = MarketPredictor()
    prediction = predictor.predict_next_day(data)
    
    print("\nPrediction for next trading day:")
    print(f"Direction: {prediction['direction']}")
    print(f"Confidence: {prediction['confidence']:.2f}")
    print(f"Confidence Range: ±{prediction['confidence_range']:.2f}")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"\nDetailed Scores:")
    print(f"Momentum: {prediction['momentum_score']:.2f}")
    print(f"Volume: {prediction['volume_score']:.2f}")
    print(f"Technical: {prediction['technical_score']:.2f}")
    print(f"Volatility: {prediction['volatility_score']:.2f}")
    print(f"Final Score: {prediction['final_score']:.2f}")

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'IBM']
    for ticker in tickers:
        print("="*40)
        print(f"\n\nPredictions for {ticker} stock:")
        main(ticker)