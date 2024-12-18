from data import load_data_from_csv
from CNN import EnhancedStockDataset, EnhancedFinancialCNN
import torch

def predict_stock(symbol, model_path='best_model.pth', sequence_length=250, device='cuda'):
    """
    对单一股票进行预测
    
    Args:
        symbol: 股票代码
        model_path: 模型权重路径
        sequence_length: 序列长度
        device: 计算设备
    """
    # 下载数据
    print(f"Downloading data for {symbol}...")
    data = load_data_from_csv("./data/GOOGL.csv")
    
    # 创建数据集
    dataset = EnhancedStockDataset(data, sequence_length=sequence_length)
    
    # 加载模型
    model = EnhancedFinancialCNN(input_dim=len(dataset.feature_order))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 获取最近的数据进行预测
    recent_data = dataset.data[:, -sequence_length:]
    recent_data = recent_data.unsqueeze(0).to(device)  # 添加batch维度
    
    # 预测
    with torch.no_grad():
        output = model(recent_data)
        prob = torch.softmax(output, dim=1)
        predicted = output.argmax(dim=1).item()
    
    # 获取预测概率
    probs = prob[0].cpu().numpy()
    
    # 获取最新日期 (从索引获取)
    latest_date = data.index[-1]
    previous_date = data.index[-2]
    
    # 打印预测结果
    print(f"\nPrediction for {symbol} as of {latest_date.strftime('%Y-%m-%d')}:")
    print(f"预测结果: {'下跌' if predicted == 0 else '横盘' if predicted == 1 else '上涨'}")
    print(f"\n预测概率:")
    print(f"下跌概率: {probs[0]:.2%}")
    print(f"横盘概率: {probs[1]:.2%}")
    print(f"上涨概率: {probs[2]:.2%}")
    
    # 打印最近的价格信息 (使用原始值,而不是归一化后的值)
    latest_close = data['Adj Close'].iloc[-1]  # 使用Adj Close而不是归一化的Close
    previous_close = data['Adj Close'].iloc[-2]
    price_change = (latest_close - previous_close) / previous_close * 100
    
    print(f"\n当前价格: ${latest_close:.2f}")
    print(f"前一日价格: ${previous_close:.2f}")
    print(f"价格变动: {price_change:+.2f}%")
    
    return predicted, probs

def main():
    # 预测GOOGL
    predicted, probs = predict_stock('GOOGL')
    
    # 可以添加其他股票
    # predict_stock('AAPL')
    # predict_stock('MSFT')

if __name__ == "__main__":
    main()