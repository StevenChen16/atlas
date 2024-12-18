import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data import download_and_prepare_data
from model import *
from train import *

def prepare_test_data(symbols=['AAPL', 'MSFT', 'GOOG', "META", "NVDA", "AMD", "AMZN", "ORCL", 'INTC', "MU", "TJX",
                               "HON", "DLR", "QUAL", "RACE", "PDD", "AVGO", "SNOW"], test_size=0.2):
    """
    准备测试数据，包括下载、处理和划分
    """
    all_data = []
    all_events = []
    
    for symbol in tqdm(symbols, desc="Downloading data"):
        try:
            # 下载和处理数据
            data = download_and_prepare_data(symbol, '2019-01-01', '2024-01-01')
            
            # 生成事件数据
            events = generate_event_data(data)
            
            all_data.append(data)
            all_events.append(events)
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # 合并数据
    combined_data = pd.concat(all_data, axis=0)
    combined_events = np.concatenate(all_events, axis=0)
    
    # 按时间顺序排序
    combined_data = combined_data.sort_index()
    
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(
        range(len(combined_data)),
        test_size=test_size,
        shuffle=False  # 保持时间顺序
    )
    
    train_data = combined_data.iloc[train_idx]
    test_data = combined_data.iloc[test_idx]
    
    train_events = combined_events[train_idx]
    test_events = combined_events[test_idx]
    
    return train_data, test_data, train_events, test_events

def calculate_direction_metrics(predictions, targets, prev_prices, threshold=0.0001):
    """
    计算方向预测的详细指标
    """
    # 计算价格变动百分比
    pred_change = (predictions - prev_prices) / prev_prices
    true_change = (targets - prev_prices) / prev_prices
    
    # 应用阈值，获取方向
    pred_direction = np.where(np.abs(pred_change) < threshold, 0,
                            np.sign(pred_change))
    true_direction = np.where(np.abs(true_change) < threshold, 0,
                            np.sign(true_change))
    
    # 计算基本指标
    total_acc = (pred_direction == true_direction).mean()
    
    # 分别计算上涨和下跌的准确率
    up_mask = true_direction > 0
    down_mask = true_direction < 0
    
    up_acc = (pred_direction[up_mask] == true_direction[up_mask]).mean() \
             if up_mask.any() else 0
    down_acc = (pred_direction[down_mask] == true_direction[down_mask]).mean() \
               if down_mask.any() else 0
    
    # 计算混淆矩阵
    # conf_matrix = confusion_matrix(true_direction, pred_direction, 
    #                              labels=[-1, 0, 1])
    
    # 计算更详细的指标
    results = {
        'total_accuracy': total_acc,
        'up_accuracy': up_acc,
        'down_accuracy': down_acc,
        # 'confusion_matrix': conf_matrix,
    }
    
    # 计算每个方向的精确率和召回率
    for direction, name in zip([-1, 0, 1], ['down', 'neutral', 'up']):
        mask = true_direction == direction
        if mask.any():
            tp = ((pred_direction == direction) & (true_direction == direction)).sum()
            fp = ((pred_direction == direction) & (true_direction != direction)).sum()
            fn = ((pred_direction != direction) & (true_direction == direction)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f'{name}_precision'] = precision
            results[f'{name}_recall'] = recall
            results[f'{name}_f1'] = f1
    
    return results

def plot_results(metrics, save_path=None):
    """
    可视化测试结果
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制准确率柱状图
    accuracies = [
        metrics['total_accuracy'],
        metrics['up_accuracy'],
        metrics['down_accuracy']
    ]
    ax1.bar(['Total', 'Up', 'Down'], accuracies)
    ax1.set_title('Direction Prediction Accuracy')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 绘制混淆矩阵热力图
    # sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
    #             xticklabels=['Down', 'Neutral', 'Up'],
    #             yticklabels=['Down', 'Neutral', 'Up'],
    #             ax=ax2)
    ax2.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_direction_prediction(model_path, symbols=['AAPL', 'MSFT', 'GOOGL'],
                            batch_size=128, threshold=0.0001):
    """
    完整的方向预测测试流程
    """
    # 准备数据
    print("Preparing test data...")
    _, test_data, _, test_events = prepare_test_data(symbols)
    
    # 创建测试数据集
    test_dataset = EnhancedStockDataset(test_data, test_events)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("Loading model...")
    model = EnhancedStockPredictor(
        input_dim=21,
        hidden_dim=128,
        event_dim=32,
        num_event_types=10,
        feature_groups=prepare_feature_groups()
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 收集预测结果
    all_predictions = []
    all_targets = []
    all_prev_prices = []
    
    print("Running predictions...")
    with torch.no_grad():
        for sequence, events, time_distances, target in tqdm(test_loader):
            # 移动数据到设备
            sequence = sequence.to(device)
            events = events.to(device)
            time_distances = time_distances.to(device)
            target = target.to(device)
            
            # 获取前一天的收盘价
            prev_price = sequence[:, -1, 3]
            
            # 模型预测
            predictions, _, _ = model(sequence, events, time_distances)
            final_predictions = predictions[:, -1, 0]
            
            # 收集结果
            all_predictions.extend(final_predictions.cpu().numpy())
            all_targets.extend(target.squeeze().cpu().numpy())
            all_prev_prices.extend(prev_price.cpu().numpy())
    
    # 计算指标
    print("\nCalculating metrics...")
    metrics = calculate_direction_metrics(
        np.array(all_predictions),
        np.array(all_targets),
        np.array(all_prev_prices),
        threshold
    )
    
    # 打印结果
    print("\nDirection Prediction Results:")
    print(f"Total Accuracy: {metrics['total_accuracy']:.4f}")
    print(f"Up Movement Accuracy: {metrics['up_accuracy']:.4f}")
    print(f"Down Movement Accuracy: {metrics['down_accuracy']:.4f}")
    
    print("\nDetailed Metrics:")
    for direction in ['up', 'down', 'neutral']:
        print(f"\n{direction.capitalize()} Movement:")
        print(f"Precision: {metrics[f'{direction}_precision']:.4f}")
        print(f"Recall: {metrics[f'{direction}_recall']:.4f}")
        print(f"F1 Score: {metrics[f'{direction}_f1']:.4f}")
    
    # 可视化结果
    plot_results(metrics, save_path='direction_prediction_results.png')
    
    return metrics

if __name__ == "__main__":
    # 测试模型
    metrics = test_direction_prediction(
        model_path='enhanced_stock_predictor.pth',
        symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', "^IXIC", "^GSPC", "^DJI", "HON", "TSLA"],
        threshold=0.001  # 0.1% 的价格变动阈值
    )