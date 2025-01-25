from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# 加载训练结果
results = pd.read_csv("runs/detect/train18/results.csv")

# 只取前17个epoch的结果
results = results.iloc[:-1]

# 绘制训练曲线
results.plot(
    y=['train/box_loss', 'train/cls_loss', 'train/dfl_loss'],  # 训练损失
    title='Training Loss'
)
plt.show()

# 检查是否存在 mAP50 列
if 'metrics/mAP50' in results.columns:
    results.plot(
        y=['metrics/mAP50'],
        title='mAP50 Curve',
        xlabel='Epoch',
        ylabel='mAP50'
    )
    plt.show()
elif 'metrics/mAP50(B)' in results.columns:  # 有些版本列名可能不同
    results.plot(
        y=['metrics/mAP50(B)'],
        title='mAP50 Curve',
        xlabel='Epoch',
        ylabel='mAP50'
    )
    plt.show()
else:
    print("未找到 mAP50 相关列，可用列名为：", results.columns)


