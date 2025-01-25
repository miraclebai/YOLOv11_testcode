from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建正确的yaml文件路径
yaml_path = os.path.join(current_dir, "sonar_dataset.yaml")

# 加载上次保存的模型
model = YOLO("runs/detect/train18/weights/last.pt") 

# 接着上次模型

# 进行模型训练
results = model.train(
    data=yaml_path,
    epochs=95,  # 总epoch数是95
    imgsz=320,
    batch=16,
    device="mps",
    lr0=0.01,  # 初始学习率
    lrf=0.1,
    resume=True,  # 关键：设置 resume=True 以继续训练
    val=True,  # 启用验证
)