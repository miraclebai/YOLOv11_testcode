from ultralytics import YOLO
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建正确的yaml文件路径
yaml_path = os.path.join(current_dir, "dataset2.yaml")

model = YOLO("yolo11n.pt")
results = model.train(
    data=yaml_path,  # 使用绝对路径
    epochs=5,
    imgsz=320,
    batch=16,
    device="mps",
    rect=True
)