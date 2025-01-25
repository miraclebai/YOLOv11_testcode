from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/detect/train11/weights/best.pt')  # 使用训练保存的最佳模型

source = "/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/original_data/JPEGImages/000018.jpg"
# 进行预测
results = model.predict(
    source=source,
    show=True,
    conf=0.35,
    iou=0.45,
)



img = cv2.imread(source)
print(img.shape)  # 输出图片的 (height, width, channels)
cv2.imshow("input", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for result in results:
    print(result.boxes)  # 检查检测框信息
img = result.plot()
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()