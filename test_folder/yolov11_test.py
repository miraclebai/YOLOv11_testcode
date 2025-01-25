from ultralytics import YOLO
import cv2
# 模型
model = YOLO("/Users/baijingyuan/jupyterPj/reproducibility/code/Yolov11/yolo11n.pt")

# 进行预测测试
results = model.predict(
    source="bus.jpg",
    show=True,
    conf=0.2
)
# 显示测试结果
for result in results:
    img = result.plot()
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



