import os
import cv2
import random
import shutil
import numpy as np
from xml.etree import ElementTree as ET

# 定义中等增强参数
AUGMENTATION_FACTOR = 3  # 每个图像生成3份（包括原始图像）
FLIP_CODES = [1]  # 只保留水平翻转

# 在文件开头添加类别映射
CLASS_MAPPING = {
    "aircraft": 0,
    "human": 1,
    "ship": 2
}

def rotate_image_and_annotation(img, angle):
    """旋转图像并调整标注"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    
    def adjust_bbox(bbox):
        x_center, y_center, w, h = bbox
        # 将YOLO格式转换为像素坐标
        xmin = int((x_center - w/2) * w)
        xmax = int((x_center + w/2) * w)
        ymin = int((y_center - h/2) * h)
        ymax = int((y_center + h/2) * h)
        
        # 旋转bbox的四个角点
        corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])
        transformed_corners = M.dot(corners_ones.T).T
        
        # 计算新的bbox并限制在图像范围内
        new_xmin = max(0, transformed_corners[:, 0].min()) / w
        new_xmax = min(w, transformed_corners[:, 0].max()) / w
        new_ymin = max(0, transformed_corners[:, 1].min()) / h
        new_ymax = min(h, transformed_corners[:, 1].max()) / h
        
        # 转换回YOLO格式
        new_x_center = (new_xmin + new_xmax) / 2
        new_y_center = (new_ymin + new_ymax) / 2
        new_w = new_xmax - new_xmin
        new_h = new_ymax - new_ymin
        
        # 增加边界框有效性检查
        if new_w <= 0 or new_h <= 0:
            return None  # 返回None表示无效边界框
        
        # 确保坐标在0-1之间
        new_x_center = max(0.01, min(0.99, new_x_center))
        new_y_center = max(0.01, min(0.99, new_y_center))
        new_w = max(0.01, min(1, new_w))
        new_h = max(0.01, min(1, new_h))
        
        return new_x_center, new_y_center, new_w, new_h
    
    return rotated_img, adjust_bbox

def flip_image_and_annotation(img, flip_code):
    """翻转图像并调整标注"""
    h, w = img.shape[:2]
    flipped_img = cv2.flip(img, flip_code)
    
    def adjust_bbox(bbox):
        x_center, y_center, w, h = bbox
        if flip_code == 1:  # 水平翻转
            new_x_center = 1.0 - x_center
            return new_x_center, y_center, w, h
        elif flip_code == 0:  # 垂直翻转
            new_y_center = 1.0 - y_center
            return x_center, new_y_center, w, h
        return x_center, y_center, w, h
    
    return flipped_img, adjust_bbox

def clip_bbox(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox
    # 确保坐标在图像范围内
    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(0, min(xmax, img_width - 1))
    ymax = max(0, min(ymax, img_height - 1))
    # 确保xmin < xmax 且 ymin < ymax
    if xmin >= xmax or ymin >= ymax:
        return None  # 返回None表示无效bbox
    return (xmin, ymin, xmax, ymax)

def validate_bbox_size(bbox, min_size=10, max_aspect_ratio=10):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    
    # 检查最小尺寸
    if width < min_size or height < min_size:
        return False
        
    # 检查宽高比
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > max_aspect_ratio:
        return False
        
    return True

def create_augmented_dataset(original_dir, annotation_dir, output_dir):
    # 创建临时目录
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(os.path.join(temp_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'labels'), exist_ok=True)

    # 获取原始图像列表
    image_files = [f for f in os.listdir(original_dir) if f.endswith('.jpg')]

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(original_dir, img_file)
        xml_path = os.path.join(annotation_dir, f'{base_name}.xml')
        
        # 读取原始图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image {img_path}, skipping")
            continue

        # 对每个图像生成AUGMENTATION_FACTOR份数据
        for i in range(AUGMENTATION_FACTOR):
            new_base_name = f"{base_name}_aug{i}"
            new_img_name = f"{new_base_name}.jpg"
            
            # 处理图像
            if i == 0:
                # 原始图像
                augmented_img = img
            elif i == 1:
                # 高斯模糊
                kernel_size = random.choice([3, 5, 7])
                augmented_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            elif i == 2:
                # 水平翻转
                augmented_img = cv2.flip(img, 1)
                
                # 处理XML标注
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # 调整bbox坐标
                width = float(root.find('size/width').text)
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    xmax = float(bbox.find('xmax').text)
                    # 水平翻转只需要调整x坐标
                    new_xmin = width - xmax
                    new_xmax = width - xmin
                    bbox.find('xmin').text = str(new_xmin)
                    bbox.find('xmax').text = str(new_xmax)
                
                # 保存修改后的XML
                tree.write(os.path.join(temp_dir, 'labels', f"{new_base_name}.xml"))
                continue
            
            # 保存图像
            cv2.imwrite(os.path.join(temp_dir, 'images', new_img_name), augmented_img)
            # 复制XML文件
            shutil.copy(xml_path, os.path.join(temp_dir, 'labels', f"{new_base_name}.xml"))

    print(f"数据增强完成，增强后的数据保存在临时目录: {temp_dir}")
    return temp_dir  # 返回临时目录路径
