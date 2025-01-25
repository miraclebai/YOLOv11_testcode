import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到Python路径
import xml.etree.ElementTree as ET
import random
import shutil
import cv2
from dataset1.data_augmentation import create_augmented_dataset  # 导入数据增强函数

# 定义类别映射
CLASS_MAPPING = {
    "aircraft": 0,
    "human": 1,
    "ship": 2
}

def convert_annotation(xml_file, output_file, target_width, target_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取原始图片尺寸
    orig_width = float(root.find('size/width').text)
    orig_height = float(root.find('size/height').text)

    with open(output_file, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in CLASS_MAPPING:
                continue  # 跳过未定义的类别
            
            # 获取类别索引
            cls_idx = CLASS_MAPPING[cls_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 调整坐标到目标尺寸
            xmin = xmin * (target_width / orig_width)
            xmax = xmax * (target_width / orig_width)
            ymin = ymin * (target_height / orig_height)
            ymax = ymax * (target_height / orig_height)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / target_width
            y_center = (ymin + ymax) / 2 / target_height
            w = (xmax - xmin) / target_width
            h = (ymax - ymin) / target_height

            # 写入类别索引而不是类别名称
            f.write(f"{cls_idx} {x_center} {y_center} {w} {h}\n")

def resize_image(img_path, target_width, target_height):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (target_width, target_height))
    return resized_img

def create_dataset(jpeg_dir, ann_dir, output_dir, train_ratio=0.8, target_width=320, target_height=320, augmented_dir=None):
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)

    # 获取所有图像文件
    all_images = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
    
    # 如果存在增强数据，合并文件列表
    if augmented_dir and os.path.exists(augmented_dir):
        augmented_images = [f for f in os.listdir(os.path.join(augmented_dir, 'images')) if f.endswith('.jpg')]
        all_images.extend(augmented_images)
        print(f"Found {len(augmented_images)} augmented images")
    
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)

    # 处理每个图像
    for i, img_name in enumerate(all_images):
        base_name = os.path.splitext(img_name)[0]
        
        # 确定训练集或验证集
        if i < split_idx:
            img_dst_dir = 'train'
        else:
            img_dst_dir = 'val'

        # 处理图像
        if augmented_dir and img_name in augmented_images:
            # 处理增强数据
            img_src = os.path.join(augmented_dir, 'images', img_name)
            xml_src = os.path.join(augmented_dir, 'labels', f'{base_name}.xml')
        else:
            # 处理原始数据
            img_src = os.path.join(jpeg_dir, img_name)
            xml_src = os.path.join(ann_dir, f'{base_name}.xml')

        # 调整图像尺寸
        img_dst = os.path.join(output_dir, img_dst_dir, 'images', img_name)
        img = cv2.imread(img_src)
        if img is None:
            print(f"Warning: Failed to read image {img_src}, skipping")
            continue
        resized_img = cv2.resize(img, (target_width, target_height))
        cv2.imwrite(img_dst, resized_img)
        
        # 处理标注
        txt_dst = os.path.join(output_dir, img_dst_dir, 'labels', f'{base_name}.txt')
        if not os.path.exists(xml_src):
            print(f"Warning: Annotation file {xml_src} not found, skipping {img_name}")
            continue
            
        convert_annotation(xml_src, txt_dst, target_width, target_height)

    print(f"数据集划分完成，训练集和验证集已分别保存至 {output_dir}/train 和 {output_dir}/val")

# 先进行数据增强
augmented_dir = create_augmented_dataset(
    original_dir="/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/SCTD/JPEGImages",
    annotation_dir="/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/SCTD/Annotations",
    output_dir="/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/yolo_dataset"
)

# 然后进行数据集划分和格式转换
create_dataset(
    jpeg_dir="/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/SCTD/JPEGImages",
    ann_dir="/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/SCTD/Annotations",
    output_dir="/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/yolo_dataset",
    augmented_dir=augmented_dir
)