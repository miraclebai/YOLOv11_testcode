import os
import xml.etree.ElementTree as ET
import random
import shutil
import cv2

# 定义类别映射
CLASS_MAPPING = {
    "ship": 0
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

def create_dataset(jpeg_dir, ann_dir, output_dir, target_width=320, target_height=320):
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

    # 处理训练集
    process_split(
        img_dir=os.path.join(jpeg_dir, 'JPEGImages_train'),
        ann_dir=os.path.join(ann_dir, 'Annotations_train'),
        output_img_dir=os.path.join(output_dir, 'images', 'train'),
        output_label_dir=os.path.join(output_dir, 'labels', 'train'),
        target_width=target_width,
        target_height=target_height
    )

    # 处理测试集
    process_split(
        img_dir=os.path.join(jpeg_dir, 'JPEGImages_test'),
        ann_dir=os.path.join(ann_dir, 'Annotations_test'),
        output_img_dir=os.path.join(output_dir, 'images', 'test'),
        output_label_dir=os.path.join(output_dir, 'labels', 'test'),
        target_width=target_width,
        target_height=target_height
    )

    print(f"数据集转换完成，结果已保存至 {output_dir}")

def process_split(img_dir, ann_dir, output_img_dir, output_label_dir, target_width, target_height):
    """处理单个数据集分割"""
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found: {img_dir}")
        return
        
    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.jpg'):
            continue
            
        base_name = os.path.splitext(img_name)[0]
        
        # 处理图像
        img_src = os.path.join(img_dir, img_name)
        img_dst = os.path.join(output_img_dir, img_name)
        img = cv2.imread(img_src)
        resized_img = cv2.resize(img, (target_width, target_height))
        cv2.imwrite(img_dst, resized_img)
        
        # 转换标注
        xml_src = os.path.join(ann_dir, f'{base_name}.xml')
        txt_dst = os.path.join(output_label_dir, f'{base_name}.txt')
        
        if not os.path.exists(xml_src):
            print(f"Warning: Annotation file {xml_src} not found, skipping {img_name}")
            continue
            
        convert_annotation(xml_src, txt_dst, target_width, target_height)

# 修改后的使用方式
create_dataset(
    jpeg_dir="/Users/baijingyuan/Desktop/audio_data/Official-SSDD-OPEN/BBox_SSDD/voc_style",
    ann_dir="/Users/baijingyuan/Desktop/audio_data/Official-SSDD-OPEN/BBox_SSDD/voc_style",
    output_dir="/Users/baijingyuan/Desktop/audio_data/Official-SSDD-OPEN/BBox_SSDD/voc_style/yolo_dataset",
    target_width=320,
    target_height=320
)