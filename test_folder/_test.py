from ultralytics import YOLO
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)

def check_dataset(data_path):
    missing_count = 0
    empty_count = 0
    total_count = 0
    
    # 确保所有标注文件都存在且不为空
    with open(data_path) as f:
        data = yaml.safe_load(f)
    
    # 确保路径是列表形式
    train_files = data['train'] if isinstance(data['train'], list) else [data['train']]
    val_files = data['val'] if isinstance(data['val'], list) else [data['val']]
    
    for file_list in [train_files, val_files]:
        for img_dir in file_list:
            # 检查路径是目录还是文件
            if os.path.isdir(img_dir):
                # 如果是目录，遍历目录中的所有jpg文件
                for img_name in os.listdir(img_dir):
                    if img_name.endswith('.jpg'):
                        total_count += 1
                        img_path = os.path.join(img_dir, img_name)
                        # 构建对应的标签路径
                        label_dir = img_dir.replace('images', 'labels')
                        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
                        
                        # 打印调试信息
                        print(f"Checking: {img_path} -> {label_path}")
                        if not os.path.exists(label_path):
                            missing_count += 1
                        elif os.path.getsize(label_path) == 0:
                            empty_count += 1
            else:
                # 如果是单个文件，按原逻辑处理
                img_name = os.path.basename(img_dir)
                label_dir = os.path.dirname(img_dir).replace('images', 'labels')
                label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
                
                print(f"Checking: {img_dir} -> {label_path}")
                if not os.path.exists(label_path):
                    print(f"Warning: Missing label file {label_path}")
                elif os.path.getsize(label_path) == 0:
                    print(f"Warning: Empty label file {label_path}")

    print(f"\nDataset check completed:")
    print(f"Total images checked: {total_count}")
    print(f"Missing labels: {missing_count}")
    print(f"Empty labels: {empty_count}")

# 添加标签验证函数
def validate_labels(label_dir):
    empty_files = []
    for label_name in os.listdir(label_dir):
        if label_name.endswith('.txt'):
            label_path = os.path.join(label_dir, label_name)
            if os.path.getsize(label_path) == 0:
                empty_files.append(label_path)
                print(f"Warning: Empty label file {label_path}")
            else:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        empty_files.append(label_path)
                        print(f"Warning: Empty label file {label_path}")
    return empty_files

# 检查训练集和验证集标签
train_label_dir = "/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/yolo_dataset/labels/train"
val_label_dir = "/Users/baijingyuan/jupyterPj/reproducibility/data/sonar_dataset/SCTD_dataset/yolo_dataset/labels/val"

empty_train = validate_labels(train_label_dir)
empty_val = validate_labels(val_label_dir)

if empty_train or empty_val:
    print("\nFound empty label files. Please fix them before training.")
    exit(1)

print("Starting dataset check...")

try:
    check_dataset("sonar_dataset.yaml")
except Exception as e:
    logging.error(f"Error occurred: {str(e)}")
    raise

print("Dataset check completed.")

if __name__ == '__main__':
    try:
        logging.info("Starting dataset check...")
        check_dataset("sonar_dataset.yaml")
        logging.info("Dataset check completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

