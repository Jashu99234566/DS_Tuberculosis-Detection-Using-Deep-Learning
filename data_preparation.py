import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(base_dir='TB_Chest_Xray_Data', output_dir='processed_data', test_size=0.2, val_size=0.1):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    classes = ['TB', 'Normal']
    for cls in classes:
        class_path = os.path.join(base_dir, cls)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=42)

        for split, imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img_path in imgs:
                shutil.copy(img_path, split_dir)

    print("✅ Dataset prepared in:", output_dir)

prepare_dataset()
