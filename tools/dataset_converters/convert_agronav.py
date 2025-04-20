import os
import shutil
import random

def split_dataset(image_dir, label_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    # Get sorted filenames (excluding extensions)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG')])
    base_names = [os.path.splitext(f)[0] for f in image_files]

    # Shuffle and split
    random.shuffle(base_names)
    val_size = int(len(base_names) * val_ratio)
    val_names = set(base_names[:val_size])
    train_names = set(base_names[val_size:])

    def copy_files(names, split):
        for name in names:
            img_src = os.path.join(image_dir, f'{name}.JPG')
            lbl_src = os.path.join(label_dir, f'{name}.png')

            img_dst = os.path.join(output_dir, 'images', split, f'{name}.JPG')
            lbl_dst = os.path.join(output_dir, 'annotations', split, f'{name}.png')

            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

    copy_files(train_names, 'train')
    copy_files(val_names, 'val')

    print(f"Done. Train: {len(train_names)} images, Val: {len(val_names)} images.")

if __name__ == "__main__":
    image_dir = '/content/TSCF-MMSeg-agronav/data/images'
    label_dir = '/content/TSCF-MMSeg-agronav/data/labels'
    output_dir = '/content/TSCF-MMSeg-agronav/data/agronav'

    split_dataset(image_dir, label_dir, output_dir)
