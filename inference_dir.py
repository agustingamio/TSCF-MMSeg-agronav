from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os

# Config and checkpoint
cwd = os.getcwd()

# Path to config file (.py)
config_file = cwd + '/configs/our_dataset/segformer_our_dataset_from_ADE.py'
# Path to checkpoint file (.pth)
checkpoint_file = cwd + '/work_dirs/segformer_our_dataset_from_ADE_20_06/iter_40000.pth'

# Init model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Directory of images to inference.
image_dir = os.path.join(cwd, 'data/our_dataset/images_rellis/test')

# Directory where results will be saved.
output_dir = os.path.join(cwd, 'work_dirs/segformer_our_dataset_from_ADE_20_06/visualizar/resultados_40k')
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir, filename)
        img = mmcv.imread(img_path, channel_order='rgb')

        # Run inference
        result = inference_model(model, img)

        # Output path for visualization
        out_path = os.path.join(output_dir, f'{filename}')
        show_result_pyplot(model, img, result, out_file=out_path, show=False)

print("Processing complete. Results saved in:", output_dir)
