from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os
#from agronav_mobile import *

# Config and checkpoint
cwd = os.getcwd()

# Path to config file (.py)
config_file = cwd + '/configs/our_dataset/our_dataset.py'
# Path to checkpoint file (.pth)
checkpoint_file = cwd + '/work_dirs/our_dataset/iter_60000.pth'

# Init model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Image path.
image_filename = '221205_0284.png'
image_path = os.path.join(cwd, image_filename)  # Change this to your image directory

# Where will be saved.
output_dir = os.path.join(cwd, '')
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
img = mmcv.imread(image_path, channel_order='rgb')

# Run inference
result = inference_model(model, img)

# Output path for visualization
out_path = os.path.join(output_dir, f'result_{image_filename}')
show_result_pyplot(model, img, result, show=False, save_dir=output_dir)

print("Processing complete. Results saved in:", output_dir)
