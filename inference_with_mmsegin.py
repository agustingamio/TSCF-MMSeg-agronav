import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os

from mmseg.apis.mmseg_inferencer import MMSegInferencer

# Config and checkpoint
cwd = os.getcwd()

image_filename = 'c3_ZoLPlT.jpg'
image_path = os.path.join(cwd, image_filename)  # Change this to your image directory
output_dir = os.path.join(cwd, '')
os.makedirs(output_dir, exist_ok=True)

img = mmcv.imread(image_path, channel_order='rgb')

out_path = os.path.join(output_dir, f'result_vit_{image_filename}')

inferencer = MMSegInferencer(
    model='san-vit-l14_coco-stuff164k-640x640',
)

inferencer(image_path, out_dir=output_dir)
print("Processing complete. Results saved in:", output_dir)
