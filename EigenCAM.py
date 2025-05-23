#!git clone https://github.com/rigvedrs/YOLO-V11-CAM

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
from ultralytics import YOLO

model = YOLO("/kaggle/input/detract-yolo11s/kaggle/working/runs/detect/train/weights/best.pt")

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/kaggle/input/test-images/test_images/MVI_40863_img00081.jpg')  # або .png
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img)

annotated_img = results[0].plot()
#print(results)
plt.imshow(annotated_img)
plt.axis('off')
plt.show()

img = cv2.imread('/kaggle/input/test-images/test_images/MVI_40863_img00081.jpg')
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255

target_layers =[model.model.model[-2]]
cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
plt.imshow(cam_image)
plt.show()