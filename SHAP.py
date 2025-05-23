import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shap
from ultralytics import YOLO
from PIL import Image


def explain_yolo11_detections(image_path, model, top_k=3, shap_samples=50):
    """
    Args:
        image_path (str): Шлях до вхідного зображення.
        model: Завантажена модель YOLO11 (ultralytics.YOLO).
        top_k (int): Кількість об'єктів для пояснення (за найвищою впевненістю).
        shap_samples (int): Кількість зразків для SHAP.

    Returns:
        Візуалізація пояснень SHAP для виявлених об'єктів.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    results = model(img_rgb, verbose=False)

    boxes = results[0].boxes.xyxy.cpu().numpy()  # Координати bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Впевненість
    classes = results[0].boxes.cls.cpu().numpy()  # Класи
    class_names = results[0].names  # Назви класів

    top_indices = np.argsort(scores)[::-1][:top_k]

    for idx in top_indices:
        class_id = int(classes[idx])
        class_name = class_names[class_id]
        score = scores[idx]
        box = boxes[idx]

        def model_predict(images, target_box=box, target_class_id=class_id):
            """
            Функція для передбачення YOLO11, сумісна з SHAP.
            Повертає впевненість для конкретного об'єкта в межах заданого bounding box.
            """
            predictions = []
            for img in images:
                img = img.transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
                img = img.astype(np.uint8)
                result = model(img, verbose=False)

                detected_boxes = result[0].boxes.xyxy.cpu().numpy()
                conf = result[0].boxes.conf.cpu().numpy()
                cls = result[0].boxes.cls.cpu().numpy()

                best_iou = 0
                best_conf = 0
                for i, (det_box, det_conf, det_cls) in enumerate(zip(detected_boxes, conf, cls)):
                    if int(det_cls) != target_class_id:
                        continue
                    x1 = max(target_box[0], det_box[0])
                    y1 = max(target_box[1], det_box[1])
                    x2 = min(target_box[2], det_box[2])
                    y2 = min(target_box[3], det_box[3])
                    intersection = max(0, x2 - x1) * max(0, y2 - y1)
                    area1 = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                    area2 = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_conf = det_conf
                predictions.append([best_conf]) 
            return np.array(predictions)

        masker = shap.maskers.Image("inpaint_telea", img_rgb.shape)

        explainer = shap.Explainer(model_predict, masker)

        img_shap = img_rgb.transpose((2, 0, 1))  # (C, H, W)
        img_shap = img_shap[np.newaxis, ...]  # batch dimension

        shap_values = explainer(img_shap, max_evals=shap_samples)

        print("SHAP values shape:", shap_values.shape)

        shap_vals = shap_values.values[0, ..., 0]  # (channels, height, width)
        shap_vals = np.transpose(shap_vals, (1, 2, 0))  # (height, width, channels)
        print("Adjusted SHAP values shape:", shap_vals.shape)

        shap_fig = shap.image_plot(shap_vals, pixel_values=img_rgb, labels=[class_name], show=False)
        plt.savefig(f'/kaggle/working/test{idx}_{shap_samples}.png', dpi=150)       

model = YOLO("/kaggle/input/detract-yolo11s/kaggle/working/runs/detect/train/weights/best.pt")

image_path = "/kaggle/input/test-images/test_images/MVI_40901_img00752.jpg" 
explain_yolo11_detections(image_path, model, top_k=3, shap_samples=50)