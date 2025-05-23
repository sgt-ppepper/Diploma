import numpy as np
import cv2
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from IPython.display import display, Image

def explain_yolo11_top3(image_path, model_path="/kaggle/input/detract-yolo11s/kaggle/working/runs/detect/train/weights/best.pt", num_samples=100, top_n=3):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)[0]
    detections = results.boxes.data.cpu().numpy()

    detections = detections[detections[:, 4].argsort()[::-1]]  
    detections = detections[:top_n]

    explainer = lime_image.LimeImageExplainer()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det.astype(int)
        label = model.names[int(cls)]

        crop_img = image_rgb[y1:y2, x1:x2]

        def yolo_predict(images):
            preds = []
            for img in images:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                result = model(img_bgr, verbose=False)[0]
                boxes = result.boxes.data.cpu().numpy()
                cls_scores = np.zeros(len(model.names))
                for box in boxes:
                    if int(box[5]) == int(cls) and box[4] > 0.5:
                        cls_scores[int(cls)] = box[4]
                preds.append(cls_scores)
            return np.array(preds)

        explanation = explainer.explain_instance(
            image_rgb,
            yolo_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(crop_img)
        plt.title(f'Detected Object: {label}', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 12), dpi=150)
        plt.imshow(mark_boundaries(image_rgb, mask))
        plt.title(f'LIME Explanation: {label}', fontsize=18)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'lime_expl_{idx}.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    image_path = "/kaggle/input/test-images/test_images/MVI_40901_img00752.jpg" 
    explain_yolo11_top3(image_path, num_samples=100)