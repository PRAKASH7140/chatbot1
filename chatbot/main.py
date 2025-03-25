from ultralytics import YOLO
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# ✅ Load YOLOv8 model (for object detection)
yolo_model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# ✅ Load EfficientNetB3 model (for classification)
efficientnet_model = EfficientNetB3(weights="imagenet")

def recognize_image(img_path):
    try:
        # ✅ YOLOv8 Object Detection
        img = cv2.imread(img_path)
        results = yolo_model(img)

        combined_objects = []  # 🔹 Stores all detected & classified objects

        for result in results:
            for box in result.boxes:
                label = yolo_model.names[int(box.cls[0])]
                confidence = round(float(box.conf[0]) * 100, 2)
                combined_objects.append({"label": label, "confidence": confidence})

        # ✅ EfficientNetB3 Classification
        img = image.load_img(img_path, target_size=(300, 300))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = efficientnet_model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        for obj in decoded_predictions:
            combined_objects.append({"label": obj[1], "confidence": round(float(obj[2]) * 100, 2)})

        # ✅ Return the combined list
        return combined_objects

    except Exception as e:
        return [{"label": "Error", "confidence": 0, "message": str(e)}]
