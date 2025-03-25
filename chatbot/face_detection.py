import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)  # Confidence threshold

def detect_faces(image_path):
    try:
        # ✅ Ensure the image path is valid
        if not image_path or not isinstance(image_path, str):
            return "Error: Invalid image path received."

        # ✅ Check if file exists
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"

        # ✅ Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: Unable to load image. Check if it's a valid format."

        # ✅ Convert image to RGB for Mediapipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ Perform face detection
        results = face_detection.process(rgb_img)

        # ✅ Count detected faces
        num_faces = len(results.detections) if results.detections else 0
        return num_faces
    except Exception as e:
        return f"Error detecting faces: {str(e)}"
