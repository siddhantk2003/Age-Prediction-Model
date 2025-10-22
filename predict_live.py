"""
Live webcam age prediction using OpenCV DNN face detector + EfficientNet model.
Displays bounding boxes with predicted age overlayed.
"""
import os
import time
import cv2
import torch
import numpy as np
from efficientnet_model import get_efficientnet_b0_regression
from utils import ensure_face_detector, default_transforms
from PIL import Image

def preprocess_face(face_bgr):
    # face_bgr: cropped face in BGR (from OpenCV)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    tf = default_transforms(train=False)
    return tf(pil).unsqueeze(0)  # shape (1,3,224,224)

def main(model_path="models/age_predictor.pth", detector_dir="models/face_detector", device='cpu'):
    # Ensure face detector files exist
    proto_path, model_file = ensure_face_detector(detector_dir)
    net = cv2.dnn.readNetFromCaffe(proto_path, model_file)

    # Load model
    device = torch.device("cpu")
    model = get_efficientnet_b0_regression(pretrained=False, device=device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train.py first.")
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Make sure camera permission is granted.")

    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                face = frame[y1:y2, x1:x2]
                # skip tiny faces
                if face.size == 0 or (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                # preprocess and predict
                inp = preprocess_face(face)
                with torch.no_grad():
                    out = model(inp)
                    age_hat = out.item()
                age_text = f"Age: {int(round(age_hat))}"

                # overlay: colored bounding box and label
                # color changes slightly with predicted age for style (wrap to 0-255)
                color = (int((age_hat*3)%255), int((age_hat*5)%255), int((age_hat*7)%255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # label background
                (text_w, text_h), baseline = cv2.getTextSize(age_text, font, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - text_h - baseline - 6), (x1 + text_w + 6, y1), color, -1)
                cv2.putText(frame, age_text, (x1 + 3, y1 - 5), font, 0.6, (255,255,255), 1, cv2.LINE_AA)

            frame_count += 1
            # show FPS every ~30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), font, 0.7, (0,255,0), 1, cv2.LINE_AA)

            cv2.imshow("Age Prediction (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # If you want to customize path, edit args here or run as module and modify parameters
    main()
