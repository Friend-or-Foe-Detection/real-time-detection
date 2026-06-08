
# Usage: python checkui.py --video hometrial.mp4 --model best.pt

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def main(video_path, model_path):
    # Load YOLO model
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    print(f"Running friend/foe test on {video_path} using {model_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.predict(frame, device=device, imgsz=320)
        res = results[0]
        data = res.boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,cls
        classes = data[:,5].astype(int)

        # Filter: if any foe (class 1), drop friend boxes
        if 1 in classes:
            mask = (classes == 1)
        else:
            mask = np.ones_like(classes, dtype=bool)
        filtered = data[mask]

        # Draw filtered boxes
        for x1,y1,x2,y2,conf,cls in filtered:
            label = 'Friend' if int(cls)==0 else 'Foe'
            color = (0,255,0) if cls==0 else (0,0,255)
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1),int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Friend/Foe Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Friend/Foe filtering on a video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--model', default='yolov11s.pt', help='YOLO model weights')
    args = parser.parse_args()
    main(args.video, args.model)
