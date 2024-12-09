import os
import cv2
import numpy as np
import torch
from torch import nn
import time
from fsrcnn import FSRCNN


def main() -> None:
    # Hyperparameters
    upscaling_factor = 2
    d = 56
    s = 12
    m = 4
    model_name = "best_model_big_x2"

    # Torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model_dir = os.path.join("..", "models")

    model = FSRCNN(upscaling_factor=upscaling_factor,
                   d=d,
                   s=s,
                   m=m).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pt"), weights_only=False)["model_state_dict"])
    model.eval()

    # Capture camera and do real time
    cap = cv2.VideoCapture(0)
    
    # FPS calculation variables
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS calculation
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)

        # Preprocess
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        frame_y = frame_ycrcb[:, :, 0]
        frame_y = torch.tensor(frame_y, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Predict
        with torch.no_grad():
            frame_y_hr = model(frame_y).cpu().numpy().clip(0, 255).astype("uint8")[0, 0, :, :]

        # Postprocess
        frame_crcb_hr = cv2.resize(frame_ycrcb[:, :, 1:], (frame_y_hr.shape[1], frame_y_hr.shape[0]), cv2.INTER_CUBIC)
        frame_ycrcb_hr = cv2.merge([frame_y_hr, frame_crcb_hr])
        frame_hr = cv2.cvtColor(frame_ycrcb_hr, cv2.COLOR_YCrCb2BGR)

        # Put FPS text on frame
        cv2.putText(frame_hr, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display both frames
        cv2.imshow("Original", frame)
        cv2.imshow("Super Resolution", frame_hr)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
