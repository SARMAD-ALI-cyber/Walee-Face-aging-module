import torch
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import sys
sys.path.append(".")
sys.path.append("face_reaging\\model")
from PIL import Image
from model.models import UNet
from test_functions import process_image
import mediapipe as mp
import time
import openvino as ov
from test_functions import init_face_detector
# Default settings
window_size = 256 #512
stride = 256 # 256

def main(model_path, source_age, target_age):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running On GPU {device}")
    else:
        device = torch.device("cpu")
        print(f"Running On CPU {device}")

    # Load your model
    unet_model = UNet().to(device)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model=unet_model.eval()
    unet_model=torch.jit.script(unet_model)
    #Initialize face detector
    detector=init_face_detector()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    desired_fps = 30
    # Optimize video capture settings
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
      

    # Confirm the FPS setting
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Requested FPS: {desired_fps}, Actual FPS: {actual_fps}")

    print("Press 'q' to quit the live feed.")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to RGB 
        
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        '''Face cut code here if needed'''

                
        '''Get time stats for each frame'''
        start=time.time()
        processed_face_pil = process_image(unet_model, rgb_frame, video=False, 
                                                source_age=source_age, target_age=target_age, 
                                               window_size=window_size, stride=stride,detector=detector)
        
        processed_face_np = np.array(processed_face_pil)
        processed_face_np= cv2.cvtColor(processed_face_np, cv2.COLOR_RGB2BGR)
        print(f"Time is {time.time()-start}")
        cv2.imshow('Webcam Live Feed', processed_face_np)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(model_path="E:\\WALEE_INTERNSHIP\\Age_Transformation\\Face_Reagging\\best_unet_model.pth", 
         target_age=90, source_age=20)


