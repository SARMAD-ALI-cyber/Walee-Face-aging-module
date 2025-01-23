import face_recognition
import numpy as np
import cupy as cp
import os
import time
import torch
from torch.amp import autocast
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import write_video
import tempfile
import subprocess
import json
from ffmpy import FFmpeg, FFprobe
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2

# Initialize face detector (do this once at startup)
def init_face_detector():
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    return detector

# Modified face detection and cropping function
def detect_and_crop_face(image, detector):
    """
    Detect face using MediaPipe and crop with the same margins as original code
    """
    # Convert BGR to RGB if needed (MediaPipe expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
    else:
        rgb_image = image
        
    # Process the image
    results = detector.process(rgb_image)
    
    if not results.detections:
        raise ValueError("No face detected in image")
    
    # Get the first face detection
    detection = results.detections[0]
    
    # Get bounding box coordinates
    bbox = detection.location_data.relative_bounding_box
    ih, iw = image.shape[:2]
    
    # Convert relative coordinates to absolute
    fl = [
        int(bbox.ymin * ih),                    # top
        int((bbox.xmin + bbox.width) * iw),     # right
        int((bbox.ymin + bbox.height) * ih),    # bottom
        int(bbox.xmin * iw)                     # left
    ]
    
    # Calculate margins (keeping the same logic as original)
    margin_y_t = int((fl[2] - fl[0]) * 0.63 * 0.85)  # Forehead margin
    margin_y_b = int((fl[2] - fl[0]) * 0.37 * 0.85)
    margin_x = int((fl[1] - fl[3]) // (2 / 0.85))
    margin_y_t += 2 * margin_x - margin_y_t - margin_y_b  # Ensure square crop

    # Calculate crop coordinates
    l_y = max([fl[0] - margin_y_t, 0])
    r_y = min([fl[2] + margin_y_b, image.shape[0]])
    l_x = max([fl[3] - margin_x, 0])
    r_x = min([fl[1] + margin_x, image.shape[1]])

    # Crop image
    cropped_image = image[l_y:r_y, l_x:r_x, :]
    orig_size = cropped_image.shape[:2]
    
    return cropped_image, orig_size, l_y, r_y, l_x, r_x

mask_file = torch.from_numpy(np.array(Image.open('assets/mask1024.jpg').convert('L'))) / 255
small_mask_file = torch.from_numpy(np.array(Image.open('assets/mask512.jpg').convert('L'))) / 255

def sliding_window_tensor(input_tensor, window_size, stride, your_model, mask=mask_file, small_mask=small_mask_file):
    """
    Apply aging operation on input tensor using a sliding-window method. This operation is done on the GPU, if available.
    """
    n, c, h, w = input_tensor.size()
    input_tensor = input_tensor.to(next(your_model.parameters()).device)
    # mask=torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    mask = mask.to(next(your_model.parameters()).device)

    # small_mask=torch.nn.functional.interpolate(small_mask.unsqueeze(0).unsqueeze(0), size=(window_size, window_size), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    small_mask = small_mask.to(next(your_model.parameters()).device)

    
    output_tensor = torch.zeros((n, 3, h, w), dtype=input_tensor.dtype, device=input_tensor.device)

    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32, device=input_tensor.device)
    
    # add = 2 if window_size % stride != 0 else 1

    # for y in range(0, h - window_size + add, stride):
    #     for x in range(0, w - window_size + add, stride):
    #         window = input_tensor[:, :, y:y + window_size, x:x + window_size]

    #         # Apply the same preprocessing as during training
    #         input_variable = Variable(window, requires_grad=False)  # Assuming GPU is available

    #         # Forward pass
    #         with torch.no_grad():
    #             output = your_model(input_variable)

    #         output_tensor[:, :, y:y + window_size, x:x + window_size] += output * small_mask
    #         count_tensor[:, :, y:y + window_size, x:x + window_size] += small_mask

    input_variable=Variable(input_tensor,requires_grad=False)
    with torch.no_grad():
        with autocast(dtype=torch.float16,device_type="cuda:0"):
            output=your_model(input_variable)
    
    output_tensor+=output* small_mask
    count_tensor+=small_mask

        

    count_tensor = torch.clamp(count_tensor, min=1.0)

    # Average the overlapping regions
    output_tensor /= count_tensor

    # Apply mask
    output_tensor *= small_mask
    
    output_tensor=output_tensor.to(next(your_model.parameters()).device)
    
    return output_tensor


def process_image(your_model, image, video, source_age, target_age=0,
                            window_size=512, stride=256, steps=18,detector=None):
    """
    Optimized aging function for real-time performance.
    """
    if video:
        target_age = 0
    input_size = (512, 512)

    # Convert image to NumPy/cupy array and ensure it's RGB
    image = cp.array(image)

    # Use GPU-accelerated face detection (already handled outside)
    # Here is time taken is more we should change face detector
    # fl = face_recognition.face_locations(image)[0]
    

    # # Calculate margins
    # margin_y_t = int((fl[2] - fl[0]) * 0.63 * 0.85)  # Forehead margin
    # margin_y_b = int((fl[2] - fl[0]) * 0.37 * 0.85)
    # margin_x = int((fl[1] - fl[3]) // (2 / 0.85))
    # margin_y_t += 2 * margin_x - margin_y_t - margin_y_b  # Ensure square crop

    # l_y = max([fl[0] - margin_y_t, 0])
    # r_y = min([fl[2] + margin_y_b, image.shape[0]])
    # l_x = max([fl[3] - margin_x, 0])
    # r_x = min([fl[1] + margin_x, image.shape[1]])

    # # Crop and resize the image for processing
    # cropped_image = image[l_y:r_y, l_x:r_x, :]
    # orig_size = cropped_image.shape[:2]
    
    cropped_image, orig_size,l_y,r_y,l_x,r_x = detect_and_crop_face(image.get(), detector)
    device = next(your_model.parameters()).device

    
    cropped_image_tensor = transforms.ToTensor()(cropped_image).to(device)
    
    
    cropped_image_resized = transforms.Resize(input_size, interpolation=Image.BILINEAR)(cropped_image_tensor)
    
    # Prepare the input tensor with age channels
    source_age_channel = torch.full_like(cropped_image_resized[:1, :, :], source_age / 100).to(device)
    target_age_channel = torch.full_like(cropped_image_resized[:1, :, :], target_age / 100).to(device)
    input_tensor = torch.cat([cropped_image_resized, source_age_channel, target_age_channel], dim=0).unsqueeze(0)
    
    
    # Perform model inference
    
    with torch.no_grad():
        with autocast(dtype=torch.float16,device_type="cuda:0"):
            aged_cropped_image = sliding_window_tensor(input_tensor, window_size, stride, your_model)
        
    
    
    # Resize back to the original size
    aged_cropped_image_resized = transforms.Resize(orig_size, interpolation=Image.BILINEAR)(aged_cropped_image)
    # Reapply the aged crop back onto the original image
    
    # image_tensor = transforms.ToTensor()(image).to(device)
    image_tensor=torch.as_tensor(image,device=device)
    image_tensor = image_tensor.permute(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    image_tensor = image_tensor / 255.0
    
    
    
    image_tensor[:, l_y:r_y, l_x:r_x] += aged_cropped_image_resized.squeeze(0)
    

    # Clamp and convert back to image
    
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    
    final_output=transforms.functional.to_pil_image(image_tensor)
    return final_output



def process_video(your_model, video_path, source_age, target_age, window_size=512, stride=256, frame_count=0):
    """
    Applying the aging to a video.
    We age as from source_age to target_age, and return an image.
    To limit the number of frames in a video, we can set frame_count.
    """

    # Extracting frames and placing them in a temporary directory
    frames_dir = tempfile.TemporaryDirectory()
    output_template = os.path.join(frames_dir.name, '%04d.jpg')

    if frame_count:
        ff = FFmpeg(
            inputs={video_path: None},
            outputs={output_template: ['-vf', f'select=lt(n\\,{frame_count})', '-q:v', '1']}
        )
    else:
        ff = FFmpeg(
            inputs={video_path: None},
            outputs={output_template: ['-q:v', '1']}
        )

    ff.run()

    # Getting framerate (for reconstruction later)
    ff = FFprobe(inputs={video_path: None},
                 global_options=['-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=r_frame_rate', '-of',
                                 'default=noprint_wrappers=1:nokey=1'])
    stdout, _ = ff.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_rate = eval(stdout.decode('utf-8').strip())


    # Applying process_image to frames
    processed_dir = tempfile.TemporaryDirectory()

    for name in os.listdir(frames_dir.name):
        image_path = os.path.join(frames_dir.name, name)
        image = Image.open(image_path).convert('RGB')
        image_aged = process_image(your_model, image, False, source_age, target_age, window_size, stride)
        image_aged.save(os.path.join(processed_dir.name, name))

    # Generating a new video
    input_template = os.path.join(processed_dir.name, '%04d.jpg')
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ff = FFmpeg(
        inputs={input_template: f'-framerate {frame_rate}'}, global_options=['-y'],
        outputs={output_file.name: ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']}
    )

    ff.run()

    frames_dir.cleanup()
    processed_dir.cleanup()

    return output_file.name

