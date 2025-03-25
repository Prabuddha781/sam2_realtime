import os
import time
import cv2
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_object_tracker

# Global variables for mouse interaction
clicked_points = []
clicked_labels = []  # 1 for foreground, 0 for background
current_frame = None

# Mouse callback function
def mouse_click(event, x, y, flags, param):
    global clicked_points, clicked_labels, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click - foreground point (label 1)
        clicked_points.append([x, y])
        clicked_labels.append(1)
        print(f"Added foreground point at ({x}, {y})")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click - background point (label 0) 
        clicked_points.append([x, y])
        clicked_labels.append(0)
        print(f"Added background point at ({x}, {y})")

class Visualizer:
    def __init__(self,
                 video_width,
                 video_height,
                 ):
        
        self.video_width = video_width
        self.video_height = video_height
        cv2.namedWindow('SAM2 Tracking')
        cv2.setMouseCallback('SAM2 Tracking', mouse_click)

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False,
                                               )
        
        return mask

    def add_frame(self, frame, mask):
        global current_frame
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        current_frame = frame.copy()
        
        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()
        
        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            frame[obj_mask] = [255, 105, 180]
                
        # Display frame with OpenCV instead of IPython
        cv2.imshow('SAM2 Tracking', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Signal to stop processing
        return True

# Set SAM2 Configuration
NUM_OBJECTS = 10  # Increased to allow multiple point selections
SAM_CHECKPOINT_FILEPATH = "./checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
DEVICE = 'cuda:0'

# Set fixed resolution for processing
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Open Webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam")
    exit()

# Use fixed resolution instead of native webcam resolution
video_width = FRAME_WIDTH
video_height = FRAME_HEIGHT

print(f"Using resolution: {video_width}x{video_height}")

# For real-time visualization
visualizer = Visualizer(video_width=video_width,
                        video_height=video_height
                        )

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False
                                )

available_slots = np.inf
tracking_initialized = False
tracked_objects = []

print("Starting webcam tracking. Click on objects to track them.")
print("- Left-click: Add foreground point")
print("- Right-click: Add background point")
print("- Press 'q' to quit")

with torch.inference_mode(), torch.autocast('cuda:0', dtype=torch.bfloat16):
    while True:
        start_time = time.time()

        # Get next frame from webcam
        ret, frame = webcam.read()
        
        # Exit if frame could not be grabbed
        if not ret:
            print("Error: Could not read frame from webcam")
            break

        # Resize frame to target resolution
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Convert frame from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Check if points have been clicked
        if len(clicked_points) > 0:
            # Process new point selections
            points = np.array(clicked_points)
            labels = np.array(clicked_labels)
            
            # Track with clicked points
            sam_out = sam.track_new_object(img=img, 
                                          points=points)
            
            # Clear points after use
            clicked_points = []
            clicked_labels = []
            tracking_initialized = True
            
        elif tracking_initialized:
            # Continue tracking existing objects
            sam_out = sam.track_all_objects(img=img)
        else:
            # Display frame without tracking until a point is selected
            dummy_mask = np.zeros((1, 1, img.shape[0], img.shape[1]), dtype=np.float32)
            if not visualizer.add_frame(frame=frame, mask=dummy_mask):
                break
            continue
            
        # Display frame with masks
        if not visualizer.add_frame(frame=frame, mask=sam_out['pred_masks']):
            break
        
        # Calculate and print FPS
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}", end="\r")
        
# Clean up
webcam.release()
cv2.destroyAllWindows()
print("\nWebcam tracking stopped")
