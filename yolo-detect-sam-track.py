import os
import time
import urllib
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image
import random

from sam2.build_sam import build_sam2_object_tracker

# Define classes we want to track
CLASSES_TO_TRACK = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']

class Visualizer:
    def __init__(self, video_width, video_height):
        self.video_width = video_width
        self.video_height = video_height
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        self.selected_box = None
        self.boxes = []
        self.masks = []
        self.class_ids = []
        self.colors = None
        cv2.setMouseCallback('Tracking', self.mouse_callback)

    def setup_yolo_colors(self, yolo_model):
        # Generate random colors for each YOLO class
        self.yolo_classes = list(yolo_model.names.values())
        self.classes_ids = [self.yolo_classes.index(clas) for clas in self.yolo_classes]
        self.colors = [random.choices(range(256), k=3) for _ in self.classes_ids]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is inside any mask
            for i, mask in enumerate(self.masks):
                # Check if coordinates are within mask bounds
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    print(mask[y, x], "mask")
                    if mask[y, x]:  # Check if clicked point is inside the mask
                        print("clicked inside")
                        self.selected_box = i
                        break
                else:
                    print("clicked outside")
        print("clicked")

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(mask,
                                            size=(self.video_height, self.video_width),
                                            mode="bilinear",
                                            align_corners=False)
        return mask

    def draw_yolo_detections(self, frame, boxes, masks, class_ids, scores=None):
        """Draw YOLO detections during selection phase"""
        frame = frame.copy()
        self.boxes = boxes
        self.masks = masks
        self.class_ids = class_ids
        
        # Create overlay for segmentation masks
        overlay = frame.copy()
        
        # Draw all YOLO detections
        for i, (box, mask, class_id) in enumerate(zip(boxes, masks, class_ids)):
            if yolo_model.names[class_id] in CLASSES_TO_TRACK:
                x1, y1 = box[0]
                x2, y2 = box[1]
                
                # Get color for this class
                color = self.colors[self.classes_ids.index(class_id)]
                
                # Fill the segmentation mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask] = color
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                
                # Draw class name and confidence score
                class_name = yolo_model.names[class_id]
                score_text = f"{scores[i]:.2f}" if scores is not None else ""
                label = f"{class_name} {score_text}"
                cv2.putText(overlay, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add instruction text
        cv2.putText(overlay, "Click on an object to track (press 'q' to quit)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Tracking', overlay)
        return cv2.waitKey(1) & 0xFF

    def draw_sam_tracking(self, frame, mask):
        """Draw SAM2 tracking mask only"""
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()
        
        # Create colored overlay for SAM2 tracking
        overlay = frame.copy()
        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            colored_mask = np.zeros_like(frame)
            colored_mask[obj_mask] = [180, 105, 255]  # BGR format for OpenCV
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
        
        # Add "Tracking" text to indicate tracking mode
        cv2.putText(overlay, "Tracking (press 'r' to reset, 'q' to quit)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
        cv2.imshow('Tracking', overlay)
        return cv2.waitKey(1) & 0xFF

# Load models and setup
SAM_CHECKPOINT_FILEPATH = "./checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
# SAM_CHECKPOINT_FILEPATH = "./checkpoints/sam2.1_hiera_large.pt"
# SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_l.yaml"
DEVICE = 'cuda:0'

# Initialize video capture and visualizer
video_stream = cv2.VideoCapture(0)  # Use default webcam (usually 0)
visualizer = Visualizer(video_width=640, video_height=640)

# Main loop
selection_phase = True
sam = None
yolo_model = None

with torch.inference_mode(), torch.autocast('cuda:0', dtype=torch.bfloat16):
    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            break

        # Resize frame to 640x640
        frame_resized = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        if selection_phase:
            # Load YOLO model only when needed
            if yolo_model is None:
                yolo_model = YOLO("yolov8x-seg.pt")
                visualizer.setup_yolo_colors(yolo_model)

            # Run YOLO segmentation
            results = yolo_model(img, verbose=False)
            # Get bounding boxes, masks, and scores for relevant classes
            boxes = []
            masks = []
            scores = []
            class_ids = []
            
            for seg in results:
                # Process each detected object
                for i, (box, cls_id, conf) in enumerate(zip(seg.boxes.xyxy, seg.boxes.cls, seg.boxes.conf)):
                    if yolo_model.names[int(cls_id)] in CLASSES_TO_TRACK:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        boxes.append([[x1, y1], [x2, y2]])
                        # Create a binary mask from the segmentation data
                        if seg.masks is not None:
                            mask = seg.masks.data[i].cpu().numpy() > 0.5
                            print(mask.shape, "mask shape")
                            masks.append(mask)
                            scores.append(conf.item())
                            class_ids.append(int(cls_id))

            # Show YOLO segmentation masks and wait for selection
            key = visualizer.draw_yolo_detections(frame_resized, boxes, masks, class_ids, scores)
            
            # If user selected a box
            if visualizer.selected_box is not None:
                selection_phase = False
                selected_box = np.array([boxes[visualizer.selected_box]])
                selected_mask = masks[visualizer.selected_box]
                
                # Make sure selected_mask is in the correct shape
                if selected_mask.ndim == 2:
                    # Resize 2D mask to 256x256
                    mask_resized = cv2.resize(selected_mask.astype(np.uint8), (256, 256), 
                                            interpolation=cv2.INTER_NEAREST)
                    # Convert to binary mask after resize
                    mask_resized = mask_resized.astype(bool)
                    # Add batch dimension to get (N, H, W) format where N=1
                    mask_resized = np.expand_dims(mask_resized, axis=0)
                elif selected_mask.ndim == 3:
                    # If mask already has batch dimension (N, H, W)
                    # Resize each mask in the batch
                    N = selected_mask.shape[0]
                    mask_resized = np.zeros((N, 256, 256), dtype=bool)
                    for i in range(N):
                        mask_resized[i] = cv2.resize(selected_mask[i].astype(np.uint8), (256, 256),
                                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    print(f"Unexpected mask shape: {selected_mask.shape}")
                    continue
                
                print(f"Mask shape before tracking: {mask_resized.shape}")
                
                # Initialize SAM2 with single object
                sam = build_sam2_object_tracker(
                    num_objects=1,
                    config_file=SAM_CONFIG_FILEPATH,
                    ckpt_path=SAM_CHECKPOINT_FILEPATH,
                    device=DEVICE,
                    verbose=False
                )
                
                # Start tracking selected object
                sam_out = sam.track_new_object(img=img, box=selected_box, mask=mask_resized)
                
                # Clean up YOLO resources
                del yolo_model
                yolo_model = None
                torch.cuda.empty_cache()
                
            if key == ord('q'):
                break
                
        else:
            # Continue tracking with SAM2
            sam_out = sam.track_all_objects(img=img)
            
            # Display frame with SAM2 mask only
            key = visualizer.draw_sam_tracking(frame_resized, mask=sam_out['pred_masks'])
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset to selection phase
                selection_phase = True
                visualizer.selected_box = None
                # Clean up SAM2 resources
                del sam
                sam = None
                torch.cuda.empty_cache()

video_stream.release()
cv2.destroyAllWindows()
