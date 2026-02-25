import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
import time
import torch
import os

class FallDetector:
    def __init__(self, model_path='yolov12n1.pt', confidence=0.5):
        """Initialize the fall detection system.
        
        Args:
            model_path (str): Path to the YOLOv12 model file (default: 'yolov12n1.pt')
            confidence (float): Detection confidence threshold (0-1)
        """
        # First verify CUDA is working
        try:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                self.device = "cuda"
                
                # Force CUDA initialization
                torch.cuda.init()
                x = torch.tensor([1.0], device="cuda")
                print(f"Test tensor device: {x.device}")
            else:
                self.device = "cpu"
                print("Using CPU for detection")
            
            # Initialize YOLOv12 model with explicit task
            print(f"Loading YOLOv12 model: {model_path}")
            
            # Download the model if needed, but DON'T overwrite model_path
            try:
                from ultralytics.utils.downloads import attempt_download
                downloaded_path = attempt_download(model_path)
                if downloaded_path:
                    print(f"Downloaded model to: {downloaded_path}")
                    model_path = str(downloaded_path)  # Only update if successful
                else:
                    print("Download returned None, using original path")
            except Exception as download_error:
                print(f"Model download error (continuing with original path): {download_error}")
            
            # Load model with explicit task type
            print(f"Using model path: {model_path}")
            self.model = YOLO(model_path, task='detect')
            self.model.to(self.device)
            print(f"Successfully loaded YOLOv12 model on {self.device}")
            
            # Verify model is on correct device
            model_device = next(self.model.parameters()).device
            print(f"Model confirmed on device: {model_device}")
            
            # Person class ID for COCO dataset
            self.person_class_id = 0
            self.confidence = confidence
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Fall detection parameters
        self.fall_threshold = 0.4  # Threshold for fall detection
        self.angle_threshold = 45  # Threshold for body angle
        self.prev_poses = []  # Store previous poses for motion analysis
        self.fall_detected = False
        self.fall_start_time = None
        self.fall_cooldown = 3  # Cooldown in seconds after fall detection
        
        # Fall type classification attributes
        self.fall_types = {
            "step_and_fall": False,
            "slip_and_fall": False, 
            "trip_and_fall": False,
            "stump_and_fall": False
        }
        self.motion_history = []  # Store motion history for fall type classification
        
        # Person tracking attributes
        self.next_person_id = 1
        self.person_trackers = {}  # Dictionary to track persons across frames {id: tracker_info}
        self.fallen_person_ids = set()  # Set of IDs of persons who are currently fallen
        
        # Landmark visualization settings
        self.show_landmarks = False  # Toggle for showing pose landmarks
        self.landmark_color_normal = (0, 255, 0)  # Green for normal
        self.landmark_color_fall = (0, 0, 255)  # Red for fall detected
        
        # Key landmark indices for visualization (MediaPipe pose landmarks)
        # 0: nose, 11-12: shoulders, 13-14: elbows, 15-16: wrists,
        # 23-24: hips, 25-26: knees, 27-28: ankles
        self.key_landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        
    def detect_person(self, frame):
        """Detect persons in the frame using YOLO."""
        # Process frame with explicit device
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != 'uint8':
                    frame = cv2.convertScaleAbs(frame)
                
            # Run inference with explicit device
            results = self.model(frame, 
                                verbose=False, 
                                conf=self.confidence,
                                device=self.device)
            
            person_boxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    if cls == self.person_class_id and conf > self.confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        person_boxes.append((x1, y1, x2, y2))
            
            return person_boxes
        except Exception as e:
            print(f"Error in detect_person: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_key_landmarks(self, frame, landmarks, person_box, is_fall=False):
        """Draw key pose landmarks on the frame.
        
        Args:
            frame: Output frame to draw on
            landmarks: List of pose landmarks (normalized coordinates)
            person_box: Person bounding box (x1, y1, x2, y2)
            is_fall: Whether a fall is detected (changes color to red)
        """
        if not self.show_landmarks or not landmarks:
            return
            
        x1, y1, x2, y2 = person_box
        h, w = y2 - y1, x2 - x1
        
        # Choose color based on fall status
        color = self.landmark_color_fall if is_fall else self.landmark_color_normal
        
        # Draw key landmark points as circles
        for idx in self.key_landmark_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                # Convert normalized coordinates to pixel coordinates within the bounding box
                px = int(lm[0] * w) + x1
                py = int(lm[1] * h) + y1
                
                # Check visibility (if available)
                visibility = lm[3] if len(lm) > 3 else 1.0
                if visibility > 0.5:
                    # Draw filled circle for the landmark
                    cv2.circle(frame, (px, py), 8, color, -1)
                    # Draw outer ring for better visibility
                    cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
        
        # Draw connections between key landmarks for skeleton visualization
        connections = [
            (11, 12),  # shoulders
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 23), (12, 24),  # torso
            (23, 24),  # hips
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28),  # right leg
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                # Check visibility
                start_vis = start_lm[3] if len(start_lm) > 3 else 1.0
                end_vis = end_lm[3] if len(end_lm) > 3 else 1.0
                
                if start_vis > 0.5 and end_vis > 0.5:
                    start_px = int(start_lm[0] * w) + x1
                    start_py = int(start_lm[1] * h) + y1
                    end_px = int(end_lm[0] * w) + x1
                    end_py = int(end_lm[1] * h) + y1
                    
                    cv2.line(frame, (start_px, start_py), (end_px, end_py), color, 3)
    
    def analyze_pose(self, frame, person_box):
        """Analyze pose for a detected person using MediaPipe.
        
        Args:
            frame: Input frame
            person_box: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            tuple: (landmarks, pose_features) or (None, None) if pose detection fails
        """
        x1, y1, x2, y2 = person_box
        
        # Extract the person from the frame
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return None, None
        
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions for MediaPipe
        h, w = rgb_img.shape[:2]
        
        # Process with MediaPipe
        results = self.pose.process(image=rgb_img)
        
        if not results.pose_landmarks:
            return None, None
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
        
        # Calculate pose features
        pose_features = self.calculate_pose_features(landmarks)
        
        return landmarks, pose_features
    
    def calculate_pose_features(self, landmarks):
        """Calculate pose features from landmarks.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            dict: Dictionary of pose features or None if calculation fails
        """
        if not landmarks:
            return None
        
        # Extract key landmarks (indices may vary based on MediaPipe version)
        # Head, shoulders, hips, knees, ankles
        key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        
        # Extract x, y coordinates of key points
        key_landmarks = [landmarks[i][:2] for i in key_points if i < len(landmarks)]
        
        if len(key_landmarks) < len(key_points):
            return None
        
        # Calculate height of the person (vertical distance between head and feet)
        head_y = landmarks[0][1]
        left_ankle_y = landmarks[27][1]
        right_ankle_y = landmarks[28][1]
        ankle_y = (left_ankle_y + right_ankle_y) / 2
        height = abs(ankle_y - head_y)
        
        # Calculate orientation (angle with the vertical)
        # Using the spine (mid-point of shoulders to mid-point of hips)
        mid_shoulder_x = (landmarks[11][0] + landmarks[12][0]) / 2
        mid_shoulder_y = (landmarks[11][1] + landmarks[12][1]) / 2
        mid_hip_x = (landmarks[23][0] + landmarks[24][0]) / 2
        mid_hip_y = (landmarks[23][1] + landmarks[24][1]) / 2
        
        dx = mid_hip_x - mid_shoulder_x
        dy = mid_hip_y - mid_shoulder_y
        
        angle = math.degrees(math.atan2(dx, dy))  # Angle with vertical axis
        
        # Extract shoulder, hip and feet positions for fall type detection
        shoulder_pos = (mid_shoulder_x, mid_shoulder_y)
        hip_pos = (mid_hip_x, mid_hip_y)
        feet_pos = ((landmarks[27][0] + landmarks[28][0])/2, (landmarks[27][1] + landmarks[28][1])/2)
        
        # Calculate bounding box aspect ratio (height/width) - helps detect lying down
        left_most_x = min(landmarks[11][0], landmarks[23][0], landmarks[25][0], landmarks[27][0])
        right_most_x = max(landmarks[12][0], landmarks[24][0], landmarks[26][0], landmarks[28][0])
        top_most_y = min(landmarks[0][1], landmarks[11][1], landmarks[12][1])
        bottom_most_y = max(landmarks[27][1], landmarks[28][1])
        
        bbox_width = right_most_x - left_most_x
        bbox_height = bottom_most_y - top_most_y
        aspect_ratio = bbox_height / max(bbox_width, 0.0001)  # Avoid division by zero
        
        # Calculate distances between key points
        shoulder_to_hip_distance = math.sqrt((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2)
        hip_to_feet_distance = math.sqrt((mid_hip_x - feet_pos[0])**2 + (mid_hip_y - feet_pos[1])**2)
        
        # Calculate velocity features if we have previous poses
        velocity_y = 0  # Vertical velocity
        velocity_x = 0  # Horizontal velocity
        acceleration = 0
        jerk = 0  # Rate of change of acceleration
        
        if self.prev_poses:
            prev = self.prev_poses[-1]
            time_diff = 1  # Assuming consistent frame rate for simplicity
            
            # Calculate vertical velocity
            velocity_y = (mid_shoulder_y - prev["mid_shoulder_y"]) / time_diff
            
            # Calculate horizontal velocity
            velocity_x = (mid_shoulder_x - prev["mid_shoulder_x"]) / time_diff
            
            # Calculate acceleration if we have at least 2 previous poses
            if len(self.prev_poses) >= 2:
                prev_velocity_y = (prev["mid_shoulder_y"] - self.prev_poses[-2]["mid_shoulder_y"]) / time_diff
                acceleration = (velocity_y - prev_velocity_y) / time_diff
                
                # Calculate jerk if we have at least 3 previous poses
                if len(self.prev_poses) >= 3:
                    prev_acceleration = (prev_velocity_y - (self.prev_poses[-2]["mid_shoulder_y"] - self.prev_poses[-3]["mid_shoulder_y"]) / time_diff) / time_diff
                    jerk = (acceleration - prev_acceleration) / time_diff
        
        features = {
            "height": height,
            "angle": angle,
            "velocity_y": velocity_y,
            "velocity_x": velocity_x,
            "acceleration": acceleration,
            "jerk": jerk,
            "mid_shoulder_y": mid_shoulder_y,
            "mid_shoulder_x": mid_shoulder_x,
            "mid_hip_y": mid_hip_y,
            "mid_hip_x": mid_hip_x,
            "shoulder_pos": shoulder_pos,
            "hip_pos": hip_pos,
            "feet_pos": feet_pos,
            "aspect_ratio": aspect_ratio,
            "shoulder_to_hip_distance": shoulder_to_hip_distance,
            "hip_to_feet_distance": hip_to_feet_distance,
            "timestamp": time.time()
        }
        
        # Keep track of previous poses, limit to 15 for better motion analysis
        self.prev_poses.append(features)
        if len(self.prev_poses) > 15:
            self.prev_poses.pop(0)
        
        return features
    
    def classify_fall_type(self, pose_features):
        """Classify the type of fall based on pose features.
        
        Args:
            pose_features: Dictionary of pose features
            
        Returns:
            str: Type of fall, one of ["step_and_fall", "slip_and_fall", "trip_and_fall", "stump_and_fall"]
        """
        # Reset fall types
        for fall_type in self.fall_types:
            self.fall_types[fall_type] = False
            
        if not pose_features or len(self.prev_poses) < 5:
            return None
            
        # Extract current and previous features
        current = pose_features
        prev = self.prev_poses[-5] if len(self.prev_poses) >= 5 else None
        
        if not prev:
            return None
        
        # Get feature differences over time
        angle_change = abs(current["angle"] - prev["angle"])
        velocity_y = current["velocity_y"]
        velocity_x = current["velocity_x"]
        acceleration = current["acceleration"]
        jerk = current["jerk"]
        aspect_ratio = current["aspect_ratio"]
        
        # Get average feature changes over the last few frames for smoother detection
        avg_velocity_y = sum([p.get("velocity_y", 0) for p in self.prev_poses[-3:]]) / min(3, len(self.prev_poses))
        avg_acceleration = sum([p.get("acceleration", 0) for p in self.prev_poses[-3:]]) / min(3, len(self.prev_poses))
        
        # Check for sharp changes in body posture
        body_posture_changed = current["aspect_ratio"] < 1.5  # Person is more horizontal than vertical
        
        # Step and Fall: Gradual angle change, moderate velocity, person loses footing
        if (angle_change > 20 and angle_change < 50 and 
            velocity_y > 0.1 and velocity_y < 0.4 and 
            abs(velocity_x) < 0.15 and 
            body_posture_changed):
            self.fall_types["step_and_fall"] = True
            return "step_and_fall"
            
        # Slip and Fall: Rapid angle change, high velocity, often backward motion
        elif (angle_change > 45 and 
              velocity_y > 0.3 and 
              acceleration > 0.2 and 
              abs(jerk) > 0.1 and
              body_posture_changed):
            self.fall_types["slip_and_fall"] = True
            return "slip_and_fall"
            
        # Trip and Fall: Forward momentum, moderate angle change, person pitches forward
        elif (angle_change > 30 and 
              abs(velocity_x) > 0.1 and 
              velocity_y > 0.2 and
              body_posture_changed):
            self.fall_types["trip_and_fall"] = True
            return "trip_and_fall"
            
        # Stump and Fall: Minimal horizontal movement, vertical drop, usually from standing still
        elif (abs(velocity_x) < 0.08 and 
              velocity_y > 0.2 and 
              angle_change > 25 and
              body_posture_changed):
            self.fall_types["stump_and_fall"] = True
            return "stump_and_fall"
            
        return None
    
    def detect_fall(self, pose_features):
        """Detect if a person has fallen based on pose features.
        
        Args:
            pose_features: Dictionary of pose features
            
        Returns:
            bool: True if fall is detected, False otherwise
        """
        if not pose_features:
            return False
        
        # Fall detection with improved criteria:
        # 1. Large angle with vertical axis (person not upright)
        # 2. Sudden change in vertical position
        # 3. Change in body proportions (aspect ratio)
        # 4. Motion patterns (velocity, acceleration, jerk)
        # 5. Final position near horizontal
        
        is_fall = False
        angle = abs(pose_features["angle"])
        aspect_ratio = pose_features["aspect_ratio"]
        
        # A fall usually involves:
        fall_criteria_met = 0
        
        # Criterion 1: Non-upright angle
        if angle > self.angle_threshold:
            fall_criteria_met += 1
            
        # Criterion 2: Aspect ratio indicates horizontal position
        if aspect_ratio < 1.5:  # More horizontal than vertical
            fall_criteria_met += 1
        
        # Only check for motion if we have enough history
        if len(self.prev_poses) > 5:
            # Get key poses to analyze movement
            current_features = pose_features
            prev_features = self.prev_poses[-6]  # 5 frames back
            
            # Criterion 3: Significant vertical movement (falling down)
            if current_features["mid_shoulder_y"] - prev_features["mid_shoulder_y"] > self.fall_threshold:
                fall_criteria_met += 1
            
            # Criterion 4: Rapid change in angle
            if abs(current_features["angle"] - prev_features["angle"]) > 30:
                fall_criteria_met += 1
                
            # Criterion 5: Acceleration spike (typical in falls)
            if abs(current_features.get("acceleration", 0)) > 0.2:
                fall_criteria_met += 1
                
            # Standing detection - check if the person appears to be standing upright
            is_standing = (
                aspect_ratio > 2.0 and  # Taller than wide (upright)
                abs(angle) < 20.0       # Nearly vertical orientation
            )
            
            # If person is clearly standing, they are not fallen regardless of other criteria
            if is_standing:
                return False
            
            # To detect a fall, we need multiple criteria to be met
            # This reduces false positives from normal movements
            if fall_criteria_met >= 3:  # Need at least 3 of the 5 criteria
                is_fall = True
                # Classify fall type
                self.classify_fall_type(pose_features)
        
        current_time = time.time()
        
        # Handle fall detection state and cooldown
        if is_fall and not self.fall_detected:
            self.fall_detected = True
            self.fall_start_time = current_time
            return True
        elif self.fall_detected:
            # Check if cooldown period has passed
            if current_time - self.fall_start_time > self.fall_cooldown:
                self.fall_detected = False
            return self.fall_detected
        
        return False
    
    def process_frame(self, frame):
        """Process a single frame for fall detection.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (output_frame, fall_detected, fall_data)
        """
        # Make a copy to avoid modifying the original
        output_frame = frame.copy()
        
        # Detect persons in the frame
        person_boxes = self.detect_person(frame)
        
        # Track persons across frames
        box_to_id = self.track_persons(frame, person_boxes)
        
        # List to store all person IDs in the current frame
        current_person_ids = list(box_to_id.values())
        
        falls_detected = False
        fall_data = {
            "person_count": len(person_boxes),
            "fall_detected": False,
            "fall_type": None,
            "person_boxes": person_boxes,
            "critical_points": [],
            "person_ids": current_person_ids,
            "fallen_ids": []
        }
        
        # Store landmarks and features for each person
        pose_data = []
        
        # Process each detected person to get pose data (we'll use this for our enhanced detection)
        for i, box in enumerate(person_boxes):
            person_id = box_to_id.get(i)
            x1, y1, x2, y2 = box
            
            # Analyze pose - use the existing method from the class 
            landmarks, pose_features = self.analyze_pose(frame, box)
            
            # Store the data for our enhanced detection
            pose_data.append((person_id, landmarks, pose_features, (x1, y1, x2, y2)))
        
        # Add better fall detection using collected pose data
        for person_id, landmarks, pose_features, (x1, y1, x2, y2) in pose_data:
            # Track if this specific person has fallen in this frame
            person_has_fallen = False
            
            # Check if this person is already marked as fallen from a previous frame
            if person_id in self.fallen_person_ids:
                person_has_fallen = True
            # Check for new fall if not already marked
            elif landmarks and pose_features:
                # Calculate angle from pose features
                angle = pose_features.get("angle", 0)
                    
                # Stricter fall detection criteria
                if abs(angle) >= (self.angle_threshold + 10):  # More horizontal posture
                    # Additional checks like aspect ratio
                    aspect_ratio = pose_features.get("aspect_ratio", 2.0)
                    if aspect_ratio < 1.5:  # More horizontal than vertical
                        # This is likely a real fall
                        person_has_fallen = True
                        
                        # Determine fall type
                        fall_type = None
                        for ft, is_active in self.fall_types.items():
                            if is_active:
                                fall_type = ft
                                break
                            
                        # If no fall type was determined, use a default
                        if not fall_type:
                            fall_type = "slip_and_fall"
                    
                        # Update fall data
                        if person_id is not None and person_id not in fall_data["fallen_ids"]:
                            fall_data["fallen_ids"].append(person_id)
                            self.fallen_person_ids.add(person_id)
                            
                        fall_data["fall_detected"] = True
                        fall_data["fall_type"] = fall_type
                        falls_detected = True
                        
                        # Add visual indication
                        cv2.putText(
                            output_frame,
                            f"FALL DETECTED: {fall_type}",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2
                        )
            
            # Draw key landmarks if enabled (after determining fall status)
            if landmarks:
                self.draw_key_landmarks(output_frame, landmarks, (x1, y1, x2, y2), is_fall=person_has_fallen)
        
        return output_frame, falls_detected, fall_data
    
    def _check_critical_points_ratio(self, keypoints):
        """Calculate ratio of critical points for fall detection."""
        # Helper method to calculate critical points ratio
        # Used by the improved fall detection logic
        
        valid_points = 0
        total_points = 0
        
        # Critical points for fall detection
        critical_indices = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        
        for idx in critical_indices:
            if idx < len(keypoints) and keypoints[idx] is not None:
                visibility = keypoints[idx].visibility if hasattr(keypoints[idx], 'visibility') else 1.0
                if visibility > 0.5:
                    valid_points += 1
            total_points += 1
        
        # Return ratio of valid to total points
        return 0 if total_points == 0 else valid_points / total_points

    def track_persons(self, frame, person_boxes):
        """Track persons across frames and assign consistent IDs.
        
        Args:
            frame: Input frame
            person_boxes: List of person bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Dictionary mapping box indices to person IDs
        """
        current_time = time.time()
        current_boxes = person_boxes
        box_to_id = {}
        
        # If no trackers yet, initialize with detected boxes
        if not self.person_trackers:
            for i, box in enumerate(current_boxes):
                self.person_trackers[self.next_person_id] = {
                    'box': box,
                    'last_seen': current_time,
                    'is_fallen': False
                }
                box_to_id[i] = self.next_person_id
                self.next_person_id += 1
            return box_to_id
        
        # Match current boxes with existing trackers using IoU
        matched_indices = []
        
        for i, current_box in enumerate(current_boxes):
            best_iou = 0.3  # Minimum IoU threshold for matching
            best_id = None
            
            for person_id, tracker_info in self.person_trackers.items():
                previous_box = tracker_info['box']
                
                # Calculate IoU between current box and previous box
                x1 = max(current_box[0], previous_box[0])
                y1 = max(current_box[1], previous_box[1])
                x2 = min(current_box[2], previous_box[2])
                y2 = min(current_box[3], previous_box[3])
                
                if x2 < x1 or y2 < y1:
                    # No overlap
                    continue
                    
                current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                previous_area = (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])
                intersection = (x2 - x1) * (y2 - y1)
                union = current_area + previous_area - intersection
                
                iou = intersection / union
                
                if iou > best_iou:
                    best_iou = iou
                    best_id = person_id
            
            if best_id is not None:
                # Match found
                box_to_id[i] = best_id
                self.person_trackers[best_id]['box'] = current_box
                self.person_trackers[best_id]['last_seen'] = current_time
                matched_indices.append(best_id)
            else:
                # New person
                self.person_trackers[self.next_person_id] = {
                    'box': current_box,
                    'last_seen': current_time,
                    'is_fallen': False
                }
                box_to_id[i] = self.next_person_id
                self.next_person_id += 1
        
        # Remove trackers for persons not seen recently (5 seconds)
        ids_to_remove = []
        for person_id, tracker_info in self.person_trackers.items():
            if current_time - tracker_info['last_seen'] > 5.0:
                ids_to_remove.append(person_id)
                # Also remove from fallen_person_ids if present
                if person_id in self.fallen_person_ids:
                    self.fallen_person_ids.remove(person_id)
        
        for person_id in ids_to_remove:
            del self.person_trackers[person_id]
        
        return box_to_id 