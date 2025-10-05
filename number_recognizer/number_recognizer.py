# number_recognizer.py
import cv2
import numpy as np
from ultralytics import YOLO


class NumberRecognizer:
    def __init__(self, model_path="models/yolo11m.pt", conf=0.4, gamma=1.4):
        """
        Initialize the jersey number recognizer with enhancement and tracking support.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.gamma = gamma
        self.tracker_dict = {}  # player_id → (jersey_number, last_seen_frame)

    # ---------- IMAGE ENHANCEMENT ----------
    def enhance_image(self, image):
        """Apply CLAHE + Gamma correction to enhance jersey visibility."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        gamma_corrected = np.array(255 * (enhanced / 255) ** (1 / self.gamma), dtype='uint8')
        return gamma_corrected

    # ---------- JERSEY NUMBER DETECTION ----------
    def recognize_number(self, frame, player_bbox, player_id=None, frame_idx=None):
        """
        Detect jersey numbers inside a player's bounding box.
        Maintains tracking linkage (reuse last number if not detected).
        """
        x1, y1, x2, y2 = map(int, player_bbox)
        player_crop = frame[y1:y2, x1:x2]

        # Enhance the cropped region
        player_crop = self.enhance_image(player_crop)

        # Run YOLO detection on the enhanced crop
        results = self.model.predict(player_crop, conf=self.conf, imgsz=320, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            # No detection → use last known number if exists
            if player_id in self.tracker_dict:
                self.tracker_dict[player_id] = (self.tracker_dict[player_id][0], frame_idx)
                return self.tracker_dict[player_id][0]
            return None

        # Pick the highest-confidence detection
        best_box = max(results[0].boxes, key=lambda b: float(b.conf))
        cls_id = int(best_box.cls)
        conf = float(best_box.conf)

        if conf < self.conf:
            if player_id in self.tracker_dict:
                self.tracker_dict[player_id] = (self.tracker_dict[player_id][0], frame_idx)
                return self.tracker_dict[player_id][0]
            return None

        jersey_number = self.model.names[cls_id]

        # Save or update the detected number in the tracker dictionary
        if player_id is not None:
            self.tracker_dict[player_id] = (jersey_number, frame_idx)

        return jersey_number

    # ---------- CLEANUP OLD TRACKS ----------
    def cleanup(self, current_frame_idx, max_inactive_frames=30):
        """
        Remove players not seen for a while from memory.
        """
        to_delete = [pid for pid, (_, last_seen) in self.tracker_dict.items()
                     if current_frame_idx - last_seen > max_inactive_frames]
        for pid in to_delete:
            del self.tracker_dict[pid]
