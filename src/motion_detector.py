import cv2
import numpy as np
from datetime import datetime
import time

class MotionDetector:
    def __init__(self, config):
        self.config = config
        self.camera_source = config['camera']['source']
        self.motion_threshold = config['camera']['motion_threshold']
        self.min_motion_frames = config['camera']['min_motion_frames']
        self.cooldown_seconds = config['camera']['cooldown_seconds']
        
        self.cap = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.motion_frames = 0
        self.last_motion_time = 0
        
    def start_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        return True
    
    def detect_motion(self, frame):
        """Detect motion in frame"""
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows and noise
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Count non-zero pixels
        motion_pixels = cv2.countNonZero(thresh)
        
        return motion_pixels > self.motion_threshold, motion_pixels
    
    def capture_frame(self):
        """Capture a single frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def process_stream(self):
        """Main processing loop - yields frames when motion detected"""
        while True:
            frame = self.capture_frame()
            if frame is None:
                break
            
            has_motion, motion_level = self.detect_motion(frame)
            
            current_time = time.time()
            
            if has_motion:
                self.motion_frames += 1
                
                # Check if we've detected enough consecutive motion frames
                if self.motion_frames >= self.min_motion_frames:
                    # Check cooldown period
                    if current_time - self.last_motion_time > self.cooldown_seconds:
                        self.last_motion_time = current_time
                        self.motion_frames = 0
                        yield frame, motion_level
            else:
                self.motion_frames = 0
            
            # Allow breaking the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
