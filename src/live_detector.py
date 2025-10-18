import cv2
import time
import numpy as np
from datetime import datetime

class LiveDetector:
    def __init__(self, config):
        self.config = config
        self.camera_source = config['camera']['source']
        self.detection_interval = config['camera'].get('detection_interval', 15)
        
        # Motion detection with background subtraction
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.motion_threshold = config['camera'].get('motion_threshold', 5000)
        
        # Load Haar Cascade for FACE ONLY
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade classifier")
                
            print("✓ Face cascade classifier loaded")
        except Exception as e:
            print(f"Error loading cascade: {str(e)}")
            raise
        
        self.cap = None
        self.last_detection_time = 0
        self.is_monitoring = False
        
        # Initialize MediaPipe for hand detection
        self.use_hand_tracking = True
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            print("✓ MediaPipe hand tracking enabled")
        except Exception as e:
            print(f"⚠ MediaPipe not available: {str(e)}")
            self.use_hand_tracking = False
        
        print("✓ Motion-first detection strategy enabled")
        print("  Camera is always on, but Azure analysis only when motion detected")
    
    def start_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        print("✓ Camera started successfully")
        return True
    
    def detect_faces(self, frame):
        """Detect faces only (no body detection)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def detect_hands(self, frame):
        """Detect hands using MediaPipe"""
        if not self.use_hand_tracking:
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks
            return []
        except:
            return []
    
    def draw_detections(self, frame, faces, hands):
        """Draw face rectangles and hand landmarks"""
        # Draw green rectangles for faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw colorful hand landmarks
        if self.use_hand_tracking and hands:
            for hand_landmarks in hands:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def detect_motion(self, frame):
        """Detect motion in frame using background subtraction"""
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows and noise
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Count non-zero pixels
        motion_pixels = cv2.countNonZero(thresh)
        
        has_motion = motion_pixels > self.motion_threshold
        
        return has_motion, motion_pixels
    
    def should_process_frame(self):
        """Check if enough time has passed for next detection"""
        current_time = time.time()
        if current_time - self.last_detection_time >= self.detection_interval:
            self.last_detection_time = current_time
            return True
        return False
    
    def process_live_stream(self):
        """Process live camera stream with motion-first strategy"""
        self.is_monitoring = True
        
        cv2.namedWindow("CASC - Live Monitoring", cv2.WINDOW_NORMAL)
        
        print("Video window opened...")
        print("Strategy: Motion Detection → Face Detection → Azure Analysis")
        print("Azure API calls only when motion + face detected")
        
        while self.is_monitoring:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # STEP 1: Check for motion FIRST
            has_motion, motion_pixels = self.detect_motion(frame)
            
            # Initialize detection results
            faces = []
            hands = []
            should_analyze = False
            
            # STEP 2: If motion detected, then check for faces and hands
            if has_motion:
                faces = self.detect_faces(frame)
                hands = self.detect_hands(frame)
                
                # STEP 3: Only send to Azure if faces detected AND cooldown passed
                if len(faces) > 0 and self.should_process_frame():
                    should_analyze = True
            
            # Draw detections
            display_frame = self.draw_detections(frame.copy(), faces, hands)
            
            # Add motion indicator
            motion_status = "MOTION DETECTED" if has_motion else "No Motion"
            motion_color = (0, 255, 0) if has_motion else (128, 128, 128)
            cv2.putText(display_frame, motion_status, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, motion_color, 2)
            
            # Add Azure analysis indicator
            if should_analyze:
                cv2.putText(display_frame, ">>> SENDING TO AZURE <<<", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Add timestamp and detection count
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(display_frame, timestamp, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f'Faces: {len(faces)} | Hands: {len(hands)} | Motion: {motion_pixels}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show next detection countdown
            time_until_next = int(self.detection_interval - (time.time() - self.last_detection_time))
            if time_until_next < 0:
                time_until_next = 0
            cv2.putText(display_frame, f'Next Azure check in: {time_until_next}s', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add monitoring status
            cv2.putText(display_frame, 'LIVE - Press Q to quit', 
                       (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("CASC - Live Monitoring", display_frame)
            
            # Process window events
            key = cv2.waitKey(1) & 0xFF
            
            # STEP 4: Only yield (send to Azure) when motion + face detected + cooldown passed
            if should_analyze:
                yield frame, faces, len(hands)
            
            # Exit on 'q' key
            if key == ord('q'):
                self.is_monitoring = False
                break
    
    def capture_single_frame(self):
        """Capture a single frame"""
        if self.cap is None or not self.cap.isOpened():
            self.start_camera()
        
        ret, frame = self.cap.read()
        if ret:
            faces = self.detect_faces(frame)
            hands = self.detect_hands(frame)
            return frame, faces, len(hands)
        return None, [], 0
    
    def release(self):
        """Release camera resources"""
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
        if self.use_hand_tracking and hasattr(self, 'hands'):
            self.hands.close()
        cv2.destroyAllWindows()
        print("✓ Camera released")

# Add standalone testing capability
def main():
    """Test the live detector standalone"""
    print("="*70)
    print("Testing Live Detector - Standalone Mode")
    print("="*70)
    print("Press 'q' to quit\n")
    
    # Create minimal config for testing
    test_config = {
        'camera': {
            'source': 0,
            'detection_interval': 5  # 5 seconds for testing
        }
    }
    
    try:
        detector = LiveDetector(test_config)
        detector.start_camera()
        
        print("Camera started. Displaying live feed...")
        print("Detections will be processed every 5 seconds")
        
        detection_count = 0
        for frame, faces, hands in detector.process_live_stream():
            detection_count += 1
            print(f"\nDetection #{detection_count}:")
            print(f"  - Faces: {len(faces)}")
            print(f"  - Hands: {hands}")
            print(f"  - Time: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        detector.release()
        print("\nTest completed.")

if __name__ == "__main__":
    main()
