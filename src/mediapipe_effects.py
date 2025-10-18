import cv2
import mediapipe as mp
import numpy as np

class MediaPipeEffects:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Custom green drawing specs with LIGHTER green and THINNER lines
        # Light green color similar to the image
        self.light_green_connection_style = self.mp_drawing.DrawingSpec(
            color=(100, 255, 100),  # Lighter green (BGR format)
            thickness=1  # Thin lines
        )
        
        # Even lighter green for tessellation (the mesh grid)
        self.tessellation_style = self.mp_drawing.DrawingSpec(
            color=(80, 200, 80),  # Subtle light green for mesh
            thickness=1  # Very thin lines
        )
        
        # THINNER green for contours (outer lines) - matching your image
        self.contour_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Bright green for outline
            thickness=1  # Thin line (changed from 2)
        )
        
        # Green for irises
        self.iris_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Bright green for eyes
            thickness=1
        )
        
        # No landmark style - we don't want dots
        self.no_landmarks = None
        
        print("✓ MediaPipe Effects initialized (Face: THIN GREEN lines | Hands: COLORFUL)")
    
    def draw_face_mesh(self, frame, draw_tesselation=True, draw_contours=True, draw_irises=True):
        """
        Draw face mesh with LIGHT GREEN tessellation (like the image)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Draw the face mesh annotations in light green (LINES ONLY, NO DOTS)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw tessellation with LIGHT GREEN (subtle mesh)
                if draw_tesselation:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.no_landmarks,
                        connection_drawing_spec=self.tessellation_style  # Light green mesh
                    )
                
                # Draw contours with BRIGHT GREEN (main outline)
                if draw_contours:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.no_landmarks,
                        connection_drawing_spec=self.contour_style  # Bright green outline
                    )
                
                # Draw irises with BRIGHT GREEN (keep them visible)
                if draw_irises:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=self.no_landmarks,
                        connection_drawing_spec=self.iris_style  # Bright green eyes
                    )
        
        return frame, results.multi_face_landmarks is not None
    
    def draw_hand_landmarks(self, frame):
        """
        Draw hand landmarks and connections on frame in ORIGINAL COLORFUL style
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks in original colorful style
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton with default colorful style
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame, results.multi_hand_landmarks is not None
    
    def draw_custom_face_effects(self, frame):
        """
        Draw custom cool effects on face in GREEN color
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get frame dimensions
                h, w, c = frame.shape
                
                # Convert normalized coordinates to pixel coordinates
                points = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
                
                # Draw green lines connecting specific facial features
                # Connect eyes
                cv2.line(frame, points[33], points[133], (0, 255, 0), 2)  # Right eye
                cv2.line(frame, points[362], points[263], (0, 255, 0), 2)  # Left eye
                
                # Connect mouth corners with a smile curve
                cv2.line(frame, points[61], points[291], (0, 255, 0), 2)
        
        return frame
    
    def apply_all_effects(self, frame, show_face_mesh=True, show_hands=True, show_custom=False):
        """
        Apply all MediaPipe effects to the frame
        Face: GREEN tessellation + contours (grid-like structure with no dots)
        Hands: COLORFUL skeleton
        """
        result_frame = frame.copy()
        face_detected = False
        hands_detected = False
        
        if show_face_mesh:
            # Enable tessellation for full grid-like structure
            result_frame, face_detected = self.draw_face_mesh(
                result_frame,
                draw_tesselation=True,   # ENABLED for grid structure
                draw_contours=True,
                draw_irises=True
            )
        
        if show_hands:
            result_frame, hands_detected = self.draw_hand_landmarks(result_frame)
        
        if show_custom:
            result_frame = self.draw_custom_face_effects(result_frame)
        
        return result_frame, face_detected, hands_detected
    
    def release(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()
        self.hands.close()
