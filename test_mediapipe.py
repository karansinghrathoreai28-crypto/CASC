"""
MediaPipe Effects Test
Test face mesh and hand tracking effects
"""

import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mediapipe_effects import MediaPipeEffects

def main():
    print("="*70)
    print("MediaPipe Effects Test")
    print("="*70)
    print("This will show cool graphics on your face and hands")
    print("\nControls:")
    print("  'f' - Toggle face mesh")
    print("  'h' - Toggle hand tracking")
    print("  'c' - Toggle custom effects")
    print("  'q' - Quit")
    print("="*70)
    
    # Initialize MediaPipe effects
    try:
        mp_effects = MediaPipeEffects()
    except Exception as e:
        print(f"Error initializing MediaPipe: {str(e)}")
        print("\nMake sure mediapipe is installed:")
        print("  pip install mediapipe")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Effect toggles
    show_face_mesh = True
    show_hands = True
    show_custom = False
    
    cv2.namedWindow("MediaPipe Effects Test", cv2.WINDOW_NORMAL)
    
    print("\nCamera started. Show your face and hands!")
    print("If effects don't appear, make sure your face/hands are visible and well-lit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Apply effects
        try:
            display_frame, face_detected, hands_detected = mp_effects.apply_all_effects(
                frame,
                show_face_mesh=show_face_mesh,
                show_hands=show_hands,
                show_custom=show_custom
            )
        except Exception as e:
            print(f"Error applying effects: {str(e)}")
            display_frame = frame.copy()
            face_detected = False
            hands_detected = False
        
        # Add status text
        y_pos = 30
        cv2.putText(display_frame, "MediaPipe Effects Demo", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_pos += 40
        cv2.putText(display_frame, f"Face Mesh: {'ON' if show_face_mesh else 'OFF'} (press 'f')", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 30
        cv2.putText(display_frame, f"Hand Tracking: {'ON' if show_hands else 'OFF'} (press 'h')", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 30
        cv2.putText(display_frame, f"Custom Effects: {'ON' if show_custom else 'OFF'} (press 'c')", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 40
        status = []
        if face_detected:
            status.append("Face Detected")
        if hands_detected:
            status.append("Hands Detected")
        
        status_text = " | ".join(status) if status else "Nothing Detected"
        status_color = (0, 255, 0) if status else (0, 0, 255)
        cv2.putText(display_frame, status_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add frame counter
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Display
        cv2.imshow("MediaPipe Effects Test", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            show_face_mesh = not show_face_mesh
            print(f"Face mesh: {'ON' if show_face_mesh else 'OFF'}")
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"Hand tracking: {'ON' if show_hands else 'OFF'}")
        elif key == ord('c'):
            show_custom = not show_custom
            print(f"Custom effects: {'ON' if show_custom else 'OFF'}")
    
    # Cleanup
    cap.release()
    mp_effects.release()
    cv2.destroyAllWindows()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
