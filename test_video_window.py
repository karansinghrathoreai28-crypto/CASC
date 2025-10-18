"""
Simple test to verify video window displays correctly
"""

import cv2
from datetime import datetime
import time

def test_video_window():
    print("="*70)
    print("Video Window Display Test")
    print("="*70)
    print("Testing if video window displays correctly")
    print("Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return False
    
    # Create window explicitly
    cv2.namedWindow("Video Test", cv2.WINDOW_NORMAL)
    print("Window created. You should see a video window now.")
    print("If you don't see it, check:")
    print("1. Is it hidden behind other windows?")
    print("2. Is it on another monitor?")
    print("3. Try Alt+Tab to find it\n")
    
    time.sleep(2)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Can't receive frame")
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Add info
        timestamp = datetime.now().strftime('%H:%M:%S')
        cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Video Test", frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Print status every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} - FPS: {fps:.1f} - Window should be visible")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTest completed. Processed {frame_count} frames.")
    
    if frame_count > 0:
        print("SUCCESS: Video window worked correctly!")
        return True
    else:
        print("FAILED: No frames processed")
        return False

if __name__ == "__main__":
    success = test_video_window()
    
    if not success:
        print("\nTROUBLESHOOTING:")
        print("1. Make sure no other application is using the camera")
        print("2. Try running: python test_camera.py")
        print("3. Check Windows Privacy Settings > Camera")
        print("4. Update your graphics drivers")
