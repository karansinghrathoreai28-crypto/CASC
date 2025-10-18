"""
Simple camera test to verify OpenCV and camera are working
"""

import cv2
from datetime import datetime

def test_camera():
    print("="*70)
    print("Testing Camera Connection")
    print("="*70)
    print("Press 'q' to quit\n")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Possible solutions:")
        print("1. Check if another application is using the camera")
        print("2. Try a different camera index (change 0 to 1, 2, etc.)")
        print("3. Check camera permissions in Windows Settings")
        return False
    
    print("Camera opened successfully!")
    print("Displaying live feed...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Can't receive frame")
            break
        
        frame_count += 1
        
        # Add info to frame
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Camera Test", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTest completed. Processed {frame_count} frames.")
    return True

if __name__ == "__main__":
    success = test_camera()
    if success:
        print("\nCamera test PASSED!")
    else:
        print("\nCamera test FAILED!")
