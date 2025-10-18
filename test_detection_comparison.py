"""
Test and compare different Haar Cascade classifiers
Shows which detectors work best for your setup
"""

import cv2
from datetime import datetime

def test_all_detectors():
    print("="*70)
    print("Haar Cascade Detection Comparison Test")
    print("="*70)
    print("This will show you which detectors work best")
    print("Press 'q' to quit\n")
    
    # Load all available cascades
    classifiers = {}
    
    try:
        classifiers['Face'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("Loaded: Face detector")
    except:
        print("Failed: Face detector")
    
    try:
        classifiers['Upper Body'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        )
        print("Loaded: Upper Body detector")
    except:
        print("Failed: Upper Body detector")
    
    try:
        classifiers['Full Body'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        print("Loaded: Full Body detector")
    except:
        print("Failed: Full Body detector")
    
    try:
        classifiers['Lower Body'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_lowerbody.xml'
        )
        print("Loaded: Lower Body detector")
    except:
        print("Failed: Lower Body detector")
    
    print("\n" + "="*70)
    print("Starting camera...\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return False
    
    print("Camera opened. Testing detections...")
    print("\nTips for better detection:")
    print("- Stand 6-10 feet from camera for full body")
    print("- Ensure good lighting")
    print("- Avoid cluttered backgrounds")
    print("- Stand upright and face the camera\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Can't receive frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast
        
        # Test each detector
        y_offset = 30
        detection_counts = {}
        
        for name, cascade in classifiers.items():
            if cascade.empty():
                continue
            
            # Adjust parameters based on detector type
            if 'Face' in name:
                detections = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                color = (0, 255, 0)  # Green for faces
            elif 'Upper' in name:
                detections = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                )
                color = (255, 165, 0)  # Orange for upper body
            else:
                detections = cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=2, minSize=(50, 100)
                )
                color = (255, 0, 0)  # Blue for bodies
            
            # Draw detections
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display count
            detection_counts[name] = len(detections)
            status = f"{name}: {len(detections)}"
            cv2.putText(frame, status, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Add timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Detection Comparison Test", frame)
        
        # Print to console periodically
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print(f"\n[{timestamp}] Detection counts:")
            for name, count in detection_counts.items():
                print(f"  {name}: {count}")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nTest completed.")
    print("\nRECOMMENDATIONS:")
    print("- If face detection works: Good for close-range monitoring")
    print("- If upper body works: Best for desk/room monitoring")
    print("- If full body works: Good for entrance/hallway monitoring")
    print("- If nothing works: Consider using motion detection instead")
    
    return True

if __name__ == "__main__":
    test_all_detectors()
