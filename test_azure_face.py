"""
Azure Face API Connection Test
Tests face detection, recognition, person management, and attributes
"""

import yaml
import sys
import cv2
import os
from datetime import datetime

def load_config():
    """Load configuration file"""
    try:
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration file loaded")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {str(e)}")
        return None

def test_client_connection(config):
    """Test basic client connection"""
    print("\n--- Test 1: Client Connection ---")
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.vision.face import FaceAdministrationClient, FaceClient
        
        endpoint = config['azure_face']['endpoint']
        api_key = config['azure_face']['api_key']
        
        if not endpoint or endpoint == "YOUR_AZURE_FACE_ENDPOINT":
            print("✗ Azure Face endpoint not configured")
            return None, None
        
        if not api_key or api_key == "YOUR_AZURE_FACE_KEY":
            print("✗ Azure Face API key not configured")
            return None, None
        
        print(f"  Endpoint: {endpoint}")
        
        # Create clients
        face_admin_client = FaceAdministrationClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        face_client = FaceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
        print("✓ Azure Face clients created successfully")
        return face_admin_client, face_client
        
    except Exception as e:
        print(f"✗ Client connection failed: {str(e)}")
        return None, None

def test_person_group_operations(face_admin_client, config):
    """Test person group creation and management"""
    print("\n--- Test 2: Person Group Operations ---")
    try:
        person_group_id = "test_group_" + datetime.now().strftime("%Y%m%d%H%M%S")
        
        from azure.ai.vision.face.models import FaceRecognitionModel
        
        # Create person group
        face_admin_client.large_person_group.create(
            large_person_group_id=person_group_id,
            name="Test Person Group",
            recognition_model=FaceRecognitionModel.RECOGNITION04
        )
        print(f"✓ Person group created: {person_group_id}")
        
        # Get person group
        group = face_admin_client.large_person_group.get(
            large_person_group_id=person_group_id
        )
        print(f"✓ Person group retrieved: {group.name}")
        
        # List all person groups
        groups = list(face_admin_client.large_person_group.get_large_person_groups())
        print(f"✓ Total person groups: {len(groups)}")
        
        return person_group_id
        
    except Exception as e:
        print(f"✗ Person group operations failed: {str(e)}")
        return None

def test_face_detection_from_camera(face_client):
    """Test face detection from webcam"""
    print("\n--- Test 3: Face Detection from Camera ---")
    try:
        from azure.ai.vision.face.models import (
            FaceDetectionModel,
            FaceRecognitionModel,
            FaceAttributeTypeRecognition04
        )
        
        print("  Opening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Could not open camera")
            return False
        
        print("  Position your face in front of camera")
        print("  Press SPACE to capture, ESC to skip")
        
        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, "Press SPACE to capture, ESC to skip", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Detection Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                print("\n  Capturing and analyzing...")
                
                # Convert frame to bytes
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()
                
                # Detect faces
                detected_faces = face_client.detect(
                    image_content=image_data,
                    detection_model=FaceDetectionModel.DETECTION03,
                    recognition_model=FaceRecognitionModel.RECOGNITION04,
                    return_face_id=True,
                    return_face_attributes=[
                        FaceAttributeTypeRecognition04.HEAD_POSE,
                        FaceAttributeTypeRecognition04.MASK,
                        FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION
                    ]
                )
                
                print(f"\n✓ Detected {len(detected_faces)} face(s)")
                
                for i, face in enumerate(detected_faces, 1):
                    print(f"\n  Face {i}:")
                    print(f"    Face ID: {face.face_id}")
                    print(f"    Rectangle: {face.face_rectangle.width}x{face.face_rectangle.height} at ({face.face_rectangle.left}, {face.face_rectangle.top})")
                    if face.face_attributes:
                        print(f"    Quality: {face.face_attributes.quality_for_recognition.value if face.face_attributes.quality_for_recognition else 'N/A'}")
                        print(f"    Mask: {face.face_attributes.mask.type.value if face.face_attributes.mask else 'N/A'}")
                
                captured = True
                break
            elif key == 27:  # ESC
                print("\n  Skipped")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured:
            print("\n✓ Face detection from camera successful")
            return True
        else:
            print("\n⚠ Face detection test skipped")
            return True
        
    except Exception as e:
        print(f"\n✗ Face detection failed: {str(e)}")
        return False

def test_person_management(face_admin_client, face_client, person_group_id):
    """Test adding and managing persons"""
    print("\n--- Test 4: Person Management ---")
    try:
        from azure.ai.vision.face.models import FaceDetectionModel
        
        # Create a test person
        person = face_admin_client.large_person_group.create_person(
            large_person_group_id=person_group_id,
            name="Test Person"
        )
        print(f"✓ Person created: {person.name} (ID: {person.person_id})")
        
        # Test with sample image URL
        test_image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/Family1-Dad1.jpg"
        
        # Add face to person
        face_admin_client.large_person_group.add_face_from_url(
            large_person_group_id=person_group_id,
            person_id=person.person_id,
            url=test_image_url,
            detection_model=FaceDetectionModel.DETECTION03
        )
        print("✓ Face added to person from URL")
        
        # Get person details
        person_details = face_admin_client.large_person_group.get_person(
            large_person_group_id=person_group_id,
            person_id=person.person_id
        )
        print(f"✓ Person has {len(person_details.persisted_face_ids)} face(s)")
        
        # List all persons
        persons = list(face_admin_client.large_person_group.get_persons(
            large_person_group_id=person_group_id
        ))
        print(f"✓ Total persons in group: {len(persons)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Person management failed: {str(e)}")
        return False

def test_training_and_identification(face_admin_client, face_client, person_group_id):
    """Test person group training and face identification"""
    print("\n--- Test 5: Training and Identification ---")
    try:
        from azure.ai.vision.face.models import (
            FaceDetectionModel,
            FaceRecognitionModel,
            FaceAttributeTypeRecognition04,
            QualityForRecognition
        )
        
        # Train the person group
        print("  Training person group...")
        poller = face_admin_client.large_person_group.begin_train(
            large_person_group_id=person_group_id,
            polling_interval=2
        )
        poller.wait()
        print("✓ Training completed")
        
        # Check training status
        training_status = face_admin_client.large_person_group.get_training_status(
            large_person_group_id=person_group_id
        )
        print(f"✓ Training status: {training_status.status.value}")
        
        # Test identification with another image
        test_image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/Face/images/identification1.jpg"
        
        # Detect faces
        detected_faces = face_client.detect_from_url(
            url=test_image_url,
            detection_model=FaceDetectionModel.DETECTION03,
            recognition_model=FaceRecognitionModel.RECOGNITION04,
            return_face_id=True,
            return_face_attributes=[FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION]
        )
        
        print(f"✓ Detected {len(detected_faces)} face(s) in test image")
        
        # Filter high quality faces
        face_ids = [face.face_id for face in detected_faces 
                   if face.face_attributes.quality_for_recognition != QualityForRecognition.LOW]
        
        if face_ids:
            # Identify faces
            identify_results = face_client.identify_from_large_person_group(
                face_ids=face_ids,
                large_person_group_id=person_group_id
            )
            
            print(f"✓ Identification completed for {len(face_ids)} face(s)")
            
            for i, result in enumerate(identify_results, 1):
                if result.candidates:
                    candidate = result.candidates[0]
                    print(f"  Face {i}: Identified with confidence {candidate.confidence:.2f}")
                else:
                    print(f"  Face {i}: Not identified")
        else:
            print("⚠ No high-quality faces to identify")
        
        return True
        
    except Exception as e:
        print(f"✗ Training/identification failed: {str(e)}")
        return False

def test_cleanup(face_admin_client, person_group_id):
    """Clean up test resources"""
    print("\n--- Test 6: Cleanup ---")
    try:
        if person_group_id:
            face_admin_client.large_person_group.delete(person_group_id)
            print(f"✓ Test person group deleted: {person_group_id}")
        return True
    except Exception as e:
        print(f"✗ Cleanup failed: {str(e)}")
        return False

def test_integration_with_casc():
    """Test integration with CASC system"""
    print("\n--- Test 7: CASC Integration ---")
    try:
        from src.azure_face_client import AzureFaceClient
        
        config = load_config()
        if not config:
            return False
        
        # Initialize CASC Azure Face client
        face_client = AzureFaceClient(config)
        print("✓ CASC Azure Face client initialized")
        
        # Test list known persons
        persons = face_client.list_known_persons()
        print(f"✓ Listed {len(persons)} known person(s)")
        
        if persons:
            print("\n  Known Persons:")
            for person in persons:
                print(f"    - {person['name']} (Faces: {person['face_count']})")
        
        return True
        
    except Exception as e:
        print(f"✗ CASC integration test failed: {str(e)}")
        return False

def main():
    print("="*70)
    print("Azure Face API Connection Tester")
    print("="*70)
    
    # Load config
    config = load_config()
    if not config:
        sys.exit(1)
    
    # Track results
    results = {}
    person_group_id = None
    
    # Test 1: Client connection
    face_admin_client, face_client = test_client_connection(config)
    results['client_connection'] = (face_admin_client is not None and face_client is not None)
    
    if not results['client_connection']:
        print("\n✗ Cannot proceed without client connection")
        sys.exit(1)
    
    # Test 2: Person group operations
    person_group_id = test_person_group_operations(face_admin_client, config)
    results['person_group'] = (person_group_id is not None)
    
    if not person_group_id:
        print("\n⚠ Skipping tests that require person group")
    else:
        # Test 3: Face detection from camera
        results['face_detection'] = test_face_detection_from_camera(face_client)
        
        # Test 4: Person management
        results['person_management'] = test_person_management(
            face_admin_client, face_client, person_group_id
        )
        
        # Test 5: Training and identification
        if results['person_management']:
            results['training_identification'] = test_training_and_identification(
                face_admin_client, face_client, person_group_id
            )
        
        # Test 6: Cleanup
        results['cleanup'] = test_cleanup(face_admin_client, person_group_id)
    
    # Test 7: CASC integration
    results['casc_integration'] = test_integration_with_casc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL AZURE FACE TESTS PASSED")
        print("  Azure Face integration is ready!")
        print("\nNext steps:")
        print("  1. Run: python src/server.py")
        print("  2. Select option 4: Manage Known Persons")
        print("  3. Add known persons to the database")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Check Azure Face endpoint and API key")
        print("  Verify you have Face API subscription")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
